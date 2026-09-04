/*
 * S3FIFOForgiveList: a copy of S3FIFOList with a forgiveness hook.
 *
 * Structurally identical to S3FIFOList, with one change: on the single-thread
 * eviction path (getEvictionCandidate), when the small (probationary) FIFO
 * would evict a NON-accessed tail object (an S3-FIFO one-hit-wonder drop), we
 * first consult the TARDIS embedding manager. If the object's embedding is
 * similar enough to recently-accessed objects, we PROMOTE it to main instead
 * of evicting it, exactly the action S3-FIFO takes for a re-accessed object.
 * This is the S3-FIFO analog of LRUForgive's tail rescue.
 *
 * The embedding manager is owned by the MM policy (MMS3FIFOForgive) and passed
 * in via setEmbeddingManager. The list holds only a non-owning pointer. The
 * key fed to the manager MUST match the MM policy's access-path key: both use
 * SpookyHashV2::Hash64(node.getKey()), NOT the folly::hasher used by hashNode.
 */
#pragma once

#include <folly/MPMCQueue.h>
#include <folly/hash/SpookyHashV2.h>
#include <folly/logging/xlog.h>

#include <algorithm>
#include <atomic>
#include <thread>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include "cachelib/allocator/serialize/gen-cpp2/objects_types.h"
#pragma GCC diagnostic pop

#include <folly/lang/Aligned.h>
#include <folly/synchronization/DistributedMutex.h>

#include "cachelib/allocator/TARDISEmbeddingManager.h"
#include "cachelib/allocator/datastruct/AtomicDList.h"
#include "cachelib/allocator/datastruct/AtomicFIFOHashTable.h"
#include "cachelib/allocator/datastruct/DList.h"
#include "cachelib/common/BloomFilter.h"
#include "cachelib/common/CompilerUtils.h"
#include "cachelib/common/Mutex.h"

namespace facebook {
namespace cachelib {

template <typename T, AtomicDListHook<T> T::*HookPtr>
class S3FIFOForgiveList {
 public:
  using Mutex = folly::DistributedMutex;
  using LockHolder = std::unique_lock<Mutex>;
  using CompressedPtr = typename T::CompressedPtr;
  using PtrCompressor = typename T::PtrCompressor;
  using ADList = AtomicDList<T, HookPtr>;
  using RefFlags = typename T::Flags;
  using S3FIFOListObject = serialization::S3FIFOListObject;

  S3FIFOForgiveList() = default;
  S3FIFOForgiveList(const S3FIFOForgiveList&) = delete;
  S3FIFOForgiveList& operator=(const S3FIFOForgiveList&) = delete;

 ~S3FIFOForgiveList() {
    fprintf(stderr, "[S3FIFOFORGIVE] nForgive=%ld nEvict=%ld\n",
            (long)nForgive_.load(), (long)nEvict_.load());
    stop_ = true;
    if (evThread_) {
      evThread_->join();
    }
  }

  S3FIFOForgiveList(PtrCompressor compressor) noexcept {
    pfifo_ = std::make_unique<ADList>(compressor);
    mfifo_ = std::make_unique<ADList>(compressor);
  }

  S3FIFOForgiveList(const S3FIFOListObject& object, PtrCompressor compressor) {
    pfifo_ = std::make_unique<ADList>(*object.pfifo(), compressor);
    mfifo_ = std::make_unique<ADList>(*object.mfifo(), compressor);
  }

  S3FIFOListObject saveState() const {
    S3FIFOListObject state;
    *state.pfifo() = pfifo_->saveState();
    *state.mfifo() = mfifo_->saveState();
    return state;
  }

  // Non-owning; the MM policy owns the manager and outlives this list.
  void setEmbeddingManager(TARDISEmbeddingManager* mgr) noexcept {
    embMgr_ = mgr;
  }
  void setForgiveParams(double threshold, int minAccess, int topK) noexcept {
    forgiveThreshold_ = threshold;
    minAccessCount_ = minAccess;
    topK_ = topK;
    if (const char* e = std::getenv("TARDIS_S3FF_MARK_ACCESSED")) markAccessed_ = (std::atoi(e) != 0);
  }

  int64_t getForgiveCount() const noexcept { return nForgive_.load(); }
  int64_t getEvictCount() const noexcept { return nEvict_.load(); }

  ADList& getListProbationary() const noexcept { return *pfifo_; }
  ADList& getListMain() const noexcept { return *mfifo_; }
  size_t size() const noexcept { return pfifo_->size() + mfifo_->size(); }

  T* getEvictionCandidate() noexcept;

  void add(T& node) noexcept {
    if (hist_.initialized() && hist_.contains(hashNode(node))) {
      mfifo_->linkAtHead(node);
      markMain(node);
      unmarkProbationary(node);
    } else {
      pfifo_->linkAtHead(node);
      markProbationary(node);
      unmarkMain(node);
    }
  }

  void markProbationary(T& node) noexcept {
    node.template setFlag<RefFlags::kMMFlag0>();
  }
  void unmarkProbationary(T& node) noexcept {
    node.template unSetFlag<RefFlags::kMMFlag0>();
  }
  bool isProbationary(const T& node) const noexcept {
    return node.template isFlagSet<RefFlags::kMMFlag0>();
  }
  void markMain(T& node) noexcept {
    node.template setFlag<RefFlags::kMMFlag2>();
  }
  void unmarkMain(T& node) noexcept {
    node.template unSetFlag<RefFlags::kMMFlag2>();
  }
  bool isMain(const T& node) const noexcept {
    return node.template isFlagSet<RefFlags::kMMFlag2>();
  }

 private:
  static uint32_t hashNode(const T& node) noexcept {
    return static_cast<uint32_t>(
        folly::hasher<folly::StringPiece>()(node.getKey()));
  }

  // MUST match MMS3FIFOForgive's access-path key (SpookyHashV2), so the
  // embedding manager sees the same id on access and on eviction.
  static uint64_t embKey(const T& node) noexcept {
    auto key = node.getKey();
    return folly::hash::SpookyHashV2::Hash64(key.data(), key.size(), 0);
  }

  // Decode the owning shard from the key. UNCACHED (unlike the MM policy's
  // cachedShardOfKey): the eviction path runs on a thread that generally does
  // not own the candidate, so the shard must be computed from the candidate's
  // own key every time. Uses the bit-shift decode matching reader.cpp's
  // (obj_id % UINT32_MAX) | (reader_id << 32) encoding.
  static int shardOfKey(const T& node) noexcept {
    auto key = node.getKey();
    uint64_t value = 0;
    for (char c : key) {
      if (c < '0' || c > '9') break;
      value = value * 10 + static_cast<uint64_t>(c - '0');
    }
    return static_cast<int>((value >> 32) % TARDISEmbeddingManager::kMaxThreads);
  }

  // The forgiveness decision: true if this object should be rescued (promoted
  // to main) instead of evicted. Mirrors MMLruForgive's 3-part gate.
  bool shouldForgive(const T& node) noexcept {
    if (embMgr_ == nullptr) return false;
    const uint64_t id = embKey(node);
    const int shard = shardOfKey(node);   // UNCACHED: eviction candidate's own shard
    if (embMgr_->get_access_count(id, shard) < minAccessCount_) return false;
    if (!embMgr_->has_embedding(id, shard)) return false;
    const double sim = embMgr_->avg_top_k_similarity_to_recent(id, shard, topK_);
    return sim >= forgiveThreshold_;
  }

  std::unique_ptr<ADList> pfifo_;
  std::unique_ptr<ADList> mfifo_;
  std::unique_ptr<ADList[]> pfifoSublists_;
  std::unique_ptr<ADList[]> mfifoSublists_;
  mutable folly::cacheline_aligned<Mutex> mtx_;
  constexpr static double pRatio_ = 0.05;
  AtomicFIFOHashTable hist_;
  constexpr static size_t nMaxEvictionCandidates_ = 64;
  folly::MPMCQueue<T*> evictCandidateQueue_{nMaxEvictionCandidates_};
  std::unique_ptr<std::thread> evThread_{nullptr};
  std::atomic<bool> stop_{false};
  bool markAccessed_ = false;  // TARDIS_S3FF_MARK_ACCESSED: forgive by setting accessed bit + re-link in small (earns normal S3-FIFO promotion next lap)

  // Forgiveness state (non-owning manager + params).
  TARDISEmbeddingManager* embMgr_{nullptr};
  double forgiveThreshold_{0.4};
  int minAccessCount_{2};
  int topK_{3};
  std::atomic<int64_t> nForgive_{0};
  std::atomic<int64_t> nEvict_{0};
};

}  // namespace cachelib
}  // namespace facebook

#include "cachelib/allocator/datastruct/S3FIFOForgiveList-inl.h"