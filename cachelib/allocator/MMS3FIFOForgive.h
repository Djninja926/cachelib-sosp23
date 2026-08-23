/*
 * MMS3FIFOForgive: S3-FIFO eviction policy augmented with TARDIS forgiveness.
 *
 * Mirrors MMS3FIFO, with these additions:
 *   - owns a TARDISEmbeddingManager (created from Config, like MMLruForgive),
 *   - passes a non-owning pointer to the manager down into the
 *     S3FIFOForgiveList (which performs the forgiveness decision at eviction),
 *   - updates the embedding on every access in recordAccess.
 *
 * Forgiveness itself lives in S3FIFOForgiveList::getEvictionCandidate: an
 * object S3-FIFO would drop from the small FIFO (freq 0) is promoted to main
 * instead, if its embedding is similar to recently-accessed objects.
 */
#pragma once

#include <atomic>
#include <cstring>
#include <memory>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <folly/Format.h>
#pragma GCC diagnostic pop
#include <folly/container/Array.h>
#include <folly/hash/SpookyHashV2.h>
#include <folly/lang/Aligned.h>
#include <folly/synchronization/DistributedMutex.h>

#include "cachelib/allocator/Cache.h"
#include "cachelib/allocator/CacheStats.h"
#include "cachelib/allocator/TARDISEmbeddingManager.h"
#include "cachelib/allocator/Util.h"
#include "cachelib/allocator/datastruct/S3FIFOForgiveList.h"
#include "cachelib/allocator/memory/serialize/gen-cpp2/objects_types.h"
#include "cachelib/common/CompilerUtils.h"
#include "cachelib/common/Mutex.h"

namespace facebook {
namespace cachelib {

class MMS3FIFOForgive {
 public:
  static const int kId;

  template <typename T>
  using Hook = AtomicDListHook<T>;

  using SerializationType = serialization::MMS3FIFOObject;
  using SerializationConfigType = serialization::MMS3FIFOConfig;
  using SerializationTypeContainer = serialization::MMS3FIFOCollection;

  enum LruType { Prob, Main, NumTypes };

  struct Config {
    explicit Config(SerializationConfigType configState)
        : Config(*configState.updateOnWrite(), *configState.updateOnRead()) {}

    Config(bool updateOnW, bool updateOnR) : Config(updateOnW, updateOnR, 0) {}

    Config(bool updateOnW, bool updateOnR, uint32_t mmReconfigureInterval)
        : updateOnWrite(updateOnW),
          updateOnRead(updateOnR),
          mmReconfigureIntervalSecs(
              std::chrono::seconds(mmReconfigureInterval)) {}

    Config() = default;
    Config(const Config& rhs) = default;
    Config(Config&& rhs) = default;
    Config& operator=(const Config& rhs) = default;
    Config& operator=(Config&& rhs) = default;

    template <typename... Args>
    void addExtraConfig(Args...) {}

    bool updateOnWrite{false};
    bool updateOnRead{true};
    bool tryLockUpdate{false};
    uint8_t lruInsertionPointSpec{0};
    std::chrono::seconds mmReconfigureIntervalSecs{};
    bool useCombinedLockForIterators{false};

    // ---- TARDIS parameters (mirror MMLruForgive defaults) ----
    double learningRate{0.2};
    double contextSpeed{0.001};
    int recentWindow{16};
    int minAccessCount{2};
    double forgiveThreshold{0.4};
    int maxForgives{5};
    int64_t maxEmbeddingEntries{-1};
    int max_K{3};
  };

  template <typename T, Hook<T> T::*HookPtr>
  struct Container {
   private:
    using FIFOList = S3FIFOForgiveList<T, HookPtr>;
    using Mutex = folly::DistributedMutex;
    using LockHolder = std::unique_lock<Mutex>;
    using PtrCompressor = typename T::PtrCompressor;
    using Time = typename Hook<T>::Time;
    using CompressedPtr = typename T::CompressedPtr;
    using RefFlags = typename T::Flags;

   public:
    Container() = default;
    Container(Config c, PtrCompressor compressor)
        : qdlist_(std::move(compressor)), config_(std::move(c)) {
      nextReconfigureTime_ =
          config_.mmReconfigureIntervalSecs.count() == 0
              ? std::numeric_limits<Time>::max()
              : static_cast<Time>(util::getCurrentTimeSec()) +
                    config_.mmReconfigureIntervalSecs.count();
      // Create the embedding manager and wire it into the list.
      embMgr_ = std::make_unique<TARDISEmbeddingManager>(
          config_.learningRate,
          config_.contextSpeed,
          config_.recentWindow,
          config_.minAccessCount,
          config_.maxEmbeddingEntries);
      qdlist_.setEmbeddingManager(embMgr_.get());
      qdlist_.setForgiveParams(
          config_.forgiveThreshold, config_.minAccessCount, config_.max_K);
    }
    Container(serialization::MMS3FIFOObject object, PtrCompressor compressor);

    Container(const Container&) = delete;
    Container& operator=(const Container&) = delete;

    class LockedIterator {
     public:
      LockedIterator(const LockedIterator&) = delete;
      LockedIterator& operator=(const LockedIterator&) = delete;
      LockedIterator(LockedIterator&&) noexcept = default;

      LockedIterator& operator++() {
        candidate_ = qdlist_->getEvictionCandidate();
        return *this;
      }
      LockedIterator& operator--() { throw std::logic_error("Not implemented"); }

      T* operator->() const noexcept { return candidate_; }
      T& operator*() const noexcept { return *candidate_; }
      explicit operator bool() const noexcept { return candidate_ != nullptr; }
      T* get() const noexcept { return candidate_; }

      void reset() noexcept {}
      void destroy() {}
      void resetToBegin() noexcept {}

     private:
      LockedIterator& operator=(LockedIterator&&) noexcept = default;
      LockedIterator(FIFOList* qdlist) {
        qdlist_ = qdlist;
        candidate_ = qdlist_->getEvictionCandidate();
      }
      FIFOList* qdlist_;
      T* candidate_;
      friend Container<T, HookPtr>;
    };

    bool recordAccess(T& node, AccessMode mode) noexcept;
    bool add(T& node) noexcept;
    bool remove(T& node) noexcept;
    void remove(LockedIterator& it) noexcept;
    bool replace(T& oldNode, T& newNode) noexcept;

    LockedIterator getEvictionIterator() noexcept;

    Config getConfig() const;
    void setConfig(const Config& newConfig);

    bool isEmpty() const noexcept { return size() == 0; }
    void reconfigureLocked(const Time& currTime);

    size_t size() const noexcept {
      return lruMutex_->lock_combine([this]() { return qdlist_.size(); });
    }

    EvictionAgeStat getEvictionAgeStat(uint64_t projectedLength) const noexcept;
    serialization::MMS3FIFOObject saveState() const noexcept;
    MMContainerStat getStats() const noexcept;

    LruType getLruType(const T& node) noexcept {
      if (isProbationary(node)) {
        return LruType::Prob;
      } else {
        XDCHECK(isMain(node));
        return LruType::Main;
      }
    }

    int64_t getForgiveCount() const noexcept {
      return qdlist_.getForgiveCount();
    }
    int64_t getEvictCount() const noexcept { return qdlist_.getEvictCount(); }

   private:
    EvictionAgeStat getEvictionAgeStatLocked(
        uint64_t projectedLength) const noexcept;

    static Time getUpdateTime(const T& node) noexcept {
      return (node.*HookPtr).getUpdateTime();
    }
    static void setUpdateTime(T& node, Time time) noexcept {
      (node.*HookPtr).setUpdateTime(time);
    }

    // MUST match S3FIFOForgiveList::embKey (SpookyHashV2), so the embedding
    // manager sees the same id on access and eviction.
    static uint64_t hashKey(const T& node) noexcept {
      auto key = node.getKey();
      return folly::hash::SpookyHashV2::Hash64(key.data(), key.size(), 0);
    }
    static int shardOfKey(const T& node) noexcept {
      auto key = node.getKey();
      uint64_t value = 0;
      for (char c : key) {
        if (c < '0' || c > '9') break;
        value = value * 10 + static_cast<uint64_t>(c - '0');
      }
      return static_cast<int>((value >> 32) % TARDISEmbeddingManager::kMaxThreads);
    }

    static int cachedShardOfKey(const T& node) noexcept {
      thread_local int cached = -1;
      if (cached < 0) {
        cached = shardOfKey(node);
      }
      return cached;
    }

    void removeLocked(T& node) noexcept;

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
    void markAccessed(T& node) noexcept {
      node.template setFlag<RefFlags::kMMFlag1>();
    }
    void unmarkAccessed(T& node) noexcept {
      node.template unSetFlag<RefFlags::kMMFlag1>();
    }
    bool isAccessed(const T& node) const noexcept {
      return node.template isFlagSet<RefFlags::kMMFlag1>();
    }

    mutable folly::cacheline_aligned<Mutex> lruMutex_;
    FIFOList qdlist_{};
    std::atomic<Time> nextReconfigureTime_{};
    Config config_{};

    // TARDIS embedding manager, owned here, pointer passed to the list.
    std::unique_ptr<TARDISEmbeddingManager> embMgr_;

    FRIEND_TEST(MMS3FIFOForgiveTest, Reconfigure);
  };
};

}  // namespace cachelib
}  // namespace facebook

#include "cachelib/allocator/MMS3FIFOForgive-inl.h"