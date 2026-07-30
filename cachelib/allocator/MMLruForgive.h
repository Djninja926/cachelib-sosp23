/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
#include "cachelib/allocator/datastruct/DList.h"
#include "cachelib/allocator/memory/serialize/gen-cpp2/objects_types.h"
#include "cachelib/common/CompilerUtils.h"
#include "cachelib/common/Mutex.h"

namespace facebook {
namespace cachelib {

// TARDIS-augmented LRU eviction policy.
// Behaves like standard MMLru but on each eviction, the policy walks from the
// LRU tail and may "forgive" up to max_forgives candidates based on their
// embedding similarity to recently accessed objects. Forgiven candidates are
// promoted to the head of the LRU instead of being evicted.
class MMLruForgive {
 public:
  // unique identifier per MMType (must not collide with other policies)
  static const int kId;

  // forward declaration
  template <typename T>
  using Hook = DListHook<T>;

  // Reuse MMLru's serialization types since the persistent state is the same
  // structure (the embedding state itself is not persisted in this prototype).
  using SerializationType = serialization::MMLruObject;
  using SerializationConfigType = serialization::MMLruConfig;
  using SerializationTypeContainer = serialization::MMLruCollection;

  // Not applicable for MMLruForgive, just for compile of cache allocator
  enum LruType { NumTypes };

  // Config class for MMLruForgive
  struct Config {
    // create from serialized config (TARDIS params use defaults)
    explicit Config(SerializationConfigType configState)
        : Config(*configState.lruRefreshTime(),
                 *configState.lruRefreshRatio(),
                 *configState.updateOnWrite(),
                 *configState.updateOnRead(),
                 *configState.tryLockUpdate(),
                 static_cast<uint8_t>(*configState.lruInsertionPointSpec())) {}

    Config(uint32_t time, bool updateOnW, bool updateOnR)
        : Config(time, updateOnW, updateOnR, false, 0) {}

    Config(uint32_t time, bool updateOnW, bool updateOnR, uint8_t ipSpec)
        : Config(time, updateOnW, updateOnR, false, ipSpec) {}

    Config(uint32_t time,
           bool updateOnW,
           bool updateOnR,
           bool tryLockU,
           uint8_t ipSpec)
        : Config(time, 0., updateOnW, updateOnR, tryLockU, ipSpec) {}

    Config(uint32_t time,
           double ratio,
           bool updateOnW,
           bool updateOnR,
           bool tryLockU,
           uint8_t ipSpec)
        : Config(time, ratio, updateOnW, updateOnR, tryLockU, ipSpec, 0) {}

    Config(uint32_t time,
           double ratio,
           bool updateOnW,
           bool updateOnR,
           bool tryLockU,
           uint8_t ipSpec,
           uint32_t mmReconfigureInterval)
        : Config(time,
                 ratio,
                 updateOnW,
                 updateOnR,
                 tryLockU,
                 ipSpec,
                 mmReconfigureInterval,
                 false) {}

    Config(uint32_t time,
           double ratio,
           bool updateOnW,
           bool updateOnR,
           bool tryLockU,
           uint8_t ipSpec,
           uint32_t mmReconfigureInterval,
           bool useCombinedLockForIterators)
        : defaultLruRefreshTime(time),
          lruRefreshTime(time),
          lruRefreshRatio(ratio),
          updateOnWrite(updateOnW),
          updateOnRead(updateOnR),
          tryLockUpdate(tryLockU),
          lruInsertionPointSpec(ipSpec),
          mmReconfigureIntervalSecs(
              std::chrono::seconds(mmReconfigureInterval)),
          useCombinedLockForIterators(useCombinedLockForIterators) {}

    Config() = default;
    Config(const Config& rhs) = default;
    Config(Config&& rhs) = default;

    Config& operator=(const Config& rhs) = default;
    Config& operator=(Config&& rhs) = default;

    template <typename... Args>
    void addExtraConfig(Args...) {}

    // ---- Standard LRU parameters ----
    uint32_t defaultLruRefreshTime{60};
    uint32_t lruRefreshTime{defaultLruRefreshTime};
    double lruRefreshRatio{0.};
    bool updateOnWrite{false};
    bool updateOnRead{true};
    bool tryLockUpdate{false};
    uint8_t lruInsertionPointSpec{0};
    std::chrono::seconds mmReconfigureIntervalSecs{};
    bool useCombinedLockForIterators{false};

    // TARDIS-specific parameters (defaults from Aryan's reference) 

    // Embedding learning rate. Controls how fast embeddings adapt to context.
    double learningRate{0.2};

    // Context vector drift speed. Controls how quickly the global context
    // decorrelates over time.
    double contextSpeed{0.001};

    // Number of recent objects whose embeddings are compared against eviction
    // candidates when computing the forgiveness score.
    int recentWindow{16};

    // Minimum number of accesses before an object gets an embedding. Filters
    // out one-hit-wonders from the embedding store.
    int minAccessCount{2};

    // Cosine similarity threshold above which an eviction candidate is forgiven
    // (promoted to MRU instead of being evicted).
    double forgiveThreshold{0.4};

    // Maximum number of forgiveness decisions per eviction call. Prevents
    // pathological loops where many tail candidates are all forgivable.
    int maxForgives{5};

    // Maximum number of embeddings to store. -1 means unlimited
    // (FORGIVE_CLEAN mode). Positive values enable MCACHE mode with LRU
    // eviction of the embedding store itself.
    int64_t maxEmbeddingEntries{-1};

    // Top-k for the similarity score computation (paper uses 3).
    int max_K{3};
  };

  // The container that actually manages the LRU + TARDIS state.
  template <typename T, Hook<T> T::*HookPtr>
  struct Container {
   private:
    using LruList = DList<T, HookPtr>;
    using Mutex = folly::DistributedMutex;
    using LockHolder = std::unique_lock<Mutex>;
    using PtrCompressor = typename T::PtrCompressor;
    using Time = typename Hook<T>::Time;
    using CompressedPtr = typename T::CompressedPtr;
    using RefFlags = typename T::Flags;

   public:
    Container() = default;
    Container(Config c, PtrCompressor compressor)
        : compressor_(std::move(compressor)),
          lru_(compressor_),
          config_(std::move(c)) {
      lruRefreshTime_ = config_.lruRefreshTime;
      nextReconfigureTime_ =
          config_.mmReconfigureIntervalSecs.count() == 0
              ? std::numeric_limits<Time>::max()
              : static_cast<Time>(util::getCurrentTimeSec()) +
                    config_.mmReconfigureIntervalSecs.count();
      // Initialize TARDIS embedding manager from config
      embMgr_ = std::make_unique<TARDISEmbeddingManager>(
          config_.learningRate,
          config_.contextSpeed,
          config_.recentWindow,
          config_.minAccessCount,
          config_.maxEmbeddingEntries);
    }
    Container(serialization::MMLruObject object, PtrCompressor compressor);

    Container(const Container&) = delete;
    Container& operator=(const Container&) = delete;

    using Iterator = typename LruList::Iterator;

    class LockedIterator : public Iterator {
     public:
      LockedIterator(const LockedIterator&) = delete;
      LockedIterator& operator=(const LockedIterator&) = delete;
      LockedIterator(LockedIterator&&) noexcept = default;

      void destroy() {
        Iterator::reset();
        if (l_.owns_lock()) {
          l_.unlock();
        }
      }

      void resetToBegin() {
        if (!l_.owns_lock()) {
          l_.lock();
        }
        Iterator::resetToBegin();
      }

     private:
      LockedIterator& operator=(LockedIterator&&) noexcept = default;
      LockedIterator(LockHolder l, const Iterator& iter) noexcept;
      friend Container<T, HookPtr>;
      LockHolder l_;
    };

    // Records an access. Updates the TARDIS embedding for the accessed object
    // before delegating to the standard LRU promotion logic.
    bool recordAccess(T& node, AccessMode mode) noexcept;

    bool add(T& node) noexcept;
    bool remove(T& node) noexcept;
    void remove(Iterator& it) noexcept;
    bool replace(T& oldNode, T& newNode) noexcept;

    // Returns an iterator starting at the LRU tail. Before returning, walks
    // the tail and promotes up to maxForgives forgivable candidates to MRU.
    LockedIterator getEvictionIterator() const noexcept;

    template <typename F>
    void withEvictionIterator(F&& f);

    Config getConfig() const;
    void setConfig(const Config& newConfig);

    bool isEmpty() const noexcept { return size() == 0; }
    void reconfigureLocked(const Time& currTime);

    size_t size() const noexcept {
      return lruMutex_->lock_combine([this]() { return lru_.size(); });
    }

    EvictionAgeStat getEvictionAgeStat(uint64_t projectedLength) const noexcept;
    serialization::MMLruObject saveState() const noexcept;
    MMContainerStat getStats() const noexcept;

    static LruType getLruType(const T& /* node */) noexcept {
      return LruType{};
    }

    // Forgiveness statistics (for debugging and validation)
    int64_t getForgiveCount() const noexcept { return nForgive_.load(); }
    int64_t getEvictCount() const noexcept { return nEvict_.load(); }

   private:
    EvictionAgeStat getEvictionAgeStatLocked(
        uint64_t projectedLength) const noexcept;

    static Time getUpdateTime(const T& node) noexcept {
      return (node.*HookPtr).getUpdateTime();
    }

    static void setUpdateTime(T& node, Time time) noexcept {
      (node.*HookPtr).setUpdateTime(time);
    }

    void ensureNotInsertionPoint(T& node) noexcept;
    void updateLruInsertionPoint() noexcept;
    void removeLocked(T& node);

    // Hash a CacheLib key (StringPiece-derived) to a uint64_t for the
    // embedding manager.
    static uint64_t hashKey(const T& node) noexcept {
      auto key = node.getKey();
      return folly::hash::SpookyHashV2::Hash64(key.data(), key.size(), 0);
    }

    // Walk the tail and promote forgivable candidates. Caller must hold the
    // lru mutex. Returns the iterator pointing at the first non-forgivable
    // tail (or end if everything was forgiven, which would be a degenerate
    // case bounded by maxForgives).
    void applyForgivenessLocked() noexcept;

    void markTail(T& node) noexcept {
      node.template setFlag<RefFlags::kMMFlag0>();
    }
    void unmarkTail(T& node) noexcept {
      node.template unSetFlag<RefFlags::kMMFlag0>();
    }
    bool isTail(T& node) const noexcept {
      return node.template isFlagSet<RefFlags::kMMFlag0>();
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
    const PtrCompressor compressor_{};
    LruList lru_{};
    T* insertionPoint_{nullptr};
    size_t tailSize_{0};
    std::atomic<Time> nextReconfigureTime_{};
    std::atomic<uint32_t> lruRefreshTime_{};
    Config config_{};

    // TARDIS embedding state. Marked mutable so const methods like
    // getEvictionIterator can update it (forgiveness happens inside).
    mutable std::unique_ptr<TARDISEmbeddingManager> embMgr_;

    // Stats (mutable for the same reason).
    mutable std::atomic<int64_t> nForgive_{0};
    mutable std::atomic<int64_t> nEvict_{0};
    mutable std::atomic<int64_t> nCandidates_{0};
    mutable std::atomic<int64_t> nQualified_{0};

    static constexpr uint32_t kLruRefreshTimeCap{900};
  };
};

}  // namespace cachelib
}  // namespace facebook

#include "cachelib/allocator/MMLruForgive-inl.h"