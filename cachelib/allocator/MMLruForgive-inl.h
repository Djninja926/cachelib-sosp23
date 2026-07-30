/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

namespace facebook {
namespace cachelib {

namespace detail {
template <typename T>
bool areBytesSameForgive(const T& one, const T& two) {
  return std::memcmp(&one, &two, sizeof(T)) == 0;
}
}  // namespace detail

/* Container Interface Implementation */

template <typename T, MMLruForgive::Hook<T> T::*HookPtr>
MMLruForgive::Container<T, HookPtr>::Container(serialization::MMLruObject object,
                                               PtrCompressor compressor)
    : compressor_(std::move(compressor)),
      lru_(*object.lru(), compressor_),
      insertionPoint_(compressor_.unCompress(
          CompressedPtr{*object.compressedInsertionPoint()})),
      tailSize_(*object.tailSize()),
      config_(*object.config()) {
  lruRefreshTime_ = config_.lruRefreshTime;
  nextReconfigureTime_ = config_.mmReconfigureIntervalSecs.count() == 0
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

template <typename T, MMLruForgive::Hook<T> T::*HookPtr>
bool MMLruForgive::Container<T, HookPtr>::recordAccess(
    T& node, AccessMode mode) noexcept {
  // TARDIS: update embedding on EVERY access, regardless of LRU refresh-time
  // gate or updateOnRead/updateOnWrite flags. This is essential for
  // correctness against the libCacheSim reference, which records every event.
  embMgr_->on_access(hashKey(node));

  // Standard MMLru recordAccess logic from here down
  if ((mode == AccessMode::kWrite && !config_.updateOnWrite) ||
      (mode == AccessMode::kRead && !config_.updateOnRead)) {
    return false;
  }

  const auto curr = static_cast<Time>(util::getCurrentTimeSec());
  if (node.isInMMContainer() &&
      ((curr >= getUpdateTime(node) +
                    lruRefreshTime_.load(std::memory_order_relaxed)) ||
       !isAccessed(node))) {
    if (!isAccessed(node)) {
      markAccessed(node);
    }

    auto func = [this, &node, curr]() {
      reconfigureLocked(curr);
      ensureNotInsertionPoint(node);
      if (node.isInMMContainer()) {
        lru_.moveToHead(node);
        setUpdateTime(node, curr);
      }
      if (isTail(node)) {
        unmarkTail(node);
        tailSize_--;
        XDCHECK_LE(0u, tailSize_);
        updateLruInsertionPoint();
      }
    };

    if (config_.tryLockUpdate) {
      if (auto lck = LockHolder{*lruMutex_, std::try_to_lock}) {
        func();
        return true;
      }
      return false;
    }

    lruMutex_->lock_combine(func);
    return true;
  }
  return false;
}

template <typename T, MMLruForgive::Hook<T> T::*HookPtr>
cachelib::EvictionAgeStat
MMLruForgive::Container<T, HookPtr>::getEvictionAgeStat(
    uint64_t projectedLength) const noexcept {
  return lruMutex_->lock_combine([this, projectedLength]() {
    return getEvictionAgeStatLocked(projectedLength);
  });
}

template <typename T, MMLruForgive::Hook<T> T::*HookPtr>
cachelib::EvictionAgeStat
MMLruForgive::Container<T, HookPtr>::getEvictionAgeStatLocked(
    uint64_t projectedLength) const noexcept {
  EvictionAgeStat stat{};
  const auto currTime = static_cast<Time>(util::getCurrentTimeSec());

  const T* node = lru_.getTail();
  stat.warmQueueStat.oldestElementAge =
      node ? currTime - getUpdateTime(*node) : 0;
  for (size_t numSeen = 0; numSeen < projectedLength && node != nullptr;
       numSeen++, node = lru_.getPrev(*node)) {
  }
  stat.warmQueueStat.projectedAge = node ? currTime - getUpdateTime(*node)
                                         : stat.warmQueueStat.oldestElementAge;
  XDCHECK(detail::areBytesSameForgive(stat.hotQueueStat, EvictionStatPerType{}));
  XDCHECK(detail::areBytesSameForgive(stat.coldQueueStat, EvictionStatPerType{}));
  return stat;
}

template <typename T, MMLruForgive::Hook<T> T::*HookPtr>
void MMLruForgive::Container<T, HookPtr>::setConfig(const Config& newConfig) {
  lruMutex_->lock_combine([this, newConfig]() {
    config_ = newConfig;
    if (config_.lruInsertionPointSpec == 0 && insertionPoint_ != nullptr) {
      auto curr = insertionPoint_;
      while (tailSize_ != 0) {
        XDCHECK(curr != nullptr);
        unmarkTail(*curr);
        tailSize_--;
        curr = lru_.getNext(*curr);
      }
      insertionPoint_ = nullptr;
    }
    lruRefreshTime_.store(config_.lruRefreshTime, std::memory_order_relaxed);
    nextReconfigureTime_ = config_.mmReconfigureIntervalSecs.count() == 0
                               ? std::numeric_limits<Time>::max()
                               : static_cast<Time>(util::getCurrentTimeSec()) +
                                     config_.mmReconfigureIntervalSecs.count();
  });
}

template <typename T, MMLruForgive::Hook<T> T::*HookPtr>
typename MMLruForgive::Config
MMLruForgive::Container<T, HookPtr>::getConfig() const {
  return lruMutex_->lock_combine([this]() { return config_; });
}

template <typename T, MMLruForgive::Hook<T> T::*HookPtr>
void MMLruForgive::Container<T, HookPtr>::updateLruInsertionPoint() noexcept {
  if (config_.lruInsertionPointSpec == 0) {
    return;
  }
  if (insertionPoint_ == nullptr) {
    insertionPoint_ = lru_.getTail();
    tailSize_ = 0;
    if (insertionPoint_ != nullptr) {
      markTail(*insertionPoint_);
      tailSize_++;
    }
  }
  if (lru_.size() <= 1) {
    return;
  }
  XDCHECK_NE(reinterpret_cast<uintptr_t>(nullptr),
             reinterpret_cast<uintptr_t>(insertionPoint_));
  const auto expectedSize = lru_.size() >> config_.lruInsertionPointSpec;
  auto curr = insertionPoint_;
  while (tailSize_ < expectedSize && curr != lru_.getHead()) {
    curr = lru_.getPrev(*curr);
    markTail(*curr);
    tailSize_++;
  }
  while (tailSize_ > expectedSize && curr != lru_.getTail()) {
    unmarkTail(*curr);
    tailSize_--;
    curr = lru_.getNext(*curr);
  }
  insertionPoint_ = curr;
}

template <typename T, MMLruForgive::Hook<T> T::*HookPtr>
bool MMLruForgive::Container<T, HookPtr>::add(T& node) noexcept {
  const auto currTime = static_cast<Time>(util::getCurrentTimeSec());
  return lruMutex_->lock_combine([this, &node, currTime]() {
    if (node.isInMMContainer()) {
      return false;
    }
    if (config_.lruInsertionPointSpec == 0 || insertionPoint_ == nullptr) {
      lru_.linkAtHead(node);
    } else {
      lru_.insertBefore(*insertionPoint_, node);
    }
    node.markInMMContainer();
    setUpdateTime(node, currTime);
    unmarkAccessed(node);
    updateLruInsertionPoint();
    return true;
  });
}

// ============================================================================
// TARDIS forgiveness loop
// ============================================================================
// Caller MUST hold the lru mutex when calling this.
// Walks from the LRU tail, up to maxForgives iterations. For each tail
// candidate, checks whether it qualifies for forgiveness and whether its
// embedding similarity exceeds the configured threshold. Forgiven candidates
// are promoted to the head (MRU) and the loop continues. Stops on the first
// non-forgivable tail candidate, or after maxForgives promotions.
template <typename T, MMLruForgive::Hook<T> T::*HookPtr>
void MMLruForgive::Container<T, HookPtr>::applyForgivenessLocked() noexcept {
  int forgivesRemaining = config_.maxForgives;
  const double threshold = config_.forgiveThreshold;
  const int minAccess = config_.minAccessCount;
  const int topK = config_.max_K;

  while (forgivesRemaining > 0) {
    T* tail = lru_.getTail();
    if (tail == nullptr) {
      return;
    }
    nCandidates_++;

    uint64_t obj_id = hashKey(*tail);
    int access_count = embMgr_->get_access_count(obj_id);

    // Candidate must have enough accesses to have an embedding
    if (access_count < minAccess) {
      return;
    }
    if (!embMgr_->has_embedding(obj_id)) {
      return;
    }
    nQualified_++;

    double similarity = embMgr_->avg_top_k_similarity_to_recent(obj_id, topK);
    if (similarity < threshold) {
      return;  // Not forgivable, stop walking
    }

    // Forgive: promote tail to head and update its timestamp
    lru_.moveToHead(*tail);
    setUpdateTime(*tail, static_cast<Time>(util::getCurrentTimeSec()));
    if (isTail(*tail)) {
      unmarkTail(*tail);
      if (tailSize_ > 0) tailSize_--;
    }
    nForgive_++;
    forgivesRemaining--;
  }
}

template <typename T, MMLruForgive::Hook<T> T::*HookPtr>
typename MMLruForgive::Container<T, HookPtr>::LockedIterator
MMLruForgive::Container<T, HookPtr>::getEvictionIterator() const noexcept {
  LockHolder l(*lruMutex_);
  // Apply forgiveness loop under the lock before returning the iterator.
  // Note: this is a const method but applyForgivenessLocked mutates state.
  // The mutable embedding manager and atomic counters allow this; the LRU
  // list itself is mutated through const_cast since the callers expect the
  // iterator to reflect the post-forgiveness state.
  const_cast<Container*>(this)->applyForgivenessLocked();
  return LockedIterator{std::move(l), lru_.rbegin()};
}

template <typename T, MMLruForgive::Hook<T> T::*HookPtr>
template <typename F>
void MMLruForgive::Container<T, HookPtr>::withEvictionIterator(F&& fun) {
  if (config_.useCombinedLockForIterators) {
    lruMutex_->lock_combine([this, &fun]() {
      applyForgivenessLocked();
      fun(Iterator{lru_.rbegin()});
    });
  } else {
    LockHolder lck{*lruMutex_};
    applyForgivenessLocked();
    fun(Iterator{lru_.rbegin()});
  }
}

template <typename T, MMLruForgive::Hook<T> T::*HookPtr>
void MMLruForgive::Container<T, HookPtr>::ensureNotInsertionPoint(
    T& node) noexcept {
  if (&node == insertionPoint_) {
    insertionPoint_ = lru_.getPrev(*insertionPoint_);
    if (insertionPoint_ != nullptr) {
      tailSize_++;
      markTail(*insertionPoint_);
    } else {
      XDCHECK_EQ(lru_.size(), 1u);
    }
  }
}

template <typename T, MMLruForgive::Hook<T> T::*HookPtr>
void MMLruForgive::Container<T, HookPtr>::removeLocked(T& node) {
  ensureNotInsertionPoint(node);
  lru_.remove(node);
  unmarkAccessed(node);
  if (isTail(node)) {
    unmarkTail(node);
    tailSize_--;
  }
  node.unmarkInMMContainer();
  updateLruInsertionPoint();
  nEvict_++;
}

template <typename T, MMLruForgive::Hook<T> T::*HookPtr>
bool MMLruForgive::Container<T, HookPtr>::remove(T& node) noexcept {
  return lruMutex_->lock_combine([this, &node]() {
    if (!node.isInMMContainer()) {
      return false;
    }
    removeLocked(node);
    return true;
  });
}

template <typename T, MMLruForgive::Hook<T> T::*HookPtr>
void MMLruForgive::Container<T, HookPtr>::remove(Iterator& it) noexcept {
  T& node = *it;
  XDCHECK(node.isInMMContainer());
  ++it;
  removeLocked(node);
}

template <typename T, MMLruForgive::Hook<T> T::*HookPtr>
bool MMLruForgive::Container<T, HookPtr>::replace(T& oldNode,
                                                  T& newNode) noexcept {
  return lruMutex_->lock_combine([this, &oldNode, &newNode]() {
    if (!oldNode.isInMMContainer() || newNode.isInMMContainer()) {
      return false;
    }
    const auto updateTime = getUpdateTime(oldNode);
    lru_.replace(oldNode, newNode);
    oldNode.unmarkInMMContainer();
    newNode.markInMMContainer();
    setUpdateTime(newNode, updateTime);
    if (isAccessed(oldNode)) {
      markAccessed(newNode);
    } else {
      unmarkAccessed(newNode);
    }
    XDCHECK(!isTail(newNode));
    if (isTail(oldNode)) {
      markTail(newNode);
      unmarkTail(oldNode);
    } else {
      unmarkTail(newNode);
    }
    if (insertionPoint_ == &oldNode) {
      insertionPoint_ = &newNode;
    }
    return true;
  });
}

template <typename T, MMLruForgive::Hook<T> T::*HookPtr>
serialization::MMLruObject
MMLruForgive::Container<T, HookPtr>::saveState() const noexcept {
  serialization::MMLruConfig configObject;
  *configObject.lruRefreshTime() =
      lruRefreshTime_.load(std::memory_order_relaxed);
  *configObject.lruRefreshRatio() = config_.lruRefreshRatio;
  *configObject.updateOnWrite() = config_.updateOnWrite;
  *configObject.updateOnRead() = config_.updateOnRead;
  *configObject.tryLockUpdate() = config_.tryLockUpdate;
  *configObject.lruInsertionPointSpec() = config_.lruInsertionPointSpec;

  serialization::MMLruObject object;
  *object.config() = configObject;
  *object.compressedInsertionPoint() =
      compressor_.compress(insertionPoint_).saveState();
  *object.tailSize() = tailSize_;
  *object.lru() = lru_.saveState();
  return object;
}

template <typename T, MMLruForgive::Hook<T> T::*HookPtr>
MMContainerStat MMLruForgive::Container<T, HookPtr>::getStats() const noexcept {
  auto stat = lruMutex_->lock_combine([this]() {
    auto* tail = lru_.getTail();
    return folly::make_array(lru_.size(),
                             tail == nullptr ? 0 : getUpdateTime(*tail),
                             lruRefreshTime_.load(std::memory_order_relaxed));
  });
  return {stat[0], stat[1], stat[2], 0, 0, 0, 0};
}

template <typename T, MMLruForgive::Hook<T> T::*HookPtr>
void MMLruForgive::Container<T, HookPtr>::reconfigureLocked(
    const Time& currTime) {
  if (currTime < nextReconfigureTime_) {
    return;
  }
  nextReconfigureTime_ = currTime + config_.mmReconfigureIntervalSecs.count();
  auto stat = getEvictionAgeStatLocked(0);
  auto lruRefreshTime = std::min(
      std::max(config_.defaultLruRefreshTime,
               static_cast<uint32_t>(stat.warmQueueStat.oldestElementAge *
                                     config_.lruRefreshRatio)),
      kLruRefreshTimeCap);
  lruRefreshTime_.store(lruRefreshTime, std::memory_order_relaxed);
}

// Iterator Context Implementation
template <typename T, MMLruForgive::Hook<T> T::*HookPtr>
MMLruForgive::Container<T, HookPtr>::LockedIterator::LockedIterator(
    LockHolder l, const Iterator& iter) noexcept
    : Iterator(iter), l_(std::move(l)) {}

}  // namespace cachelib
}  // namespace facebook