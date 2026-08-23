/*
 * MMS3FIFOForgive-inl.h
 *
 * Mirrors MMS3FIFO-inl.h. The ONLY functional addition is the
 * embMgr_->on_access(hashKey(node)) call at the top of recordAccess, so the
 * embedding updates on every access (matching MMLruForgive). Forgiveness at
 * eviction is handled inside S3FIFOForgiveList::getEvictionCandidate.
 */
#pragma once

namespace facebook {
namespace cachelib {

template <typename T, MMS3FIFOForgive::Hook<T> T::*HookPtr>
MMS3FIFOForgive::Container<T, HookPtr>::Container(
    serialization::MMS3FIFOObject object, PtrCompressor compressor)
    : qdlist_(*object.qdlist(), compressor), config_(*object.config()) {
  nextReconfigureTime_ = config_.mmReconfigureIntervalSecs.count() == 0
                             ? std::numeric_limits<Time>::max()
                             : static_cast<Time>(util::getCurrentTimeSec()) +
                                   config_.mmReconfigureIntervalSecs.count();
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

template <typename T, MMS3FIFOForgive::Hook<T> T::*HookPtr>
bool MMS3FIFOForgive::Container<T, HookPtr>::recordAccess(
    T& node, AccessMode mode) noexcept {
  // TARDIS: update the embedding on EVERY access, regardless of the
  // updateOnRead/updateOnWrite gate, matching MMLruForgive and the
  // libCacheSim reference (which records every event).
  embMgr_->on_access(hashKey(node), cachedShardOfKey(node));

  if ((mode == AccessMode::kWrite && !config_.updateOnWrite) ||
      (mode == AccessMode::kRead && !config_.updateOnRead)) {
    return false;
  }

  const auto curr = static_cast<Time>(util::getCurrentTimeSec());
  if (node.isInMMContainer()) {
    if (!isAccessed(node)) {
      markAccessed(node);
    }
    setUpdateTime(node, curr);
    return true;
  }
  return false;
}

template <typename T, MMS3FIFOForgive::Hook<T> T::*HookPtr>
cachelib::EvictionAgeStat
MMS3FIFOForgive::Container<T, HookPtr>::getEvictionAgeStat(
    uint64_t projectedLength) const noexcept {
  return lruMutex_->lock_combine([this, projectedLength]() {
    return getEvictionAgeStatLocked(projectedLength);
  });
}

template <typename T, MMS3FIFOForgive::Hook<T> T::*HookPtr>
cachelib::EvictionAgeStat
MMS3FIFOForgive::Container<T, HookPtr>::getEvictionAgeStatLocked(
    uint64_t projectedLength) const noexcept {
  EvictionAgeStat stat{};
  return stat;
}

template <typename T, MMS3FIFOForgive::Hook<T> T::*HookPtr>
void MMS3FIFOForgive::Container<T, HookPtr>::setConfig(
    const Config& newConfig) {
  lruMutex_->lock_combine([this, newConfig]() {
    config_ = newConfig;
    nextReconfigureTime_ = config_.mmReconfigureIntervalSecs.count() == 0
                               ? std::numeric_limits<Time>::max()
                               : static_cast<Time>(util::getCurrentTimeSec()) +
                                     config_.mmReconfigureIntervalSecs.count();
  });
}

template <typename T, MMS3FIFOForgive::Hook<T> T::*HookPtr>
typename MMS3FIFOForgive::Config
MMS3FIFOForgive::Container<T, HookPtr>::getConfig() const {
  return lruMutex_->lock_combine([this]() { return config_; });
}

template <typename T, MMS3FIFOForgive::Hook<T> T::*HookPtr>
bool MMS3FIFOForgive::Container<T, HookPtr>::add(T& node) noexcept {
  const auto currTime = static_cast<Time>(util::getCurrentTimeSec());
  if (node.isInMMContainer()) {
    return false;
  }
  qdlist_.add(node);
  unmarkAccessed(node);
  node.markInMMContainer();
  setUpdateTime(node, currTime);
  return true;
}

template <typename T, MMS3FIFOForgive::Hook<T> T::*HookPtr>
typename MMS3FIFOForgive::Container<T, HookPtr>::LockedIterator
MMS3FIFOForgive::Container<T, HookPtr>::getEvictionIterator() noexcept {
  return LockedIterator{&qdlist_};
}

template <typename T, MMS3FIFOForgive::Hook<T> T::*HookPtr>
void MMS3FIFOForgive::Container<T, HookPtr>::removeLocked(T& node) noexcept {
  LruType type = getLruType(node);
  switch (type) {
  case LruType::Prob:
    qdlist_.getListProbationary().remove(node);
    break;
  case LruType::Main:
    qdlist_.getListMain().remove(node);
    break;
  case LruType::NumTypes:
    XDCHECK(false);
  }
  node.unmarkInMMContainer();
  return;
}

template <typename T, MMS3FIFOForgive::Hook<T> T::*HookPtr>
bool MMS3FIFOForgive::Container<T, HookPtr>::remove(T& node) noexcept {
  return lruMutex_->lock_combine([this, &node]() {
    if (!node.isInMMContainer()) {
      return false;
    }
    removeLocked(node);
    return true;
  });
}

template <typename T, MMS3FIFOForgive::Hook<T> T::*HookPtr>
void MMS3FIFOForgive::Container<T, HookPtr>::remove(LockedIterator& it) noexcept {
  T& node = *it;
  XDCHECK(node.isInMMContainer());
  node.unmarkInMMContainer();
}

template <typename T, MMS3FIFOForgive::Hook<T> T::*HookPtr>
bool MMS3FIFOForgive::Container<T, HookPtr>::replace(T& oldNode,
                                                     T& newNode) noexcept {
  return lruMutex_->lock_combine([this, &oldNode, &newNode]() {
    if (!oldNode.isInMMContainer() || newNode.isInMMContainer()) {
      return false;
    }
    const auto updateTime = getUpdateTime(oldNode);
    LruType type = getLruType(oldNode);
    switch (type) {
    case LruType::Prob:
      markProbationary(newNode);
      qdlist_.getListProbationary().replace(oldNode, newNode);
      break;
    case LruType::Main:
      markMain(newNode);
      qdlist_.getListMain().replace(oldNode, newNode);
      break;
    case LruType::NumTypes:
      XDCHECK(false);
    }
    oldNode.unmarkInMMContainer();
    newNode.markInMMContainer();
    setUpdateTime(newNode, updateTime);
    if (isAccessed(oldNode)) {
      markAccessed(newNode);
    } else {
      unmarkAccessed(newNode);
    }
    return true;
  });
}

template <typename T, MMS3FIFOForgive::Hook<T> T::*HookPtr>
serialization::MMS3FIFOObject
MMS3FIFOForgive::Container<T, HookPtr>::saveState() const noexcept {
  serialization::MMS3FIFOConfig configObject;
  *configObject.updateOnWrite() = config_.updateOnWrite;
  *configObject.updateOnRead() = config_.updateOnRead;

  serialization::MMS3FIFOObject object;
  *object.config() = configObject;
  *object.qdlist() = qdlist_.saveState();
  return object;
}

template <typename T, MMS3FIFOForgive::Hook<T> T::*HookPtr>
MMContainerStat MMS3FIFOForgive::Container<T, HookPtr>::getStats()
    const noexcept {
  auto stat = lruMutex_->lock_combine(
      [this]() { return folly::make_array(qdlist_.size()); });
  return {stat[0], 0, 0, 0, 0, 0, 0};
}

template <typename T, MMS3FIFOForgive::Hook<T> T::*HookPtr>
void MMS3FIFOForgive::Container<T, HookPtr>::reconfigureLocked(
    const Time& currTime) {
  if (currTime < nextReconfigureTime_) {
    return;
  }
  nextReconfigureTime_ = currTime + config_.mmReconfigureIntervalSecs.count();
}

}  // namespace cachelib
}  // namespace facebook