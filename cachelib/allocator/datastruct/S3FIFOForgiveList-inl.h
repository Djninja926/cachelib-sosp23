/*
 * S3FIFOForgiveList-inl.h
 *
 * getEvictionCandidate is a copy of S3FIFOList::getEvictionCandidate with ONE
 * addition: at the small-FIFO (pfifo_) branch, when S3-FIFO would evict a
 * non-accessed object (freq 0), we call shouldForgive(). If forgivable, we
 * promote it to main (the same three lines S3-FIFO uses for an accessed
 * object) and continue the walk, instead of returning it for eviction.
 */
#pragma once

namespace facebook {
namespace cachelib {

template <typename T, AtomicDListHook<T> T::*HookPtr>
T* S3FIFOForgiveList<T, HookPtr>::getEvictionCandidate() noexcept {
  size_t listSize = pfifo_->size() + mfifo_->size();
  if (listSize == 0) {
    return nullptr;
  }

  int forgivesThisCall = 0;
  const int maxForgivesPerCall = 50;  // guarantee progress

  T* curr = nullptr;
  if (!hist_.initialized()) {
    LockHolder l(*mtx_);
    if (!hist_.initialized()) {
      hist_.setFIFOSize(listSize / 2);
      hist_.initHashtable();
    }
  }

  while (true) {
    if (pfifo_->size() > (double)(pfifo_->size() + mfifo_->size()) * pRatio_) {
      // evict from probationary FIFO
      curr = pfifo_->removeTail();
      if (curr == nullptr) {
        if (pfifo_->size() != 0) {
          printf("pfifo_->size() = %zu\n", pfifo_->size());
          abort();
        }
        continue;
      }
      if (pfifo_->isAccessed(*curr)) {
        // S3-FIFO: re-accessed in small FIFO -> promote to main.
        pfifo_->unmarkAccessed(*curr);
        XDCHECK(isProbationary(*curr));
        unmarkProbationary(*curr);
        markMain(*curr);
        mfifo_->linkAtHead(*curr);
      } else if (forgivesThisCall < maxForgivesPerCall && shouldForgive(*curr)) {
          forgivesThisCall++;
          if (markAccessed_) {
            // set accessed bit and keep in small: earns normal S3-FIFO promotion
            // on its next lap to the small-FIFO tail (via the isAccessed branch).
            pfifo_->markAccessed(*curr);
            pfifo_->linkAtHead(*curr);
          } else if (keepInSmall_) {
            pfifo_->linkAtHead(*curr);
          } else {
            unmarkProbationary(*curr);
            markMain(*curr);
            mfifo_->linkAtHead(*curr);
          }
        nForgive_.fetch_add(1, std::memory_order_relaxed);
      } else {
        // S3-FIFO: drop the one-hit-wonder (record in history, evict).
        hist_.insert(hashNode(*curr));
        nEvict_.fetch_add(1, std::memory_order_relaxed);
        return curr;
      }
    } else {
      curr = mfifo_->removeTail();
      if (curr == nullptr) {
        continue;
      }
      if (mfifo_->isAccessed(*curr)) {
        mfifo_->unmarkAccessed(*curr);
        mfifo_->linkAtHead(*curr);
      } else {
        nEvict_.fetch_add(1, std::memory_order_relaxed);
        return curr;
      }
    }
  }
}

}  // namespace cachelib
}  // namespace facebook