/*
 * TARDIS Embedding Manager for CacheLib - Log-Structured (Option B + Stage 1 batching)
 *
 * Same as Option B (per-thread SPSC buffers + single applier doing an N-way
 * merge by seq), with one throughput change: the applier accumulates a
 * seq-ordered BATCH of up to kBatchSize events and applies the whole batch
 * under a SINGLE state-lock acquisition (applyBatch), instead of locking once
 * per event. This cuts lock acquire/release overhead by ~kBatchSize and keeps
 * the embedding state hot in cache across the batch.
 *
 * Semantics are identical to Option B: events are still applied in seq order,
 * with the same per-event logic. The only behavioral difference is that a
 * low-seq event arriving while a batch is mid-apply is deferred to the next
 * batch, giving a cross-batch reorder distance of ~kBatchSize (kept small so
 * miss ratio matches; the Option A window sweep showed 256 matches B exactly).
 *
 * Tradeoff: larger kBatchSize = fewer lock acquisitions (faster applier) but
 * longer eviction-reader stalls (the applier holds the write lock per batch).
 */

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <list>
#include <memory>
#include <random>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <folly/ProducerConsumerQueue.h>

namespace facebook {
namespace cachelib {

class TARDISEmbeddingManager {
 public:
  static constexpr int K = 8;
  static constexpr int D = 8;

  using EmbeddingArray = std::array<std::array<double, D>, K>;

  struct AccessEvent {
    uint64_t obj_id;
    uint64_t seq;
  };

  static constexpr int kMaxThreads = 64;
  static constexpr size_t kQueueCapacity = 1u << 20;
  // Number of events applied per lock acquisition. Tradeoff: throughput (more)
  // vs eviction-reader stall + cross-batch reorder distance (less). 256 matched
  // Option B miss ratio exactly in the window sweep.
  static constexpr size_t kBatchSize = 256;

  TARDISEmbeddingManager(double lr,
                         double ctx_speed,
                         int recent_window,
                         int min_access_count,
                         int64_t max_entries = -1,
                         uint64_t seed = 42,
                         int embed_threshold = -1)
      : lr_(lr),
        ctx_speed_(ctx_speed),
        recent_window_(recent_window),
        min_access_count_(min_access_count),
        max_entries_(max_entries),
        embed_threshold_(embed_threshold < 0 ? min_access_count : embed_threshold),
        rng_(seed) {
    init_context();
    recent_embeddings_.resize(recent_window_);
    recent_ids_.resize(recent_window_);
    recent_valid_.resize(recent_window_, false);

    for (int i = 0; i < kMaxThreads; i++) {
      buffers_[i] =
          std::make_unique<folly::ProducerConsumerQueue<AccessEvent>>(
              kQueueCapacity);
    }
    applier_ = std::thread(&TARDISEmbeddingManager::applierLoop, this);
  }

  ~TARDISEmbeddingManager() {
    stop_.store(true, std::memory_order_release);
    if (applier_.joinable()) {
      applier_.join();
    }
  }

  TARDISEmbeddingManager(const TARDISEmbeddingManager&) = delete;
  TARDISEmbeddingManager& operator=(const TARDISEmbeddingManager&) = delete;

  // ---- HOT PATH ----
  void on_access(uint64_t obj_id) {
    const int tid = threadSlot();
    AccessEvent ev{obj_id, current_trace_seq};
    while (!buffers_[tid]->write(ev)) {
      std::this_thread::yield();
    }
  }

  // ---- READ PATH (eviction-time forgiveness) ----
  double avg_top_k_similarity_to_recent(uint64_t obj_id, int top_k = 3) {
    std::shared_lock<std::shared_mutex> lock(state_mutex_);
    auto it = embeddings_.find(obj_id);
    if (it == embeddings_.end()) return 0.0;

    std::vector<double> sims;
    sims.reserve(recent_count_);
    for (int i = 0; i < recent_count_; i++) {
      if (recent_ids_[i] == obj_id) continue;
      if (!recent_valid_[i]) continue;
      const auto& emb = it->second;
      const auto& recent_emb = recent_embeddings_[i];
      double sum = 0.0;
      for (int k = 0; k < K; k++) {
        sum += cosine_similarity(emb[k].data(), recent_emb[k].data());
      }
      sims.push_back(sum / K);
    }
    if (sims.empty()) return 0.0;
    std::sort(sims.begin(), sims.end(), std::greater<double>());
    int count = std::min(top_k, static_cast<int>(sims.size()));
    double total = 0.0;
    for (int i = 0; i < count; i++) total += sims[i];
    return total / count;
  }

  int get_access_count(uint64_t obj_id) const {
    std::shared_lock<std::shared_mutex> lock(state_mutex_);
    auto it = access_count_.find(obj_id);
    return (it != access_count_.end()) ? it->second : 0;
  }

  bool has_embedding(uint64_t obj_id) const {
    std::shared_lock<std::shared_mutex> lock(state_mutex_);
    return embeddings_.count(obj_id) > 0;
  }

  size_t get_num_embeddings() const {
    std::shared_lock<std::shared_mutex> lock(state_mutex_);
    return embeddings_.size();
  }

  static void set_trace_seq(uint64_t s) { current_trace_seq = s; }

 private:
  static constexpr uint64_t kNoSeq = ~0ull;

  static int threadSlot() {
    thread_local int slot = -1;
    if (slot < 0) {
      slot = next_slot_.fetch_add(1, std::memory_order_relaxed);
      slot %= kMaxThreads;
    }
    return slot;
  }

  // ---- BACKGROUND APPLIER (batched) ----
  void applierLoop() {
    std::vector<AccessEvent> head(kMaxThreads);
    std::vector<bool> haveHead(kMaxThreads, false);
    std::vector<AccessEvent> batch;
    batch.reserve(kBatchSize);

    auto refillHead = [&](int i) {
      if (haveHead[i]) return;
      AccessEvent ev;
      if (buffers_[i]->read(ev)) {
        head[i] = ev;
        haveHead[i] = true;
      }
    };

    while (true) {
      // Top up all heads, then build a seq-ordered batch by repeated global-min.
      for (int i = 0; i < kMaxThreads; i++) refillHead(i);

      batch.clear();
      while (batch.size() < kBatchSize) {
        int minIdx = -1;
        uint64_t minSeq = kNoSeq;
        for (int i = 0; i < kMaxThreads; i++) {
          if (haveHead[i] && head[i].seq < minSeq) {
            minSeq = head[i].seq;
            minIdx = i;
          }
        }
        if (minIdx < 0) break;  // nothing available right now
        batch.push_back(head[minIdx]);
        haveHead[minIdx] = false;
        refillHead(minIdx);  // refill only the consumed buffer
      }

      if (!batch.empty()) {
        applyBatch(batch);
        continue;
      }

      // Batch empty: nothing available.
      if (stop_.load(std::memory_order_acquire)) {
        bool any = false;
        for (int i = 0; i < kMaxThreads; i++) {
          refillHead(i);
          if (haveHead[i]) any = true;
        }
        if (!any) break;
        continue;
      }
      std::this_thread::yield();
    }
  }

  // Apply a whole batch under ONE lock acquisition. Events are pre-sorted by
  // seq by the caller, so this preserves seq-order application.
  void applyBatch(const std::vector<AccessEvent>& batch) {
    std::unique_lock<std::shared_mutex> lock(state_mutex_);
    for (const auto& ev : batch) {
      applyEventLocked(ev);
    }
  }

  // Per-event logic, identical to single-threaded on_access. Caller holds lock.
  void applyEventLocked(const AccessEvent& ev) {
    const uint64_t obj_id = ev.obj_id;

    if (++perturb_counter_ >= 10) {
      perturb_counter_ = 0;
      update_context_batch();
    }

    int& count = access_count_[obj_id];
    count++;

    auto it = embeddings_.find(obj_id);
    bool has_emb = (it != embeddings_.end());

    if (has_emb) {
      update_embedding(obj_id);
      if (max_entries_ >= 0) {
        move_to_front(obj_id);
      }
      update_recent(obj_id);
    } else if (count == embed_threshold_) {
      if (max_entries_ >= 0) {
        while (static_cast<int64_t>(embeddings_.size()) >= max_entries_ &&
               !lru_list_.empty()) {
          evict_lru();
        }
        init_embedding(obj_id);
        add_to_lru(obj_id);
      } else {
        init_embedding(obj_id);
      }
      update_recent(obj_id);
    }
  }

  // ---- embedding-cache LRU (capped/MCACHE mode) ----
  void add_to_lru(uint64_t obj_id) {
    lru_list_.push_front(obj_id);
    lru_map_[obj_id] = lru_list_.begin();
  }
  void move_to_front(uint64_t obj_id) {
    auto it = lru_map_.find(obj_id);
    if (it != lru_map_.end()) {
      lru_list_.erase(it->second);
      lru_list_.push_front(obj_id);
      it->second = lru_list_.begin();
    }
  }
  void evict_lru() {
    if (lru_list_.empty()) return;
    uint64_t lru_id = lru_list_.back();
    lru_list_.pop_back();
    lru_map_.erase(lru_id);
    embeddings_.erase(lru_id);
  }

  void init_context() {
    for (int k = 0; k < K; k++)
      for (int d = 0; d < D; d++) context_[k][d] = normal_dist_(rng_);
  }
  void update_context_batch() {
    double effective_speed = 10.0 * ctx_speed_;
    double decay = std::sqrt(1.0 - effective_speed);
    double noise_scale = std::sqrt(effective_speed);
    for (int k = 0; k < K; k++)
      for (int d = 0; d < D; d++)
        context_[k][d] =
            decay * context_[k][d] + noise_scale * normal_dist_(rng_);
  }
  void init_embedding(uint64_t obj_id) {
    auto& emb = embeddings_[obj_id];
    for (int k = 0; k < K; k++)
      for (int d = 0; d < D; d++) emb[k][d] = normal_dist_(rng_);
  }
  void update_embedding(uint64_t obj_id) {
    auto& emb = embeddings_[obj_id];
    double decay = std::sqrt(1.0 - lr_);
    double ctx_scale = std::sqrt(lr_);
    for (int k = 0; k < K; k++)
      for (int d = 0; d < D; d++)
        emb[k][d] = decay * emb[k][d] + ctx_scale * context_[k][d];
  }
  void update_recent(uint64_t obj_id) {
    recent_ids_[recent_idx_] = obj_id;
    auto it = embeddings_.find(obj_id);
    if (it != embeddings_.end()) {
      recent_embeddings_[recent_idx_] = it->second;
      recent_valid_[recent_idx_] = true;
    } else {
      recent_valid_[recent_idx_] = false;
    }
    recent_idx_ = (recent_idx_ + 1) % recent_window_;
    if (recent_count_ < recent_window_) recent_count_++;
  }
  static double cosine_similarity(const double* a, const double* b) {
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (int i = 0; i < D; i++) {
      dot += a[i] * b[i];
      norm_a += a[i] * a[i];
      norm_b += b[i] * b[i];
    }
    double denom = std::sqrt(norm_a * norm_b);
    if (denom < 1e-10) return 0.0;
    return dot / denom;
  }

  // ---- parameters ----
  double lr_;
  double ctx_speed_;
  int recent_window_;
  int min_access_count_;
  int embed_threshold_;
  int64_t max_entries_;

  // ---- embedding state (applier-owned; read under mutex) ----
  mutable std::shared_mutex state_mutex_;
  double context_[K][D];
  std::unordered_map<uint64_t, EmbeddingArray> embeddings_;
  std::unordered_map<uint64_t, int> access_count_;
  std::vector<EmbeddingArray> recent_embeddings_;
  std::vector<uint64_t> recent_ids_;
  std::vector<bool> recent_valid_;
  int recent_idx_ = 0;
  int recent_count_ = 0;
  int perturb_counter_ = 0;
  std::mt19937 rng_;
  std::normal_distribution<double> normal_dist_{0.0, 1.0};
  std::list<uint64_t> lru_list_;
  std::unordered_map<uint64_t, std::list<uint64_t>::iterator> lru_map_;

  // ---- log infrastructure ----
  std::unique_ptr<folly::ProducerConsumerQueue<AccessEvent>>
      buffers_[kMaxThreads];
  std::thread applier_;
  std::atomic<bool> stop_{false};

  static std::atomic<int> next_slot_;
  static thread_local uint64_t current_trace_seq;
};

inline std::atomic<int> TARDISEmbeddingManager::next_slot_{0};
inline thread_local uint64_t TARDISEmbeddingManager::current_trace_seq = 0;

}  // namespace cachelib
}  // namespace facebook