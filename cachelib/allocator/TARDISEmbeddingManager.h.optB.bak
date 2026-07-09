/*
 * TARDIS Embedding Manager for CacheLib - Log-Structured (Option B)
 * Ported from Aryan Khatri's LRUForgiveEmbCacheEmbeddingManager (libCacheSim)
 *
 * Log-structured lazy update design:
 *   - Hot path (on_access): appends an AccessEvent to the calling thread's own
 *     single-producer-single-consumer ring buffer. No lock, no embedding math.
 *   - Background applier thread: drains all per-thread buffers in sequence order
 *     (N-way merge by seq) and performs the actual embedding/context/recency
 *     updates. Single writer, so no lock is needed on the embedding state.
 *   - Read methods (eviction-time forgiveness): single-writer/multi-reader,
 *     protected by a low-traffic shared_mutex. NOT on the hot path.
 *
 * v1: free-running applier, strict in-order application (apply global-min seq
 * next). Seq is a global atomic counter (enqueue order, not true trace order);
 * sufficient because strict application serializes all updates through one
 * total order. True trace-order seq is a later refinement if needed.
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

  // A single logged access event. 16 bytes.
  struct AccessEvent {
    uint64_t obj_id;
    uint64_t seq;
  };

  // Max worker threads we support. Each gets its own SPSC buffer.
  static constexpr int kMaxThreads = 64;
  // Per-thread ring buffer capacity. Generously sized so the free-running
  // applier rarely causes a producer to spin. ~16 bytes * 1M = 16MB/thread.
  static constexpr size_t kQueueCapacity = 1u << 20;

  TARDISEmbeddingManager(double lr,
                         double ctx_speed,
                         int recent_window,
                         int min_access_count,
                         int64_t max_entries = -1,
                         uint64_t seed = 42)
      : lr_(lr),
        ctx_speed_(ctx_speed),
        recent_window_(recent_window),
        min_access_count_(min_access_count),
        max_entries_(max_entries),
        rng_(seed) {
    init_context();
    recent_embeddings_.resize(recent_window_);
    recent_ids_.resize(recent_window_);
    recent_valid_.resize(recent_window_, false);

    // Allocate per-thread SPSC buffers up front.
    for (int i = 0; i < kMaxThreads; i++) {
      buffers_[i] =
          std::make_unique<folly::ProducerConsumerQueue<AccessEvent>>(
              kQueueCapacity);
      buf_active_[i].store(false, std::memory_order_relaxed);
      buf_head_seq_[i] = kNoSeq;
    }

    // Start the background applier.
    applier_ = std::thread(&TARDISEmbeddingManager::applierLoop, this);
  }

  ~TARDISEmbeddingManager() {
    // Signal shutdown and let the applier drain whatever remains.
    stop_.store(true, std::memory_order_release);
    if (applier_.joinable()) {
      applier_.join();
    }
  }

  TARDISEmbeddingManager(const TARDISEmbeddingManager&) = delete;
  TARDISEmbeddingManager& operator=(const TARDISEmbeddingManager&) = delete;

  // ---- HOT PATH ----
  // Append the access to this thread's own buffer. No lock, no math.
  // void on_access(uint64_t obj_id) {
  //   const int tid = threadSlot();
  //   const uint64_t seq = seq_counter_.fetch_add(1, std::memory_order_relaxed);
  //   AccessEvent ev{obj_id, seq};
  //   // SPSC write. If the buffer is full (applier fell behind), spin briefly.
  //   // With kQueueCapacity sized generously this is rare.
  //   while (!buffers_[tid]->write(ev)) {
  //     std::this_thread::yield();
  //   }
  // }
  void on_access(uint64_t obj_id) {
    const int tid = threadSlot();
    AccessEvent ev{obj_id, current_trace_seq};   // true trace order, not atomic counter
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

  // Assign each calling thread a stable slot index in [0, kMaxThreads).
  static int threadSlot() {
    thread_local int slot = -1;
    if (slot < 0) {
      slot = next_slot_.fetch_add(1, std::memory_order_relaxed);
      // If we exceed kMaxThreads, wrap (collision: two threads share a buffer,
      // which is still correct because the applier reads by seq, just with a
      // small ordering imperfection). Should not happen in our sweeps (<=16).
      slot %= kMaxThreads;
    }
    // Mark the slot active so the applier knows to drain it.
    return slot;
  }

  // ---- BACKGROUND APPLIER ----
  // Strict in-order N-way merge: repeatedly find the buffer whose head has the
  // global-minimum seq and apply it. Free-running; drains remaining events on
  // shutdown.
  void applierLoop() {
    // Local cache of each buffer's current head event (peeked but not yet
    // applied). buf_head_seq_[i] == kNoSeq means "no pending head for i".
    std::vector<AccessEvent> head(kMaxThreads);
    std::vector<bool> haveHead(kMaxThreads, false);

    auto refillHead = [&](int i) {
      if (haveHead[i]) return;
      AccessEvent ev;
      if (buffers_[i]->read(ev)) {
        head[i] = ev;
        haveHead[i] = true;
      }
    };

    while (true) {
      // Try to refill any missing heads.
      for (int i = 0; i < kMaxThreads; i++) refillHead(i);

      // Find the buffer with the global-minimum seq among available heads.
      int minIdx = -1;
      uint64_t minSeq = kNoSeq;
      for (int i = 0; i < kMaxThreads; i++) {
        if (haveHead[i] && head[i].seq < minSeq) {
          minSeq = head[i].seq;
          minIdx = i;
        }
      }

      if (minIdx < 0) {
        // Nothing available right now.
        if (stop_.load(std::memory_order_acquire)) {
          // One more sweep to catch late arrivals, then exit if still empty.
          bool any = false;
          for (int i = 0; i < kMaxThreads; i++) {
            refillHead(i);
            if (haveHead[i]) any = true;
          }
          if (!any) break;
          continue;
        }
        std::this_thread::yield();
        continue;
      }

      // STRICT ORDERING NOTE: ideally we would wait until every still-active
      // producer has a head before committing to the global min, to guarantee
      // we never apply seq=k+1 before a not-yet-arrived seq=k. In v1 we apply
      // the min across currently-available heads. Because each producer's own
      // events are monotonic in seq and we always take the global min of what's
      // present, the only reordering possible is when a slow producer's lower
      // seq has not yet landed. We accept that small window in v1 and measure;
      // tightening it (barrier on all active producers) is the strict-vs-
      // approximate experiment.
      applyEvent(head[minIdx]);
      haveHead[minIdx] = false;
    }
  }

  // Apply a single access event to the embedding state. Mirrors the original
  // single-threaded on_access logic exactly. Holds the state mutex so eviction
  // readers see consistent data.
  void applyEvent(const AccessEvent& ev) {
    std::unique_lock<std::shared_mutex> lock(state_mutex_);
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
    } else if (count == min_access_count_) {
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
  int64_t max_entries_;

  // ---- embedding state (owned exclusively by the applier; read under mutex) ----
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
  std::atomic<bool> buf_active_[kMaxThreads];
  uint64_t buf_head_seq_[kMaxThreads];
  std::atomic<uint64_t> seq_counter_{0};
  std::thread applier_;
  std::atomic<bool> stop_{false};

  static std::atomic<int> next_slot_;
  static thread_local uint64_t current_trace_seq;
};

// Definition of the static thread-slot allocator.
inline std::atomic<int> TARDISEmbeddingManager::next_slot_{0};
inline thread_local uint64_t TARDISEmbeddingManager::current_trace_seq = 0;

}  // namespace cachelib
}  // namespace facebook