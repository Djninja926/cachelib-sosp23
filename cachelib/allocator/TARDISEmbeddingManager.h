/*
 * TARDIS Embedding Manager for CacheLib - Log-Structured (Option B + Stage 1 batching)
 *
 * Per-thread SPSC buffers + a single background applier doing an N-way merge by
 * seq, applying seq-ordered batches of up to kBatchSize under one state-lock
 * acquisition. Includes the SIMD read-path optimizations (norms cached in
 * EmbeddingData, single-pass bounded top-3, cosine taking precomputed norms).
 *
 * WAIT STRATEGY (adaptive backoff): bare yield() busy-waits burned ~84% of
 * syscall time on sched_yield (producer/consumer rate imbalance turning into a
 * yield storm). Both idle sites now use idle_backoff: a cheap spin first (no
 * syscall, absorbs short bursts), then a few yields, then a short sleep once
 * genuinely idle. This roughly doubled throughput with zero fidelity change
 * (it touches only how idle threads wait, never what is applied). A blocking
 * condition-variable variant was tried and lost: at high thread counts no
 * thread is deeply idle, so CV signaling became a futex storm; brief spinning
 * wins in this tight rate-matched regime. It adds no threads, the only lever
 * that helps on this (40-core, pool-heavy) box.
 */

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include <cstdlib>

#include <folly/ProducerConsumerQueue.h>

namespace facebook {
namespace cachelib {

class TARDISEmbeddingManager {
 public:
  static constexpr int K = 8;
  static constexpr int D = 8;

  // OPTIMIZATION 1: Cache the norms alongside the vectors to eliminate
  // std::sqrt and redundant calculations on the eviction path.
  struct EmbeddingData {
    std::array<std::array<double, D>, K> vectors;
    std::array<double, K> norms;
  };

  struct AccessEvent {
    uint64_t obj_id;
    uint64_t seq;
  };

  static constexpr int kMaxThreads = 64;
  static constexpr size_t kQueueCapacity = 1u << 20;
  static constexpr size_t kBatchSize = 256;

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
    if (const char* r = std::getenv("TARDIS_SAMPLE_RATE")) sample_rate_ = std::atof(r);
    if (const char* m = std::getenv("TARDIS_SAMPLE_MODE"))  sample_mode_ = std::atoi(m);
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

  void on_access(uint64_t obj_id) {
    std::unique_lock<std::shared_mutex> lock(state_mutex_);
    // Variant 1 (mode 0): subsample the entire update.
    if (sample_mode_ == 0 && sample_rate_ < 1.0) {
      if (std::uniform_real_distribution<double>(0.0, 1.0)(rng_) >= sample_rate_) {
        return;  // skip whole event: no count, no recency, no embedding
      }
    }
    // Variant 2 (mode 2): deterministic global every-Nth access, skip whole event.
    if (sample_mode_ == 2 && sample_rate_ < 1.0) {
      long N = std::llround(1.0 / sample_rate_);   // 0.25 -> every 4th, 0.1 -> every 10th
      if (N < 1) N = 1;
      if ((++sample_counter_ % (uint64_t)N) != 0) {
        return;  // skip unless this is the Nth access
      }
    }
    applyEventLocked(AccessEvent{obj_id, current_trace_seq});
  }

  // ---- READ PATH (eviction-time forgiveness) ----
  double avg_top_k_similarity_to_recent(uint64_t obj_id, int top_k = 3) {
    std::shared_lock<std::shared_mutex> lock(state_mutex_);
    auto it = embeddings_.find(obj_id);
    if (it == embeddings_.end()) return 0.0;

    // OPTIMIZATION 2: Bounded top-k in a single pass. 
    // Removes std::vector allocation and std::sort. Caps top_k at 3.
    double top[3] = {-2.0, -2.0, -2.0}; // Cosine similarity is >= -1.0
    int valid_count = 0;
    const auto& cand = it->second;

    for (int i = 0; i < recent_count_; i++) {
      if (recent_ids_[i] == obj_id) continue;
      if (!recent_valid_[i]) continue;
      
      const auto& recent_emb = recent_embeddings_[i];
      double sum = 0.0;
      for (int k = 0; k < K; k++) {
        sum += cosine_similarity(cand.vectors[k].data(), 
                                 recent_emb.vectors[k].data(), 
                                 cand.norms[k], 
                                 recent_emb.norms[k]);
      }
      double avg_sim = sum / K;
      valid_count++;

      // Shift down to maintain the top 3
      if (avg_sim > top[0]) {
        top[2] = top[1]; top[1] = top[0]; top[0] = avg_sim;
      } else if (avg_sim > top[1]) {
        top[2] = top[1]; top[1] = avg_sim;
      } else if (avg_sim > top[2]) {
        top[2] = avg_sim;
      }
    }

    if (valid_count == 0) return 0.0;
    
    // Safety cap incase a user passes top_k > 3, we cap at the array bounds
    int count = std::min(top_k, std::min(3, valid_count)); 
    double total = 0.0;
    for (int i = 0; i < count; i++) total += top[i];
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

  // Adaptive idle backoff: cheap spins first (no syscall, absorbs short
  // bursts), then a few yields, then a short sleep once genuinely idle. Tuned
  // to reach the sleep tier soon after the burst window so the residual
  // sched_yield is small, while keeping the cheap-spin tier that makes this
  // fast. `n` is the caller's consecutive-idle count (reset to 0 on progress).
  static inline void idle_backoff(uint32_t n) {
    if (n < 64) {
      for (volatile int i = 0; i < 32; i++) {
      }
    } else if (n < 512) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
  }

  // ---- BACKGROUND APPLIER (batched) ----
  void applierLoop() {
    std::vector<AccessEvent> head(kMaxThreads);
    std::vector<bool> haveHead(kMaxThreads, false);
    std::vector<AccessEvent> batch;
    batch.reserve(kBatchSize);
    uint32_t idle = 0;

    auto refillHead = [&](int i) {
      if (haveHead[i]) return;
      AccessEvent ev;
      if (buffers_[i]->read(ev)) {
        head[i] = ev;
        haveHead[i] = true;
      }
    };

    while (true) {
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
        if (minIdx < 0) break;  
        batch.push_back(head[minIdx]);
        haveHead[minIdx] = false;
        refillHead(minIdx);  
      }

      if (!batch.empty()) {
        idle = 0;
        applyBatch(batch);
        continue;
      }

      if (stop_.load(std::memory_order_acquire)) {
        bool any = false;
        for (int i = 0; i < kMaxThreads; i++) {
          refillHead(i);
          if (haveHead[i]) any = true;
        }
        if (!any) break;
        continue;
      }
      idle_backoff(idle++);
    }
  }

  void applyBatch(const std::vector<AccessEvent>& batch) {
    std::unique_lock<std::shared_mutex> lock(state_mutex_);
    for (const auto& ev : batch) {
      applyEventLocked(ev);
    }
  }

  void applyEventLocked(const AccessEvent& ev) {
    const uint64_t obj_id = ev.obj_id;

    // Variant 2 (mode 1): subsample only the heavy embedding math. access_count
    // still increments every access; context advance, embedding update/init,
    // and recency are skipped on non-sampled accesses.
    bool doEmbedding = true;
    if (sample_mode_ == 1 && sample_rate_ < 1.0) {
      doEmbedding =
          (std::uniform_real_distribution<double>(0.0, 1.0)(rng_) < sample_rate_);
    }

    if (doEmbedding && ++perturb_counter_ >= 10) {
      perturb_counter_ = 0;
      update_context_batch();
    }

    int& count = access_count_[obj_id];
    count++;

    if (!doEmbedding) {
      return;  // bookkeeping only; skip embedding update/init and recency
    }

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
    for (int k = 0; k < K; k++) {
      double sq_norm = 0.0;
      for (int d = 0; d < D; d++) {
        double val = normal_dist_(rng_);
        emb.vectors[k][d] = val;
        sq_norm += val * val;
      }
      // Cache the computed norm immediately
      emb.norms[k] = std::sqrt(sq_norm); 
    }
  }

  void update_embedding(uint64_t obj_id) {
    auto& emb = embeddings_[obj_id];
    double decay = std::sqrt(1.0 - lr_);
    double ctx_scale = std::sqrt(lr_);
    
    for (int k = 0; k < K; k++) {
      double sq_norm = 0.0;
      for (int d = 0; d < D; d++) {
        emb.vectors[k][d] = decay * emb.vectors[k][d] + ctx_scale * context_[k][d];
        sq_norm += emb.vectors[k][d] * emb.vectors[k][d];
      }
      // Cache the computed norm immediately
      emb.norms[k] = std::sqrt(sq_norm) + 1e-10;
    }
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

  // Refactored to accept precomputed norms
  static double cosine_similarity(const double* a, const double* b, double norm_a, double norm_b) {
    double dot = 0.0;
    for (int i = 0; i < D; i++) dot += a[i] * b[i];
    double denom = norm_a * norm_b;
    if (denom < 1e-10) return 0.0;   // preserve old suppression behavior
    return dot / denom;
  }

  // ---- parameters ----
  double lr_;
  double ctx_speed_;
  int recent_window_;
  int min_access_count_;
  int64_t max_entries_;
  double sample_rate_ = 1.0;   // TARDIS_SAMPLE_RATE
  int    sample_mode_ = 0;     // TARDIS_SAMPLE_MODE: 0=skip whole event, 1=skip only embedding math
  uint64_t sample_counter_ = 0;   // for mode 2 deterministic every-Nth

  // ---- embedding state (applier-owned; read under mutex) ----
  mutable std::shared_mutex state_mutex_;
  double context_[K][D];
  std::unordered_map<uint64_t, EmbeddingData> embeddings_;
  std::unordered_map<uint64_t, int> access_count_;
  std::vector<EmbeddingData> recent_embeddings_;
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