/*
 * TARDIS Embedding Manager for CacheLib
 * Ported from Aryan Khatri's LRUForgiveEmbCacheEmbeddingManager (libCacheSim)
 *
 * Maintains per-object embeddings and a global context vector that evolves
 * as a random walk with Gaussian noise. Embeddings of co-occurring objects
 * become similar, allowing cosine similarity to predict near-future reuse.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <list>
#include <random>
#include <unordered_map>
#include <vector>

namespace facebook {
namespace cachelib {

class TARDISEmbeddingManager {
 public:
  // Number of parallel context vectors (K) and dimensions per vector (D).
  // These match Aryan's reference implementation defaults.
  static constexpr int K = 8;
  static constexpr int D = 8;

  using EmbeddingArray = std::array<std::array<double, D>, K>;

  // Parameters match Aryan's defaults from LRUForgiveEmbCache.cpp:
  //   lr=0.2, ctx_speed=0.001, recent_window=16, min_access=2,
  //   max_entries=-1 (unlimited), seed=42
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
  }

  // Called on every cache access. Updates the object's embedding toward the
  // current context, advances the context periodically, and tracks the object
  // in the recent-access buffer.
  void on_access(uint64_t obj_id) {
    // Batched context update every 10 accesses (variance-preserving)
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
        move_to_front(obj_id);  // LRU on embedding cache (capped mode)
      }
      update_recent(obj_id);
    } else if (count == min_access_count_) {
      if (max_entries_ >= 0) {
        // Capped mode: evict LRU before adding new embedding if at capacity
        while (static_cast<int64_t>(embeddings_.size()) >= max_entries_ &&
               !lru_list_.empty()) {
          evict_lru();
        }
        init_embedding(obj_id);
        add_to_lru(obj_id);
      } else {
        // Unlimited mode (FORGIVE_CLEAN): just add without LRU tracking
        init_embedding(obj_id);
      }
      update_recent(obj_id);
    }
  }

  // Returns the average of the top-k cosine similarities between this object's
  // embedding and the embeddings of objects in the recent-access buffer.
  double avg_top_k_similarity_to_recent(uint64_t obj_id, int top_k = 3) {
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
    auto it = access_count_.find(obj_id);
    return (it != access_count_.end()) ? it->second : 0;
  }

  bool has_embedding(uint64_t obj_id) const {
    return embeddings_.count(obj_id) > 0;
  }

  size_t get_num_embeddings() const { return embeddings_.size(); }

 private:
  // LRU list management for the embedding cache (capped mode only)
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

  // Initialize context vectors ~ N(0, I)
  void init_context() {
    for (int k = 0; k < K; k++) {
      for (int d = 0; d < D; d++) {
        context_[k][d] = normal_dist_(rng_);
      }
    }
  }

  // Variance-preserving context update (batched every 10 accesses)
  //   C = sqrt(1 - 10*speed) * C + sqrt(10*speed) * N(0, I)
  void update_context_batch() {
    double effective_speed = 10.0 * ctx_speed_;
    double decay = std::sqrt(1.0 - effective_speed);
    double noise_scale = std::sqrt(effective_speed);
    for (int k = 0; k < K; k++) {
      for (int d = 0; d < D; d++) {
        context_[k][d] = decay * context_[k][d] + noise_scale * normal_dist_(rng_);
      }
    }
  }

  // Initialize embedding ~ N(0, I)
  void init_embedding(uint64_t obj_id) {
    auto& emb = embeddings_[obj_id];
    for (int k = 0; k < K; k++) {
      for (int d = 0; d < D; d++) {
        emb[k][d] = normal_dist_(rng_);
      }
    }
  }

  // Variance-preserving embedding update toward current context
  //   E = sqrt(1 - lr) * E + sqrt(lr) * C
  void update_embedding(uint64_t obj_id) {
    auto& emb = embeddings_[obj_id];
    double decay = std::sqrt(1.0 - lr_);
    double ctx_scale = std::sqrt(lr_);
    for (int k = 0; k < K; k++) {
      for (int d = 0; d < D; d++) {
        emb[k][d] = decay * emb[k][d] + ctx_scale * context_[k][d];
      }
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

  // Parameters
  double lr_;
  double ctx_speed_;
  int recent_window_;
  int min_access_count_;
  int64_t max_entries_;  // -1 means unlimited (FORGIVE_CLEAN mode)

  // Core state
  double context_[K][D];
  std::unordered_map<uint64_t, EmbeddingArray> embeddings_;
  std::unordered_map<uint64_t, int> access_count_;

  // Recent access buffer (circular)
  std::vector<EmbeddingArray> recent_embeddings_;
  std::vector<uint64_t> recent_ids_;
  std::vector<bool> recent_valid_;
  int recent_idx_ = 0;
  int recent_count_ = 0;
  int perturb_counter_ = 0;

  // RNG
  std::mt19937 rng_;
  std::normal_distribution<double> normal_dist_{0.0, 1.0};

  // LRU tracking for embedding cache (capped mode only)
  std::list<uint64_t> lru_list_;
  std::unordered_map<uint64_t, std::list<uint64_t>::iterator> lru_map_;
};

}  // namespace cachelib
}  // namespace facebook