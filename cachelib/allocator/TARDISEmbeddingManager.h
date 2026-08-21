/*
 * TARDIS Embedding Manager for CacheLib - Log-Structured (Option B + Stage 1 batching)
 *
 * Same as Option B (per-thread SPSC buffers + single applier doing an N-way
 * merge by seq), with one throughput change: the applier accumulates a
 * seq-ordered BATCH of up to kBatchSize events and applies the whole batch
 * under a SINGLE state-lock acquisition (applyBatch), instead of locking once
 * per event.
 *
 * ---------------------------------------------------------------------------
 * PHASE 1b: recency-reorder isolation (env-gated, off by default)
 *
 * Phase 1a (hard-barrier windowing) was a dead end: in a single applier, reorder
 * and applier-staleness are coupled, so a barrier that enforces order induces
 * head-of-line staleness that destroyed the gap at t>1 even at W=1. Only
 * parallel appliers decouple them.
 *
 * Phase 1b isolates the ONE thing sharding actually approximates, the recency
 * buffer's write order, with zero barrier and zero staleness confound. Because
 * applyBatch holds the write lock for the whole batch and eviction reads take
 * the shared lock, no read observes the ring mid-batch, so we can:
 *   1. apply every event in seq order exactly as baseline (embeddings, context,
 *      counts byte-faithful), capturing each recency write's (obj_id, embedding
 *      snapshot) at its true seq position instead of writing it inline;
 *   2. flush the captured recency writes into the ring at batch end, in shard
 *      order (bucketed by hash(obj_id) % K).
 * Between batches the ring holds the same writes as baseline, only reordered, so
 * there is no new staleness. K=1 reproduces baseline byte-for-byte.
 *
 * Knobs:
 *   TARDIS_RECENCY_SHARDS=K (default 1 = OFF = inline baseline). K>1 simulates
 *     K parallel appliers sharing the recency ring.
 *   TARDIS_RECENCY_MODE=0 (default) = interleave (round-robin across buckets;
 *     realistic, reorder ~K). =1 = group (concatenate buckets; pessimistic,
 *     reorder ~batch).
 * Reorder is capped at the batch size (kBatchSize); a realistic skew bound.
 * ---------------------------------------------------------------------------
 *
 * ---------------------------------------------------------------------------
 * Thread-local redesign
 *
 * Every mybench worker thread owns a disjoint key space, so each shard is
 * logically written on behalf of exactly one origin thread.
 *
 * ---------------------------------------------------------------------------
 * Per-shard appliers
 *
 * One SPSC queue + one applier thread + one shared_mutex per shard, instead
 * of a single global queue-merge/lock/applier. A shard's applier is spawned
 * lazily on that shard's first on_access. Since no shard depends on another shard's
 * state, each applier operates fully independently.
 * ---------------------------------------------------------------------------
 */

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <list>
#include <memory>
#include <random>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <pthread.h>
#include <sched.h>

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
  };

  // ---- Phase 1b recency-intent record ----
  struct RecencyIntent {
    uint64_t obj_id;
    EmbeddingData emb;
    bool valid;
  };

  static constexpr int kMaxThreads = 64;
  static constexpr size_t kQueueCapacity = 1u << 20;
  static constexpr size_t kBatchSize = 256;

  // Per-origin-thread embedding state. Written only by that shard's own
  // applier thread, so each ShardState is single-writer with no cross-shard
  // dependency anywhere below.
  struct ShardState {
    double context[K][D];
    std::unordered_map<uint64_t, EmbeddingData> embeddings;
    std::unordered_map<uint64_t, int> access_count;
    std::vector<EmbeddingData> recent_embeddings;
    std::vector<uint64_t> recent_ids;
    std::vector<bool> recent_valid;
    int recent_idx = 0;
    int recent_count = 0;
    int perturb_counter = 0;
    std::mt19937 rng;
    uint64_t sample_counter = 0;   // for mode 2 deterministic every-Nth
    std::normal_distribution<double> normal_dist{0.0, 1.0};

    // ---- embedding-cache LRU (capped/MCACHE mode) ----
    std::list<uint64_t> lru_list;
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator> lru_map;

    std::vector<RecencyIntent> recency_intents;
    std::vector<std::vector<RecencyIntent>> recency_buckets;
  };

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
        max_entries_(max_entries) {
    for (int i = 0; i < kMaxThreads; i++) {
      auto& s = shards_[i];
      s.rng.seed(seed + static_cast<uint64_t>(i));
      init_context(s);
      s.recent_embeddings.resize(recent_window_);
      s.recent_ids.resize(recent_window_);
      s.recent_valid.resize(recent_window_, false);
    }

    // Phase 1b recency-reorder knobs (read BEFORE any applier thread spawns).
    if (const char* s = std::getenv("TARDIS_RECENCY_SHARDS")) {
      recency_shards_ = std::atoi(s);
      if (recency_shards_ < 1) recency_shards_ = 1;
    }
    if (const char* m = std::getenv("TARDIS_RECENCY_MODE")) {
      recency_mode_ = std::atoi(m);
    }
    // Base core for pinning per-shard appliers: shard i pins to core
    // applier_core_ + 2*i, so it doesn't float onto a core mybench has
    // pinned a worker to (workers only ever use even cores, see
    // mybench/benchMT.cpp) or migrate cross-NUMA away from its shard state.
    // -1 disables pinning entirely.
    if (const char* c = std::getenv("TARDIS_APPLIER_CORE")) {
      applier_core_ = std::atoi(c);
    }
        if (const char* r = std::getenv("TARDIS_SAMPLE_RATE")) {
      sample_rate_ = std::atof(r);
    }
    if (const char* m = std::getenv("TARDIS_SAMPLE_MODE")) {
      sample_mode_ = std::atoi(m);
    }

    for (int i = 0; i < kMaxThreads; i++) {
      buffers_[i] =
          std::make_unique<folly::ProducerConsumerQueue<AccessEvent>>(
              kQueueCapacity);
      applierStarted_[i].store(false, std::memory_order_relaxed);
    }
  }

  ~TARDISEmbeddingManager() {
    stop_.store(true, std::memory_order_release);
    for (int i = 0; i < kMaxThreads; i++) {
      if (appliers_[i].joinable()) {
        appliers_[i].join();
      }
    }
  }

  TARDISEmbeddingManager(const TARDISEmbeddingManager&) = delete;
  TARDISEmbeddingManager& operator=(const TARDISEmbeddingManager&) = delete;

  // ---- HOT PATH ----
  // `shard` is the object's owning shard, decoded by the caller from the key
  void on_access(uint64_t obj_id, int shard) {
    // Inline apply: each shard is single-writer in replicated mode, so the
    // per-shard applier thread is unnecessary. Applying directly here removes
    // one spinning applier thread per shard (the Phase 2 spin-thrash cost) and
    // preserves per-shard chronological order trivially. The per-shard lock is
    // kept because on one shared cache, an eviction on another thread can read
    // this shard during forgiveness (avg_top_k takes a shared_lock).
    // ORIGINAL
    // std::unique_lock<std::shared_mutex> lock(state_mutex_[shard]);
    // applyEventLocked(shards_[shard], AccessEvent{obj_id});

    // NEW
    std::unique_lock<std::shared_mutex> lock(state_mutex_[shard]);
    ShardState& s = shards_[shard];
    // Subsampling. mode 0 (whole): skip the entire update on non-sampled
    // accesses. mode 1 (update): always do bookkeeping, sample only the heavy
    // embedding math (gate is inside applyEventLocked).
    if (sample_mode_ == 0 && sample_rate_ < 1.0) {
      if (std::uniform_real_distribution<double>(0.0, 1.0)(s.rng) >= sample_rate_) {
        return;
      }
    }
    if (sample_mode_ == 2 && sample_rate_ < 1.0) {
      long N = std::llround(1.0 / sample_rate_);
      if (N < 1) N = 1;
      if ((++s.sample_counter % (uint64_t)N) != 0) {
        return;
      }
    }
    applyEventLocked(s, AccessEvent{obj_id});
  }

  // ---- READ PATH (eviction-time forgiveness) ----
  // `shard` is the candidate's owning shard
  double avg_top_k_similarity_to_recent(uint64_t obj_id, int shard, int top_k = 3) {
    std::shared_lock<std::shared_mutex> lock(state_mutex_[shard]);
    const auto& s = shards_[shard];
    auto it = s.embeddings.find(obj_id);
    if (it == s.embeddings.end()) return 0.0;

    // OPTIMIZATION 2: Bounded top-k in a single pass.
    // Removes std::vector allocation and std::sort. Caps top_k at 3.
    double top[3] = {-2.0, -2.0, -2.0}; // Cosine similarity is >= -1.0
    int valid_count = 0;
    const auto& cand = it->second;

    for (int i = 0; i < s.recent_count; i++) {
      if (s.recent_ids[i] == obj_id) continue;
      if (!s.recent_valid[i]) continue;

      const auto& recent_emb = s.recent_embeddings[i];
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

  int get_access_count(uint64_t obj_id, int shard) const {
    std::shared_lock<std::shared_mutex> lock(state_mutex_[shard]);
    const auto& s = shards_[shard];
    auto it = s.access_count.find(obj_id);
    return it != s.access_count.end() ? it->second : 0;
  }

  bool has_embedding(uint64_t obj_id, int shard) const {
    std::shared_lock<std::shared_mutex> lock(state_mutex_[shard]);
    return shards_[shard].embeddings.count(obj_id) > 0;
  }

  size_t get_num_embeddings() const {
    size_t total = 0;
    for (int i = 0; i < kMaxThreads; i++) {
      std::shared_lock<std::shared_mutex> lock(state_mutex_[i]);
      total += shards_[i].embeddings.size();
    }
    return total;
  }

 private:
  static inline uint64_t hashObj(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    return x;
  }

  // Lazily spawn shard i's dedicated applier thread on its first access, so
  // shards a run never uses (most of the 64 slots, for typical thread
  // counts) never spin up a thread. Safe under concurrent first-touch even
  // though in practice exactly one origin thread ever calls this for a
  // given shard.
  void ensureApplierStarted(int shard) {
    if (applierStarted_[shard].load(std::memory_order_acquire)) {
      return;
    }
    bool expected = false;
    if (applierStarted_[shard].compare_exchange_strong(
            expected, true, std::memory_order_acq_rel)) {
      appliers_[shard] =
          std::thread(&TARDISEmbeddingManager::applierLoopForShard, this,
                      shard);
      if (applier_core_ >= 0) {
        const int core = applier_core_ + 2 * shard;
        if (core < static_cast<int>(std::thread::hardware_concurrency())) {
          cpu_set_t cpuset;
          CPU_ZERO(&cpuset);
          CPU_SET(core, &cpuset);
          pthread_setaffinity_np(appliers_[shard].native_handle(),
                                  sizeof(cpu_set_t), &cpuset);
        }
      }
    }
  }

  // ---- BACKGROUND APPLIER (batched), one per shard ----
  // buffers_[shard] is a single-producer/single-consumer queue, so draining
  // it in order already preserves that origin thread's chronological order.
  // No shard depends on another shard's state, so this applier never needs
  // to coordinate with any other shard's applier or readers.
  void applierLoopForShard(int shard) {
    auto& buf = *buffers_[shard];
    ShardState& s = shards_[shard];
    std::shared_mutex& mtx = state_mutex_[shard];
    std::vector<AccessEvent> batch;
    batch.reserve(kBatchSize);

    auto drainOnce = [&]() {
      batch.clear();
      AccessEvent ev;
      while (batch.size() < kBatchSize && buf.read(ev)) {
        batch.push_back(ev);
      }
    };

    while (true) {
      drainOnce();

      if (!batch.empty()) {
        applyBatchForShard(s, mtx, batch);
        continue;
      }

      if (stop_.load(std::memory_order_acquire)) {
        drainOnce();
        if (batch.empty()) break;
        applyBatchForShard(s, mtx, batch);
        continue;
      }
      std::this_thread::yield();
    }
  }

  void applyBatchForShard(ShardState& s, std::shared_mutex& mtx,
                           const std::vector<AccessEvent>& batch) {
    std::unique_lock<std::shared_mutex> lock(mtx);
    if (recency_shards_ > 1) s.recency_intents.clear();
    for (const auto& ev : batch) {
      applyEventLocked(s, ev);
    }
    if (recency_shards_ > 1) {
      flushRecencyReordered(s);
    }
  }

  void applyEventLocked(ShardState& s, const AccessEvent& ev) {
    const uint64_t obj_id = ev.obj_id;

    // Variant 2 (mode 1): subsample only the heavy embedding math. Bookkeeping
    // (access_count) still runs every access; the context advance, embedding
    // update, and embedding init are skipped on non-sampled accesses.
    bool doEmbedding = true;
    if (sample_mode_ == 1 && sample_rate_ < 1.0) {
      doEmbedding =
          (std::uniform_real_distribution<double>(0.0, 1.0)(s.rng) < sample_rate_);
    }

    if (doEmbedding && ++s.perturb_counter >= 10) {
      s.perturb_counter = 0;
      update_context_batch(s);
    }

    int& count = s.access_count[obj_id];
    count++;

    if (!doEmbedding) {
      return;  // cheap bookkeeping only; skip embedding update/init and recency
    }

    auto it = s.embeddings.find(obj_id);
    bool has_emb = (it != s.embeddings.end());
    if (has_emb) {
      update_embedding(s, obj_id);
      if (max_entries_ >= 0) {
        move_to_front(s, obj_id);
      }
      record_recent(s, obj_id);
    } else if (count == min_access_count_) {
      if (max_entries_ >= 0) {
        while (static_cast<int64_t>(s.embeddings.size()) >= max_entries_ &&
               !s.lru_list.empty()) {
          evict_lru(s);
        }
        init_embedding(s, obj_id);
        add_to_lru(s, obj_id);
      } else {
        init_embedding(s, obj_id);
      }
      record_recent(s, obj_id);
    }
  }

  // void applyEventLocked(ShardState& s, const AccessEvent& ev) {
  //   const uint64_t obj_id = ev.obj_id;

  //   if (++s.perturb_counter >= 10) {
  //     s.perturb_counter = 0;
  //     update_context_batch(s);
  //   }

  //   int& count = s.access_count[obj_id];
  //   count++;

  //   auto it = s.embeddings.find(obj_id);
  //   bool has_emb = (it != s.embeddings.end());

  //   if (has_emb) {
  //     update_embedding(s, obj_id);
  //     if (max_entries_ >= 0) {
  //       move_to_front(s, obj_id);
  //     }
  //     record_recent(s, obj_id);
  //   } else if (count == min_access_count_) {
  //     if (max_entries_ >= 0) {
  //       while (static_cast<int64_t>(s.embeddings.size()) >= max_entries_ &&
  //              !s.lru_list.empty()) {
  //         evict_lru(s);
  //       }
  //       init_embedding(s, obj_id);
  //       add_to_lru(s, obj_id);
  //     } else {
  //       init_embedding(s, obj_id);
  //     }
  //     record_recent(s, obj_id);
  //   }
  // }

  // Recency sink: inline (baseline) when not sharding, else capture the write
  // (obj_id + embedding snapshot at THIS seq position) for a reordered flush.
  void record_recent(ShardState& s, uint64_t obj_id) {
    if (recency_shards_ <= 1) {
      update_recent(s, obj_id);   // baseline, byte-identical
      return;
    }
    RecencyIntent in;
    in.obj_id = obj_id;
    auto it = s.embeddings.find(obj_id);
    if (it != s.embeddings.end()) {
      in.emb = it->second;
      in.valid = true;
    } else {
      in.valid = false;
    }
    s.recency_intents.push_back(std::move(in));
  }

  void writeRecent(ShardState& s, const RecencyIntent& in) {
    s.recent_ids[s.recent_idx] = in.obj_id;
    if (in.valid) {
      s.recent_embeddings[s.recent_idx] = in.emb;
      s.recent_valid[s.recent_idx] = true;
    } else {
      s.recent_valid[s.recent_idx] = false;
    }
    s.recent_idx = (s.recent_idx + 1) % recent_window_;
    if (s.recent_count < recent_window_) s.recent_count++;
  }

  // Flush captured recency writes into the ring in shard order. Buckets by
  // hash(obj_id) % K, preserving in-shard (seq) order; interleave or group.
  void flushRecencyReordered(ShardState& s) {
    const int Ks = recency_shards_;
    if (s.recency_buckets.size() != static_cast<size_t>(Ks)) {
      s.recency_buckets.assign(Ks, {});
    } else {
      for (auto& b : s.recency_buckets) b.clear();
    }
    for (auto& in : s.recency_intents) {
      s.recency_buckets[hashObj(in.obj_id) % Ks].push_back(std::move(in));
    }

    if (recency_mode_ == 1) {
      // GROUP (pessimistic): concatenate shards, reorder ~batch.
      for (int b = 0; b < Ks; b++)
        for (auto& in : s.recency_buckets[b]) writeRecent(s, in);
    } else {
      // INTERLEAVE (realistic): round-robin across shards, reorder ~K.
      size_t maxLen = 0;
      for (int b = 0; b < Ks; b++)
        maxLen = std::max(maxLen, s.recency_buckets[b].size());
      for (size_t r = 0; r < maxLen; r++)
        for (int b = 0; b < Ks; b++)
          if (r < s.recency_buckets[b].size()) writeRecent(s, s.recency_buckets[b][r]);
    }
  }

  // ---- embedding-cache LRU (capped/MCACHE mode) ----
  void add_to_lru(ShardState& s, uint64_t obj_id) {
    s.lru_list.push_front(obj_id);
    s.lru_map[obj_id] = s.lru_list.begin();
  }
  void move_to_front(ShardState& s, uint64_t obj_id) {
    auto it = s.lru_map.find(obj_id);
    if (it != s.lru_map.end()) {
      s.lru_list.erase(it->second);
      s.lru_list.push_front(obj_id);
      it->second = s.lru_list.begin();
    }
  }
  void evict_lru(ShardState& s) {
    if (s.lru_list.empty()) return;
    uint64_t lru_id = s.lru_list.back();
    s.lru_list.pop_back();
    s.lru_map.erase(lru_id);
    s.embeddings.erase(lru_id);
  }

  void init_context(ShardState& s) {
    for (int k = 0; k < K; k++)
      for (int d = 0; d < D; d++) s.context[k][d] = s.normal_dist(s.rng);
  }

  void update_context_batch(ShardState& s) {
    double effective_speed = 10.0 * ctx_speed_;
    double decay = std::sqrt(1.0 - effective_speed);
    double noise_scale = std::sqrt(effective_speed);
    for (int k = 0; k < K; k++)
      for (int d = 0; d < D; d++)
        s.context[k][d] =
            decay * s.context[k][d] + noise_scale * s.normal_dist(s.rng);
  }

  void init_embedding(ShardState& s, uint64_t obj_id) {
    auto& emb = s.embeddings[obj_id];
    for (int k = 0; k < K; k++) {
      double sq_norm = 0.0;
      for (int d = 0; d < D; d++) {
        double val = s.normal_dist(s.rng);
        emb.vectors[k][d] = val;
        sq_norm += val * val;
      }
      // Cache the computed norm immediately
      emb.norms[k] = std::sqrt(sq_norm);
    }
  }

  void update_embedding(ShardState& s, uint64_t obj_id) {
    auto& emb = s.embeddings[obj_id];
    double decay = std::sqrt(1.0 - lr_);
    double ctx_scale = std::sqrt(lr_);

    for (int k = 0; k < K; k++) {
      double sq_norm = 0.0;
      for (int d = 0; d < D; d++) {
        emb.vectors[k][d] = decay * emb.vectors[k][d] + ctx_scale * s.context[k][d];
        sq_norm += emb.vectors[k][d] * emb.vectors[k][d];
      }
      // Cache the computed norm immediately
      emb.norms[k] = std::sqrt(sq_norm) + 1e-10;
    }
  }

  void update_recent(ShardState& s, uint64_t obj_id) {
    s.recent_ids[s.recent_idx] = obj_id;
    auto it = s.embeddings.find(obj_id);
    if (it != s.embeddings.end()) {
      s.recent_embeddings[s.recent_idx] = it->second;
      s.recent_valid[s.recent_idx] = true;
    } else {
      s.recent_valid[s.recent_idx] = false;
    }
    s.recent_idx = (s.recent_idx + 1) % recent_window_;
    if (s.recent_count < recent_window_) s.recent_count++;
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
  int recency_shards_ = 1;   // TARDIS_RECENCY_SHARDS: 1 = OFF (inline baseline)
  int recency_mode_ = 0;     // TARDIS_RECENCY_MODE: 0 = interleave, 1 = group
  int applier_core_ = -1;     // TARDIS_APPLIER_CORE: base core for per-shard
                             // applier pinning, shard i -> core + 2*i
                             // (-1 = off)
  double sample_rate_ = 1.0;      // TARDIS_SAMPLE_RATE: fraction of accesses that update embeddings
  int    sample_mode_ = 0;        // TARDIS_SAMPLE_MODE: 0 = skip whole event, 1 = skip only embedding math

  // ---- embedding state, sharded per origin thread. state_mutex_[i] guards
  // shards_[i] only; shard i's applier is state_mutex_[i]'s sole writer, so
  // no shard's lock is ever touched by another shard's applier or readers.
  // ----
  mutable std::array<std::shared_mutex, kMaxThreads> state_mutex_;
  std::array<ShardState, kMaxThreads> shards_;

  // ---- log infrastructure: one SPSC queue + one applier thread per shard,
  // the latter spawned lazily on that shard's first access. ----
  std::unique_ptr<folly::ProducerConsumerQueue<AccessEvent>>
      buffers_[kMaxThreads];
  std::array<std::thread, kMaxThreads> appliers_;
  std::array<std::atomic<bool>, kMaxThreads> applierStarted_;
  std::atomic<bool> stop_{false};
};

}  // namespace cachelib
}  // namespace facebook
