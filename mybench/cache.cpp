

// #include <fmt/core.h>
// #include <folly/Conv.h>
// #include <folly/FBString.h>

// using namespace folly;

#include "cachelib/allocator/LruTailAgeStrategy.h"
#include "cachelib/allocator/MarginalHitsStrategy.h"
#include "cachelib/allocator/HitsPerSlabStrategy.h"
#include "cachelib/allocator/RebalanceStrategy.h"
#include "cachelib/common/Time.h"

#include <chrono>

#include "request.h"
#include "cache.h"


static void print_config(Cache::Config &config) {
  printf(
      "poolRebalancingEnabled %d, poolOptimizerEnabled %d, "
      "itemsReaperEnabled %d\n",
      config.poolRebalancingEnabled(), config.poolOptimizerEnabled(),
      config.itemsReaperEnabled());

  auto configmap = config.serialize();
  for (auto &kv : configmap) {
    printf("%s %s\n", kv.first.c_str(), kv.second.c_str());
  }
}

void mycache_init(int64_t cache_size_in_mb, unsigned int hashpower,
                       Cache **cache_p, PoolId *pool_p) {
  Cache::Config config;

  // auto rebalance_strategy = std::make_shared<HitsPerSlabStrategy>();
  // only works for LRU
  // auto rebalance_strategy = std::make_shared<LruTailAgeStrategy>();
  // only works for 2Q
  // auto rebalance_strategy = std::make_shared<MarginalHitsStrategy>();

  config.setCacheSize(cache_size_in_mb * 1024 * 1024)
      .setCacheName("My cache")
#ifdef USE_STRICTLRU
      .setAccessConfig({hashpower, 1})
#else
      .setAccessConfig({hashpower, hashpower})
#endif
      .validate();

  // print_config(config);
  *cache_p = new Cache(config);

#if defined(USE_STRICTLRU)
  Cache::MMConfig mm_config;
  mm_config.lruRefreshTime = 0;
  *pool_p = (*cache_p)->addPool("default",
                                (*cache_p)->getCacheMemoryStats().ramCacheSize,
                                {}, mm_config);
#elif defined(USE_LRUFORGIVE)
  Cache::MMConfig mm_config;
  // MCACHE mode: bound the embedding store at 5x estimated cache item count.
  // Average object size 34KB is the TARDIS paper Table 5 number for the
  // Wikipedia text CDN workload. Adjust constant if running other workloads.
  constexpr int64_t kAvgObjSize = 34000;
  int64_t est_cache_items =
      (cache_size_in_mb * 1024LL * 1024LL) / kAvgObjSize;
  mm_config.maxEmbeddingEntries = 5 * est_cache_items;
  fprintf(stderr,
          "[LRUFORGIVE] cache_mb=%ld, est_cache_items=%ld, "
          "maxEmbeddingEntries=%ld (MCACHE 5x)\n",
          (long)cache_size_in_mb,
          (long)est_cache_items,
          (long)mm_config.maxEmbeddingEntries);
  *pool_p = (*cache_p)->addPool("default",
                                (*cache_p)->getCacheMemoryStats().ramCacheSize,
                                {}, mm_config);
#elif defined(USE_S3FIFOFORGIVE)
  Cache::MMConfig mm_config;
  constexpr int64_t kAvgObjSize = 34000;
  int64_t est_cache_items =
      (cache_size_in_mb * 1024LL * 1024LL) / kAvgObjSize;
  mm_config.maxEmbeddingEntries = 5 * est_cache_items;
  fprintf(stderr,
          "[S3FIFOFORGIVE] cache_mb=%ld, est_cache_items=%ld, "
          "maxEmbeddingEntries=%ld (MCACHE 5x)\n",
          (long)cache_size_in_mb, (long)est_cache_items,
          (long)mm_config.maxEmbeddingEntries);
  *pool_p = (*cache_p)->addPool("default",
                                (*cache_p)->getCacheMemoryStats().ramCacheSize,
                                {}, mm_config);
#else
  *pool_p = (*cache_p)->addPool("default",
                                (*cache_p)->getCacheMemoryStats().ramCacheSize);
#endif

  util::setCurrentTimeSec(1);
  assert(util::getCurrentTimeSec() == 1);
}

static inline std::string gen_key(struct request *req) {
  auto key = fmt::format_int(*(uint64_t *)(req->key)).str();

  return key;
}

int cache_get(Cache *cache, PoolId pool, struct request *req) {
  static __thread char buf[1024 * 1024];

  auto key = gen_key(req);

  Cache::ReadHandle item_handle = cache->find(key);
  if (item_handle) {
    if (item_handle->isExpired()) {
      return 1;
    } else {
      assert(item_handle->getSize() == req->val_len);
      assert(!item_handle->isExpired());
      const char *data =
          reinterpret_cast<const char *>(item_handle->getMemory());

      return 0;
    }
  } else {
    return 1;
  }
}

int cache_set(Cache *cache, PoolId pool, struct request *req) {

  auto key = gen_key(req);
  req->val_len += req->key_len - key.size();
  if (req->val_len < 0) req->val_len = 0;

  Cache::WriteHandle item_handle =
      cache->allocate(pool, key, req->val_len, req->ttl, req->timestamp);
  if (item_handle == nullptr || item_handle->getMemory() == nullptr) {
    return 1;
  }

  std::memcpy(item_handle->getMemory(), req->val, req->val_len);
  cache->insertOrReplace(item_handle);

  return 0;
}

int cache_del(Cache *cache, PoolId pool, struct request *req) {
  auto key = gen_key(req);
  auto rm = cache->remove(key);
  if (rm == Cache::RemoveRes::kSuccess)
    return 0;
  else
    return 1;
}

double cache_utilization(Cache *cache) {
  int64_t used_size = 0;
  for (const auto &itr : *cache) {
    auto key = itr.getKey();
    used_size += key.size() + itr.getSize();
  }

  return (double)used_size / (double)cache->getCacheMemoryStats().ramCacheSize;
}
