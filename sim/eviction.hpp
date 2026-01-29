#pragma once
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/container/node_hash_map.h>

#include "utils.hpp"

struct ZoneStats {
    u64 hits = 0;      // hits
    u64 misses = 0;    // misses (which in turn cause fetches)
    u64 rv_fetch = 0;  // revalidations only
    u64 rv_wasted = 0; // revalidations that were confirmed wasted
    u64 rv_good = 0;   // revalidations that were confirmed useful
};

struct CacheEntry {
    u64 key;
    u64 size;
    u32 ttl;
    u32 ttstale;
    u32 entry_create_ts;
    u32 accesses_since_update;
    u32 last_access_ts;
    u32 last_update_ts;
    u32 next_access_ts;
    u32 content_type;
    ZoneStats *zs;
    std::list<u64>::iterator list_pos;
    u32 frequency = 0; // for gdsf / sieve
    f64 priority = 0;  // for gdsf
};

// using cachemap_t = absl::node_hash_map<u64, CacheEntry>;
using cachemap_t = std::map<u64, CacheEntry>;
using exit_cb = std::function<void(CacheEntry &)>;

class Cache
{
  public:
    Cache(u64 capacity, exit_cb cb) : max_bytes(capacity), on_exit_cb(cb) {}
    virtual ~Cache() = default;

    auto find(u64 key) -> CacheEntry *
    {
        if (auto it = cachemap.find(key); it != cachemap.end())
            return &it->second;
        else
            return nullptr;
    }
    auto occupied_bytes() const { return cur_bytes; }

    virtual auto touch(u64 key) -> void = 0;
    virtual auto admit(u64 key, const CacheEntry entry) -> void = 0;
    virtual auto remove(u64 key) -> void = 0;

  protected:
    auto erase(cachemap_t::iterator it) -> void
    {
        on_exit_cb(it->second); // call this first before entry is invalidated
        cur_bytes -= it->second.size;
        cachemap.erase(it);
    }

    cachemap_t cachemap;
    u64 max_bytes;
    u64 cur_bytes = 0;
    exit_cb on_exit_cb;
};

template <typename T> using uptr = std::unique_ptr<T>;

auto inline validate_cache_type(str type) -> bool
{
    static const absl::flat_hash_set<str> valid_types = {"lru", "gdsf", "sieve",
                                                         "fifo", "arc"};
    return valid_types.contains(type);
}

auto make_cache(str type, u64 max_bytes, exit_cb cb) -> uptr<Cache>;
