#include "utils.hpp"
#include <absl/container/node_hash_map.h>
#include <absl/container/node_hash_set.h>
#include <algorithm>
#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>
#include <cassert>
#include <csignal>
#include <cstdio>
#include <fmt/core.h>
#include <fmt/os.h>
#include <fmt/ranges.h>
#include <iostream>
#include <list>
#include <semaphore>
#include <thread>
#include <vector>
#include <zstd.h>

const bool ENABLE_FORCE_TTL = false;
const u64 FORCE_TTL = 3600;
const bool DEBUG_PRINT_FIRST_ENTRIES = true;
namespace po = boost::program_options;

class TraceReader
{
  public:
    TraceReader(bp::ipstream &stream) : zout(stream) {}
    bool get(Req &req)
    {
        if (zout.read(reinterpret_cast<char *>(&req), sizeof(Req)))
            return zout.gcount() == sizeof(Req);
        return false;
    }

  private:
    bp::ipstream &zout;
};

struct CacheEntry {
    u64 key;
    u64 size;
    u32 ttl;
    u32 ttstale;
    u64 entry_create_ts;
    u64 last_access_ts;
    u64 last_update_ts;
    u32 accesses_since_update;
    std::list<u64>::iterator lru_pos;
};

struct ExpiryHeapEntry {
    u64 expiry_ts;
    u64 key;

    bool operator>(const ExpiryHeapEntry &other) const
    {
        return expiry_ts > other.expiry_ts;
    }
};

class ExpiryHeap
{
  public:
    void push(const ExpiryHeapEntry &entry)
    {
        heap.push_back(entry);
        std::push_heap(heap.begin(), heap.end(), std::greater<>());
    }
    void pop()
    {
        std::pop_heap(heap.begin(), heap.end(), std::greater<>());
        heap.pop_back();
    }
    auto top() -> ExpiryHeapEntry & { return heap.front(); }
    auto empty() const { return heap.empty(); }

    vec<ExpiryHeapEntry> heap;

  private:
};

class LRUCache
{
  public:
    LRUCache(u64 capacity_bytes) : capacity(capacity_bytes), current_size(0) {}

    // Find an entry in the cache
    auto find(u64 key) -> absl::node_hash_map<u64, CacheEntry>::iterator
    {
        return cache.find(key);
    }

    auto end() -> absl::node_hash_map<u64, CacheEntry>::iterator
    {
        return cache.end();
    }

    // Get entry by key (assumes key exists)
    auto get(u64 key) -> CacheEntry & { return cache.at(key); }

    // Insert a new cache entry
    void insert(u64 key, const CacheEntry &entry)
    {
        lru_list.push_back(key);
        current_size += entry.size;

        CacheEntry new_entry = entry;
        new_entry.lru_pos = std::prev(lru_list.end());
        cache[key] = new_entry;
    }

    // Touch an entry (move to most recently used position)
    auto touch(std::list<u64>::iterator lru_pos,
               u64 key) -> std::list<u64>::iterator
    {
        lru_list.erase(lru_pos);
        lru_list.push_back(key);
        return std::prev(lru_list.end());
    }

    // Remove an entry from the cache (by iterator)
    void erase(absl::node_hash_map<u64, CacheEntry>::iterator it)
    {
        lru_list.erase(it->second.lru_pos);
        current_size -= it->second.size;
        cache.erase(it);
    }

    // Remove an entry from the cache (by key)
    void erase(u64 key)
    {
        auto it = cache.find(key);
        if (it != cache.end())
            erase(it);
    }

    // Get the least recently used key (for eviction)
    auto get_lru_key() const -> u64 { return lru_list.front(); }

    // Evict the LRU entry and return the evicted key
    auto evict_lru() -> u64
    {
        auto lru_key = lru_list.front();
        lru_list.pop_front();

        auto it = cache.find(lru_key);
        if (it != cache.end()) {
            current_size -= it->second.size;
            cache.erase(it);
        }

        return lru_key;
    }

    // Check if cache is over capacity
    bool is_over_capacity() const { return current_size > capacity; }
    auto get_current_size() const -> u64 { return current_size; }

  private:
    absl::node_hash_map<u64, CacheEntry> cache;
    std::list<u64> lru_list;
    u64 capacity;
    u64 current_size;
};

auto run_sim(SimConfig cfg, TraceReader &reader) -> SimStats
{
    Req req;
    SimStats stats;

    absl::node_hash_set<u64> seen_keys;

    // cache should be adjusted for key sampling
    LRUCache lru_cache(cfg.capacity_mb * 1024 * 1024 / cfg.key_sample_ratio);
    ExpiryHeap expiry_heap;

    while (reader.get(req)) {
        // skip too large objects
        if (req.size > 5ull * 1024 * 1024 * 1024) // 5 gib
            continue;

        // cap ttls
        if (req.ttl > UINT32_MAX)
            req.ttl = UINT32_MAX;
        if (req.ttstale > UINT32_MAX)
            req.ttstale = UINT32_MAX;

        if (req.key % cfg.key_sample_ratio != 0) // should be uniformly dist
            continue;
        stats.all++;
        auto ts = req.ts;

        if (ENABLE_FORCE_TTL)
            req.ttl = FORCE_TTL;

        // handle expired entries
        if (cfg.rv_enable) {
            // get all expired entries up to current ts
            while (!expiry_heap.empty() && expiry_heap.top().expiry_ts <= ts) {
                auto [expiry_ts, key] = expiry_heap.top();
                expiry_heap.pop();

                auto entryf = lru_cache.find(key);
                if (entryf == lru_cache.end())
                    continue; // already evicted

                auto &[k, entry] = *entryf;
                assert(k == key);
                assert(entry.ttl + entry.last_update_ts == expiry_ts);

                // check if we should revalidate
                auto should_revalidate =
                    entry.ttl >= cfg.rv_min_ttl &&
                    entry.accesses_since_update >= cfg.rv_min_freq;

                // revalidate: reset ttl and accesses_since_update
                if (should_revalidate) {
                    entry.last_update_ts = ts;
                    entry.accesses_since_update = 0;
                    expiry_heap.push(
                        ExpiryHeapEntry{ts + entry.ttl, entry.key});
                    continue;
                }

                if (!cfg.evict_expired)
                    continue;

                // evict if not revalidated
                seen_keys.insert(key);
                lru_cache.erase(key);
            }
        }

        // handle cache logic
        auto entryf = lru_cache.find(req.key);

        // miss
        if (entryf == lru_cache.end()) {
            // check miss type, is it mandatory or not
            if (seen_keys.find(req.key) == seen_keys.end())
                stats.miss_mandatory++;
            else
                stats.miss_evicted++;

            // insert into cache
            CacheEntry new_entry{
                .key = req.key,
                .size = req.size,
                .ttl = (u32)req.ttl,
                .ttstale = (u32)req.ttstale,
                .entry_create_ts = ts,
                .last_access_ts = ts,
                .last_update_ts = ts,
                .accesses_since_update = 1,
            };
            lru_cache.insert(req.key, new_entry);
            expiry_heap.push(ExpiryHeapEntry{ts + req.ttl, req.key});

            // evict until under capacity
            while (lru_cache.is_over_capacity()) {
                auto evicted_key = lru_cache.evict_lru();
                seen_keys.insert(evicted_key);
            }

            continue;
        }

        // hit
        auto &[k, entry] = *entryf;
        assert(k == req.key);

        // determine if hit is fresh or stale
        auto cache_age = ts - entry.last_update_ts;

        // 3 types of hits:
        // fresh: age <= ttl
        // reval: fresh, but # of accesses since last update = 0
        // stale: ttl < age <= ttl + ttstale
        if (cache_age <= entry.ttl && entry.accesses_since_update > 0)
            stats.hit_fresh++;
        else if (cache_age <= entry.ttl && entry.accesses_since_update == 0)
            stats.hit_reval++;
        else if (cache_age <= entry.ttl + entry.ttstale)
            stats.hit_stale++;
        else
            stats.miss_expired++;

        // update entry metadata
        entry.last_access_ts = ts;
        entry.accesses_since_update++;

        // move to head of LRU list (most recently used)
        entry.lru_pos = lru_cache.touch(entry.lru_pos, req.key);
    }

    return stats;
}

auto main(int argc, char **argv) -> int
{
    vec<SimConfig> configs;

    // clang-format off
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "produce help message")
    ("csvout,o", po::value<str>()->default_value("results.csv"), "all revalidation results output file")
    ("capacity,c", po::value<u64>()->default_value(2048), "cache capacity in MB")
    ("parallel,p", po::value<u32>()->default_value(1), "number of parallel simulations to run")

    ("input-file,i", po::value<vec<str>>(), "input trace file (zstd compressed)")
    ("key-sample-ratio,s", po::value<vec<u64>>(), "key sampling ratio (list)")
    ("rv-enable,r", po::value<bool>()->default_value(true), "enable revalidation (1/0)")
    ("rv-min-ttl,t", po::value<vec<u64>>(), "min ttl to revalidate (list)")
    ("rv-min-freq,f", po::value<vec<u64>>(), "min accesses to revalidate (list)")
    ("rv-max-za,z", po::value<vec<f64>>(), "max zone amplification (list)")
    ;
    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        desc.print(std::cout);
        return 0;
    }

    auto capacity = vm["capacity"].as<u64>();
    auto rv_enable = vm["rv-enable"].as<bool>();
    auto csvout_file = vm["csvout"].as<str>();
    auto parallel = vm["parallel"].as<u32>();

    auto input_files = vec<str>{"traces/sim/cf_a.bin.zst"};
    if (vm.count("input-file"))
        input_files = vm["input-file"].as<vec<str>>();
    auto key_sample_ratio = vec<u64>{4};
    if (vm.count("key-sample-ratio"))
        key_sample_ratio = vm["key-sample-ratio"].as<vec<u64>>();
    auto rv_min_ttl = vec<u64>{15};
    if (vm.count("rv-min-ttl"))
        rv_min_ttl = vm["rv-min-ttl"].as<vec<u64>>();
    auto rv_min_freq = vec<u64>{3};
    if (vm.count("rv-min-freq"))
        rv_min_freq = vm["rv-min-freq"].as<vec<u64>>();
    auto rv_max_za = vec<f64>{2};
    if (vm.count("rv-max-za"))
        rv_max_za = vm["rv-max-za"].as<vec<f64>>();

    for (auto infile : input_files)
        for (auto ksr : key_sample_ratio)
            for (auto rvt : rv_min_ttl)
                for (auto rvf : rv_min_freq)
                    for (auto rvza : rv_max_za)
                        configs.push_back(SimConfig{
                            .infile = infile,
                            .capacity_mb = capacity,
                            .key_sample_ratio = ksr,
                            .evict_expired = true,
                            .rv_enable = rv_enable,
                            .rv_min_ttl = rvt,
                            .rv_min_freq = rvf,
                            .rv_max_zone_amp = rvza,
                        });

    fmt::print("CSV output file: {}\n", csvout_file);
    auto outf = fmt::output_file(csvout_file);
    outf.print("{},{}\n", SimConfig::csv_hdr(), SimStats::csv_hdr());

    std::counting_semaphore sem{parallel};
    vec<std::jthread> threads;

    for (auto i = 0; i < configs.size(); i++) {
        auto &cfg = configs[i];
        threads.push_back(std::jthread([&sem, &cfg, &outf, i]() {
            sem.acquire();

            fmt::print("Running config #{}: {}\n", i + 1, cfg.repr());
            bp::ipstream zout;
            bp::child zstdcat("zstdcat",
                              bp::std_in<cfg.infile, bp::std_out> zout);
            TraceReader reader(zout);

            auto tstart = tnow();
            auto stats = run_sim(cfg, reader);
            auto total_time = tsince(tstart);

            auto mrps = (double)stats.all / (double)total_time / 1'000'000.0;
            fmt::print("Results #{} (took {}s, {:.2f} Mreq/s)\n{}:\n{}", i + 1,
                       total_time, mrps, cfg.repr(), stats.human_str());
            outf.print("{},{}\n", cfg.csv(), stats.csv());

            zstdcat.terminate();
            zstdcat.wait();
            sem.release();
        }));
    }

    for (auto &t : threads)
        t.join();

    return 0;
}