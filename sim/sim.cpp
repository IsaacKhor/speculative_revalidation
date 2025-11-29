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
#include <thread>
#include <vector>
#include <zstd.h>

const bool ENABLE_FORCE_TTL = false;
const u64 FORCE_TTL = 86400 * 2;
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
    u32 entry_create_ts;
    u32 last_access_ts;
    u32 last_update_ts;
    u32 accesses_since_update;
    u32 next_access_ts;
    std::list<u64>::iterator lru_pos;
};

// using cachemap = absl::node_hash_map<u64, CacheEntry>;
// using keyset = absl::node_hash_set<u64>;
using cachemap = std::unordered_map<u64, CacheEntry>;
using keyset = std::unordered_set<u64>;

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
    LRUCache(u64 capacity_bytes) : capacity(capacity_bytes) {}

    // Find an entry in the cache
    auto find(u64 key) -> cachemap::iterator { return cache.find(key); }

    auto end() -> cachemap::iterator { return cache.end(); }

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
    void erase(cachemap::iterator it)
    {
        lru_list.erase(it->second.lru_pos);
        current_size -= it->second.size;
        cache.erase(it);
    }

    // Remove an entry from the cache (by key)
    void evict_key(u64 key)
    {
        auto it = cache.find(key);
        if (it != cache.end())
            erase(it);
    }

    // Evict the LRU entry and return the evicted entry
    auto evict_lru() -> CacheEntry
    {
        auto lru_key = lru_list.front();
        lru_list.pop_front();

        auto it = cache.find(lru_key);
        auto entry = it->second;
        if (it != cache.end()) {
            current_size -= it->second.size;
            cache.erase(it);
        }

        return entry;
    }

    // Check if cache is over capacity
    bool is_over_capacity() const { return current_size > capacity; }
    auto get_current_size() const -> u64 { return current_size; }
    auto get_lru_key() const -> u64 { return lru_list.front(); }

  private:
    cachemap cache;
    std::list<u64> lru_list;
    u64 capacity = 0;
    u64 current_size = 0;
};

class CacheSimulator
{
  private:
    SimConfig cfg;
    LRUCache lru_cache;
    ExpiryHeap expiry_heap;
    keyset seen_keys;
    SimStats stats;

  public:
    CacheSimulator(SimConfig cfg)
        : cfg(cfg), lru_cache(cfg.capacity_gib * 1024 * 1024 * 1024 /
                              cfg.key_sample_ratio)
    {
    }

    auto evict(u64 key)
    {
        seen_keys.insert(key);
        lru_cache.evict_key(key);
    }

    auto reval(CacheEntry &entry, u64 now_ts)
    {
        entry.last_update_ts = now_ts;
        entry.accesses_since_update = 0;
        expiry_heap.push(ExpiryHeapEntry{now_ts + entry.ttl, entry.key});
        stats.revals++;
    }

    auto run_sim(TraceReader &reader, bool print_progress) -> SimStats
    {
        Req req;

        while (reader.get(req)) {
            // skip too large objects
            if (req.size > 5ull * 1024 * 1024 * 1024) // 5 gib
                continue;

            if (req.key % cfg.key_sample_ratio != 0) // should be uniformly dist
                continue;

            sim_request(req);

            if (print_progress && stats.all % 1'000'000 == 0)
                fmt::print(stderr, ".");
            if (print_progress && stats.all % 50'000'000 == 0)
                fmt::print(stderr, " {}m requests\n", stats.all / 1'000'000);
        }

        return stats;
    }

    auto on_expire(CacheEntry &entry, u32 now_ts)
    {
        auto key = entry.key;

        // check if it was a wasted revalidation
        if (entry.accesses_since_update == 0)
            stats.reval_wasted++;

        if (cfg.rv_mode == RevalidateMode::NEVER) {
            evict(key);
            return;
        }

        if (cfg.rv_mode == RevalidateMode::ALWAYS) {
            reval(entry, now_ts);
            return;
        }

        if (cfg.rv_mode == RevalidateMode::ORACLE) {
            // oracle means reval if future access < ttl and cache cycle
            // TODO implement cache cycle time check
            if (entry.next_access_ts <= now_ts + entry.ttl)
                reval(entry, now_ts);
            else
                evict(key);
            return;
        }

        if (cfg.rv_mode == RevalidateMode::HEURISTICS) {
            // check if we should revalidate
            auto should_revalidate =
                entry.ttl >= cfg.rv_min_ttl &&
                entry.accesses_since_update >= cfg.rv_min_freq;

            if (should_revalidate)
                reval(entry, now_ts);
            else
                evict(key);

            return;
        }

        if (cfg.rv_mode == RevalidateMode::ML) {
            // TODO implement ML-based revalidation
            throw std::runtime_error("ML revalidation not implemented");
        }

        throw std::runtime_error("unknown RevalidateMode");
    }

    auto sim_request(Req req) -> void
    {
        stats.all++;
        auto ts = req.ts;

        if (ENABLE_FORCE_TTL)
            req.ttl = FORCE_TTL;

        // handle expired entries
        // TODO handle ttstale
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

            on_expire(entry, ts);
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
                .ttl = req.ttl,
                .ttstale = req.ttstale,
                .entry_create_ts = ts,
                .last_access_ts = ts,
                .last_update_ts = ts,
                .accesses_since_update = 1,
                .next_access_ts = req.next_req_ts,
            };
            lru_cache.insert(req.key, new_entry);
            stats.fetches++;
            expiry_heap.push(ExpiryHeapEntry{ts + req.ttl, req.key});

            // evict until under capacity
            while (lru_cache.is_over_capacity()) {
                auto evicted_entry = lru_cache.evict_lru();
                seen_keys.insert(evicted_entry.key);
                if (evicted_entry.accesses_since_update == 0)
                    stats.reval_wasted++;
            }

            return;
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
        entry.next_access_ts = req.next_req_ts;

        // move to head of LRU list (most recently used)
        entry.lru_pos = lru_cache.touch(entry.lru_pos, req.key);
    }
};

auto main(int argc, char **argv) -> int
{
    vec<SimConfig> configs;

    // clang-format off
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "produce help message")
    ("csvout,o", po::value<str>()->default_value("results.csv"), "all revalidation results output file")
    ("capacity,c", po::value<u64>()->default_value(2048), "cache capacity in GiB")
    ("parallel,p", po::value<u32>()->default_value(1), "number of parallel simulations to run")

    ("input-file,i", po::value<vec<str>>(), "input trace file (zstd compressed)")
    ("key-sample-ratio,s", po::value<vec<u64>>(), "key sampling ratio (list)")
    ("rv-mode,m", po::value<vec<str>>(), "revalidation mode (never, always, oracle, heuristics, ml)")
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
    auto csvout_file = vm["csvout"].as<str>();
    auto parallel = vm["parallel"].as<u32>();

    if (vm.count("rv-mode") == 0)
        throw std::runtime_error("must specify at least one rv-mode");
    auto mode = vm["rv-mode"].as<vec<str>>();

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
        if (!std::filesystem::exists(infile))
            throw std::runtime_error("input file does not exist: " + infile);

    for (auto infile : input_files)
        for (auto ksr : key_sample_ratio)
            for (auto mode : mode) {
                auto rv_mode = rv_mode_from_str(mode);
                if (rv_mode != RevalidateMode::HEURISTICS)
                    configs.push_back(SimConfig{
                        .infile = infile,
                        .capacity_gib = capacity,
                        .key_sample_ratio = ksr,
                        .evict_expired = true,
                        .rv_mode = rv_mode,
                        .rv_min_ttl = 0,
                        .rv_min_freq = 0,
                        .rv_max_zone_amp = 0.0,
                    });
                else
                    for (auto rvt : rv_min_ttl)
                        for (auto rvf : rv_min_freq)
                            for (auto rvza : rv_max_za)
                                configs.push_back(SimConfig{
                                    .infile = infile,
                                    .capacity_gib = capacity,
                                    .key_sample_ratio = ksr,
                                    .evict_expired = true,
                                    .rv_mode = rv_mode,
                                    .rv_min_ttl = rvt,
                                    .rv_min_freq = rvf,
                                    .rv_max_zone_amp = rvza,
                                });
            }

    fmt::print("CSV output file: {}\n", csvout_file);
    auto outf = fmt::output_file(csvout_file);
    outf.print("{},{}\n", SimConfig::csv_hdr(), SimStats::csv_hdr());

    fmt::print("Running {} simulations with parallelism {}\n", configs.size(),
               parallel);

    vec<std::jthread> threads;
    std::atomic<u32> cfg_idx{0};

    for (auto i = 0; i < parallel; i++)
        threads.push_back(std::jthread([&cfg_idx, &configs, &outf]() {
            int j;
            while ((j = cfg_idx.fetch_add(1)) < configs.size()) {
                auto &cfg = configs[j];
                fmt::print("Running config #{}: {}\n", j + 1, cfg.repr());
                bp::ipstream zout;
                bp::child zstdcat("zstdcat",
                                  bp::std_in<cfg.infile, bp::std_out> zout);
                TraceReader reader(zout);
                CacheSimulator sim(cfg);

                auto tstart = tnow();
                auto stats = sim.run_sim(reader, configs.size() == 1);
                auto total_time = tsince(tstart);

                auto mrps =
                    (double)stats.all / (double)total_time / 1'000'000.0;
                fmt::print("Results #{} (took {}s, {:.2f} Mreq/s)\n{}:\n{}",
                           j + 1, total_time, mrps, cfg.repr(),
                           stats.human_str());
                outf.print("{},{}\n", cfg.csv(), stats.csv());

                zstdcat.terminate();
                zstdcat.wait();
            }
        }));

    for (auto &t : threads)
        t.join();

    return 0;
}