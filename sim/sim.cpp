#include "libonnxruntime/onnxruntime_cxx_api.h"
#include "model.h"
#include "utils.hpp"
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/container/node_hash_map.h>
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

struct ZoneStats {
    u64 fetches = 0;   // all origin fetches, revals and misses
    u64 revals = 0;    // revalidations only
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
    std::list<u64>::iterator lru_pos;
};

using cachemap_t = absl::node_hash_map<u64, CacheEntry>;

struct ExpiryHeapEntry {
    u64 key;
    u64 expiry_ts;

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

class RevalPredictor
{
  private:
    static constexpr const u32 IN_FEATURES = 7;
    static constexpr const u32 OUT_FEATURES = 2;
    static constexpr const char *inames[1] = {"in"};
    static constexpr const char *onames[1] = {"probabilities"};

    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "predictor");
    Ort::Session session;
    Ort::MemoryInfo meminfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<i64, 2> ishape{1, IN_FEATURES};
    std::array<i64, 2> oshape{1, OUT_FEATURES};
    Ort::Value itensor;
    Ort::Value otensor;
    std::array<f32, IN_FEATURES> idata;
    std::array<f32, OUT_FEATURES> oprob;

  public:
    RevalPredictor(const str model_path)
        : session(env, model_path.c_str(), Ort::SessionOptions{})
    {
        itensor = Ort::Value::CreateTensor<f32>(
            meminfo, idata.data(), idata.size(), ishape.data(), ishape.size());
        otensor = Ort::Value::CreateTensor<f32>(
            meminfo, oprob.data(), oprob.size(), oshape.data(), oshape.size());
    }

    auto predict(u32 ttl, u32 freq, u32 generations, u32 t_since_last, u32 mime,
                 u32 size) -> f32
    {
        static const auto runopts = Ort::RunOptions{};
        idata[0] = static_cast<f32>(ttl);
        idata[1] = static_cast<f32>(freq);
        idata[2] = static_cast<f32>(generations);
        idata[3] = static_cast<f32>(t_since_last);
        idata[4] = static_cast<f32>(t_since_last) / static_cast<f32>(ttl);
        idata[5] = static_cast<f32>(mime);
        idata[6] = static_cast<f32>(size);
        session.Run(runopts, inames, &itensor, 1, onames, &otensor, 1);
        return oprob[1];
    }

    auto predict(CacheEntry &e, u32 now_ts) -> f32
    {
        auto ttl = e.ttl;
        auto freq = e.accesses_since_update;
        auto generations = (now_ts - e.entry_create_ts - 1) / ttl;
        auto t_since_last = now_ts - e.last_access_ts;
        auto mime = e.content_type;
        auto size = e.size;
        return predict(ttl, freq, generations, t_since_last, mime, size);
    }
};

class LRUCache
{
  public:
    LRUCache(u64 capacity_bytes) : capacity(capacity_bytes) {}

    // Check if cache is over capacity
    auto is_over_capacity() const { return current_size > capacity; }
    auto get_current_size() const { return current_size; }
    auto &tail() { return find(lru_list.front())->second; }
    auto end() -> cachemap_t::iterator { return cache.end(); }
    auto find(u64 key) -> cachemap_t::iterator { return cache.find(key); }
    auto at(u64 key) -> CacheEntry & { return cache.at(key); }

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
    auto erase(cachemap_t::iterator it)
    {
        lru_list.erase(it->second.lru_pos);
        current_size -= it->second.size;
        cache.erase(it);
    }

  private:
    cachemap_t cache;
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
    absl::flat_hash_set<u64> seen_keys;
    absl::node_hash_map<u64, ZoneStats> zone_stats;
    SimStats stats;
    FILE *trace_outf;
    std::optional<RevalPredictor> ml_predictor = std::nullopt;

  public:
    CacheSimulator(SimConfig cfg)
        : cfg(cfg), lru_cache(cfg.capacity_gib * 1024 * 1024 * 1024 /
                              cfg.key_sample_ratio),
          trace_outf(cfg.trace_outf)
    {
        if (cfg.rv_mode == RevalidateMode::ML)
            ml_predictor.emplace(cfg.model_path);
    }

    auto evict(CacheEntry &entry)
    {
        // check if it was a wasted revalidation
        if (entry.entry_create_ts < entry.last_update_ts &&
            entry.accesses_since_update == 0) {
            stats.rv_wasted++;
            entry.zs->rv_wasted++;
        }

        auto key = entry.key;
        seen_keys.insert(key);
        lru_cache.erase(lru_cache.find(key));
    }

    auto reval(CacheEntry &entry, u32 now_ts)
    {
        // check if it was a wasted revalidation
        if (entry.entry_create_ts < entry.last_update_ts &&
            entry.accesses_since_update == 0) {
            stats.rv_wasted++;
            entry.zs->rv_wasted++;
        }

        stats.revals++;
        entry.zs->revals++;
        entry.last_update_ts = now_ts;
        entry.accesses_since_update = 0;

        if (entry.zs->revals > 2)
            breakpoint();

        // don't bother if ttl is too big (trace lasts roughly 1 month)
        if (entry.ttl > 60 * 60 * 24 * 30)
            return;
        expiry_heap.push(ExpiryHeapEntry{entry.key, now_ts + entry.ttl});
    }

    auto run_sim(TraceReader &reader, bool print_progress) -> SimStats
    {
        Req req;

        while (reader.get(req)) {
            // skip too large objects
            if (req.size > 5ull * 1024 * 1024 * 1024) // 5 gib
                continue;

            // filter out ttl=0
            if (req.ttl == 0)
                continue;

            if (req.key % cfg.key_sample_ratio != 0) // should be uniformly dist
                continue;

            sim_request(req);

            if (print_progress && stats.all % 100'000 == 0)
                fmt::print(stderr, ".");
            if (print_progress && stats.all % 5'000'000 == 0)
                fmt::print(stderr, " {}m requests\n", stats.all / 1'000'000);
        }

        return stats;
    }

    auto record_expiry(CacheEntry &entry, u32 now_ts)
    {
        if (trace_outf == nullptr)
            return;

        auto zs = *entry.zs;
        fmt::print(trace_outf, "{},{},{},{},{},{},{},{},{},{},{},{}\n", now_ts,
                   entry.next_access_ts, entry.ttl, entry.entry_create_ts,
                   entry.accesses_since_update, entry.last_access_ts,
                   entry.last_update_ts, zs.revals, zs.rv_good, zs.rv_wasted,
                   entry.content_type, entry.size);
    }

    auto on_expire(CacheEntry &entry, u32 now_ts)
    {
        if (cfg.rv_mode == RevalidateMode::NEVER) {
            evict(entry);
            return;
        }

        if (cfg.rv_mode == RevalidateMode::ALWAYS) {
            reval(entry, now_ts);
            return;
        }

        if (cfg.rv_mode == RevalidateMode::ORACLE) {
            // oracle means reval if future access < ttl and cache cycle

            // TODO implement cache cycle time check properly
            if (entry.ttl < 7 * 86400 && entry.next_access_ts > now_ts &&
                entry.next_access_ts <= now_ts + entry.ttl)
                reval(entry, now_ts);
            else
                evict(entry);
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
                evict(entry);

            return;
        }

        if (cfg.rv_mode == RevalidateMode::ML) {
            assert(ml_predictor.has_value());
            auto confidence = ml_predictor->predict(entry, now_ts);
            if (confidence > cfg.conf_thres)
                reval(entry, now_ts);
            else
                evict(entry);
            return;
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
        while (!expiry_heap.empty() && expiry_heap.top().expiry_ts < ts) {
            auto [key, expiry_ts] = expiry_heap.top();
            expiry_heap.pop();

            auto entryf = lru_cache.find(key);
            if (entryf == lru_cache.end())
                continue; // already evicted

            auto &[k, entry] = *entryf;
            assert(k == key);
            assert(entry.ttl + entry.last_update_ts == expiry_ts);

            record_expiry(entry, ts);
            on_expire(entry, ts);
        }

        // handle cache logic
        auto entryf = lru_cache.find(req.key);

        // miss
        if (entryf == lru_cache.end()) {
            // get zone
            if (zone_stats.find(req.zone) == zone_stats.end())
                zone_stats[req.zone] = ZoneStats{};
            auto *zs = &zone_stats[req.zone];

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
                .accesses_since_update = 1,
                .last_access_ts = ts,
                .last_update_ts = ts,
                .next_access_ts = req.next_req_ts,
                .content_type = req.content_type,
                .zs = zs,
            };
            lru_cache.insert(req.key, new_entry);
            expiry_heap.push(ExpiryHeapEntry{req.key, ts + req.ttl});

            stats.fetches++;
            zs->fetches++;

            // evict until under capacity
            while (lru_cache.is_over_capacity())
                evict(lru_cache.tail());

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
        if (cache_age <= entry.ttl && entry.accesses_since_update > 0) {
            stats.hit_fresh++;
        } else if (cache_age <= entry.ttl && entry.accesses_since_update == 0) {
            stats.hit_reval++;
            entry.zs->rv_good++;
        } else if (cache_age <= entry.ttl + entry.ttstale) {
            stats.hit_stale++;
        } else {
            stats.miss_expired++;
        }

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
    ("trace-expiry,e", po::value<bool>()->default_value(false), "output a trace of expiry events (for ml training, writes to traces/expiry/)")
    ("input-file,i", po::value<vec<str>>(), "input trace file (zstd compressed)")
    ("key-sample-ratio,s", po::value<vec<u64>>(), "key sampling ratio (list)")
    ("rv-mode,m", po::value<vec<str>>(), "revalidation mode (never, always, oracle, heuristics, ml)")

    // heuristics params
    ("rv-min-ttl", po::value<vec<u64>>()->default_value({}, ""), "min ttl to revalidate (list)")
    ("rv-min-freq", po::value<vec<u64>>()->default_value({}, ""), "min accesses to revalidate (list)")
    ("rv-max-za", po::value<vec<f32>>()->default_value({}, ""), "max zone amplification (list)")

    // ml params
    ("ml-model-path", po::value<vec<str>>()->default_value({}, ""), "path to revalidation model file")
    ("ml-conf-thres", po::value<vec<f32>>()->default_value({}, ""), "threshold at which to revalidate")
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
    auto trace_expiry = vm["trace-expiry"].as<bool>();

    if (vm.count("rv-mode") == 0)
        throw std::runtime_error("must specify at least one rv-mode");
    auto mode = vm["rv-mode"].as<vec<str>>();

    auto input_files = vec<str>{"traces/sim/cf_a.bin.zst"};
    if (vm.count("input-file"))
        input_files = vm["input-file"].as<vec<str>>();
    auto key_sample_ratio = vm["key-sample-ratio"].as<vec<u64>>();
    auto rv_min_ttl = vm["rv-min-ttl"].as<vec<u64>>();
    auto rv_min_freq = vm["rv-min-freq"].as<vec<u64>>();
    auto rv_max_za = vm["rv-max-za"].as<vec<f32>>();
    auto model_path = vm["ml-model-path"].as<vec<str>>();
    auto conf_thres = vm["ml-conf-thres"].as<vec<f32>>();

    for (auto infile : input_files)
        if (!std::filesystem::exists(infile))
            FAIL("input file does not exist: " + infile);
    for (auto mpath : model_path)
        if (!std::filesystem::exists(mpath))
            FAIL("model path does not exist: " + mpath);

    for (auto infile : input_files)
        for (auto ksr : key_sample_ratio)
            for (auto mode : mode) {
                auto rv_mode = rv_mode_from_str(mode);
                if (rv_mode == RevalidateMode::ML) {
                    if (model_path.empty())
                        FAIL("must specify model-path for ML mode");
                    if (conf_thres.empty())
                        FAIL("must specify confidence thresholds for ML mode");
                    for (auto mpath : model_path)
                        for (auto thres : conf_thres)
                            configs.push_back(SimConfig{
                                .infile = infile,
                                .capacity_gib = capacity,
                                .key_sample_ratio = ksr,
                                .rv_mode = rv_mode,
                                .model_path = mpath,
                                .conf_thres = thres,
                            });
                } else if (rv_mode == RevalidateMode::HEURISTICS) {
                    for (auto rvt : rv_min_ttl)
                        for (auto rvf : rv_min_freq)
                            for (auto rvza : rv_max_za)
                                configs.push_back(SimConfig{
                                    .infile = infile,
                                    .capacity_gib = capacity,
                                    .key_sample_ratio = ksr,
                                    .rv_mode = rv_mode,
                                    .rv_min_ttl = rvt,
                                    .rv_min_freq = rvf,
                                    .rv_max_zone_amp = rvza,
                                });
                } else {
                    configs.push_back(SimConfig{
                        .infile = infile,
                        .capacity_gib = capacity,
                        .key_sample_ratio = ksr,
                        .rv_mode = rv_mode,
                    });
                }
            }

    if (trace_expiry) {
        for (auto &cfg : configs) {
            std::filesystem::create_directories("traces/expiry/");
            auto trace_outpath =
                fmt::format("traces/expiry/{}.csv", cfg.infile_base());
            auto f = fopen(trace_outpath.c_str(), "w");
            cfg.trace_outf = f;
        }
    }

    fmt::print("CSV output file: {}\n", csvout_file);
    auto outf = fmt::output_file(csvout_file);
    outf.print("{},{}\n", SimConfig::csv_hdr(), SimStats::csv_hdr());
    outf.flush();

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
                outf.flush();

                zstdcat.terminate();
                zstdcat.wait();
            }
        }));

    for (auto &t : threads)
        t.join();

    return 0;
}