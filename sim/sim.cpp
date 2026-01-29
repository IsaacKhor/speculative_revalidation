#include "eviction.hpp"
#include "libonnxruntime/onnxruntime_cxx_api.h"
#include "model.h"
#include "utils.hpp"
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

const u64 MAX_EXPIRY_TIME = 30 * 24 * 60 * 60; // 30 days
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

struct ExpiryHeapEntry {
    u64 key;
    u32 expiry_ts;

    bool operator>(const ExpiryHeapEntry &other) const
    {
        return expiry_ts > other.expiry_ts;
    }
};

template <typename T> class MinHeap
{
  public:
    void push(const T &entry)
    {
        // if (entry.expiry_ts > MAX_EXPIRY_TIME)
        //     return;
        heap.push_back(entry);
        std::push_heap(heap.begin(), heap.end(), std::greater<T>());
    }
    void pop()
    {
        std::pop_heap(heap.begin(), heap.end(), std::greater<T>());
        heap.pop_back();
    }
    auto top() -> T & { return heap.front(); }
    auto empty() const { return heap.empty(); }

    vec<T> heap;

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

class CacheSimulator
{
  private:
    SimConfig cfg;
    std::unique_ptr<Cache> cache;
    MinHeap<ExpiryHeapEntry> expiry_heap;
    absl::flat_hash_set<u64> seen_keys;
    absl::node_hash_map<u64, ZoneStats> zone_stats;
    SimStats stats;
    FILE *trace_outf;
    std::optional<RevalPredictor> ml_predictor = std::nullopt;

  public:
    CacheSimulator(SimConfig cfg) : cfg(cfg), trace_outf(cfg.trace_outf)
    {
        // Create the appropriate cache implementation
        auto capacity =
            cfg.capacity_gib * 1024 * 1024 * 1024 / cfg.key_sample_ratio;
        auto exit_callback = [this](CacheEntry &entry) {
            on_cache_exit(entry);
        };

        cache = make_cache(cfg.cache_type, capacity, exit_callback);
        if (cfg.rv_mode == RevalidateMode::ML)
            ml_predictor.emplace(cfg.model_path);
    }

    auto on_cache_exit(CacheEntry &entry) -> void
    {
        // check if it was a wasted revalidation
        if (entry.entry_create_ts < entry.last_update_ts &&
            entry.accesses_since_update == 0) {
            stats.rv_wasted++;
            entry.zs->rv_wasted++;
        }

        auto key = entry.key;
        seen_keys.insert(key);
    }

    auto reval(CacheEntry &entry, u32 now_ts)
    {
        // check if it was a wasted revalidation
        if (entry.entry_create_ts < entry.last_update_ts &&
            entry.accesses_since_update == 0) {
            stats.rv_wasted++;
            entry.zs->rv_wasted++;
        }

        stats.rv_fetch++;
        entry.zs->rv_fetch++;
        entry.last_update_ts = now_ts;
        entry.accesses_since_update = 0;

        // don't bother if ttl is too big (trace lasts roughly 1 month)
        if (entry.ttl > 86400 * 30)
            return;

        expiry_heap.push(ExpiryHeapEntry{entry.key, capadd(now_ts, entry.ttl)});
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
                   entry.last_update_ts, zs.rv_fetch, zs.rv_good, zs.rv_wasted,
                   entry.content_type, entry.size);
    }

    auto on_expire(CacheEntry &entry, u32 now_ts)
    {
        if (cfg.rv_mode == RevalidateMode::NEVER) {
            cache->remove(entry.key);
            return;
        }

        if (cfg.rv_mode == RevalidateMode::ALWAYS) {
            reval(entry, now_ts);
            return;
        }

        if (cfg.rv_mode == RevalidateMode::ORACLE) {
            // oracle means reval if future access < ttl and cache cycle

            // different thresholds: how many ttl's ahead to look?
            auto allowable_delay = capmul(entry.ttl, cfg.conf_thres);

            // TODO implement cache cycle time check
            if (entry.next_access_ts >= now_ts &&
                entry.next_access_ts <= capadd(now_ts, allowable_delay))
                reval(entry, now_ts);
            else
                cache->remove(entry.key);
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
                cache->remove(entry.key);

            return;
        }

        if (cfg.rv_mode == RevalidateMode::ML) {
            assert(ml_predictor.has_value());
            auto confidence = ml_predictor->predict(entry, now_ts);
            if (confidence > cfg.conf_thres)
                reval(entry, now_ts);
            else
                cache->remove(entry.key);
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

            auto entryf = cache->find(key);
            if (entryf == nullptr)
                continue; // already evicted

            // assert(capadd(entryf->ttl, entryf->last_update_ts) == expiry_ts);

            record_expiry(*entryf, ts);
            on_expire(*entryf, ts);
        }

        // handle cache logic
        auto entryf = cache->find(req.key);

        // miss
        if (entryf == nullptr) {
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
            cache->admit(req.key, new_entry);
            expiry_heap.push(ExpiryHeapEntry{req.key, capadd(ts, req.ttl)});

            stats.misses_all++;
            zs->misses++;

            return;
        }

        // hit
        auto &entry = *entryf;
        assert(entry.key == req.key);

        if (entry.next_access_ts != ts)
            breakpoint();

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
        entry.zs->hits++;
        entry.last_access_ts = ts;
        entry.accesses_since_update++;
        entry.next_access_ts = req.next_req_ts;

        // update cache state for lru/gdsf/sieve
        cache->touch(req.key);
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
    ("cache-type,y", po::value<vec<str>>(), "cache type (lru, gdsf)")
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
    ("ml-conf-thres,t", po::value<vec<f32>>()->default_value({}, ""), "threshold at which to revalidate")
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

    auto cache_types = vec<str>{"lru"};
    if (vm.count("cache-type"))
        cache_types = vm["cache-type"].as<vec<str>>();
    for (auto &ct : cache_types)
        if (!validate_cache_type(ct))
            FAIL("unknown cache type: " + ct);

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
        for (auto ct : cache_types)
            for (auto ksr : key_sample_ratio)
                for (auto mode : mode) {
                    auto rv_mode = rv_mode_from_str(mode);
                    if (rv_mode == RevalidateMode::ML) {
                        if (model_path.empty())
                            FAIL("must specify model-path for ML mode");
                        if (conf_thres.empty())
                            FAIL("must specify confidence thresholds for ML "
                                 "mode");
                        for (auto mpath : model_path)
                            for (auto thres : conf_thres)
                                configs.push_back(SimConfig{
                                    .infile = infile,
                                    .capacity_gib = capacity,
                                    .key_sample_ratio = ksr,
                                    .cache_type = ct,
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
                                        .cache_type = ct,
                                        .rv_mode = rv_mode,
                                        .rv_min_ttl = rvt,
                                        .rv_min_freq = rvf,
                                        .rv_max_zone_amp = rvza,
                                    });
                    } else if (rv_mode == RevalidateMode::ORACLE) {
                        for (auto thres : conf_thres)
                            configs.push_back(SimConfig{
                                .infile = infile,
                                .capacity_gib = capacity,
                                .key_sample_ratio = ksr,
                                .cache_type = ct,
                                .rv_mode = rv_mode,
                                .conf_thres = thres,
                            });
                    } else {
                        configs.push_back(SimConfig{
                            .infile = infile,
                            .capacity_gib = capacity,
                            .key_sample_ratio = ksr,
                            .cache_type = ct,
                            .rv_mode = rv_mode,
                        });
                    }
                }

    if (trace_expiry) {
        std::filesystem::create_directories("traces/expiry/");
        for (auto &cfg : configs) {
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