#pragma once

#ifdef NDEBUG
#define DEBUG 0
#define FAIL(msg)                                                              \
    do {                                                                       \
        fmt::print(stderr, "FATAL: {}\n", msg);                                \
        std::abort();                                                          \
    } while (0)
#define breakpoint()
#else
#define DEBUG 1
#define FAIL(msg)                                                              \
    do {                                                                       \
        fmt::print(stderr, "FATAL: {}\n", msg);                                \
        asm("int3; nop");                                                      \
        throw std::runtime_error(msg);                                         \
    } while (0)
#define breakpoint() asm("int3; nop");
#endif

#include <boost/algorithm/string.hpp>
#include <boost/process.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <fmt/core.h>
#include <vector>

namespace bp = boost::process;
using f32 = float;
using f64 = double;
using u64 = uint64_t;
using u32 = uint32_t;
using i64 = int64_t;
using i32 = int32_t;
using str = std::string;
using strv = std::string_view;
template <typename T> using vec = std::vector<T>;

inline auto tnow() { return std::chrono::steady_clock::now(); }
inline auto tsince(std::chrono::time_point<std::chrono::steady_clock> &start)
{
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::steady_clock::now() - start)
        .count();
}

inline auto capadd(u32 x, u32 y) -> u32
{
    if (x >= UINT32_MAX - y)
        return UINT32_MAX;
    return x + y;
}

inline auto capmul(u32 x, u32 y) -> u32
{
    if (x == 0 || y == 0)
        return 0;
    if (x >= UINT32_MAX / y)
        return UINT32_MAX;
    return x * y;
}

enum class RevalidateMode {
    NEVER,
    ALWAYS,
    ORACLE,
    HEURISTICS,
    ML,
};

enum class CacheType {
    LRU,
    GDSF,
};

constexpr auto rv_mode_str(RevalidateMode m) -> str
{
    switch (m) {
    case RevalidateMode::NEVER:
        return "never";
    case RevalidateMode::ALWAYS:
        return "always";
    case RevalidateMode::ORACLE:
        return "oracle";
    case RevalidateMode::HEURISTICS:
        return "heuristics";
    case RevalidateMode::ML:
        return "ml";
    default:
        FAIL("unknown RevalidateMode");
    }
}

inline auto rv_mode_from_str(const str &s) -> RevalidateMode
{
    auto lower = boost::algorithm::to_lower_copy(s);
    if (lower == "never")
        return RevalidateMode::NEVER;
    if (lower == "always")
        return RevalidateMode::ALWAYS;
    if (lower == "oracle")
        return RevalidateMode::ORACLE;
    if (lower == "heuristics")
        return RevalidateMode::HEURISTICS;
    if (lower == "ml")
        return RevalidateMode::ML;
    FAIL("unknown RevalidateMode string: " + s);
}

struct SimConfig {
    str infile;                 // must be zstd-compressed binary trace
    u64 capacity_gib = 2048;    // cache capacity; reduced by key_sample_ratio
    u64 key_sample_ratio = 1;   // only sample 1 in N keys
    str cache_type = "lru";     // cache implementation to use
    FILE *trace_outf = nullptr; // expiry trace file output, null for none
    RevalidateMode rv_mode = RevalidateMode::NEVER;

    // ml params
    str model_path = "";
    f32 conf_thres = 1;

    // heuristics params
    u64 rv_min_ttl = 0;
    u64 rv_min_freq = 0;
    f64 rv_max_zone_amp = 0;

    bool evict_expired = false;
    FILE *zonestats_outf = nullptr;
    FILE *ts_outf = nullptr; // per-100k-request time series, null for none

    inline auto infile_base() const -> str
    {
        auto posl = infile.find_last_of('/');
        posl = posl == str::npos ? 0 : posl + 1;
        auto stripl = infile.substr(posl);
        auto posr = stripl.find_first_of('.');
        posr = posr == str::npos ? stripl.size() : posr;
        return stripl.substr(0, posr);
    }

    inline auto repr() const -> str
    {
        return fmt::format(
            "SimConfig(in={}, cache_gib={}, ksr={}, cache_type={}, mode={}, "
            "model={}, "
            "mlthres={}, rv_min_ttl={}, rv_min_freq={}, rv_max_za={}, "
            "evict_expired={})",
            infile, capacity_gib, key_sample_ratio, cache_type,
            rv_mode_str(rv_mode), model_path, conf_thres, rv_min_ttl,
            rv_min_freq, rv_max_zone_amp, evict_expired);
    }

    inline static auto csv_hdr() -> str
    {
         return "in,gib,ksr,cache_type,mode,model,mlthres,rv_min_ttl,rv_min_"
             "freq,rv_max_za,evict_expired";
    }

    inline auto csv() const -> str
    {
        return fmt::format("{},{},{},{},{},{},{},{},{},{},{}", infile_base(),
                           capacity_gib, key_sample_ratio, cache_type,
                           rv_mode_str(rv_mode), model_path, conf_thres,
                           rv_min_ttl, rv_min_freq, rv_max_zone_amp,
                           evict_expired);
    }
};

struct SimStats {
    u64 all = 0;
    u64 hit_fresh = 0;
    u64 hit_reval = 0;
    u64 hit_stale = 0;
    u64 miss_mandatory = 0;
    u64 miss_expired = 0;
    u64 miss_evicted = 0;

    u64 misses_all = 0; // all origin fetches, misses and revalidations
    u64 rv_fetch = 0;   // revalidation fetches
    u64 rv_wasted = 0;  // revalidiations that were wasted

    inline auto human_str() const -> str
    {
        auto denom = (double)all / 100.0;
        auto hit_fresh_pc = (double)this->hit_fresh / denom;
        auto hit_reval_pc = (double)this->hit_reval / denom;
        auto hit_stale_pc = (double)this->hit_stale / denom;
        auto all_hit = hit_fresh + hit_reval + hit_stale;
        auto miss_mandatory_pc = (double)this->miss_mandatory / denom;
        auto miss_expired_pc = (double)this->miss_expired / denom;
        auto miss_evicted_pc = (double)this->miss_evicted / denom;
        auto all_miss = miss_mandatory + miss_expired + miss_evicted;

        auto good = hit_reval;
        auto bad = rv_wasted;
        auto pending = rv_fetch - good - bad;
        auto revals_pc = (double)rv_fetch / misses_all;
        auto good_pc = (double)good / misses_all;
        auto pending_pc = (double)pending / misses_all;
        auto wasted_pc = (double)rv_wasted / misses_all;

        auto amp_lower = (double)(bad) / (double)(all_miss);
        auto amp_upper = (double)(bad + pending) / (double)(all_miss);

        auto revals_wasted_pc = (double)rv_wasted / (double)rv_fetch * 100.0;

        auto cachestr = fmt::format(
            R"(
Total Requests: {}
Hits:
  Fresh: {:.02f}% ({})
  Reval: {:.02f}% ({})
  Stale: {:.02f}% ({})
Misses:
  First: {:.02f}% ({})
  Expir: {:.02f}% ({})
  Evict: {:.02f}% ({})
        )",
            all, hit_fresh_pc, hit_fresh, hit_reval_pc, hit_reval, hit_stale_pc,
            hit_stale, miss_mandatory_pc, miss_mandatory, miss_expired_pc,
            miss_expired, miss_evicted_pc, miss_evicted);

        auto originstr = fmt::format(
            R"(
Origin: fetches/revals/good/pending/wasted
Origin: 100/{:.02f}/{:.02f}/{:.02f}/{:.02f}
Origin: {}/{}/{}/{}/{}
Amp: {:.02f} - {:.02f}x
% revals wasted: {:.02f}%
)",
            revals_pc, good_pc, pending_pc, wasted_pc, misses_all, rv_fetch,
            good, pending, bad, amp_lower, amp_upper, revals_wasted_pc);

        // return cachestr + originstr;
        return fmt::format(
            "Amp: {:.02f} - {:.02f}x, % revals wasted: {:.02f}%\n", amp_lower,
            amp_upper, revals_wasted_pc);
    }

    inline auto csv() const -> str
    {
        return fmt::format("{},{},{},{},{},{},{},{},{},{}", all, hit_fresh,
                           hit_reval, hit_stale, miss_mandatory, miss_expired,
                           miss_evicted, misses_all, rv_fetch, rv_wasted);
    }

    inline static auto csv_hdr() -> str
    {
        return "all,hit_fresh,hit_reval,hit_stale,miss_mandatory,miss_expired,"
               "miss_evicted,fetches,revals,reval_wasted";
    }

    // snapshot of cumulative counters at the last time-series emit; used to
    // produce per-window deltas that effectively reset every TS_WINDOW requests
    static constexpr u64 TS_WINDOW = 100'000;
    u64 ts_last_all = 0;
    u64 ts_last_hit_fresh = 0;
    u64 ts_last_hit_reval = 0;
    u64 ts_last_hit_stale = 0;
    u64 ts_last_miss_mandatory = 0;
    u64 ts_last_miss_expired = 0;
    u64 ts_last_miss_evicted = 0;

    inline static auto ts_csv_hdr() -> str
    {
        return "all,now_ts,hits,revals,stales,miss_mandatory,miss_expired,"
               "miss_evicted";
    }

    // emit one time-series row if at least TS_WINDOW requests have elapsed
    // since the last emit, then advance the snapshot. now_ts is the current
    // simulation timestamp (trace wall-clock seconds).
    inline auto emit_ts(FILE *outf, u32 now_ts) -> void
    {
        if (outf == nullptr)
            return;
        if (all - ts_last_all < TS_WINDOW)
            return;
        fmt::print(outf, "{},{},{},{},{},{},{},{}\n", all, now_ts,
                   hit_fresh - ts_last_hit_fresh,
                   hit_reval - ts_last_hit_reval,
                   hit_stale - ts_last_hit_stale,
                   miss_mandatory - ts_last_miss_mandatory,
                   miss_expired - ts_last_miss_expired,
                   miss_evicted - ts_last_miss_evicted);
        ts_last_all = all;
        ts_last_hit_fresh = hit_fresh;
        ts_last_hit_reval = hit_reval;
        ts_last_hit_stale = hit_stale;
        ts_last_miss_mandatory = miss_mandatory;
        ts_last_miss_expired = miss_expired;
        ts_last_miss_evicted = miss_evicted;
        fflush(outf);
    }
};
