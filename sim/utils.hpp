#pragma once

#ifdef DEBUG
#define DEBUG 1
#else
#define DEBUG 0
#endif

#include <boost/process.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <fmt/core.h>
#include <vector>

namespace bp = boost::process;
using f64 = double;
using u64 = uint64_t;
using u32 = uint32_t;
using str = std::string;
template <typename T> using vec = std::vector<T>;

struct Req {
    u64 ts;
    u64 key;
    u64 zone;
    u64 size;
    u64 ttl;
    u64 ttstale;
    bool is_purge;

    inline auto str() const -> str
    {
        return fmt::format("Req(ts={}, key={:16x}, size={}, ttl={})", ts, key,
                           size, ttl);
    }
};

inline auto tnow() { return std::chrono::steady_clock::now(); }
inline auto tsince(std::chrono::time_point<std::chrono::steady_clock> &start)
{
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::steady_clock::now() - start)
        .count();
}

struct SimConfig {
    str infile;
    u64 capacity_mb;
    u64 key_sample_ratio;
    bool evict_expired;

    // revalidation params
    bool rv_enable;
    u64 rv_min_ttl;
    u64 rv_min_freq;
    f64 rv_max_zone_amp; // not currently implemented

    inline auto repr() const -> str
    {
        return fmt::format(
            "SimConfig(infile={}, capacity_mb={}, key_sample_ratio={}, "
            "evict_expired={}, "
            "rv_enable={}, rv_min_ttl={}, rv_min_freq={}, rv_max_zone_amp={})",
            infile, capacity_mb, key_sample_ratio, evict_expired, rv_enable,
            rv_min_ttl, rv_min_freq, rv_max_zone_amp);
    }

    inline static auto csv_hdr() -> str
    {
        return "infile,capacity_mb,key_sample_ratio,evict_expired,rv_enable,rv_"
               "min_ttl,rv_min_freq,rv_max_zone_amp";
    }

    inline auto csv() const -> str
    {
        return fmt::format("{},{},{},{},{},{},{},{}", infile, capacity_mb,
                           key_sample_ratio, evict_expired, rv_enable,
                           rv_min_ttl, rv_min_freq, rv_max_zone_amp);
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

    inline auto human_str() const -> str
    {
        auto denom = (double)all / 100.0;
        auto hit_fresh_pc = (double)this->hit_fresh / denom;
        auto hit_reval_pc = (double)this->hit_reval / denom;
        auto hit_stale_pc = (double)this->hit_stale / denom;
        auto miss_mandatory_pc = (double)this->miss_mandatory / denom;
        auto miss_expired_pc = (double)this->miss_expired / denom;
        auto miss_evicted_pc = (double)this->miss_evicted / denom;

        return fmt::format("Total Requests: {}\n"
                           "Hits:\n"
                           "  Fresh: {:.02f}% ({})\n"
                           "  Reval: {:.02f}% ({})\n"
                           "  Stale: {:.02f}% ({})\n"
                           "Misses:\n"
                           "  First: {:.02f}% ({})\n"
                           "  Expir: {:.02f}% ({})\n"
                           "  Evict: {:.02f}% ({})\n",
                           all, hit_fresh_pc, hit_fresh, hit_reval_pc,
                           hit_reval, hit_stale_pc, hit_stale,
                           miss_mandatory_pc, miss_mandatory, miss_expired_pc,
                           miss_expired, miss_evicted_pc, miss_evicted);
    }

    inline auto csv() const -> str
    {
        return fmt::format("{},{},{},{},{},{},{}", all, hit_fresh, hit_reval,
                           hit_stale, miss_mandatory, miss_expired,
                           miss_evicted);
    }

    inline static auto csv_hdr() -> str
    {
        return "all,hit_fresh,hit_reval,hit_stale,miss_mandatory,miss_expired,"
               "miss_evicted";
    }
};
