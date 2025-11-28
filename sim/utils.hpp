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

#include "model.h"

namespace bp = boost::process;
using f64 = double;
using u64 = uint64_t;
using u32 = uint32_t;
using str = std::string;
template <typename T> using vec = std::vector<T>;

inline auto tnow() { return std::chrono::steady_clock::now(); }
inline auto tsince(std::chrono::time_point<std::chrono::steady_clock> &start)
{
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::steady_clock::now() - start)
        .count();
}

struct SimConfig {
    str infile;
    u64 capacity_gib;
    u64 key_sample_ratio;
    bool evict_expired;

    // revalidation params
    bool rv_enable;
    u64 rv_min_ttl;
    u64 rv_min_freq;
    f64 rv_max_zone_amp; // not currently implemented

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
            "SimConfig(infile={}, capacity_mb={}, key_sample_ratio={}, "
            "evict_expired={}, "
            "rv_enable={}, rv_min_ttl={}, rv_min_freq={}, rv_max_zone_amp={})",
            infile, capacity_gib, key_sample_ratio, evict_expired, rv_enable,
            rv_min_ttl, rv_min_freq, rv_max_zone_amp);
    }

    inline static auto csv_hdr() -> str
    {
        return "infile,capacity_mb,key_sample_ratio,evict_expired,rv_enable,rv_"
               "min_ttl,rv_min_freq,rv_max_zone_amp";
    }

    inline auto csv() const -> str
    {
        return fmt::format("{},{},{},{},{},{},{},{}", infile_base(),
                           capacity_gib, key_sample_ratio, evict_expired,
                           rv_enable, rv_min_ttl, rv_min_freq, rv_max_zone_amp);
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

    // metric we want: origin amplification, both optimistic and pessimistic
    // this means failed origin fetch / total origin fetch
    // total_revals = hit_reval + reval_wasted + reval_pending
    // total_fetches = all miss + total_revals
    // TODO make a per-zone version
    u64 fetches = 0;
    u64 revals = 0;
    u64 reval_wasted = 0;

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
        auto bad = reval_wasted;
        auto pending = revals - good - bad;
        auto revals_pc = (double)revals / fetches;
        auto good_pc = (double)good / fetches;
        auto pending_pc = (double)pending / fetches;
        auto wasted_pc = (double)reval_wasted / fetches;

        auto amp_lower = (double)(bad) / (double)(all_miss);
        auto amp_upper = (double)(bad + pending) / (double)(all_miss);

        auto revals_wasted_pc = (double)reval_wasted / (double)revals * 100.0;

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
            revals_pc, good_pc, pending_pc, wasted_pc, fetches, revals, good,
            pending, bad, amp_lower, amp_upper, revals_wasted_pc);

        return cachestr + originstr;
    }

    inline auto csv() const -> str
    {
        return fmt::format("{},{},{},{},{},{},{},{},{},{}", all, hit_fresh,
                           hit_reval, hit_stale, miss_mandatory, miss_expired,
                           miss_evicted, fetches, revals, reval_wasted);
    }

    inline static auto csv_hdr() -> str
    {
        return "all,hit_fresh,hit_reval,hit_stale,miss_mandatory,miss_expired,"
               "miss_evicted,fetches,revals,reval_wasted";
    }
};
