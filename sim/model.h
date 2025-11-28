#pragma once

#include <cstdint>
#include <fmt/core.h>

using str = std::string;
using u32 = uint32_t;
using u64 = uint64_t;

struct Req {
    u64 key;
    u64 zone;
    u64 size;
    u32 ts;
    u32 next_req_ts;
    u32 ttl;
    u32 ttstale;
    bool is_purge;

    inline auto str() const -> str
    {
        return fmt::format(
            "Req(ts={}, next_ts={}, key={:16x}, size={}, ttl={})", ts,
            next_req_ts, key, size, ttl);
    }
};
