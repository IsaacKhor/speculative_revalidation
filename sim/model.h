#pragma once

#include <cstdint>

using u64 = uint64_t;

struct CdnRequest {
    u64 timestamp;
    u64 key;
    u64 zone;
    u64 size;
    u64 expiry;
    u64 stale;
    bool is_purge;
};
