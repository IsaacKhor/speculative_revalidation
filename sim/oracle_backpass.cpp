#include "absl/container/flat_hash_map.h"
#include "model.h"
#include <cstdio>
#include <fcntl.h>
#include <map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define FAIL(cond, msg)                                                        \
    if (cond)                                                                  \
        do {                                                                   \
            perror(msg);                                                       \
            exit(EXIT_FAILURE);                                                \
    } while (0)

auto main(int argc, char **argv) -> int
{
    if (argc != 2) {
        fmt::print("Usage: {} <input_file>\n", argv[0]);
        return 1;
    }

    auto input_file = argv[1];
    auto fd = open(input_file, O_RDWR);
    FAIL(fd < 0, "Failed to open input file");

    struct stat st;
    auto res = fstat(fd, &st);
    FAIL(res < 0, "Failed to stat input file");

    auto num_reqs = st.st_size / sizeof(Req);
    FAIL(st.st_size % sizeof(Req) != 0,
         "Input file size is not a multiple of Req size");

    fmt::print("Oracle backpass on file: {}\n", input_file);
    fmt::print("File size: {} bytes, num_reqs: {}\n", st.st_size, num_reqs);

    auto addr =
        mmap(nullptr, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    FAIL(addr == MAP_FAILED, "Failed to mmap input file");

    std::map<u64, u64> last_seen;
    // absl::flat_hash_map<u64, u64> last_seen;
    Req *file_reqs = (Req *)addr;

    for (u64 i = 0; i < num_reqs; i++) {
        Req *req = &file_reqs[num_reqs - i - 1];
        if (i < 5)
            fmt::print("Req: {}\n", req->str());

        if (i % 1'000'000 == 0 && i > 0) {
            fmt::print(".", i / 1'000'000);
            fflush(stdout);
        }

        if (i % 50'000'000 == 0 && i > 0)
            fmt::print(" {}m reqs\n", i / 1'000'000);

        auto k = req->key;

        auto it = last_seen.find(k);
        if (it != last_seen.end())
            req->next_req_ts = it->second;
        else
            req->next_req_ts = 0;

        last_seen[req->key] = req->ts;
    }

    fmt::print("\nOracle backpass complete.\n");

    res = munmap(addr, st.st_size);
    FAIL(res < 0, "Failed to munmap input file");
    close(fd);
}