#include "eviction.hpp"
#include "libonnxruntime/onnxruntime_cxx_api.h"
#include "model.h"
#include "utils.hpp"
#include <algorithm>
#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>
#include <cassert>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fmt/core.h>
#include <fmt/os.h>
#include <fmt/ranges.h>
#include <iostream>
#include <limits>
#include <thread>
#include <unordered_map>
#include <vector>
#include <zstd.h>

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

int main(int argc, char **argv)
{
    if (argc < 2) {
        fmt::print("Usage: {} <model_path>\n", argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    RevalPredictor predictor(model_path);

    const int iterations = 1000000;

    // Dummy data
    u32 ttl = 3600;
    u32 freq = 5;
    u32 generations = 2;
    u32 t_since_last = 60;
    u32 mime = 1;
    u32 size = 1024;

    fmt::print("Running {} iterations...\n", iterations);
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        volatile float res = predictor.predict(ttl, freq, generations, t_since_last + (i % 10), mime, size);
        (void)res;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    double avg_time_ns = (diff.count() * 1e9) / iterations;

    fmt::print("Total time: {:.4f} s\n", diff.count());
    fmt::print("Average time per prediction: {:.4f} ns\n", avg_time_ns);

    return 0;
}
