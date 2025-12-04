#include "fmt/core.h"
#include "libonnxruntime/onnxruntime_cxx_api.h"
#include "utils.hpp"

class RevalPredictor
{
  private:
    static constexpr const u32 IN_FEATURES = 6;
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

    auto predict(u32 ttl, u32 freq, u32 generations, u32 t_since_last,
                 u32 mime) -> f32
    {
        static const auto runopts = Ort::RunOptions{};
        idata[0] = static_cast<f32>(ttl);
        idata[1] = static_cast<f32>(freq);
        idata[2] = static_cast<f32>(generations);
        idata[3] = static_cast<f32>(t_since_last);
        idata[4] = static_cast<f32>(t_since_last) / static_cast<f32>(ttl);
        idata[5] = static_cast<f32>(mime);
        session.Run(runopts, inames, &itensor, 1, onames, &otensor, 1);
        return oprob[1];
    }
};


auto main() -> int
{
    RevalPredictor predictor("models/rfc_cf_all.onnx");
    auto res = predictor.predict(86400, 1, 3, 17382, 19);
    fmt::print("Prediction: {}\n", res);
    return 0;
}
