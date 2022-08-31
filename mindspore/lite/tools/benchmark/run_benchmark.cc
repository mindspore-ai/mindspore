/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tools/benchmark/run_benchmark.h"
#include <string>
#include <memory>
#include "tools/benchmark/benchmark_unified_api.h"
#ifndef ENABLE_CLOUD_FUSION_INFERENCE
#include "tools/benchmark/benchmark_c_api.h"
#endif

namespace mindspore {
namespace lite {
int RunBenchmark(int argc, const char **argv) {
  BenchmarkFlags flags;
  Option<std::string> err = flags.ParseFlags(argc, argv);
#ifdef SUPPORT_NNIE
  if (SvpSysInit() != RET_OK) {
    std::cerr << "SVP Init failed" << std::endl;
    return RET_ERROR;
  }
#endif
  if (err.IsSome()) {
    std::cerr << err.Get() << std::endl;
    std::cerr << flags.Usage() << std::endl;
    return RET_ERROR;
  }

  if (flags.help) {
    std::cerr << flags.Usage() << std::endl;
    return RET_OK;
  }

  auto api_type = std::getenv("MSLITE_API_TYPE");
  if (api_type != nullptr) {
    MS_LOG(INFO) << "MSLITE_API_TYPE = " << api_type;
    std::cout << "MSLITE_API_TYPE = " << api_type << std::endl;
  }
  BenchmarkBase *benchmark = nullptr;
  if (api_type == nullptr || std::string(api_type) == "NEW") {
    benchmark = new (std::nothrow) BenchmarkUnifiedApi(&flags);
  } else if (std::string(api_type) == "C") {
#ifndef ENABLE_CLOUD_FUSION_INFERENCE
    benchmark = new (std::nothrow) tools::BenchmarkCApi(&flags);
#endif
  } else {
    BENCHMARK_LOG_ERROR("Invalid MSLITE_API_TYPE, (NEW/C, default:NEW)");
    return RET_ERROR;
  }
  if (benchmark == nullptr) {
    BENCHMARK_LOG_ERROR("new benchmark failed ");
    return RET_ERROR;
  }

  auto status = benchmark->Init();
  if (status != 0) {
    BENCHMARK_LOG_ERROR("Benchmark init Error : " << status);
    delete benchmark;
    return RET_ERROR;
  }
  auto model_name = flags.model_file_.substr(flags.model_file_.find_last_of(DELIM_SLASH) + 1);

  status = benchmark->RunBenchmark();
  if (status != 0) {
    BENCHMARK_LOG_ERROR("Run Benchmark " << model_name << " Failed : " << status);
    delete benchmark;
    return RET_ERROR;
  }

  MS_LOG(INFO) << "Run Benchmark " << model_name << " Success.";
  std::cout << "Run Benchmark " << model_name << " Success." << std::endl;
  delete benchmark;
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
