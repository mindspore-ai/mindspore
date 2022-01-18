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
#include "tools/benchmark/benchmark.h"
#include "tools/benchmark/benchmark_unified_api.h"
#include "tools/benchmark/benchmark_c_api.h"

namespace mindspore {
namespace lite {
int RunBenchmark(int argc, const char **argv) {
  BenchmarkFlags flags;
  Option<std::string> err = flags.ParseFlags(argc, argv);
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

  std::unique_ptr<BenchmarkBase> benchmark;
  if (flags.config_file_ != "" || (api_type != nullptr && std::string(api_type) == "NEW")) {
    benchmark = std::make_unique<BenchmarkUnifiedApi>(&flags);
  } else if (api_type == nullptr || std::string(api_type) == "OLD") {
    benchmark = std::make_unique<Benchmark>(&flags);
  } else if (std::string(api_type) == "C") {
    benchmark = std::make_unique<tools::BenchmarkCApi>(&flags);
  } else {
    BENCHMARK_LOG_ERROR("Invalid MSLITE_API_TYPE, (OLD/NEW/C, default:OLD)");
    return RET_ERROR;
  }
  if (benchmark == nullptr) {
    BENCHMARK_LOG_ERROR("new benchmark failed ");
    return RET_ERROR;
  }

  auto status = benchmark->Init();
  if (status != 0) {
    BENCHMARK_LOG_ERROR("Benchmark init Error : " << status);
    return RET_ERROR;
  }
  auto model_name = flags.model_file_.substr(flags.model_file_.find_last_of(DELIM_SLASH) + 1);

  status = benchmark->RunBenchmark();
  if (status != 0) {
    BENCHMARK_LOG_ERROR("Run Benchmark " << model_name << " Failed : " << status);
    return RET_ERROR;
  }

  MS_LOG(INFO) << "Run Benchmark " << model_name << " Success.";
  std::cout << "Run Benchmark " << model_name << " Success." << std::endl;
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
