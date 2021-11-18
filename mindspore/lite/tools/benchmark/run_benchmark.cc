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

  // get dump data output path
  auto new_api = std::getenv("ENABLE_NEW_API");
  bool run_old_api = (new_api == nullptr || std::string(new_api) != "true");
  if (flags.config_file_ != "") {
    run_old_api = false;
  }
  std::unique_ptr<BenchmarkBase> benchmark;
  if (run_old_api) {
    benchmark = std::make_unique<Benchmark>(&flags);
  } else {
    benchmark = std::make_unique<BenchmarkUnifiedApi>(&flags);
  }
  if (benchmark == nullptr) {
    MS_LOG(ERROR) << "new benchmark failed ";
    std::cerr << "new benchmark failed" << std::endl;
    return RET_ERROR;
  }
  auto status = benchmark->Init();
  if (status != 0) {
    MS_LOG(ERROR) << "Benchmark init Error : " << status;
    std::cerr << "Benchmark init Error : " << status << std::endl;
    return RET_ERROR;
  }

  status = benchmark->RunBenchmark();
  if (status != 0) {
    MS_LOG(ERROR) << "Run Benchmark "
                  << flags.model_file_.substr(flags.model_file_.find_last_of(DELIM_SLASH) + 1).c_str()
                  << " Failed : " << status;
    std::cerr << "Run Benchmark " << flags.model_file_.substr(flags.model_file_.find_last_of(DELIM_SLASH) + 1).c_str()
              << " Failed : " << status << std::endl;
    return RET_ERROR;
  }

  MS_LOG(INFO) << "Run Benchmark " << flags.model_file_.substr(flags.model_file_.find_last_of(DELIM_SLASH) + 1).c_str()
               << " Success.";
  std::cout << "Run Benchmark " << flags.model_file_.substr(flags.model_file_.find_last_of(DELIM_SLASH) + 1).c_str()
            << " Success." << std::endl;
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
