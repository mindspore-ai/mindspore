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

#include "tools/benchmark_train/run_net_train.h"
#include <string>
#include "tools/benchmark_train/net_train.h"
#include "tools/benchmark_train/net_train_c_api.h"

namespace mindspore {
namespace lite {
int RunNetTrain(int argc, const char **argv) {
  NetTrainFlags flags;
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
  if (flags.unified_api_) {
    return NetTrain::RunNr(&flags);
  }

  auto api_type = std::getenv("MSLITE_API_TYPE");
  if (api_type != nullptr) {
    MS_LOG(INFO) << "MSLITE_API_TYPE = " << api_type;
    std::cout << "MSLITE_API_TYPE = " << api_type << std::endl;
  }

  NetTrainBase *net_trainer = nullptr;
  if (api_type == nullptr || std::string(api_type) == "NEW") {
    net_trainer = new (std::nothrow) NetTrain(&flags);
  } else if (std::string(api_type) == "C") {
    net_trainer = new (std::nothrow) NetTrainCApi(&flags);
  } else {
    MS_LOG(ERROR) << "Invalid MSLITE_API_TYPE, (NEW/C, default:NEW)";
    return RET_ERROR;
  }

  if (net_trainer == nullptr) {
    MS_LOG(ERROR) << "new net_trainer failed.";
    return RET_ERROR;
  }
  auto status = net_trainer->Init();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "NetTrain init Error : " << status;
    std::cerr << "NetTrain init Error : " << status << std::endl;
    return RET_ERROR;
  }

  status = net_trainer->RunNetTrain();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Run NetTrain " << flags.model_file_.substr(flags.model_file_.find_last_of("/") + 1).c_str()
                  << " Failed : " << status;
    std::cerr << "Run NetTrain " << flags.model_file_.substr(flags.model_file_.find_last_of("/") + 1).c_str()
              << " Failed : " << status << std::endl;
    return RET_ERROR;
  }

  MS_LOG(INFO) << "Run NetTrain " << flags.model_file_.substr(flags.model_file_.find_last_of("/") + 1).c_str()
               << " Success.";
  std::cout << "Run NetTrain " << flags.model_file_.substr(flags.model_file_.find_last_of("/") + 1).c_str()
            << " Success." << std::endl;
  delete net_trainer;
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
