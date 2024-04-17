/**
 * Copyright 2023-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_BENCHMARK_NET_TRAIN_C_API_H
#define MINDSPORE_LITE_TOOLS_BENCHMARK_NET_TRAIN_C_API_H

#include <getopt.h>
#include <csignal>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <map>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <cfloat>
#include <utility>
#include <algorithm>
#include <nlohmann/json.hpp>
#include "include/api/model.h"
#include "include/api/types.h"
#include "include/api/context.h"
#include "include/api/cfg.h"

#include "include/c_api/model_c.h"
#include "include/c_api/context_c.h"

#ifdef ENABLE_FP16
#include <arm_neon.h>
#endif
#include "tools/common/flag_parser.h"
#include "src/common/file_utils.h"
#include "src/common/utils.h"
#include "tools/benchmark_train/net_train_base.h"

namespace mindspore::lite {
#ifdef __cplusplus
extern "C" {
#endif
bool TimeProfilingBeforeCallback(const MSTensorHandleArray inputs, const MSTensorHandleArray outputs,
                                 const MSCallBackParamC kernel_Info);
bool TimeProfilingAfterCallback(const MSTensorHandleArray inputs, const MSTensorHandleArray outputs,
                                const MSCallBackParamC kernel_Info);
#ifdef __cplusplus
}
#endif

class MS_API NetTrainCApi : public NetTrainBase {
 public:
  explicit NetTrainCApi(NetTrainFlags *flags) : NetTrainBase(flags) {}
  virtual ~NetTrainCApi() {}

 protected:
  // call GenerateRandomData to fill inputTensors
  int GenerateInputData() override;

  int ReadInputFile() override;

  int LoadStepInput(size_t step);

  int InitMSContext();

  void InitTrainCfg();

  char **TransStrVectorToCharArrays(const std::vector<std::string> &s);

  std::vector<std::string> TransCharArraysToStrVector(char **c, const size_t &num);

  int CreateAndRunNetwork(const std::string &filename, const std::string &bb_filename, bool is_train, int epochs,
                          bool check_accuracy = true) override;

  int CreateAndRunNetworkForInference(const std::string &filename, const MSContextHandle &context);

  int CreateAndRunNetworkForTrain(const std::string &filename, const std::string &bb_filename,
                                  const MSContextHandle &context, const MSTrainCfgHandle &train_cfg, int epochs);

  int InitDumpTensorDataCallbackParameter() override;

  int InitTimeProfilingCallbackParameter() override;

  int PrintResult(const std::vector<std::string> &title,
                  const std::map<std::string, std::pair<int, float>> &result) override;

  int PrintInputData();

  int MarkPerformance() override;

  int MarkAccuracy(bool enforce_accuracy = true) override;

  int CompareOutput() override;

  int SaveModels() override;

  MSModelHandle ms_model_;
  MSTensorHandleArray ms_inputs_for_api_;
  MSContextHandle context_ = nullptr;
  MSTrainCfgHandle train_cfg_ = nullptr;
  MSKernelCallBackC before_call_back_{nullptr};
  MSKernelCallBackC after_call_back_{nullptr};
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_TOOLS_BENCHMARK_NET_TRAIN_C_API_H
