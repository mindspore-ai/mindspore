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

#ifndef MINDSPORE_LITE_TOOLS_BENCHMARK_TRAIN_NET_TRAIN_H_
#define MINDSPORE_LITE_TOOLS_BENCHMARK_TRAIN_NET_TRAIN_H_

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

#ifdef ENABLE_FP16
#include <arm_neon.h>
#endif
#include "tools/common/flag_parser.h"
#include "src/common/file_utils.h"
#include "src/common/utils.h"
#include "tools/benchmark_train/net_train_base.h"

namespace mindspore::lite {
extern const std::unordered_map<int, std::string> kTypeIdMap;
extern const std::unordered_map<mindspore::Format, std::string> kTensorFormatMap;

class MS_API NetTrain : public NetTrainBase {
 public:
  explicit NetTrain(NetTrainFlags *flags) : NetTrainBase(flags) {}
  virtual ~NetTrain() {}

 protected:
  // call GenerateRandomData to fill inputTensors
  int GenerateInputData() override;

  int ReadInputFile() override;

  int LoadStepInput(size_t step);

  void InitMSContext(const std::shared_ptr<Context> &context);

  void InitTrainCfg(const std::shared_ptr<TrainCfg> &train_cfg);

  int CreateAndRunNetwork(const std::string &filename, const std::string &bb_filename, bool is_train, int epochs,
                          bool check_accuracy = true) override;

  int CreateAndRunNetworkForInference(const std::string &filename, const std::shared_ptr<mindspore::Context> &context);

  int CreateAndRunNetworkForTrain(const std::string &filename, const std::string &bb_filename,
                                  const std::shared_ptr<mindspore::Context> &context,
                                  const std::shared_ptr<TrainCfg> &train_cfg, int epochs);

  int InitDumpTensorDataCallbackParameter() override;

  int InitTimeProfilingCallbackParameter() override;

  int PrintResult(const std::vector<std::string> &title,
                  const std::map<std::string, std::pair<int, float>> &result) override;

  template <typename T>
  void PrintInputData(mindspore::MSTensor *input) {
    MS_ASSERT(input != nullptr);
    static int i = 0;
    auto inData = reinterpret_cast<T *>(input->MutableData());
    size_t tensorSize = input->ElementNum();
    size_t len = (tensorSize < 20) ? tensorSize : 20;
    std::cout << "InData" << i++ << ": ";
    for (size_t j = 0; j < len; j++) {
      std::cout << inData[j] << " ";
    }
    std::cout << std::endl;
  }

  int MarkPerformance() override;
  int MarkAccuracy(bool enforce_accuracy = true) override;
  int CompareOutput() override;
  int SaveModels() override;

  // callback parameters
  uint64_t op_begin_ = 0;
  int op_call_times_total_ = 0;
  float op_cost_total_ = 0.0f;
  std::map<std::string, std::pair<int, float>> op_times_by_type_;
  std::map<std::string, std::pair<int, float>> op_times_by_name_;

  std::shared_ptr<mindspore::Model> ms_model_ = nullptr;
  std::vector<mindspore::MSTensor> ms_inputs_for_api_;

  mindspore::MSKernelCallBack before_call_back_{nullptr};
  mindspore::MSKernelCallBack after_call_back_{nullptr};
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_TOOLS_BENCHMARK_TRAIN_NET_TRAIN_H_
