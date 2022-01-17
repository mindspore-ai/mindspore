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

#ifndef MINDSPORE_BENCHMARK_BENCHMARK_UNIFIED_API_H_
#define MINDSPORE_BENCHMARK_BENCHMARK_UNIFIED_API_H_

#include <signal.h>
#include <random>
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
#include <nlohmann/json.hpp>
#include "tools/benchmark/benchmark_base.h"
#include "include/model.h"
#include "tools/common/flag_parser.h"
#include "src/common/file_utils.h"
#include "src/common/utils.h"
#include "include/api/types.h"
#include "include/api/model.h"

namespace mindspore::lite {

class MS_API BenchmarkUnifiedApi : public BenchmarkBase {
 public:
  explicit BenchmarkUnifiedApi(BenchmarkFlags *flags) : BenchmarkBase(flags) {}

  virtual ~BenchmarkUnifiedApi();

  int RunBenchmark() override;

 protected:
  int CompareDataGetTotalBiasAndSize(const std::string &name, mindspore::MSTensor *tensor, float *total_bias,
                                     int *total_size);
  void InitContext(const std::shared_ptr<mindspore::Context> &context);

  // call GenerateRandomData to fill inputTensors
  int GenerateInputData() override;

  int ReadInputFile() override;

  void InitMSContext(const std::shared_ptr<Context> &context);

  int GetDataTypeByTensorName(const std::string &tensor_name) override;

  int CompareOutput() override;

  int InitTimeProfilingCallbackParameter() override;

  int InitPerfProfilingCallbackParameter() override;

  int InitDumpTensorDataCallbackParameter() override;

  int InitPrintTensorDataCallbackParameter() override;

  int PrintInputData();

  template <typename T>
  std::vector<int64_t> ConverterToInt64Vector(const std::vector<T> &srcDims) {
    std::vector<int64_t> dims;
    for (auto shape : srcDims) {
      dims.push_back(static_cast<int64_t>(shape));
    }
    return dims;
  }

  int MarkPerformance();

  int MarkAccuracy();

 private:
  mindspore::Model ms_model_;
  std::vector<mindspore::MSTensor> ms_inputs_for_api_;

  MSKernelCallBack ms_before_call_back_ = nullptr;
  MSKernelCallBack ms_after_call_back_ = nullptr;
};

}  // namespace mindspore::lite
#endif  // MINNIE_BENCHMARK_BENCHMARK_H_
