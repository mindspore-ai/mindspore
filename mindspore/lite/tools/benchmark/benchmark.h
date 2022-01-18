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

#ifndef MINNIE_BENCHMARK_BENCHMARK_H_
#define MINNIE_BENCHMARK_BENCHMARK_H_

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
#include "include/lite_session.h"

namespace mindspore::lite {

class MS_API Benchmark : public BenchmarkBase {
 public:
  explicit Benchmark(BenchmarkFlags *flags) : BenchmarkBase(flags) {}

  virtual ~Benchmark();

  int RunBenchmark() override;

 protected:
  // call GenerateRandomData to fill inputTensors
  int GenerateInputData() override;

  int ReadInputFile() override;

  int GetDataTypeByTensorName(const std::string &tensor_name) override;

  void InitContext(const std::shared_ptr<Context> &context);

  int CompareOutput() override;

  int CompareDataGetTotalBiasAndSize(const std::string &name, tensor::MSTensor *tensor, float *total_bias,
                                     int *total_size);

  int InitTimeProfilingCallbackParameter() override;

  int InitPerfProfilingCallbackParameter() override;

  int InitDumpTensorDataCallbackParameter() override;

  int InitPrintTensorDataCallbackParameter() override;

  int PrintInputData();

  int MarkPerformance();

  int MarkAccuracy();

  int CheckInputNames();

 private:
  session::LiteSession *session_{nullptr};
  std::vector<mindspore::tensor::MSTensor *> ms_inputs_;
  std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> ms_outputs_;

  KernelCallBack before_call_back_ = nullptr;
  KernelCallBack after_call_back_ = nullptr;
};

}  // namespace mindspore::lite
#endif  // MINNIE_BENCHMARK_BENCHMARK_H_
