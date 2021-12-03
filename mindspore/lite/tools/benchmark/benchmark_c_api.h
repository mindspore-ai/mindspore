/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_TOOLS_BENCHMARK_BENCHMARK_C_API_H_
#define MINDSPORE_LITE_TOOLS_BENCHMARK_BENCHMARK_C_API_H_

#include <vector>
#include <string>
#include "tools/benchmark/benchmark_base.h"
#include "include/c_api/model_c.h"
#include "include/c_api/context_c.h"

#ifdef __cplusplus
extern "C" {
#endif
bool TimeBeforeCallback(const MSTensorHandleArray inputs, const MSTensorHandleArray outputs,
                        const MSCallBackParamC kernel_Info);
bool TimeAfterCallback(const MSTensorHandleArray inputs, const MSTensorHandleArray outputs,
                       const MSCallBackParamC kernel_Info);
#ifdef __cplusplus
}
#endif

using mindspore::lite::BenchmarkBase;
using mindspore::lite::BenchmarkFlags;

namespace mindspore::tools {
class MS_API BenchmarkCApi : public BenchmarkBase {
 public:
  explicit BenchmarkCApi(BenchmarkFlags *flags) : BenchmarkBase(flags) {}

  virtual ~BenchmarkCApi() { MSModelDestroy(&model_); }

  int RunBenchmark() override;

  int LoadInput() override;

 protected:
  int CompareDataGetTotalBiasAndSize(const std::string &name, MSTensorHandle tensor, float *total_bias,
                                     int *total_size);
  int InitContext();
  int GenerateInputData() override;
  int ReadInputFile() override;
  int GetDataTypeByTensorName(const std::string &tensor_name) override;
  int CompareOutput() override;

  int InitTimeProfilingCallbackParameter() override;
  int InitPerfProfilingCallbackParameter() override;
  int InitDumpTensorDataCallbackParameter() override;
  int InitPrintTensorDataCallbackParameter() override;

  int PrintInputData();
  int MarkPerformance();
  int MarkAccuracy();

 private:
  MSModelHandle model_ = nullptr;
  MSContextHandle context_ = nullptr;
  MSTensorHandleArray inputs_;
  MSTensorHandleArray outputs_;

  MSKernelCallBackC before_call_back_ = nullptr;
  MSKernelCallBackC after_call_back_ = nullptr;
};
}  // namespace mindspore::tools
#endif  // MINDSPORE_LITE_TOOLS_BENCHMARK_BENCHMARK_C_API_H_
