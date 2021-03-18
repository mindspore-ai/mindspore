/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_WINOGRAD_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_WINOGRAD_H_

#include <string>
#include <vector>
#include "src/runtime/kernel/opencl/kernel/conv2d.h"

namespace mindspore::kernel {

class WinogradOpenCLKernel : public Conv2DOpenCLKernel {
 public:
  WinogradOpenCLKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                       const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : Conv2DOpenCLKernel(parameter, inputs, outputs, ctx) {
    use_winograd_ = true;
  }

  ~WinogradOpenCLKernel() override = default;

  void SetConstArgs() override;
  void SetGlobalLocal() override;
  int Run() override;

  std::vector<BaseTuningParameter> GenerateTuningParam() override { return {}; }
  int Tune() override { return RET_OK; }
  double GetProfilingTimeMs() override;

 private:
  void BuildKernel() override;
  void InitFilter() override;
  void AllocateMemory();

  cl::Kernel kernel_4x4to36_;
  cl::Kernel kernel_36to4x4_;
  cl::Event kernel2_event_;
  cl::NDRange global_4x4to36_, local_4x4to36_;
  cl::NDRange global_36to4x4_, local_36to4x4_;
  cl::Event kernel3_event_;
  void *winograd_mem0_{nullptr};
  void *winograd_mem1_{nullptr};
};

}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_WINOGRAD_H_
