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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_ACTIVATION_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_ACTIVATION_H_

#include <vector>
#include <string>

#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "nnacl/fp32/activation_fp32.h"

namespace mindspore::kernel {

class ActivationOpenCLKernel : public OpenCLKernel {
 public:
  ActivationOpenCLKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : OpenCLKernel(parameter, inputs, outputs, ctx),
        type_(reinterpret_cast<ActivationParameter *>(parameter)->type_),
        alpha_(reinterpret_cast<ActivationParameter *>(parameter)->alpha_) {}
  ~ActivationOpenCLKernel() override = default;

  int Run() override;
  int Prepare() override;
  int CheckSpecs() override;
  void SetConstArgs() override;
  void SetGlobalLocal() override;

 private:
  static std::string GetActTypeString(int act_type);
  int type_;
  float alpha_;
  GpuTensorInfo outShape;
};

}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_ACTIVATION_H_
