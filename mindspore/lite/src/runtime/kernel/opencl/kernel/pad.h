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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_PAD_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_PAD_H_

#include <vector>
#include <string>
#include "src/tensor.h"
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "schema/model_generated.h"
#include "nnacl/pad_parameter.h"

namespace mindspore::kernel {

class PadOpenCLKernel : public OpenCLKernel {
 public:
  PadOpenCLKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                  const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : OpenCLKernel(parameter, inputs, outputs, ctx), param_(reinterpret_cast<PadParameter *>(op_parameter_)) {}
  ~PadOpenCLKernel() override = default;

  int CheckSpecs() override;

  int Prepare() override;
  void SetConstArgs() override;

  int Run() override;

 private:
  PadParameter *param_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_PAD_H_
