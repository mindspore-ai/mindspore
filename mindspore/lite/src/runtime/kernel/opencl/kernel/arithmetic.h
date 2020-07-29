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

#ifndef MINDSPORE_LITE_SRC_BACKEND_OPENCL_ARITHMETIC_H_
#define MINDSPORE_LITE_SRC_BACKEND_OPENCL_ARITHMETIC_H_

#include <vector>
#include "src/runtime/kernel/arm/fp32/arithmetic.h"
#include "src/runtime/opencl/opencl_runtime.h"

namespace mindspore::kernel {

class ArithmeticOpenCLKernel : public ArithmeticCPUKernel {
 public:
  explicit ArithmeticOpenCLKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                                  const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx)
      : ArithmeticCPUKernel(parameter, inputs, outputs, ctx) {}
  ~ArithmeticOpenCLKernel() override {};

  int Init() override;
  int Run() override;

 private:
  cl::Kernel kernel_;
  bool is_bias_add_{false};
  float weight_{1.f};
  float bias_{.0f};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_BACKEND_OPENCL_ARITHMETIC_H_

