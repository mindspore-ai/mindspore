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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_STACK_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_STACK_H_

#include <vector>
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "nnacl/stack_parameter.h"

namespace mindspore::kernel {

class StackOpenCLKernel : public OpenCLKernel {
 public:
  explicit StackOpenCLKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                             const std::vector<lite::Tensor *> &outputs)
      : OpenCLKernel(parameter, inputs, outputs) {}

  ~StackOpenCLKernel() override{};

  int Init() override;

  int ReSize() override;

  int Run() override;

 private:
  int RunAxis0();

  int InferInTensorShapeTo4D(int *arg_cn);

  int InferOutTensorShapeTo4D(cl_int4 *output_shape);

  cl::Kernel kernel_;
  int axis_{0};
  size_t N_{1};
  size_t H_{1};
  size_t W_{1};
  size_t C_{1};
  size_t OH_{1};
  size_t OW_{1};
  size_t OC_{1};
};

}  // namespace mindspore::kernel
#endif
