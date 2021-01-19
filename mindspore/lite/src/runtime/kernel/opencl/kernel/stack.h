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
  using OpenCLKernel::OpenCLKernel;

  ~StackOpenCLKernel() override{};
  int Prepare() override;
  int CheckSpecs() override;
  void SetConstArgs() override;
  void SetGlobalLocal() override;

  int ReSize() override;

  int Run() override;

 private:
  int RunAxis0();

  cl::Kernel kernel_;
  int axis_{0};
  size_t OH_{1};
  size_t OW_{1};
  size_t OC_{1};
  bool buffer_button_{false};
  bool enable_fp16_{false};
  cl_int stride_w_in{1};
  cl_int stride_w_out{1};
  cl_int4 in_shape_ = {};
  cl_int4 out_shape_ = {};
};

}  // namespace mindspore::kernel
#endif
