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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_CONCAT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_CONCAT_H_

#include <vector>
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "nnacl/concat_parameter.h"

namespace mindspore::kernel {

class ConcatOpenCLKernel : public OpenCLKernel {
 public:
  using OpenCLKernel::OpenCLKernel;

  ~ConcatOpenCLKernel() override = default;

  int Prepare() override;

  int CheckSpecs() override;
  void SetConstArgs() override;
  void SetGlobalLocal() override;
  int Run() override;

 private:
  std::vector<size_t> local;
  uint32_t OH = {1};
  uint32_t OW = {1};
  uint32_t OC = {1};
  std::vector<size_t> global;
  bool Align_{true};
  std::vector<void *> weight_ptrs_;
  cl_int stride_w{1};
  cl_int4 in_shape_{};
  cl_int4 out_shape_{};
  int axis_{0};

 private:
  int RunAxis0();
  int ConvertWeightToTensor();
};

}  // namespace mindspore::kernel
#endif
