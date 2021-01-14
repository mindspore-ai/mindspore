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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_SPLIT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_SPLIT_H_

#include <vector>
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "nnacl/split_parameter.h"

namespace mindspore::kernel {

class SplitOpenCLKernel : public OpenCLKernel {
 public:
  using OpenCLKernel::OpenCLKernel;

  ~SplitOpenCLKernel() override = default;

  int Prepare() override;

  int CheckSpecs() override;
  void SetConstArgs() override;
  void SetGlobalLocal() override;
  int Run() override;

 private:
  void AlignSplitSizes(SplitParameter *param, const std::vector<int> &in_shape);
  int RunAxis0();

 private:
  cl_int4 in_shape_{};
  cl_int4 out_shape_ = {};
  bool Align_{true};
  bool enable_fp16_{false};
  size_t num_split_ = 1;
  int *split_sizes_{nullptr};
  int split_dim_ = 0;
  cl_int stride_w{1};
  uint32_t OH = {1};
  uint32_t OW = {1};
  uint32_t OC = {1};
};

}  // namespace mindspore::kernel
#endif
