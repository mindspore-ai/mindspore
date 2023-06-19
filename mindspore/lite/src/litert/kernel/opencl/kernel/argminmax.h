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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_ARGMINMAX_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_ARGMINMAX_H_

#include <vector>
#include "src/litert/kernel/opencl/opencl_kernel.h"
#include "nnacl/arg_min_max_parameter.h"
#include "nnacl/kernel/arg_min_max.h"

namespace mindspore::kernel {
class ArgMinMaxOpenCLKernel : public OpenCLKernel {
 public:
  using OpenCLKernel::OpenCLKernel;

  ~ArgMinMaxOpenCLKernel() override = default;

  int Run() override;
  int Prepare() override;

  int CheckSpecs() override;
  int SetConstArgs() override;
  int SetGlobalLocal() override;
  int InitWeights() override;
  int Tune() override { return lite::RET_OK; }

 private:
  int SetGlobalLocalPre();

  void *buff_{nullptr};
  void *ids_{nullptr};
  GpuTensorInfo im_in_;
  GpuTensorInfo im_out_;
  cl_int4 src_size_;
  cl_int4 cus_size_;
  cl_int4 strides_;
  ArgMinMaxComputeParam compute_;
};
}  // namespace mindspore::kernel
#endif
