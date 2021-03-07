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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_SOFTMAX_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_SOFTMAX_H_

#include <vector>

#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "nnacl/fp32/softmax_fp32.h"

namespace mindspore::kernel {

class SoftmaxOpenCLKernel : public OpenCLKernel {
 public:
  using OpenCLKernel::OpenCLKernel;
  ~SoftmaxOpenCLKernel() override = default;

  int Run() override;
  int Prepare() override;
  int CheckSpecs() override;
  void SetConstArgs() override;
  void SetGlobalLocal() override;
  int Tune() override;

 private:
  std::vector<float> GetMaskForLastChannel(int channels);

  bool onexone_flag_{false};
  std::vector<size_t> local_size_;
  std::vector<size_t> global_size_;
  int axis_{0};
  GpuTensorInfo out_shape_;
};

}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_SOFTMAX_H_
