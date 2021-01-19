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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_SCALE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_SCALE_H_

#include <vector>
#include "nnacl/scale.h"
#include "src/runtime/kernel/opencl/opencl_kernel.h"

namespace mindspore::kernel {

class ScaleOpenCLKernel : public OpenCLKernel {
 public:
  using OpenCLKernel::OpenCLKernel;
  ~ScaleOpenCLKernel() override;

  int CheckSpecs() override;
  int Prepare() override;
  int Run() override;
  int InitWeights() override;

 private:
  void Image2dGetWorkGroupSize();

  bool weight_vector_flag_{true};
  bool broadcast_flag_{false};
  bool broadcast_H_flag_{false};
  void *scale_ptr_{nullptr};
  void *offset_ptr_{nullptr};
  int axis_{0};

  std::vector<size_t> local_size_;
  std::vector<size_t> global_size_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_SCALE_H_
