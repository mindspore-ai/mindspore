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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_REDUCE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_REDUCE_H_

#include <vector>
#include <string>
#include "src/lite_kernel.h"
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "nnacl/reduce_parameter.h"

namespace mindspore::kernel {
class ReduceOpenCLKernel : public OpenCLKernel {
 public:
  using OpenCLKernel::OpenCLKernel;
  ~ReduceOpenCLKernel() override = default;

  int Run() override;
  int Prepare() override;
  int CheckSpecs() override;
  void SetConstArgs() override;
  void SetGlobalLocal() override;
  int Tune() override;

 private:
  int SetAxes();
  cl_float4 GenC4Mask();
  static std::string GetReduceTypeStr(int type);
  GpuTensorInfo inShape;
  bool use_local_{false};
  bool wc_reduce_{false};
  bool hw_reduce_{false};
  bool c_reduce_{false};
  bool reduce_axes_[4]{false};
  static const size_t LOCAL_CACHE_THREAD{16};
  int axes_[MAX_SHAPE_SIZE];
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_REDUCE_H_
