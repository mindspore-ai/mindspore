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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_LAYER_NORM_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_LAYER_NORM_H_

#include <vector>
#include "src/runtime/kernel/opencl/opencl_kernel.h"

namespace mindspore::kernel {

class LayerNormOpenCLKernel : public OpenCLKernel {
 public:
  using OpenCLKernel::OpenCLKernel;

  ~LayerNormOpenCLKernel() override = default;

  int Run() override;
  int Prepare() override;

  int CheckSpecs() override;
  void SetConstArgs() override;
  void SetGlobalLocal() override;

 private:
  int Initweight();
  void GetMeanVar();

 private:
  cl::Kernel kernel_mean_var_;
  cl::NDRange global_mean_var_, local_mean_var_;
  bool use_fp16_enable_{false};
  void *gamma_{nullptr};
  void *mean_{nullptr};
  void *var_{nullptr};
  void *beta_{nullptr};
  cl_int4 in_shape_{};
  int32_t normalized_axis_{3};  // default is C
  int normalized_shape_size_{1};
  float epsilon_{0.0f};
  cl::Kernel kernel_;
};

}  // namespace mindspore::kernel
#endif
