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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_SPARSE_TO_DENSE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_SPARSE_TO_DENSE_H_

#include <vector>
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "mindspore/lite/nnacl/fp32/sparse_to_dense_fp32.h"

namespace mindspore::kernel {

class SparseToDenseOpenCLKernel : public OpenCLKernel {
 public:
  using OpenCLKernel::OpenCLKernel;

  ~SparseToDenseOpenCLKernel() override = default;

  int Prepare() override;
  int Run() override;
  int InitWeights() override;
  void SetConstArgs() override;
  void SetGlobalLocal() override;
  int CheckSpecs() override;

 private:
  int InferShapeTo4D();
  int InitOutputToDefault();

 private:
  bool enable_fp16_{false};
  float default_{0.0f};
  float weight_scalar_{0.f};
  void *weight_vector_{nullptr};
  int input_dim_{1};
  int inshapeindex1_dim{1};
  cl_int stride_w{1};
  std::vector<int32_t> output_shape_;

  cl_int n_{1};
  cl_int h_{1};
  cl_int w_{1};
  cl_int c_{1};

  cl_int out_n_{1};
  cl_int out_h_{1};
  cl_int out_w_{1};
  cl_int out_c_{1};
};
}  // namespace mindspore::kernel
#endif
