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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_DEPTHWISE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_DEPTHWISE_H_

#include <vector>
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "nnacl/conv_parameter.h"

using mindspore::lite::opencl::MemType;

namespace mindspore::kernel {

class DepthwiseConv2dOpenCLKernel : public OpenCLKernel {
 public:
  DepthwiseConv2dOpenCLKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                              const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : OpenCLKernel(parameter, inputs, outputs, ctx) {
    bool is_adreno = ocl_runtime_->GetGpuInfo().type == lite::opencl::GpuType::ADRENO;
    filter_type_ = is_adreno ? MemType::IMG : MemType::BUF;
  }

  ~DepthwiseConv2dOpenCLKernel() override = default;

  int Run() override;
  int Prepare() override;

  int CheckSpecs() override;
  int InitWeights() override;
  int InitBias();
  void SetConstArgs() override;
  void SetGlobalLocal() override;

 private:
  void *packed_weight_{nullptr};
  void *bias_data_{nullptr};
  struct {
    int H{2};
    int W{2};
    int C{1};
  } block_size_;
  MemType filter_type_{MemType::BUF};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_DEPTHWISE_H_
