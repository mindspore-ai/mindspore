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

#ifndef MINDSPORE_LITE_SRC_BACKEND_OPENCL_DEPTHWISE_H_
#define MINDSPORE_LITE_SRC_BACKEND_OPENCL_DEPTHWISE_H_

#include <vector>

#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/opclib/conv_parameter.h"
#include "src/runtime/opencl/opencl_runtime.h"


namespace mindspore::kernel {

class DepthwiseConv2dOpenCLKernel : public LiteKernel {
 public:
  explicit DepthwiseConv2dOpenCLKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                                       const std::vector<lite::tensor::Tensor *> &outputs)
      : LiteKernel(parameter, inputs, outputs),
      packed_weight_(nullptr), bias_data_(nullptr), kernel_(nullptr) {}
  ~DepthwiseConv2dOpenCLKernel() override {};

  int Init() override;
  int ReSize() override;
  int Run() override;
  int InitBuffer();

 private:
  FLOAT_t *packed_weight_;
  FLOAT_t *bias_data_;
  cl::Kernel kernel_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_BACKEND_OPENCL_DEPTHWISE_H_

