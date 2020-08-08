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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_CONVOLUTION_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_CONVOLUTION_H_

#include <vector>
#include <string>
#include "src/ir/tensor.h"
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "schema/model_generated.h"
#include "src/runtime/opencl/opencl_runtime.h"
#include "src/runtime/kernel/arm/nnacl/conv_parameter.h"

namespace mindspore::kernel {

class ConvolutionOpenCLKernel : public OpenCLKernel {
 public:
  explicit ConvolutionOpenCLKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                                   const std::vector<lite::tensor::Tensor *> &outputs)
      : OpenCLKernel(parameter, inputs, outputs) {}
  ~ConvolutionOpenCLKernel() override{};

  int Init() override;
  int Run() override;
  int InitBuffer();
  int GetImageSize(size_t idx, std::vector<size_t> *img_size) override;

 private:
  float *packed_weight_ = nullptr;
  float *packed_bias_ = nullptr;
  cl::Kernel kernel_;

  std::string CodeGen();
  int GetGlobalLocal(std::vector<size_t> *global, std::vector<size_t> *local);
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_CONVOLUTION_H_
