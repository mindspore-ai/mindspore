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

#ifndef MINDSPORE_LITE_SRC_BACKEND_OPENCL_MATMUL_H_
#define MINDSPORE_LITE_SRC_BACKEND_OPENCL_MATMUL_H_

#include <vector>

#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "nnacl/conv_parameter.h"
#include "src/runtime/opencl/opencl_runtime.h"

namespace mindspore::kernel {

class MatMulOpenCLKernel : public OpenCLKernel {
 public:
  explicit MatMulOpenCLKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                              const std::vector<lite::tensor::Tensor *> &outputs, bool hasBias)
      : OpenCLKernel(parameter, inputs, outputs) {
    hasBias_ = hasBias;
  }
  ~MatMulOpenCLKernel() override{};

  int Init() override;
  int ReSize() override;
  int Run() override;
  void PadWeight();
  int GetImageSize(size_t idx, std::vector<size_t> *img_size) override;

 private:
  cl::Kernel kernel_;
  void *padWeight_;
  void *bias_;
  bool hasBias_{false};
  bool enable_fp16_{false};
  cl_int2 sizeCI;
  cl_int2 sizeCO;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_BACKEND_OPENCL_MATMUL_H_
