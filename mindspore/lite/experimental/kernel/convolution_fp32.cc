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
#include "nnacl/op_base.h"
#include "experimental/kernel/convolution_fp32.h"
#include "src/common/tensor_util.h"

namespace mindspore::kernel {
ConvolutionCPUFp32::ConvolutionCPUFp32(OpParameter *parameter, std::vector<lite::Tensor *> in_tensors,
                                       std::vector<lite::Tensor *> out_tensors, const lite::Context *ctx)
    : LiteKernel(parameter, in_tensors, out_tensors, ctx) {
  for (size_t i = 0; i < in_tensors.size(); i++) {
    in[i] = reinterpret_cast<TensorC *>(malloc(sizeof(TensorC)));
    Tensor2TensorC(in_tensors[i], in[i]);
  }
  out[0] = reinterpret_cast<TensorC *>(malloc(sizeof(TensorC)));
  Tensor2TensorC(out_tensors[0], out[0]);
}

ConvolutionCPUFp32::~ConvolutionCPUFp32() {
  if (kernel == nullptr) {
    return;
  }
  kernel->release(kernel);
  free(kernel);
  for (size_t i = 0; i < in_tensors_.size(); i++) {
    free(in[i]);
  }
  free(out[0]);
}

int ConvolutionCPUFp32::Prepare() {
  kernel = CreateKernel(op_parameter_, in, in_tensors_.size(), out, 1);
  if (kernel == nullptr) {
    return -1;
  }

  auto ret = kernel->resize(kernel, in, in_tensors_.size(), out, 1);
  if (ret != NNACL_OK) {
    return ret;
  }

  return kernel->prepare(kernel);
}

int ConvolutionCPUFp32::Run() { return kernel->compute(kernel); }
int ConvolutionCPUFp32::ReSize() { return kernel->resize(kernel, in, in_tensors_.size(), out, 1); }
}  // namespace mindspore::kernel
