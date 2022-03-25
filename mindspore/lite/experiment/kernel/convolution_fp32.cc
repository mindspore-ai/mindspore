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
#include "experiment/kernel/convolution_fp32.h"

namespace mindspore::kernel {
ConvolutionCPUFp32::ConvolutionCPUFp32(OpParameter *parameter, std::vector<lite::Tensor *> in_tensors,
                                       std::vector<lite::Tensor *> out_tensors, const lite::Context *ctx)
    : InnerKernel(parameter, in_tensors, out_tensors, ctx) {
  in[0] = &in_tensors[0]->TensorC();
  in[1] = &in_tensors[1]->TensorC();
  out[0] = &out_tensors[0]->TensorC();
}

ConvolutionCPUFp32::~ConvolutionCPUFp32() {
  kernel->release(kernel);
  free(kernel);
}

int ConvolutionCPUFp32::Prepare() {
  kernel = CreateKernel(parameter, in, 2, out, 1);
  if (kernel == nullptr) {
    return -1;
  }

  if (kernel->resize(kernel, in, 2, out, 1) != NNACL_OK) {
    return ret;
  }

  return kernel->prepare(kernel, NULL);
}

int ConvolutionCPUFp32::Run() { return kernel->compute(kernel); }
int ConvolutionCPUFp32::Resize() { return kernel->resize(kernel, in, 2, out, 1); }
}  // namespace mindspore::kernel
