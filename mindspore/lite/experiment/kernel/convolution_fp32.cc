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
  TensorC *in[C4NUM];
  size_t insize = 0;
  for (; insize < in_tensors.size() && insize < C4NUM; insize++) {
    in[insize] = &in_tensors[insize]->TensorC();
  }

  TensorC *out[1];
  size_t outsize = 0;
  for (; outsize < out_tensors.size() && outsize < 1; outsize++) {
    out[outsize] = &out_tensors[outsize]->TensorC();
  }
  kernel = CreateKernel(parameter, in, insize, out, outsize);
}

ConvolutionCPUFp32::~ConvolutionCPUFp32() {
  kernel->release(kernel);
  free(kernel);
}

int ConvolutionCPUFp32::Prepare() {
  if (kernel == nullptr) {
    return -1;
  }
  kernel->init(kernel, &ctx_);  // init kernel, pack weight
  return 0;
}

int ConvolutionCPUFp32::PreProcess() {
  // allocate output tensor

  return 0;
}

int ConvolutionCPUFp32::Run() { return kernel->compute(kernel); }

int ConvolutionCPUFp32::PostProcess() { return kernel->compute(kernel); }
}  // namespace mindspore::kernel
