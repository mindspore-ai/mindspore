/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "experimental/kernel/convolution_1x1.h"
#include "nnacl/tensor_c.h"
#include "src/common/tensor_util.h"
#include "nnacl/kernel.h"

namespace mindspore::kernel::experimental {
Convolution1x1CPU::Convolution1x1CPU(OpParameter *parameter, std::vector<lite::Tensor *> in_tensors,
                                     std::vector<lite::Tensor *> out_tensors, const lite::Context *ctx)
    : LiteKernel(parameter, in_tensors, out_tensors, ctx) {
  for (size_t i = 0; i < in_tensors.size(); i++) {
    in[i] = reinterpret_cast<TensorC *>(malloc(sizeof(TensorC)));
    Tensor2TensorC(in_tensors[i], in[i]);
  }
  out[0] = reinterpret_cast<TensorC *>(malloc(sizeof(TensorC)));
  Tensor2TensorC(out_tensors[0], out[0]);
}

Convolution1x1CPU::~Convolution1x1CPU() {
  if (kernel == nullptr) {
    return;
  }
  kernel->release(kernel);

  free(kernel);
  kernel = nullptr;

  for (size_t i = 0; i < in_tensors().size(); i++) {
    free(in[i]);
    in[i] = nullptr;
  }
  free(out[0]);
  out[0] = nullptr;
}

int Convolution1x1CPU::Prepare() {
  kernel = CreateKernel(op_parameter_, in, in_tensors().size(), out, 1);
  if (kernel == nullptr) {
    return -1;
  }
  return kernel->prepare(kernel);
}

int Convolution1x1CPU::ReSize() {
  if (kernel == nullptr) {
    return -1;
  }
  return kernel->resize(kernel, in, in_tensors().size(), out, 1);
}

int Convolution1x1CPU::Run() {
  if (kernel == nullptr) {
    return -1;
  }

  kernel->in[0]->data_ = in_tensors().front()->data();
  kernel->out[0]->data_ = out_tensors().front()->data();

  kernel->compute(kernel);
  return 0;
}
}  // namespace mindspore::kernel::experimental
