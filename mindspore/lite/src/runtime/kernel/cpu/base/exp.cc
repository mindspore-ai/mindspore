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
#include "src/runtime/kernel/cpu/base/exp.h"
#include <cmath>
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/common/tensor_util.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ExpFusion;

namespace mindspore::kernel {
ExpCPUKernel::~ExpCPUKernel() {
  if (kernel == nullptr) {
    return;
  }
  kernel->release(kernel);

  free(kernel);
  kernel = nullptr;

  free(in[0]);
  in[0] = nullptr;
  free(out[0]);
  out[0] = nullptr;
}

int ExpCPUKernel::Prepare() {
  CHECK_NOT_EQUAL_RETURN(in_tensors_.size(), 1);
  CHECK_NOT_EQUAL_RETURN(out_tensors_.size(), 1);

  in[0] = reinterpret_cast<TensorC *>(malloc(sizeof(TensorC)));
  if (in[0] == nullptr) {
    return RET_ERROR;
  }
  Tensor2TensorC(in_tensors_[0], in[0]);
  out[0] = reinterpret_cast<TensorC *>(malloc(sizeof(TensorC)));
  if (out[0] == nullptr) {
    return RET_ERROR;
  }
  Tensor2TensorC(out_tensors_[0], out[0]);

  kernel = CreateKernel(op_parameter_, in, 1, out, 1);
  if (kernel == nullptr) {
    return RET_ERROR;
  }

  int ret = kernel->prepare(kernel);
  if (ret != RET_OK) {
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ExpCPUKernel::ReSize() {
  if (kernel == nullptr) {
    return RET_ERROR;
  }
  Tensor2TensorC(in_tensors_[0], in[0]);
  return kernel->resize(kernel, in, 1, out, 1);
}

int ExpCPUKernel::Run() {
  if (kernel == nullptr) {
    return RET_ERROR;
  }

  kernel->in[0]->data_ = in_tensors().front()->data();
  kernel->out[0]->data_ = out_tensors().front()->data();

  return kernel->compute(kernel);
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_ExpFusion, LiteKernelCreator<ExpCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ExpFusion, LiteKernelCreator<ExpCPUKernel>)
}  // namespace mindspore::kernel
