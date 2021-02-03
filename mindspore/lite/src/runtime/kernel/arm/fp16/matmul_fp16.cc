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

#include "src/runtime/kernel/arm/fp16/matmul_fp16.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_MatMul;

namespace mindspore::kernel {
void MatmulFP16CPUKernel::InitAShape() {
  auto a_shape = in_tensors_[0]->shape();
  if (a_shape.empty()) {
    return;
  }
  int batch = 1;
  for (size_t i = 0; i < a_shape.size() - 2; ++i) {
    batch *= a_shape[i];
  }
  params_->batch = batch;
  params_->row_ = params_->a_transpose_ ? a_shape[a_shape.size() - 1] : a_shape[a_shape.size() - 2];
  params_->deep_ = params_->a_transpose_ ? a_shape[a_shape.size() - 2] : a_shape[a_shape.size() - 1];
  params_->row_16_ = UP_ROUND(params_->row_, C16NUM);
}

void MatmulFP16CPUKernel::InitBShape() {
  auto b_shape = in_tensors_[1]->shape();
  if (b_shape.empty()) {
    return;
  }
  int batch = 1;
  for (size_t i = 0; i < b_shape.size() - 2; ++i) {
    batch *= b_shape[i];
  }
  params_->batch = batch;
  params_->col_ = params_->b_transpose_ ? b_shape[b_shape.size() - 2] : b_shape[b_shape.size() - 1];
  params_->col_8_ = UP_ROUND(params_->col_, 8);
  params_->deep_ = params_->b_transpose_ ? b_shape[b_shape.size() - 1] : b_shape[b_shape.size() - 2];
}

int MatmulFP16CPUKernel::Init() {
  MatmulBaseFP16CPUKernel::InitParameter();

  if (params_->a_const_) {
    InitAShape();
  }
  if (params_->b_const_) {
    InitBShape();
  }

  auto ret = MatmulBaseFP16CPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int MatmulFP16CPUKernel::ReSize() {
  InitAShape();
  InitBShape();
  return MatmulBaseFP16CPUKernel::ReSize();
}

int MatmulFP16CPUKernel::Run() {
  auto ret = MatmulBaseFP16CPUKernel::Run();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MatmulFP16CPUKernel run failed";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_MatMul, LiteKernelCreator<MatmulFP16CPUKernel>)
}  // namespace mindspore::kernel
