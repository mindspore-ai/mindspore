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

#include "src/runtime/kernel/arm/fp32/matmul_fp32.h"
#include "include/errorcode.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_MatMul;

namespace mindspore::kernel {
void MatmulCPUKernel::InitShapeA() {
  auto a_shape = in_tensors_[kInputIndex]->shape();
  int batch = 1;
  MS_CHECK_TRUE_RET_VOID(a_shape.size() >= 2);
  for (size_t i = 0; i < a_shape.size() - 2; ++i) {
    batch *= a_shape[i];
  }
  params_->batch = batch;
  params_->row_ = params_->a_transpose_ ? a_shape[a_shape.size() - 1] : a_shape[a_shape.size() - 2];
  params_->deep_ = params_->a_transpose_ ? a_shape[a_shape.size() - 2] : a_shape[a_shape.size() - 1];
}

void MatmulCPUKernel::InitShapeB() {
  auto b_shape = in_tensors_[kWeightIndex]->shape();
  int batch = 1;
  MS_CHECK_TRUE_RET_VOID(b_shape.size() >= 2);
  for (size_t i = 0; i < b_shape.size() - 2; ++i) {
    batch *= b_shape[i];
  }
  params_->batch = batch;
  params_->col_ = params_->b_transpose_ ? b_shape[b_shape.size() - 2] : b_shape[b_shape.size() - 1];
  params_->deep_ = params_->b_transpose_ ? b_shape[b_shape.size() - 1] : b_shape[b_shape.size() - 2];
}

int MatmulCPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  MatmulFp32BaseCPUKernel::InitParameter();

  if (params_->a_const_ == true) {
    InitShapeA();
  }

  if (params_->b_const_ == true) {
    InitShapeB();
  }

  auto ret = MatmulFp32BaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int MatmulCPUKernel::ReSize() {
  InitShapeA();
  InitShapeB();

  return MatmulFp32BaseCPUKernel::ReSize();
}

int MatmulCPUKernel::Run() {
  auto ret = MatmulFp32BaseCPUKernel::Run();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MatmulFp32BaseCPUKernel failed!";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_MatMul, LiteKernelCreator<MatmulCPUKernel>)
}  // namespace mindspore::kernel
