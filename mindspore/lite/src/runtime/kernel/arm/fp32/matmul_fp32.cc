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

#include <algorithm>
#include "src/runtime/kernel/arm/fp32/matmul_fp32.h"
#include "include/errorcode.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_MatMul;

namespace mindspore::kernel {
namespace {
constexpr size_t HWDIMS = 2;
constexpr size_t CHWDIMS = 3;
constexpr size_t NCHWDIMS = 4;
}  // namespace

void MatmulCPUKernel::InitShapeA() {
  auto a_shape = in_tensors_.at(0)->shape();
  int batch = 1;
  for (size_t i = 0; i < a_shape.size() - 2; ++i) {
    batch *= a_shape[i];
  }
  a_batch_ = batch;
  params_->row_ = params_->a_transpose_ ? a_shape[a_shape.size() - 1] : a_shape[a_shape.size() - 2];
  params_->deep_ = params_->a_transpose_ ? a_shape[a_shape.size() - 2] : a_shape[a_shape.size() - 1];
}

void MatmulCPUKernel::InitShapeB() {
  auto b_shape = in_tensors_.at(1)->shape();
  int batch = 1;
  for (size_t i = 0; i < b_shape.size() - 2; ++i) {
    batch *= b_shape[i];
  }
  b_batch_ = batch;
  params_->col_ = params_->b_transpose_ ? b_shape[b_shape.size() - 2] : b_shape[b_shape.size() - 1];
  params_->deep_ = params_->b_transpose_ ? b_shape[b_shape.size() - 1] : b_shape[b_shape.size() - 2];
}

int MatmulCPUKernel::Init() {
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

int MatmulCPUKernel::InitBroadcastParams() {
  auto a_shape = in_tensors_[kInputIndex]->shape();
  if (a_shape.size() < NCHWDIMS) {
    size_t add_nums = NCHWDIMS - a_shape.size();
    for (size_t i = 0; i < add_nums; ++i) {
      a_shape.insert(a_shape.begin(), 1);
    }
  }
  auto b_shape = in_tensors_[kWeightIndex]->shape();
  if (b_shape.size() < NCHWDIMS) {
    size_t add_nums = NCHWDIMS - b_shape.size();
    for (size_t i = 0; i < add_nums; ++i) {
      b_shape.insert(b_shape.begin(), 1);
    }
  }

  for (int i = a_shape.size() - CHWDIMS; i >= 0; --i) {
    if (static_cast<int>(a_shape.size() - CHWDIMS) == i) {
      batch_sizes_[i] = std::max(a_shape[i], b_shape[i]);
      a_batch_sizes_[i] = a_shape[i];
      b_batch_sizes_[i] = b_shape[i];
    } else {
      batch_sizes_[i] = batch_sizes_[i + 1] * std::max(a_shape[i], b_shape[i]);
      a_batch_sizes_[i] = a_batch_sizes_[i + 1] * a_shape[i];
      b_batch_sizes_[i] = b_batch_sizes_[i + 1] * b_shape[i];
    }
  }

  int out_batch = 1;
  for (size_t i = 0; i < a_shape.size() - HWDIMS; ++i) {
    out_batch *= MSMAX(a_shape[i], b_shape[i]);
    if (a_shape[i] < b_shape[i] && a_shape[i] == 1) {
      a_broadcast_ = true;
    } else if (a_shape[i] > b_shape[i] && b_shape[i] == 1) {
      b_broadcast_ = true;
    } else if (a_shape[i] != b_shape[i]) {
      MS_LOG(ERROR) << "matmul don't support broadcast for dimension " << a_shape << " and " << b_shape;
      return RET_ERROR;
    }
  }
  params_->batch = out_batch;
  return RET_OK;
}

int MatmulCPUKernel::ReSize() {
  InitShapeA();
  InitShapeB();
  InitBroadcastParams();

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
