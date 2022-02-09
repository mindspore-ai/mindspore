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
#include <algorithm>
#include "include/errorcode.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "src/kernel_registry.h"

using mindspore::lite::kCHWDimNumber;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::kHWDimNumber;
using mindspore::lite::kNCHWDimNumber;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_MatMulFusion;

namespace mindspore::kernel {
void MatmulCPUKernel::InitShapeA() {
  auto a_shape = in_tensors_[kInputIndex]->shape();
  int batch = 1;
  MS_CHECK_TRUE_RET_VOID(a_shape.size() >= 2);
  for (size_t i = 0; i < a_shape.size() - 2; ++i) {
    batch *= a_shape[i];
  }
  a_batch_ = batch;
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
  b_batch_ = batch;
  params_->col_ = params_->b_transpose_ ? b_shape[b_shape.size() - 2] : b_shape[b_shape.size() - 1];
  params_->deep_ = params_->b_transpose_ ? b_shape[b_shape.size() - 1] : b_shape[b_shape.size() - 2];
}

int MatmulCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  MatmulFp32BaseCPUKernel::InitParameter();

  if (params_->a_const_) {
    InitShapeA();
  }

  if (params_->b_const_) {
    InitShapeB();
  }

  auto ret = MatmulFp32BaseCPUKernel::Prepare();
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
  if (a_shape.size() < kNCHWDimNumber) {
    size_t add_nums = kNCHWDimNumber - a_shape.size();
    for (size_t i = 0; i < add_nums; ++i) {
      a_shape.insert(a_shape.begin(), 1);
    }
  }
  auto b_shape = in_tensors_[kWeightIndex]->shape();
  if (b_shape.size() < kNCHWDimNumber) {
    size_t add_nums = kNCHWDimNumber - b_shape.size();
    for (size_t i = 0; i < add_nums; ++i) {
      b_shape.insert(b_shape.begin(), 1);
    }
  }

  int batch_sizes[MAX_SHAPE_SIZE] = {0};
  int a_batch_sizes[MAX_SHAPE_SIZE] = {0};
  int b_batch_sizes[MAX_SHAPE_SIZE] = {0};
  for (int i = a_shape.size() - kCHWDimNumber; i >= 0; --i) {
    if (static_cast<int>(a_shape.size() - kCHWDimNumber) == i) {
      batch_sizes[i] = std::max(a_shape[i], b_shape[i]);
      a_batch_sizes[i] = a_shape[i];
      b_batch_sizes[i] = b_shape[i];
    } else {
      batch_sizes[i] = batch_sizes[i + 1] * std::max(a_shape[i], b_shape[i]);
      a_batch_sizes[i] = a_batch_sizes[i + 1] * a_shape[i];
      b_batch_sizes[i] = b_batch_sizes[i + 1] * b_shape[i];
    }
  }

  int out_batch = 1;
  for (size_t i = 0; i < a_shape.size() - kHWDimNumber; ++i) {
    int max_v = MSMAX(a_shape[i], b_shape[i]);
    int min_v = MSMIN(a_shape[i], b_shape[i]) > 0 ? MSMIN(a_shape[i], b_shape[i]) : 1;
    out_batch *= max_v;
    if (max_v != min_v && max_v % min_v != 0) {
      MS_LOG(ERROR) << "matmul don't support broadcast for dimension " << a_shape << " and " << b_shape;
      return RET_ERROR;
    }
  }
  params_->batch = out_batch;

  a_offset_.resize(params_->batch, 0);
  b_offset_.resize(params_->batch, 0);
  for (int i = 0; i < params_->batch; ++i) {
    int delta = i;
    int a_offset = 0;
    int b_offset = 0;
    for (size_t j = 0; j < a_shape.size() - kHWDimNumber; ++j) {
      if (j > 0) {
        delta = delta % batch_sizes[j];
      }
      if (j < (a_shape.size() - kCHWDimNumber)) {
        a_offset += (delta / batch_sizes[j + 1] * a_shape[j] / std::max(a_shape[j], b_shape[j])) * a_batch_sizes[j + 1];
        b_offset += (delta / batch_sizes[j + 1] * b_shape[j] / std::max(a_shape[j], b_shape[j])) * b_batch_sizes[j + 1];
      } else {
        a_offset += (delta * a_shape[j] / std::max(a_shape[j], b_shape[j]));
        b_offset += (delta * b_shape[j] / std::max(a_shape[j], b_shape[j]));
      }
    }
    a_offset_[i] = a_offset;
    b_offset_[i] = b_offset;
  }

  return RET_OK;
}

int MatmulCPUKernel::ReSize() {
  InitShapeA();
  InitShapeB();
  (void)InitBroadcastParams();

  return MatmulFp32BaseCPUKernel::ReSize();
}

int MatmulCPUKernel::Run() {
  CHECK_NULL_RETURN(in_tensors_.at(FIRST_INPUT));
  CHECK_NULL_RETURN(in_tensors_.at(SECOND_INPUT));
  CHECK_NULL_RETURN(out_tensors_.at(FIRST_INPUT));
  auto ret = MatmulFp32BaseCPUKernel::Run();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MatmulFp32BaseCPUKernel failed!";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_MatMulFusion, LiteKernelCreator<MatmulCPUKernel>)
}  // namespace mindspore::kernel
