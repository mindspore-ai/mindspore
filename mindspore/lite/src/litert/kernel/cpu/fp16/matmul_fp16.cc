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

#include "src/litert/kernel/cpu/fp16/matmul_fp16.h"
#include <algorithm>
#include "include/errorcode.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/cpu/fp32/matmul_fp32_base.h"

using mindspore::lite::kCHWDimNumber;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::kHWDimNumber;
using mindspore::lite::kNCHWDimNumber;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_MatMulFusion;

namespace mindspore::kernel {
int MatmulFP16CPUKernel::InitAShape() {
  auto a_shape = in_tensors_[0]->shape();
  MS_CHECK_TRUE_MSG(a_shape.size() >= DIMENSION_2D, RET_ERROR, "A-metric tensor's shape is invalid.");
  int batch = 1;
  for (size_t i = 0; i < a_shape.size() - DIMENSION_2D; ++i) {
    batch *= a_shape[i];
  }
  a_batch_ = batch;
  params_->row_ = params_->a_transpose_ ? a_shape[a_shape.size() - 1] : a_shape[a_shape.size() - 2];
  params_->deep_ = params_->a_transpose_ ? a_shape[a_shape.size() - 2] : a_shape[a_shape.size() - 1];
  params_->row_16_ = UP_ROUND(params_->row_, row_tile_);
  return RET_OK;
}

int MatmulFP16CPUKernel::InitBShape() {
  auto b_shape = in_tensors_[1]->shape();
  MS_CHECK_TRUE_MSG(b_shape.size() >= DIMENSION_2D, RET_ERROR, "B-metric tensor's shape is invalid.");
  int batch = 1;
  for (size_t i = 0; i < b_shape.size() - DIMENSION_2D; ++i) {
    batch *= b_shape[i];
  }
  b_batch_ = batch;
  params_->col_ = params_->b_transpose_ ? b_shape[b_shape.size() - 2] : b_shape[b_shape.size() - 1];
  params_->col_8_ = UP_ROUND(params_->col_, C8NUM);
  params_->deep_ = params_->b_transpose_ ? b_shape[b_shape.size() - 1] : b_shape[b_shape.size() - 2];
  return RET_OK;
}

int MatmulFP16CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
#ifdef ENABLE_ARM64
  row_tile_ = C1NUM;
  col_tile_ = C4NUM;
#else
  row_tile_ = C12NUM;
  col_tile_ = C8NUM;
#endif
  auto ret = MatmulBaseFP16CPUKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Do matmul prepare failed.";
  }
  return ret;
}

int MatmulFP16CPUKernel::ReSize() {
  InitAShape();
  InitBShape();
  auto ret = MatmulFp32BaseCPUKernel::InitBroadcastParams(
    in_tensors_[kInputIndex]->shape(), in_tensors_[kWeightIndex]->shape(), params_, &a_offset_, &b_offset_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitBroadcastParams failed.";
    return RET_ERROR;
  }

  return MatmulBaseFP16CPUKernel::ReSize();
}

int MatmulFP16CPUKernel::Run() {
  if (IsTrainable() && (IsTrain())) {
    is_repack_ = true;
  }
  auto ret = MatmulBaseFP16CPUKernel::Run();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MatmulFP16CPUKernel run failed";
  }
  is_repack_ = false;
  return ret;
}

int MatmulFP16CPUKernel::Eval() {
  LiteKernel::Eval();
  if (IsTrainable()) {
    is_repack_ = true;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_MatMulFusion, LiteKernelCreator<MatmulFP16CPUKernel>)
}  // namespace mindspore::kernel
