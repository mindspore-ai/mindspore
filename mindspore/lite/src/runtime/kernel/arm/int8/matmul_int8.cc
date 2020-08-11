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

#include "src/runtime/kernel/arm/int8/matmul_int8.h"
#include "src/runtime/kernel/arm/nnacl/int8/matmul_int8.h"
#include "src/runtime/kernel/arm/nnacl/common_func.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
MatmulInt8CPUKernel::~MatmulInt8CPUKernel() {
  ctx_->allocator->Free(a_c8_ptr_);
  ctx_->allocator->Free(b_r8_ptr_);
  ctx_->allocator->Free(c_r8x8_ptr_);
}

int MatmulInt8CPUKernel::Init() {
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    set_need_reinit();
    return RET_OK;
  }
  int batch = 1;
  auto x_shape = in_tensors_[0]->shape();
  auto o_shape = out_tensors_[0]->shape();
  for (int i = 0; i < x_shape.size() - 2; ++i) {
    batch *= x_shape[i];
  }
  params_->batch = batch;
  params_->row_ = o_shape[o_shape.size() - 2];
  params_->col_ = o_shape[o_shape.size() - 1];
  params_->deep_ = params_->a_transpose_ ? x_shape[x_shape.size() - 2] : x_shape[x_shape.size() - 1];
  params_->row_8_ = UP_ROUND(params_->row_, 8);
  params_->col_8_ = UP_ROUND(params_->col_, 8);
  thread_count_ = MSMIN(thread_count_, UP_DIV(params_->col_8_, 8));
  thread_stride_ = UP_DIV(UP_DIV(params_->col_8_, 8), thread_count_);

  a_c8_ptr_ = reinterpret_cast<int8_t *>(ctx_->allocator->Malloc(params_->row_8_ * params_->deep_ * sizeof(int8_t)));
  if (!a_c8_ptr_) {
    return RET_MEMORY_FAILED;
  }
  memset(a_c8_ptr_, 0, params_->row_8_ * params_->deep_ * sizeof(int8_t));
  b_r8_ptr_ = reinterpret_cast<int8_t *>(ctx_->allocator->Malloc(params_->col_8_ * params_->deep_ * sizeof(int8_t)));
  if (!b_r8_ptr_) {
    return RET_MEMORY_FAILED;
  }
  memset(b_r8_ptr_, 0, params_->col_8_ * params_->deep_ * sizeof(int8_t));
  c_r8x8_ptr_ = reinterpret_cast<int *>(ctx_->allocator->Malloc(params_->row_8_ * params_->col_8_ * sizeof(int)));
  if (!c_r8x8_ptr_) {
    return RET_MEMORY_FAILED;
  }
  memset(c_r8x8_ptr_, 0, params_->row_8_ * params_->col_8_ * sizeof(int));

  auto input_tensor = in_tensors_[0];
  auto params = input_tensor->GetQuantParams();
  MS_ASSERT(params.size() == 1);
  quant_params_.input.zp_ = params.front().zeroPoint;
  quant_params_.input.scale_ = params.front().scale;
  auto weight_tensor = in_tensors_[1];
  params = weight_tensor->GetQuantParams();
  MS_ASSERT(params.size() == 1);
  quant_params_.weight.zp_ = params.front().zeroPoint;
  quant_params_.weight.scale_ = params.front().scale;
  auto output_tensor = out_tensors_[0];
  params = output_tensor->GetQuantParams();
  MS_ASSERT(params.size() == 1);
  quant_params_.output.zp_ = params.front().zeroPoint;
  quant_params_.output.scale_ = params.front().scale;

  double real_multiplier = quant_params_.input.scale_ * quant_params_.weight.scale_ / quant_params_.output.scale_;
  QuantizeRoundParameter(real_multiplier, &quant_params_.quant_multiplier, &quant_params_.left_shift,
                         &quant_params_.right_shift);
  return RET_OK;
}

int MatmulInt8CPUKernel::ReSize() { return RET_OK; }

int MatmulInt8CPUKernel::RunImpl(int task_id) {
  int cur_oc = MSMIN(thread_stride_, UP_DIV(params_->col_8_, 8) - task_id * thread_stride_);
  if (cur_oc <= 0) {
    return RET_OK;
  }
  auto cur_b = b_r8_ptr_ + task_id * thread_stride_ * C8NUM * params_->deep_;
  auto cur_c = c_r8x8_ptr_ + task_id * thread_stride_ * C8NUM * params_->row_8_;
  MatMulInt8(a_c8_ptr_, cur_b, cur_c, params_->row_8_, cur_oc * 8, params_->deep_, quant_params_.input.zp_,
             quant_params_.weight.zp_);
  return RET_OK;
}

int MatmulInt8Run(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto op = reinterpret_cast<MatmulInt8CPUKernel *>(cdata);
  auto ret = op->RunImpl(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MatmulInt8Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int MatmulInt8CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }
  auto a_ptr = reinterpret_cast<int8_t *>(in_tensors_[0]->Data());
  auto b_ptr = reinterpret_cast<int8_t *>(in_tensors_[1]->Data());
  auto c_ptr = reinterpret_cast<int8_t *>(out_tensors_[0]->Data());
  auto a_stride = params_->row_ * params_->deep_;
  auto b_stride = params_->deep_ * params_->col_;
  auto c_stride = params_->row_ * params_->col_;

  for (int i = 0; i < params_->batch; ++i) {
    auto cur_a_ptr = a_ptr + i * a_stride;
    auto cur_b_ptr = b_ptr + i * b_stride;
    auto cur_c_ptr = c_ptr + i * c_stride;
    if (params_->a_transpose_) {
      RowMajor2Row8MajorInt8(cur_a_ptr, a_c8_ptr_, params_->deep_, params_->row_);
    } else {
      RowMajor2Col8MajorInt8(cur_a_ptr, a_c8_ptr_, params_->row_, params_->deep_);
    }
    if (params_->b_transpose_) {
      RowMajor2Col8MajorInt8(cur_b_ptr, b_r8_ptr_, params_->col_, params_->deep_);
    } else {
      RowMajor2Row8MajorInt8(cur_b_ptr, b_r8_ptr_, params_->deep_, params_->col_);
    }
    LiteBackendParallelLaunch(MatmulInt8Run, this, thread_count_);
    auto &q = quant_params_;
    SimplePostFuncInt8(c_r8x8_ptr_, cur_c_ptr, params_->col_, params_->row_, params_->row_8_, q.quant_multiplier,
                       q.left_shift, q.right_shift, q.output.zp_);
  }

  return RET_OK;
}

}  // namespace mindspore::kernel
