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
#include "nnacl/int8/matmul_int8.h"
#include "nnacl/common_func.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
MatmulInt8CPUKernel::~MatmulInt8CPUKernel() { FreeTmpBuffer(); }

int MatmulInt8CPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int MatmulInt8CPUKernel::ReSize() {
  FreeTmpBuffer();
  int batch = 1;
  auto x_shape = in_tensors_[0]->shape();
  auto o_shape = out_tensors_[0]->shape();
  for (size_t i = 0; i < x_shape.size() - 2; ++i) {
    batch *= x_shape[i];
  }
  params_->batch = batch;
  params_->row_ = o_shape[o_shape.size() - 2];
  params_->col_ = o_shape[o_shape.size() - 1];
  params_->deep_ = params_->a_transpose_ ? x_shape[x_shape.size() - 2] : x_shape[x_shape.size() - 1];
  params_->row_8_ = UP_ROUND(params_->row_, 8);
  params_->col_8_ = UP_ROUND(params_->col_, 8);

  r4_ = UP_ROUND(params_->row_, 4);
  c4_ = UP_ROUND(params_->col_, 4);
  d16_ = UP_ROUND(params_->deep_, 16);
  a_r4x16_ptr_ = reinterpret_cast<int8_t *>(ctx_->allocator->Malloc(r4_ * d16_ * sizeof(int8_t)));
  if (!a_r4x16_ptr_) return RET_MEMORY_FAILED;
  memset(a_r4x16_ptr_, 0, r4_ * d16_ * sizeof(int8_t));
  b_c16x4_ptr_ = reinterpret_cast<int8_t *>(ctx_->allocator->Malloc(c4_ * d16_ * sizeof(int8_t)));
  if (!b_c16x4_ptr_) return RET_MEMORY_FAILED;
  memset(b_c16x4_ptr_, 0, c4_ * d16_ * sizeof(int8_t));
  input_sums_ = reinterpret_cast<int *>(ctx_->allocator->Malloc(r4_ * sizeof(int)));
  if (!input_sums_) return RET_MEMORY_FAILED;
  memset(input_sums_, 0, r4_ * sizeof(int));
  weight_bias_sums_ = reinterpret_cast<int *>(ctx_->allocator->Malloc(c4_ * sizeof(int)));
  if (!weight_bias_sums_) return RET_MEMORY_FAILED;
  memset(weight_bias_sums_, 0, c4_ * sizeof(int));
  thread_count_ = MSMIN(thread_count_, UP_DIV(c4_, 4));
  thread_stride_ = UP_DIV(UP_DIV(c4_, 4), thread_count_);

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

int MatmulInt8CPUKernel::RunImpl(int task_id) {
  int cur_oc = MSMIN(thread_stride_, UP_DIV(c4_, 4) - task_id * thread_stride_);
  if (cur_oc <= 0) {
    return RET_OK;
  }
  int cur_oc_res = MSMIN(thread_stride_ * C4NUM, params_->col_ - task_id * thread_stride_ * C4NUM);
  auto cur_b = b_c16x4_ptr_ + task_id * thread_stride_ * 4 * d16_;
  auto cur_bias = weight_bias_sums_ + task_id * thread_stride_ * 4;
  auto cur_c = c_ptr_ + task_id * thread_stride_ * 4;

  auto &p = quant_params_;
#ifdef ENABLE_ARM64
  MatmulInt8Neon64(a_r4x16_ptr_, cur_b, cur_c, r4_, cur_oc * C4NUM, d16_, input_sums_, cur_bias, INT8_MIN, INT8_MAX,
                   p.output.zp_, p.quant_multiplier, p.left_shift, p.right_shift, params_->row_, cur_oc_res,
                   params_->col_ * sizeof(int8_t));
#else
  MatmulInt8(a_r4x16_ptr_, cur_b, cur_c, input_sums_, cur_bias, INT8_MIN, INT8_MAX, p.output.zp_, p.quant_multiplier,
             p.left_shift, p.right_shift, params_->row_, cur_oc_res, d16_, params_->col_);
#endif

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

    if (params_->a_transpose_) {
      RowMajor2Col16x4Major(cur_a_ptr, params_->deep_, params_->row_, a_r4x16_ptr_, d16_);
    } else {
      RowMajor2Row4x16Major(cur_a_ptr, params_->row_, params_->deep_, a_r4x16_ptr_, d16_);
    }
    if (params_->b_transpose_) {
      RowMajor2Row4x16Major(cur_b_ptr, params_->col_, params_->deep_, b_c16x4_ptr_, d16_);
    } else {
      RowMajor2Col16x4Major(cur_b_ptr, params_->deep_, params_->col_, b_c16x4_ptr_, d16_);
    }
    c_ptr_ = c_ptr + i * c_stride;
    auto &q = quant_params_;
    CalcInputSums(cur_a_ptr, params_->row_, params_->deep_, q.weight.zp_, input_sums_);
    CalcWeightBiasSums(cur_b_ptr, params_->deep_, params_->col_, q.input.zp_, q.weight.zp_, NULL, weight_bias_sums_);
    ret = LiteBackendParallelLaunch(MatmulInt8Run, this, thread_count_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "MatmulInt8Run error: [" << ret << "]";
      return ret;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
