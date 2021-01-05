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
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_MatMul;

namespace mindspore::kernel {
MatmulInt8CPUKernel::~MatmulInt8CPUKernel() { FreeTmpBuffer(); }

void MatmulInt8CPUKernel::FreeTmpBuffer() {
  if (a_r4x16_ptr_ != nullptr) {
    context_->allocator->Free(a_r4x16_ptr_);
    a_r4x16_ptr_ = nullptr;
  }
  if (input_sums_ != nullptr) {
    context_->allocator->Free(input_sums_);
    input_sums_ = nullptr;
  }
  if (b_c16x4_batch_ != nullptr) {
    context_->allocator->Free(b_c16x4_batch_);
    b_c16x4_batch_ = nullptr;
  }
  if (weight_bias_sums_batch_ != nullptr) {
    context_->allocator->Free(weight_bias_sums_batch_);
    weight_bias_sums_batch_ = nullptr;
  }
  if (bias_ptr_ != nullptr) {
    context_->allocator->Free(bias_ptr_);
    bias_ptr_ = nullptr;
  }
  return;
}

int MatmulInt8CPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int MatmulInt8CPUKernel::ReSize() {
  FreeTmpBuffer();
  int batch = 1;
  auto x_shape = in_tensors_.at(0)->shape();
  auto o_shape = out_tensors_.at(0)->shape();
  MS_ASSERT(x_shape.size() >= 2);
  for (size_t i = 0; i < x_shape.size() - 2; ++i) {
    batch *= x_shape[i];
  }
  params_->batch = batch;
  MS_ASSERT(o_shape.size() >= 2);
  params_->row_ = o_shape[o_shape.size() - 2];
  params_->col_ = o_shape[o_shape.size() - 1];
  params_->deep_ = params_->a_transpose_ ? x_shape[x_shape.size() - 2] : x_shape[x_shape.size() - 1];
  params_->row_4_ = UP_ROUND(params_->row_, 4);
  params_->col_4_ = UP_ROUND(params_->col_, 4);
  params_->deep_16_ = UP_ROUND(params_->deep_, 16);
  a_r4x16_ptr_ =
    reinterpret_cast<int8_t *>(context_->allocator->Malloc(params_->row_4_ * params_->deep_16_ * sizeof(int8_t)));
  if (!a_r4x16_ptr_) return RET_MEMORY_FAILED;
  memset(a_r4x16_ptr_, 0, params_->row_4_ * params_->deep_16_ * sizeof(int8_t));
  input_sums_ = reinterpret_cast<int *>(context_->allocator->Malloc(params_->row_4_ * sizeof(int)));
  if (!input_sums_) return RET_MEMORY_FAILED;
  memset(input_sums_, 0, params_->row_4_ * sizeof(int));
  b_c16x4_batch_ = reinterpret_cast<int8_t *>(
    context_->allocator->Malloc(params_->batch * params_->col_4_ * params_->deep_16_ * sizeof(int8_t)));
  if (!b_c16x4_batch_) return RET_MEMORY_FAILED;
  memset(b_c16x4_batch_, 0, params_->batch * params_->col_4_ * params_->deep_16_ * sizeof(int8_t));
  weight_bias_sums_batch_ =
    reinterpret_cast<int *>(context_->allocator->Malloc(params_->batch * params_->col_4_ * sizeof(int)));
  if (!weight_bias_sums_batch_) return RET_MEMORY_FAILED;
  memset(weight_bias_sums_batch_, 0, params_->batch * params_->col_4_ * sizeof(int));
  if (in_tensors_.size() == 3) {
    auto bias_size = params_->col_4_ * sizeof(int);
    bias_ptr_ = reinterpret_cast<int *>(context_->allocator->Malloc(bias_size));
    if (!bias_ptr_) return RET_MEMORY_FAILED;
    memcpy(bias_ptr_, in_tensors_[2]->data_c(), bias_size);
  } else {
    bias_ptr_ = NULL;
  }
  thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(params_->col_4_, 4));
  thread_stride_ = UP_DIV(UP_DIV(params_->col_4_, 4), thread_count_);

  auto input_tensor = in_tensors_.at(0);
  auto params = input_tensor->quant_params();
  MS_ASSERT(params.size() == 1);
  quant_params_.input.zp_ = params.front().zeroPoint;
  quant_params_.input.scale_ = params.front().scale;
  auto weight_tensor = in_tensors_.at(1);
  params = weight_tensor->quant_params();
  MS_ASSERT(params.size() == 1);
  quant_params_.weight.zp_ = params.front().zeroPoint;
  quant_params_.weight.scale_ = params.front().scale;
  auto output_tensor = out_tensors_.at(0);
  params = output_tensor->quant_params();
  MS_ASSERT(params.size() == 1);
  quant_params_.output.zp_ = params.front().zeroPoint;
  quant_params_.output.scale_ = params.front().scale;

  params_->b_const_ = (in_tensors_.at(1)->data_c() != nullptr);
  if (params_->b_const_) {
    auto b_ptr = reinterpret_cast<int8_t *>(in_tensors_.at(1)->data_c());
    for (int i = 0; i < params_->batch; ++i) {
      auto cur_b = b_ptr + i * params_->deep_ * params_->col_;
      auto cur_b_pack = b_c16x4_batch_ + i * params_->col_4_ * params_->deep_16_;
      auto cur_sums = weight_bias_sums_batch_ + i * params_->col_4_;
      if (params_->b_transpose_) {
        RowMajor2Row16x4MajorInt8(cur_b, cur_b_pack, params_->col_, params_->deep_);
        CalcWeightBiasSums(cur_b, params_->deep_, params_->col_, quant_params_.input.zp_, &quant_params_.weight.zp_,
                           bias_ptr_, cur_sums, ColMajor, false);
      } else {
        RowMajor2Col16x4MajorInt8(cur_b, params_->deep_, params_->col_, cur_b_pack);
        CalcWeightBiasSums(cur_b, params_->deep_, params_->col_, quant_params_.input.zp_, &quant_params_.weight.zp_,
                           bias_ptr_, cur_sums, RowMajor, false);
      }
    }
  }
  double real_multiplier = quant_params_.input.scale_ * quant_params_.weight.scale_ / quant_params_.output.scale_;
  QuantizeRoundParameterWithDoublePrecision(real_multiplier, &quant_params_.quant_multiplier, &quant_params_.left_shift,
                                            &quant_params_.right_shift);
  return RET_OK;
}

int MatmulInt8CPUKernel::RunImpl(int task_id) {
  int cur_oc = MSMIN(thread_stride_, UP_DIV(params_->col_4_, 4) - task_id * thread_stride_);
  if (cur_oc <= 0) {
    return RET_OK;
  }
  int cur_oc_res = MSMIN(thread_stride_ * C4NUM, params_->col_ - task_id * thread_stride_ * C4NUM);
  auto cur_b = b_c16x4_ptr_ + task_id * thread_stride_ * 4 * params_->deep_16_;
  auto cur_bias = weight_bias_sums_ + task_id * thread_stride_ * 4;
  auto cur_c = c_ptr_ + task_id * thread_stride_ * 4;

  auto &p = quant_params_;
#ifdef ENABLE_ARM64
  MatmulInt8Neon64(a_r4x16_ptr_, cur_b, cur_c, params_->row_4_, cur_oc * C4NUM, params_->deep_16_, input_sums_,
                   cur_bias, INT8_MIN, INT8_MAX, p.output.zp_, &p.quant_multiplier, &p.left_shift, &p.right_shift,
                   params_->row_, cur_oc_res, params_->col_ * sizeof(int8_t), false);
#else
  MatMulInt8_16x4_r(a_r4x16_ptr_, cur_b, cur_c, params_->row_, cur_oc_res, params_->deep_16_, params_->col_,
                    input_sums_, cur_bias, &p.left_shift, &p.right_shift, &p.quant_multiplier, p.output.zp_, INT8_MIN,
                    INT8_MAX, false);
#endif

  return RET_OK;
}

int MatmulInt8Run(void *cdata, int task_id) {
  auto op = reinterpret_cast<MatmulInt8CPUKernel *>(cdata);
  auto ret = op->RunImpl(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MatmulInt8Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int MatmulInt8CPUKernel::Run() {
  auto a_ptr = reinterpret_cast<int8_t *>(in_tensors_.at(0)->data_c());
  auto c_ptr = reinterpret_cast<int8_t *>(out_tensors_.at(0)->data_c());
  auto a_stride = params_->row_ * params_->deep_;
  auto b_stride = params_->deep_ * params_->col_;
  auto c_stride = params_->row_ * params_->col_;

  if (!params_->b_const_) {
    auto b_ptr = reinterpret_cast<int8_t *>(in_tensors_.at(1)->data_c());
    for (int i = 0; i < params_->batch; ++i) {
      auto cur_b = b_ptr + i * b_stride;
      auto cur_b_pack = b_c16x4_batch_ + i * params_->col_4_ * params_->deep_16_;
      auto cur_sums = weight_bias_sums_batch_ + i * params_->col_4_;
      if (params_->b_transpose_) {
        RowMajor2Row16x4MajorInt8(cur_b, cur_b_pack, params_->col_, params_->deep_);
        CalcWeightBiasSums(cur_b, params_->deep_, params_->col_, quant_params_.input.zp_, &quant_params_.weight.zp_,
                           bias_ptr_, cur_sums, ColMajor, false);
      } else {
        RowMajor2Col16x4MajorInt8(cur_b, params_->deep_, params_->col_, cur_b_pack);
        CalcWeightBiasSums(cur_b, params_->deep_, params_->col_, quant_params_.input.zp_, &quant_params_.weight.zp_,
                           bias_ptr_, cur_sums, RowMajor, false);
      }
    }
  }

  for (int i = 0; i < params_->batch; ++i) {
    auto cur_a_ptr = a_ptr + i * a_stride;
    if (params_->a_transpose_) {
      RowMajor2Col16x4MajorInt8(cur_a_ptr, params_->deep_, params_->row_, a_r4x16_ptr_);
      CalcInputSums(cur_a_ptr, params_->row_, params_->deep_, quant_params_.weight.zp_, input_sums_, ColMajor);
    } else {
      RowMajor2Row16x4MajorInt8(cur_a_ptr, a_r4x16_ptr_, params_->row_, params_->deep_);
      CalcInputSums(cur_a_ptr, params_->row_, params_->deep_, quant_params_.weight.zp_, input_sums_, RowMajor);
    }
    b_c16x4_ptr_ = b_c16x4_batch_ + i * params_->col_4_ * params_->deep_16_;
    weight_bias_sums_ = weight_bias_sums_batch_ + i * params_->col_4_;
    c_ptr_ = c_ptr + i * c_stride;
    auto ret = ParallelLaunch(this->context_->thread_pool_, MatmulInt8Run, this, thread_count_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "MatmulInt8Run error: [" << ret << "]";
      return ret;
    }
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_MatMul, LiteKernelCreator<MatmulInt8CPUKernel>)
}  // namespace mindspore::kernel
