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

#include "src/runtime/kernel/arm/int8/fullconnection_int8.h"
#include "nnacl/int8/matmul_int8.h"
#include "nnacl/common_func.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int FullconnectionInt8CPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int FullconnectionInt8CPUKernel::ReSize() {
  FreeTmpBuffer();
  fc_param_->row_ = (in_tensors_[0]->shape())[0];
  fc_param_->col_ = (in_tensors_[1]->shape())[0];
  fc_param_->deep_ = (in_tensors_[1]->shape())[1];
  fc_param_->row_8_ = UP_ROUND(fc_param_->row_, 8);
  fc_param_->col_8_ = UP_ROUND(fc_param_->col_, 8);

  r4_ = UP_ROUND(fc_param_->row_, 4);
  c4_ = UP_ROUND(fc_param_->col_, 4);
  d16_ = UP_ROUND(fc_param_->deep_, 16);
  thread_count_ = MSMIN(thread_count_, UP_DIV(c4_, 4));
  thread_stride_ = UP_DIV(UP_DIV(c4_, 4), thread_count_);
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
  auto weight_data = reinterpret_cast<int8_t *>(in_tensors_[1]->Data());
  RowMajor2Row4x16Major(weight_data, fc_param_->col_, fc_param_->deep_, b_c16x4_ptr_, d16_);
  if (in_tensors_.size() == 3) {
    auto bias_len = fc_param_->col_8_ * sizeof(int);
    bias_ptr_ = reinterpret_cast<int *>(ctx_->allocator->Malloc(bias_len));
    if (!bias_ptr_) return RET_MEMORY_FAILED;
    memcpy(bias_ptr_, in_tensors_[2]->Data(), bias_len);
  } else {
    bias_ptr_ = NULL;
  }

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
  CalculateActivationRangeQuantized(fc_param_->act_type_ == ActType_Relu, fc_param_->act_type_ == ActType_Relu6,
                                    quant_params_.output.zp_, quant_params_.output.scale_, &quant_params_.out_act_min,
                                    &quant_params_.out_act_max);
  CalcWeightBiasSums(weight_data, fc_param_->deep_, fc_param_->col_, quant_params_.input.zp_, quant_params_.weight.zp_,
                     bias_ptr_, weight_bias_sums_);
  return RET_OK;
}

int FullconnectionInt8CPUKernel::RunImpl(int task_id) {
  int cur_oc = MSMIN(thread_stride_, UP_DIV(c4_, 4) - task_id * thread_stride_);
  if (cur_oc <= 0) {
    return RET_OK;
  }
  int cur_oc_res = MSMIN(thread_stride_ * C4NUM, fc_param_->col_ - task_id * thread_stride_ * C4NUM);
  auto &q = quant_params_;
  auto &p = fc_param_;
  auto cur_b = b_c16x4_ptr_ + task_id * thread_stride_ * C4NUM * d16_;
  auto cur_bias = weight_bias_sums_ + task_id * thread_stride_ * C4NUM;
  auto output_ptr = reinterpret_cast<int8_t *>(out_tensors_[0]->Data());
  auto cur_c = output_ptr + task_id * thread_stride_ * C4NUM;
#ifdef ENABLE_ARM64
  MatmulInt8Neon64(a_r4x16_ptr_, cur_b, cur_c, r4_, cur_oc * C4NUM, d16_, input_sums_, cur_bias, q.out_act_min,
                   q.out_act_max, q.output.zp_, q.quant_multiplier, q.left_shift, q.right_shift, p->row_, cur_oc_res,
                   p->col_ * sizeof(int8_t));
#else
  MatmulInt8(a_r4x16_ptr_, cur_b, cur_c, input_sums_, cur_bias, q.out_act_min, q.out_act_max, q.output.zp_,
             q.quant_multiplier, q.left_shift, q.right_shift, p->row_, cur_oc_res, d16_, p->col_);
#endif

  return RET_OK;
}

int FcInt8Run(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto fc = reinterpret_cast<FullconnectionInt8CPUKernel *>(cdata);
  auto ret = fc->RunImpl(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FcInt8Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int FullconnectionInt8CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }
  auto input_ptr = reinterpret_cast<int8_t *>(in_tensors_[0]->Data());
  RowMajor2Row4x16Major(input_ptr, fc_param_->row_, fc_param_->deep_, a_r4x16_ptr_, d16_);
  CalcInputSums(input_ptr, fc_param_->row_, fc_param_->deep_, quant_params_.weight.zp_, input_sums_);
  LiteBackendParallelLaunch(FcInt8Run, this, thread_count_);
  return RET_OK;
}

}  // namespace mindspore::kernel
