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
#include "src/runtime/kernel/arm/nnacl/int8/matmul_int8.h"
#include "src/runtime/kernel/arm/nnacl/common_func.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int FullconnectionInt8CPUKernel::Init() {
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    set_need_reinit();
    return RET_OK;
  }
  fc_param_->row_ = (in_tensors_[0]->shape())[0];
  fc_param_->col_ = (in_tensors_[1]->shape())[0];
  fc_param_->deep_ = (in_tensors_[1]->shape())[1];
  fc_param_->row_8_ = UP_ROUND(fc_param_->row_, 8);
  fc_param_->col_8_ = UP_ROUND(fc_param_->col_, 8);

  thread_count_ = MSMIN(thread_count_, UP_DIV(fc_param_->col_8_, 8));
  thread_stride_ = UP_DIV(UP_DIV(fc_param_->col_8_, 8), thread_count_);

  a_c8_ptr_ =
    reinterpret_cast<int8_t *>(ctx_->allocator->Malloc(fc_param_->row_8_ * fc_param_->deep_ * sizeof(int8_t)));
  if (!a_c8_ptr_) {
    return RET_MEMORY_FAILED;
  }
  memset(a_c8_ptr_, 0, fc_param_->row_8_ * fc_param_->deep_ * sizeof(int8_t));
  b_r8_ptr_ =
    reinterpret_cast<int8_t *>(ctx_->allocator->Malloc(fc_param_->col_8_ * fc_param_->deep_ * sizeof(int8_t)));
  if (!b_r8_ptr_) {
    return RET_MEMORY_FAILED;
  }
  memset(b_r8_ptr_, 0, fc_param_->col_8_ * fc_param_->deep_ * sizeof(int8_t));
  auto weight_data = reinterpret_cast<int8_t *>(in_tensors_[1]->Data());
  RowMajor2Col8MajorInt8(weight_data, b_r8_ptr_, fc_param_->col_, fc_param_->deep_);
  c_r8x8_ptr_ = reinterpret_cast<int *>(ctx_->allocator->Malloc(fc_param_->row_8_ * fc_param_->col_8_ * sizeof(int)));
  if (!c_r8x8_ptr_) {
    return RET_MEMORY_FAILED;
  }
  memset(c_r8x8_ptr_, 0, fc_param_->row_8_ * fc_param_->col_8_ * sizeof(int));
  auto bias_len = fc_param_->col_8_ * sizeof(int);
  bias_ptr_ = reinterpret_cast<int *>(ctx_->allocator->Malloc(bias_len));
  if (!bias_ptr_) {
    return RET_MEMORY_FAILED;
  }
  memset(bias_ptr_, 0, bias_len);
  if (in_tensors_.size() == 3) {
    memcpy(bias_ptr_, in_tensors_[2]->Data(), bias_len);
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
                                    quant_params_.output.zp_, quant_params_.output.scale_, &quant_params_.out_act_max,
                                    &quant_params_.out_act_min);
  return RET_OK;
}

int FullconnectionInt8CPUKernel::ReSize() { return RET_OK; }

int FullconnectionInt8CPUKernel::RunImpl(int task_id) {
  int cur_oc = MSMIN(thread_stride_, UP_DIV(fc_param_->col_8_, 8) - task_id * thread_stride_);
  if (cur_oc <= 0) {
    return RET_OK;
  }
  auto &p = quant_params_;
  auto cur_b = b_r8_ptr_ + task_id * thread_stride_ * C8NUM * fc_param_->deep_;
  auto cur_c = c_r8x8_ptr_ + task_id * thread_stride_ * C8NUM * fc_param_->row_8_;
  MatMulInt8(a_c8_ptr_, cur_b, cur_c, fc_param_->row_8_, cur_oc * 8, fc_param_->deep_, p.input.zp_, p.weight.zp_);
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
  auto a_ptr = reinterpret_cast<int8_t *>(in_tensors_[0]->Data());
  auto output_ptr = reinterpret_cast<int8_t *>(out_tensors_[0]->Data());
  auto &p = quant_params_;
  RowMajor2Col8MajorInt8(a_ptr, a_c8_ptr_, fc_param_->row_, fc_param_->deep_);
  LiteBackendParallelLaunch(FcInt8Run, this, thread_count_);
  PostFuncInt8(c_r8x8_ptr_, bias_ptr_, output_ptr, fc_param_->col_, fc_param_->row_, fc_param_->row_8_,
               p.quant_multiplier, p.left_shift, p.right_shift, p.output.zp_, p.out_act_min, p.out_act_max);
  return RET_OK;
}

}  // namespace mindspore::kernel
