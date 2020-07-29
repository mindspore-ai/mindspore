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
#include "src/runtime/kernel/arm/opclib/int8/matmul.h"
#include "src/runtime/kernel/arm/opclib/common_func.h"
#include "include/errorcode.h"

using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int FullconnectionInt8CPUKernel::Init() {
  fc_param_->row_ = (inputs_[0]->shape())[0];
  fc_param_->col_ = (inputs_[1]->shape())[1];
  fc_param_->deep_ = (inputs_[1]->shape())[0];
  fc_param_->row_8_ = UP_ROUND(fc_param_->row_, 8);
  fc_param_->col_8_ = UP_ROUND(fc_param_->col_, 8);

  a_c8_ptr_ =
    reinterpret_cast<int8_t *>(ctx_->allocator->Malloc(fc_param_->row_8_ * fc_param_->deep_ * sizeof(int8_t)));
  memset(a_c8_ptr_, 0, fc_param_->row_8_ * fc_param_->deep_ * sizeof(int8_t));
  b_r8_ptr_ =
    reinterpret_cast<int8_t *>(ctx_->allocator->Malloc(fc_param_->col_8_ * fc_param_->deep_ * sizeof(int8_t)));
  memset(b_r8_ptr_, 0, fc_param_->col_8_ * fc_param_->deep_ * sizeof(int8_t));
  c_r8x8_ptr_ = reinterpret_cast<int *>(ctx_->allocator->Malloc(fc_param_->row_8_ * fc_param_->col_8_ * sizeof(int)));
  memset(c_r8x8_ptr_, 0, fc_param_->row_8_ * fc_param_->col_8_ * sizeof(int));
  if (!a_c8_ptr_ || !b_r8_ptr_ || !c_r8x8_ptr_) {
    return RET_MEMORY_FAILED;
  }

  auto input_tensor = inputs_[0];
  auto params = input_tensor->GetQuantParams();
  MS_ASSERT(params.size() == 1);
  quant_params_.input.zp_ = params.front().zeroPoint;
  quant_params_.input.scale_ = params.front().scale;
  auto weight_tensor = inputs_[1];
  params = weight_tensor->GetQuantParams();
  MS_ASSERT(params.size() == 1);
  quant_params_.weight.zp_ = params.front().zeroPoint;
  quant_params_.weight.scale_ = params.front().scale;
  auto output_tensor = outputs_[0];
  params = output_tensor->GetQuantParams();
  MS_ASSERT(params.size() == 1);
  quant_params_.output.zp_ = params.front().zeroPoint;
  quant_params_.output.scale_ = params.front().scale;

  double real_multiplier = quant_params_.input.scale_ * quant_params_.weight.scale_ / quant_params_.output.scale_;
  QuantizeMultiplier(real_multiplier, &quant_params_.quant_multiplier, &quant_params_.output_shift);
  CalculateActivationRangeQuantized(fc_param_->maxf_, fc_param_->minf_, quant_params_.output.scale_,
                                    quant_params_.output.zp_, &quant_params_.out_act_max, &quant_params_.out_act_min);

  return RET_OK;
}

int FullconnectionInt8CPUKernel::ReSize() { return RET_OK; }

int FullconnectionInt8CPUKernel::Run() {
  auto a_ptr = reinterpret_cast<int8_t *>(inputs_.at(0)->Data());
  auto b_ptr = reinterpret_cast<int8_t *>(inputs_.at(1)->Data());
  auto bias_ptr = reinterpret_cast<int *>(inputs_.at(2)->Data());
  auto output_ptr = reinterpret_cast<int8_t *>(outputs_.at(0)->Data());
  auto &p = quant_params_;

  // rows*depth -> rows*depth, col_8 major
  RowMajor2Col8MajorInt8(a_ptr, a_c8_ptr_, fc_param_->row_, fc_param_->deep_);
  // cols*depth -> cols*depth, col_8 major == depth*cols, row_8 major
  RowMajor2Col8MajorInt8(b_ptr, b_r8_ptr_, fc_param_->col_, fc_param_->deep_);
  MatMulInt8(a_c8_ptr_, b_r8_ptr_, c_r8x8_ptr_, fc_param_->row_8_, fc_param_->col_8_, fc_param_->deep_, p.input.zp_,
             p.weight.zp_);
  PostFuncInt8(c_r8x8_ptr_, bias_ptr, output_ptr, fc_param_->col_, fc_param_->row_, fc_param_->col_8_,
               fc_param_->row_8_, p.quant_multiplier, p.output_shift, p.output.zp_, p.out_act_min, p.out_act_max);

  return RET_OK;
}
}  // namespace mindspore::kernel
