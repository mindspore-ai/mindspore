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
#include "src/runtime/runtime_api.h"
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_FullConnection;

namespace mindspore::kernel {
void FullconnectionInt8CPUKernel::FreeQuantParam() {
  if (quant_.filter_scale_ != nullptr) {
    free(quant_.filter_scale_);
    quant_.filter_scale_ = nullptr;
  }
  if (quant_.filter_zp_ != nullptr) {
    free(quant_.filter_zp_);
    quant_.filter_zp_ = nullptr;
  }
  if (quant_.left_shift_ != nullptr) {
    free(quant_.left_shift_);
    quant_.left_shift_ = nullptr;
  }
  if (quant_.right_shift_ != nullptr) {
    free(quant_.right_shift_);
    quant_.right_shift_ = nullptr;
  }
  if (quant_.quant_multiplier_ != nullptr) {
    free(quant_.quant_multiplier_);
    quant_.quant_multiplier_ = nullptr;
  }
  return;
}

void FullconnectionInt8CPUKernel::FreeTmpBuffer() {
  if (pack_a_ptr_ != nullptr) {
    free(pack_a_ptr_);
    pack_a_ptr_ = nullptr;
  }
  if (pack_b_ptr_ != nullptr) {
    free(pack_b_ptr_);
    pack_b_ptr_ = nullptr;
  }
  if (input_sums_ != nullptr) {
    free(input_sums_);
    input_sums_ = nullptr;
  }
  if (weight_bias_sums_ != nullptr) {
    free(weight_bias_sums_);
    weight_bias_sums_ = nullptr;
  }
  if (bias_ptr_ != nullptr) {
    free(bias_ptr_);
    bias_ptr_ = nullptr;
  }
  return;
}

int FullconnectionInt8CPUKernel::MallocQuantParam() {
  auto weight_tensor = in_tensors_.at(1);
  auto weight_quant_params = weight_tensor->quant_params();
  int col = weight_tensor->shape().front();
  filter_per_channel_ = (weight_quant_params.size() > 1);

  int init_size = filter_per_channel_ ? col : 1;

  quant_.filter_scale_ = reinterpret_cast<float *>(malloc(init_size * sizeof(float)));
  if (quant_.filter_scale_ == nullptr) {
    return RET_ERROR;
  }
  quant_.filter_zp_ = reinterpret_cast<int32_t *>(malloc(init_size * sizeof(int32_t)));
  if (quant_.filter_zp_ == nullptr) {
    return RET_ERROR;
  }
  quant_.left_shift_ = reinterpret_cast<int32_t *>(malloc(init_size * sizeof(int32_t)));
  if (quant_.left_shift_ == nullptr) {
    return RET_ERROR;
  }
  quant_.right_shift_ = reinterpret_cast<int32_t *>(malloc(init_size * sizeof(int32_t)));
  if (quant_.right_shift_ == nullptr) {
    return RET_ERROR;
  }
  quant_.quant_multiplier_ = reinterpret_cast<int32_t *>(malloc(init_size * sizeof(int32_t)));
  if (quant_.quant_multiplier_ == nullptr) {
    return RET_ERROR;
  }
  return RET_OK;
}

int FullconnectionInt8CPUKernel::Init() {
  auto ret = MallocQuantParam();
  if (ret != RET_OK) {
    FreeQuantParam();
    return ret;
  }

  auto in_quant_params = in_tensors_.at(0)->quant_params();
  quant_.input_.zp_ = in_quant_params.front().zeroPoint;
  quant_.input_.scale_ = in_quant_params.front().scale;

  auto out_quant_params = out_tensors_.at(0)->quant_params();
  quant_.output_.zp_ = out_quant_params.front().zeroPoint;
  quant_.output_.scale_ = out_quant_params.front().scale;

  auto weight_tensor = in_tensors_.at(1);
  fc_param_->b_const_ = (weight_tensor->data_c() != nullptr);
  int weight_quant_num = filter_per_channel_ ? weight_tensor->shape().front() : 1;
  auto weight_quant_params = weight_tensor->quant_params();

  for (int i = 0; i < weight_quant_num; i++) {
    quant_.filter_zp_[i] = weight_quant_params[i].zeroPoint;
    quant_.filter_scale_[i] = weight_quant_params[i].scale;
  }

  for (int i = 0; i < weight_quant_num; ++i) {
    const double in_scale = static_cast<double>(quant_.input_.scale_ * quant_.filter_scale_[i]);
    double real_multiplier = in_scale / static_cast<double>(quant_.output_.scale_);
    QuantizeRoundParameterWithDoublePrecision(real_multiplier, &quant_.quant_multiplier_[i], &quant_.left_shift_[i],
                                              &quant_.right_shift_[i]);
  }

  CalculateActivationRangeQuantized(fc_param_->act_type_ == ActType_Relu, fc_param_->act_type_ == ActType_Relu6,
                                    quant_.output_.zp_, quant_.output_.scale_, &quant_.out_act_min_,
                                    &quant_.out_act_max_);

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

void FullconnectionInt8CPUKernel::InitParam() {
  int row = 1;
  for (size_t i = 0; i < out_tensors_.at(0)->shape().size() - 1; ++i) {
    row *= (out_tensors_.at(0)->shape()).at(i);
  }
  fc_param_->row_ = row;
  fc_param_->col_ = out_tensors_.at(0)->shape().back();
  fc_param_->deep_ = (in_tensors_.at(1)->shape()).at(1);

  fc_param_->row_4_ = UP_ROUND(fc_param_->row_, C4NUM);
  fc_param_->col_4_ = UP_ROUND(fc_param_->col_, C4NUM);
  fc_param_->col_8_ = UP_ROUND(fc_param_->col_, C8NUM);
  fc_param_->deep_16_ = UP_ROUND(fc_param_->deep_, C16NUM);

  thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(fc_param_->col_4_, C4NUM));
  thread_stride_ = UP_DIV(UP_DIV(fc_param_->col_4_, C4NUM), thread_count_);
  return;
}

int FullconnectionInt8CPUKernel::ReSize() {
  FreeTmpBuffer();

  InitParam();

  pack_a_ptr_ = reinterpret_cast<int8_t *>(malloc(fc_param_->row_4_ * fc_param_->deep_16_ * sizeof(int8_t)));
  if (pack_a_ptr_ == nullptr) {
    FreeTmpBuffer();
    return RET_ERROR;
  }
  pack_b_ptr_ = reinterpret_cast<int8_t *>(malloc(fc_param_->col_4_ * fc_param_->deep_16_ * sizeof(int8_t)));
  if (pack_b_ptr_ == nullptr) {
    FreeTmpBuffer();
    return RET_ERROR;
  }
  input_sums_ = reinterpret_cast<int *>(malloc(fc_param_->row_4_ * sizeof(int)));
  if (input_sums_ == nullptr) {
    FreeTmpBuffer();
    return RET_ERROR;
  }
  weight_bias_sums_ = reinterpret_cast<int *>(malloc(fc_param_->col_4_ * sizeof(int)));
  if (weight_bias_sums_ == nullptr) {
    FreeTmpBuffer();
    return RET_ERROR;
  }

  memset(pack_a_ptr_, 0, fc_param_->row_4_ * fc_param_->deep_16_ * sizeof(int8_t));
  memset(pack_b_ptr_, 0, fc_param_->col_4_ * fc_param_->deep_16_ * sizeof(int8_t));
  memset(input_sums_, 0, fc_param_->row_4_ * sizeof(int));
  memset(weight_bias_sums_, 0, fc_param_->col_4_ * sizeof(int));

  if (in_tensors_.size() == 3) {
    bias_ptr_ = reinterpret_cast<int *>(malloc(fc_param_->col_4_ * sizeof(int)));
    if (bias_ptr_ == nullptr) {
      MS_LOG(ERROR) << "Memory allocation failed";
      FreeTmpBuffer();
      return RET_MEMORY_FAILED;
    }
    memcpy(bias_ptr_, in_tensors_.at(2)->data_c(), fc_param_->col_ * sizeof(int));
  } else {
    bias_ptr_ = nullptr;
  }

  if (fc_param_->b_const_) {
    auto weight_data = reinterpret_cast<int8_t *>(in_tensors_.at(1)->data_c());
    RowMajor2Row16x4MajorInt8(weight_data, pack_b_ptr_, fc_param_->col_, fc_param_->deep_);
    CalcWeightBiasSums(weight_data, fc_param_->deep_, fc_param_->col_, quant_.input_.zp_, quant_.filter_zp_, bias_ptr_,
                       weight_bias_sums_, ColMajor, filter_per_channel_);
  }
  return RET_OK;
}

int FullconnectionInt8CPUKernel::RunImpl(int task_id) {
  int stride = thread_stride_ * C4NUM;
  int cur_stride = task_id * stride;
  int res_stride = fc_param_->col_ - cur_stride;
  int cur_oc = MSMIN(stride, res_stride);
  if (cur_oc <= 0) {
    return RET_OK;
  }

  int32_t *cur_left = filter_per_channel_ ? quant_.left_shift_ + cur_stride : quant_.left_shift_;
  int32_t *cur_right = filter_per_channel_ ? quant_.right_shift_ + cur_stride : quant_.right_shift_;
  int32_t *cur_mul = filter_per_channel_ ? quant_.quant_multiplier_ + cur_stride : quant_.quant_multiplier_;
  int32_t *cur_zp = filter_per_channel_ ? quant_.filter_zp_ + cur_stride : quant_.filter_zp_;

  MatmulInt8Opt(pack_a_ptr_, pack_b_ptr_ + cur_stride * fc_param_->deep_16_, c_ptr_ + cur_stride, fc_param_->row_,
                cur_oc, fc_param_->deep_16_, input_sums_, weight_bias_sums_ + cur_stride, quant_.out_act_min_,
                quant_.out_act_max_, quant_.output_.zp_, cur_mul, cur_left, cur_right, fc_param_->col_,
                filter_per_channel_, cur_zp);

  return RET_OK;
}

int FcInt8Run(void *cdata, int task_id) {
  auto fc = reinterpret_cast<FullconnectionInt8CPUKernel *>(cdata);
  auto ret = fc->RunImpl(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FcInt8Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int FullconnectionInt8CPUKernel::Run() {
  auto input_ptr = reinterpret_cast<int8_t *>(in_tensors_.at(0)->data_c());
  RowMajor2Row16x4MajorInt8(input_ptr, pack_a_ptr_, fc_param_->row_, fc_param_->deep_);

  int32_t tmp_weight_zp = filter_per_channel_ ? 1 : quant_.filter_zp_[0];
  CalcInputSums(input_ptr, fc_param_->row_, fc_param_->deep_, tmp_weight_zp, input_sums_, RowMajor);

  if (!fc_param_->b_const_) {
    auto weight_data = reinterpret_cast<int8_t *>(in_tensors_.at(1)->data_c());
    RowMajor2Row16x4MajorInt8(weight_data, pack_b_ptr_, fc_param_->col_, fc_param_->deep_);
    CalcWeightBiasSums(weight_data, fc_param_->deep_, fc_param_->col_, quant_.input_.zp_, quant_.filter_zp_, bias_ptr_,
                       weight_bias_sums_, ColMajor, filter_per_channel_);
  }

  c_ptr_ = reinterpret_cast<int8_t *>(out_tensors_.at(0)->data_c());
  auto ret = ParallelLaunch(this->context_->thread_pool_, FcInt8Run, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParallelLaunch failed";
    return ret;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_FullConnection, LiteKernelCreator<FullconnectionInt8CPUKernel>)
}  // namespace mindspore::kernel
