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

#include "src/runtime/kernel/arm/int8/matmul_base_int8.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int MatmulBaseInt8Run(void *cdata, int task_id) {
  auto op = reinterpret_cast<MatmulBaseInt8CPUKernel *>(cdata);
  auto ret = op->RunImpl(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MatmulInt8Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int MatmulBaseInt8CPUKernel::RunImpl(int task_id) {
  int stride = thread_stride_ * col_tile_;
  int cur_stride = task_id * stride;
  int res_stride = param_->col_ - cur_stride;
  int cur_oc = MSMIN(stride, res_stride);
  if (cur_oc <= 0) {
    return RET_OK;
  }

  int32_t *cur_left = filter_per_channel_ ? quant_.left_shift_ + cur_stride : quant_.left_shift_;
  int32_t *cur_right = filter_per_channel_ ? quant_.right_shift_ + cur_stride : quant_.right_shift_;
  int32_t *cur_mul = filter_per_channel_ ? quant_.quant_multiplier_ + cur_stride : quant_.quant_multiplier_;
  int32_t *cur_zp = filter_per_channel_ ? quant_.filter_zp_ + cur_stride : quant_.filter_zp_;

  MatmulInt8Opt(pack_a_ptr_, batch_b_ptr_ + cur_stride * param_->deep_16_, batch_c_ptr_ + cur_stride, param_->row_,
                cur_oc, param_->deep_16_, input_sums_, weight_bias_sums_ + cur_stride, quant_.out_act_min_,
                quant_.out_act_max_, quant_.output_.zp_, cur_mul, cur_left, cur_right, param_->col_,
                filter_per_channel_, cur_zp);

  return RET_OK;
}

MatmulBaseInt8CPUKernel::~MatmulBaseInt8CPUKernel() {
  FreeQuantParam();

  FreeTmpBuffer();

  if (bias_ptr_ != nullptr) {
    free(bias_ptr_);
    bias_ptr_ = nullptr;
  }
  return;
}

void MatmulBaseInt8CPUKernel::FreeQuantParam() {
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

int MatmulBaseInt8CPUKernel::MallocQuantParam() {
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

void MatmulBaseInt8CPUKernel::InitQuantParam() {
  auto in_quant_params = in_tensors_.at(0)->quant_params();
  quant_.input_.zp_ = in_quant_params.front().zeroPoint;
  quant_.input_.scale_ = in_quant_params.front().scale;

  auto out_quant_params = out_tensors_.at(0)->quant_params();
  quant_.output_.zp_ = out_quant_params.front().zeroPoint;
  quant_.output_.scale_ = out_quant_params.front().scale;

  auto weight_tensor = in_tensors_.at(1);
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

  CalculateActivationRangeQuantized(param_->act_type_ == ActType_Relu, param_->act_type_ == ActType_Relu6,
                                    quant_.output_.zp_, quant_.output_.scale_, &quant_.out_act_min_,
                                    &quant_.out_act_max_);
}

void MatmulBaseInt8CPUKernel::InitParameter() {
  param_->a_const_ = (in_tensors_[0]->data_c() != nullptr);
  param_->b_const_ = (in_tensors_[1]->data_c() != nullptr);
#ifdef ENABLE_ARM32
  row_tile_ = C4NUM;
  col_tile_ = C2NUM;
#else
  row_tile_ = C4NUM;
  col_tile_ = C4NUM;
#endif
  return;
}

void MatmulBaseInt8CPUKernel::ResizeParameter() {
  param_->row_align_ = UP_ROUND(param_->row_, row_tile_);
  param_->col_align_ = UP_ROUND(param_->col_, col_tile_);
  param_->deep_16_ = UP_ROUND(param_->deep_, C16NUM);

  thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(param_->col_align_, col_tile_));
  thread_stride_ = UP_DIV(UP_DIV(param_->col_align_, col_tile_), thread_count_);
  return;
}

void MatmulBaseInt8CPUKernel::FreeTmpBuffer() {
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
  return;
}

void MatmulBaseInt8CPUKernel::TransferB() {
  auto weight_data = reinterpret_cast<int8_t *>(in_tensors_.at(1)->data_c());
  for (int i = 0; i < param_->batch; i++) {
    auto current_weight = weight_data + i * param_->deep_ * param_->col_;
    auto current_b_pack = pack_b_ptr_ + i * param_->col_align_ * param_->deep_16_;
    auto current_sums = weight_bias_sums_ + i * param_->col_align_;
    if (param_->b_transpose_) {
#ifdef ENABLE_ARM32
      RowMajor2Row2x16MajorInt8(current_weight, current_b_pack, param_->col_, param_->deep_);
#else
      RowMajor2Row16x4MajorInt8(current_weight, current_b_pack, param_->col_, param_->deep_);
#endif
      CalcWeightBiasSums(current_weight, param_->deep_, param_->col_, quant_.input_.zp_, quant_.filter_zp_, bias_ptr_,
                         current_sums, ColMajor, filter_per_channel_);
    } else {
#ifdef ENABLE_ARM32
      RowMajor2Col16x2MajorInt8(current_weight, current_b_pack, param_->deep_, param_->col_);
#else
      RowMajor2Col16x4MajorInt8(current_weight, param_->deep_, param_->col_, current_b_pack);
#endif
      CalcWeightBiasSums(current_weight, param_->deep_, param_->col_, quant_.input_.zp_, quant_.filter_zp_, bias_ptr_,
                         current_sums, RowMajor, false);
    }
  }
  return;
}

int MatmulBaseInt8CPUKernel::InitTmpBuffer() {
  pack_a_ptr_ = reinterpret_cast<int8_t *>(malloc(param_->row_align_ * param_->deep_16_ * sizeof(int8_t)));
  if (pack_a_ptr_ == nullptr) {
    FreeTmpBuffer();
    return RET_ERROR;
  }
  pack_b_ptr_ =
    reinterpret_cast<int8_t *>(malloc(param_->batch * param_->col_align_ * param_->deep_16_ * sizeof(int8_t)));
  if (pack_b_ptr_ == nullptr) {
    FreeTmpBuffer();
    return RET_ERROR;
  }
  input_sums_ = reinterpret_cast<int *>(malloc(param_->row_align_ * sizeof(int)));
  if (input_sums_ == nullptr) {
    FreeTmpBuffer();
    return RET_ERROR;
  }
  weight_bias_sums_ = reinterpret_cast<int *>(malloc(param_->batch * param_->col_align_ * sizeof(int)));
  if (weight_bias_sums_ == nullptr) {
    FreeTmpBuffer();
    return RET_ERROR;
  }

  memset(pack_a_ptr_, 0, param_->row_align_ * param_->deep_16_ * sizeof(int8_t));
  memset(pack_b_ptr_, 0, param_->batch * param_->col_align_ * param_->deep_16_ * sizeof(int8_t));
  memset(input_sums_, 0, param_->row_align_ * sizeof(int));
  memset(weight_bias_sums_, 0, param_->batch * param_->col_align_ * sizeof(int));

  return RET_OK;
}

int MatmulBaseInt8CPUKernel::InitBias() {
  if (in_tensors_.size() == 3) {
    auto bias_tensor = in_tensors_[2];
    int max_bias_data = UP_ROUND(bias_tensor->ElementsNum(), C4NUM);
    bias_ptr_ = reinterpret_cast<int *>(malloc(max_bias_data * sizeof(int)));
    if (bias_ptr_ == nullptr) {
      MS_LOG(ERROR) << "Memory allocation failed";
      FreeTmpBuffer();
      return RET_MEMORY_FAILED;
    }
    memcpy(bias_ptr_, bias_tensor->data_c(), bias_tensor->ElementsNum() * sizeof(int));
  } else {
    bias_ptr_ = nullptr;
  }
  return RET_OK;
}

int MatmulBaseInt8CPUKernel::Init() {
  auto ret = MallocQuantParam();
  if (ret != RET_OK) {
    FreeQuantParam();
    return ret;
  }

  InitQuantParam();

  ret = InitBias();
  if (ret != RET_OK) {
    FreeQuantParam();
    return ret;
  }

  return RET_OK;
}

int MatmulBaseInt8CPUKernel::ReSize() {
  FreeTmpBuffer();

  ResizeParameter();

  auto ret = InitTmpBuffer();
  if (ret != RET_OK) {
    FreeQuantParam();
    return ret;
  }

  if (param_->b_const_ == true) {
    TransferB();
  }
  return RET_OK;
}

int MatmulBaseInt8CPUKernel::Run() {
  if (param_->b_const_ == false) {
    TransferB();
  }

  int8_t *a_ptr = reinterpret_cast<int8_t *>(in_tensors_.at(0)->data_c());
  int8_t *c_ptr = reinterpret_cast<int8_t *>(out_tensors_.at(0)->data_c());
  int32_t tmp_weight_zp = filter_per_channel_ ? 1 : quant_.filter_zp_[0];
  for (int i = 0; i < param_->batch; i++) {
    auto current_src_a = a_ptr + i * param_->row_ * param_->deep_;
    if (param_->a_transpose_) {
      RowMajor2Col16x4MajorInt8(current_src_a, param_->deep_, param_->row_, pack_a_ptr_);
      CalcInputSums(current_src_a, param_->row_, param_->deep_, tmp_weight_zp, input_sums_, ColMajor);
    } else {
      RowMajor2Row16x4MajorInt8(current_src_a, pack_a_ptr_, param_->row_, param_->deep_);
      CalcInputSums(current_src_a, param_->row_, param_->deep_, tmp_weight_zp, input_sums_, RowMajor);
    }

    batch_b_ptr_ = pack_b_ptr_ + i * param_->col_align_ * param_->deep_16_;
    batch_sums_ = weight_bias_sums_ + i * param_->col_align_;
    batch_c_ptr_ = c_ptr + i * param_->row_ * param_->col_;

    auto ret = ParallelLaunch(this->context_->thread_pool_, MatmulBaseInt8Run, this, thread_count_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "MatmulInt8Run error: [" << ret << "]";
      return ret;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
