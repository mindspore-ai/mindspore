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

#include "src/litert/kernel/cpu/int8/matmul_base_int8.h"
#include "src/litert/kernel/cpu/int8/opt_op_handler.h"
#include "src/litert/kernel/cpu/fp32/matmul_fp32_base.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int MatmulBaseInt8Run(void *cdata, int task_id, float, float) {
  CHECK_NULL_RETURN(cdata);
  auto op = reinterpret_cast<MatmulBaseInt8CPUKernel *>(cdata);
  auto ret = op->RunImpl(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MatmulInt8Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

#if defined(ENABLE_ARM64) && !defined(SUPPORT_NNIE) && !defined(SUPPORT_34XX) && (!defined(MACHINE_LINUX_ARM64))
int Arm64SdotPreRun(void *cdata, int task_id, float, float) {
  CHECK_NULL_RETURN(cdata);
  auto op = reinterpret_cast<MatmulBaseInt8CPUKernel *>(cdata);
  auto ret = op->Arm64SdotPre(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MatmulInt8Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int Arm64SdotRun(void *cdata, int task_id, float, float) {
  CHECK_NULL_RETURN(cdata);
  auto op = reinterpret_cast<MatmulBaseInt8CPUKernel *>(cdata);
  auto ret = op->Arm64SdotImpl(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MatmulInt8Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int MatmulBaseInt8CPUKernel::Arm64SdotPre(int task_id) {
  int row_thread_count = MSMIN(op_parameter_->thread_num_, UP_DIV(param_->row_align_, row_tile_));
  int row_stride = UP_DIV(UP_DIV(param_->row_align_, row_tile_), row_thread_count) * row_tile_;

  int row_current_stride = task_id * row_stride;
  int row_res_stride = param_->row_ - row_current_stride;
  int cur_r = MSMIN(row_res_stride, row_stride);
  if (cur_r <= 0) {
    return RET_OK;
  }

  int tmp_weight_zp = filter_per_channel_ ? 1 : quant_param_->filter_zp_[0];
  auto current_a_pack = pack_a_ptr_ + row_current_stride * param_->deep_align_;

  if (param_->a_transpose_) {
    auto current_src_a = batch_input_ptr_ + row_current_stride;
    PackInput2Col4x4AndInputSumPert(current_src_a, current_a_pack, input_sums_ + row_current_stride, param_->deep_,
                                    cur_r, param_->row_, tmp_weight_zp);
  } else {
    auto current_src_a = batch_input_ptr_ + row_current_stride * param_->deep_;
    PackInput4x4AndInputSumPert(current_src_a, current_a_pack, input_sums_ + row_current_stride, param_->deep_, cur_r,
                                tmp_weight_zp);
  }
  return RET_OK;
}

int MatmulBaseInt8CPUKernel::Arm64SdotImpl(int task_id) {
  int stride = thread_stride_ * col_tile_;
  int cur_stride = task_id * stride;
  int res_stride = param_->col_ - cur_stride;
  int cur_oc = MSMIN(stride, res_stride);
  if (cur_oc <= 0) {
    return RET_OK;
  }
  if (param_->b_const_ == false) {
    auto current_sums = batch_sums_ + cur_stride;
    auto current_b_pack = batch_b_ptr_ + cur_stride * param_->deep_align_;
    auto current_filter_zp = filter_per_channel_ ? quant_param_->filter_zp_ + cur_stride : quant_param_->filter_zp_;
    auto current_bias = bias_ptr_ == nullptr ? nullptr : bias_ptr_ + cur_stride;
    if (param_->b_transpose_) {
      auto current_weight = batch_weight_ptr_ + cur_stride * param_->deep_;

      RowMajor2Row4x16MajorInt8(current_weight, current_b_pack, cur_oc, param_->deep_);
      CalcPartWeightBiasSums(current_weight, param_->deep_, param_->col_, cur_oc, quant_param_->input_.zp_,
                             current_filter_zp, current_bias, current_sums, ColMajor, filter_per_channel_);
    } else {
      auto current_weight = batch_weight_ptr_ + cur_stride;
      RowMajor2Col4x16MajorPartInt8(current_weight, current_b_pack, param_->deep_, param_->col_, cur_oc);
      CalcPartWeightBiasSums(current_weight, param_->deep_, param_->col_, cur_oc, quant_param_->input_.zp_,
                             current_filter_zp, current_bias, current_sums, RowMajor, filter_per_channel_);
    }
  }

  int32_t *cur_left = filter_per_channel_ ? quant_param_->left_shift_ + cur_stride : quant_param_->left_shift_;
  int32_t *cur_right = filter_per_channel_ ? quant_param_->right_shift_ + cur_stride : quant_param_->right_shift_;
  int32_t *cur_mul =
    filter_per_channel_ ? quant_param_->quant_multiplier_ + cur_stride : quant_param_->quant_multiplier_;
  int32_t *cur_zp = filter_per_channel_ ? quant_param_->filter_zp_ + cur_stride : quant_param_->filter_zp_;

  MatmulInt8DpOpt(pack_a_ptr_, batch_b_ptr_ + cur_stride * param_->deep_align_, batch_c_ptr_ + cur_stride, param_->row_,
                  cur_oc, param_->deep_align_, input_sums_, batch_sums_ + cur_stride, quant_param_->out_act_min_,
                  quant_param_->out_act_max_, quant_param_->output_.zp_, cur_mul, cur_left, cur_right, param_->col_,
                  filter_per_channel_, cur_zp);

  return RET_OK;
}
#endif

int MatmulBaseInt8CPUKernel::RunImpl(int task_id) {
  int stride = thread_stride_ * col_tile_;
  int cur_stride = task_id * stride;
  int res_stride = param_->col_ - cur_stride;
  int cur_oc = MSMIN(stride, res_stride);
  if (cur_oc <= 0) {
    return RET_OK;
  }

  int32_t *cur_left = filter_per_channel_ ? quant_param_->left_shift_ + cur_stride : quant_param_->left_shift_;
  int32_t *cur_right = filter_per_channel_ ? quant_param_->right_shift_ + cur_stride : quant_param_->right_shift_;
  int32_t *cur_mul =
    filter_per_channel_ ? quant_param_->quant_multiplier_ + cur_stride : quant_param_->quant_multiplier_;
  int32_t *cur_zp = filter_per_channel_ ? quant_param_->filter_zp_ + cur_stride : quant_param_->filter_zp_;

  MatmulInt8Opt(pack_a_ptr_, batch_b_ptr_ + cur_stride * param_->deep_align_, batch_c_ptr_ + cur_stride, param_->row_,
                cur_oc, param_->deep_align_, input_sums_, batch_sums_ + cur_stride, quant_param_->out_act_min_,
                quant_param_->out_act_max_, quant_param_->output_.zp_, cur_mul, cur_left, cur_right, param_->col_,
                filter_per_channel_, cur_zp);

  return RET_OK;
}

MatmulBaseInt8CPUKernel::~MatmulBaseInt8CPUKernel() {
  FreeQuantParam();

  FreeTmpBuffer();
}

void MatmulBaseInt8CPUKernel::FreeQuantParam() {
  if (quant_param_ != nullptr) {
    if (quant_param_->filter_scale_ != nullptr) {
      free(quant_param_->filter_scale_);
      quant_param_->filter_scale_ = nullptr;
    }
    if (quant_param_->filter_zp_ != nullptr) {
      free(quant_param_->filter_zp_);
      quant_param_->filter_zp_ = nullptr;
    }
    if (quant_param_->left_shift_ != nullptr) {
      free(quant_param_->left_shift_);
      quant_param_->left_shift_ = nullptr;
    }
    if (quant_param_->right_shift_ != nullptr) {
      free(quant_param_->right_shift_);
      quant_param_->right_shift_ = nullptr;
    }
    if (quant_param_->quant_multiplier_ != nullptr) {
      free(quant_param_->quant_multiplier_);
      quant_param_->quant_multiplier_ = nullptr;
    }
    free(quant_param_);
    quant_param_ = nullptr;
  }

  if (save_b_const_ != nullptr) {
    free(save_b_const_);
    save_b_const_ = nullptr;
  }
}

int MatmulBaseInt8CPUKernel::MallocQuantParam() {
  auto weight_tensor = in_tensors_.at(1);
  auto weight_quant_params = weight_tensor->quant_params();

  MS_CHECK_TRUE_MSG(weight_quant_params.size() >= 1, lite::RET_ERROR, "weight quant params size should >= 1");
  filter_per_channel_ = (weight_quant_params.size() > 1);
  channel_num_ = weight_quant_params.size();

  const int &init_size = channel_num_;

  quant_param_ = reinterpret_cast<MatmulQuantParameter *>(malloc(sizeof(MatmulQuantParameter)));
  if (quant_param_ == nullptr) {
    MS_LOG(ERROR) << "Malloc MatmulQuantParameter for Matmul int8 op failed!";
    return RET_ERROR;
  }
  (void)memset(quant_param_, 0, sizeof(MatmulQuantParameter));
  quant_param_->filter_scale_ = reinterpret_cast<float *>(malloc(init_size * sizeof(float)));
  if (quant_param_->filter_scale_ == nullptr) {
    return RET_ERROR;
  }
  quant_param_->filter_zp_ = reinterpret_cast<int32_t *>(malloc(init_size * sizeof(int32_t)));
  if (quant_param_->filter_zp_ == nullptr) {
    return RET_ERROR;
  }
  quant_param_->left_shift_ = reinterpret_cast<int32_t *>(malloc(init_size * sizeof(int32_t)));
  if (quant_param_->left_shift_ == nullptr) {
    return RET_ERROR;
  }
  quant_param_->right_shift_ = reinterpret_cast<int32_t *>(malloc(init_size * sizeof(int32_t)));
  if (quant_param_->right_shift_ == nullptr) {
    return RET_ERROR;
  }
  quant_param_->quant_multiplier_ = reinterpret_cast<int32_t *>(malloc(init_size * sizeof(int32_t)));
  if (quant_param_->quant_multiplier_ == nullptr) {
    return RET_ERROR;
  }
  return RET_OK;
}

int MatmulBaseInt8CPUKernel::InitQuantParam() {
  auto in_quant_params = in_tensors_.at(0)->quant_params();
  if (in_quant_params.size() < 1) {
    MS_LOG(ERROR) << "invalid in quant param";
    return RET_ERROR;
  }
  quant_param_->input_.zp_ = in_quant_params.front().zeroPoint;
  quant_param_->input_.scale_ = in_quant_params.front().scale;

  auto out_quant_params = out_tensors_.at(0)->quant_params();
  if (out_quant_params.size() < 1) {
    MS_LOG(ERROR) << "invalid out quant param";
    return RET_ERROR;
  }
  quant_param_->output_.zp_ = out_quant_params.front().zeroPoint;
  quant_param_->output_.scale_ = out_quant_params.front().scale;

  auto weight_tensor = in_tensors_.at(1);
  const int &weight_quant_num = channel_num_;
  auto weight_quant_params = weight_tensor->quant_params();
  MS_CHECK_TRUE_RET(static_cast<int>(weight_quant_params.size()) == weight_quant_num, RET_ERROR);

  for (int i = 0; i < weight_quant_num; i++) {
    quant_param_->filter_zp_[i] = weight_quant_params[i].zeroPoint;
    quant_param_->filter_scale_[i] = weight_quant_params[i].scale;
  }

  for (int i = 0; i < weight_quant_num; ++i) {
    const double in_scale = static_cast<double>(quant_param_->input_.scale_ * quant_param_->filter_scale_[i]);
    double real_multiplier = in_scale / static_cast<double>(quant_param_->output_.scale_);
    QuantizeRoundParameterWithDoublePrecision(real_multiplier, &quant_param_->quant_multiplier_[i],
                                              &quant_param_->left_shift_[i], &quant_param_->right_shift_[i]);
  }

  CalculateActivationRangeQuantized(param_->act_type_ == ActType_Relu, param_->act_type_ == ActType_Relu6,
                                    quant_param_->output_.zp_, quant_param_->output_.scale_,
                                    &quant_param_->out_act_min_, &quant_param_->out_act_max_);
  return RET_OK;
}

void MatmulBaseInt8CPUKernel::InitParameter() {
  param_->a_const_ = (in_tensors_[0]->data() != nullptr);
  param_->b_const_ = (in_tensors_[1]->data() != nullptr);
#ifdef ENABLE_ARM32
  row_tile_ = C4NUM;
  col_tile_ = C2NUM;
  deep_tile_ = C16NUM;
#elif ENABLE_ARM64
  support_sdot_ = mindspore::lite::IsSupportSDot();
  row_tile_ = C4NUM;
  if (support_sdot_) {
    col_tile_ = C16NUM;
    deep_tile_ = C4NUM;
  } else {
    col_tile_ = C4NUM;
    deep_tile_ = C16NUM;
  }
#else
  row_tile_ = C4NUM;
  col_tile_ = C4NUM;
  deep_tile_ = C16NUM;
#endif
  if (param_->a_transpose_) {
    a_pack_func_ = RowMajor2Col16x4MajorInt8;
  } else {
    a_pack_func_ = RowMajor2Row16x4MajorInt8;
  }
  if (param_->b_transpose_) {
#ifdef ENABLE_ARM32
    b_pack_func_ = RowMajor2Row2x16MajorInt8;
#elif ENABLE_ARM64
    if (support_sdot_) {
      b_pack_func_ = RowMajor2Row4x16MajorInt8;
    } else {
      b_pack_func_ = RowMajor2Row16x4MajorInt8;
    }
#else
    b_pack_func_ = RowMajor2Row16x4MajorInt8;
#endif
  } else {
#ifdef ENABLE_ARM32
    b_pack_func_ = RowMajor2Col16x2MajorInt8;
#elif ENABLE_ARM64
    if (support_sdot_) {
      b_pack_func_ = RowMajor2Col4x16MajorInt8;
    } else {
      b_pack_func_ = RowMajor2Col16x4MajorInt8;
    }
#else
    b_pack_func_ = RowMajor2Col16x4MajorInt8;
#endif
  }
  return;
}

void MatmulBaseInt8CPUKernel::ResizeParameter() {
  param_->row_align_ = UP_ROUND(param_->row_, row_tile_);
  param_->col_align_ = UP_ROUND(param_->col_, col_tile_);
  param_->deep_align_ = UP_ROUND(param_->deep_, deep_tile_);

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

int MatmulBaseInt8CPUKernel::TransferB() {
  auto weight_data = (save_b_const_ == nullptr) ? reinterpret_cast<int8_t *>(in_tensors_.at(1)->data())
                                                : reinterpret_cast<int8_t *>(save_b_const_);
  CHECK_NULL_RETURN(weight_data);
  CHECK_NULL_RETURN(b_pack_func_);
  for (int i = 0; i < param_->batch; i++) {
    auto current_weight = weight_data + b_offset_[i] * param_->deep_ * param_->col_;
    auto current_b_pack = pack_b_ptr_ + b_offset_[i] * param_->col_align_ * param_->deep_align_;
    auto current_sums = weight_bias_sums_ + b_offset_[i] * param_->col_align_;
    if (param_->b_transpose_) {
      b_pack_func_(current_weight, current_b_pack, param_->col_, param_->deep_);
      CalcWeightBiasSums(current_weight, param_->deep_, param_->col_, quant_param_->input_.zp_,
                         quant_param_->filter_zp_, bias_ptr_, current_sums, ColMajor, filter_per_channel_);
    } else {
      b_pack_func_(current_weight, current_b_pack, param_->deep_, param_->col_);
      CalcWeightBiasSums(current_weight, param_->deep_, param_->col_, quant_param_->input_.zp_,
                         quant_param_->filter_zp_, bias_ptr_, current_sums, RowMajor, filter_per_channel_);
    }
  }
  if (save_b_const_ != nullptr) {
    free(save_b_const_);
    save_b_const_ = nullptr;
  }
  return RET_OK;
}

int MatmulBaseInt8CPUKernel::InitTmpBuffer() {
  pack_a_ptr_ = reinterpret_cast<int8_t *>(malloc(param_->row_align_ * param_->deep_align_ * sizeof(int8_t)));
  if (pack_a_ptr_ == nullptr) {
    FreeTmpBuffer();
    return RET_ERROR;
  }
  pack_b_ptr_ =
    reinterpret_cast<int8_t *>(malloc(param_->batch * param_->col_align_ * param_->deep_align_ * sizeof(int8_t)));
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

  (void)memset(pack_a_ptr_, 0, param_->row_align_ * param_->deep_align_ * sizeof(int8_t));
  (void)memset(pack_b_ptr_, 0, param_->batch * param_->col_align_ * param_->deep_align_ * sizeof(int8_t));
  (void)memset(input_sums_, 0, param_->row_align_ * sizeof(int));
  (void)memset(weight_bias_sums_, 0, param_->batch * param_->col_align_ * sizeof(int));

  return RET_OK;
}

int MatmulBaseInt8CPUKernel::InitBias() {
  if (in_tensors_.size() == kInputSize2) {
    auto bias_tensor = in_tensors_[kBiasIndex];
    if (bias_tensor->data_type() != kNumberTypeInt32) {
      MS_LOG(ERROR) << "Invalid bias tensor type.";
      FreeTmpBuffer();
      return RET_MEMORY_FAILED;
    }
    bias_ptr_ = reinterpret_cast<int *>(malloc(bias_tensor->ElementsNum() * sizeof(int)));
    if (bias_ptr_ == nullptr) {
      MS_LOG(ERROR) << "Memory allocation failed";
      FreeTmpBuffer();
      return RET_MEMORY_FAILED;
    }
    (void)memcpy(bias_ptr_, bias_tensor->data(), bias_tensor->ElementsNum() * sizeof(int));
  } else {
    bias_ptr_ = nullptr;
  }
  return RET_OK;
}

int MatmulBaseInt8CPUKernel::Prepare() {
  auto ret = MallocQuantParam();
  if (ret != RET_OK) {
    FreeQuantParam();
    return ret;
  }

  ret = InitQuantParam();
  if (ret != RET_OK) {
    FreeQuantParam();
    return ret;
  }

  ret = InitBias();
  if (ret != RET_OK) {
    FreeQuantParam();
    return ret;
  }
  if (!InferShapeDone()) {
    if (param_->b_const_) {
      auto weight_tensor = in_tensors_.at(1);
      CHECK_NULL_RETURN(weight_tensor);
      CHECK_NULL_RETURN(weight_tensor->data());
      save_b_const_ = reinterpret_cast<int8_t *>(malloc(weight_tensor->ElementsNum() * sizeof(int8_t)));
      (void)memcpy(save_b_const_, weight_tensor->data(), weight_tensor->ElementsNum() * sizeof(int8_t));
    }
  }
  return RET_OK;
}
int MatmulBaseInt8CPUKernel::MatmulReSize() {
  auto ret = MatmulFp32BaseCPUKernel::InitBroadcastParams(
    in_tensors_[kInputIndex]->shape(), in_tensors_[kWeightIndex]->shape(), param_, &a_offset_, &b_offset_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitBroadcastParams failed.";
    return RET_ERROR;
  }
  return MatmulBaseInt8CPUKernel::ReSize();
}

int MatmulBaseInt8CPUKernel::ReSize() {
  FreeTmpBuffer();

  ResizeParameter();

  auto ret = InitTmpBuffer();
  if (ret != RET_OK) {
    FreeQuantParam();
    return ret;
  }

  if (param_->b_const_) {
    if (TransferB() != RET_OK) {
      MS_LOG(ERROR) << "TransferB error";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

#if defined(ENABLE_ARM64) && !defined(SUPPORT_NNIE) && !defined(SUPPORT_34XX) && (!defined(MACHINE_LINUX_ARM64))
int MatmulBaseInt8CPUKernel::RunArm64Sdot() {
  int8_t *a_ptr = reinterpret_cast<int8_t *>(in_tensors_.at(0)->data());
  int8_t *b_ptr = reinterpret_cast<int8_t *>(in_tensors_.at(1)->data());
  int8_t *c_ptr = reinterpret_cast<int8_t *>(out_tensors_.at(0)->data());
  CHECK_NULL_RETURN(a_ptr);
  CHECK_NULL_RETURN(b_ptr);
  CHECK_NULL_RETURN(c_ptr);

  for (int i = 0; i < param_->batch; i++) {
    batch_input_ptr_ = a_ptr + i * param_->row_ * param_->deep_;
    auto ret = ParallelLaunch(this->ms_context_, Arm64SdotPreRun, this, op_parameter_->thread_num_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Arm64SdotPreRun error: [" << ret << "]";
      return ret;
    }

    batch_weight_ptr_ = b_ptr + i * param_->col_ * param_->deep_;
    batch_b_ptr_ = pack_b_ptr_ + i * param_->col_align_ * param_->deep_align_;
    batch_sums_ = weight_bias_sums_ + i * param_->col_align_;
    batch_c_ptr_ = c_ptr + i * param_->row_ * param_->col_;

    ret = ParallelLaunch(this->ms_context_, Arm64SdotRun, this, thread_count_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Arm64SdotRun error: [" << ret << "]";
      return ret;
    }
  }
  return RET_OK;
}
#endif

int MatmulBaseInt8CPUKernel::Run() {
#if defined(ENABLE_ARM64) && !defined(SUPPORT_NNIE) && !defined(SUPPORT_34XX) && (!defined(MACHINE_LINUX_ARM64))
  if (support_sdot_) {
    return RunArm64Sdot();
  }
#endif
  if (!param_->b_const_) {
    if (TransferB() != RET_OK) {
      MS_LOG(ERROR) << "TransferB error";
      return RET_ERROR;
    }
  }
  int8_t *a_ptr = reinterpret_cast<int8_t *>(in_tensors_.at(0)->data());
  int8_t *c_ptr = reinterpret_cast<int8_t *>(out_tensors_.at(0)->data());
  CHECK_NULL_RETURN(a_ptr);
  CHECK_NULL_RETURN(c_ptr);
  int32_t tmp_weight_zp = filter_per_channel_ ? 1 : quant_param_->filter_zp_[0];
  for (int i = 0; i < param_->batch; i++) {
    auto current_src_a = a_ptr + a_offset_[i] * param_->row_ * param_->deep_;
    if (param_->a_transpose_) {
      MS_CHECK_TRUE_RET(a_pack_func_ != nullptr, RET_ERROR);
      a_pack_func_(current_src_a, pack_a_ptr_, param_->deep_, param_->row_);
      CalcInputSums(current_src_a, param_->row_, param_->deep_, tmp_weight_zp, input_sums_, ColMajor);
    } else {
      MS_CHECK_TRUE_RET(a_pack_func_ != nullptr, RET_ERROR);
      a_pack_func_(current_src_a, pack_a_ptr_, param_->row_, param_->deep_);
      CalcInputSums(current_src_a, param_->row_, param_->deep_, tmp_weight_zp, input_sums_, RowMajor);
    }

    batch_b_ptr_ = pack_b_ptr_ + b_offset_[i] * param_->col_align_ * param_->deep_align_;
    batch_sums_ = weight_bias_sums_ + b_offset_[i] * param_->col_align_;
    batch_c_ptr_ = c_ptr + i * param_->row_ * param_->col_;

    auto ret = ParallelLaunch(this->ms_context_, MatmulBaseInt8Run, this, thread_count_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "MatmulInt8Run error: [" << ret << "]";
      return ret;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
