/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/int8/matmul_dynamic_base_int8.h"
#include "nnacl/int8/dynamic_matmul_int8.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
constexpr int kHasBiasSize = 3;
constexpr int kMinInputSize = 2;
constexpr int kOutputSize = 1;
constexpr int kSize1 = 1;
constexpr int kSize2 = 2;
}  // namespace

MatmulDynamicBaseInt8CPUKernel::~MatmulDynamicBaseInt8CPUKernel() {
  FreeQuantParam();
  FreeTmpBuffer();
}

void MatmulDynamicBaseInt8CPUKernel::FreeQuantParam() {
  if (quant_param_ != nullptr) {
    if (quant_param_->filter_scale_ != nullptr) {
      free(quant_param_->filter_scale_);
      quant_param_->filter_scale_ = nullptr;
    }
    if (quant_param_->filter_zp_ != nullptr) {
      free(quant_param_->filter_zp_);
      quant_param_->filter_zp_ = nullptr;
    }
    free(quant_param_);
    quant_param_ = nullptr;
  }
}

int MatmulDynamicBaseInt8CPUKernel::MallocQuantParam() {
  quant_param_ = reinterpret_cast<MatmulDynamicQuantParameter *>(malloc(sizeof(MatmulQuantParameter)));
  if (quant_param_ == nullptr) {
    MS_LOG(ERROR) << "Malloc MatmulDynamicQuantParameter for Matmul int8 op failed!";
    return RET_ERROR;
  }
  memset(quant_param_, 0, sizeof(MatmulQuantParameter));
  return RET_OK;
}

int MatmulDynamicBaseInt8CPUKernel::InitFilterQuantParam() {
  if (quant_param_->filter_scale_ != nullptr) {
    free(quant_param_->filter_scale_);
    quant_param_->filter_scale_ = nullptr;
  }
  if (quant_param_->filter_zp_ != nullptr) {
    free(quant_param_->filter_zp_);
    quant_param_->filter_zp_ = nullptr;
  }

  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto weight_quant_params = weight_tensor->quant_params();
  auto w_shape = weight_tensor->shape();
  if (w_shape.size() < DIMENSION_2D) {
    MS_LOG(ERROR) << weight_tensor->tensor_name() << " dims < 2.";
    return RET_ERROR;
  }
  int col = param_->b_transpose_ ? w_shape[w_shape.size() - kSize2] : w_shape[w_shape.size() - kSize1];
  filter_per_channel_ = (weight_quant_params.size() > 1);
  channel_num_ = filter_per_channel_ ? col : 1;
  if (static_cast<int>(weight_quant_params.size()) != channel_num_) {
    MS_LOG(ERROR) << weight_tensor->tensor_name() << " quant params size:" << weight_quant_params.size()
                  << " != channel_num_:" << channel_num_;
    return RET_ERROR;
  }
  quant_param_->filter_scale_ = reinterpret_cast<float *>(malloc(channel_num_ * sizeof(float)));
  CHECK_NULL_RETURN(quant_param_->filter_scale_);
  memset(quant_param_->filter_scale_, 0, sizeof(channel_num_));
  quant_param_->filter_zp_ = reinterpret_cast<int32_t *>(malloc(channel_num_ * sizeof(int32_t)));
  CHECK_NULL_RETURN(quant_param_->filter_zp_);
  memset(quant_param_->filter_zp_, 0, sizeof(channel_num_));

  for (int i = 0; i < channel_num_; i++) {
    quant_param_->filter_scale_[i] = static_cast<float>(weight_quant_params[i].scale);
    quant_param_->filter_zp_[i] = weight_quant_params[i].zeroPoint;
  }
  return RET_OK;
}

void MatmulDynamicBaseInt8CPUKernel::ResizeMatrixBParameter() {
  auto w_shape = in_tensors_.at(kWeightIndex)->shape();
  int batch = 1;
  for (size_t i = 0; i < w_shape.size() - kSize2; ++i) {
    batch *= w_shape[i];
  }
  param_->batch = batch;
  param_->col_ = param_->b_transpose_ ? w_shape[w_shape.size() - kSize2] : w_shape[w_shape.size() - kSize1];
  param_->deep_ = param_->b_transpose_ ? w_shape[w_shape.size() - kSize1] : w_shape[w_shape.size() - kSize2];

  param_->col_align_ = UP_ROUND(param_->col_, col_tile_);
  param_->deep_align_ = UP_ROUND(param_->deep_, deep_tile_);

  thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(param_->col_align_, col_tile_));
  thread_stride_ = UP_DIV(UP_DIV(param_->col_align_, col_tile_), thread_count_);
  return;
}

void MatmulDynamicBaseInt8CPUKernel::FreeTmpBuffer() {
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
  if (weight_sums_ != nullptr) {
    free(weight_sums_);
    weight_sums_ = nullptr;
  }
  if (fp32_bias_ptr_ != nullptr) {
    free(fp32_bias_ptr_);
    fp32_bias_ptr_ = nullptr;
  }
  return;
}

int MatmulDynamicBaseInt8CPUKernel::InitInputQuantParam() {
  auto in_quant_params = in_tensors_.at(kInputIndex)->quant_params();
  if (in_quant_params.empty()) {
    MS_LOG(ERROR) << "invalid in quant param";
    return RET_ERROR;
  }
  quant_param_->input_zp_ = in_quant_params.front().zeroPoint;
  quant_param_->input_scale_ = static_cast<float>(in_quant_params.front().scale);
  return RET_OK;
}

int MatmulDynamicBaseInt8CPUKernel::TransferB() {
  auto weight_data = reinterpret_cast<int8_t *>(in_tensors_.at(kWeightIndex)->data());
  CHECK_NULL_RETURN(weight_data);
  for (int i = 0; i < param_->batch; i++) {
    auto current_weight = weight_data + i * param_->deep_ * param_->col_;
    auto current_b_pack = pack_b_ptr_ + i * param_->col_align_ * param_->deep_align_;
    auto current_sums = weight_sums_ + i * param_->col_align_;
    CHECK_NULL_RETURN(b_pack_func_);
    if (param_->b_transpose_) {
      b_pack_func_(current_weight, current_b_pack, param_->col_, param_->deep_);
      CalcWeightSums(current_weight, param_->deep_, param_->col_, current_sums, ColMajor);
    } else {
      b_pack_func_(current_weight, current_b_pack, param_->deep_, param_->col_);
      CalcWeightSums(current_weight, param_->deep_, param_->col_, current_sums, RowMajor);
    }
  }
  return RET_OK;
}

int MatmulDynamicBaseInt8CPUKernel::InitMatrixABuffer() {
  if (pack_a_ptr_ != nullptr) {
    free(pack_a_ptr_);
    pack_a_ptr_ = nullptr;
  }
  pack_a_ptr_ = reinterpret_cast<int8_t *>(malloc(param_->row_align_ * param_->deep_align_ * sizeof(int8_t)));
  if (pack_a_ptr_ == nullptr) {
    FreeTmpBuffer();
    return RET_ERROR;
  }
  if (input_sums_ != nullptr) {
    free(input_sums_);
    input_sums_ = nullptr;
  }
  input_sums_ = reinterpret_cast<int *>(malloc(param_->row_align_ * sizeof(int)));
  if (input_sums_ == nullptr) {
    FreeTmpBuffer();
    return RET_ERROR;
  }
  memset(pack_a_ptr_, 0, param_->row_align_ * param_->deep_align_ * sizeof(int8_t));
  memset(input_sums_, 0, param_->row_align_ * sizeof(int));
  return RET_OK;
}

int MatmulDynamicBaseInt8CPUKernel::InitMatrixBBuffer() {
  if (pack_b_ptr_ != nullptr) {
    free(pack_b_ptr_);
    pack_b_ptr_ = nullptr;
  }
  pack_b_ptr_ =
    reinterpret_cast<int8_t *>(malloc(param_->batch * param_->col_align_ * param_->deep_align_ * sizeof(int8_t)));
  if (pack_b_ptr_ == nullptr) {
    FreeTmpBuffer();
    return RET_ERROR;
  }
  if (weight_sums_ != nullptr) {
    free(weight_sums_);
    weight_sums_ = nullptr;
  }
  weight_sums_ = reinterpret_cast<int *>(malloc(param_->batch * param_->col_align_ * sizeof(int)));
  if (weight_sums_ == nullptr) {
    FreeTmpBuffer();
    return RET_ERROR;
  }
  memset(pack_b_ptr_, 0, param_->batch * param_->col_align_ * param_->deep_align_ * sizeof(int8_t));
  memset(weight_sums_, 0, param_->batch * param_->col_align_ * sizeof(int));
  return RET_OK;
}

int MatmulDynamicBaseInt8CPUKernel::CopyBias() {
  if (in_tensors_.size() == kHasBiasSize) {
    CHECK_NULL_RETURN(in_tensors_[kBiasIndex]);
    auto bias_tensor = in_tensors_[kBiasIndex];
    fp32_bias_ptr_ = static_cast<float *>(malloc(bias_tensor->Size()));
    if (fp32_bias_ptr_ == nullptr) {
      MS_LOG(ERROR) << "Memory allocation failed";
      FreeTmpBuffer();
      return RET_MEMORY_FAILED;
    }
    memcpy(fp32_bias_ptr_, bias_tensor->data(), bias_tensor->ElementsNum() * sizeof(float));
  } else {
    fp32_bias_ptr_ = nullptr;
  }
  return RET_OK;
}

int MatmulDynamicBaseInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), kMinInputSize);
  CHECK_LESS_RETURN(out_tensors_.size(), kOutputSize);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(in_tensors_[1]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  if (in_tensors_[0]->data_type() != mindspore::kNumberTypeInt8 ||
      in_tensors_[1]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[0]->data_type() != mindspore::kNumberTypeFloat32) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[0]->data_type() << ", input1 data_type is "
                  << in_tensors_[1]->data_type() << ", output data_type is " << out_tensors_[0]->data_type();
    return RET_ERROR;
  }
  InitParameter();
  auto ret = MallocQuantParam();
  if (ret != RET_OK) {
    FreeQuantParam();
    return ret;
  }
  if (param_->b_const_) {
    ResizeMatrixBParameter();
    ret = InitFilterQuantParam();
    if (ret != RET_OK) {
      FreeQuantParam();
      return ret;
    }
    ret = InitMatrixBBuffer();
    if (ret != RET_OK) {
      FreeQuantParam();
      return ret;
    }

    ret = TransferB();
    if (ret != RET_OK) {
      FreeQuantParam();
      return ret;
    }
  }

  ret = CopyBias();
  if (ret != RET_OK) {
    FreeQuantParam();
    return ret;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int MatmulDynamicBaseInt8CPUKernel::ReSize() {
  auto x_shape = in_tensors_.at(0)->shape();
  auto y_shape = in_tensors_.at(1)->shape();
  auto o_shape = out_tensors_.at(0)->shape();
  MS_ASSERT(o_shape.size() >= kSize2);
  unsigned int i = 0;
  param_->row_ = param_->a_transpose_ ? x_shape[x_shape.size() - kSize1] : x_shape[x_shape.size() - kSize2];
  param_->batch = 1;
  for (; i < x_shape.size() - kSize2; i++) {
    if (x_shape[i] != y_shape[i]) {
      break;
    }
    param_->batch *= x_shape[i];
  }
  for (; i < x_shape.size() - kSize2; i++) {
    param_->row_ *= x_shape[i];
  }

  param_->row_align_ = UP_ROUND(param_->row_, row_tile_);
  param_->deep_ = param_->a_transpose_ ? x_shape[x_shape.size() - kSize2] : x_shape[x_shape.size() - kSize1];
  param_->deep_align_ = UP_ROUND(param_->deep_, deep_tile_);

  auto ret = InitMatrixABuffer();
  if (ret != RET_OK) {
    FreeQuantParam();
    return ret;
  }

  if (!param_->b_const_) {
    ResizeMatrixBParameter();
    ret = InitMatrixBBuffer();
    if (ret != RET_OK) {
      FreeQuantParam();
      return ret;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
