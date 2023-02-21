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

using mindspore::lite::kCHWDimNumber;
using mindspore::lite::kHWDimNumber;
using mindspore::lite::kNCHWDimNumber;
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
  b_batch_ = batch;
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
  if (pack_b_ptr_ != nullptr && !weight_is_packed_) {
    free(pack_b_ptr_);
    pack_b_ptr_ = nullptr;
  }
  if (input_sums_ != nullptr) {
    free(input_sums_);
    input_sums_ = nullptr;
  }
  if (weight_sums_ != nullptr && !weight_is_packed_) {
    free(weight_sums_);
    weight_sums_ = nullptr;
  }
  if (fp32_bias_ptr_ != nullptr) {
    free(fp32_bias_ptr_);
    fp32_bias_ptr_ = nullptr;
  }
#ifdef ENABLE_FP16
  if (fp16_bias_ptr_ != nullptr) {
    free(fp16_bias_ptr_);
    fp16_bias_ptr_ = nullptr;
  }
#endif
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
  if (weight_is_packed_) {
    CHECK_NULL_RETURN(weight_sums_tensor_);
    pack_b_ptr_ = static_cast<int8_t *>(in_tensors_.at(kWeightIndex)->data());
    weight_sums_ = static_cast<int *>(weight_sums_tensor_->data());
    return RET_OK;
  }
  auto weight_data = reinterpret_cast<int8_t *>(in_tensors_.at(kWeightIndex)->data());
  CHECK_NULL_RETURN(weight_data);
  for (int i = 0; i < b_batch_; i++) {
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
  if (weight_is_packed_) {
    return RET_OK;
  }

  if (pack_b_ptr_ != nullptr) {
    free(pack_b_ptr_);
    pack_b_ptr_ = nullptr;
  }
  pack_b_ptr_ =
    reinterpret_cast<int8_t *>(malloc(b_batch_ * param_->col_align_ * param_->deep_align_ * sizeof(int8_t)));
  if (pack_b_ptr_ == nullptr) {
    FreeTmpBuffer();
    return RET_ERROR;
  }
  if (weight_sums_ != nullptr) {
    free(weight_sums_);
    weight_sums_ = nullptr;
  }
  weight_sums_ = reinterpret_cast<int *>(malloc(b_batch_ * param_->col_align_ * sizeof(int)));
  if (weight_sums_ == nullptr) {
    FreeTmpBuffer();
    return RET_ERROR;
  }
  memset(pack_b_ptr_, 0, b_batch_ * param_->col_align_ * param_->deep_align_ * sizeof(int8_t));
  memset(weight_sums_, 0, b_batch_ * param_->col_align_ * sizeof(int));
  return RET_OK;
}

int MatmulDynamicBaseInt8CPUKernel::CopyBias() {
  if (in_tensors_.size() == kHasBiasSize) {
    CHECK_NULL_RETURN(in_tensors_[kBiasIndex]);
    auto bias_tensor = in_tensors_[kBiasIndex];

#ifdef ENABLE_FP16
    if (enable_fp16_) {
      fp16_bias_ptr_ = static_cast<float16_t *>(malloc(bias_tensor->Size()));
      if (fp16_bias_ptr_ == nullptr) {
        MS_LOG(ERROR) << "Memory allocation failed";
        FreeTmpBuffer();
        return RET_MEMORY_FAILED;
      }
      memcpy(fp16_bias_ptr_, bias_tensor->data(), bias_tensor->ElementsNum() * sizeof(float16_t));
    }
#endif
    if (!enable_fp16_) {
      fp32_bias_ptr_ = static_cast<float *>(malloc(bias_tensor->Size()));
      if (fp32_bias_ptr_ == nullptr) {
        MS_LOG(ERROR) << "Memory allocation failed";
        FreeTmpBuffer();
        return RET_MEMORY_FAILED;
      }
      memcpy(fp32_bias_ptr_, bias_tensor->data(), bias_tensor->ElementsNum() * sizeof(float));
    }
  } else {
    fp32_bias_ptr_ = nullptr;
#ifdef ENABLE_FP16
    fp16_bias_ptr_ = nullptr;
#endif
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
      in_tensors_[1]->data_type() != mindspore::kNumberTypeInt8) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[0]->data_type() << ", input1 data_type is "
                  << in_tensors_[1]->data_type();
    return RET_ERROR;
  }
#ifdef ENABLE_FP16
  enable_fp16_ = ms_context_->device_list_[0].device_info_.cpu_device_info_.enable_float16_;
#endif
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
  // In the framework, the out_tensors data_type is forced to kNumberTypeFloat32
  if (enable_fp16_) {
    out_tensors_[0]->set_data_type(kNumberTypeFloat16);
  }
  auto x_shape = in_tensors_.at(0)->shape();
  auto o_shape = out_tensors_.at(0)->shape();
  MS_ASSERT(o_shape.size() >= kSize2);

  param_->row_ = o_shape[o_shape.size() - kSize2];
  param_->row_align_ = UP_ROUND(param_->row_, row_tile_);
  param_->deep_ = param_->a_transpose_ ? x_shape[x_shape.size() - kSize2] : x_shape[x_shape.size() - kSize1];
  param_->deep_align_ = UP_ROUND(param_->deep_, deep_tile_);

  auto ret = InitBroadcastParams(in_tensors_[kInputIndex]->shape(), in_tensors_[kWeightIndex]->shape(), param_,
                                 &a_offset_, &b_offset_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitBroadcastParams failed.";
    return RET_ERROR;
  }

  ret = InitMatrixABuffer();
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

int MatmulDynamicBaseInt8CPUKernel::InitBroadcastParams(const std::vector<int> &a_shape_const,
                                                        const std::vector<int> &b_shape_const, MatMulParameter *params,
                                                        std::vector<int> *a_offsets, std::vector<int> *b_offsets) {
  std::vector<int> a_shape = a_shape_const;
  if (a_shape.size() < kNCHWDimNumber) {
    size_t add_nums = kNCHWDimNumber - a_shape.size();
    for (size_t i = 0; i < add_nums; ++i) {
      (void)a_shape.insert(a_shape.begin(), 1);
    }
  }
  std::vector<int> b_shape = b_shape_const;
  if (b_shape.size() < kNCHWDimNumber) {
    size_t add_nums = kNCHWDimNumber - b_shape.size();
    for (size_t i = 0; i < add_nums; ++i) {
      (void)b_shape.insert(b_shape.begin(), 1);
    }
  }

  int batch_sizes[MAX_SHAPE_SIZE] = {0};
  int a_batch_sizes[MAX_SHAPE_SIZE] = {0};
  int b_batch_sizes[MAX_SHAPE_SIZE] = {0};
  for (int i = a_shape.size() - kCHWDimNumber; i >= 0; --i) {
    if (static_cast<int>(a_shape.size() - kCHWDimNumber) == i) {
      batch_sizes[i] = std::max(a_shape[i], b_shape[i]);
      a_batch_sizes[i] = a_shape[i];
      b_batch_sizes[i] = b_shape[i];
    } else {
      batch_sizes[i] = batch_sizes[i + 1] * std::max(a_shape[i], b_shape[i]);
      a_batch_sizes[i] = a_batch_sizes[i + 1] * a_shape[i];
      b_batch_sizes[i] = b_batch_sizes[i + 1] * b_shape[i];
    }
  }

  int out_batch = 1;
  for (size_t i = 0; i < a_shape.size() - kHWDimNumber; ++i) {
    int max_v = MSMAX(a_shape[i], b_shape[i]);
    int min_v = MSMIN(a_shape[i], b_shape[i]) > 0 ? MSMIN(a_shape[i], b_shape[i]) : 1;
    out_batch *= max_v;
    if (max_v != min_v && max_v % min_v != 0) {
      MS_LOG(ERROR) << "matmul don't support broadcast for dimension " << a_shape << " and " << b_shape;
      return RET_ERROR;
    }
  }
  params->batch = out_batch;

  a_offsets->resize(params->batch, 0);
  b_offsets->resize(params->batch, 0);
  for (int i = 0; i < params->batch; ++i) {
    int64_t delta = i;
    int a_offset = 0;
    int b_offset = 0;
    for (size_t j = 0; j < a_shape.size() - kHWDimNumber; ++j) {
      if (j > 0) {
        delta = delta % batch_sizes[j];
      }
      if (j < (a_shape.size() - kCHWDimNumber)) {
        a_offset += (delta / batch_sizes[j + 1] * a_shape[j] / std::max(a_shape[j], b_shape[j])) * a_batch_sizes[j + 1];
        b_offset += (delta / batch_sizes[j + 1] * b_shape[j] / std::max(a_shape[j], b_shape[j])) * b_batch_sizes[j + 1];
      } else {
        a_offset += (delta * a_shape[j] / std::max(a_shape[j], b_shape[j]));
        b_offset += (delta * b_shape[j] / std::max(a_shape[j], b_shape[j]));
      }
    }
    (*a_offsets)[i] = a_offset;
    (*b_offsets)[i] = b_offset;
  }

  return RET_OK;
}
}  // namespace mindspore::kernel
