/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp16/lstm_fp16_base.h"
#include <cfloat>
#include "nnacl/fp16/lstm_fp16.h"
#include "nnacl/fp16/cast_fp16.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
constexpr int kGateNum = 4;
constexpr int kTempInputBufferIndex = 0;
constexpr int kTempInputGateBufferIndex = 1;
constexpr int kTempStateBufferIndex = 2;
constexpr int kTempStateGateBufferIndex = 3;
constexpr int kTempCellStateBufferIndex = 4;
constexpr int kTempHiddenStateBufferIndex = 5;
constexpr int kTempProjectInputBufferIndex = 6;
}  // namespace

LstmFp16BaseCPUKernel::~LstmFp16BaseCPUKernel() { FreePackBuffer(); }

int LstmFp16BaseCPUKernel::Prepare() {
  for (size_t i = 0; i < in_tensors_.size(); ++i) {
    CHECK_NULL_RETURN(in_tensors_[i]);
  }
  CHECK_LESS_RETURN(out_tensors_.size(), C3NUM);
  for (size_t i = 0; i < out_tensors_.size(); ++i) {
    CHECK_NULL_RETURN(out_tensors_[i]);
  }
  CHECK_NULL_RETURN(lstm_param_);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int LstmFp16BaseCPUKernel::ReSize() {
  auto ret = InitParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmFp16 InitParam failed.";
    return RET_ERROR;
  }
  if (running_pack_) {
    return RET_OK;
  }
  return PackWeightAndBias();
}

int LstmFp16BaseCPUKernel::Run() {
  auto input_ptr = reinterpret_cast<float16_t *>(in_tensors_[FIRST_INPUT]->data());
  CHECK_NULL_RETURN(input_ptr);
  auto output_ptr = reinterpret_cast<float16_t *>(out_tensors_[FIRST_INPUT]->data());
  CHECK_NULL_RETURN(output_ptr);

  auto hidden_init = in_tensors_[hidden_init_index_]->data();
  CHECK_NULL_RETURN(hidden_init);
  auto cell_init = in_tensors_[cell_init_index_]->data();
  CHECK_NULL_RETURN(cell_init);

  auto output_hidden = out_tensors_[SECOND_INPUT]->data();
  CHECK_NULL_RETURN(output_hidden);
  (void)memcpy(output_hidden, hidden_init, in_tensors_[hidden_init_index_]->ElementsNum() * sizeof(float16_t));
  auto output_cell = out_tensors_[THIRD_INPUT]->data();
  CHECK_NULL_RETURN(output_cell);
  (void)memcpy(output_cell, cell_init, in_tensors_[cell_init_index_]->ElementsNum() * sizeof(float16_t));

  if (running_pack_) {
    auto ret = PackWeightAndBias();
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "LstmFp16 PackWeightAndBias failed.";
      return ret;
    }
  }
  auto ret = MallocRunBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmFp16CPUKernel MallocRunBuffer error.";
    FreeRunBuffer();
    if (running_pack_) {
      FreePackBuffer();
    }
    return RET_ERROR;
  }
  LstmFp16(output_ptr, input_ptr, weight_i_ptr_, weight_h_ptr_, input_bias_, state_bias_, weight_project_ptr_,
           project_bias_, reinterpret_cast<float16_t *>(output_hidden), reinterpret_cast<float16_t *>(output_cell),
           running_buffer_, lstm_param_);
  FreeRunBuffer();
  if (running_pack_) {
    FreePackBuffer();
  }
  return RET_OK;
}

int LstmFp16BaseCPUKernel::InitParam() {
  auto in_shape = in_tensors_[FIRST_INPUT]->shape();
  MS_CHECK_TRUE_MSG(in_shape.size() == C3NUM, lite::RET_INPUT_TENSOR_ERROR,
                    "The dims of LSTM's first input must be 3.");
  lstm_param_->seq_len_ = in_shape[0];
  lstm_param_->batch_ = in_shape[1];
  lstm_param_->input_size_ = in_shape.back();

  auto h_init_shape = in_tensors_.at(hidden_init_index_)->shape();
  auto c_init_shape = in_tensors_.at(cell_init_index_)->shape();
  lstm_param_->hidden_size_ = c_init_shape.back();
  lstm_param_->output_size_ = h_init_shape.back();

  lstm_param_->output_step_ = lstm_param_->bidirectional_ ? C2NUM * lstm_param_->batch_ * lstm_param_->output_size_
                                                          : lstm_param_->batch_ * lstm_param_->output_size_;
  weight_segment_num_ = lstm_param_->bidirectional_ ? C2NUM * kGateNum : kGateNum;
#ifdef ENABLE_ARM64
  lstm_param_->input_row_align_ = UP_ROUND(lstm_param_->seq_len_ * lstm_param_->batch_, C1NUM);
  lstm_param_->input_col_align_ = UP_ROUND(lstm_param_->hidden_size_, C4NUM);

  lstm_param_->state_row_align_ = UP_ROUND(lstm_param_->batch_, C1NUM);
  lstm_param_->state_col_align_ = UP_ROUND(lstm_param_->hidden_size_, C4NUM);
  lstm_param_->proj_col_align_ = UP_ROUND(lstm_param_->output_size_, C4NUM);
  weight_need_pack_ = true;
#else
  lstm_param_->input_row_align_ = UP_ROUND(lstm_param_->seq_len_ * lstm_param_->batch_, C16NUM);
  lstm_param_->input_col_align_ = UP_ROUND(lstm_param_->hidden_size_, C8NUM);

  lstm_param_->state_row_align_ =
    lstm_param_->batch_ == 1 ? lstm_param_->batch_ : UP_ROUND(lstm_param_->batch_, C16NUM);
  lstm_param_->state_col_align_ =
    lstm_param_->batch_ == 1 ? lstm_param_->hidden_size_ : UP_ROUND(lstm_param_->hidden_size_, C8NUM);
  lstm_param_->proj_col_align_ =
    lstm_param_->batch_ == 1 ? lstm_param_->output_size_ : UP_ROUND(lstm_param_->output_size_, C8NUM);
  weight_need_pack_ = lstm_param_->batch_ != 1;
#endif
  return RET_OK;
}

int LstmFp16BaseCPUKernel::PackWeightAndBias() {
  FreePackBuffer();
  auto ret = InitInputWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmFp16 InitInputWeightBias failed.";
    FreePackBuffer();
    return RET_ERROR;
  }

  ret = InitStateWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmFp16 InitStateWeightBias failed.";
    FreePackBuffer();
    return RET_ERROR;
  }

  ret = InitProjectWeight();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmFp16 InitProjectWeight failed.";
    FreePackBuffer();
    return RET_ERROR;
  }
  return RET_OK;
}

int LstmFp16BaseCPUKernel::PackInputWeight(const void *src, const int32_t *order, TypeId src_data_type) {
  weight_i_ptr_ = reinterpret_cast<float16_t *>(
    malloc(weight_segment_num_ * lstm_param_->input_col_align_ * lstm_param_->input_size_ * sizeof(float16_t)));
  MS_CHECK_TRUE_MSG(weight_i_ptr_ != nullptr, lite::RET_NULL_PTR, "LstmCPUKernel fp16 malloc weight_i_ptr_ failed.");
  pack_buffer_.push_back(weight_i_ptr_);
  if (src_data_type == kNumberTypeFloat32) {
    PackLstmWeightFp32ToFp16(weight_i_ptr_, reinterpret_cast<const float *>(src), weight_segment_num_,
                             lstm_param_->input_size_, lstm_param_->hidden_size_, lstm_param_->input_col_align_, order);
  } else if (src_data_type == kNumberTypeFloat16) {
    PackLstmWeightFp16(weight_i_ptr_, reinterpret_cast<const float16_t *>(src), weight_segment_num_,
                       lstm_param_->input_size_, lstm_param_->hidden_size_, lstm_param_->input_col_align_, order);
  } else {
    MS_LOG(ERROR) << "Unsupported data type of weight_i tensor for lstm.";
    return RET_ERROR;
  }
  return lite::RET_OK;
}

int LstmFp16BaseCPUKernel::PackInputBias(const void *src, const int32_t *order, TypeId src_data_type) {
  auto bias_size = weight_segment_num_ * lstm_param_->input_col_align_ * sizeof(float16_t);
  input_bias_ = reinterpret_cast<float16_t *>(malloc(bias_size));
  MS_CHECK_TRUE_MSG(input_bias_ != nullptr, lite::RET_NULL_PTR, "LstmCPUKernel fp16 malloc input_bias_ failed.");
  pack_buffer_.push_back(input_bias_);
  (void)memset(input_bias_, 0, bias_size);
  if (!lstm_param_->has_bias_) {
    return lite::RET_OK;
  }
  if (src_data_type == kNumberTypeFloat32) {
    PackLstmBiasFp32ToFp16(input_bias_, reinterpret_cast<const float *>(src), weight_segment_num_,
                           lstm_param_->hidden_size_, lstm_param_->input_col_align_, lstm_param_->bidirectional_,
                           order);
  } else if (src_data_type == kNumberTypeFloat16) {
    PackLstmBiasFp16(input_bias_, reinterpret_cast<const float16_t *>(src), weight_segment_num_,
                     lstm_param_->hidden_size_, lstm_param_->input_col_align_, lstm_param_->bidirectional_, order);
  } else {
    MS_LOG(ERROR) << "Unsupported data type of bias tensor for lstm.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

int LstmFp16BaseCPUKernel::PackStateWeight(const void *src, const int32_t *order, TypeId src_data_type) {
  weight_h_ptr_ = reinterpret_cast<float16_t *>(
    malloc(weight_segment_num_ * lstm_param_->state_col_align_ * lstm_param_->output_size_ * sizeof(float16_t)));
  MS_CHECK_TRUE_MSG(weight_h_ptr_ != nullptr, lite::RET_NULL_PTR, "LstmCPUKernel fp16 malloc weight_h_ptr_ failed.");
  pack_buffer_.push_back(weight_h_ptr_);
  if (weight_need_pack_) {
    if (src_data_type == kNumberTypeFloat32) {
      PackLstmWeightFp32ToFp16(weight_h_ptr_, reinterpret_cast<const float *>(src), weight_segment_num_,
                               lstm_param_->output_size_, lstm_param_->hidden_size_, lstm_param_->state_col_align_,
                               order);
    } else if (src_data_type == kNumberTypeFloat16) {
      PackLstmWeightFp16(weight_h_ptr_, reinterpret_cast<const float16_t *>(src), weight_segment_num_,
                         lstm_param_->output_size_, lstm_param_->hidden_size_, lstm_param_->state_col_align_, order);
    } else {
      MS_LOG(ERROR) << "Unsupported data type of weight_h tensor for lstm.";
      return RET_ERROR;
    }
  } else {
    auto element_num = weight_segment_num_ * lstm_param_->hidden_size_ * lstm_param_->output_size_;
    if (src_data_type == kNumberTypeFloat32) {
      Float32ToFloat16(reinterpret_cast<const float *>(src), weight_h_ptr_, element_num);
    } else if (src_data_type == kNumberTypeFloat16) {
      (void)memcpy(weight_h_ptr_, src, element_num * sizeof(float16_t));
    } else {
      MS_LOG(ERROR) << "Unsupported data type of weight_h tensor for lstm.";
      return RET_ERROR;
    }
  }
  return lite::RET_OK;
}

int LstmFp16BaseCPUKernel::PackStateBias(const void *src, const int32_t *order, TypeId src_data_type) {
  state_bias_ =
    reinterpret_cast<float16_t *>(malloc(weight_segment_num_ * lstm_param_->state_col_align_ * sizeof(float16_t)));
  MS_CHECK_TRUE_MSG(state_bias_ != nullptr, lite::RET_NULL_PTR, "LstmCPUKernel fp16 malloc state_bias_ failed.");
  (void)memset(state_bias_, 0, weight_segment_num_ * lstm_param_->state_col_align_ * sizeof(float16_t));
  pack_buffer_.push_back(state_bias_);
  if (!lstm_param_->has_bias_) {
    return RET_OK;
  }
  if (src_data_type == kNumberTypeFloat32) {
    PackLstmBiasFp32ToFp16(state_bias_, reinterpret_cast<const float *>(src), weight_segment_num_,
                           lstm_param_->hidden_size_, lstm_param_->state_col_align_, lstm_param_->bidirectional_,
                           order);
  } else if (src_data_type == kNumberTypeFloat16) {
    PackLstmBiasFp16(state_bias_, reinterpret_cast<const float16_t *>(src), weight_segment_num_,
                     lstm_param_->hidden_size_, lstm_param_->state_col_align_, lstm_param_->bidirectional_, order);
  } else {
    MS_LOG(ERROR) << "Unsupported data type of bias tensor for lstm.";
    return RET_ERROR;
  }
  return RET_OK;
}

int LstmFp16BaseCPUKernel::PackProjectWeight(const void *src, const int32_t *order, TypeId src_data_type) {
  int batch = lstm_param_->bidirectional_ ? C2NUM : C1NUM;
  weight_project_ptr_ = reinterpret_cast<float16_t *>(
    malloc(batch * lstm_param_->hidden_size_ * lstm_param_->proj_col_align_ * sizeof(float16_t)));
  MS_CHECK_TRUE_MSG(weight_project_ptr_ != nullptr, lite::RET_NULL_PTR,
                    "LstmNonMindirCPUKernel malloc weight_project_ptr_ failed.");
  pack_buffer_.push_back(weight_project_ptr_);

  if (weight_need_pack_) {
    if (src_data_type == kNumberTypeFloat32) {
      PackLstmWeightFp32ToFp16(weight_project_ptr_, reinterpret_cast<const float *>(src), batch,
                               lstm_param_->hidden_size_, lstm_param_->output_size_, lstm_param_->proj_col_align_,
                               order);
    } else if (src_data_type == kNumberTypeFloat16) {
      PackLstmWeightFp16(weight_project_ptr_, reinterpret_cast<const float16_t *>(src), batch,
                         lstm_param_->hidden_size_, lstm_param_->output_size_, lstm_param_->proj_col_align_, order);
    } else {
      MS_LOG(ERROR) << "Unsupported data type of weight_project tensor for lstm.";
      return RET_ERROR;
    }
  } else {
    auto element_num = batch * lstm_param_->hidden_size_ * lstm_param_->project_size_;
    if (src_data_type == kNumberTypeFloat32) {
      Float32ToFloat16(reinterpret_cast<const float *>(src), weight_project_ptr_, element_num);
    } else if (src_data_type == kNumberTypeFloat16) {
      (void)memcpy(weight_project_ptr_, src, element_num * sizeof(float16_t));
    } else {
      MS_LOG(ERROR) << "Unsupported data type of weight_project tensor for lstm.";
      return RET_ERROR;
    }
  }
  return lite::RET_OK;
}

void LstmFp16BaseCPUKernel::FreePackBuffer() {
  for (auto buffer : pack_buffer_) {
    if (buffer) {
      free(buffer);
    }
  }
  pack_buffer_.clear();
}

int LstmFp16BaseCPUKernel::MallocRunBuffer() {
  for (int i = 0; i < C7NUM; i++) {
    running_buffer_[i] = nullptr;
  }
  bool need_pack_input = true;
#ifdef ENABLE_ARM64
  need_pack_input = lstm_param_->seq_len_ * lstm_param_->batch_ >= C4NUM;
#endif
  if (need_pack_input) {
    running_buffer_[kTempInputBufferIndex] = reinterpret_cast<float16_t *>(
      ms_context_->allocator->Malloc(lstm_param_->input_row_align_ * lstm_param_->input_size_ * sizeof(float16_t)));
    if (running_buffer_[kTempInputBufferIndex] == nullptr) {
      MS_LOG(ERROR) << "LstmFp16CPUKernel malloc input * weight left matirx error.";
      return RET_ERROR;
    }
  }

  running_buffer_[kTempInputGateBufferIndex] = reinterpret_cast<float16_t *>(ms_context_->allocator->Malloc(
    kGateNum * lstm_param_->seq_len_ * lstm_param_->batch_ * lstm_param_->hidden_size_ * sizeof(float16_t)));
  if (running_buffer_[kTempInputGateBufferIndex] == nullptr) {
    MS_LOG(ERROR) << "LstmFp16CPUKernel malloc state * weight left matirx error.";
    return RET_ERROR;
  }

#ifdef ENABLE_ARM64
  need_pack_input = lstm_param_->batch_ >= C4NUM;
#else
  need_pack_input = lstm_param_->batch_ != 1;
#endif
  if (need_pack_input) {
    running_buffer_[kTempStateBufferIndex] = reinterpret_cast<float16_t *>(
      ms_context_->allocator->Malloc(lstm_param_->state_row_align_ * lstm_param_->output_size_ * sizeof(float16_t)));
    if (running_buffer_[kTempStateBufferIndex] == nullptr) {
      MS_LOG(ERROR) << "LstmFp16CPUKernel malloc state * weight left matirx error.";
      return RET_ERROR;
    }
  }

  running_buffer_[kTempStateGateBufferIndex] = reinterpret_cast<float16_t *>(
    ms_context_->allocator->Malloc(kGateNum * lstm_param_->batch_ * lstm_param_->hidden_size_ * sizeof(float16_t)));
  if (running_buffer_[kTempStateGateBufferIndex] == nullptr) {
    MS_LOG(ERROR) << "LstmFp16CPUKernel malloc state gate buffer_ error.";
    return RET_ERROR;
  }

  if (!(lstm_param_->zoneout_cell_ >= -FLT_EPSILON && lstm_param_->zoneout_cell_ <= FLT_EPSILON)) {
    int buffer_size = lstm_param_->batch_ * lstm_param_->hidden_size_ * sizeof(float16_t);
    running_buffer_[kTempCellStateBufferIndex] =
      reinterpret_cast<float16_t *>(ms_context_->allocator->Malloc(buffer_size));
    if (running_buffer_[kTempCellStateBufferIndex] == nullptr) {
      MS_LOG(ERROR) << "LstmFp16CPUKernel malloc state_buffer for cell error.";
      return RET_ERROR;
    }
  }
  if (!(lstm_param_->zoneout_hidden_ >= -FLT_EPSILON && lstm_param_->zoneout_hidden_ <= FLT_EPSILON)) {
    int buffer_size = lstm_param_->batch_ * lstm_param_->output_size_ * sizeof(float16_t);
    running_buffer_[kTempHiddenStateBufferIndex] =
      reinterpret_cast<float16_t *>(ms_context_->allocator->Malloc(buffer_size));
    if (running_buffer_[kTempHiddenStateBufferIndex] == nullptr) {
      MS_LOG(ERROR) << "LstmFp16CPUKernel malloc state_buffer for hidden error.";
      return RET_ERROR;
    }
  }
  if (need_pack_input && in_tensors_.size() == C7NUM) {
    running_buffer_[kTempProjectInputBufferIndex] = reinterpret_cast<float16_t *>(
      ms_context_->allocator->Malloc(lstm_param_->state_row_align_ * lstm_param_->hidden_size_ * sizeof(float16_t)));
    if (running_buffer_[kTempProjectInputBufferIndex] == nullptr) {
      MS_LOG(ERROR) << "LstmFp16CPUKernel malloc project_buffer for hidden error.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

void LstmFp16BaseCPUKernel::FreeRunBuffer() {
  for (int i = 0; i < C7NUM; ++i) {
    if (running_buffer_[i] != nullptr) {
      ms_context_->allocator->Free(running_buffer_[i]);
      running_buffer_[i] = nullptr;
    }
  }
}
}  // namespace mindspore::kernel
