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

#include "src/litert/kernel/cpu/fp16/lstm_mindir_fp16.h"
#include "nnacl/fp16/lstm_fp16.h"

namespace mindspore::kernel {
namespace {
constexpr size_t kMindirInputTensorNum = 4;
constexpr int kWeightsIndex = 3;
constexpr int kGateNum = 4;
const int kWeightsOrderMap[4] = {0, 2, 3, 1};  // IFGO order to IOFG order
}  // namespace

int LstmMindirFp16CPUKernel::Prepare() {
  CHECK_NOT_EQUAL_RETURN(in_tensors_.size(), kMindirInputTensorNum);
  running_pack_ = trainable_ || !in_tensors_[FOURTH_INPUT]->IsConst();
  if (lstm_param_->bidirectional_) {
    MS_LOG(ERROR) << "LstmMindirFp16CPUKernel doesn't support Bidirection.";
    return lite::RET_NOT_SUPPORT;
  }
  return LstmFp16BaseCPUKernel::Prepare();
}

int LstmMindirFp16CPUKernel::ReSize() {
  auto ret = InitParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmFp16 InitParam failed.";
    return lite::RET_ERROR;
  }
  // determine FB origin
  gpu_orig_state_ = false;
  auto weight_t = in_tensors_.at(kWeightsIndex);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(lstm_param_->hidden_size_, lstm_param_->input_size_, lite::RET_ERROR);
  int hi_unit_size = lstm_param_->hidden_size_ * lstm_param_->input_size_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(weight_segment_num_, hi_unit_size, lite::RET_ERROR);
  int hi_whole_size = weight_segment_num_ * hi_unit_size;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(lstm_param_->hidden_size_, lstm_param_->output_size_, lite::RET_ERROR);
  int hh_unit_size = lstm_param_->hidden_size_ * lstm_param_->output_size_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(weight_segment_num_, hh_unit_size, lite::RET_ERROR);
  int hh_whole_size = weight_segment_num_ * hh_unit_size;
  int scale = lstm_param_->bidirectional_ ? C2NUM : C1NUM;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(lstm_param_->hidden_size_, lstm_param_->project_size_, lite::RET_ERROR);
  int hp_unit_size = lstm_param_->hidden_size_ * lstm_param_->project_size_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(scale, hp_unit_size, lite::RET_ERROR);
  int hp_whole_size = scale * hp_unit_size;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(weight_segment_num_ * C2NUM, lstm_param_->hidden_size_, lite::RET_ERROR);
  int bias_whole_size = weight_segment_num_ * C2NUM * lstm_param_->hidden_size_;
  auto whole_size = weight_t->ElementsNum();
  bool has_bias = (hi_whole_size + hh_whole_size + hp_whole_size < whole_size) ? true : false;
  // if bias exist we can determine the gpu_orig_state_
  if (has_bias) {
    gpu_orig_state_ = (hi_whole_size + hh_whole_size + hp_whole_size + bias_whole_size == whole_size) ? true : false;
  } else {
    bias_whole_size = 0;
  }
  if (gpu_orig_state_) {
    return lite::RET_OK;
  }
  bias_whole_size /= C2NUM;
  if (hi_whole_size + hh_whole_size + hp_whole_size + bias_whole_size != whole_size) {
    MS_LOG(ERROR) << "LstmMindir is invalid when original model exports from CPU.";
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  if (running_pack_) {
    return lite::RET_OK;
  }
  return PackWeightAndBias();
}

int LstmMindirFp16CPUKernel::InitInputWeightBias() {
  auto weight_data = in_tensors_.at(kWeightsIndex)->data();
  CHECK_NULL_RETURN(weight_data);
  auto data_type = in_tensors_.at(kWeightsIndex)->data_type();
  auto ret = PackInputWeight(weight_data, kWeightsOrderMap, data_type);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "LstmMindir fp16 PackInputWeight failed.";
    return ret;
  }
  int hi_unit_size = lstm_param_->input_size_ * lstm_param_->hidden_size_;
  int hh_unit_size = lstm_param_->hidden_size_ * lstm_param_->output_size_;
  int scale = lstm_param_->bidirectional_ ? C2NUM : C1NUM;
  int offset = (weight_segment_num_ * (hi_unit_size + hh_unit_size) +
                scale * lstm_param_->project_size_ * lstm_param_->hidden_size_) *
               lite::DataTypeSize(data_type);
  ret = PackInputBias(reinterpret_cast<uint8_t *>(weight_data) + offset, kWeightsOrderMap, data_type);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "LstmMindir fp16 PackInputBias failed.";
    return ret;
  }
  return lite::RET_OK;
}

int LstmMindirFp16CPUKernel::InitStateWeightBias() {
  auto weight_data = in_tensors_.at(kWeightsIndex)->data();
  CHECK_NULL_RETURN(weight_data);
  auto data_type = in_tensors_.at(kWeightsIndex)->data_type();
  int hi_unit_size = lstm_param_->input_size_ * lstm_param_->hidden_size_;
  auto offset =
    (gpu_orig_state_ ? kGateNum * hi_unit_size : weight_segment_num_ * hi_unit_size) * lite::DataTypeSize(data_type);
  auto ret = PackStateWeight(reinterpret_cast<uint8_t *>(weight_data) + offset, kWeightsOrderMap, data_type);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "LstmMindir fp16 PackStateWeight failed.";
    return ret;
  }

  // state bias
  int hi_whole_size = weight_segment_num_ * lstm_param_->hidden_size_ * lstm_param_->input_size_;
  int hh_whole_size = weight_segment_num_ * lstm_param_->hidden_size_ * lstm_param_->output_size_;
  int proj_size =
    (lstm_param_->bidirectional_ ? C2NUM : C1NUM) * lstm_param_->project_size_ * lstm_param_->hidden_size_;
  offset =
    (hi_whole_size + hh_whole_size + proj_size + lstm_param_->hidden_size_ * kGateNum) * lite::DataTypeSize(data_type);
  ret = PackStateBias(reinterpret_cast<uint8_t *>(weight_data) + offset, kWeightsOrderMap, data_type);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "LstmMindir fp16 PackStateBias failed.";
    return ret;
  }
  return RET_OK;
}

int LstmMindirFp16CPUKernel::InitProjectWeight() {
  if (lstm_param_->project_size_ == 0) {
    return RET_OK;
  }
  auto weight_data = in_tensors_.at(kWeightsIndex)->data();
  CHECK_NULL_RETURN(weight_data);
  auto data_type = in_tensors_.at(kWeightsIndex)->data_type();
  int hi_whole_size = weight_segment_num_ * lstm_param_->hidden_size_ * lstm_param_->input_size_;
  int hh_whole_size = weight_segment_num_ * lstm_param_->hidden_size_ * lstm_param_->output_size_;
  auto offset = (hi_whole_size + hh_whole_size) * lite::DataTypeSize(data_type);
  auto ret = PackProjectWeight(reinterpret_cast<uint8_t *>(weight_data) + offset, nullptr, data_type);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "LstmMindir fp16 PackProjectWeight failed.";
    return ret;
  }
  size_t bias_size = UP_ROUND(lstm_param_->output_size_, C8NUM) * sizeof(float16_t);
  project_bias_ = reinterpret_cast<float16_t *>(malloc(bias_size));
  MS_CHECK_TRUE_MSG(project_bias_ != nullptr, lite::RET_NULL_PTR,
                    "LstmNonMindirCPUKernel malloc project_bias_ failed.");
  pack_buffer_.push_back(project_bias_);
  (void)memset(project_bias_, 0, bias_size);
  return RET_OK;
}
}  // namespace mindspore::kernel
