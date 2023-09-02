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

#include "src/litert/kernel/cpu/fp16/lstm_non_mindir_fp16.h"
#include "nnacl/fp16/lstm_fp16.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
constexpr int kGateNum = 4;
constexpr size_t kInputTensorNumMin = 6;
}  // namespace

int LstmNonMindirFp16CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), kInputTensorNumMin);
  running_pack_ = train_mode_;
  for (size_t i = 1; i <= FOURTH_INPUT; ++i) {
    running_pack_ = running_pack_ || !in_tensors_[i]->IsConst();
  }
  return LstmFp16BaseCPUKernel::Prepare();
}

int LstmNonMindirFp16CPUKernel::ReSize() {
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

int LstmNonMindirFp16CPUKernel::InitInputWeightBias() {
  // malloc and init input * weight right matrix buffer
  // input -- row: seq_len * batch; col: input_size
  // weight -- row: hidden_size; col: input_size, need transpose
  // result -- row: seq_len * batch; col: hidden_size
  auto weight_i = in_tensors_.at(1);
  auto weight_i_data = weight_i->data();
  CHECK_NULL_RETURN(weight_i_data);
  auto ret = PackInputWeight(weight_i_data, nullptr, weight_i->data_type());
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "LstmNonMindir fp16 PackInputWeight failed.";
    return ret;
  }
  // input bias
  auto bias = in_tensors_.at(FOURTH_INPUT);
  auto bias_data = bias->data();
  CHECK_NULL_RETURN(bias_data);
  lstm_param_->has_bias_ = true;
  ret = PackInputBias(bias_data, nullptr, bias->data_type());
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "LstmNonMindir fp16 PackInputBias failed.";
    return ret;
  }
  return RET_OK;
}

int LstmNonMindirFp16CPUKernel::InitStateWeightBias() {
  // malloc and init state * weight right matrix buffer, state * weight will be executed seq_len_ times.
  // state -- row: batch; col: hidden_size
  // weight -- row: hidden_size; col: hidden_size, need transpose
  // result -- row: batch; col: hidden_size
  auto weight_h = in_tensors_.at(THIRD_INPUT);
  auto weight_h_data = weight_h->data();
  CHECK_NULL_RETURN(weight_h_data);
  auto ret = PackStateWeight(weight_h_data, nullptr, weight_h->data_type());
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "LstmNonMindir fp16 PackStateWeight failed.";
    return ret;
  }

  // state bias
  auto bias = in_tensors_[FOURTH_INPUT];
  auto bias_data = bias->data();
  CHECK_NULL_RETURN(bias_data);
  lstm_param_->has_bias_ = true;
  auto offset = kGateNum * lstm_param_->hidden_size_ * lite::DataTypeSize(bias->data_type());
  ret = PackStateBias(reinterpret_cast<uint8_t *>(bias_data) + offset, nullptr, bias->data_type());
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "LstmNonMindir fp16 PackStateBias failed.";
    return ret;
  }
  return RET_OK;
}

int LstmNonMindirFp16CPUKernel::InitProjectWeight() {
  if (in_tensors_.size() < C7NUM) {
    return RET_OK;
  }
  auto weight_pro = in_tensors_[SEVENTH_INPUT];
  auto shape = weight_pro->shape();
  MS_CHECK_TRUE_MSG(shape.size() == C3NUM, lite::RET_ERROR, "Project-weight's shape must be 3D.");
  auto weight_pro_data = weight_pro->data();
  CHECK_NULL_RETURN(weight_pro_data);
  int batch = lstm_param_->bidirectional_ ? C2NUM : C1NUM;
  if (shape[0] != batch) {
    MS_LOG(ERROR) << "Project-weight's shape[0] must be 1(bidirectional=false) or 2(bidirectional=true).";
    return RET_ERROR;
  }
  auto ret = PackProjectWeight(weight_pro_data, nullptr, weight_pro->data_type());
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "LstmNonMindir fp16 PackProjectWeight failed.";
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
