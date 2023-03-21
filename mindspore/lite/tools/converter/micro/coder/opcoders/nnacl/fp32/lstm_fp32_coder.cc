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

#include "coder/opcoders/nnacl/fp32/lstm_fp32_coder.h"
#include <cfloat>
#include <string>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/log.h"
#include "coder/opcoders/file_collector.h"
#include "coder/utils/common.h"

using mindspore::schema::PrimitiveType_LSTM;

namespace mindspore::lite::micro::nnacl {
constexpr int kFifthIndex = 4;
constexpr int kSixthIndex = 5;

int LstmFP32Coder::InitInputWeightBias(CoderContext *const context) {
  NNaclFp32Serializer init_code;
  Tensor *weight_i = input_tensors_.at(kWeightIndex);
  MS_CHECK_PTR(weight_i);
  size_t weight_i_size = weight_batch_ * lstm_param_->input_col_align_ * lstm_param_->input_size_ * sizeof(float);
  weight_i_ptr_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, kOnlineSize, kOnlinePackWeight));
  MS_CHECK_PTR(weight_i_ptr_);

  size_t w_buf_size = 0;

  init_code.CodeBufferOffsetExpression(weight_i_ptr_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), weight_i_size);
  w_buf_size += weight_i_size;
  init_code.CodeFunction("PackLstmWeight", weight_i_ptr_, weight_i, weight_batch_, lstm_param_->input_size_,
                         lstm_param_->hidden_size_, lstm_param_->input_col_align_, "NULL");

  Tensor *bias_i = input_tensors_.at(kInputSize2);
  MS_CHECK_PTR(bias_i);
  input_bias_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, kOnlineSize, kOnlinePackWeight));
  MS_CHECK_PTR(input_bias_);
  size_t bias_i_size = weight_batch_ * lstm_param_->input_col_align_ * sizeof(float);
  w_buf_size += bias_i_size;
  init_code.CodeBufferOffsetExpression(input_bias_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), bias_i_size);
  init_code.CodeFunction("PackLstmBias", input_bias_, bias_i, weight_batch_, lstm_param_->hidden_size_,
                         lstm_param_->input_col_align_, lstm_param_->bidirectional_, "NULL");

  context->AppendInitWeightSizeCode(w_buf_size);
  context->AppendInitCode(init_code.str());
  return RET_OK;
}

int LstmFP32Coder::InitStateWeightBias(CoderContext *const context) {
  NNaclFp32Serializer init_code;
  size_t w_buf_size = 0;

  Tensor *weight_h = input_tensors().at(kInputSize1);
  MS_CHECK_PTR(weight_h);
  if (!is_vec_) {
    size_t weight_h_size = weight_batch_ * lstm_param_->state_col_align_ * lstm_param_->hidden_size_ * sizeof(float);
    weight_h_ptr_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, kOnlineSize, kOnlinePackWeight));
    MS_CHECK_PTR(weight_h_ptr_);
    init_code.CodeBufferOffsetExpression(weight_h_ptr_, context->weight_name(), context->weight_offset_name(),
                                         context->weight_size_name(), weight_h_size);
    w_buf_size += weight_h_size;
    init_code.CodeFunction("PackLstmWeight", weight_h_ptr_, weight_h, weight_batch_, lstm_param_->hidden_size_,
                           lstm_param_->hidden_size_, lstm_param_->state_col_align_, "NULL");
  }

  state_bias_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, kOnlineSize, kOnlinePackWeight));
  size_t state_bias_size = weight_batch_ * lstm_param_->state_col_align_ * sizeof(float);
  MS_CHECK_PTR(state_bias_);
  init_code.CodeBufferOffsetExpression(state_bias_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), state_bias_size);
  w_buf_size += state_bias_size;

  Tensor *bias_i = input_tensors_.at(kInputSize2);
  MS_CHECK_PTR(bias_i);
  std::string state_bias_addr =
    allocator_->GetRuntimeAddr(bias_i) + "+" + std::to_string(kFour * lstm_param_->hidden_size_);
  init_code.CodeFunction("PackLstmBias", state_bias_, state_bias_addr, weight_batch_, lstm_param_->hidden_size_,
                         lstm_param_->state_col_align_, lstm_param_->bidirectional_, "NULL");

  context->AppendInitWeightSizeCode(w_buf_size);
  context->AppendInitCode(init_code.str());
  return RET_OK;
}

int LstmFP32Coder::InitParam() {
  std::vector<int> in_shape = input_tensor_->shape();
  lstm_param_->seq_len_ = in_shape.at(0);
  lstm_param_->batch_ = in_shape.at(1);
  lstm_param_->input_size_ = in_shape.at(kTwo);

  auto weight_i = input_tensors_.at(1);
  MS_ASSERT(weight_i != nullptr);
  std::vector<int> w_shape = weight_i->shape();
  lstm_param_->hidden_size_ = w_shape.at(1) / kFour;
  lstm_param_->output_step_ = lstm_param_->bidirectional_ ? kTwo * lstm_param_->batch_ * lstm_param_->hidden_size_
                                                          : lstm_param_->batch_ * lstm_param_->hidden_size_;
  weight_batch_ = lstm_param_->bidirectional_ ? kEight : kFour;

  if (target_ == kARM32 || target_ == kCortex_M) {
    row_tile_ = C12NUM;
    col_tile_ = C4NUM;
  } else {
    row_tile_ = C12NUM;
    col_tile_ = C8NUM;
  }
  lstm_param_->input_row_align_ = UP_ROUND(lstm_param_->seq_len_ * lstm_param_->batch_, row_tile_);
  lstm_param_->input_col_align_ = UP_ROUND(lstm_param_->hidden_size_, col_tile_);

  is_vec_ = lstm_param_->batch_ == 1;
  lstm_param_->state_row_align_ = is_vec_ ? 1 : UP_ROUND(lstm_param_->batch_, row_tile_);
  lstm_param_->state_col_align_ = is_vec_ ? lstm_param_->hidden_size_ : UP_ROUND(lstm_param_->hidden_size_, col_tile_);
  return RET_OK;
}

int LstmFP32Coder::MallocRunBuffer(CoderContext *const context) {
  constexpr int kMultiply = 4;
  constexpr int kThirdInput = 2;

  buffer_[0] = reinterpret_cast<float *>(allocator_->Malloc(
    kNumberTypeFloat32, lstm_param_->input_row_align_ * lstm_param_->input_size_ * sizeof(float), kWorkspace));
  MS_CHECK_PTR(buffer_[0]);
  buffer_[1] = reinterpret_cast<float *>(allocator_->Malloc(
    kNumberTypeFloat32,
    kMultiply * lstm_param_->seq_len_ * lstm_param_->batch_ * lstm_param_->hidden_size_ * sizeof(float), kWorkspace));
  MS_CHECK_PTR(buffer_[1]);
  if (!is_vec_) {
    buffer_[kThirdInput] = reinterpret_cast<float *>(allocator_->Malloc(
      kNumberTypeFloat32, lstm_param_->state_row_align_ * lstm_param_->hidden_size_ * sizeof(float), kWorkspace));
    MS_CHECK_PTR(buffer_[kThirdInput]);
  }
  buffer_[kThree] = reinterpret_cast<float *>(allocator_->Malloc(
    kNumberTypeFloat32, kMultiply * lstm_param_->batch_ * lstm_param_->hidden_size_ * sizeof(float), kWorkspace));
  MS_CHECK_PTR(buffer_[kThree]);
  if (!(lstm_param_->zoneout_cell_ >= -FLT_EPSILON && lstm_param_->zoneout_cell_ <= FLT_EPSILON)) {
    buffer_[kFour] = reinterpret_cast<float *>(allocator_->Malloc(
      kNumberTypeFloat32, lstm_param_->batch_ * lstm_param_->hidden_size_ * sizeof(float), kWorkspace));
    MS_CHECK_PTR(buffer_[kFour]);
  }
  if (!(lstm_param_->zoneout_hidden_ >= -FLT_EPSILON && lstm_param_->zoneout_hidden_ <= FLT_EPSILON)) {
    buffer_[kFive] = reinterpret_cast<float *>(allocator_->Malloc(
      kNumberTypeFloat32, lstm_param_->batch_ * lstm_param_->hidden_size_ * sizeof(float), kWorkspace));
    MS_CHECK_PTR(buffer_[kFive]);
  }
  buffer_[kSix] = nullptr;
  return RET_OK;
}

int LstmFP32Coder::ReSize(CoderContext *const context) {
  MS_CHECK_RET_CODE(InitParam(), "init params of lstm coder failed");
  MS_CHECK_RET_CODE(InitInputWeightBias(context), "init input weight and bias failed");
  MS_CHECK_RET_CODE(InitStateWeightBias(context), "init state weight and bias failed");
  MS_CHECK_RET_CODE(MallocRunBuffer(context), "malloc run buffer failed");
  return RET_OK;
}

int LstmFP32Coder::Prepare(CoderContext *const context) {
  lstm_param_ = reinterpret_cast<LstmParameter *>(parameter_);
  return ReSize(context);
}

int LstmFP32Coder::DoCode(CoderContext *context) {
  Collect(context,
          {
            "nnacl/lstm_parameter.h",
            "nnacl/fp32/lstm_fp32.h",
          },
          {
            "lstm_fp32.c",
            "mul_fp32.c",
          });
  if (target_ == kARM32 || target_ == kARM64) {
    Collect(context, {}, {},
            {
              "MatVecMulFp32.S",
            });
  }
  Tensor *hidden_state = input_tensors_.at(kFifthIndex);
  MS_CHECK_PTR(hidden_state);
  Tensor *cell_state = input_tensors_.at(kSixthIndex);
  MS_CHECK_PTR(cell_state);
  Tensor *output_hidden_state = output_tensors_[1];
  MS_CHECK_PTR(output_hidden_state);
  Tensor *output_cell_state = output_tensors_[kTwo];
  MS_CHECK_PTR(output_hidden_state);

  std::vector<std::string> buffers_addr;
  for (const auto &buf : buffer_) {
    std::string addr = buf == nullptr ? "NULL" : allocator_->GetRuntimeAddr(buf);
    buffers_addr.push_back(addr);
  }
  NNaclFp32Serializer code;
  code.CodeStruct("lstm_param", *lstm_param_);
  code.CodeArray("buffer", buffers_addr.data(), buffers_addr.size(), false);
  code.CodeFunction("memcpy", output_hidden_state, hidden_state, hidden_state->Size());
  code.CodeFunction("memcpy", output_cell_state, cell_state, cell_state->Size());
  if (weight_h_ptr_ != nullptr) {
    code.CodeFunction("Lstm", output_tensor_, input_tensor_, weight_i_ptr_, weight_h_ptr_, input_bias_, state_bias_,
                      output_hidden_state, output_cell_state, "buffer", "&lstm_param");
  } else {
    auto *weight_h_tensor = input_tensors().at(kInputSize1);
    auto weight_h = allocator_->GetRuntimeAddr(weight_h_tensor, weight_h_tensor->IsConst());
    code.CodeFunction("Lstm", output_tensor_, input_tensor_, weight_i_ptr_, weight_h, input_bias_, state_bias_,
                      output_hidden_state, output_cell_state, "buffer", "&lstm_param");
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_LSTM, CPUOpCoderCreator<LstmFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
