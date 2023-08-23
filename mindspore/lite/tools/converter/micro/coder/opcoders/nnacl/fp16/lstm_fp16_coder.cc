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

#include "coder/opcoders/nnacl/fp16/lstm_fp16_coder.h"
#include <cfloat>
#include <string>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/log.h"
#include "coder/opcoders/file_collector.h"
#include "coder/utils/common.h"
#include "base/float16.h"

using mindspore::schema::PrimitiveType_LSTM;

namespace mindspore::lite::micro::nnacl {
int LstmFP16Coder::InitInputWeightBias(CoderContext *const context) {
  NNaclFp32Serializer init_code;
  Tensor *weight_i = input_tensors_.at(SECOND_INPUT);
  MS_CHECK_PTR(weight_i);
  size_t weight_i_size =
    weight_batch_ * lstm_param_->input_col_align_ * lstm_param_->input_size_ * DataTypeSize(data_type_);
  weight_i_ptr_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
  MS_CHECK_PTR(weight_i_ptr_);

  size_t w_buf_size = 0;

  init_code.CodeBufferOffsetExpression(weight_i_ptr_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), weight_i_size);
  w_buf_size += weight_i_size;
  auto packed_weight_i_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(reinterpret_cast<float16 *>(weight_i_ptr_));
  auto weight_i_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(weight_i);
  init_code.CodeFunction("PackLstmWeightFp16", packed_weight_i_str, weight_i_str, weight_batch_,
                         lstm_param_->input_size_, lstm_param_->hidden_size_, lstm_param_->input_col_align_);

  Tensor *bias_i = input_tensors_.at(FOURTH_INPUT);
  MS_CHECK_PTR(bias_i);
  input_bias_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
  MS_CHECK_PTR(input_bias_);
  size_t bias_i_size = weight_batch_ * lstm_param_->input_col_align_ * DataTypeSize(data_type_);
  w_buf_size += bias_i_size;
  init_code.CodeBufferOffsetExpression(input_bias_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), bias_i_size);
  auto input_bias_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(reinterpret_cast<float16 *>(input_bias_));
  auto bias_i_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(bias_i);
  init_code.CodeFunction("memset", input_bias_str, 0, bias_i_size);
  init_code.CodeFunction("PackLstmBiasFp16", input_bias_str, bias_i_str, weight_batch_, lstm_param_->hidden_size_,
                         lstm_param_->input_col_align_, lstm_param_->bidirectional_);

  context->AppendInitWeightSizeCode(w_buf_size);
  context->AppendInitCode(init_code.str());
  return RET_OK;
}

int LstmFP16Coder::InitStateWeightBias(CoderContext *const context) {
  NNaclFp32Serializer init_code;
  size_t w_buf_size = 0;

  Tensor *weight_h = input_tensors().at(THIRD_INPUT);
  MS_CHECK_PTR(weight_h);
  size_t weight_h_size =
    weight_batch_ * lstm_param_->state_col_align_ * lstm_param_->output_size_ * DataTypeSize(data_type_);
  weight_h_ptr_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
  MS_CHECK_PTR(weight_h_ptr_);
  init_code.CodeBufferOffsetExpression(weight_h_ptr_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), weight_h_size);
  w_buf_size += weight_h_size;
  auto pack_weight_h_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(reinterpret_cast<float16 *>(weight_h_ptr_));
  auto weight_h_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(weight_h);
  if (!is_vec_) {
    init_code.CodeFunction("PackLstmWeightFp16", pack_weight_h_str, weight_h_str, weight_batch_,
                           lstm_param_->project_size_, lstm_param_->hidden_size_, lstm_param_->state_col_align_);
  } else {
    init_code.CodeFunction("memcpy", pack_weight_h_str, weight_h_str, weight_h->Size());
  }

  size_t state_bias_size = weight_batch_ * lstm_param_->state_col_align_ * DataTypeSize(data_type_);
  state_bias_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
  MS_CHECK_PTR(state_bias_);
  init_code.CodeBufferOffsetExpression(state_bias_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), state_bias_size);
  w_buf_size += state_bias_size;

  Tensor *bias_i = input_tensors_.at(FOURTH_INPUT);
  MS_CHECK_PTR(bias_i);
  std::string state_bias_addr =
    allocator_->GetRuntimeAddr(bias_i) + "+" + std::to_string(kFour * lstm_param_->hidden_size_);
  auto state_bias_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(reinterpret_cast<float16 *>(state_bias_));
  init_code.CodeFunction("PackLstmBiasFp16", state_bias_str, state_bias_addr, weight_batch_, lstm_param_->hidden_size_,
                         lstm_param_->state_col_align_, lstm_param_->bidirectional_);

  context->AppendInitWeightSizeCode(w_buf_size);
  context->AppendInitCode(init_code.str());
  return RET_OK;
}

int LstmFP16Coder::InitProjectWeight(CoderContext *const context) {
  if (input_tensors_.size() < C7NUM) {
    return RET_OK;
  }

  NNaclFp32Serializer init_code;
  size_t w_buf_size = 0;
  Tensor *weight_pro = input_tensors().at(Index6);
  MS_CHECK_PTR(weight_pro);
  int batch = lstm_param_->bidirectional_ ? C2NUM : C1NUM;
  int col_align = is_vec_ ? lstm_param_->output_size_ : UP_ROUND(lstm_param_->output_size_, col_tile_);
  size_t weight_pro_size = batch * lstm_param_->hidden_size_ * col_align * DataTypeSize(data_type_);
  weight_pro_ptr_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
  MS_CHECK_PTR(weight_pro_ptr_);
  init_code.CodeBufferOffsetExpression(weight_pro_ptr_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), weight_pro_size);
  w_buf_size += weight_pro_size;
  auto pack_weight_pro_str =
    MemoryAllocator::GetInstance()->GetRuntimeAddr(reinterpret_cast<float16 *>(weight_pro_ptr_));
  auto weight_pro_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(weight_pro);
  if (!is_vec_) {
    init_code.CodeFunction("PackLstmWeightFp16", pack_weight_pro_str, weight_pro_str, batch, lstm_param_->hidden_size_,
                           lstm_param_->project_size_, col_align);
  } else {
    init_code.CodeFunction("memcpy", pack_weight_pro_str, weight_pro_str, weight_pro->Size());
  }

  size_t bias_pro_size = UP_ROUND(lstm_param_->output_size_, col_tile_) * DataTypeSize(data_type_);
  bias_pro_ptr_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
  MS_CHECK_PTR(bias_pro_ptr_);
  init_code.CodeBufferOffsetExpression(bias_pro_ptr_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), bias_pro_size);
  w_buf_size += bias_pro_size;
  auto bias_pro_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(reinterpret_cast<float16 *>(bias_pro_ptr_));
  init_code.CodeFunction("memset", bias_pro_str, 0, bias_pro_size);

  context->AppendInitWeightSizeCode(w_buf_size);
  context->AppendInitCode(init_code.str());
  return RET_OK;
}

int LstmFP16Coder::MallocRunBuffer(CoderContext *const context) {
  bool need_pack = target_ != kARM64 || (lstm_param_->batch_ * lstm_param_->seq_len_ != 1);
  if (need_pack) {
    buffer_fp16_[FIRST_INPUT] = allocator_->Malloc(
      data_type_, lstm_param_->input_row_align_ * lstm_param_->input_size_ * DataTypeSize(data_type_), kWorkspace);
    MS_CHECK_PTR(buffer_fp16_[FIRST_INPUT]);
  }
  buffer_fp16_[SECOND_INPUT] = allocator_->Malloc(
    data_type_,
    kFour * lstm_param_->seq_len_ * lstm_param_->batch_ * lstm_param_->hidden_size_ * DataTypeSize(data_type_),
    kWorkspace);
  MS_CHECK_PTR(buffer_fp16_[SECOND_INPUT]);
  if (!is_vec_) {
    buffer_fp16_[THIRD_INPUT] = allocator_->Malloc(
      data_type_, lstm_param_->state_row_align_ * lstm_param_->output_size_ * DataTypeSize(data_type_), kWorkspace);
    MS_CHECK_PTR(buffer_fp16_[THIRD_INPUT]);
  }
  buffer_fp16_[FOURTH_INPUT] = allocator_->Malloc(
    data_type_, kFour * lstm_param_->batch_ * lstm_param_->hidden_size_ * DataTypeSize(data_type_), kWorkspace);
  MS_CHECK_PTR(buffer_fp16_[FOURTH_INPUT]);
  if (!(lstm_param_->zoneout_cell_ >= -FLT_EPSILON && lstm_param_->zoneout_cell_ <= FLT_EPSILON)) {
    buffer_fp16_[FIFTH_INPUT] = allocator_->Malloc(
      data_type_, lstm_param_->batch_ * lstm_param_->hidden_size_ * DataTypeSize(data_type_), kWorkspace);
    MS_CHECK_PTR(buffer_fp16_[FIFTH_INPUT]);
  }
  if (!(lstm_param_->zoneout_hidden_ >= -FLT_EPSILON && lstm_param_->zoneout_hidden_ <= FLT_EPSILON)) {
    buffer_fp16_[SIXTH_INPUT] = allocator_->Malloc(
      data_type_, lstm_param_->batch_ * lstm_param_->output_size_ * DataTypeSize(data_type_), kWorkspace);
    MS_CHECK_PTR(buffer_fp16_[SIXTH_INPUT]);
  }
  if (!is_vec_) {
    buffer_fp16_[Index6] = allocator_->Malloc(
      data_type_, lstm_param_->state_row_align_ * lstm_param_->hidden_size_ * DataTypeSize(data_type_), kWorkspace);
    MS_CHECK_PTR(buffer_fp16_[Index6]);
  }
  return RET_OK;
}

int LstmFP16Coder::Prepare(CoderContext *const context) {
  if (input_tensor_->data_type() != kNumberTypeFloat16) {
    MS_LOG(INFO) << "Input tensor data type is invalid";
    return RET_INPUT_PARAM_INVALID;
  }
  MS_CHECK_RET_CODE(LstmFP32Coder::Prepare(context), "Common Prepare failed");
  MS_CHECK_RET_CODE(InitProjectWeight(context), "Init Project weight failed");
  return RET_OK;
}

int LstmFP16Coder::DoCode(CoderContext *context) {
  Collect(context,
          {
            "nnacl/lstm_parameter.h",
            "nnacl/fp16/lstm_fp16.h",
          },
          {
            "lstm_fp16.c",
            "arithmetic_fp16.c",
          });
  if (target_ == kARM32 || target_ == kARM64) {
    Collect(context, {}, {},
            {
              "MatVecMulFp16.S",
            });
  }

  Tensor *hidden_state = input_tensors_.at(FIFTH_INPUT);
  MS_CHECK_PTR(hidden_state);
  Tensor *cell_state = input_tensors_.at(SIXTH_INPUT);
  MS_CHECK_PTR(cell_state);
  Tensor *output_hidden_state = output_tensors_.at(SECOND_INPUT);
  MS_CHECK_PTR(output_hidden_state);
  Tensor *output_cell_state = output_tensors_[THIRD_INPUT];
  MS_CHECK_PTR(output_hidden_state);

  NNaclFp32Serializer code;
  code << "float16_t *buffer[7] = {";
  for (const auto &buf : buffer_fp16_) {
    std::string addr = buf == nullptr ? "NULL" : allocator_->GetRuntimeAddr(static_cast<float16 *>(buf));
    code << addr << ", ";
  }
  code << "};\n";

  code.CodeStruct("lstm_param", *lstm_param_);
  code.CodeFunction("memcpy", output_hidden_state, hidden_state, hidden_state->Size());
  code.CodeFunction("memcpy", output_cell_state, cell_state, cell_state->Size());
  auto weight_i_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(weight_i_ptr_));
  auto weight_h_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(weight_h_ptr_));
  auto weight_pro_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(weight_pro_ptr_));
  auto input_bias_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(input_bias_));
  auto state_bias_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(state_bias_));
  auto pro_bias_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(bias_pro_ptr_));

  code.CodeFunction("LstmFp16", output_tensor_, input_tensor_, weight_i_str, weight_h_str, input_bias_str,
                    state_bias_str, weight_pro_str, pro_bias_str, output_hidden_state, output_cell_state, "buffer",
                    "&lstm_param");
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_LSTM, CPUOpCoderCreator<LstmFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_LSTM, CPUOpCoderCreator<LstmFP16Coder>)
}  // namespace mindspore::lite::micro::nnacl
