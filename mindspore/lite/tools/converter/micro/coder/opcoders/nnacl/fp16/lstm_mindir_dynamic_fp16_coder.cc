/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "coder/opcoders/nnacl/fp16/lstm_mindir_dynamic_fp16_coder.h"
#include <cfloat>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/utils/coder_utils.h"
#include "tools/common/string_util.h"

using mindspore::schema::PrimitiveType_LSTM;

namespace mindspore::lite::micro::nnacl {
namespace {
constexpr size_t kMindirInputTensorNum = 4;
}  // namespace

int LstmMindirDynamicFP16Coder::Prepare(CoderContext *const context) {
  CHECK_NULL_RETURN(context);
  CHECK_NOT_EQUAL_RETURN(input_tensors_.size(), kMindirInputTensorNum);
  for (auto in : input_tensors_) {
    MS_CHECK_TRUE_MSG(in != nullptr, RET_INPUT_TENSOR_ERROR, "LstmMindirDynamicFP16Coder input is a nullptr.");
    MS_CHECK_TRUE_MSG(in->data_type() == kNumberTypeFloat16, RET_INPUT_TENSOR_ERROR,
                      "LstmMindirDynamicFP16Coder input must be fp16.");
    MS_CHECK_TRUE_MSG(in->shape().size() == C3NUM, RET_INPUT_TENSOR_ERROR,
                      "LstmMindirDynamicFP16Coder input must be 3D.");
  }
  MS_CHECK_TRUE_MSG(input_tensors_[FOURTH_INPUT]->IsConst(), RET_INPUT_TENSOR_ERROR,
                    "LstmMindirDynamicFP16Coder last three inputs must be all constant.");
  lstm_param_ = reinterpret_cast<LstmParameter *>(parameter_);
  CHECK_NULL_RETURN(lstm_param_);
  return InitParam();
}

int LstmMindirDynamicFP16Coder::DoCode(CoderContext *const context) {
  Collect(context, {"nnacl/lstm_parameter.h", "nnacl/fp16/lstm_fp16.h"},
          {"lstm_fp16.c", "activation_fp16.c", "arithmetic_fp16.c", "matmul_fp16.c", "pack_fp16.c"},
          {"MatmulBaseFp16Neon.S"});
  auto ret = InitInputWeightBias(context);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "Lstm InitInputWeightBias failed.");
  ret = InitStateWeightBias(context);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "Lstm InitStateWeightBias failed.");
  ret = InitProjectWeight(context);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "Lstm InitProjectWeight failed.");
  ret = ComputeWorkSpace();
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "Lstm ComputeWorkSpace failed.");
  CreateBufferAddrStr();
  NNaclFp32Serializer code;
  code << "float16_t *buffer[7] = {";
  for (const auto &buf : buffers_str_) {
    code << "(float16_t *)(" << buf << "), ";
  }
  code << "};\n";
  auto input1 = dynamic_mem_manager_->GetVarTensorAddr(input_tensors_[FIRST_INPUT]);
  auto hidden_init = input_tensors_[SECOND_INPUT]->IsConst()
                       ? allocator_->GetRuntimeAddr(input_tensors_[SECOND_INPUT], true)
                       : dynamic_mem_manager_->GetVarTensorAddr(input_tensors_[SECOND_INPUT]);
  auto cell_init = input_tensors_[THIRD_INPUT]->IsConst()
                     ? allocator_->GetRuntimeAddr(input_tensors_[THIRD_INPUT], true)
                     : dynamic_mem_manager_->GetVarTensorAddr(input_tensors_[THIRD_INPUT]);
  auto output1 = dynamic_mem_manager_->GetVarTensorAddr(output_tensors_[FIRST_INPUT]);
  auto hidden_output = dynamic_mem_manager_->GetVarTensorAddr(output_tensors_[SECOND_INPUT]);
  auto cell_output = dynamic_mem_manager_->GetVarTensorAddr(output_tensors_[THIRD_INPUT]);
  MS_CHECK_TRUE_MSG(!input1.empty() && !hidden_init.empty() && !cell_init.empty() && !output1.empty() &&
                      !hidden_output.empty() && !cell_output.empty(),
                    RET_ERROR, "Lstm cannot get addr.");
  code.CodeStruct("lstm_param", *lstm_param_, dynamic_lstm_param_);
  int data_item_size = static_cast<int>(DataTypeSize(kNumberTypeFloat16));
  auto input_shape2 = shape_info_container_->GetTemplateShape(input_tensors_[SECOND_INPUT]);
  std::string input_shape2_size = AccumulateShape(input_shape2, 0, input_shape2.size());
  code.CodeFunction("memcpy", hidden_output, hidden_init, input_shape2_size + " * " + std::to_string(data_item_size));
  auto input_shape3 = shape_info_container_->GetTemplateShape(input_tensors_[THIRD_INPUT]);
  std::string input_shape3_size = AccumulateShape(input_shape3, 0, input_shape3.size());
  code.CodeFunction("memcpy", cell_output, cell_init, input_shape3_size + " * " + std::to_string(data_item_size));
  GenerateStateWeightBiasStr();
  code.CodeFunction("LstmFp16", "(float16_t *)(" + output1 + ")", "(float16_t *)(" + input1 + ")", weight_i_str_,
                    weight_h_str_, input_bias_str_, state_bias_str_, weight_pro_str_, pro_bias_str_,
                    "(float16_t *)(" + hidden_output + ")", "(float16_t *)(" + cell_output + ")", "buffer",
                    "&lstm_param");
  context->AppendCode(code.str());
  return RET_OK;
}

int LstmMindirDynamicFP16Coder::InitParam() {
  auto in_shape1 = shape_info_container_->GetTemplateShape(input_tensors_[FIRST_INPUT]);
  MS_CHECK_TRUE_MSG(in_shape1.size() == C3NUM, RET_INPUT_TENSOR_ERROR, "LstmMindir first input's dim must be 3D.");
  dynamic_lstm_param_.batch_ = in_shape1[1];
  dynamic_lstm_param_.seq_len_ = in_shape1[0];
  MS_CHECK_TRUE_MSG(IsNumber(in_shape1[C2NUM]), RET_NOT_SUPPORT,
                    "LstmMindir doesn't support input_size is dynamical in micro.");
  lstm_param_->input_size_ = std::atoi(in_shape1[C2NUM].c_str());

  auto h_init_shape = input_tensors_[SECOND_INPUT]->shape();
  auto c_init_shape = input_tensors_[THIRD_INPUT]->shape();
  lstm_param_->hidden_size_ = c_init_shape.back();
  lstm_param_->output_size_ = h_init_shape.back();

  lstm_param_->output_step_ = lstm_param_->bidirectional_ ? C2NUM * lstm_param_->batch_ * lstm_param_->output_size_
                                                          : lstm_param_->batch_ * lstm_param_->output_size_;
  weight_segment_num_ = lstm_param_->bidirectional_ ? C8NUM : C4NUM;
  dynamic_lstm_param_.input_row_align_ =
    "(" + dynamic_lstm_param_.batch_ + " * " + dynamic_lstm_param_.seq_len_ + " + 3) / 4 * 4";
  lstm_param_->input_col_align_ = UP_ROUND(lstm_param_->hidden_size_, C4NUM);

  dynamic_lstm_param_.state_row_align_ = "(" + dynamic_lstm_param_.batch_ + " + 3) / 4 * 4";
  lstm_param_->state_col_align_ = UP_ROUND(lstm_param_->hidden_size_, C4NUM);
  lstm_param_->proj_col_align_ = UP_ROUND(lstm_param_->project_size_, C4NUM);
  dynamic_lstm_param_.output_step_ =
    std::to_string((lstm_param_->bidirectional_ ? C2NUM : C1NUM) * lstm_param_->output_size_) + " * " +
    dynamic_lstm_param_.batch_;
  size_t scale = lstm_param_->bidirectional_ ? C2NUM : C1NUM;
  hi_size_ = scale * C4NUM * lstm_param_->hidden_size_ * lstm_param_->input_size_;
  hh_size_ = scale * C4NUM * lstm_param_->hidden_size_ * lstm_param_->output_size_;
  hp_size_ = scale * lstm_param_->project_size_ * lstm_param_->hidden_size_;
  bias_size_ = scale * C8NUM * lstm_param_->hidden_size_;
  auto real_whole_size = input_tensors_[FOURTH_INPUT]->ElementsNum();
  gpu_state_ = (hi_size_ + hh_size_ + hp_size_ + bias_size_) == static_cast<size_t>(real_whole_size);
  if (gpu_state_) {
    MS_LOG(ERROR) << "LstmMindirDynamicFP16Coder doesn't support model which exported from GPU.";
    return RET_NOT_SUPPORT;
  }
  if (hi_size_ + hh_size_ + hp_size_ == static_cast<size_t>(real_whole_size)) {
    bias_size_ = 0;
    return RET_OK;
  }
  bias_size_ /= C2NUM;
  if ((hi_size_ + hh_size_ + hp_size_ + bias_size_) != static_cast<size_t>(real_whole_size)) {
    MS_LOG(ERROR) << "Bias of LstmMindir exported from cpu  only exist in hi-part.";
    return RET_INPUT_TENSOR_ERROR;
  }
  return RET_OK;
}

int LstmMindirDynamicFP16Coder::InitInputWeightBias(CoderContext *const context) {
  NNaclFp32Serializer init_code;

  size_t weight_hi_size =
    weight_segment_num_ * lstm_param_->input_col_align_ * lstm_param_->input_size_ * DataTypeSize(data_type_);
  weight_i_ptr_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
  MS_CHECK_PTR(weight_i_ptr_);

  size_t w_buf_size = 0;

  init_code.CodeBufferOffsetExpression(weight_i_ptr_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), weight_hi_size);
  w_buf_size += weight_hi_size;
  auto weight_i_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(input_tensors_[FOURTH_INPUT]);
  MS_CHECK_TRUE_MSG(!weight_i_str.empty(), RET_INPUT_TENSOR_ERROR, "Lstm cannot get weight.");
  auto packed_weight_i_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(reinterpret_cast<float16 *>(weight_i_ptr_));
  init_code << "  int32_t order[4] = {0, 2, 3, 1};\n";
  init_code.CodeFunction("PackLstmWeightFp16", packed_weight_i_str, weight_i_str, weight_segment_num_,
                         lstm_param_->input_size_, lstm_param_->hidden_size_, lstm_param_->input_col_align_, "order");

  auto bias_stride = hi_size_ + hh_size_ + hp_size_;
  input_bias_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
  MS_CHECK_PTR(input_bias_);
  size_t bias_i_size = weight_segment_num_ * lstm_param_->input_col_align_ * DataTypeSize(data_type_);
  w_buf_size += bias_i_size;
  init_code.CodeBufferOffsetExpression(input_bias_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), bias_i_size);
  auto input_bias_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(reinterpret_cast<float16 *>(input_bias_));
  init_code.CodeFunction("memset", input_bias_str, 0, bias_i_size);
  if (bias_size_ != 0) {
    init_code.CodeFunction("PackLstmBiasFp16", input_bias_str, weight_i_str + " + " + std::to_string(bias_stride),
                           weight_segment_num_, lstm_param_->hidden_size_, lstm_param_->input_col_align_,
                           lstm_param_->bidirectional_, "order");
  }

  context->AppendInitWeightSizeCode(w_buf_size);
  context->AppendInitCode(init_code.str());
  return RET_OK;
}

int LstmMindirDynamicFP16Coder::InitStateWeightBias(CoderContext *const context) {
  NNaclFp32Serializer init_code;

  size_t weight_hh_size =
    weight_segment_num_ * lstm_param_->state_col_align_ * lstm_param_->project_size_ * DataTypeSize(data_type_);
  weight_h_ptr_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
  MS_CHECK_PTR(weight_h_ptr_);

  size_t w_buf_size = 0;

  init_code.CodeBufferOffsetExpression(weight_h_ptr_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), weight_hh_size);
  w_buf_size += weight_hh_size;
  auto weight_hh_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(input_tensors_[FOURTH_INPUT]);
  MS_CHECK_TRUE_MSG(!weight_hh_str.empty(), RET_INPUT_TENSOR_ERROR, "Lstm cannot get weight.");
  auto packed_weight_hh_str =
    MemoryAllocator::GetInstance()->GetRuntimeAddr(reinterpret_cast<float16 *>(weight_h_ptr_));
  init_code << "  int32_t order[4] = {0, 2, 3, 1};\n";
  init_code.CodeFunction("PackLstmWeightFp16", packed_weight_hh_str, weight_hh_str + " + " + std::to_string(hi_size_),
                         weight_segment_num_, lstm_param_->project_size_, lstm_param_->hidden_size_,
                         lstm_param_->state_col_align_, "order");

  hh_bias_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
  MS_CHECK_PTR(hh_bias_);
  size_t bias_hh_size = weight_segment_num_ * lstm_param_->state_col_align_ * DataTypeSize(data_type_);
  w_buf_size += bias_hh_size;
  init_code.CodeBufferOffsetExpression(hh_bias_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), bias_hh_size);
  auto hh_bias_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(reinterpret_cast<float16 *>(hh_bias_));
  init_code.CodeFunction("memset", hh_bias_str, 0, bias_hh_size);

  context->AppendInitWeightSizeCode(w_buf_size);
  context->AppendInitCode(init_code.str());
  return RET_OK;
}

int LstmMindirDynamicFP16Coder::InitProjectWeight(CoderContext *const context) {
  if (hp_size_ == 0) {
    return RET_OK;
  }

  NNaclFp32Serializer init_code;
  size_t w_buf_size = 0;
  int scale = lstm_param_->bidirectional_ ? C2NUM : C1NUM;
  int col_align = UP_ROUND(lstm_param_->project_size_, C8NUM);
  size_t weight_pro_size = scale * lstm_param_->hidden_size_ * col_align * DataTypeSize(data_type_);
  weight_project_ptr_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
  MS_CHECK_PTR(weight_project_ptr_);
  init_code.CodeBufferOffsetExpression(weight_project_ptr_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), weight_pro_size);
  w_buf_size += weight_pro_size;
  auto weight_hp_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(input_tensors_[FOURTH_INPUT]);
  MS_CHECK_TRUE_MSG(!weight_hp_str.empty(), RET_INPUT_TENSOR_ERROR, "Lstm cannot get weight.");
  auto weight_pro_str =
    MemoryAllocator::GetInstance()->GetRuntimeAddr(reinterpret_cast<float16 *>(weight_project_ptr_));
  init_code.CodeFunction("PackLstmWeightFp16", weight_pro_str,
                         weight_hp_str + " + " + std::to_string(hi_size_ + hh_size_), scale, lstm_param_->hidden_size_,
                         lstm_param_->project_size_, col_align, "NULL");

  size_t bias_pro_size = col_align * DataTypeSize(data_type_);
  project_bias_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
  MS_CHECK_PTR(project_bias_);
  init_code.CodeBufferOffsetExpression(project_bias_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), bias_pro_size);
  w_buf_size += bias_pro_size;
  auto bias_pro_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(reinterpret_cast<float16 *>(project_bias_));
  init_code.CodeFunction("memset", bias_pro_str, 0, bias_pro_size);

  context->AppendInitWeightSizeCode(w_buf_size);
  context->AppendInitCode(init_code.str());
  return RET_OK;
}

int LstmMindirDynamicFP16Coder::ComputeWorkSpace() {
  auto in_shape1 = shape_info_container_->GetTemplateShape(input_tensors_[FIRST_INPUT]);
  auto seq_lens = shape_info_container_->GetRealNums(in_shape1[0]);
  MS_CHECK_TRUE_MSG(!seq_lens.empty(), RET_ERROR, "Lstm cannot get seq_len");
  auto batches = shape_info_container_->GetRealNums(in_shape1[1]);
  MS_CHECK_TRUE_MSG(!batches.empty(), RET_ERROR, "Lstm cannot get batch");
  size_t scene_num = seq_lens.size() > batches.size() ? seq_lens.size() : batches.size();
  for (size_t i = 0; i < scene_num; ++i) {
    int seq_len = seq_lens[i % seq_lens.size()];
    int batch = batches[i % batches.size()];
    size_t buffer1 =
      seq_len * batch <= C3NUM ? 0 : seq_len * batch * lstm_param_->input_size_ * DataTypeSize(data_type_);
    size_t buffer2 = C4NUM * seq_len * batch * lstm_param_->hidden_size_ * DataTypeSize(data_type_);
    size_t buffer3 = batch <= C3NUM ? 0 : batch * lstm_param_->output_size_ * DataTypeSize(data_type_);
    size_t buffer4 = C4NUM * batch * lstm_param_->hidden_size_ * DataTypeSize(data_type_);
    size_t buffer5 = (lstm_param_->zoneout_cell_ >= -FLT_EPSILON && lstm_param_->zoneout_cell_ <= FLT_EPSILON)
                       ? 0
                       : batch * lstm_param_->hidden_size_ * DataTypeSize(data_type_);
    size_t buffer6 = (lstm_param_->zoneout_hidden_ >= -FLT_EPSILON && lstm_param_->zoneout_hidden_ <= FLT_EPSILON)
                       ? 0
                       : batch * lstm_param_->output_size_ * DataTypeSize(data_type_);
    size_t buffer7 = (batch <= C3NUM || lstm_param_->project_size_ == 0)
                       ? 0
                       : batch * lstm_param_->hidden_size_ * DataTypeSize(data_type_);
    auto whole_size = buffer1 + buffer2 + buffer3 + buffer4 + buffer5 + buffer6 + buffer7;
    buffers_start_ = dynamic_mem_manager_->AllocWorkSpace(whole_size, i);
    MS_CHECK_TRUE_MSG(!buffers_start_.empty(), RET_ERROR, "Lstm cannot alloc workspace.");
  }

  return RET_OK;
}

void LstmMindirDynamicFP16Coder::CreateBufferAddrStr() {
  auto in_shape1 = shape_info_container_->GetTemplateShape(input_tensors_[FIRST_INPUT]);
  auto seq_len = in_shape1[0];
  auto batch = in_shape1[1];
  buffers_str_.push_back("(" + seq_len + " * " + batch + " <= 3) ? NULL : " + buffers_start_);
  auto offset = "((" + seq_len + " * " + batch + " <= 3) ? 0 : (" + seq_len + " * " + batch + ") * " +
                std::to_string(lstm_param_->input_size_ * DataTypeSize(data_type_)) + ")";
  buffers_str_.push_back(buffers_start_ + " + " + offset);
  offset = "(" + offset + " + " + seq_len + " * " + batch + " * " +
           std::to_string(C4NUM * lstm_param_->hidden_size_ * DataTypeSize(data_type_)) + ")";
  buffers_str_.push_back(batch + " <= 3 ? NULL : (" + buffers_start_ + " + " + offset + ")");
  offset = "(" + offset + " + (" + batch + " <= 3 ? 0 : (" + batch + ") * " +
           std::to_string(lstm_param_->output_size_ * DataTypeSize(data_type_)) + "))";
  buffers_str_.push_back(buffers_start_ + " + " + offset);
  offset = "(" + offset + " + " + batch + " * " +
           std::to_string(C4NUM * lstm_param_->hidden_size_ * DataTypeSize(data_type_)) + ")";
  if (lstm_param_->zoneout_cell_ < -FLT_EPSILON || lstm_param_->zoneout_cell_ > FLT_EPSILON) {
    buffers_str_.push_back(buffers_start_ + " + " + offset);
    offset =
      "(" + offset + " + " + batch + " * " + std::to_string(lstm_param_->hidden_size_ * DataTypeSize(data_type_)) + ")";
  } else {
    buffers_str_.emplace_back("NULL");
  }
  if (lstm_param_->zoneout_hidden_ < -FLT_EPSILON && lstm_param_->zoneout_hidden_ > FLT_EPSILON) {
    buffers_str_.push_back(buffers_start_ + " + " + offset);
    offset =
      "(" + offset + " + " + batch + " * " + std::to_string(lstm_param_->output_size_ * DataTypeSize(data_type_)) + ")";
  } else {
    buffers_str_.emplace_back("NULL");
  }
  if (lstm_param_->project_size_ == 0) {
    buffers_str_.emplace_back("NULL");
  } else {
    buffers_str_.emplace_back(batch + " <= 3 ? NULL : " + "(" + buffers_start_ + " + " + offset + ")");
  }
}

void LstmMindirDynamicFP16Coder::GenerateStateWeightBiasStr() {
  weight_i_str_.clear();
  weight_h_str_.clear();
  weight_pro_str_.clear();
  input_bias_str_.clear();
  state_bias_str_.clear();
  pro_bias_str_.clear();
  weight_i_str_ = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(weight_i_ptr_));
  weight_h_str_ = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(weight_h_ptr_));
  weight_pro_str_ = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(weight_project_ptr_));
  input_bias_str_ = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(input_bias_));
  state_bias_str_ = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(hh_bias_));
  pro_bias_str_ = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(project_bias_));
}
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_LSTM,
                           CPUOpCoderCreator<LstmMindirDynamicFP16Coder>)
}  // namespace mindspore::lite::micro::nnacl
