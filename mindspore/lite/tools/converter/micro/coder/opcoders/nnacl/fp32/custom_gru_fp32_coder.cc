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
#include "coder/opcoders/nnacl/fp32/custom_gru_fp32_coder.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "nnacl/custom_gru_parameter.h"

using mindspore::schema::PrimitiveType_Custom;

namespace mindspore::lite::micro::nnacl {
namespace {
constexpr size_t kOutputNum = 3;
constexpr size_t kInputDims = 3;
constexpr size_t kWeightDims = 2;
constexpr size_t kInputSize = 6;
}  // namespace
int CustomGruFP32Coder::Prepare(CoderContext *const context) {
  if (input_tensors_.size() != kInputSize) {
    MS_LOG(ERROR) << "built-in CustomGru must have 6 input." << node_->name_;
    return RET_ERROR;
  }
  for (size_t i = 1; i < kInputSize - 1; ++i) {
    if (!input_tensors_[i]->IsConst()) {
      MS_LOG(ERROR) << "built-in CustomGru only support first-input and last-input is variable." << node_->name_;
      return RET_NOT_SUPPORT;
    }
  }
  if (InitParamter() != RET_OK) {
    MS_LOG(ERROR) << "Init built-in CustomGru Parameter failed." << node_->name_;
    return RET_ERROR;
  }
  return ReSize();
}

int CustomGruFP32Coder::InitParamter() {
  param_ = reinterpret_cast<CustomGruParameter *>(parameter_);
  param_->op_parameter_.thread_num_ = 1;
  auto weight_in_shape = input_tensors_[1]->shape();
  auto weight_hidden_shape = input_tensors_[C2NUM]->shape();
  if (weight_in_shape.size() != kWeightDims || weight_hidden_shape.size() != kWeightDims) {
    MS_LOG(ERROR) << "built-in CustomGru's weight must be 2D." << node_->name_;
    return RET_ERROR;
  }
  if (weight_in_shape[0] != weight_hidden_shape[0]) {
    MS_LOG(ERROR) << "Built-in CustomGru's weight-in and weight-hidden first-dim must be same." << node_->name_;
    return RET_ERROR;
  }
  if (weight_hidden_shape[0] != weight_hidden_shape[1] * C3NUM) {
    MS_LOG(ERROR) << "Built-in CustomGru's weight-hidden first-dim must be 3 * second-dim." << node_->name_;
    return RET_ERROR;
  }
  auto bias_in_shape = input_tensors_[C3NUM]->shape();
  auto bias_hidden_shape = input_tensors_[C4NUM]->shape();
  if (bias_in_shape.size() != 1) {
    MS_LOG(ERROR) << "built-in CustomGru's bias must be 1D." << node_->name_;
    return RET_ERROR;
  }
  if (bias_in_shape != bias_hidden_shape) {
    MS_LOG(ERROR) << "built-in CustomGru's bias-in and bias-hidden must have same shape." << node_->name_;
    return RET_ERROR;
  }
  if (bias_in_shape.back() != weight_in_shape.front()) {
    MS_LOG(ERROR) << "built-in CustomGru's bias-in shape don't match with the first-dim of weight." << node_->name_;
    return RET_ERROR;
  }
  if (bias_in_shape.front() % C3NUM != 0) {
    MS_LOG(ERROR) << "The first-dim of CustomGru's weight must be 3 * hidden.";
    return RET_ERROR;
  }
  param_->input_size = weight_in_shape.back();
  param_->hidden_size = bias_in_shape.front() / C3NUM;
  return RET_OK;
}

int CustomGruFP32Coder::ReSize() {
  auto in_shape = input_tensor_->shape();
  if (in_shape.size() != kInputDims) {
    MS_LOG(ERROR) << "built-in CustomGru's first-input must be 3D." << node_->name_;
    return RET_ERROR;
  }
  param_->num_step = in_shape[0];
  param_->batch_size = in_shape[1];
  if (in_shape.back() != param_->input_size) {
    MS_LOG(ERROR) << "built-in CustomGru's fisrt-input don't match its weight." << node_->name_;
    return RET_ERROR;
  }
  return InitWeightAndBias();
}

int CustomGruFP32Coder::InitWeightAndBias() {
  auto col_align = UP_ROUND(param_->hidden_size, col_tile_);
  bias_pack_size_ = col_align * data_type_size_;
  weight_in_pack_size_ = static_cast<size_t>(col_align * param_->input_size) * data_type_size_;
  weight_input_ = allocator_->Malloc(kNumberTypeUInt8, weight_in_pack_size_ * C3NUM, kOnlinePackWeight,
                                     input_tensors_.at(1)->tensor_name() + "_online_pack");
  MS_CHECK_TRUE_MSG(weight_input_ != nullptr, RET_NULL_PTR, "Init weight-in failed.");
  weight_hidden_pack_size_ = static_cast<size_t>(col_align * param_->hidden_size) * data_type_size_;
  weight_hidden_ = allocator_->Malloc(kNumberTypeUInt8, weight_hidden_pack_size_ * C3NUM, kOnlinePackWeight,
                                      input_tensors_.at(C2NUM)->tensor_name() + "_online_pack");
  MS_CHECK_TRUE_MSG(weight_hidden_ != nullptr, RET_NULL_PTR, "Init weight-hidden failed.");
  bias_input_ = allocator_->Malloc(kNumberTypeUInt8, bias_pack_size_ * C3NUM, kOnlinePackWeight,
                                   input_tensors_.at(C3NUM)->tensor_name() + "_online_pack");
  MS_CHECK_TRUE_MSG(bias_input_ != nullptr, RET_NULL_PTR, "Init bias-in failed.");
  bias_hidden_ = allocator_->Malloc(kNumberTypeUInt8, bias_pack_size_ * C3NUM, kOnlinePackWeight,
                                    input_tensors_.at(C4NUM)->tensor_name() + "_online_pack");
  MS_CHECK_TRUE_MSG(bias_hidden_ != nullptr, RET_NULL_PTR, "Init bias-hidden failed.");
  auto row_align = UP_ROUND(param_->batch_size, row_tile_);
  auto work_space =
    (row_align * (param_->input_size + param_->hidden_size) + param_->batch_size * param_->hidden_size * C6NUM) *
    data_type_size_;
  run_buffer_ = allocator_->Malloc(kNumberTypeUInt8, work_space, kWorkspace);
  MS_CHECK_TRUE_MSG(run_buffer_ != nullptr, RET_NULL_PTR, "Init run_buffer failed.");
  return RET_OK;
}

void CustomGruFP32Coder::InitNnaclFile(CoderContext *const context) {
  Collect(context, {"nnacl/fp32/custom_gru_fp32.h"},
          {"custom_gru_fp32.c", "pack_fp32.c", "matmul_fp32.c", "arithmetic_fp32.c", "activation_fp32.c"});
}

void CustomGruFP32Coder::InitPackMatrixB(NNaclFp32Serializer *init_code, const std::string &src, const std::string &dst,
                                         int row, int col) {
  init_code->CodeFunction("RowMajor2Col8MajorParallel", src, dst, row, col, 0, row);
}

void CustomGruFP32Coder::InitBiasCode(CoderContext *const context, NNaclFp32Serializer *init_code) {
  init_code->CodeBufferOffsetExpression(bias_input_, context->weight_name(), context->weight_offset_name(),
                                        context->weight_size_name(), bias_pack_size_ * C3NUM);
  auto bias_in_str = "( " + data_type + " *)" + MemoryAllocator::GetInstance()->GetRuntimeAddr(bias_input_);
  auto bias_in_tensor = MemoryAllocator::GetInstance()->GetRuntimeAddr(input_tensors_[C3NUM]);
  for (int i = 0; i < C3NUM; ++i) {
    auto dst_bias_in = bias_in_str + " + " + std::to_string(i * bias_pack_size_ / data_type_size_);
    auto src_bias_in = bias_in_tensor + " + " + std::to_string(i * param_->hidden_size);
    init_code->CodeFunction("memcpy", dst_bias_in, src_bias_in, param_->hidden_size * data_type_size_);
  }
  init_code->CodeBufferOffsetExpression(bias_hidden_, context->weight_name(), context->weight_offset_name(),
                                        context->weight_size_name(), bias_pack_size_ * C3NUM);
  auto bias_hidden_str = "( " + data_type + " *)" + MemoryAllocator::GetInstance()->GetRuntimeAddr(bias_hidden_);
  auto bias_hidden_tensor = MemoryAllocator::GetInstance()->GetRuntimeAddr(input_tensors_[C4NUM]);
  for (int i = 0; i < C3NUM; ++i) {
    auto dst_bias_hidden = bias_hidden_str + " + " + std::to_string(i * bias_pack_size_ / data_type_size_);
    auto src_bias_hidden = bias_hidden_tensor + " + " + std::to_string(i * param_->hidden_size);
    init_code->CodeFunction("memcpy", dst_bias_hidden, src_bias_hidden, param_->hidden_size * data_type_size_);
  }
}

void CustomGruFP32Coder::InitWeightCode(CoderContext *const context, NNaclFp32Serializer *init_code) {
  init_code->CodeBufferOffsetExpression(weight_input_, context->weight_name(), context->weight_offset_name(),
                                        context->weight_size_name(), weight_in_pack_size_ * C3NUM);
  auto weight_input_str = "( " + data_type + " *)" + MemoryAllocator::GetInstance()->GetRuntimeAddr(weight_input_);
  auto weight_in_tensor = MemoryAllocator::GetInstance()->GetRuntimeAddr(input_tensors_[1]);
  for (int i = 0; i < C3NUM; ++i) {
    auto dst_weight_in = weight_input_str + " + " + std::to_string(i * weight_in_pack_size_ / data_type_size_);
    auto src_weight_in = weight_in_tensor + " + " + std::to_string(i * param_->hidden_size * param_->input_size);
    InitPackMatrixB(init_code, src_weight_in, dst_weight_in, param_->hidden_size, param_->input_size);
  }

  init_code->CodeBufferOffsetExpression(weight_hidden_, context->weight_name(), context->weight_offset_name(),
                                        context->weight_size_name(), weight_hidden_pack_size_ * C3NUM);
  auto weight_hidden_str = "( " + data_type + " *)" + MemoryAllocator::GetInstance()->GetRuntimeAddr(weight_hidden_);
  auto weight_hidden_tensor = MemoryAllocator::GetInstance()->GetRuntimeAddr(input_tensors_[C2NUM]);
  for (int i = 0; i < C3NUM; ++i) {
    auto dst_weight_hidden = weight_hidden_str + " + " + std::to_string(i * weight_hidden_pack_size_ / data_type_size_);
    auto src_weight_hidden =
      weight_hidden_tensor + " + " + std::to_string(i * param_->hidden_size * param_->hidden_size);
    InitPackMatrixB(init_code, src_weight_hidden, dst_weight_hidden, param_->hidden_size, param_->hidden_size);
  }
}

int CustomGruFP32Coder::DoCode(CoderContext *const context) {
  NNaclFp32Serializer code, init_code;
  code.CodeStruct("custom_gru_parm", *param_);
  InitNnaclFile(context);
  InitWeightCode(context, &init_code);
  InitBiasCode(context, &init_code);
  auto row_align = UP_ROUND(param_->batch_size, row_tile_);
  auto buffer_name = "( " + data_type + " *)" + MemoryAllocator::GetInstance()->GetRuntimeAddr(run_buffer_);
  int offset1 = row_align * param_->input_size;
  int offset2 = offset1 + param_->batch_size * param_->hidden_size * C3NUM;
  int offset3 = offset2 + row_align * param_->hidden_size;
  code << data_type + " *buffer[4] = {" << buffer_name << ", " << buffer_name + " + " << offset1 << ", "
       << buffer_name + " + " << offset2 << ", " << buffer_name + " + " << offset3 << "};\n";
  auto weight_input_str = "( " + data_type + " *)" + MemoryAllocator::GetInstance()->GetRuntimeAddr(weight_input_);
  auto weight_hidden_str = "( " + data_type + " *)" + MemoryAllocator::GetInstance()->GetRuntimeAddr(weight_hidden_);
  auto bias_input_str = "( " + data_type + " *)" + MemoryAllocator::GetInstance()->GetRuntimeAddr(bias_input_);
  auto bias_hidden_str = "( " + data_type + " *)" + MemoryAllocator::GetInstance()->GetRuntimeAddr(bias_hidden_);
  code.CodeFunction(op_func_, output_tensor_, input_tensor_, weight_input_str, weight_hidden_str, bias_input_str,
                    bias_hidden_str, input_tensors_[C5NUM], "buffer", "&custom_gru_parm");
  context->AppendInitWeightSizeCode((weight_in_pack_size_ + weight_hidden_pack_size_) * C3NUM +
                                    bias_pack_size_ * C6NUM);
  context->AppendInitCode(init_code.str());
  context->AppendCode(code.str());
  return RET_OK;
}

REG_BUILIN_CUSTOM_CODER(kARM64, kNumberTypeFloat32, "CustomGRU", CPUOpCoderCreator<CustomGruFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
