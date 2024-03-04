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

#include "coder/opcoders/nnacl/fp16/convolution_1x1_dynamic_fp16_coder.h"
#include <algorithm>
#include "nnacl/fp32/winograd_utils.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "coder/utils/coder_utils.h"

namespace mindspore::lite::micro::nnacl {
int Convolution1x1DynamicFP16Coder::Prepare(CoderContext *const context) {
  CHECK_LESS_RETURN(input_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(output_tensors_.size(), 1);
  for (size_t i = 0; i < input_tensors_.size(); ++i) {
    MS_CHECK_TRUE_MSG(input_tensors_[i]->data_type() == kNumberTypeFloat16, RET_INPUT_PARAM_INVALID,
                      "Tensor data type is invalid");
  }
  for (size_t i = 0; i < output_tensors_.size(); ++i) {
    MS_CHECK_TRUE_MSG(output_tensors_[i]->data_type() == kNumberTypeFloat16, RET_PARAM_INVALID,
                      "Tensor data type is invalid");
  }
  if (target_ == kARM64) {
    row_tile_ = (output_tensor_->format() == NC4HW4) ? C16NUM : C12NUM;
    col_tile_ = (output_tensor_->format() == NC4HW4) ? C8NUM : C16NUM;
  }
  if (matmul_param_ == nullptr) {
    matmul_param_ = new (std::nothrow) MatMulParameter();
    MS_CHECK_PTR(matmul_param_);
  }
  conv_param_ = reinterpret_cast<ConvParameter *>(parameter_);
  filter_tensor_ = input_tensors_.at(kWeightIndex);
  MS_CHECK_PTR(filter_tensor_);
  if (input_tensors_.size() == kInputSize2) {
    bias_tensor_ = input_tensors_.at(kBiasIndex);
    MS_CHECK_PTR(bias_tensor_);
  } else {
    MS_CHECK_TRUE(input_tensors_.size() == kInputSize1, "wrong input size");
  }
  dynamic_param_.input_batch_ = shape_info_container_->GetTemplateShape(input_tensor_)[0];
  conv_param_->input_h_ = input_tensor_->Height();
  conv_param_->input_w_ = input_tensor_->Width();
  conv_param_->input_channel_ = input_tensor_->Channel();
  dynamic_param_.output_batch_ = shape_info_container_->GetTemplateShape(output_tensor_)[0];
  conv_param_->output_h_ = output_tensor_->Height();
  conv_param_->output_w_ = output_tensor_->Width();
  conv_param_->output_channel_ = output_tensor_->Channel();
  MS_CHECK_RET_CODE(InitWeightBias(context), "Init weight bias failed.");
  MS_CHECK_RET_CODE(InitMatmulParam(), "Init matmul param failed.");
  MS_CHECK_RET_CODE(InitTmpBuffer(context), "Init tmp buffer failed.");
  return RET_OK;
}

int Convolution1x1DynamicFP16Coder::DoCode(CoderContext *const context) {
  CollectFilesForFunc(context);
  NNaclFp32Serializer code;
  MS_CHECK_RET_CODE(ComputeWorkspace(), "ComputeWorkspace failed.");
  auto tmp_input_str = "(float16_t *)(" + allocator_->GetRuntimeAddr(static_cast<float16 *>(tmp_input_)) + ")";
  auto input_str =
    "(float16_t *)(" + GetTensorAddr(input_tensor_, input_tensor_->IsConst(), dynamic_mem_manager_, allocator_) + ")";
  auto output_str =
    "(float16_t *)(" + GetTensorAddr(output_tensor_, output_tensor_->IsConst(), dynamic_mem_manager_, allocator_) + ")";
  auto packed_weight_str = allocator_->GetRuntimeAddr(static_cast<float16 *>(packed_weight_));

  code << "  for (int batch_index = 0; batch_index < " << dynamic_param_.input_batch_ << "; batch_index++) {\n";
  output_ptr_ = output_str + " + batch_index * " + std::to_string(matmul_param_->row_ * matmul_param_->col_);
  auto batch_in = input_str + " + batch_index * " +
                  std::to_string(conv_param_->input_h_ * conv_param_->input_w_ * conv_param_->input_channel_);
  if (pre_trans_input_) {
    code.CodeStruct("conv_parameter", *conv_param_, dynamic_param_);
    code.CodeFunction("Conv1x1InputPack", batch_in, tmp_input_str, "&conv_parameter", DataTypeSize(data_type_));
  } else {
    tmp_input_str = batch_in;
  }

  if (output_tensor_->format() == NC4HW4) {
    code.CodeFunction(target_ == kARM64 ? "RowMajor2Col16MajorFp16Opt" : "RowMajor2Col12MajorFp16Opt", tmp_input_str,
                      "(float16_t *)(" + pack_input_str_ + ")", matmul_param_->row_, matmul_param_->deep_);
  } else {
    code.CodeFunction("RowMajor2Col12MajorFp16Opt", tmp_input_str, "(float16_t *)(" + pack_input_str_ + ")",
                      matmul_param_->row_, matmul_param_->deep_);
  }

  if (output_tensor_->format() == NC4HW4) {
    code.CodeStruct("matmul_param", *matmul_param_);
    code.CodeFunction("Conv1x1OutNc8hw8MultiThreadByWeightFp16", tmp_input_str,
                      "(float16_t *)(" + pack_input_str_ + ")", packed_weight_str, bias_data_, output_ptr_,
                      kDefaultTaskId, "&matmul_param");
  } else {
    code.CodeFunction(target_ == kARM64 ? "MatMul12x16Fp16Opt" : "MatMul12x8A32Fp16",
                      "(float16_t *)(" + pack_input_str_ + ")", packed_weight_str, output_ptr_, bias_data_,
                      matmul_param_->act_type_, matmul_param_->deep_, matmul_param_->row_, matmul_param_->col_,
                      matmul_param_->col_, OutType_Nhwc);
  }
  code << "  }\n";
  context->AppendCode(code.str());
  return RET_OK;
}

Convolution1x1DynamicFP16Coder::~Convolution1x1DynamicFP16Coder() {
  FreeTmpBuffer();
  if (matmul_param_ != nullptr) {
    delete matmul_param_;
    matmul_param_ = nullptr;
  }
  return;
}

void Convolution1x1DynamicFP16Coder::FreeTmpBuffer() {
  if (pre_trans_input_ && tmp_input_ != nullptr) {
    free(tmp_input_);
    tmp_input_ = nullptr;
  }
  return;
}

int Convolution1x1DynamicFP16Coder::ComputeWorkspace() {
  pack_input_size_ = matmul_param_->row_align_ * matmul_param_->deep_ * DataTypeSize(data_type_);
  auto input_shape = shape_info_container_->GetTemplateShape(input_tensor_);
  size_t scene_num = 0;
  for (auto &dim_template : input_shape) {
    auto dim_nums = shape_info_container_->GetRealNums(dim_template);
    MS_CHECK_TRUE_MSG(!dim_nums.empty(), RET_ERROR, "Dynamic shape's num must be greater than 0.");
    scene_num = std::max(scene_num, dim_nums.size());
  }
  for (size_t i = 0; i < scene_num; ++i) {
    pack_input_str_ = dynamic_mem_manager_->AllocWorkSpace(pack_input_size_, i);
    MS_CHECK_TRUE_MSG(!pack_input_str_.empty(), RET_ERROR, "Convolution cannot alloc workspace.");
  }
  return RET_OK;
}

int Convolution1x1DynamicFP16Coder::InitMatmulParam() {
  matmul_param_->row_ = conv_param_->output_h_ * conv_param_->output_w_;
  matmul_param_->col_ = conv_param_->output_channel_;
  matmul_param_->deep_ = conv_param_->input_channel_;
  matmul_param_->row_align_ = UP_ROUND(matmul_param_->row_, row_tile_);
  matmul_param_->col_align_ = UP_ROUND(matmul_param_->col_, col_tile_);
  matmul_param_->act_type_ = conv_param_->act_type_;
  return RET_OK;
}

int Convolution1x1DynamicFP16Coder::InitWeightBias(CoderContext *const context) {
  auto input_channel = filter_tensor_->Channel();
  auto output_channel = filter_tensor_->Batch();
  MS_CHECK_TRUE_RET(input_channel > 0 && output_channel > 0, RET_ERROR);
  pack_weight_size_ = input_channel * UP_ROUND(output_channel, col_tile_) * DataTypeSize(data_type_);
  packed_weight_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
  MS_CHECK_PTR(packed_weight_);

  NNaclFp32Serializer init_code;
  std::string ori_weight_addr = allocator_->GetRuntimeAddr(filter_tensor_);
  size_t w_buf_size = 0;
  w_buf_size += pack_weight_size_;
  auto packed_weight_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(packed_weight_));
  init_code.CodeBufferOffsetExpression(packed_weight_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), pack_weight_size_);
  if (target_ == kARM64 && output_tensor_->format() != NC4HW4) {
    init_code.CodeFunction("RowMajor2Col16MajorFp16Opt", ori_weight_addr, packed_weight_str, output_channel,
                           input_channel);
  } else {
    init_code.CodeFunction("ColMajor2Row8MajorFp16", ori_weight_addr, packed_weight_str, input_channel, output_channel,
                           true);
  }
  bias_data_size_ = UP_ROUND(output_channel, col_tile_) * DataTypeSize(data_type_);
  if (input_tensors_.size() == kInputSize2) {
    bias_data_ =
      allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight, bias_tensor_->tensor_name() + "_online_pack");
    MS_CHECK_PTR(bias_data_);
    init_code.CodeBufferOffsetExpression(bias_data_, context->weight_name(), context->weight_offset_name(),
                                         context->weight_size_name(), bias_data_size_);
    w_buf_size += bias_data_size_;
    auto bias_data_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(bias_data_));
    std::string bias_tensor_str = allocator_->GetRuntimeAddr(bias_tensor_);
    init_code.CodeFunction("memcpy", bias_data_str, bias_tensor_str, bias_tensor_->Size());
  } else {
    bias_data_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight, node_->name_ + "_bias_online_pack");
    MS_CHECK_PTR(bias_data_);
    init_code.CodeFunction("memset", bias_data_, 0, bias_data_size_);
  }
  context->AppendInitWeightSizeCode(w_buf_size);
  context->AppendInitCode(init_code.str());
  return RET_OK;
}

int Convolution1x1DynamicFP16Coder::InitTmpBuffer(CoderContext *const context) {
  NNaclFp32Serializer init_code;
  pre_trans_input_ = (conv_param_->pad_u_ != 0 || conv_param_->pad_l_ != 0 || conv_param_->stride_h_ != 1 ||
                      conv_param_->stride_w_ != 1);
  size_t w_size = 0;
  if (pre_trans_input_) {
    tmp_input_size_ = matmul_param_->row_ * matmul_param_->deep_ * DataTypeSize(data_type_);
    tmp_input_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
    MS_CHECK_PTR(tmp_input_);
    w_size += tmp_input_size_;
    init_code.CodeBufferOffsetExpression(tmp_input_, context->weight_name(), context->weight_offset_name(),
                                         context->weight_size_name(), tmp_input_size_);
    init_code.CodeFunction("memset", tmp_input_, 0, tmp_input_size_);
  }
  context->AppendInitWeightSizeCode(w_size);
  context->AppendInitCode(init_code.str());
  return RET_OK;
}

void Convolution1x1DynamicFP16Coder::CollectFilesForFunc(CoderContext *const context) {
  if (target_ == kARM64) {
    Collect(context, {}, {},
            {
              "MatmulFp16.S",
              "MatmulFp16Opt.S",
              "Matmul12X16Fp16.S",
            });
  } else {
    Collect(context, {}, {},
            {
              "Matmul12x8Fp16.S",
            });
  }
  Collect(context,
          {
            "nnacl/fp16/matmul_fp16.h",
            "nnacl/conv_parameter.h",
            "nnacl/op_base.h",
            "nnacl/fp16/conv_fp16.h",
            "nnacl/base/conv1x1_base.h",
          },
          {
            "common_func.c",
            "matmul_fp16.c",
            "conv_fp16.c",
            "conv1x1_base.c",
          });
}
}  // namespace mindspore::lite::micro::nnacl
