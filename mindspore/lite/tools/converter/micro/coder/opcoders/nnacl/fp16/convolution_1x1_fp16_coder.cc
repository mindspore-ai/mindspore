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

#include "coder/opcoders/nnacl/fp16/convolution_1x1_fp16_coder.h"
#include <string>
#include <vector>
#include "nnacl/fp32/winograd_utils.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"

namespace mindspore::lite::micro::nnacl {
int Convolution1x1FP16Coder::Prepare(CoderContext *const context) {
  if (input_tensor_->data_type() != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Tensor data type is invalid";
    return lite::RET_INPUT_PARAM_INVALID;
  }
  if (target_ == kARM64) {
    row_tile_ = (output_tensor_->format() == NC4HW4) ? C16NUM : C12NUM;
    col_tile_ = (output_tensor_->format() == NC4HW4) ? C8NUM : C16NUM;
  }

  if (matmul_param_ == nullptr) {
    matmul_param_ = new (std::nothrow) MicroMatmulParameter();
    if (matmul_param_ == nullptr) {
      MS_LOG(ERROR) << "Init matmul_param_ failed.";
      return RET_ERROR;
    }
  }
  auto ret = Conv2DBaseCoder::Init();
  MS_CHECK_RET_CODE(ret, "ConvolutionBase init failed.");
  ret = InitWeightBias(context);
  MS_CHECK_RET_CODE(ret, "Init weight bias failed.");
  ret = InitMatmulParam();
  MS_CHECK_RET_CODE(ret, "Init matmul param failed.");
  pack_input_size_ = matmul_param_->row_align_ * matmul_param_->deep_ * DataTypeSize(data_type_);
  pack_input_ = allocator_->Malloc(data_type_, pack_input_size_, kWorkspace);
  MS_CHECK_PTR(pack_input_);
  return RET_OK;
}

Convolution1x1FP16Coder::~Convolution1x1FP16Coder() {
  FreeTmpBuffer();
  if (matmul_param_ != nullptr) {
    delete matmul_param_;
    matmul_param_ = nullptr;
  }
  return;
}

void Convolution1x1FP16Coder::FreeTmpBuffer() {
  if (pre_trans_input_ && tmp_input_ != nullptr) {
    free(tmp_input_);
    tmp_input_ = nullptr;
  }
  return;
}

int Convolution1x1FP16Coder::InitMatmulParam() {
  // init matmul param
  matmul_param_->row_ = conv_param_->output_h_ * conv_param_->output_w_;
  matmul_param_->col_ = conv_param_->output_channel_;
  matmul_param_->deep_ = conv_param_->input_channel_;
  matmul_param_->row_align_ = UP_ROUND(matmul_param_->row_, row_tile_);
  matmul_param_->col_align_ = UP_ROUND(matmul_param_->col_, col_tile_);
  matmul_param_->act_type_ = conv_param_->act_type_;
  return RET_OK;
}

int Convolution1x1FP16Coder::InitWeightBias(CoderContext *const context) {
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

  pre_trans_input_ = (conv_param_->pad_u_ != 0 || conv_param_->pad_l_ != 0 || conv_param_->stride_h_ != 1 ||
                      conv_param_->stride_w_ != 1);
  if (pre_trans_input_) {
    tmp_input_size_ = matmul_param_->row_ * matmul_param_->deep_ * DataTypeSize(data_type_);
    tmp_input_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
    MS_CHECK_PTR(tmp_input_);
    init_code.CodeFunction("memset", tmp_input_, 0, tmp_input_size_);
  }

  context->AppendInitWeightSizeCode(w_buf_size);
  context->AppendInitCode(init_code.str());
  return RET_OK;
}

void Convolution1x1FP16Coder::CollectFilesForFunc(CoderContext *const context) {
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
            "wrapper/base/micro_parameter.h",
          },
          {
            "common_func.c",
            "matmul_fp16.c",
            "conv_fp16.c",
            "conv1x1_base.c",
          });
}

int Convolution1x1FP16Coder::DoCode(CoderContext *const context) {
  CollectFilesForFunc(context);
  NNaclFp32Serializer code;
  auto tmp_input_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(tmp_input_));
  auto input_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(input_tensor_);
  auto output_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(output_tensor_);
  output_ptr_ = output_str + " + batch_index * " + std::to_string(matmul_param_->row_ * matmul_param_->col_);
  auto packed_weight_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(packed_weight_));
  auto pack_input_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(pack_input_));

  code << "  for (int batch_index = 0; batch_index < " << conv_param_->input_batch_ << "; batch_index++) {\n";
  auto batch_in = input_str + " + batch_index * " +
                  std::to_string(conv_param_->input_h_ * conv_param_->input_w_ * conv_param_->input_channel_);
  if (pre_trans_input_) {
    code.CodeStruct("conv_parameter", *conv_param_);
    code.CodeFunction("Conv1x1InputPack", batch_in, tmp_input_str, "&conv_parameter", DataTypeSize(data_type_));
  } else {
    tmp_input_str = batch_in;
  }

  if (output_tensor_->format() == NC4HW4 && target_ == kARM64) {
    code.CodeFunction("RowMajor2Col16MajorFp16Opt", tmp_input_str, pack_input_str, matmul_param_->row_,
                      matmul_param_->deep_);
  } else {
    code.CodeFunction("RowMajor2Col12MajorFp16Opt", tmp_input_str, pack_input_str, matmul_param_->row_,
                      matmul_param_->deep_);
  }

  if (output_tensor_->format() == NC4HW4) {
    code.CodeStruct("matmul_param", *matmul_param_);
    code.CodeFunction("Conv1x1OutNc8hw8MultiThreadByWeightFp16", tmp_input_str, pack_input_str, packed_weight_str,
                      bias_data_, output_ptr_, kDefaultTaskId, "&matmul_param");
  } else {
    if (target_ == kARM64) {
      code.CodeFunction("MatMul12x16Fp16Opt", pack_input_str, packed_weight_str, output_ptr_, bias_data_,
                        matmul_param_->act_type_, matmul_param_->deep_, matmul_param_->row_, matmul_param_->col_,
                        matmul_param_->col_, OutType_Nhwc);
    } else {
      code.CodeFunction("MatMul12x8A32Fp16", pack_input_str, packed_weight_str, output_ptr_, bias_data_,
                        matmul_param_->act_type_, matmul_param_->deep_, matmul_param_->row_, matmul_param_->col_,
                        matmul_param_->col_, OutType_Nhwc);
    }
  }
  code << "  }\n";
  context->AppendCode(code.str());
  return RET_OK;
}
}  // namespace mindspore::lite::micro::nnacl
