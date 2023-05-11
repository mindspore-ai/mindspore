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

#include "coder/opcoders/nnacl/fp16/conv_depthwise_3x3_fp16_coder.h"
#include <string>
#include <vector>
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "base/float16.h"

namespace mindspore::lite::micro::nnacl {
int ConvolutionDepthwise3x3FP16Coder::Prepare(CoderContext *const context) {
  if (input_tensor_->data_type() != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Tensor data type is invalid";
    return lite::RET_INPUT_PARAM_INVALID;
  }
  auto ret = Conv2DBaseCoder::Init();
  MS_CHECK_RET_CODE(ret, "ConvolutionBase init failed.");
  ret = InitWeightBias(context);
  MS_CHECK_RET_CODE(ret, "Init weight bias failed.");
  size_t units = UP_DIV(conv_param_->output_w_, C2NUM);  // F(2, 3) contains 2 conv units
  size_t c8 = UP_ROUND(conv_param_->input_channel_, C8NUM);
  buffer_size_ = units * c8 * C12NUM * conv_param_->op_parameter_.thread_num_ * DataTypeSize(data_type_);
  buffer_ = allocator_->Malloc(data_type_, buffer_size_, kWorkspace);
  MS_CHECK_PTR(buffer_);
  return RET_OK;
}

int ConvolutionDepthwise3x3FP16Coder::InitWeightBias(CoderContext *const context) {
  NNaclFp32Serializer init_code;
  // init weight
  int channel = filter_tensor_->Batch();
  size_t pack_weight_size = UP_ROUND(channel, C8NUM) * C12NUM * DataTypeSize(data_type_);
  packed_weight_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
  MS_CHECK_PTR(packed_weight_);
  std::string ori_weight_addr = allocator_->GetRuntimeAddr(filter_tensor_);
  size_t w_buf_size = 0;
  w_buf_size += pack_weight_size;
  auto packed_weight_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(packed_weight_));
  init_code.CodeBufferOffsetExpression(packed_weight_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), pack_weight_size);
  init_code.CodeFunction("PackWeightConvDw3x3Fp16", ori_weight_addr, packed_weight_str, channel);

  // init bias
  size_t bias_size = UP_ROUND(channel, C8NUM) * DataTypeSize(data_type_);
  if (input_tensors_.size() == kInputSize2) {
    bias_data_ =
      allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight, bias_tensor_->tensor_name() + "_online_pack");
    MS_CHECK_PTR(bias_data_);
    init_code.CodeBufferOffsetExpression(bias_data_, context->weight_name(), context->weight_offset_name(),
                                         context->weight_size_name(), bias_size);
    w_buf_size += bias_size;
    auto bias_data_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(bias_data_);
    std::string bias_tensor_str = allocator_->GetRuntimeAddr(bias_tensor_);
    init_code.CodeFunction("memcpy", bias_data_str, bias_tensor_str, bias_tensor_->Size());
  } else {
    bias_data_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight, node_->name_ + "_bias_online_pack");
    MS_CHECK_PTR(bias_data_);
    init_code.CodeFunction("memset", bias_data_, 0, bias_size);
  }

  context->AppendInitWeightSizeCode(w_buf_size);
  context->AppendInitCode(init_code.str());
  return RET_OK;
}

void ConvolutionDepthwise3x3FP16Coder::CollectFilesForFunc(CoderContext *const context) {
  if (target_ == kARM64) {
    Collect(context, {}, {},
            {
              "ConvDwFp16Center.S",
              "ConvDwFp16Border.S",
              "DeconvDwFp16Center.S",
              "DeconvDwFp16Border.S",
              "ConvDwFp16Row.S",
            });
  }
  Collect(context,
          {
            "nnacl/fp16/conv_depthwise_fp16.h",
            "nnacl/fp16/pack_fp16.h",
          },
          {
            "conv_depthwise_fp16.c",
            "pack_fp16.c",
          },
          {});
}

int ConvolutionDepthwise3x3FP16Coder::DoCode(CoderContext *const context) {
  CollectFilesForFunc(context);
  NNaclFp32Serializer code;
  auto input_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(input_tensor_);
  auto output_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(output_tensor_);
  auto packed_weight_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(packed_weight_));
  auto bias_data_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(bias_data_));
  auto buffer_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(buffer_));
  code.CodeFunction("memset", buffer_str, "0", buffer_size_);
  code.CodeStruct("conv_parameter", *conv_param_);

  size_t units = UP_DIV(conv_param_->output_w_, C2NUM);  // F(2, 3) contains 2 conv units
  size_t c8 = UP_ROUND(conv_param_->input_channel_, C8NUM);
  code << "    float16_t *tmp_buffer = " << buffer_str << " + " << C12NUM * c8 * units * kDefaultTaskId << ";\n";
  int step_oh = UP_DIV(conv_param_->output_h_, conv_param_->op_parameter_.thread_num_);
  int start_oh = step_oh * kDefaultTaskId;
  int end_oh = MSMIN(start_oh + step_oh, conv_param_->output_h_);
  code.CodeFunction("ConvDw3x3Fp16", output_str, "tmp_buffer", input_str, packed_weight_str, bias_data_str,
                    "&conv_parameter", start_oh, end_oh);
  context->AppendCode(code.str());
  return RET_OK;
}
}  // namespace mindspore::lite::micro::nnacl
