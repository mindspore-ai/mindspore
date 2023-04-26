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

#include "coder/opcoders/nnacl/fp16/conv_depthwise_fp16_coder.h"
#include <string>
#include "coder/opcoders/parallel.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "base/float16.h"

namespace mindspore::lite::micro::nnacl {
int ConvolutionDepthwiseFP16Coder::Prepare(CoderContext *const context) {
  if (input_tensor_->data_type() != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Tensor data type is invalid";
    return lite::RET_INPUT_PARAM_INVALID;
  }
  return ConvolutionDepthwiseFP32Coder::Prepare(context);
}

void ConvolutionDepthwiseFP16Coder::InitCodeOnline(CoderContext *const context) {
  NNaclFp32Serializer init_code;
  init_code.CodeBufferOffsetExpression(packed_weight_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), packed_weight_size_);
  auto ori_weight_addr = allocator_->GetRuntimeAddr(filter_tensor_);
  auto packed_weight_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(packed_weight_));
  init_code.CodeFunction("PackNCHWToNHWCFp16", ori_weight_addr, packed_weight_str, 1,
                         filter_tensor_->Height() * filter_tensor_->Width(), filter_tensor_->Batch(), 0, 0);

  // init bias
  init_code.CodeBufferOffsetExpression(bias_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), packed_bias_size_);
  if (input_tensors_.size() == kInputSize2) {
    auto bias_str = allocator_->GetRuntimeAddr(bias_tensor_);
    init_code.CodeFunction("memcpy", bias_, bias_str, bias_tensor_->Size());
  } else {
    init_code.CodeFunction("memcpy", bias_, 0, packed_bias_size_);
  }
  context->AppendInitWeightSizeCode(packed_weight_size_ + packed_bias_size_);
  context->AppendInitCode(init_code.str());
}

void ConvolutionDepthwiseFP16Coder::CollectFilesForFunc(CoderContext *const context) {
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
            "nnacl/fp16/activation_fp16.h",
          },
          {
            "conv_depthwise_fp16.c",
            "pack_fp16.c",
            "activation_fp16.c",
          },
          {});
}

int ConvolutionDepthwiseFP16Coder::DoCode(CoderContext *const context) {
  MS_CHECK_TRUE(conv_param_->input_channel_ == conv_param_->output_channel_,
                "Only support input channel equals output channel.");
  CollectFilesForFunc(context);
  InitCodeOnline(context);

  nnacl::NNaclFp32Serializer code;
  // call the op function
  code.CodeStruct("conv_parameter", *conv_param_);
  auto input_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(input_tensor_);
  auto output_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(output_tensor_);
  auto packed_weight_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(packed_weight_));
  code.CodeFunction("ConvDwFp16", output_str, input_str, packed_weight_str, bias_, "&conv_parameter", kDefaultTaskId);
  context->AppendCode(code.str());
  return RET_OK;
}
}  // namespace mindspore::lite::micro::nnacl
