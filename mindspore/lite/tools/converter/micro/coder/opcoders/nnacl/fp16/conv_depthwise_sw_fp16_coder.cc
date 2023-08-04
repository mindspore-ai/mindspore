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

#include "coder/opcoders/nnacl/fp16/conv_depthwise_sw_fp16_coder.h"
#include <string>
#include <vector>
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "base/float16.h"

namespace mindspore::lite::micro::nnacl {
int ConvolutionDepthwiseSWFP16Coder::Prepare(CoderContext *const context) {
  if (input_tensor_->data_type() != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Tensor data type is invalid";
    return lite::RET_INPUT_PARAM_INVALID;
  }
  if (sw_param_ == nullptr) {
    sw_param_ = new (std::nothrow) SlidingWindowParam();
    if (sw_param_ == nullptr) {
      MS_LOG(ERROR) << "Init sw_param_ failed.";
      return RET_ERROR;
    }
  }
  auto ret = Conv2DBaseCoder::Init();
  MS_CHECK_RET_CODE(ret, "ConvolutionBase init failed.");
  ret = InitWeightBias(context);
  MS_CHECK_RET_CODE(ret, "Init weight bias failed.");
  return RET_OK;
}

int ConvolutionDepthwiseSWFP16Coder::InitWeightBias(CoderContext *const context) {
  NNaclFp32Serializer init_code;
  // init weight
  int channel = filter_tensor_->Batch();
  int kernel_h = filter_tensor_->Height();
  int kernel_w = filter_tensor_->Width();
  size_t w_buf_size = 0;
  size_t pack_weight_size = UP_ROUND(channel, C8NUM) * C8NUM * kernel_h * kernel_w * DataTypeSize(data_type_);
  packed_weight_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
  MS_CHECK_PTR(packed_weight_);
  std::string ori_weight_addr = allocator_->GetRuntimeAddr(filter_tensor_);
  w_buf_size += pack_weight_size;
  auto packed_weight_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(packed_weight_));
  init_code.CodeBufferOffsetExpression(packed_weight_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), pack_weight_size);
  init_code.CodeFunction("PackNCHWFp16ToNC8HW8Fp16", ori_weight_addr, packed_weight_str, 1, kernel_h * kernel_w,
                         channel);

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

void ConvolutionDepthwiseSWFP16Coder::CollectFilesForFunc(CoderContext *const context) {
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
            "nnacl/fp32/conv_depthwise_fp32.h",
            "nnacl/fp16/conv_depthwise_fp16.h",
            "nnacl/fp16/pack_fp16.h",
          },
          {
            "conv_depthwise_fp32.c",
            "conv_depthwise_fp16.c",
            "pack_fp16.c",
          },
          {});
}

int ConvolutionDepthwiseSWFP16Coder::DoCode(CoderContext *const context) {
  CollectFilesForFunc(context);
  NNaclFp32Serializer code;
  input_ptr_ = MemoryAllocator::GetInstance()->GetRuntimeAddr(input_tensor_);
  output_ptr_ = MemoryAllocator::GetInstance()->GetRuntimeAddr(output_tensor_);
  auto packed_weight_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(packed_weight_));
  auto bias_data_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(bias_data_));
  code.CodeStruct("conv_parameter", *conv_param_);
  code.CodeStruct("sliding_parameter", *sw_param_);
  code.CodeFunction("InitSlidingParamConvDw", "&sliding_parameter", "&conv_parameter", C8NUM);

  bool need_align = (conv_param_->input_channel_ % C8NUM != 0);
  if (need_align) {
    auto C8 = UP_DIV(conv_param_->input_channel_, C8NUM);
    pack_input_size_ =
      conv_param_->input_batch_ * conv_param_->input_h_ * conv_param_->input_w_ * C8NUM * C8 * DataTypeSize(data_type_);
    packed_input_ = reinterpret_cast<float16 *>(allocator_->Malloc(data_type_, pack_input_size_, kWorkspace));
    MS_CHECK_PTR(packed_input_);
    auto packed_input_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(packed_input_);
    code.CodeFunction("PackNHWCToNHWC8Fp16", input_ptr_, packed_input_str, conv_param_->input_batch_,
                      conv_param_->input_h_ * conv_param_->input_w_, conv_param_->input_channel_);
    pack_output_size_ = conv_param_->output_batch_ * conv_param_->output_h_ * conv_param_->output_w_ * C8NUM * C8 *
                        DataTypeSize(data_type_);
    packed_output_ = reinterpret_cast<float16 *>(allocator_->Malloc(data_type_, pack_output_size_, kWorkspace));
    MS_CHECK_PTR(packed_output_);
  }
  auto packed_input_str = (need_align) ? MemoryAllocator::GetInstance()->GetRuntimeAddr(packed_input_) : input_ptr_;
  auto packed_output_str = (need_align) ? MemoryAllocator::GetInstance()->GetRuntimeAddr(packed_output_) : output_ptr_;
  code.CodeFunction("ConvDwC8Fp16", packed_output_str, packed_input_str, packed_weight_str, bias_data_str,
                    "&conv_parameter", "&sliding_parameter", kDefaultTaskId);

  if (need_align) {
    code.CodeFunction("PackNHWC8ToNHWCFp16", packed_output_str, output_ptr_, conv_param_->output_batch_,
                      conv_param_->output_h_ * conv_param_->output_w_, conv_param_->output_channel_);
  }

  context->AppendCode(code.str());
  return RET_OK;
}
}  // namespace mindspore::lite::micro::nnacl
