/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "coder/opcoders/nnacl/int8/convolution_depthwise_int8_coder.h"
#include <string>
#include "coder/log.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "nnacl/int8/conv_depthwise_int8.h"

namespace mindspore::lite::micro {

int ConvolutionDepthwiseINT8Coder::Prepare(CoderContext *const context) {
  Conv2DBaseCoder::Init();
  // init sliding window param
  MS_CHECK_RET_CODE(SetQuantParam(), "Set quant param failed.");
  MS_CHECK_RET_CODE(InitWeightBias(context), "dwconvolution do init weightbais failed");
  MS_CHECK_RET_CODE(InitBuffer(context), "dwconvolution do init buffer failed");
  return RET_OK;
}

int ConvolutionDepthwiseINT8Coder::InitBuffer(CoderContext *const context) {
  // malloc pack input and output buffer
  row_buffer_size_ = thread_num_ * conv_param_->output_w_ * conv_param_->output_channel_ * sizeof(int32_t);
  row_buffer_ = reinterpret_cast<int32_t *>(allocator_->Malloc(kNumberTypeInt32, row_buffer_size_, kWorkspace));
  MS_CHECK_PTR(row_buffer_);
  return RET_OK;
}

int ConvolutionDepthwiseINT8Coder::InitWeightBias(CoderContext *const context) {
  // init weight, int8 -> int16
  int channel = filter_tensor_->Batch();
  int pack_weight_size = channel * filter_tensor_->Height() * filter_tensor_->Width();
  auto tmp_weight_data_size = static_cast<size_t>(pack_weight_size * sizeof(int8_t));

  nnacl::NNaclInt8Serializer code;

  int8_t *tmp_weight = reinterpret_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, kOnlineSize, kOnlinePackWeight));
  MS_CHECK_PTR(tmp_weight);
  code.CodeMallocExpression(tmp_weight, tmp_weight_data_size);
  code.CodeFunction("memset", tmp_weight, 0, tmp_weight_data_size);
  code.CodeFunction("PackNCHWToNHWCInt8", filter_tensor_, tmp_weight, 1,
                    filter_tensor_->Height() * filter_tensor_->Width(), filter_tensor_->Batch());
  int weight_zp = conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_;
  auto packed_weight_data_size = static_cast<size_t>(pack_weight_size * sizeof(int16_t));
  packed_weight_ = reinterpret_cast<int16_t *>(allocator_->Malloc(kNumberTypeInt16, kOnlineSize, kOnlinePackWeight));
  MS_CHECK_PTR(packed_weight_);
  code.CodeMallocExpression(packed_weight_, packed_weight_data_size);
  code << "for (int i = 0; i < " << filter_tensor_->ElementsNum() << "; i++) {\n";
  code << "  " << allocator_->GetRuntimeAddr(packed_weight_) << "[i] = (int16_t)("
       << allocator_->GetRuntimeAddr(tmp_weight) << "[i] - " << weight_zp << ");\n";
  code << "}\n";

  auto channel_data_size = static_cast<size_t>(channel * sizeof(int32_t));
  bias_data_ = reinterpret_cast<int32_t *>(allocator_->Malloc(kNumberTypeInt32, kOnlineSize, kOnlinePackWeight));
  MS_CHECK_PTR(bias_data_);
  code.CodeMallocExpression(bias_data_, channel_data_size);
  code.CodeFunction("memset", bias_data_, 0, channel_data_size);
  // init bias
  if (input_tensors_.size() == kInputSize2) {
    code.CodeFunction("memcpy", bias_data_, bias_tensor_, bias_tensor_->ElementsNum() * sizeof(int32_t));
  }
  context->AppendInitCode(code.str());
  return RET_OK;
}

int ConvolutionDepthwiseINT8Coder::DoCode(CoderContext *const context) {
  MS_CHECK_TRUE(conv_param_->input_channel_ == conv_param_->output_channel_,
                "Only support input channel equals output channel.");
  Collect(
    context,
    {"nnacl/int8/conv_depthwise_int8.h", "nnacl/int8/pack_int8.h", "wrapper/int8/convolution_depthwise_int8_wrapper.h"},
    {"conv_depthwise_int8.c", "fixed_point.c", "pack_int8.c", "conv_int8.c", "winograd_transform.c",
     "convolution_depthwise_int8_wrapper.c"},
    {"ConvDwInt8Row.S", "ConvDwInt8PostAlign4.S", "ConvDwInt8PostAlign4PerChannel.S", "ConvDwInt8Center.S",
     "DeconvDwInt8Center.S", "DeconvDwInt8Post.S"});
  if (target_ == kARM64) {
    Collect(context, {}, {},
            {"ConvDw3x3Int8.S", "ConvDw3x3Int8Corner.S", "ConvDw3x3Int8Horizontal.S", "ConvDw3x3Int8Stride2.S",
             "ConvDw3x3Int8Vertical.S", "MatmulDpInt8Opt.S", "MatmulOptR4Int8.S"});
  }
  nnacl::NNaclInt8Serializer code;
  code.precision(kPrecision);
  // call the op function
  code.CodeFunction("memset", row_buffer_, 0, row_buffer_size_);
  code.CodeStruct("conv_param", *conv_param_);
  code.CodeBaseStruct("ConvDepthwiseInt8Args", kRunArgs, output_tensor_, row_buffer_, input_tensor_, packed_weight_,
                      bias_data_, "&conv_param");
  if (support_parallel_) {
    code.CodeFunction(kParallelLaunch, gThreadPool, "ConvDepthwiseInt8Run", kRunArgsAddr, "conv_param.thread_num_");
  } else {
    code.CodeFunction("ConvDepthwiseInt8Run", kRunArgsAddr, kDefaultTaskId);
  }
  context->AppendCode(code.str());
  return RET_OK;
}

}  // namespace mindspore::lite::micro
