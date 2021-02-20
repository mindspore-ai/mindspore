/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "micro/coder/opcoders/nnacl/fp32/convolution_depthwise_fp32_coder.h"
#include <string>
#include "micro/coder/log.h"
#include "micro/coder/opcoders/file_collector.h"
#include "micro/coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"

using mindspore::schema::PrimitiveType_DepthwiseConv2D;
namespace mindspore::lite::micro::nnacl {
int ConvolutionDepthwiseFP32Coder::Prepare(CoderContext *const context) {
  Conv2DBaseCoder::Init();
  MS_CHECK_RET_CODE(InitWeightBias(), "dwconvolution do init weightbais failed");
  conv_param_->thread_num_ = MSMIN(thread_num_, conv_param_->output_h_);
  return RET_OK;
}

int ConvolutionDepthwiseFP32Coder::InitWeightBias() {
  // init weight: o, h, w, i; o == group, i == 1
  auto *origin_weight = reinterpret_cast<float *>(filter_tensor_->data_c());
  int channel = filter_tensor_->Batch();
  size_t pack_weight_size = filter_tensor_->Batch() * filter_tensor_->Height() * filter_tensor_->Width();
  size_t packed_weight_data_size = pack_weight_size * sizeof(float);
  packed_weight_ =
    reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, packed_weight_data_size, kOfflinePackWeight));
  MS_CHECK_PTR(packed_weight_);
  MS_CHECK_RET_CODE(memset_s(packed_weight_, packed_weight_data_size, 0, packed_weight_data_size),
                    "memset packed weight failed!");
  PackNCHWToNHWCFp32(origin_weight, packed_weight_, 1, filter_tensor_->Height() * filter_tensor_->Width(), channel);

  auto channel_size = static_cast<size_t>(channel);
  auto bias_size = static_cast<size_t>(channel_size * sizeof(float));
  bias_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, bias_size, kOfflinePackWeight));
  MS_CHECK_PTR(bias_);
  MS_CHECK_RET_CODE(memset_s(bias_, bias_size, 0, bias_size), "memset bias failed!");
  // init bias
  if (input_tensors_.size() == kInputSize2) {
    auto *ori_bias = reinterpret_cast<float *>(bias_tensor_->data_c());
    MS_CHECK_TRUE(bias_tensor_->ElementsNum() > 0, "invalid bias length");
    MS_CHECK_RET_CODE(memcpy_s(bias_, static_cast<size_t>(bias_tensor_->ElementsNum() * sizeof(float)), ori_bias,
                               static_cast<size_t>(bias_tensor_->ElementsNum() * sizeof(float))),
                      "memcpy_s bias failed!");
  }
  return RET_OK;
}

int ConvolutionDepthwiseFP32Coder::DoCode(CoderContext *const context) {
  MS_CHECK_TRUE(conv_param_->input_channel_ == conv_param_->output_channel_,
                "Only support input channel equals output channel.");
  // generate code .h .c
  Collect(context, {"nnacl/fp32/conv_depthwise.h"}, {"conv_depthwise.c"});

  nnacl::NNaclFp32Serializer code;
  // call the op function
  code.CodeStruct("conv_parameter", *conv_param_);
  int task_id = 0;
  code.CodeFunction("ConvDw", output_tensor_, input_tensor_, packed_weight_, bias_, "&conv_parameter", task_id);
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_DepthwiseConv2D,
                   CPUOpCoderCreator<ConvolutionDepthwiseFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
