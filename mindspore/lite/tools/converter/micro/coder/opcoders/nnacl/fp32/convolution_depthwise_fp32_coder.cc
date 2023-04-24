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

#include "coder/opcoders/nnacl/fp32/convolution_depthwise_fp32_coder.h"
#include <string>
#include "coder/log.h"
#include "coder/opcoders/parallel.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "src/common/utils.h"

namespace mindspore::lite::micro::nnacl {
int ConvolutionDepthwiseFP32Coder::Prepare(CoderContext *const context) {
  MS_CHECK_RET_CODE(Conv2DBaseCoder::Init(), "Conv2DBaseCoder::Init() failed!");
  MS_CHECK_RET_CODE(InitParameter(), "dwconvolution do InitParamter failed");
  if (data_type_ == kNumberTypeFloat32) {
    is_weight_online_ = Configurator::GetInstance()->keep_original_weight();
  }
  if (is_weight_online_) {
    MS_CHECK_RET_CODE(InitWeightBiasOnline(), "dwconvolution do InitWeightBiasOnline failed");
  } else {
    MS_CHECK_RET_CODE(InitWeightBiasOffline(), "dwconvolution do InitWeightBiasOffline failed");
  }
  conv_param_->thread_num_ = MSMIN(thread_num_, conv_param_->output_h_);
  return RET_OK;
}

int ConvolutionDepthwiseFP32Coder::InitParameter() {
  auto shape = filter_tensor_->shape();
  MS_CHECK_TRUE_MSG(shape.size() == C4NUM, RET_ERROR, "Conv: filter-weight's shape must be 4D.");
  packed_weight_size_ =
    filter_tensor_->Batch() * filter_tensor_->Height() * filter_tensor_->Width() * DataTypeSize(data_type_);
  packed_bias_size_ = filter_tensor_->Batch() * DataTypeSize(data_type_);
  return RET_OK;
}

int ConvolutionDepthwiseFP32Coder::InitWeightBiasOffline() {
  auto *origin_weight = reinterpret_cast<float *>(filter_tensor_->data());
  MS_CHECK_PTR(origin_weight);
  int channel = filter_tensor_->Batch();
  packed_weight_ = allocator_->Malloc(data_type_, packed_weight_size_, kOfflinePackWeight);
  MS_CHECK_PTR(packed_weight_);
  MS_CHECK_RET_CODE(memset_s(packed_weight_, packed_weight_size_, 0, packed_weight_size_),
                    "memset packed weight failed!");
  PackNCHWToNHWCFp32(origin_weight, reinterpret_cast<float *>(packed_weight_), 1,
                     filter_tensor_->Height() * filter_tensor_->Width(), channel, kDefaultTaskId, 0);

  bias_ = allocator_->Malloc(data_type_, packed_bias_size_, kOfflinePackWeight);
  MS_CHECK_PTR(bias_);
  MS_CHECK_RET_CODE(memset_s(bias_, packed_bias_size_, 0, packed_bias_size_), "memset bias failed!");
  // init bias
  if (input_tensors_.size() == kInputSize2) {
    auto *ori_bias = reinterpret_cast<float *>(bias_tensor_->data());
    MS_CHECK_TRUE(bias_tensor_->ElementsNum() > 0, "invalid bias length");
    MS_CHECK_RET_CODE(memcpy_s(bias_, packed_bias_size_, ori_bias, bias_tensor_->Size()), "memcpy_s bias failed!");
  }
  return RET_OK;
}

int ConvolutionDepthwiseFP32Coder::InitWeightBiasOnline() {
  packed_weight_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
  MS_CHECK_PTR(packed_weight_);
  bias_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
  MS_CHECK_PTR(bias_);
  return RET_OK;
}

void ConvolutionDepthwiseFP32Coder::InitCodeOnline(CoderContext *const context) {
  if (!is_weight_online_) {
    return;
  }
  Collect(context,
          {
            "nnacl/fp32/pack_fp32.h",
          },
          {"pack_fp32.c"});
  NNaclFp32Serializer init_code;
  init_code.CodeBufferOffsetExpression(packed_weight_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), packed_weight_size_);
  auto filter_str = allocator_->GetRuntimeAddr(filter_tensor_);
  init_code.CodeFunction("PackNCHWToNHWCFp32", filter_str, reinterpret_cast<float *>(packed_weight_), 1,
                         filter_tensor_->Height() * filter_tensor_->Width(), filter_tensor_->Batch(), 0, 0);
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

void ConvolutionDepthwiseFP32Coder::CollectFilesForFunc(CoderContext *const context) {
  if (target_ != kX86) {
    Collect(context, {}, {},
            {
              "ConvDwFp32Center.S",
              "ConvDwFp32Border.S",
              "DeconvDwFp32Center.S",
              "ConvDwFp32Row.S",
            });
  }
  Collect(context,
          {
            "nnacl/fp32/conv_depthwise_fp32.h",
          },
          {
            "conv_depthwise_fp32.c",
            "activation_fp32.c",
          },
          {});
}

int ConvolutionDepthwiseFP32Coder::DoCode(CoderContext *const context) {
  MS_CHECK_TRUE(conv_param_->input_channel_ == conv_param_->output_channel_,
                "Only support input channel equals output channel.");
  // generate code .h .c
  CollectFilesForFunc(context);
  InitCodeOnline(context);
  nnacl::NNaclFp32Serializer code;
  // call the op function
  std::string param_name = "conv_parameter";
  code.CodeStruct(param_name, *conv_param_);
  if (support_parallel_) {
    code << "    " << param_name << ".op_parameter_.thread_num_ = 1;\n";
    code << "    " << param_name << ".thread_num_ = 1;\n";
  }
  auto packed_weight_str = "(float *)" + MemoryAllocator::GetInstance()->GetRuntimeAddr(packed_weight_);
  auto bias_str = "(float *)" + MemoryAllocator::GetInstance()->GetRuntimeAddr(bias_);
  code.CodeFunction("ConvDw", output_tensor_, input_tensor_, packed_weight_str, bias_str, "&" + param_name,
                    kDefaultTaskId);
  context->AppendCode(code.str());
  return RET_OK;
}
}  // namespace mindspore::lite::micro::nnacl
