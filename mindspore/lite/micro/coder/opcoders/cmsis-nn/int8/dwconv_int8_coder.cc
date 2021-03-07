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

#include "coder/opcoders/cmsis-nn/int8/dwconv_int8_coder.h"
#include <string>
#include "coder/opcoders/serializers/serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/log.h"

namespace mindspore::lite::micro::cmsis {

int DWConvInt8Coder::Prepare(CoderContext *const context) {
  Conv2DBaseCoder::Init();
  MS_CHECK_RET_CODE(micro::Conv2DBaseCoder::CheckLayout(input_tensor_), "Check layout failed.");
  MS_CHECK_RET_CODE(micro::Conv2DBaseCoder::SetQuantParam(), "SetQuantParam failed");
  MS_CHECK_RET_CODE(Conv2DBaseCoder::SetQuantArgs(), "SetQuantArgs failed");
  MS_CHECK_RET_CODE(InitWeightBias(), "InitWeightBias failed");
  MS_CHECK_RET_CODE(SetParameters(), "SetParameters failed");
  CheckSupportOptimize();
  MS_CHECK_RET_CODE(InitTmpBuffer(), "InitTmpBuffer failed");
  return RET_OK;
}

int DWConvInt8Coder::DoCode(CoderContext *const context) {
  Serializer code;
  code.precision(kPrecision);

  std::vector<std::string> h_files;
  std::vector<std::string> c_files;

  h_files.emplace_back("CMSIS/NN/Include/arm_nnfunctions.h");
  code.CodeArray("output_shift", output_shift_, output_ch_);
  code.CodeArray("output_mult", output_mult_, output_ch_);
  switch (optimize_) {
    case Conv_3x3:
      c_files.emplace_back("arm_depthwise_conv_3x3_s8.c");
      Collect(context, h_files, c_files);
      code.CodeFunction("arm_depthwise_conv_3x3_s8", input_tensor_, input_x_, input_y_, input_ch_, filter_tensor_,
                        output_ch_, pad_x_, pad_y_, stride_x_, stride_y_, bias_tensor_, output_tensor_, "output_shift",
                        "output_mult", output_x_, output_y_, output_offset_, input_offset_, output_activation_min_,
                        output_activation_max_, dilation_x_, dilation_y_, "NULL");
      break;
    case Conv_opt:
      // arm_depthwise_conv_s8_opt also depends on arm_depthwise_conv_s8
      c_files.emplace_back("arm_depthwise_conv_s8.c");
      c_files.emplace_back("arm_depthwise_conv_s8_opt.c");
      Collect(context, h_files, c_files);
      code.CodeFunction("arm_depthwise_conv_s8_opt", input_tensor_, input_x_, input_y_, input_ch_, filter_tensor_,
                        output_ch_, kernel_x_, kernel_y_, pad_x_, pad_y_, stride_x_, stride_y_, bias_tensor_,
                        output_tensor_, "output_shift", "output_mult", output_x_, output_y_, output_offset_,
                        input_offset_, output_activation_min_, output_activation_max_, dilation_x_, dilation_y_,
                        "NULL");
      break;
    case Basic:
      c_files.emplace_back("arm_depthwise_conv_s8.c");
      Collect(context, h_files, c_files);
      code.CodeFunction("arm_depthwise_conv_s8", input_tensor_, input_x_, input_y_, input_ch_, filter_tensor_,
                        output_ch_, ch_mult_, kernel_x_, kernel_y_, pad_x_, pad_y_, stride_x_, stride_y_, bias_tensor_,
                        output_tensor_, "output_shift", "output_mult", output_x_, output_y_, output_offset_,
                        input_offset_, output_activation_min_, output_activation_max_, dilation_x_, dilation_y_,
                        "NULL");
      break;
    default:
      MS_LOG(ERROR) << "unsupported optimize_r";
      break;
  }
  context->AppendCode(code.str());
  return RET_OK;
}

int DWConvInt8Coder::InitWeightBias() {
  auto *origin_weight = reinterpret_cast<int8_t *>(filter_tensor_->data_c());
  MS_CHECK_PTR(origin_weight);
  auto pack_weight_size =
    static_cast<size_t>(filter_tensor_->Batch() * filter_tensor_->Height() * filter_tensor_->Width());
  packed_weight_ =
    static_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, pack_weight_size * sizeof(int8_t), kOfflinePackWeight));
  MS_ASSERT(packed_weight_);
  PackNCHWToNHWCInt8(origin_weight, packed_weight_, 1, filter_tensor_->Height() * filter_tensor_->Width(),
                     filter_tensor_->Batch());
  return RET_OK;
}

int DWConvInt8Coder::SetParameters() {
  input_x_ = input_tensor_->Width();
  input_y_ = input_tensor_->Height();
  input_ch_ = input_tensor_->Channel();
  output_ch_ = output_tensor_->Channel();

  // depth_multiplier
  ch_mult_ = output_tensor_->Channel() / input_tensor_->Channel();

  kernel_x_ = filter_tensor_->Width();
  kernel_y_ = filter_tensor_->Height();

  pad_y_ = conv_param_->pad_u_;
  pad_x_ = conv_param_->pad_l_;

  stride_y_ = conv_param_->stride_h_;
  stride_x_ = conv_param_->stride_w_;

  QuantArg input_quant_arg = input_tensor_->quant_params().at(0);
  QuantArg output_quant_arg = output_tensor_->quant_params().at(0);

  output_x_ = output_tensor_->Width();
  output_y_ = output_tensor_->Height();
  input_offset_ = -input_quant_arg.zeroPoint;
  output_offset_ = output_quant_arg.zeroPoint;

  CalculateActivationRangeQuantized(conv_param_->act_type_ == ActType_Relu, conv_param_->act_type_ == ActType_Relu6,
                                    output_quant_arg.zeroPoint, output_quant_arg.scale, &output_activation_min_,
                                    &output_activation_max_);
  return RET_OK;
}

void DWConvInt8Coder::CheckSupportOptimize() {
  if (ch_mult_ == 1) {
    if ((kernel_x_ == 3) && (kernel_y_ == 3) && (pad_y_ <= 1)) {
      optimize_ = Conv_3x3;
      buffer_size_ = 0;
    } else {
      optimize_ = Conv_opt;
      buffer_size_ = input_ch_ * kernel_x_ * kernel_y_ * sizeof(int16_t);
    }
  } else {
    optimize_ = Basic;
    buffer_size_ = 0;
  }
}

int DWConvInt8Coder::InitTmpBuffer() {
  if (buffer_size_ != 0) {
    buffer = static_cast<int16_t *>(allocator_->Malloc(kNumberTypeInt16, buffer_size_, kWorkspace));
    MS_CHECK_PTR(buffer);
  } else {
    buffer = nullptr;
  }
  return 0;
}

}  // namespace mindspore::lite::micro::cmsis
