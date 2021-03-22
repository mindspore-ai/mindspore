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

#include "coder/opcoders/base/conv2d_base_coder.h"
#include <string>
#include <vector>
#include "nnacl/fp32/winograd_utils.h"
#include "nnacl/int8/quantize.h"
#include "coder/log.h"
namespace mindspore::lite::micro {

Conv2DBaseCoder::~Conv2DBaseCoder() {
  FreeConvQuantParams();
  conv_param_ = nullptr;
  conv_quant_arg_ = nullptr;
  filter_tensor_ = nullptr;
  bias_tensor_ = nullptr;
}

void Conv2DBaseCoder::FreeConvQuantParams() {
  if (conv_quant_arg_ == nullptr) {
    return;
  }
  if (conv_quant_arg_->real_multiplier_ != nullptr) {
    free(conv_quant_arg_->real_multiplier_);
    conv_quant_arg_->real_multiplier_ = nullptr;
  }
  if (conv_quant_arg_->left_shift_ != nullptr) {
    free(conv_quant_arg_->left_shift_);
    conv_quant_arg_->left_shift_ = nullptr;
  }
  if (conv_quant_arg_->right_shift_ != nullptr) {
    free(conv_quant_arg_->right_shift_);
    conv_quant_arg_->right_shift_ = nullptr;
  }
  if (conv_quant_arg_->quant_multiplier_ != nullptr) {
    free(conv_quant_arg_->quant_multiplier_);
    conv_quant_arg_->quant_multiplier_ = nullptr;
  }
  if (conv_quant_arg_->out_act_min_ != nullptr) {
    free(conv_quant_arg_->out_act_min_);
    conv_quant_arg_->out_act_min_ = nullptr;
  }
  if (conv_quant_arg_->out_act_max_ != nullptr) {
    free(conv_quant_arg_->out_act_max_);
    conv_quant_arg_->out_act_max_ = nullptr;
  }
  if (conv_quant_arg_->input_quant_args_ != nullptr) {
    free(conv_quant_arg_->input_quant_args_);
    conv_quant_arg_->input_quant_args_ = nullptr;
  }
  if (conv_quant_arg_->filter_quant_args_ != nullptr) {
    free(conv_quant_arg_->filter_quant_args_);
    conv_quant_arg_->filter_quant_args_ = nullptr;
  }
  if (conv_quant_arg_->output_quant_args_ != nullptr) {
    free(conv_quant_arg_->output_quant_args_);
    conv_quant_arg_->output_quant_args_ = nullptr;
  }
}

int Conv2DBaseCoder::MallocConvQuantParams(size_t input_arg_num, size_t filter_arg_num, size_t output_arg_num) {
  MS_CHECK_TRUE(input_arg_num > 0, "invalid value of input_arg_num");
  MS_CHECK_TRUE(filter_arg_num > 0, "invalid value of filter_arg_num");
  MS_CHECK_TRUE(output_arg_num > 0, "invalid value of output_arg_num");
  conv_quant_arg_->input_quant_args_ = reinterpret_cast<::QuantArg *>(malloc(input_arg_num * sizeof(::QuantArg)));
  if (conv_quant_arg_->input_quant_args_ == nullptr) {
    FreeConvQuantParams();
    return RET_ERROR;
  }
  conv_quant_arg_->filter_quant_args_ = reinterpret_cast<::QuantArg *>(malloc(filter_arg_num * sizeof(::QuantArg)));
  if (conv_quant_arg_->filter_quant_args_ == nullptr) {
    FreeConvQuantParams();
    return RET_ERROR;
  }
  conv_quant_arg_->output_quant_args_ = reinterpret_cast<::QuantArg *>(malloc(output_arg_num * sizeof(::QuantArg)));
  if (conv_quant_arg_->output_quant_args_ == nullptr) {
    FreeConvQuantParams();
    return RET_ERROR;
  }
  return RET_OK;
}

std::string Conv2DBaseCoder::LayoutTransformFp32(schema::Format src_format, schema::Format dst_format) {
  std::string ret;
  if (src_format == schema::Format_NHWC && dst_format == schema::Format_NC4HW4) {
    ret = "PackNHWCToNC4HW4Fp32";
  } else if (src_format == schema::Format_NHWC && dst_format == schema::Format_NHWC4) {
    ret = "PackNHWCToNHWC4Fp32";
  } else if (src_format == schema::Format_NC4HW4 && dst_format == schema::Format_NHWC4) {
    ret = "PackNC4HW4ToNHWC4Fp32";
  } else if (src_format == schema::Format_NCHW && dst_format == schema::Format_NC4HW4) {
    ret = "PackNCHWToNC4HW4Fp32";
  } else if (src_format == schema::Format_NC4HW4 && dst_format == schema::Format_NHWC) {
    ret = "PackNC4HW4ToNHWCFp32";
  } else {
    MS_LOG(ERROR) << "Unsupported transform from " << schema::EnumNameFormat(src_format) << " to "
                  << schema::EnumNameFormat(dst_format);
  }
  return ret;
}

std::string Conv2DBaseCoder::LayoutTransformInt8(schema::Format src_format, schema::Format dst_format) {
  std::string ret;
  if (src_format == schema::Format_NHWC && dst_format == schema::Format_NHWC4) {
    ret = "PackNHWCToNHWC4Int8";
  } else {
    MS_LOG(ERROR) << "Unsupported transform from " << schema::EnumNameFormat(src_format) << " to "
                  << schema::EnumNameFormat(dst_format);
  }
  return ret;
}

std::string Conv2DBaseCoder::LayoutTransform(TypeId data_type, schema::Format src_format, schema::Format dst_format) {
  std::string ret;
  switch (data_type) {
    case kNumberTypeInt8:
      ret = LayoutTransformInt8(src_format, dst_format);
      break;
    case kNumberTypeFloat32:
      ret = LayoutTransformFp32(src_format, dst_format);
      break;
    default:
      MS_LOG(WARNING) << "unsupported data type";
  }
  return ret;
}

int Conv2DBaseCoder::SetIfPerChannel() {
  auto input_channel = static_cast<size_t>(filter_tensor_->Channel());
  auto output_channel = static_cast<size_t>(filter_tensor_->Batch());

  uint8_t per_channel = 0b0;
  if (conv_quant_arg_->input_arg_num_ != kPerTensor) {
    MS_CHECK_TRUE(conv_quant_arg_->input_arg_num_ == input_channel,
                  "input per channel quant param length is not equal to input channel.");
    per_channel = per_channel | INPUT_PER_CHANNEL;
  }

  if (conv_quant_arg_->filter_arg_num_ != kPerTensor) {
    MS_CHECK_TRUE(conv_quant_arg_->filter_arg_num_ == output_channel,
                  "weight per channel quant param length is not equal to filter num.");
    per_channel = per_channel | FILTER_PER_CHANNEL;
  }

  if (conv_quant_arg_->output_arg_num_ != kPerTensor) {
    MS_CHECK_TRUE(conv_quant_arg_->output_arg_num_ != output_channel,
                  "output per channel quant param length is not equal to output channel.");
    per_channel = per_channel | OUTPUT_PER_CHANNEL;
  }
  conv_quant_arg_->per_channel_ = per_channel;
  return RET_OK;
}

int Conv2DBaseCoder::MallocQuantParam() {
  conv_quant_arg_ = &conv_param_->conv_quant_arg_;
  size_t input_arg_num = input_tensor_->quant_params().size();
  size_t filter_arg_num = filter_tensor_->quant_params().size();
  size_t output_arg_num = output_tensor_->quant_params().size();
  conv_quant_arg_->input_arg_num_ = input_arg_num;
  conv_quant_arg_->filter_arg_num_ = filter_arg_num;
  conv_quant_arg_->output_arg_num_ = output_arg_num;
  MallocConvQuantParams(input_arg_num, filter_arg_num, output_arg_num);
  return RET_OK;
}

int Conv2DBaseCoder::SetInputTensorQuantParam() {
  size_t in_arg_num = conv_quant_arg_->input_arg_num_;
  if (in_arg_num == kPerTensor) {
    QuantArg input_quant_arg = input_tensor_->quant_params().at(0);
    conv_quant_arg_->input_quant_args_[0].zp_ = input_quant_arg.zeroPoint;
    conv_quant_arg_->input_quant_args_[0].scale_ = static_cast<float>(input_quant_arg.scale);
    return RET_OK;
  } else {
    // per channel
    MS_LOG(ERROR) << "Not Support Per Channel for input now.";
    return RET_ERROR;
  }
}

int Conv2DBaseCoder::SetFilterTensorQuantParam() {
  size_t weight_arg_num = conv_quant_arg_->filter_arg_num_;
  if (weight_arg_num == kPerTensor) {
    QuantArg weight_quant_arg = filter_tensor_->quant_params().at(0);
    conv_quant_arg_->filter_quant_args_[0].zp_ = weight_quant_arg.zeroPoint;
    conv_quant_arg_->filter_quant_args_[0].scale_ = static_cast<float>(weight_quant_arg.scale);
  } else {
    std::vector<QuantArg> weight_quant_arg = filter_tensor_->quant_params();
    for (int i = 0; i < static_cast<int>(weight_arg_num); ++i) {
      conv_quant_arg_->filter_quant_args_[i].zp_ = weight_quant_arg[i].zeroPoint;
      conv_quant_arg_->filter_quant_args_[i].scale_ = static_cast<float>(weight_quant_arg[i].scale);
    }
  }
  return RET_OK;
}

int Conv2DBaseCoder::SetOutputTensorQuantParam() {
  size_t out_arg_num = conv_quant_arg_->output_arg_num_;
  if (out_arg_num == kPerTensor) {
    QuantArg output_quant_arg = output_tensor_->quant_params().at(0);
    conv_quant_arg_->output_quant_args_[0].zp_ = output_quant_arg.zeroPoint;
    conv_quant_arg_->output_quant_args_[0].scale_ = static_cast<float>(output_quant_arg.scale);
  } else {
    MS_LOG(ERROR) << "Not Support Per Channel for input now.";
    return RET_ERROR;
  }
  return RET_OK;
}

int Conv2DBaseCoder::SetQuantMultiplier() {
  // now only support weight tensor is per channel, others are per tensor.
  int weight_arg_num = kPerTensor;
  if (conv_quant_arg_->per_channel_ & FILTER_PER_CHANNEL) {
    weight_arg_num = conv_quant_arg_->filter_arg_num_;
  }
  conv_quant_arg_->real_multiplier_ = reinterpret_cast<double *>(malloc(weight_arg_num * sizeof(double)));
  if (conv_quant_arg_->real_multiplier_ == nullptr) {
    FreeConvQuantParams();
    return RET_ERROR;
  }
  conv_quant_arg_->left_shift_ = reinterpret_cast<int32_t *>(malloc(weight_arg_num * sizeof(int32_t)));
  if (conv_quant_arg_->left_shift_ == nullptr) {
    FreeConvQuantParams();
    return RET_ERROR;
  }
  conv_quant_arg_->right_shift_ = reinterpret_cast<int32_t *>(malloc(weight_arg_num * sizeof(int32_t)));
  if (conv_quant_arg_->right_shift_ == nullptr) {
    FreeConvQuantParams();
    return RET_ERROR;
  }
  conv_quant_arg_->quant_multiplier_ = reinterpret_cast<int32_t *>(malloc(weight_arg_num * sizeof(int32_t)));
  if (conv_quant_arg_->quant_multiplier_ == nullptr) {
    FreeConvQuantParams();
    return RET_ERROR;
  }
  conv_quant_arg_->out_act_min_ = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t)));
  if (conv_quant_arg_->out_act_min_ == nullptr) {
    FreeConvQuantParams();
    return RET_ERROR;
  }
  conv_quant_arg_->out_act_max_ = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t)));
  if (conv_quant_arg_->out_act_max_ == nullptr) {
    FreeConvQuantParams();
    return RET_ERROR;
  }
  for (int i = 0; i < weight_arg_num; ++i) {
    const auto in_scale =
      static_cast<double>(conv_quant_arg_->input_quant_args_[0].scale_ * conv_quant_arg_->filter_quant_args_[i].scale_);
    double real_multiplier = in_scale / static_cast<double>(conv_quant_arg_->output_quant_args_[0].scale_);
    conv_quant_arg_->real_multiplier_[i] = real_multiplier;
    if (conv_quant_arg_->quant_multiplier_mode_ == Method_SinglePrecision) {
      QuantizeRoundParameterWithSinglePrecision(real_multiplier, &conv_quant_arg_->quant_multiplier_[i],
                                                &conv_quant_arg_->left_shift_[i], &conv_quant_arg_->right_shift_[i]);
    } else if (conv_quant_arg_->quant_multiplier_mode_ == Method_DoublePrecision) {
      QuantizeRoundParameterWithDoublePrecision(real_multiplier, &conv_quant_arg_->quant_multiplier_[i],
                                                &conv_quant_arg_->left_shift_[i], &conv_quant_arg_->right_shift_[i]);
    }
  }
  return RET_OK;
}

int Conv2DBaseCoder::CheckResizeValid() {
  // ===============check in channel================= //
  int32_t filter_in_channel = filter_tensor_->Channel();
  int32_t resize_in_channel = input_tensor_->Channel();
  MS_CHECK_TRUE(filter_in_channel == resize_in_channel,
                "Channel of resized input should be equal to in channel of filter.");
  return RET_OK;
}

void Conv2DBaseCoder::SetRoundingAndMultipilerMode() {
  auto input_quant_arg = input_tensor_->quant_params().front();
  int round_type = input_quant_arg.roundType;
  switch (round_type) {
    case 1:
      conv_quant_arg_->round_mode_ = Rounding_Away_from_zero;
      break;
    case 2:
      conv_quant_arg_->round_mode_ = Rounding_Up;
      break;
    default:
      conv_quant_arg_->round_mode_ = Rounding_No;
  }
  int cal_multiplier_type = input_quant_arg.multiplier;
  switch (cal_multiplier_type) {
    case 0:
      conv_quant_arg_->quant_multiplier_mode_ = Method_SinglePrecision;
      break;
    case 1:
      conv_quant_arg_->quant_multiplier_mode_ = Method_DoublePrecision;
      break;
    default:
      conv_quant_arg_->quant_multiplier_mode_ = Method_No;
  }
}

int Conv2DBaseCoder::SetQuantParam() {
  MS_CHECK_RET_CODE(MallocQuantParam(), "Malloc quant param failed.");
  MS_CHECK_RET_CODE(SetInputTensorQuantParam(), "Set Input Tensor Quant Param Failed.");
  MS_CHECK_RET_CODE(SetFilterTensorQuantParam(), "Set Filter Tensor Quant Param Failed.");
  MS_CHECK_RET_CODE(SetOutputTensorQuantParam(), "Set Output Tensor Quant Param Failed.");
  MS_CHECK_RET_CODE(SetIfPerChannel(), "Set if per tensor channel failed.");
  SetRoundingAndMultipilerMode();
  MS_CHECK_RET_CODE(SetQuantMultiplier(), "Set Quant Multiplier Failed.");
  bool relu = conv_param_->act_type_ == ActType_Relu;
  bool relu6 = conv_param_->act_type_ == ActType_Relu6;
  CalculateActivationRangeQuantized(relu, relu6, conv_param_->conv_quant_arg_.output_quant_args_[0].zp_,
                                    conv_param_->conv_quant_arg_.output_quant_args_[0].scale_,
                                    &conv_param_->conv_quant_arg_.out_act_min_[0],
                                    &conv_param_->conv_quant_arg_.out_act_max_[0]);

  return RET_OK;
}

int Conv2DBaseCoder::Init() {
  this->conv_param_ = reinterpret_cast<ConvParameter *>(parameter_);
  filter_tensor_ = input_tensors_.at(kWeightIndex);
  MS_CHECK_PTR(filter_tensor_);
  MS_CHECK_PTR(filter_tensor_->data_c());
  if (input_tensors_.size() == kInputSize2) {
    bias_tensor_ = input_tensors_.at(kBiasIndex);
    MS_CHECK_PTR(bias_tensor_);
    MS_CHECK_PTR(bias_tensor_->data_c());
  } else {
    MS_CHECK_TRUE(input_tensors_.size() == kInputSize1, "wrong input size");
  }

  conv_param_->input_batch_ = input_tensor_->Batch();
  conv_param_->input_h_ = input_tensor_->Height();
  conv_param_->input_w_ = input_tensor_->Width();
  conv_param_->input_channel_ = input_tensor_->Channel();
  conv_param_->output_batch_ = output_tensor_->Batch();
  conv_param_->output_h_ = output_tensor_->Height();
  conv_param_->output_w_ = output_tensor_->Width();
  conv_param_->output_channel_ = output_tensor_->Channel();
  return RET_OK;
}

int Conv2DBaseCoder::CheckLayout(lite::Tensor *input_tensor) {
  mindspore::TypeId data_type = input_tensor->data_type();
  schema::Format input_format = input_tensor->format();
  schema::Format execute_format = schema::Format_NHWC4;
  convert_func_ = LayoutTransform(data_type, input_format, execute_format);
  MS_CHECK_TRUE(!convert_func_.empty(), "layout convert func is nullptr.");
  return RET_OK;
}
}  // namespace mindspore::lite::micro
