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

#include "src/runtime/kernel/arm/base/convolution_base.h"
#include <float.h>
#include "schema/model_generated.h"
#include "src/kernel_factory.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType;
using mindspore::schema::PadMode;

namespace mindspore::kernel {
ConvolutionBaseCPUKernel::~ConvolutionBaseCPUKernel() {
  if (bias_data_ != nullptr) {
    free(bias_data_);
    bias_data_ = nullptr;
  }
  if (nhwc4_input_ != nullptr) {
    free(nhwc4_input_);
    nhwc4_input_ = nullptr;
  }
}

void ConvolutionBaseCPUKernel::FreeQuantParam() {
  ConvQuantArg *conv_quant_arg_ = &conv_param_->conv_quant_arg_;
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
  }
  if (conv_quant_arg_->filter_quant_args_ != nullptr) {
    free(conv_quant_arg_->filter_quant_args_);
  }
  if (conv_quant_arg_->output_quant_args_ != nullptr) {
    free(conv_quant_arg_->output_quant_args_);
  }
}

int ConvolutionBaseCPUKernel::Init() {
  auto input = this->in_tensors_.front();
  auto output = this->out_tensors_.front();
  conv_param_->input_batch_ = input->Batch();
  conv_param_->input_h_ = input->Height();
  conv_param_->input_w_ = input->Width();
  conv_param_->input_channel_ = input->Channel();
  conv_param_->output_batch_ = output->Batch();
  conv_param_->output_h_ = output->Height();
  conv_param_->output_w_ = output->Width();
  conv_param_->output_channel_ = output->Channel();
  conv_param_->thread_num_ = ctx_->thread_num_;
  return RET_OK;
}

int ConvolutionBaseCPUKernel::CheckLayout(lite::tensor::Tensor *input_tensor) {
  auto data_type = input_tensor->data_type();
  auto input_format = input_tensor->GetFormat();
  schema::Format execute_format = schema::Format_NHWC4;
  convert_func_ = LayoutTransform(data_type, input_format, execute_format);
  if (convert_func_ == nullptr) {
    MS_LOG(ERROR) << "layout convert func is nullptr.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionBaseCPUKernel::SetIfPerChannel() {
  uint8_t per_channel = 0b0;
  if (conv_quant_arg_->input_arg_num_ != kPerTensor) {
    int in_channel = conv_param_->input_channel_;
    if (conv_quant_arg_->input_arg_num_ != in_channel) {
      MS_LOG(ERROR) << "input per channel quant param length is not equal to input channel.";
      return RET_ERROR;
    }
    per_channel = per_channel | INPUT_PER_CHANNEL;
  }

  if (conv_quant_arg_->filter_arg_num_ != kPerTensor) {
    int filter_num = conv_param_->output_channel_;
    if (conv_quant_arg_->filter_arg_num_ != filter_num) {
      MS_LOG(ERROR) << "weight per channel quant param length is not equal to filter num.";
      return RET_ERROR;
    }
    per_channel = per_channel | FILTER_PER_CHANNEL;
  }

  if (conv_quant_arg_->output_arg_num_ != kPerTensor) {
    int out_channel = conv_param_->output_channel_;
    if (conv_quant_arg_->output_arg_num_ != out_channel) {
      MS_LOG(ERROR) << "output per channel quant param length is not equal to output channel.";
      return RET_ERROR;
    }
    per_channel = per_channel | OUTPUT_PER_CHANNEL;
  }
  conv_quant_arg_->per_channel_ = per_channel;
  return RET_OK;
}

int ConvolutionBaseCPUKernel::SetIfAsymmetric() {
  uint8_t asymmetric = 0b0;
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  auto filter_ele_num = filter_tensor->ElementsNum();
  auto filter_data = reinterpret_cast<int8_t *>(filter_tensor->Data());
  int min_value = INT8_MAX;
  int max_value = INT8_MIN;
  for (int i = 0; i < filter_ele_num; ++i) {
    min_value = min_value < filter_data[i] ? min_value : filter_data[i];
    max_value = max_value > filter_data[i] ? max_value : filter_data[i];
  }
  if (conv_quant_arg_->filter_arg_num_ == kPerTensor) {
    auto filter_zp = conv_quant_arg_->filter_quant_args_[0].zp_;
    if (filter_zp != 0 && min_value >= -128 && max_value <= 127) {
      asymmetric = asymmetric | FILTER_ASYMMETRIC;
    }
  } else {
    auto filter_arg = conv_quant_arg_->filter_quant_args_;
    for (int i = 0; i < conv_param_->output_channel_; ++i) {
      if (filter_arg[i].zp_ != 0 && min_value >= -128 && max_value <= 127) {
        asymmetric = asymmetric | FILTER_ASYMMETRIC;
      }
    }
  }
  conv_quant_arg_->asymmetric_ = asymmetric;
  return RET_OK;
}

int ConvolutionBaseCPUKernel::MallocQuantParam() {
  conv_quant_arg_ = &conv_param_->conv_quant_arg_;
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto output_tensor = out_tensors_.at(kOutputIndex);
  size_t input_arg_num = input_tensor->GetQuantParams().size();
  size_t filter_arg_num = weight_tensor->GetQuantParams().size();
  size_t output_arg_num = output_tensor->GetQuantParams().size();
  conv_quant_arg_->input_arg_num_ = input_arg_num;
  conv_quant_arg_->filter_arg_num_ = filter_arg_num;
  conv_quant_arg_->output_arg_num_ = output_arg_num;

  conv_quant_arg_->input_quant_args_ = reinterpret_cast<QuantArg *>(malloc(input_arg_num * sizeof(QuantArg)));
  if (conv_quant_arg_->input_quant_args_ == nullptr) {
    MS_LOG(ERROR) << "malloc input_quant_args_ failed.";
    return RET_ERROR;
  }
  conv_quant_arg_->filter_quant_args_ = reinterpret_cast<QuantArg *>(malloc(filter_arg_num * sizeof(QuantArg)));
  if (conv_quant_arg_->filter_quant_args_ == nullptr) {
    MS_LOG(ERROR) << "malloc filter_quant_args_ failed.";
    return RET_ERROR;
  }
  conv_quant_arg_->output_quant_args_ = reinterpret_cast<QuantArg *>(malloc(output_arg_num * sizeof(QuantArg)));
  if (conv_quant_arg_->output_quant_args_ == nullptr) {
    MS_LOG(ERROR) << "malloc output_quant_args_ failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionBaseCPUKernel::SetInputTensorQuantParam() {
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto in_arg_num = conv_quant_arg_->input_arg_num_;
  if (in_arg_num == kPerTensor) {
    auto input_quant_arg = input_tensor->GetQuantParams().front();
    conv_quant_arg_->input_quant_args_[0].zp_ = input_quant_arg.zeroPoint;
    conv_quant_arg_->input_quant_args_[0].scale_ = input_quant_arg.scale;
  } else {
    // per channel
    MS_LOG(ERROR) << "Not Support Per Channel for input now.";
    return RET_ERROR;
    //    auto input_quant_arg = input_tensor->GetQuantParams();
    //    for (int i = 0; i < in_arg_num; ++i) {
    //      conv_quant_arg_->input_quant_args_[i].zp_ = input_quant_arg[i].zeroPoint;
    //      conv_quant_arg_->input_quant_args_[i].scale_ = input_quant_arg[i].scale;
    //    }
  }
  return RET_OK;
}

int ConvolutionBaseCPUKernel::SetFilterTensorQuantParam() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto weight_arg_num = conv_quant_arg_->filter_arg_num_;
  if (weight_arg_num == kPerTensor) {
    auto weight_quant_arg = weight_tensor->GetQuantParams().front();
    conv_quant_arg_->filter_quant_args_[0].zp_ = weight_quant_arg.zeroPoint;
    conv_quant_arg_->filter_quant_args_[0].scale_ = weight_quant_arg.scale;
  } else {
    auto weight_quant_arg = weight_tensor->GetQuantParams();
    for (int i = 0; i < weight_arg_num; ++i) {
      conv_quant_arg_->filter_quant_args_[i].zp_ = weight_quant_arg[i].zeroPoint;
      conv_quant_arg_->filter_quant_args_[i].scale_ = weight_quant_arg[i].scale;
    }
  }
  return RET_OK;
}

int ConvolutionBaseCPUKernel::SetOutputTensorQuantParam() {
  auto output_tensor = out_tensors_.at(kOutputIndex);
  auto out_arg_num = conv_quant_arg_->output_arg_num_;
  if (out_arg_num == kPerTensor) {
    auto output_quant_arg = output_tensor->GetQuantParams().front();
    conv_quant_arg_->output_quant_args_[0].zp_ = output_quant_arg.zeroPoint;
    conv_quant_arg_->output_quant_args_[0].scale_ = output_quant_arg.scale;
  } else {
    MS_LOG(ERROR) << "Not Support Per Channel for input now.";
    return RET_ERROR;
    //    auto output_quant_arg = output_tensor->GetQuantParams();
    //    for (int i = 0; i < out_arg_num; ++i) {
    //      conv_quant_arg_->output_quant_args_[i].zp_ = output_quant_arg[i].zeroPoint;
    //      conv_quant_arg_->output_quant_args_[i].scale_ = output_quant_arg[i].scale;
    //    }
  }
  return RET_OK;
}

int ConvolutionBaseCPUKernel::SetQuantMultiplier() {
  // now only support weight tensor is per channel, others are per tensor.
  int weight_arg_num = kPerTensor;
  if (conv_quant_arg_->per_channel_ & FILTER_PER_CHANNEL) {
    weight_arg_num = conv_quant_arg_->filter_arg_num_;
  }
  conv_quant_arg_->real_multiplier_ = reinterpret_cast<double *>(malloc(weight_arg_num * sizeof(double)));
  conv_quant_arg_->left_shift_ = reinterpret_cast<int32_t *>(malloc(weight_arg_num * sizeof(int32_t)));
  conv_quant_arg_->right_shift_ = reinterpret_cast<int32_t *>(malloc(weight_arg_num * sizeof(int32_t)));
  conv_quant_arg_->quant_multiplier_ = reinterpret_cast<int32_t *>(malloc(weight_arg_num * sizeof(int32_t)));
  conv_quant_arg_->out_act_min_ = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t)));
  conv_quant_arg_->out_act_max_ = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t)));

  for (int i = 0; i < weight_arg_num; ++i) {
    double real_multiplier = conv_quant_arg_->filter_quant_args_[i].scale_ *
                             conv_quant_arg_->input_quant_args_[0].scale_ /
                             conv_quant_arg_->output_quant_args_[0].scale_;
    conv_quant_arg_->real_multiplier_[i] = real_multiplier;
    QuantizeRoundParameter(real_multiplier, &conv_quant_arg_->quant_multiplier_[i], &conv_quant_arg_->left_shift_[i],
                           &conv_quant_arg_->right_shift_[i]);
  }
  return RET_OK;
}

int ConvolutionBaseCPUKernel::SetQuantParam() {
  auto ret = MallocQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc quant param failed.";
    return ret;
  }
  ret = SetInputTensorQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set Input Tensor Quant Param Failed.";
    return ret;
  }
  ret = SetFilterTensorQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set Filter Tensor Quant Param Failed.";
    return ret;
  }
  ret = SetOutputTensorQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set Output Tensor Quant Param Failed.";
    return ret;
  }
  ret = SetQuantMultiplier();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set Quant Multiplier Failed.";
    return ret;
  }
  // now only consider per tensor for output
  CalculateActivationRangeQuantized(
    conv_param_->is_relu_, conv_param_->is_relu6_, conv_param_->conv_quant_arg_.output_quant_args_[0].zp_,
    conv_param_->conv_quant_arg_.output_quant_args_[0].scale_, &conv_param_->conv_quant_arg_.out_act_min_[0],
    &conv_param_->conv_quant_arg_.out_act_max_[0]);

  ret = SetIfPerChannel();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set if per tensor channel failed.";
    return ret;
  }
  ret = SetIfAsymmetric();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set if per asymmetric failed.";
    return ret;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
