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

  if (conv_quant_arg_->quant_args_ != nullptr) {
    for (int i = 0; i < 3; ++i) {
      if (*(conv_quant_arg_->quant_args_ + i) != nullptr) {
        free(*(conv_quant_arg_->quant_args_ + i));
      }
    }
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

int ConvolutionBaseCPUKernel::SetQuantParam() {
  ConvQuantArg *conv_quant_arg_ = &conv_param_->conv_quant_arg_;
  conv_quant_arg_->quant_args_ = reinterpret_cast<QuantArg **>(malloc(3 * sizeof(QuantArg *)));
  if (conv_quant_arg_->quant_args_ == nullptr) {
    MS_LOG(ERROR) << "malloc quant_args_ failed.";
    return RET_ERROR;
  }
  // per-tensor init
  for (int j = 0; j < 3; ++j) {
    conv_quant_arg_->quant_args_[j] = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg)));
    if (conv_quant_arg_->quant_args_[j] == nullptr) {
      MS_LOG(ERROR) << "malloc quant_args_ failed.";
      return RET_ERROR;
    }
  }
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto output_tensor = out_tensors_.at(kOutputIndex);
  auto input_quant_arg = input_tensor->GetQuantParams().front();
  auto weight_quant_arg = weight_tensor->GetQuantParams().front();
  auto output_quant_arg = output_tensor->GetQuantParams().front();
  // input
  conv_quant_arg_->quant_args_[0][0].zp_ = input_quant_arg.zeroPoint;
  conv_quant_arg_->quant_args_[0][0].scale_ = input_quant_arg.scale;
  // weight
  conv_quant_arg_->quant_args_[1][0].zp_ = weight_quant_arg.zeroPoint;
  conv_quant_arg_->quant_args_[1][0].scale_ = weight_quant_arg.scale;
  // output
  conv_quant_arg_->quant_args_[2][0].zp_ = output_quant_arg.zeroPoint;
  conv_quant_arg_->quant_args_[2][0].scale_ = output_quant_arg.scale;

  conv_quant_arg_->real_multiplier_ = reinterpret_cast<double *>(malloc(sizeof(double)));
  conv_quant_arg_->left_shift_ = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t)));
  conv_quant_arg_->right_shift_ = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t)));
  conv_quant_arg_->quant_multiplier_ = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t)));
  conv_quant_arg_->out_act_min_ = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t)));
  conv_quant_arg_->out_act_max_ = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t)));

  double real_multiplier = weight_quant_arg.scale * input_quant_arg.scale / output_quant_arg.scale;
  conv_quant_arg_->real_multiplier_[0] = real_multiplier;
  QuantizeRoundParameter(real_multiplier, &conv_quant_arg_->quant_multiplier_[0], &conv_quant_arg_->left_shift_[0],
                         &conv_quant_arg_->right_shift_[0]);

  CalculateActivationRangeQuantized(
    conv_param_->is_relu_, conv_param_->is_relu6_, conv_param_->conv_quant_arg_.quant_args_[2][0].zp_,
    conv_param_->conv_quant_arg_.quant_args_[2][0].scale_, &conv_param_->conv_quant_arg_.out_act_min_[0],
    &conv_param_->conv_quant_arg_.out_act_max_[0]);
  return RET_OK;
}
}  // namespace mindspore::kernel
