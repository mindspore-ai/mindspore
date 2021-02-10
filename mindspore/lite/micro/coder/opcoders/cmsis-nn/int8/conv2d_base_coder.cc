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

#include "coder/opcoders/cmsis-nn/int8/conv2d_base_coder.h"
#include "nnacl/int8/quantize.h"

namespace mindspore::lite::micro::cmsis {

int Conv2DBaseCoder::SetQuantArgs() {
  int channel = output_tensor_->Channel();
  size_t channel_data_size = static_cast<size_t>(channel) * sizeof(int32_t);
  output_mult_ = reinterpret_cast<int32_t *>(malloc(channel_data_size));
  MS_CHECK_PTR(output_mult_);
  output_shift_ = reinterpret_cast<int32_t *>(malloc(channel_data_size));
  MS_CHECK_PTR(output_shift_);

  const ::QuantArg *filter_quant_args = conv_quant_arg_->filter_quant_args_;
  auto input_scale = static_cast<double>(conv_quant_arg_->input_quant_args_[0].scale_);
  auto output_scale = static_cast<double>(conv_quant_arg_->output_quant_args_[0].scale_);
  int32_t significand;
  int channel_shift;
  if (conv_quant_arg_->filter_arg_num_ > 1) {
    for (int i = 0; i < channel; ++i) {
      // If per-tensor quantization parameter is specified, broadcast it along the
      // quantization dimension (channels_out).
      MS_CHECK_TRUE(conv_quant_arg_->filter_arg_num_ == static_cast<size_t>(channel), "quant num not match");
      const auto filter_scale = static_cast<double>(filter_quant_args[i].scale_);
      const double effective_output_scale = input_scale * filter_scale / output_scale;
      QuantizeMultiplier(effective_output_scale, &significand, &channel_shift);
      output_mult_[i] = significand;
      output_shift_[i] = channel_shift;
    }
  } else {
    // broadcast multiplier and shift to all array if per-tensor
    const auto filter_scale = static_cast<double>(filter_quant_args[0].scale_);
    const double effective_output_scale = input_scale * filter_scale / output_scale;
    QuantizeMultiplier(effective_output_scale, &significand, &channel_shift);
    for (int i = 0; i < channel; ++i) {
      output_mult_[i] = significand;
      output_shift_[i] = channel_shift;
    }
  }

  return RET_OK;
}

}  // namespace mindspore::lite::micro::cmsis
