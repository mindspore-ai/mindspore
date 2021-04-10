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

#include "nnacl/infer/deconv2d_infer.h"
#include "nnacl/infer/infer_register.h"

int Deconv2dInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                       OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullInputSize(inputs, inputs_size, outputs, outputs_size, parameter, 2);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  const TensorC *weight = inputs[1];
  TensorC *output = outputs[0];
  output->format_ = input->format_;
  output->data_type_ = input->data_type_;

  ConvParameter *param = (ConvParameter *)parameter;
  if (param->group_ == 0) {
    param->group_ = weight->shape_[0];
  }
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  int32_t input_h = GetHeight(input);
  int32_t input_w = GetWidth(input);

  int32_t output_n = GetBatch(input);
  int32_t output_h = 0;
  int32_t output_w = 0;
  int32_t output_c = GetChannel(weight);
  if (param->group_ == GetChannel(input) && param->group_ == GetBatch(weight) && 1 == GetChannel(weight)) {
    output_c = GetBatch(weight); /* depthwise */
  }

  int kernel_w = param->kernel_w_;
  int kernel_h = param->kernel_h_;
  int stride_w = param->stride_w_;
  int stride_h = param->stride_h_;
  int dilate_w = param->dilation_w_;
  int dilate_h = param->dilation_h_;
  int pad_mode = param->pad_mode_;
  if (pad_mode == Pad_pad) {
    output_h = (input_h - 1) * stride_h + ((kernel_h - 1) * dilate_h + 1) - param->pad_u_ - param->pad_d_;
    output_w = (input_w - 1) * stride_w + ((kernel_w - 1) * dilate_w + 1) - param->pad_l_ - param->pad_r_;
  } else if (pad_mode == Pad_same) {
    output_h = input_h * stride_h;
    output_w = input_w * stride_w;
  } else if (pad_mode == Pad_valid) {
    output_h = (input_h - 1) * stride_h + kernel_h;
    output_w = (input_w - 1) * stride_w + kernel_w;
  } else {
    return NNACL_ERR;
  }

  output_h += param->output_padding_h_;
  output_w += param->output_padding_w_;

  output->shape_size_ = 4;
  output->shape_[0] = output_n;
  output->shape_[1] = output_h;
  output->shape_[2] = output_w;
  output->shape_[3] = output_c;

  if (pad_mode == Pad_same) {
    param->pad_u_ = ((input_h - 1) * stride_h + (kernel_h - 1) * dilate_h + 1 - output_h) / 2;
    param->pad_l_ = ((input_w - 1) * stride_w + (kernel_w - 1) * dilate_w + 1 - output_w) / 2;
  } else if (pad_mode == Pad_valid) {
    param->pad_u_ = 0;
    param->pad_l_ = 0;
  }

  const int *in_shape = input->shape_;
  param->input_batch_ = in_shape[0];
  param->input_h_ = in_shape[1];
  param->input_w_ = in_shape[2];
  param->input_channel_ = in_shape[3];
  param->output_batch_ = output_n;
  param->output_h_ = output_h;
  param->output_w_ = output_w;
  param->output_channel_ = output_c;
  return NNACL_OK;
}

REG_INFER(Conv2dTranspose, PrimType_Conv2dTransposeFusion, Deconv2dInferShape)
