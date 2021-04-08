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

#include "nnacl/infer/depthwise_conv2d_infer.h"
#include "nnacl/infer/infer_register.h"

int DepthwiseConv2dInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                              OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSizeInputTwo(inputs, inputs_size, outputs, outputs_size, parameter, 2, 3, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, input);
  ConvParameter *param = (ConvParameter *)parameter;

  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  int input_h = input->shape_[1];
  int input_w = input->shape_[2];
  int input_channel = input->shape_[3];
  int output_w = 0, output_h = 0;
  param->input_channel_ = input_channel;

  if (param->pad_mode_ == Pad_same) {
    output_h = ceil((float)(input_h) / (float)(param->stride_h_));
    output_w = ceil((float)(input_w) / (float)(param->stride_w_));
    int pad_h_all = ((output_h - 1) * param->stride_h_ + (param->kernel_h_ - 1) * param->dilation_h_ + 1 - input_h);
    int pad_w_all = ((output_w - 1) * param->stride_w_ + (param->kernel_w_ - 1) * param->dilation_w_ + 1 - input_w);
    if (pad_h_all > 0) {
      param->pad_u_ = pad_h_all / 2;
      param->pad_d_ = pad_h_all - param->pad_u_;
    }
    if (pad_w_all > 0) {
      param->pad_l_ = pad_w_all / 2;
      param->pad_r_ = pad_w_all - param->pad_l_;
    }
  } else {
    output_h = ceil(((float)(input_h) + param->pad_u_ + param->pad_d_ -
                     ((float)(param->kernel_h_) - 1) * (float)(param->dilation_h_)) /
                    (float)(param->stride_h_));
    output_w = ceil(((float)(input_w) + param->pad_l_ + param->pad_r_ -
                     ((float)(param->kernel_w_) - 1) * (float)(param->dilation_w_)) /
                    (float)(param->stride_w_));
  }
  int out_shape[MAX_SHAPE_SIZE];
  size_t out_shape_size = 0;
  ShapeSet(out_shape, &out_shape_size, input->shape_, input->shape_size_);
  out_shape[1] = output_h;
  out_shape[2] = output_w;
  if (param->channel_multiplie_ != 1) {
    return NNACL_ERR;
  }
  out_shape[3] = input_channel;  // in_channel * out_channel
  SetShapeArray(output, out_shape, out_shape_size);
  return 0;
}
