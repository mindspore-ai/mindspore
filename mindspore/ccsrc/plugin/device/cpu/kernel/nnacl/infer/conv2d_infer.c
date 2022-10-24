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
#include "nnacl/infer/conv2d_infer.h"
#include "nnacl/infer/infer_register.h"

int ConvInferShape(int input_h, int input_w, int *output_h, int *output_w, ConvParameter *param) {
  int kernel_w = param->kernel_w_;
  int kernel_h = param->kernel_h_;
  int stride_w = param->stride_w_;
  int stride_h = param->stride_h_;
  int dilate_w = param->dilation_w_;
  int dilate_h = param->dilation_h_;

  if (stride_w == 0 || stride_h == 0) {
    return NNACL_PARAM_INVALID;
  }
  if (INT_MUL_OVERFLOW(kernel_h, dilate_h) || INT_MUL_OVERFLOW(kernel_w, dilate_w)) {
    return NNACL_ERRCODE_MUL_OVERFLOW;
  }
  if (param->pad_mode_ == Pad_same) {  // maybe error
    *output_w = ceil((float)(input_w) / (float)(stride_w));
    *output_h = ceil((float)(input_h) / (float)(stride_h));
    int pad_h_all = ((*output_h - 1) * stride_h + (kernel_h - 1) * dilate_h + 1 - input_h);
    int pad_w_all = ((*output_w - 1) * stride_w + (kernel_w - 1) * dilate_w + 1 - input_w);
    if (pad_h_all < 0) {
      param->pad_u_ = param->pad_d_ = 0;
    } else {
      param->pad_u_ = pad_h_all / 2;
      param->pad_d_ = pad_h_all - param->pad_u_;
    }
    if (pad_w_all < 0) {
      param->pad_l_ = param->pad_r_ = 0;
    } else {
      param->pad_l_ = pad_w_all / 2;
      param->pad_r_ = pad_w_all - param->pad_l_;
    }
  } else if (param->pad_mode_ == Pad_valid) {
    *output_w = ceil(((float)(input_w) + param->pad_l_ + param->pad_r_ - ((float)(kernel_w)-1) * (float)(dilate_w)) /
                     (float)(stride_w));
    *output_h = ceil(((float)(input_h) + param->pad_u_ + param->pad_d_ - ((float)(kernel_h)-1) * (float)(dilate_h)) /
                     (float)(stride_h));
  } else {
    int kernel_width = (kernel_w - 1) * dilate_w + 1;
    int kernel_height = (kernel_h - 1) * dilate_h + 1;
    *output_w = ((input_w) + param->pad_l_ + param->pad_r_ - kernel_width) / stride_w + 1;
    *output_h = ((input_h) + param->pad_u_ + param->pad_d_ - kernel_height) / stride_h + 1;
  }

  if (param->kernel_h_ > input_h + param->pad_u_ + param->pad_d_ ||
      param->kernel_w_ > input_w + param->pad_l_ + param->pad_r_) {
    return NNACL_PARAM_INVALID;
  }
  return NNACL_OK;
}

static const int MAX_CONV_KERNEL_DIM = 10000;  // One big value that should not be adopted as the conv kernel dimension.

int CheckConvAttr(const int input_c, const TensorC *weight_tensor, const ConvParameter *param) {
  // common conv: input_c == weight_tensor->shape_[3]
  // conv depthwise: input_c == 1
  // group conv: input_c / group == weight_tensor->shape_[3]
  MS_CHECK_FALSE(param->group_ == 0, NNACL_PARAM_INVALID);
  if (input_c != weight_tensor->shape_[3] && input_c != 1 && (input_c / param->group_) != weight_tensor->shape_[3]) {
    return NNACL_PARAM_INVALID;
  }

  // common conv: group == 1
  // conv depthwise: group == input_c == output_c
  // group conv: group == input_c / weight_tensor->shape_[3]
  MS_CHECK_FALSE(weight_tensor->shape_[3] == 0, NNACL_PARAM_INVALID);
  if (param->group_ != 1 && param->group_ != input_c && param->group_ != (input_c / weight_tensor->shape_[3])) {
    return NNACL_PARAM_INVALID;
  }
  if (param->stride_h_ <= 0 || param->stride_w_ <= 0) {
    return NNACL_PARAM_INVALID;
  }

  if ((param->kernel_h_ >= MAX_CONV_KERNEL_DIM) || (param->kernel_w_ >= MAX_CONV_KERNEL_DIM)) {
    return NNACL_PARAM_INVALID;
  }

  MS_CHECK_TRUE_RET(param->kernel_h_ == weight_tensor->shape_[1], NNACL_PARAM_INVALID);
  MS_CHECK_TRUE_RET(param->kernel_w_ == weight_tensor->shape_[2], NNACL_PARAM_INVALID);
  return NNACL_OK;
}

int Conv2dInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter) {
  int check_ret = CheckAugmentNullSizeInputTwo(inputs, inputs_size, outputs, outputs_size, parameter, 2, 3, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input_tensor = inputs[0];
  if (input_tensor->format_ != Format_NHWC && input_tensor->format_ != Format_KHWC &&
      input_tensor->format_ != Format_NC4HW4 && input_tensor->format_ != Format_NC8HW8) {
    return NNACL_FORMAT_ERROR;
  }
  const TensorC *weight_tensor = inputs[1];
  if (weight_tensor->format_ != Format_NHWC && weight_tensor->format_ != Format_KHWC) {
    return NNACL_FORMAT_ERROR;
  }
  TensorC *out_tensor = outputs[0];
  if (out_tensor->format_ != Format_NC4HW4) {
    out_tensor->format_ = input_tensor->format_;
  }
  out_tensor->data_type_ = input_tensor->data_type_;
  ConvParameter *param = (ConvParameter *)parameter;
  if (param->group_ == 0) {
    param->group_ = weight_tensor->shape_[0];
  }
  param->output_channel_ = weight_tensor->shape_[0];
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  param->kernel_h_ = param->kernel_h_ != -1 ? param->kernel_h_ : weight_tensor->shape_[1];
  param->kernel_w_ = param->kernel_w_ != -1 ? param->kernel_w_ : weight_tensor->shape_[2];

  if (input_tensor->shape_size_ == 0) {
    return NNACL_INFER_INVALID;
  }

  int ret = CheckConvAttr(GetChannel(input_tensor), weight_tensor, param);
  if (ret != NNACL_OK) {
    return ret;
  }

  int output_w = 0, output_h = 0;
  ret = ConvInferShape(GetHeight(input_tensor), GetWidth(input_tensor), &output_h, &output_w, param);
  if (ret != NNACL_OK) {
    return ret;
  }

  out_tensor->shape_size_ = input_tensor->shape_size_;
  SetBatch(out_tensor, GetBatch(input_tensor));
  SetChannel(out_tensor, GetBatch(weight_tensor));
  output_h = output_h >= 0 ? output_h : 1;
  SetHeight(out_tensor, output_h);
  output_w = output_w >= 0 ? output_w : 1;
  SetWidth(out_tensor, output_w);

  param->input_batch_ = GetBatch(input_tensor);
  param->input_h_ = GetHeight(input_tensor);
  param->input_w_ = GetWidth(input_tensor);
  param->input_channel_ = GetChannel(input_tensor);
  param->output_batch_ = GetBatch(out_tensor);
  param->output_h_ = GetHeight(out_tensor);
  param->output_w_ = GetWidth(out_tensor);
  param->output_channel_ = GetChannel(out_tensor);
  param->out_format_ = out_tensor->format_;
  return NNACL_OK;
}

REG_INFER(Adder, PrimType_AdderFusion, Conv2dInferShape)
REG_INFER(Conv2D, PrimType_Conv2DFusion, Conv2dInferShape)
