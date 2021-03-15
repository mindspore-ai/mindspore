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

#include "nnacl/infer/pooling_infer.h"
#include <math.h>
#include "nnacl/infer/infer_register.h"

int PoolingInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                      OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, input);
  PoolingParameter *param = (PoolingParameter *)parameter;
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  int input_h = input->shape_[1];
  int input_w = input->shape_[2];

  int window_h = param->window_h_;
  int window_w = param->window_w_;
  if (param->global_) {
    window_h = input_h;
    window_w = input_w;
  }
  int output_h = 0;
  int output_w = 0;
  if (param->pad_mode_ == Pad_same) {  // maybe error
    output_w = ceil((float)(input_w) / (float)(param->stride_w_));
    output_h = ceil((float)(input_h) / (float)(param->stride_h_));
    int pad_h_all = ((output_h - 1) * param->stride_h_ + (window_h - 1) + 1 - input_h);
    int pad_w_all = ((output_w - 1) * param->stride_w_ + (window_w - 1) + 1 - input_w);
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
  } else {
    int round_mode = (RoundMode)param->round_mode_;
    if (round_mode == RoundMode_Floor) {
      output_h = floor((float)(input_h + param->pad_u_ + param->pad_d_ - window_h) / param->stride_h_) + 1;
      output_w = floor((float)(input_w + param->pad_l_ + param->pad_r_ - window_w) / param->stride_w_) + 1;
    } else if (round_mode == RoundMode_Ceil) {
      output_h = ceil((float)(input_h + param->pad_u_ + param->pad_d_ - window_h) / param->stride_h_) + 1;
      output_w = ceil((float)(input_w + param->pad_l_ + param->pad_r_ - window_w) / param->stride_w_) + 1;
    } else {
      return NNACL_ERR;
    }
  }
  int input_shape[MAX_SHAPE_SIZE];
  size_t input_shape_size = 0;
  ShapeSet(input_shape, &input_shape_size, input->shape_, input->shape_size_);
  input_shape[1] = output_h > 0 ? output_h : 1;
  input_shape[2] = output_w > 0 ? output_w : 1;
  SetShapeArray(output, input_shape, input_shape_size);
  return NNACL_OK;
}

REG_INFER(MaxPool, PrimType_MaxPoolFusion, PoolingInferShape)
REG_INFER(AvgPool, PrimType_AvgPoolFusion, PoolingInferShape)
