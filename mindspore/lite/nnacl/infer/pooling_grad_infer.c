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

#include "nnacl/infer/pooling_grad_infer.h"
#include <math.h>
#include "nnacl/infer/infer_register.h"

int PoolingGradInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                          OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 3, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  int input_h = input->shape_[1];
  int input_w = input->shape_[2];

  PoolingParameter *param = (PoolingParameter *)parameter;
  int window_h = param->window_h_;
  int window_w = param->window_w_;
  if (param->global_) {
    window_h = input_h;
    window_w = input_w;
  }

  if (param->pad_mode_ == Pad_same) {
    int output_w = ceil((float)(input_w) / (float)(param->stride_w_));
    int output_h = ceil((float)(input_h) / (float)(param->stride_h_));
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
  }
  SetDataTypeFormat(outputs[0], input);
  SetShapeTensor(outputs[0], input);
  return NNACL_OK;
}

REG_INFER(AvgPoolGrad, PrimType_AvgPoolGrad, PoolingGradInferShape)
REG_INFER(MaxPoolGrad, PrimType_MaxPoolGrad, PoolingGradInferShape)
