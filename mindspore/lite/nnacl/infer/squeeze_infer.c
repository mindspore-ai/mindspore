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

#include "nnacl/infer/squeeze_infer.h"
#include "nnacl/infer/infer_register.h"

int SqueezeInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                      OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  SqueezeParameter *param = (SqueezeParameter *)parameter;
  SetDataTypeFormat(outputs[0], input);
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  int out_shape[MAX_SHAPE_SIZE];
  size_t out_shape_size = 0;

  for (size_t i = 0; i < param->axis_size_; i++) {
    param->axis_[i] = param->axis_[i] >= 0 ? param->axis_[i] : param->axis_[i] + input->shape_size_;
  }

  if (param->axis_size_ == 0) {
    for (size_t i = 0; i < input->shape_size_; i++) {
      if (input->shape_[i] != 1) {
        ShapePush(out_shape, &out_shape_size, input->shape_[i]);
      }
    }
  } else {
    size_t axisIdx = 0;
    for (size_t i = 0; i < input->shape_size_; i++) {
      if (axisIdx < param->axis_size_ && param->axis_[axisIdx] == (int)(i)) {
        if (input->shape_[i] != 1) return NNACL_PARAM_INVALID;
        axisIdx++;
        continue;
      } else {
        ShapePush(out_shape, &out_shape_size, input->shape_[i]);
      }
    }
  }
  SetShapeArray(outputs[0], out_shape, out_shape_size);
  return NNACL_OK;
}

REG_INFER(Squeeze, PrimType_Squeeze, SqueezeInferShape)
