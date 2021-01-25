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

#include "nnacl/infer/argmin_infer.h"

int ArgminInferShape(const TensorC *const *inputs, const size_t inputs_size, TensorC **outputs,
                     const size_t outputs_size, OpParameter *parameter) {
  if (inputs_size != 1 || outputs_size > 2) {
    return NNACL_ERR;
  }
  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, input);
  ArgMinMaxParameter *param = (ArgMinMaxParameter *)parameter;
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  int input_shape_size = input->shape_size_;
  int axis = param->axis_ < 0 ? param->axis_ + input_shape_size : param->axis_;
  if (axis >= input_shape_size || axis < 0) {
    return NNACL_PARAM_INVALID;
  }
  int output_shape[MAX_SHAPE_SIZE];
  size_t output_shape_size = 0;
  ShapeSet(output_shape, &output_shape_size, input->shape_, input->shape_size_);
  if (param->topk_ == 1 && !param->keep_dims_) {
    ShapeErase(output_shape, &output_shape_size, axis);
  } else {
    output_shape[axis] = param->topk_;
  }

  SetShapeArray(output, output_shape, output_shape_size);
  if (outputs_size == 2) {
    SetDataTypeFormat(outputs[1], input);
    SetShapeTensor(outputs[1], output);
  }
  return NNACL_OK;
}
