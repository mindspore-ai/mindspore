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

#include "nnacl/infer/flatten_infer.h"
#include "nnacl/infer/infer_register.h"
#include "nnacl/flatten_parameter.h"

int FlattenInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                      OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];

  SetDataTypeFormat(output, input);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  if (input->shape_size_ <= 0 || input->shape_size_ > MAX_SHAPE_SIZE) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  int input_shape[MAX_SHAPE_SIZE] = {0};
  size_t input_shape_size = 0;
  ShapeSet(input_shape, &input_shape_size, input->shape_, input->shape_size_);
  FlattenParameter *param = (FlattenParameter *)parameter;
  int axis = param->axis_;
  // The value for axis must be in the range[-r, r], where r is
  // the rank of the input tensor.Negative value means counting
  // dimensions from the back.
  axis = axis < 0 ? (int)input_shape_size - axis : axis;
  if (axis >= (int)input_shape_size) {
    return NNACL_ERR;
  }
  int output_shape[2];
  output_shape[0] = axis == 0 ? 1 : input_shape[0];
  for (size_t i = 1; i < (size_t)axis; i++) {
    output_shape[0] *= input_shape[i];
  }
  output_shape[1] = input_shape[axis];
  for (size_t i = (size_t)axis + 1; i < input_shape_size; i++) {
    output_shape[1] *= input_shape[i];
  }
  SetShapeArray(output, output_shape, 2);
  return NNACL_OK;
}

REG_INFER(Flatten, PrimType_Flatten, FlattenInferShape)
