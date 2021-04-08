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

int FlattenInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                      OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];

  SetDataTypeFormat(output, input);
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  int input_shape[MAX_SHAPE_SIZE];
  size_t input_shape_size = 0;
  ShapeSet(input_shape, &input_shape_size, input->shape_, input->shape_size_);
  int output_shape[2];
  output_shape[0] = input_shape[0];
  output_shape[1] = 1;
  for (size_t i = 1; i < input_shape_size; i++) {
    output_shape[1] *= input_shape[i];
  }
  SetShapeArray(output, output_shape, 2);
  return NNACL_OK;
}

REG_INFER(Flatten, PrimType_Flatten, FlattenInferShape)
