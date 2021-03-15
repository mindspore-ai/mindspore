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

#include "nnacl/infer/sparse_to_dense_infer.h"
#include "nnacl/infer/infer_register.h"

int SparseToDenseInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                            OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  TensorC *output = outputs[0];
  const TensorC *input1 = inputs[1];
  const TensorC *input2 = inputs[2];
  SetDataTypeFormat(output, input2);
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  int *input1_data = (int *)(input1->data_);
  int output_shape[MAX_SHAPE_SIZE];
  size_t output_shape_size = 0;
  for (int i = 0; i < GetElementNum(input1); i++) {
    ShapePush(output_shape, &output_shape_size, input1_data[i]);
  }
  SetShapeArray(output, output_shape, output_shape_size);
  return NNACL_OK;
}

REG_INFER(SparseToDense, PrimType_SparseToDense, SparseToDenseInferShape)
