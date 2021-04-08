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

#include "nnacl/infer/full_connection_infer.h"
#include "nnacl/infer/infer_register.h"

int FullConnectionInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                             OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input0 = inputs[0];
  const TensorC *input1 = inputs[1];
  TensorC *output = outputs[0];
  MatMulParameter *param = (MatMulParameter *)parameter;
  SetDataTypeFormat(output, input0);
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  if ((param->has_bias_ && inputs_size != 3) || (!param->has_bias_ && inputs_size != 2)) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  if (param->use_axis_ && (param->axis_ < 1 || param->axis_ > (int)(input0->shape_size_))) {
    return NNACL_ERR;
  }
  int new_k = 1;
  if (param->use_axis_) {
    for (size_t i = param->axis_; i < input0->shape_size_; ++i) {
      new_k *= input0->shape_[i];
    }
    if (new_k != input1->shape_[1]) {
      return NNACL_INPUT_TENSOR_ERROR;
    }
  } else {
    new_k = input1->shape_[1];
  }
  if (param->has_bias_) {
    if (inputs[2]->shape_[0] != input1->shape_[0]) {
      return NNACL_INPUT_TENSOR_ERROR;
    }
  }
  int out_shape[MAX_SHAPE_SIZE];
  size_t out_shape_size = 0;
  ShapeSet(out_shape, &out_shape_size, inputs[0]->shape_, inputs[0]->shape_size_);
  if (param->use_axis_) {
    out_shape_size = param->axis_ + 1;
    out_shape[param->axis_] = input1->shape_[0];
  } else {
    int total = 1;
    for (size_t i = 0; i < input0->shape_size_; ++i) {
      total *= input0->shape_[i];
    }
    out_shape_size = 2;
    int batch_size = total / new_k;
    out_shape[0] = batch_size;
    out_shape[1] = input1->shape_[0];
  }
  SetShapeArray(output, out_shape, out_shape_size);

  return NNACL_OK;
}

REG_INFER(FullConnection, PrimType_FullConnection, FullConnectionInferShape)
