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

#include "nnacl/infer/argmin_max_infer.h"
#include "nnacl/infer/infer_register.h"

int ArgMinMaxInferShape(const TensorC *const *inputs, const size_t inputs_size, TensorC **outputs,
                        const size_t outputs_size, OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  if (inputs_size != 1 || outputs_size > 2) {
    return NNACL_ERR;
  }
#endif

  ArgMinMaxParameter *param = (ArgMinMaxParameter *)parameter;
  const TensorC *input = inputs[0];
  TensorC *output_1 = NULL;
  TensorC *output_2 = NULL;
  if (outputs_size == 2) {
    output_1 = outputs[0];
    output_2 = outputs[1];
  } else if (param->out_value_) {
    output_2 = outputs[0];
  } else {
    output_1 = outputs[0];
  }

  if (output_1 != NULL) {
    output_1->format_ = input->format_;
    output_1->data_type_ = kNumberTypeInt32;
  }
  if (output_2 != NULL) {
    SetDataTypeFormat(output_2, input);
  }
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  int output_shape[MAX_SHAPE_SIZE];
  size_t output_shape_size = 0;
  ShapeSet(output_shape, &output_shape_size, input->shape_, input->shape_size_);
  size_t input_shape_size = input->shape_size_;
  int axis = param->axis_ < 0 ? param->axis_ + (int)input_shape_size : param->axis_;
  if (axis >= input_shape_size || axis < 0) {
    return NNACL_PARAM_INVALID;
  }
  if (param->topk_ == 1 && !param->keep_dims_) {
    ShapeErase(output_shape, &output_shape_size, axis);
  } else {
    output_shape[axis] = param->topk_;
  }

  if (output_1 != NULL) {
    SetShapeArray(output_1, output_shape, output_shape_size);
  }
  if (output_2 != NULL) {
    SetShapeArray(output_2, output_shape, output_shape_size);
  }
  return NNACL_OK;
}

REG_INFER(ArgMin, PrimType_ArgMinFusion, ArgMinMaxInferShape)
REG_INFER(ArgMax, PrimType_ArgMaxFusion, ArgMinMaxInferShape)
