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

#include "nnacl/infer/unstack_infer.h"
#include "nnacl/infer/infer_register.h"

int UnstackInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                      OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  UnstackParameter *param = (UnstackParameter *)parameter;
  int axis = param->axis_ < 0 ? param->axis_ + input->shape_size_ : param->axis_;
  if (axis < 0 || axis >= input->shape_size_) {
    return NNACL_PARAM_INVALID;
  }
  for (size_t i = 0; i < outputs_size; i++) {
    SetDataTypeFormat(outputs[i], input);
  }

  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  int output_shape[MAX_SHAPE_SIZE];
  size_t output_shape_size = 0;
  for (size_t i = 0; i < input->shape_size_; ++i) {
    if (i != axis) {
      ShapePush(output_shape, &output_shape_size, input->shape_[i]);
    }
  }
  for (size_t i = 0; i < outputs_size; i++) {
    if (outputs[i] == NULL) {
      return NNACL_NULL_PTR;
    }
    SetShapeArray(outputs[i], output_shape, output_shape_size);
  }
  return NNACL_OK;
}

REG_INFER(Unstack, PrimType_Unstack, UnstackInferShape)
