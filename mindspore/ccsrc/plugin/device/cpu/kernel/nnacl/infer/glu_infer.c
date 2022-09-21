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
#include "nnacl/infer/glu_infer.h"
#include "nnacl/infer/infer_register.h"
#include "nnacl/glu_parameter.h"

int GluInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
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
  SetShapeTensor(output, input);
  GluParameter *param = (GluParameter *)parameter;
  NNACL_CHECK_NULL_RETURN_ERR(param);
  if (param->axis_ >= (int)input->shape_size_ || (param->axis_ < 0 && ((int)input->shape_size_ + param->axis_) < 0)) {
    return NNACL_ERR;
  }
  int axis = param->axis_ > 0 ? param->axis_ : (int)input->shape_size_ + param->axis_;
  if (axis < 0 || axis >= MAX_SHAPE_SIZE) {
    return NNACL_BUFFER_OVERFLOW;
  }
  output->shape_[axis] /= 2;
  return NNACL_OK;
}

REG_INFER(GLU, PrimType_GLU, GluInferShape)
