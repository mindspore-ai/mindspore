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

#include "nnacl/infer/power_infer.h"
#include "nnacl/infer/infer_register.h"

int PowerInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                    OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *x_tensor = inputs[0];
  TensorC *exp_tensor = NULL;
  if (inputs_size == 2) {
    exp_tensor = (TensorC *)inputs[1];
    PowerParameter *param = (PowerParameter *)parameter;
    float *exp_data = (float *)(exp_tensor->data_);
    if (exp_data == NULL) {
      return NNACL_INFER_INVALID;
    }
    param->power_ = *exp_data;
  }
  TensorC *output_tensor = outputs[0];

  SetDataTypeFormat(output_tensor, x_tensor);
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  if (exp_tensor != NULL) {
    bool exp_x_equal = ShapeEqual(exp_tensor->shape_, exp_tensor->shape_size_, x_tensor->shape_, x_tensor->shape_size_);
    if (!exp_x_equal && GetElementNum(exp_tensor) != 1) {
      return NNACL_INPUT_TENSOR_ERROR;
    }
  }

  SetShapeTensor(output_tensor, x_tensor);
  return NNACL_OK;
}

REG_INFER(Pow, PrimType_PowFusion, PowerInferShape)
