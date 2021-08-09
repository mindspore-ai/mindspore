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

#include "nnacl/infer/control/tensor_array_infer.h"
#include "nnacl/infer/infer_register.h"
#include "nnacl/tensor_array_parameter.h"

int TensorArrayInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                          OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  TensorC *output = outputs[0];

  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  TensorArrayParameter *param = (TensorArrayParameter *)parameter;
  if (param == NULL) {
    return NNACL_NULL_PTR;
  }

  output->data_type_ = param->data_type_;
  SetShapeArray(output, param->element_shape_, param->element_shape_size_);

  return NNACL_OK;
}

REG_INFER(TensorArray, PrimType_TensorArray, TensorArrayInferShape)
