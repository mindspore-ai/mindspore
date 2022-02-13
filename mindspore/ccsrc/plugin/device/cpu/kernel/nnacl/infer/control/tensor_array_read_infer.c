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

#include "nnacl/infer/control/tensor_array_read_infer.h"
#include "nnacl/infer/infer_register.h"
#include "nnacl/tensor_array_parameter.h"

int TensorArrayReadInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                              OpParameter *parameter) {
  // { prim, handle, index } -> node
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 3, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  MS_CHECK_TRUE_RET(inputs_size >= 1, NNACL_ERR);
  MS_CHECK_TRUE_RET(outputs_size >= 1, NNACL_ERR);
  TensorC *handle = (TensorC *)inputs[0];
  TensorC *output = outputs[0];

  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  output->data_type_ = handle->data_type_;
  SetShapeArray(output, handle->shape_, handle->shape_size_);

  return NNACL_OK;
}

REG_INFER(TensorArrayRead, PrimType_TensorArrayRead, TensorArrayReadInferShape)
