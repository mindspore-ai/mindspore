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

#include "nnacl/infer/control/tensor_array_write_infer.h"
#include "nnacl/infer/infer_register.h"
#include "nnacl/tensor_array_parameter.h"

int TensorArrayWriteInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                               OpParameter *parameter) {
  // { handle, index, value, flow_in } -> empty
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 4, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  MS_CHECK_TRUE_RET(inputs_size >= 3, NNACL_ERR);
  TensorC *handle = (TensorC *)inputs[0];
  TensorC *value = (TensorC *)inputs[2];

  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  TensorArrayParameter *param = (TensorArrayParameter *)parameter;
  if (param == NULL) {
    return NNACL_NULL_PTR;
  }

  if (handle->shape_size_ != value->shape_size_) {
    return NNACL_INFER_INVALID;
  }

  for (int i = 0; i < handle->shape_size_; ++i) {
    if (handle->shape_[i] != value->shape_[i]) {
      return NNACL_INFER_INVALID;
    }
  }

  return NNACL_OK;
}

REG_INFER(TensorArrayWrite, PrimType_TensorArrayWrite, TensorArrayWriteInferShape)
