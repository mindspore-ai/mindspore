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

#include "nnacl/infer/one_hot_infer.h"
#include "nnacl/infer/infer_register.h"

int OneHotInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  if (inputs_size != 4 && inputs_size != 3) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
#endif

  const TensorC *input = inputs[0];
  const TensorC *depth_tensor = inputs[1];
  const TensorC *on_value = inputs[2];
  TensorC *output = outputs[0];
  const int *depth = (int *)(depth_tensor->data_);
  if (depth == NULL) {
    return NNACL_NULL_PTR;
  }
  SetDataTypeFormat(output, on_value);
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  OneHotParameter *param = (OneHotParameter *)parameter;
  int axis = param->axis_;
  int input_rank = (int)(input->shape_size_);
  if (axis < 0) {
    axis += input_rank + 1;
  }
  ShapeSet(output->shape_, &(output->shape_size_), input->shape_, input->shape_size_);
  int res_insert = ShapeInsert(output->shape_, &output->shape_size_, axis, *depth);
  if (res_insert == NNACL_ERR) {
    return NNACL_ERR;
  }

  return NNACL_OK;
}

REG_INFER(OneHot, PrimType_OneHot, OneHotInferShape)
