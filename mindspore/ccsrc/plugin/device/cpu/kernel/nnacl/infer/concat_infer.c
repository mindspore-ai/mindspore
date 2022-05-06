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

#include "nnacl/infer/concat_infer.h"
#include "nnacl/infer/infer_register.h"

int DataTypeJudge(const TensorC *input, const TensorC *output) {
  if ((input->data_type_ != output->data_type_) &&
      !((input->data_type_ == kNumberTypeFloat16 && output->data_type_ == kNumberTypeFloat32) ||
        (input->data_type_ == kNumberTypeFloat32 && output->data_type_ == kNumberTypeFloat16))) {
    return NNACL_PARAM_INVALID;
  }
  return NNACL_OK;
}

int ConcatInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter) {
  int check_ret = CheckAugmentNullOutputSize(inputs, inputs_size, outputs, outputs_size, parameter, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  if (inputs_size < 1) {
    return NNACL_INPUT_TENSOR_ERROR;
  }

  const TensorC *input0 = inputs[0];
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, input0);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  const int *input0_shape = inputs[0]->shape_;
  size_t input0_shape_size = inputs[0]->shape_size_;

  ConcatParameter *param = (ConcatParameter *)parameter;
  int axis = param->axis_ < 0 ? param->axis_ + (int)input0_shape_size : param->axis_;
  if (axis < 0 || axis >= (int)input0_shape_size) {
    return NNACL_ERR;
  }
  if (input0_shape_size > MAX_SHAPE_SIZE) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  int input0_shape_without_axis[MAX_SHAPE_SIZE] = {0};
  size_t input0_shape_without_axis_size = 0;
  ShapeSet(input0_shape_without_axis, &input0_shape_without_axis_size, input0_shape, input0_shape_size);
  int erase_ret = ShapeErase(input0_shape_without_axis, &input0_shape_without_axis_size, axis);
  if (erase_ret != NNACL_OK) {
    return NNACL_ERR;
  }
  int output_axis_dim = input0_shape[axis];
  for (size_t i = 1; i < inputs_size; ++i) {
    size_t input_i_shape_size = inputs[i]->shape_size_;
    if (input_i_shape_size != input0_shape_size) {
      return NNACL_PARAM_INVALID;
    }
    int shape_tmp[MAX_SHAPE_SIZE] = {0};
    size_t shape_tmp_size = 0;
    ShapeSet(shape_tmp, &shape_tmp_size, inputs[i]->shape_, inputs[i]->shape_size_);
    int data_type_judge = DataTypeJudge(inputs[i], output);
    if (data_type_judge != NNACL_OK) {
      return data_type_judge;
    }
    int axis_tmp = shape_tmp[axis];
    erase_ret = ShapeErase(shape_tmp, &shape_tmp_size, axis);
    if (erase_ret != NNACL_OK) {
      return NNACL_ERR;
    }
    if (!ShapeEqual(input0_shape_without_axis, input0_shape_without_axis_size, shape_tmp, shape_tmp_size)) {
      return NNACL_ERR;
    }
    output_axis_dim += axis_tmp;
  }
  int output_shape[MAX_SHAPE_SIZE];
  size_t output_shape_size = input0_shape_size;
  for (size_t i = 0; i < input0_shape_size; i++) {
    output_shape[i] = input0_shape[i];
  }
  output_shape[axis] = output_axis_dim;
  SetShapeArray(outputs[0], output_shape, output_shape_size);
  return NNACL_OK;
}

REG_INFER(Concat, PrimType_Concat, ConcatInferShape)
