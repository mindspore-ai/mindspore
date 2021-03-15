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

#include "nnacl/infer/batch_to_space_infer.h"
#include "nnacl/infer/infer_register.h"

int SetOutputShapeFromParam(const TensorC *const *inputs, TensorC **outputs, OpParameter *parameter) {
  int input_shape[MAX_SHAPE_SIZE];
  size_t input_shape_size = 0;
  ShapeSet(input_shape, &input_shape_size, inputs[0]->shape_, inputs[0]->shape_size_);

  if (input_shape_size != 4) {
    return NNACL_PARAM_INVALID;
  }

  BatchToSpaceParameter *param = (BatchToSpaceParameter *)parameter;
  int32_t *block_shape = param->block_shape_;
  int32_t *crops = param->crops_;
  int mul_block_shape = 1;

  for (size_t i = 0; i < 2; ++i) {
    if (block_shape[i] <= 0) {
      return NNACL_PARAM_INVALID;
    }
    if (input_shape[kNHWC_N] % block_shape[i]) {
      return NNACL_ERR;
    }
    mul_block_shape *= block_shape[i];
  }

  if (input_shape[kNHWC_N] < mul_block_shape) {
    return NNACL_PARAM_INVALID;
  }
  for (size_t i = 0; i < 4; ++i) {
    if (crops[i] < 0) {
      return NNACL_PARAM_INVALID;
    }
  }
  int32_t output_shape[MAX_SHAPE_SIZE];
  size_t output_shape_size = input_shape_size;
  output_shape[kNHWC_N] = input_shape[kNHWC_N] / mul_block_shape;
  output_shape[kNHWC_H] = input_shape[kNHWC_H] * block_shape[0] - crops[0] - crops[1];
  output_shape[kNHWC_W] = input_shape[kNHWC_W] * block_shape[1] - crops[2] - crops[3];
  output_shape[kNHWC_C] = input_shape[kNHWC_C];
  SetShapeArray(outputs[0], output_shape, output_shape_size);
  return NNACL_OK;
}

int SetOutputShapeFromInput(const TensorC *const *inputs, TensorC **outputs) {
  int input_shape[MAX_SHAPE_SIZE];
  size_t input_shape_size = 0;
  ShapeSet(input_shape, &input_shape_size, inputs[0]->shape_, inputs[0]->shape_size_);
  if (input_shape_size != 4) {
    return NNACL_PARAM_INVALID;
  }
  int *block_shape = (int *)(inputs[1]->data_);
  int *crops = (int *)(inputs[2]->data_);
  if (GetElementNum(inputs[1]) != 2) {
    return NNACL_PARAM_INVALID;
  }
  if (GetElementNum(inputs[2]) != 4) {
    return NNACL_PARAM_INVALID;
  }
  int mul_block_shape_ = 1;

  for (size_t i = 0; i < 2; ++i) {
    if (block_shape[i] <= 0) {
      return NNACL_PARAM_INVALID;
    }
    if (input_shape[kNHWC_N] % block_shape[i]) {
      return 1;
    }
    mul_block_shape_ *= block_shape[i];
  }

  if (input_shape[kNHWC_N] < mul_block_shape_) {
    return NNACL_PARAM_INVALID;
  }
  for (size_t i = 0; i < 4; ++i) {
    if (crops[i] < 0) {
      return NNACL_PARAM_INVALID;
    }
  }
  int32_t output_shape[MAX_SHAPE_SIZE];
  size_t output_shape_size = input_shape_size;
  output_shape[kNHWC_N] = input_shape[kNHWC_N] / mul_block_shape_;
  output_shape[kNHWC_H] = input_shape[kNHWC_H] * block_shape[0] - crops[0] - crops[1];
  output_shape[kNHWC_W] = input_shape[kNHWC_W] * block_shape[1] - crops[2] - crops[3];
  output_shape[kNHWC_C] = input_shape[kNHWC_C];
  SetShapeArray(outputs[0], output_shape, output_shape_size);
  return NNACL_OK;
}

int BatchToSpaceInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                           OpParameter *parameter) {
  if (outputs_size != 1 || (inputs_size != 1 && inputs_size != 3)) {
    return NNACL_PARAM_INVALID;
  }

  const TensorC *input = inputs[0];
  if (input->format_ != Format_NHWC) {
    return NNACL_ERR;
  }
  SetDataTypeFormat(outputs[0], input);
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  if (inputs_size == 1) {
    int ret = SetOutputShapeFromParam(inputs, outputs, parameter);
    return ret;
  }
  if (inputs_size == 3) {
    if (inputs[0]->data_ == NULL) {
      return NNACL_INFER_INVALID;
    }
    if (inputs[1]->data_ == NULL || inputs[2]->data_ == NULL) {
      return NNACL_ERR;
    }
    int ret = SetOutputShapeFromInput(inputs, outputs);
    return ret;
  }

  return NNACL_OK;
}

REG_INFER(BatchToSpace, PrimType_BatchToSpace, BatchToSpaceInferShape)
