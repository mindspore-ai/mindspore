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

#include "nnacl/infer/tile_infer.h"
#include <limits.h>
#include "nnacl/infer/infer_register.h"

void TileParamCaffe2Tflite(TileParameter *param, size_t out_shape_size) {
  if (param->dims_size_ != 0) {
    int multiples_size_tmp[5] = {0};
    for (size_t i = 0; i < out_shape_size; i++) {
      multiples_size_tmp[i] = 1;
    }
    for (size_t i = 0; i < param->dims_size_; i++) {
      if (i >= MAX_TILE_DIM_SIZE) {
        return;
      }
      multiples_size_tmp[param->dims_[i]] = param->multiples_[i];
    }
    for (size_t i = 0; i < 5; i++) {
      param->multiples_[i] = multiples_size_tmp[i];
    }
  }
}

int TileInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                   OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 2, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];

  SetDataTypeFormat(output, input);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  int out_shape[MAX_SHAPE_SIZE] = {0};
  size_t out_shape_size = 0;
  TileParameter *param = (TileParameter *)parameter;

  size_t multiples_size = 0;
  int input1_shape_size = inputs[1]->shape_size_;
  if (input1_shape_size > (int)(input->shape_size_) || input->shape_size_ > MAX_SHAPE_SIZE) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  if (input1_shape_size > MAX_TILE_DIM_SIZE) {
    return NNACL_ERR;
  }
  int data_num = GetElementNum(inputs[1]);
  multiples_size = (size_t)(data_num);
  if (inputs[1]->data_type_ != kNumberTypeInt && inputs[1]->data_type_ != kNumberTypeInt32) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  int *input1_data = inputs[1]->data_;
  if (input1_data == NULL) {
    return NNACL_INFER_INVALID;
  }
  for (int i = 0; i < data_num; i++) {
    param->multiples_[i] = input1_data[i];
  }

  int *dims = param->dims_;
  size_t dims_size = param->dims_size_;
  if (dims_size == 0) {
    int dim_num = GetElementNum(inputs[1]);
    if (dim_num > MAX_SHAPE_SIZE) {
      return NNACL_ERR;
    }
    for (int dim = 0; dim < dim_num; ++dim) {
      ShapePush(dims, &dims_size, dim);
    }
    param->dims_size_ = dims_size;
  }
  if (multiples_size != dims_size) {
    return NNACL_ERR;
  }
  for (size_t i = 0; i < input->shape_size_; ++i) {
    ShapePush(out_shape, &out_shape_size, input->shape_[i]);
  }
  for (size_t i = 0; i < dims_size; ++i) {
    if (dims[i] >= MAX_SHAPE_SIZE || input->shape_[dims[i]] == 0) {
      return NNACL_ERR;
    }
    if (input->shape_[dims[i]] != 0 && param->multiples_[i] > INT_MAX / input->shape_[dims[i]]) {
      return NNACL_ERR;
    }
    MS_CHECK_FALSE(INT_MUL_OVERFLOW(input->shape_[dims[i]], (param->multiples_[i])), NNACL_ERR);
    out_shape[dims[i]] = input->shape_[dims[i]] * (param->multiples_[i]);
  }
  // change caffe param format to tflite
  TileParamCaffe2Tflite(param, out_shape_size);
  SetShapeArray(output, out_shape, out_shape_size);
  return NNACL_OK;
}

REG_INFER(Tile, PrimType_TileFusion, TileInferShape)
