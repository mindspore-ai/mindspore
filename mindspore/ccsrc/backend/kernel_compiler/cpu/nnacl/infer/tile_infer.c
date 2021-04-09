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
      multiples_size_tmp[param->dims_[i]] = param->multiples_[i];
    }
    for (size_t i = 0; i < 5; i++) {
      param->multiples_[i] = multiples_size_tmp[i];
    }
  }
}

int TileInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                   OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];

  SetDataTypeFormat(output, input);
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  int out_shape[MAX_SHAPE_SIZE];
  size_t out_shape_size = 0;
  TileParameter *param = (TileParameter *)parameter;

  size_t multiples_size = 0;
  if (inputs_size != 2) {
    return NNACL_ERR;
  }
  int data_num = GetElementNum(inputs[1]);
  if (data_num > (int)(input->shape_size_)) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  multiples_size = data_num;
  int *input1_data = inputs[1]->data_;
  if (input1_data == NULL) {
    return NNACL_INFER_INVALID;
  }
  for (size_t i = 0; i < data_num; i++) {
    param->multiples_[i] = input1_data[i];
  }

#ifdef SUPPORT_TRAIN
  const size_t in_dims = input->shape_size_;
  const size_t delta_dims = in_dims - multiples_size;

  size_t i = 0;
  for (; i < delta_dims; ++i) {
    int tmp = input->shape_[i];
    ShapePush(out_shape, &out_shape_size, tmp);
  }
  for (; i < in_dims; ++i) {
    int tmp = input->shape_[i] * (param->multiples_[i - delta_dims]);
    ShapePush(out_shape, &out_shape_size, tmp);
  }
#else
  int *dims = param->dims_;
  size_t dims_size = param->dims_size_;
  if (dims_size == 0) {
    for (int dim = 0; dim < GetElementNum(inputs[1]); ++dim) {
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
    if (input->shape_[dims[i]] != 0 && param->multiples_[i] > INT_MAX / input->shape_[dims[i]]) {
      return NNACL_ERR;
    }
    out_shape[dims[i]] = input->shape_[dims[i]] * (param->multiples_[i]);
  }
  // change caffe param format to tflite
  TileParamCaffe2Tflite(param, out_shape_size);
#endif
  SetShapeArray(output, out_shape, out_shape_size);
  return NNACL_OK;
}

REG_INFER(Tile, PrimType_TileFusion, TileInferShape)
