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

#include "nnacl/infer/slice_infer.h"
#include "nnacl/infer/infer_register.h"

static bool CheckInputsDataType(const TensorC *const *inputs, size_t inputs_size) {
  // not support data_type of slice's begin and size is not int32
  if (inputs_size >= 2) {
    if (inputs[1]->data_type_ != kNumberTypeInt32) {
      return false;
    }
  }
  if (inputs_size == 3) {
    if (inputs[2]->data_type_ != kNumberTypeInt32) {
      return false;
    }
  }
  return true;
}

int InitBeginAndSizeParam(const TensorC *const *inputs, SliceParameter *param) {
  /* init begin parameter */
  int slice_begin_size = GetElementNum(inputs[1]);
  int *begin_ptr = (int *)(inputs[1]->data_);
  if (slice_begin_size != param->param_length_ || begin_ptr == NULL) {
    return NNACL_INFER_INVALID;
  }
  if (slice_begin_size > MAX_AXIS_SIZE) {
    return NNACL_ERR;
  }
  for (size_t i = 0; i < slice_begin_size; i++) {
    param->begin_[i] = begin_ptr[i];
  }

  /* init size parameter */
  int slice_size_size = GetElementNum(inputs[2]);
  int *size_ptr = (int *)(inputs[2]->data_);
  if (slice_size_size != param->param_length_ || size_ptr == NULL) {
    return NNACL_INFER_INVALID;
  }
  if (slice_size_size > MAX_AXIS_SIZE) {
    return NNACL_ERR;
  }
  for (size_t i = 0; i < slice_size_size; i++) {
    param->size_[i] = size_ptr[i];
  }
  return NNACL_OK;
}

int SliceInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                    OpParameter *parameter) {
  int ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 3, 1);
  if (ret != NNACL_OK) {
    return ret;
  }

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, input);

  if (!CheckInputsDataType(inputs, inputs_size)) {
    return NNACL_ERR;
  }

  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  if (input->shape_size_ > MAX_SHAPE_SIZE) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  SliceParameter *param = (SliceParameter *)parameter;
  param->param_length_ = (int)(input->shape_size_);
  output->shape_size_ = input->shape_size_;

  ret = InitBeginAndSizeParam(inputs, param);
  if (ret != NNACL_OK) {
    return ret;
  }

  /* infer output shape information */
  int begin[MAX_SHAPE_SIZE];
  int size[MAX_SHAPE_SIZE];
  for (int32_t i = 0; i < param->param_length_; ++i) {
    begin[param->axis_[i]] = param->begin_[i];
    size[param->axis_[i]] = param->size_[i];
  }

  for (int32_t i = 0; i < param->param_length_; ++i) {
    if (size[i] < 0 && size[i] != -1) {
      return NNACL_PARAM_INVALID;
    }
    if (begin[i] < 0) {
      return NNACL_PARAM_INVALID;
    }
    if (input->shape_[i] < begin[i]) {
      return NNACL_PARAM_INVALID;
    }
    if (size[i] > (input->shape_[i] - begin[i])) {
      return NNACL_PARAM_INVALID;
    }

    output->shape_[i] = size[i] < 0 ? input->shape_[i] - begin[i] : size[i];
  }
  return NNACL_OK;
}

REG_INFER(Slice, PrimType_SliceFusion, SliceInferShape)
