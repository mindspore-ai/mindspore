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

#include "nnacl/infer/split_infer.h"
#include "nnacl/infer/infer_register.h"

int UpdateSplitSize(const TensorC *const *inputs, size_t inputs_size, SplitParameter *param) {
  // get split size from the second input.
  if (inputs_size == DIMENSION_2D && inputs[SECOND_INPUT]->data_ != NULL) {
    if (inputs[SECOND_INPUT]->data_type_ != kNumberTypeInt32) {
      return NNACL_ERR;
    }
    int split_count = 1;
    for (size_t i = 0; i < inputs[SECOND_INPUT]->shape_size_; i++) {
      split_count *= inputs[SECOND_INPUT]->shape_[i];
    }
    param->split_count_ = split_count;
    for (int i = 0; i < split_count; i++) {
      param->split_sizes_[i] = ((int *)(inputs[SECOND_INPUT]->data_))[i];
    }
  }
  if (param->split_count_ == 0) {
    const TensorC *input = inputs[0];
    if (input->shape_[param->split_dim_] % param->num_split_ != 0) {
      return NNACL_ERR;
    }
    for (int i = 0; i < param->num_split_; ++i) {
      param->split_sizes_[i] = input->shape_[param->split_dim_] / param->num_split_;
    }
  }
  return NNACL_OK;
}

int SetSplitOutputShape(const TensorC *input, TensorC **outputs, SplitParameter *param) {
  for (int i = 0; i < param->num_split_; ++i) {
    int output_shape[MAX_SHAPE_SIZE];
    size_t output_shape_size = 0;
    ShapeSet(output_shape, &output_shape_size, input->shape_, input->shape_size_);
    int split_dim_i = input->shape_[param->split_dim_];
    if (i == param->num_split_ - 1 && param->split_sizes_[i] == -1) {
      if (param->num_split_ - 1 < 0) {
        return NNACL_ERR;
      }
      for (int j = 0; j < param->num_split_ - 1; ++j) {
        split_dim_i -= param->split_sizes_[j];
      }
      param->split_sizes_[i] = split_dim_i;
    } else {
      split_dim_i = param->split_sizes_[i];
    }
    MS_CHECK_TRUE_RET(split_dim_i >= 0 && split_dim_i <= input->shape_[param->split_dim_], NNACL_ERR);
    output_shape[param->split_dim_] = split_dim_i;
    SetShapeArray(outputs[i], output_shape, output_shape_size);
    SetDataTypeFormat(outputs[i], input);
  }
  return NNACL_OK;
}

int SplitInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                    OpParameter *parameter) {
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  for (size_t i = 0; i < outputs_size; i++) {
    SetDataTypeFormat(outputs[i], input);
  }

  SplitParameter *param = (SplitParameter *)parameter;

  int num_split = param->num_split_ == 0 ? (int)(outputs_size) : param->num_split_;
  if (num_split == 0) {
    return NNACL_ERR;
  }
  param->num_split_ = num_split;
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  if (input->shape_size_ > MAX_SHAPE_SIZE) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  int split_dim = param->split_dim_ < 0 ? ((int)(input->shape_size_)) + param->split_dim_ : param->split_dim_;
  if (split_dim >= (int)(input->shape_size_) || split_dim < 0) {
    return NNACL_ERR;
  }
  param->split_dim_ = split_dim;
  if ((int)(outputs_size) != num_split) {
    return NNACL_ERR;
  }

  int ret = UpdateSplitSize(inputs, inputs_size, param);
  if (ret != NNACL_OK) {
    return ret;
  }
  ret = SetSplitOutputShape(input, outputs, param);
  if (ret != NNACL_OK) {
    return ret;
  }
  return NNACL_OK;
}

REG_INFER(Split, PrimType_Split, SplitInferShape)
