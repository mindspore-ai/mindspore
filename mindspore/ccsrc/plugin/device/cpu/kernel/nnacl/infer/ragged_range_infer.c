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
#include "nnacl/infer/ragged_range_infer.h"
#include <math.h>
#include "nnacl/infer/infer_register.h"

int CheckInputTensor(const TensorC *const *inputs) {
  if (inputs[0]->data_ == NULL || inputs[1]->data_ == NULL || inputs[2]->data_ == NULL) {
    return NNACL_INFER_INVALID;
  }
  if (inputs[0]->shape_size_ != 0 && inputs[0]->shape_size_ != 1) {
    return NNACL_ERR;
  }
  return NNACL_OK;
}

int GetRows(const TensorC *const *inputs, bool starts_is_scalar, bool limits_is_scalar, bool deltas_is_scalar,
            int *rows) {
  NNACL_CHECK_NULL_RETURN_ERR(rows);
  int sizes[3];
  int not_scalar_count = 0;
  if (!starts_is_scalar) {
    sizes[not_scalar_count++] = inputs[0]->shape_[0];
  }
  if (!limits_is_scalar) {
    sizes[not_scalar_count++] = inputs[1]->shape_[0];
  }
  if (!deltas_is_scalar) {
    sizes[not_scalar_count++] = inputs[2]->shape_[0];
  }
  for (int i = 1; i < not_scalar_count; i++) {
    if (sizes[i] != sizes[i - 1]) {
      return NNACL_ERR;
    }
  }
  *rows = not_scalar_count == 0 ? 1 : sizes[0];
  return NNACL_OK;
}

int GetOutputValueElementNum(const TensorC *const *inputs, bool starts_is_scalar, bool limits_is_scalar,
                             bool deltas_is_scalar, int rows, int *output_value_element_num) {
  int count = 0;
  switch (inputs[0]->data_type_) {
    case kNumberTypeInt32: {
      int *starts = (int *)(inputs[0]->data_);
      int *limits = (int *)(inputs[1]->data_);
      int *deltas = (int *)(inputs[2]->data_);
      for (int i = 0; i < rows; i++) {
        int start = starts_is_scalar ? starts[0] : starts[i];
        int limit = limits_is_scalar ? limits[0] : limits[i];
        int delta = deltas_is_scalar ? deltas[0] : deltas[i];
        NNACL_CHECK_ZERO_RETURN_ERR(delta);
        count += MSMAX((int)(ceil((float)(limit - start) / delta)), 0);
      }
    } break;
    case kNumberTypeFloat32: {
      float *starts = (float *)(inputs[0]->data_);
      float *limits = (float *)(inputs[1]->data_);
      float *deltas = (float *)(inputs[2]->data_);
      for (int i = 0; i < rows; i++) {
        float start = starts_is_scalar ? starts[0] : starts[i];
        float limit = limits_is_scalar ? limits[0] : limits[i];
        float delta = deltas_is_scalar ? deltas[0] : deltas[i];
        NNACL_CHECK_ZERO_RETURN_ERR(delta);
        count += MSMAX((ceil((limit - start) / delta)), 0);
      }
    } break;
    default: {
      return NNACL_ERR;
    }
  }
  *output_value_element_num = count;
  return NNACL_OK;
}

int RaggedRangeInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                          OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 3, 2);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  outputs[0]->data_type_ = kNumberTypeInt32;
  outputs[0]->format_ = inputs[0]->format_;
  SetDataTypeFormat(outputs[1], inputs[0]);

  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  int ret = CheckInputTensor(inputs);
  if (ret != NNACL_OK) {
    return ret;
  }

  bool starts_is_scalar = inputs[0]->shape_size_ == 0;
  bool limits_is_scalar = inputs[1]->shape_size_ == 0;
  bool deltas_is_scalar = inputs[2]->shape_size_ == 0;
  int rows;
  ret = GetRows(inputs, starts_is_scalar, limits_is_scalar, deltas_is_scalar, &rows);
  if (ret != NNACL_OK) {
    return ret;
  }
  int output_value_element_num;
  ret = GetOutputValueElementNum(inputs, starts_is_scalar, limits_is_scalar, deltas_is_scalar, rows,
                                 &output_value_element_num);
  if (ret != NNACL_OK) {
    return ret;
  }
  outputs[0]->shape_size_ = 1;
  outputs[0]->shape_[0] = rows + 1;
  outputs[1]->shape_size_ = 1;
  outputs[1]->shape_[0] = output_value_element_num;
  return NNACL_OK;
}

REG_INFER(RaggedRange, PrimType_RaggedRange, RaggedRangeInferShape)
