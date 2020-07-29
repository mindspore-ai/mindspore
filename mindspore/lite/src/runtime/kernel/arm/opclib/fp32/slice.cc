/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/opclib/fp32/slice.h"
#include "src/runtime/kernel/arm/opclib/op_base.h"

void PadSliceParameterTo4D(SliceParameter *param) {
  int32_t begin[DIMENSION_4D];
  int32_t end[DIMENSION_4D];
  int32_t slice_size[DIMENSION_4D];
  int32_t data_shape[DIMENSION_4D];
  for (int32_t i = 0; i < param->param_length_; ++i) {
    begin[i] = param->begin_[i];
    end[i] = param->end_[i];
    slice_size[i] = param->size_[i];
    data_shape[i] = param->shape_[i];
  }
  int32_t real_index = param->param_length_ - 1;
  for (int32_t i = DIMENSION_4D - 1; i >= 0; --i) {
    if (real_index >= 0) {
      param->begin_[i] = begin[real_index];
      param->end_[i] = end[real_index];
      param->size_[i] = slice_size[real_index];
      param->shape_[i] = data_shape[real_index--];
    } else {
      param->begin_[i] = 0;
      param->end_[i] = 1;
      param->size_[i] = 1;
      param->shape_[i] = 1;
    }
  }
  param->param_length_ = DIMENSION_4D;
}

int DoSlice(const float *input, SliceParameter *param, float *output) {
  if (param->param_length_ > DIMENSION_4D) {
    return -1;
  }

  for (int i = 0; i < param->param_length_; ++i) {
    if (param->size_[i] < 0) {
      param->size_[i] = param->shape_[i] - param->begin_[i];
    }
    param->end_[i] = param->begin_[i] + param->size_[i];
  }

  if (param->param_length_ < DIMENSION_4D) {
    PadSliceParameterTo4D(param);
  }
  size_t dim_offset[DIMENSION_4D - 1];
  dim_offset[2] = param->shape_[3];
  dim_offset[1] = dim_offset[2] * param->shape_[2];
  dim_offset[0] = dim_offset[1] * param->shape_[1];
  size_t output_index = 0;
  for (int32_t dim0 = param->begin_[0]; dim0 < param->end_[0]; ++dim0) {
    for (int32_t dim1 = param->begin_[1]; dim1 < param->end_[1]; ++dim1) {
      for (int32_t dim2 = param->begin_[2]; dim2 < param->end_[2]; ++dim2) {
        for (int32_t dim3 = param->begin_[3]; dim3 < param->end_[3]; ++dim3) {
          output[output_index++] = *(input + dim0 * dim_offset[0]
            + dim1 * dim_offset[1] + dim2 * dim_offset[2] + dim3);
        }
      }
    }
  }
  return 0;
}

