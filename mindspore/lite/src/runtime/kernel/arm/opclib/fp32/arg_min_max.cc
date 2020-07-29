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
#include "src/runtime/kernel/arm/opclib/fp32/arg_min_max.h"
#include <float.h>

void GetCalcParameter(const int *shape, int dims_number, int axis, int *pre_axis_count, int *axis_count,
                      int *after_axis_count) {
  *pre_axis_count = 1;
  for (int i = 0; i < axis; ++i) {
    *pre_axis_count = (*pre_axis_count) * shape[i];
  }

  *axis_count = shape[axis];

  *after_axis_count = 1;
  for (int i = axis + 1; i < dims_number; ++i) {
    *after_axis_count = (*after_axis_count) * shape[i];
  }
}

void ArgMax(const float *input, const int *shape, int dims_number, int axis, bool out_value, float *output) {
  int pre_axis_count = 1;
  int axis_count = 1;
  int after_axis_count = 1;
  GetCalcParameter(shape, dims_number, axis, &pre_axis_count, &axis_count, &after_axis_count);

  for (int i = 0; i < pre_axis_count; ++i) {
    int64_t output_offset = i * after_axis_count;
    int64_t input_offset = output_offset * axis_count;

    for (int j = 0; j < after_axis_count; ++j) {
      float value = -FLT_MAX;
      float index = 0.0f;
      for (int k = 0; k < axis_count; ++k) {
        float value_tmp = input[input_offset + k * after_axis_count + j];
        if (value_tmp > value) {
          value = value_tmp;
          index = k;
        }
      }
      output[output_offset + j] = out_value ? value : index;
    }
  }
}

void ArgMin(const float *input, const int *shape, int dims_number, int axis, bool out_value, float *output) {
  int pre_axis_count = 1;
  int axis_count = 1;
  int after_axis_count = 1;
  GetCalcParameter(shape, dims_number, axis, &pre_axis_count, &axis_count, &after_axis_count);

  for (int i = 0; i < pre_axis_count; ++i) {
    int64_t output_offset = i * after_axis_count;
    int64_t input_offset = output_offset * axis_count;
    for (int j = 0; j < after_axis_count; ++j) {
      float value = FLT_MAX;
      float index = 0.0f;
      for (int k = 0; k < axis_count; ++k) {
        float value_tmp = input[input_offset + k * after_axis_count + j];
        if (value_tmp < value) {
          value = value_tmp;
          index = k;
        }
      }
      output[output_offset + j] = out_value ? value : index;
    }
  }
}

