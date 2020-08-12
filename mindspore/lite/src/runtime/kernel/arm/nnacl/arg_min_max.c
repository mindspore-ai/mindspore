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
#include "nnacl/arg_min_max.h"
#include "nnacl/fp32/arg_min_max.h"

#define FLOAT_DATA_TYPE 43

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

void ArgMinMaxTopk1(const void *input, void *output, const int *shape, ArgMinMaxParameter *param) {
  int pre_axis_count = 1;
  int axis_count = 1;
  int after_axis_count = 1;
  GetCalcParameter(shape, param->dims_size_, param->axis_, &pre_axis_count, &axis_count, &after_axis_count);
  switch (param->data_type_) {
    case FLOAT_DATA_TYPE: {
      if (param->get_max_) {
        ArgMax(input, output, param, pre_axis_count, axis_count, after_axis_count);
      } else {
        ArgMin(input, output, param, pre_axis_count, axis_count, after_axis_count);
      }
      break;
    }
    default:
      break;
  }
}

void ArgMinMaxTopknFp32(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  if (param->get_max_) {
    switch (param->axis_) {
      case 0:
        ArgMaxDim0(input, output, in_shape, param);
        break;
      case 1:
        ArgMaxDim1(input, output, in_shape, param);
        break;
      case 2:
        ArgMaxDim2(input, output, in_shape, param);
        break;
      case 3:
        ArgMaxDim3(input, output, in_shape, param);
        break;
    }
  } else {
    switch (param->axis_) {
      case 0:
        ArgMinDim0(input, output, in_shape, param);
        break;
      case 1:
        ArgMinDim1(input, output, in_shape, param);
        break;
      case 2:
        ArgMinDim2(input, output, in_shape, param);
        break;
      case 3:
        ArgMinDim3(input, output, in_shape, param);
        break;
    }
  }
}

void ArgMinMax(const void *input, void *output, const int *in_shape, ArgMinMaxParameter *param) {
  if (param->topk_ == 1 && !param->keep_dims_) {
    ArgMinMaxTopk1(input, output, in_shape, param);
    return;
  }

  switch (param->data_type_) {
    case FLOAT_DATA_TYPE: {
      ArgMinMaxTopknFp32(input, output, in_shape, param);
      return;
    }
    default:
      break;
  }
}

#undef FLOAT_DATA_TYPE
#undef INT8_DATA_TYPE
