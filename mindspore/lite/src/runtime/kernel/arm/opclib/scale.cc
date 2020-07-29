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

#include "src/runtime/kernel/arm/opclib/scale.h"
#include "src/runtime/kernel/arm/opclib/errorcode.h"

int DoScale(float *in_data, float *out_data, float *scale, float *offset, int units_offset, int num_unit,
            ScaleParameter *scale_param) {
  if (in_data == nullptr || out_data == nullptr || scale == nullptr || offset == nullptr || scale_param == nullptr) {
    return OPCLIB_ERR;
  }

  int in_stride_j = units_offset * scale_param->in_stride_;
  for (int j = units_offset; j < units_offset + num_unit; j++) {
    int channel = j % scale_param->channel_;
    for (int k = 0; k < scale_param->in_stride_; k++) {
      out_data[in_stride_j + k] = in_data[in_stride_j + k] * scale[channel] + offset[channel];
    }
    in_stride_j = in_stride_j + scale_param->in_stride_;
  }
  return OPCLIB_OK;
}

int DoScale(float *in_data, float *out_data, float *scale, int units_offset, int num_unit,
            ScaleParameter *scale_param) {
  if (in_data == nullptr || out_data == nullptr || scale == nullptr || scale_param == nullptr) {
    return OPCLIB_ERR;
  }

  int in_stride_j = units_offset * scale_param->in_stride_;
  for (int j = units_offset; j < units_offset + num_unit; j++) {
    int channel = j % scale_param->channel_;
    for (int k = 0; k < scale_param->in_stride_; k++) {
      out_data[in_stride_j + k] = in_data[in_stride_j + k] * scale[channel];
    }
    in_stride_j = in_stride_j + scale_param->in_stride_;
  }
  return OPCLIB_OK;
}

