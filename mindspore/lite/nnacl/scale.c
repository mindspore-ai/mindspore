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

#include "nnacl/scale.h"
#include "nnacl/errorcode.h"

int DoScale(float *in_data, float *out_data, float *scale, float *offset, int task_id, ScaleParameter *scale_param) {
  if (in_data == NULL || out_data == NULL || scale == NULL || offset == NULL || scale_param == NULL) {
    return NNACL_ERR;
  }

  if (scale_param->has_offset_) {
    for (int out = task_id; out < scale_param->outer_size_; out += scale_param->op_parameter_.thread_num_) {
      int out_offset = out * scale_param->axis_size_ * scale_param->inner_size_;
      for (int i = 0; i < scale_param->axis_size_; i++) {
        int axis_offset = out_offset + i * scale_param->inner_size_;
        for (int in = 0; in < scale_param->inner_size_; in++) {
          int in_offset = axis_offset + in;
          out_data[in_offset] = in_data[in_offset] * scale[i] + offset[i];
        }
      }
    }
  } else {
    for (int out = task_id; out < scale_param->outer_size_; out += scale_param->op_parameter_.thread_num_) {
      int out_offset = out * scale_param->axis_size_ * scale_param->inner_size_;
      for (int i = 0; i < scale_param->axis_size_; i++) {
        int axis_offset = out_offset + i * scale_param->inner_size_;
        for (int in = 0; in < scale_param->inner_size_; in++) {
          int in_offset = axis_offset + in;
          out_data[in_offset] = in_data[in_offset] * scale[i];
        }
      }
    }
  }
  return NNACL_OK;
}
