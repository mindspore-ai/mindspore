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

#include "nnacl/fp32/broadcast_to_fp32.h"
#include <string.h>
#include "nnacl/op_base.h"
#include "nnacl/errorcode.h"

void PadBroadcastShapeInfo(BroadcastShapeInfo *shape_info) {
  if (shape_info->input_shape_size_ < DIMENSION_4D) {
    int input_shape_tmp[DIMENSION_4D];
    for (int i = 0; i < shape_info->input_shape_size_; ++i) {
      input_shape_tmp[i] = shape_info->input_shape_[i];
    }
    int input_shape_index = shape_info->input_shape_size_ - 1;
    for (int i = DIMENSION_4D - 1; i >= 0; --i) {
      if (input_shape_index >= 0) {
        shape_info->input_shape_[i] = input_shape_tmp[input_shape_index--];
      } else {
        shape_info->input_shape_[i] = 1;
      }
    }
  }
  if (shape_info->output_shape_size_ < DIMENSION_4D) {
    int output_shape_tmp[DIMENSION_4D];
    for (int i = 0; i < shape_info->output_shape_size_; ++i) {
      output_shape_tmp[i] = shape_info->output_shape_[i];
    }
    int output_shape_index = shape_info->output_shape_size_ - 1;
    for (int i = DIMENSION_4D - 1; i >= 0; --i) {
      if (output_shape_index >= 0) {
        shape_info->output_shape_[i] = output_shape_tmp[output_shape_index--];
      } else {
        shape_info->output_shape_[i] = 1;
      }
    }
  }
}

int BroadcastTo(const float *input, BroadcastShapeInfo *shape_info, float *output) {
  if (shape_info->input_shape_size_ > DIMENSION_4D || shape_info->output_shape_size_ > DIMENSION_4D) {
    return NNACL_ERR;
  }
  PadBroadcastShapeInfo(shape_info);
  size_t input_dim_offset[DIMENSION_4D - 1];
  input_dim_offset[2] = shape_info->input_shape_[3] * 4;
  input_dim_offset[1] = input_dim_offset[2] * shape_info->input_shape_[2];
  input_dim_offset[0] = input_dim_offset[1] * shape_info->input_shape_[1];
  size_t output_dim_offset[DIMENSION_4D - 1];
  output_dim_offset[2] = shape_info->output_shape_[3] * 4;
  output_dim_offset[1] = output_dim_offset[2] * shape_info->output_shape_[2];
  output_dim_offset[0] = output_dim_offset[1] * shape_info->output_shape_[1];
  uint8_t *in_base = (uint8_t *)input;
  uint8_t *out_base = (uint8_t *)(output);
  for (int32_t dim0 = 0; dim0 < shape_info->input_shape_[0]; ++dim0) {
    for (int32_t dim1 = 0; dim1 < shape_info->input_shape_[1]; ++dim1) {
      for (int32_t dim2 = 0; dim2 < shape_info->input_shape_[2]; ++dim2) {
        if (shape_info->input_shape_[3] == shape_info->output_shape_[3]) {
          memcpy(out_base + output_dim_offset[0] * dim0 + output_dim_offset[1] * dim1 + output_dim_offset[2] * dim2,
                 in_base + input_dim_offset[0] * dim0 + input_dim_offset[1] * dim1 + input_dim_offset[2] * dim2,
                 input_dim_offset[2]);
        } else {
          for (int32_t dim3 = 0; dim3 < shape_info->output_shape_[3]; ++dim3) {
            memcpy(out_base + output_dim_offset[0] * dim0 + output_dim_offset[1] * dim1 + output_dim_offset[2] * dim2 +
                     dim3 * 4,
                   in_base + input_dim_offset[0] * dim0 + input_dim_offset[1] * dim1 + input_dim_offset[2] * dim2, 4);
          }
        }
      }
      if (shape_info->input_shape_[2] != shape_info->output_shape_[2]) {
        for (int32_t dim2 = 0; dim2 < shape_info->output_shape_[2]; ++dim2) {
          memcpy(out_base + output_dim_offset[0] * dim0 + output_dim_offset[1] * dim1 + dim2 * output_dim_offset[2],
                 out_base + output_dim_offset[0] * dim0 + output_dim_offset[1] * dim1, output_dim_offset[2]);
        }
      }
    }
    if (shape_info->input_shape_[1] != shape_info->output_shape_[1]) {
      for (int32_t dim1 = 0; dim1 < shape_info->output_shape_[1]; ++dim1) {
        memcpy(out_base + output_dim_offset[0] * dim0 + output_dim_offset[1] * dim1,
               out_base + output_dim_offset[0] * dim0, output_dim_offset[1]);
      }
    }
  }
  if (shape_info->input_shape_[0] != shape_info->output_shape_[0]) {
    for (int32_t dim0 = 0; dim0 < shape_info->output_shape_[0]; ++dim0) {
      memcpy(out_base + output_dim_offset[0] * dim0, out_base, output_dim_offset[0]);
    }
  }
  return NNACL_OK;
}
