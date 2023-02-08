/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "nnacl/base/space_to_depth_base.h"
#include "nnacl/common_func.h"
#include "nnacl/errorcode.h"

int SpaceToDepthForNHWC(const void *input, void *output, const int *in_shape, const int *out_shape, int shape_size,
                        SpaceToDepthParameter *param, int task_id) {
  if (param->op_parameter_.thread_num_ == 0) {
    return NNACL_ERR;
  }
  int output_h = out_shape[kNHWC_H];
  int unit_per_thread = UP_DIV(output_h, param->op_parameter_.thread_num_);
  int h_start = unit_per_thread * task_id;
  int h_end = MSMIN(h_start + unit_per_thread, output_h);

  int block_size = param->block_size_;
  int in_strides[C4NUM];
  int out_strides[C4NUM];
  ComputeStrides(in_shape, in_strides, shape_size);
  ComputeStrides(out_shape, out_strides, shape_size);
  for (int i = 0; i < out_shape[0]; ++i) {
    int64_t in_offset_n = i * in_strides[0];
    int64_t out_offset_n = i * out_strides[0];
    for (int j = h_start; j < h_end; ++j) {
      int64_t in_offset_h = in_offset_n + j * block_size * in_strides[1];
      int64_t out_offset_h = out_offset_n + j * out_strides[1];
      for (int k = 0; k < out_shape[2]; ++k) {
        int64_t in_offset_w = in_offset_h + k * block_size * in_strides[2];
        int64_t out_offset_w = out_offset_h + k * out_strides[2];
        for (int l = 0; l < block_size; ++l) {
          memcpy((int8_t *)output + (out_offset_w + l * block_size * in_strides[DIMENSION_2D]) * param->date_type_len,
                 (const int8_t *)input + (in_offset_w + l * in_strides[DIMENSION_1D]) * param->date_type_len,
                 block_size * in_strides[DIMENSION_2D] * param->date_type_len);
        }
      }
    }
  }
  return NNACL_OK;
}
