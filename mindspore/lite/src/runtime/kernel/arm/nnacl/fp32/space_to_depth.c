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
#include "nnacl/fp32/space_to_depth.h"
#include "nnacl/arithmetic_common.h"
#include "nnacl/errorcode.h"
#include "nnacl/op_base.h"

int SpaceToDepthForNHWC(const float *input, float *output, int *in_shape, int *out_shape, int shape_size,
                        int block_size, int h_start, int h_end) {
  if (input == NULL || output == NULL) {
    return NNACL_NULL_PTR;
  }
  if (shape_size != C4NUM) {
    return NNACL_PARAM_INVALID;
  }
  if (h_start < 0 || h_start >= h_end || h_end > out_shape[1]) {
    return NNACL_PARAM_INVALID;
  }
  int in_strides[C4NUM];
  ComputeStrides(in_shape, in_strides, shape_size);
  int out_strides[C4NUM];
  ComputeStrides(out_shape, out_strides, shape_size);
  for (int i = 0; i < out_shape[0]; ++i) {
    size_t in_offset_n = i * in_strides[0];
    size_t out_offset_n = i * out_strides[0];
    for (int j = h_start; j < h_end; ++j) {
      size_t in_offset_h = in_offset_n + j * block_size * in_strides[1];
      size_t out_offset_h = out_offset_n + j * out_strides[1];
      for (int k = 0; k < out_shape[2]; ++k) {
        size_t in_offset_w = in_offset_h + k * block_size * in_strides[2];
        size_t out_offset_w = out_offset_h + k * out_strides[2];
        for (int l = 0; l < block_size; ++l) {
          memcpy(output + out_offset_w + l * block_size * in_strides[2], input + in_offset_w + l * in_strides[1],
                 block_size * in_strides[2] * sizeof(float));
        }
      }
    }
  }
  return NNACL_OK;
}
