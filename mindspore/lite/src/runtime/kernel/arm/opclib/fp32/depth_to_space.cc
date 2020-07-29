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
#include "src/runtime/kernel/arm/opclib/fp32/depth_to_space.h"
#include "src/runtime/kernel/arm/opclib/arithmetic_common.h"

void DepthToSpaceForNHWC(const float *input, float *output, int *in_shape, int *out_shape, int shape_size,
                         int block_size) {
  int *in_strides = (int *)(malloc(sizeof(int) * shape_size));
  ComputeStrides(in_shape, in_strides, shape_size);
  int *out_strides = (int *)(malloc(sizeof(int) * shape_size));
  ComputeStrides(out_shape, out_strides, shape_size);
  for (int i = 0; i < in_shape[0]; ++i) {
    size_t in_offset_n = i * in_strides[0];
    size_t out_offset_n = i * out_strides[0];
    for (int j = 0; j < in_shape[1]; ++j) {
      size_t in_offset_h = in_offset_n + j * in_strides[1];
      size_t out_offset_h = out_offset_n + j * block_size * out_strides[1];
      for (int k = 0; k < in_shape[2]; ++k) {
        size_t in_offset_w = in_offset_h + k * in_strides[2];
        size_t out_offset_w = out_offset_h + k * block_size * out_strides[2];
        for (int l = 0; l < block_size; ++l) {
          memcpy(output + out_offset_w + l * out_strides[1], input + in_offset_w + l * block_size * out_strides[2],
                 block_size * out_strides[2] * 4);
        }
      }
    }
  }
  free(out_strides);
  free(in_strides);
}
