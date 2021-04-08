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

#include "nnacl/base/depth_to_space_base.h"

void DepthToSpaceForNHWC(const void *input, void *output, const int *in_shape, const DepthToSpaceParameter *param) {
  int32_t block_size = param->block_size_;
  int32_t in_shape_dim2 = in_shape[2];
  int32_t in_shape_dim1 = in_shape[1];
  size_t copy_size = block_size * param->out_stride_dim2_ * param->data_type_size_;
  for (int i = 0; i < in_shape[0]; ++i) {
    size_t in_offset_n = i * param->in_stride_dim0_;
    size_t out_offset_n = i * param->out_stride_dim0_;
    for (int j = 0; j < in_shape_dim1; ++j) {
      size_t in_offset_h = in_offset_n + j * param->in_stride_dim1_;
      size_t out_offset_h = out_offset_n + j * block_size * param->out_stride_dim1_;
      for (int k = 0; k < in_shape_dim2; ++k) {
        size_t in_offset_w = in_offset_h + k * param->in_stride_dim2_;
        size_t out_offset_w = out_offset_h + k * block_size * param->out_stride_dim2_;
        for (int l = 0; l < block_size; ++l) {
          size_t out_offset = (out_offset_w + l * param->out_stride_dim1_) * param->data_type_size_;
          size_t in_offset = (in_offset_w + l * block_size * param->out_stride_dim2_) * param->data_type_size_;
          memcpy((int8_t *)output + out_offset, (int8_t *)input + in_offset, copy_size);
        }
      }
    }
  }
}
