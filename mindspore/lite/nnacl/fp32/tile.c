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

#include "nnacl/fp32/tile.h"
#include <string.h>

void DoCopyData(float *input_data, float *output_data, size_t size, size_t multiple) {
  float *out_data = output_data;
  for (size_t i = 0; i < multiple; ++i) {
    (void)memcpy(out_data, input_data, size * sizeof(float));
    out_data += size;
  }
}

int DoTileOneDimension(float *input_data, float *output_data, size_t dim, TileParameter *parameter) {
  size_t src_dim_size = parameter->in_shape_[dim];
  if (dim == parameter->in_dim_ - 1) {
    DoCopyData(input_data, output_data, src_dim_size, parameter->multiples_[dim]);
    return 0;
  }
  for (size_t i = 0; i < src_dim_size; ++i) {
    for (size_t j = 0; j < parameter->multiples_[dim]; ++j) {
      size_t in_pos = parameter->in_strides_[dim] * i;
      size_t out_pos = parameter->out_strides_[dim] * (i + j * src_dim_size);
      DoTileOneDimension(input_data + in_pos, output_data + out_pos, dim + 1, parameter);
    }
  }
  return 0;
}

void Tile(float *input_data, float *output_data, TileParameter *parameter) {
  DoTileOneDimension(input_data, output_data, 0, parameter);
}
