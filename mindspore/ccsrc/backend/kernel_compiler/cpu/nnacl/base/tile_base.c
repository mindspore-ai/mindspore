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

#include "nnacl/base/tile_base.h"
#include <string.h>

void DoCopyData(const uint8_t *input_data, uint8_t *output_data, size_t size, size_t data_size, size_t multiple) {
  uint8_t *out_data = output_data;
  for (size_t i = 0; i < multiple; ++i) {
    (void)memcpy(out_data, input_data, size * sizeof(uint8_t) * data_size);
    out_data += size * data_size;
  }
}

int DoTileOneDimension(uint8_t *input_data, uint8_t *output_data, size_t dim, TileParameter *parameter) {
  size_t src_dim_size = parameter->in_shape_[dim];
  if (dim == parameter->in_dim_ - 1) {
    DoCopyData(input_data, output_data, src_dim_size, parameter->data_size_, parameter->multiples_[dim]);
    return 0;
  }
  for (size_t i = 0; i < src_dim_size; ++i) {
    for (size_t j = 0; j < parameter->multiples_[dim]; ++j) {
      size_t in_pos = parameter->in_strides_[dim] * i;
      size_t out_pos = parameter->out_strides_[dim] * (i + j * src_dim_size);
      DoTileOneDimension(input_data + in_pos * parameter->data_size_, output_data + out_pos * parameter->data_size_,
                         dim + 1, parameter);
    }
  }
  return 0;
}

void Tile(void *input_data, void *output_data, TileParameter *parameter) {
  DoTileOneDimension((uint8_t *)input_data, (uint8_t *)output_data, 0, parameter);
}

void TileSimple(void *input_data, void *output_data, size_t begin, size_t end, TileParameter *parameter) {
  uint8_t *out_data = output_data;
  uint8_t *in_data = input_data;
  for (size_t i = begin; i < end; ++i) {
    uint8_t *src = in_data + i * parameter->fast_stride_ * parameter->data_size_;
    uint8_t *dst = out_data + i * parameter->fast_stride_ * parameter->fast_multiple_ * parameter->data_size_;
    for (size_t m = 0; m < parameter->fast_multiple_; ++m) {
      (void)memcpy(dst, src, parameter->fast_stride_ * parameter->data_size_);
      dst += parameter->fast_stride_ * parameter->data_size_;
    }
  }
}
