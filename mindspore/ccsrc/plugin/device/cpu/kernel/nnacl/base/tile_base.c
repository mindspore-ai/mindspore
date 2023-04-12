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
#include "nnacl/errorcode.h"

void DoCopyData(const uint8_t *input_data, uint8_t *output_data, size_t size, size_t data_size, size_t multiple) {
  uint8_t *out_data = output_data;
  for (size_t i = 0; i < multiple; ++i) {
    (void)memcpy(out_data, input_data, size * sizeof(uint8_t) * data_size);
    out_data += size * data_size;
  }
}

int DoTileOneDimension(uint8_t *input_data, uint8_t *output_data, size_t dim, const TileStruct *tile) {
  int src_dim_size = tile->in_shape_[dim];
  if (dim == tile->in_dim_ - 1) {
    DoCopyData(input_data, output_data, src_dim_size, tile->data_size_, tile->multiples_[dim]);
    return NNACL_OK;
  }
  for (int i = 0; i < src_dim_size; ++i) {
    for (int j = 0; j < tile->multiples_[dim]; ++j) {
      int in_pos = tile->in_strides_[dim] * i;
      int out_pos = tile->out_strides_[dim] * (i + j * src_dim_size);
      DoTileOneDimension(input_data + in_pos * tile->data_size_, output_data + out_pos * tile->data_size_, dim + 1,
                         tile);
    }
  }
  return NNACL_OK;
}

void Tile(void *input_data, void *output_data, const TileStruct *tile) {
  DoTileOneDimension((uint8_t *)input_data, (uint8_t *)output_data, 0, tile);
}

void TileSimple(void *input_data, void *output_data, size_t begin, size_t end, const TileStruct *tile) {
  uint8_t *out_data = output_data;
  uint8_t *in_data = input_data;
  size_t dst_one_row_size = tile->fast_stride_ * tile->fast_multiple_ * tile->data_size_;
  for (size_t i = begin; i < end; ++i) {
    uint8_t *src = in_data + i * tile->fast_stride_ * tile->data_size_;
    uint8_t *dst = out_data + i * tile->fast_stride_ * tile->fast_multiple_ * tile->data_size_;
    size_t offset = tile->fast_stride_ * tile->data_size_;
    (void)memcpy(dst, src, offset);
    // copy size double each time
    while (2 * offset <= dst_one_row_size) {
      (void)memcpy(dst + offset, dst, offset);
      offset *= 2;
    }
    if (2 * offset > dst_one_row_size) {
      (void)memcpy(dst + offset, dst, dst_one_row_size - offset);
    }
  }
}
