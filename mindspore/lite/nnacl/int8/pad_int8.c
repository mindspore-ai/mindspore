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

#include "nnacl/int8/pad_int8.h"
#include "nnacl/common_func.h"
#include "nnacl/errorcode.h"

int PadConstant4D(const int8_t *in_data, int8_t *out_data, const int32_t *in_dims, const int32_t *out_dims,
                  const int32_t *paddings, const int tid, const int thread_num) {
  int32_t copy_size = in_dims[3];
  for (int n = 0; n < in_dims[0]; n++) {
    for (int h = tid; h < in_dims[1]; h += thread_num) {
      for (int w = 0; w < in_dims[2]; w++) {
        const int8_t *in = in_data + offset(in_dims, n, h, w, 0);
        int8_t *out = out_data + offset(out_dims, n + paddings[0], h + paddings[2], w + paddings[4], paddings[6]);
        memcpy(out, in, copy_size * sizeof(int8_t));
      }
    }
  }
  return NNACL_OK;
}

int TransOut2InputDimIndexInt8(int out_dim_index, int left_pad, int in_dim, int offset) {
  if (out_dim_index < left_pad) {
    // left pad
    const int index_sum = left_pad + offset - 1;
    return MSMAX(index_sum - out_dim_index, offset);
  }
  out_dim_index -= left_pad;
  if (out_dim_index < in_dim) {
    return out_dim_index;
  }
  // right pad
  out_dim_index -= in_dim;
  const int index_sum = in_dim - 1 - offset;
  return MSMAX(index_sum - out_dim_index, 0);
}

int GetInputFlattenIndexInt8(int out_flatten_index, const int *input_shape, const PadParameter *pad_param) {
  int in_flatten_index = 0;
  int i;
  for (i = 0; i < COMM_SHAPE_SIZE; ++i) {
    int left_pad = pad_param->paddings_[i * 2];
    int out_dim_index = out_flatten_index / pad_param->out_strides[i];
    out_flatten_index %= pad_param->out_strides[i];
    int in_dim_index = TransOut2InputDimIndexInt8(out_dim_index, left_pad, input_shape[i], pad_param->mirror_offset_);
    in_flatten_index += in_dim_index * pad_param->in_strides[i];
  }
  return in_flatten_index;
}

void MirrorPadInt8(const int8_t *input_data, int8_t *output_data, const int *input_shape, const PadParameter *pad_param,
                   int begin, int end) {
  int i = 0;
  for (i = begin; i < end; ++i) {
    output_data[i] = input_data[GetInputFlattenIndexInt8(i, input_shape, pad_param)];
  }
}
