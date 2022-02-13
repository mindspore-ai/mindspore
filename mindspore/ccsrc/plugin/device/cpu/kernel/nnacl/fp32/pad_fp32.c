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

#include "nnacl/fp32/pad_fp32.h"
#include "nnacl/common_func.h"
#include "nnacl/errorcode.h"

void Pad(const float *input_data, float *output_data, const int *input_shape, const int *output_shape,
         const int *paddings, int tid, int thread_num) {
  if (thread_num == 0) {
    return;
  }
  int in[DEFAULT_PAD_NDIMS], out[DEFAULT_PAD_NDIMS];
  for (in[0] = 0; in[0] < input_shape[0]; in[0]++) {
    out[0] = in[0] + paddings[0];
    for (in[1] = tid; in[1] < input_shape[1]; in[1] += thread_num) {
      out[1] = in[1] + paddings[2];
      for (in[2] = 0; in[2] < input_shape[2]; in[2]++) {
        out[2] = in[2] + paddings[4];
        for (in[3] = 0; in[3] < input_shape[3]; in[3]++) {
          out[3] = in[3] + paddings[6];
          for (in[4] = 0; in[4] < input_shape[4]; in[4]++) {
            out[4] = in[4] + paddings[8];
            float *dst = output_data + Offset6d(output_shape, out) + paddings[10];
            const float *src = input_data + Offset6d(input_shape, in);
            memcpy(dst, src, input_shape[5] * (int)(sizeof(float)));
          }
        }
      }
    }
  }
}

int TransOut2InputDimIndex(int out_dim_index, int left_pad, int in_dim, int offset) {
  if (out_dim_index < left_pad) {
    // left pad
    const int index_sum = left_pad + offset - 1;
    int in_index = MSMAX(index_sum - out_dim_index, offset);
    return MSMIN(in_index, in_dim - 1);
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

int GetInputFlattenIndex(int out_flatten_index, const int *input_shape, const PadParameter *pad_param) {
  int in_flatten_index = 0;
  for (int i = 0; i < DEFAULT_PAD_NDIMS; ++i) {
    int left_pad = pad_param->paddings_[i * 2];
    NNACL_CHECK_ZERO_RETURN_ERR(pad_param->out_strides[i]);
    int out_dim_index = out_flatten_index / pad_param->out_strides[i];
    out_flatten_index %= pad_param->out_strides[i];
    int in_dim_index = TransOut2InputDimIndex(out_dim_index, left_pad, input_shape[i], pad_param->mirror_offset_);
    in_flatten_index += in_dim_index * pad_param->in_strides[i];
  }
  return in_flatten_index;
}

void MirrorPad(const float *input_data, float *output_data, const int *input_shape, const PadParameter *pad_param,
               int begin, int end) {
  int i = 0;
  for (i = begin; i < end; ++i) {
    output_data[i] = input_data[GetInputFlattenIndex(i, input_shape, pad_param)];
  }
}
