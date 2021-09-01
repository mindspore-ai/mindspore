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
#include "nnacl/int8/space_to_batch_int8.h"
#include "nnacl/common_func.h"

void DoSpaceToBatchNHWCInt8(const int8_t *input, int8_t *output, const int *block_sizes, const int *in_shape,
                            const int *out_shape) {
  int out_dim0 = out_shape[0];
  int out_dim1 = out_shape[1];
  int out_dim2 = out_shape[2];
  int copy_num = out_shape[3];
  int block_w = block_sizes[1];
  int block_h = block_sizes[0];
  int in_strides[4] = {0};
  ComputeStrides(in_shape, in_strides, 4);
  int out_strides[4] = {0};
  ComputeStrides(out_shape, out_strides, 4);
  size_t copy_size = copy_num * sizeof(int8_t);
  size_t out_offset = 0;

  NNACL_CHECK_ZERO_RETURN(in_shape[0]);
  NNACL_CHECK_ZERO_RETURN(block_w);
  for (int n = 0; n < out_dim0; ++n) {
    int in_n = n % in_shape[0];
    int32_t stride_w = (n / in_shape[0]) % block_w;
    int32_t stride_h = (n / in_shape[0]) / block_w;
    size_t in_offset0 = in_n * in_strides[0];
    for (int h = 0; h < out_dim1; ++h) {
      size_t in_offset1 = in_offset0 + (h * block_h + stride_h) * in_strides[1];
      for (int w = 0; w < out_dim2; ++w) {
        size_t in_offset2 = in_offset1 + (w * block_w + stride_w) * in_strides[2];
        memcpy(output + out_offset, input + in_offset2, copy_size);
        out_offset += copy_num;
      }
    }
  }
}

void DoSpaceToBatchPaddingNHWCInt8(const int8_t *input, int8_t *output, SpaceToBatchParameter *param, int32_t zp) {
  int block_shape_h = param->block_sizes_[0];
  int block_shape_w = param->m_ == 2 ? param->block_sizes_[1] : 1;
  int in_b = param->input_shape_[0];
  int in_h = param->input_shape_[1];
  int in_w = param->input_shape_[2];
  int channel = param->input_shape_[3];
  int out_h = param->output_shape_[1];
  int out_w = param->output_shape_[2];
  int pad_t = param->paddings_[0];
  int pad_l = param->m_ == 2 ? param->paddings_[2] : 0;

  NNACL_CHECK_ZERO_RETURN(in_b);
  NNACL_CHECK_ZERO_RETURN(block_shape_w);
  for (int i = 0; i < param->output_shape_[0]; ++i) {
    int in_batch = i % in_b;
    int offset_w = (i / in_b) % block_shape_w;
    int offset_h = (i / in_b) / block_shape_w;
    int in_b_offset = in_batch * in_h * in_w * channel;
    int out_b_offset = i * out_h * out_w * channel;
    for (int j = 0; j < out_h; ++j) {
      int out_h_offset = out_b_offset + j * out_w * channel;
      for (int k = 0; k < out_w; ++k) {
        int8_t *out_ptr = output + out_h_offset + k * channel;
        int index_h = j * block_shape_h + offset_h;
        int index_w = k * block_shape_w + offset_w;
        if (index_h < pad_t || index_h >= (pad_t + in_h) || index_w < pad_l || index_w >= (pad_l + in_w)) {
          memset(out_ptr, zp, channel * sizeof(int8_t));
        } else {
          int in_plane_offset = in_b_offset + ((index_h - pad_t) * in_w + (index_w - pad_l)) * channel;
          const int8_t *in_ptr = input + in_plane_offset;
          memcpy(out_ptr, in_ptr, channel * sizeof(int8_t));
        }
      }
    }
  }
}
