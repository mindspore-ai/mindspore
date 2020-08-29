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
#include "nnacl/fp32/space_to_batch.h"
#include "nnacl/arithmetic_common.h"
#include "nnacl/errorcode.h"
#include "nnacl/op_base.h"

void DoSpaceToBatchNHWC(const float *input, float *output, SpaceToBatchParameter *param, int *in_shape,
                        int *out_shape) {
  int out_dim0 = out_shape[0];
  int out_dim1 = out_shape[1];
  int out_dim2 = out_shape[2];
  int copy_num = out_shape[3];
  int block_w = param->block_sizes_[1];
  int block_h = param->block_sizes_[0];
  int in_strides[4];
  ComputeStrides(in_shape, in_strides, 4);
  int out_strides[4];
  ComputeStrides(out_shape, out_strides, 4);
  size_t copy_size = copy_num * sizeof(float);
  size_t out_offset = 0;
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

void DoSpaceToBatchPaddingNHWC(const float *input, float *output, int *in_shape, int *padding, int *out_shape,
                              const float *pedding_h_data, const float *pedding_w_data) {
  int in_h = in_shape[1];
  int in_w = in_shape[2];
  int in_c = in_shape[3];
  int out_w = out_shape[2];
  int out_c = out_shape[3];
  size_t ped_h_num = out_w * out_c;
  size_t ped_h_size = ped_h_num * sizeof(float);
  size_t ped_w_size = out_c * sizeof(float);
  size_t out_offset = 0;
  int in_strides[4];
  ComputeStrides(in_shape, in_strides, 4);
  int out_strides[4];
  ComputeStrides(out_shape, out_strides, 4);
  size_t copy_size = in_c * sizeof(float);
  for (int i = 0; i < in_shape[0]; ++i) {
    size_t in_offset0 = i * in_strides[0];
    for (int pad_h_top = 0; pad_h_top < padding[0]; ++pad_h_top) {
        memcpy(output + out_offset, pedding_h_data, ped_h_size);
        out_offset += ped_h_num;
    }
    for (int j = 0; j < in_h; ++j) {
      size_t in_offset1 = in_offset0 + j * in_strides[1];
      for (int pad_w_left = 0; pad_w_left < padding[2]; ++pad_w_left) {
        memcpy(output + out_offset, pedding_w_data, ped_w_size);
        out_offset += out_c;
      }
      for (int k = 0; k < in_w; ++k) {
        size_t in_offset2 = in_offset1 + k * in_strides[2];
        memcpy(output + out_offset, input + in_offset2, copy_size);
        out_offset += in_c;
      }
      for (int pad_w_right = 0; pad_w_right < padding[3]; ++pad_w_right) {
        memcpy(output + out_offset, pedding_w_data, ped_w_size);
        out_offset += out_c;
      }
    }
    for (int pad_h_bottom = 0; pad_h_bottom < padding[1]; ++pad_h_bottom) {
      memcpy(output + out_offset, pedding_h_data, ped_h_size);
      out_offset += ped_h_num;
    }
  }
}
