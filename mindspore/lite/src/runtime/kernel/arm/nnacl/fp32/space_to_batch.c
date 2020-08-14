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
#include "nnacl/fp32/concat.h"
#include "nnacl/op_base.h"

int EnumElement(int *shape, int n_dims) {
  int total = 1;
  for (int i = 0; i < n_dims; i++) {
    total *= shape[i];
  }
  return total;
}

void TransposeForNHWC(const float *in_data, float *out_data, int *strides, int *out_strides, int *perm,
                      int *output_shape) {
  const int stride0 = strides[perm[0]];
  const int stride1 = strides[perm[1]];
  const int stride2 = strides[perm[2]];
  const int stride3 = strides[perm[3]];
  const int stride4 = strides[perm[4]];
  const int out_stride0 = out_strides[0];
  const int out_stride1 = out_strides[1];
  const int out_stride2 = out_strides[2];
  const int out_stride3 = out_strides[3];
  const int out_stride4 = out_strides[4];
  const int output0 = output_shape[0];
  const int output1 = output_shape[1];
  const int output2 = output_shape[2];
  const int output3 = output_shape[3];
  const int output4 = output_shape[4];

  for (int i = 0; i < output0; ++i) {
    int out_stride0_i = i * out_stride0;
    int stride0_i = i * stride0;
    for (int j = 0; j < output1; ++j) {
      int out_stride1_j = j * out_stride1;
      int stride1_j = j * stride1;
      for (int k = 0; k < output2; ++k) {
        int out_stride2_k = k * out_stride2;
        int stride2_k = k * stride2;
        for (int m = 0; m < output3; ++m) {
          int out_stride3_m = m * out_stride3;
          int stride3_m = m * stride3;
          for (int n = 0; n < output4; ++n) {
            int out_stride4_n = n * out_stride4;
            int stride4_n = n * stride4;
            memcpy(out_data + out_stride0_i + out_stride1_j + out_stride2_k + out_stride3_m + out_stride4_n,
                   in_data + stride0_i + stride1_j + stride2_k + stride3_m + stride4_n, stride4 * sizeof(float));
          }
        }
      }
    }
  }
}

int SpaceToBatchForNHWC(const float *input, float *output, int *in_shape, int shape_size, int *block_sizes) {
  int trans_in_shape[6] = {in_shape[0],    in_shape[1] / block_sizes[0],
                           block_sizes[0], in_shape[2] / block_sizes[1],
                           block_sizes[1], in_shape[3]};
  int trans_out_shape[6] = {
    in_shape[0], block_sizes[0], block_sizes[1], in_shape[1] / block_sizes[0], in_shape[2] / block_sizes[1],
    in_shape[3]};
  int in_strides[C4NUM + 2];
  ComputeStrides(trans_in_shape, in_strides, shape_size + 2);
  int out_strides[C4NUM + 2];
  ComputeStrides(trans_out_shape, out_strides, shape_size + 2);

  int perm[6] = {0, 2, 4, 1, 3, 5};
  TransposeForNHWC(input, output, in_strides, out_strides, perm, trans_out_shape);
  return NNACL_OK;
}

void DoPadding(const float *input, float *padded_input, SpaceToBatchParameter param, float *tmp_space[]) {
  float *tmp = padded_input;
  (void)memcpy(tmp, input, param.num_elements_ * sizeof(float));
  float *target = tmp_space[0];
  float *tmp_zeros = tmp_space[1];
  float *tmp2 = NULL;
  int cur_shape[param.n_dims_], cur_start_shape[param.n_dims_], cur_end_shape[param.n_dims_],
    cur_target_shape[param.n_dims_];
  float *concat_inputs[3];
  int *concat_shapes[4];

  for (int i = 0; i < param.n_dims_; i++) {
    cur_shape[i] = param.in_shape_[i];
    cur_start_shape[i] = param.in_shape_[i];
    cur_end_shape[i] = param.in_shape_[i];
    cur_target_shape[i] = param.in_shape_[i];
  }
  for (int i = 0; i < param.n_space_dims_; ++i) {
    if (param.padded_in_shape_[i + 1] > param.in_shape_[i + 1]) {
      int concat_idx = 0;
      cur_target_shape[i + 1] = 0;
      if (param.paddings_[2 * i] != 0) {
        cur_start_shape[i + 1] = param.paddings_[2 * i];
        concat_inputs[concat_idx] = tmp_zeros;
        concat_shapes[concat_idx++] = cur_start_shape;
        cur_target_shape[i + 1] += cur_start_shape[i + 1];
      }

      concat_inputs[concat_idx] = tmp;
      concat_shapes[concat_idx++] = cur_shape;
      cur_target_shape[i + 1] += cur_shape[i + 1];
      if (param.paddings_[2 * i + 1] != 0) {
        cur_end_shape[i + 1] = param.paddings_[2 * i + 1];
        concat_inputs[concat_idx] = tmp_zeros;
        concat_shapes[concat_idx++] = cur_end_shape;
        cur_target_shape[i + 1] += cur_end_shape[i + 1];
      }
      concat_shapes[concat_idx] = cur_target_shape;
      Concat((void **)concat_inputs, concat_idx, i + 1, concat_shapes, param.n_dims_, target);

      tmp2 = tmp;
      tmp = target;
      target = tmp2;
      cur_start_shape[i + 1] = cur_end_shape[i + 1] = cur_shape[i + 1] = concat_shapes[concat_idx][i + 1];
    }
  }
  if (padded_input != tmp) {
    memcpy(padded_input, tmp, param.num_elements_padded_ * sizeof(float));
  }
}

int SpaceToBatch(const float *input, float *output, SpaceToBatchParameter param, float *tmp_space[3]) {
  float *padded_input = NULL;
  int ret;
  if (param.need_paddings_) {
    if (tmp_space[0] == NULL || tmp_space[1] == NULL || tmp_space[2] == NULL) {
      return NNACL_NULL_PTR;
    }
    padded_input = tmp_space[0];
    DoPadding(input, padded_input, param, tmp_space + 1);
  }

  if (param.need_paddings_) {
    ret = SpaceToBatchForNHWC(padded_input, output, param.padded_in_shape_, param.n_dims_, param.block_sizes_);
  } else {
    ret = SpaceToBatchForNHWC(input, output, param.padded_in_shape_, param.n_dims_, param.block_sizes_);
  }
  return ret;
}
