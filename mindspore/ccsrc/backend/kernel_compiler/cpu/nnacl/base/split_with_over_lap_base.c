/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "nnacl/base/split_with_over_lap_base.h"
#include "nnacl/split_parameter.h"
#include <string.h>
#include "nnacl/errorcode.h"

int DoSplitWithOverlap(char *in_data, char **out_data, int num_split, int split_dim_size, int element_bytes,
                       int outer_total_dim, int inner_stride, const int *start_indices, const int *end_indices) {
  int input_stride = split_dim_size * inner_stride * element_bytes;
  for (int slice_idx = 0; slice_idx < num_split; slice_idx++) {
    int out_stride = (end_indices[slice_idx] - start_indices[slice_idx]) * inner_stride * element_bytes;
    char *src_ptr = in_data + start_indices[slice_idx] * inner_stride * element_bytes;
    for (int out_idx = 0; out_idx < outer_total_dim; out_idx++) {
      (void)(memcpy(out_data[slice_idx] + out_idx * out_stride, src_ptr, out_stride));
      src_ptr += input_stride;
    }
  }
  return NNACL_OK;
}

int DoSplitWithOverlapParallel(char *in_data, char **out_data, int slice_idx, int split_dim_size, int element_bytes,
                               int outer_total_dim, int inner_stride, const int *start_indices,
                               const int *end_indices) {
  int input_stride = split_dim_size * inner_stride * element_bytes;
  int out_stride = (end_indices[slice_idx] - start_indices[slice_idx]) * inner_stride * element_bytes;
  char *src_ptr = in_data + start_indices[slice_idx] * inner_stride * element_bytes;
  for (int i = 0; i < outer_total_dim; i++) {
    (void)memcpy(out_data[slice_idx] + i * out_stride, src_ptr, out_stride);
    src_ptr += input_stride;
  }
  return NNACL_OK;
}
