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
#include <string.h>
#include "nnacl/errorcode.h"

int DoSplitWithOverlapParallel(const char *in_data, char **out_data, int slice_idx,
                               const SplitWithOverlapParameter *param, const int *start_indices,
                               const int *end_indices) {
  int start_index = start_indices[slice_idx];
  int end_index = end_indices[slice_idx];

  int input_stride = param->split_dim_size_ * param->inner_stride_ * param->element_bytes_;
  int out_stride = (end_index - start_index) * param->inner_stride_ * param->element_bytes_;

  const char *src_ptr = in_data + start_index * param->inner_stride_ * param->element_bytes_;
  char *dst_ptr = out_data[slice_idx];

  for (int i = 0; i < param->outer_total_dim_; i++) {
    (void)memcpy(dst_ptr + i * out_stride, src_ptr, out_stride);
    src_ptr += input_stride;
  }
  return NNACL_OK;
}
