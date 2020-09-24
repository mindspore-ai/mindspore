/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "nnacl/split.h"
#include "nnacl/split_parameter.h"
#include <string.h>
#include "nnacl/errorcode.h"

int DoSplit(float *in_data, float **out_data, const int *input_shape, int offset, int num_unit,
            SplitParameter *split_param) {
  if (in_data == NULL || out_data == NULL) {
    return NNACL_ERR;
  }
  int num_split = split_param->num_split_;
  int *split_sizes = split_param->split_sizes_;
  int *strides = split_param->strides_;
  int split_dim = split_param->split_dim_;
  int in_stride = strides[split_dim];

  float *src;
  int size_float = (int)(sizeof(float));
  int in_stride_bytes = in_stride * size_float;

  int split_which;
  int split_times;
  int stride_per_split = in_stride * input_shape[split_dim];

  split_which = offset % num_split;
  split_times = offset / num_split;
  src = in_data + split_times * stride_per_split;

  for (int i = 0; i < split_which; i++) {
    src += split_sizes[i] * in_stride;
  }

  for (int i = offset; i < offset + num_unit; i++) {
    split_which = i % num_split;
    split_times = i / num_split;
    int split_size = split_sizes[split_which];
    float *dst = out_data[split_which] + split_times * in_stride * split_size;
    (void)memcpy(dst, src, split_size * in_stride_bytes);
    src += split_size * in_stride;
  }

  return NNACL_OK;
}
