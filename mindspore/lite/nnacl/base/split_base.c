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

#include "nnacl/base/split_base.h"
#include "nnacl/split_parameter.h"
#include <string.h>
#include "nnacl/errorcode.h"

int DoSplit(void *in_data, void **out_data, const int *input_shape, int offset, int num_unit,
            SplitParameter *split_param, int data_size) {
  if (in_data == NULL || out_data == NULL) {
    return NNACL_ERR;
  }

  int8_t *int8_in = (int8_t *)in_data;

  int num_split = split_param->num_split_;
  int *split_sizes = split_param->split_sizes_;
  int *strides = split_param->strides_;
  int split_dim = split_param->split_dim_;
  int in_stride = strides[split_dim];

  int in_stride_bytes = in_stride * data_size;

  int split_which;
  int split_times;
  int stride_per_split = in_stride * input_shape[split_dim];

  split_which = offset % num_split;
  split_times = offset / num_split;
  int8_t *src = int8_in + split_times * stride_per_split * data_size;

  for (int i = 0; i < split_which; i++) {
    src += split_sizes[i] * in_stride * data_size;
  }

  for (int i = offset; i < offset + num_unit; i++) {
    split_which = i % num_split;
    split_times = i / num_split;
    int split_size = split_sizes[split_which];
    int8_t *int8_out = (int8_t *)out_data[split_which];
    int8_t *dst = int8_out + split_times * in_stride * split_size * data_size;
    (void)memcpy(dst, src, split_size * in_stride_bytes);
    src += split_size * in_stride * data_size;
  }

  return NNACL_OK;
}
