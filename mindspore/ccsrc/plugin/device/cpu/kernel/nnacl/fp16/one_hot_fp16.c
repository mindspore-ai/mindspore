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

#include "nnacl/fp16/one_hot_fp16.h"
#include "nnacl/errorcode.h"
int OneHotToFp16(const int *indices, float16_t on_value, float16_t off_value, float16_t *output,
                 const OneHotStruct *one_hot_param, const int tid, const int thread_num) {
  if (indices == NULL || one_hot_param == NULL || output == NULL) {
    return NNACL_NULL_PTR;
  }
  if (thread_num == 0) {
    return NNACL_PARAM_INVALID;
  }

  int outer_size = one_hot_param->outer_size_;
  int inner_size = one_hot_param->inner_size_;
  int depth = one_hot_param->depth_;
  int i, j, k;
  for (i = tid; i < outer_size; i += thread_num) {
    float16_t *output_ptr = output + i * depth * inner_size;
    for (k = 0; k < depth; k++) {  // output layout: outer_size * depth * inner_size
      const int *indices_ptr = indices + i * inner_size;
      for (j = 0; j < inner_size; j++) {
        *output_ptr = off_value;
        int index = *(indices_ptr++);
        if (one_hot_param->support_neg_index_ && index < 0) {
          index += depth;
        }
        if (index == k) {
          *output_ptr = on_value;
        }
        output_ptr++;
      }
    }
  }
  return NNACL_OK;
}
