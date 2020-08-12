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

#include "nnacl/int8/topk_int8.h"

int DescendCmpInt8(const void *a, const void *b) {
  return ((const TopkNodeInt8 *)b)->element - ((const TopkNodeInt8 *)a)->element;
}

int AscendCmpInt8(const void *a, const void *b) {
  return ((const TopkNodeInt8 *)a)->element - ((const TopkNodeInt8 *)b)->element;
}

void TopkInt8(int8_t *input_data, int8_t *output_data, int32_t *output_index, TopkParameter *parameter) {
  int last_dim_size = parameter->last_dim_size_;
  int loop_num = parameter->loop_num_;
  int k = parameter->k_;
  TopkNodeInt8 *top_map = (TopkNodeInt8 *)parameter->topk_node_list_;

  int8_t *cur_input_data = input_data;
  int8_t *cur_output_data = output_data;
  int32_t *cur_output_index = output_index;
  for (int i = 0; i < loop_num; i++) {
    for (int j = 0; j < last_dim_size; j++) {
      top_map[j].element = *(cur_input_data + j);
      top_map[j].index = j;
    }
    if (parameter->sorted_) {
      qsort(top_map, last_dim_size, sizeof(top_map[0]), DescendCmpInt8);
    } else {
      qsort(top_map, last_dim_size, sizeof(top_map[0]), AscendCmpInt8);
    }
    for (int m = 0; m < k; m++) {
      cur_output_data[m] = top_map[m].element;
      cur_output_index[m] = top_map[m].index;
    }
    cur_input_data += last_dim_size;
    cur_output_data += k;
    cur_output_index += k;
  }
}
