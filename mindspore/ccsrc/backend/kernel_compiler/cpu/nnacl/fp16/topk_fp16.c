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
#include "nnacl/fp16/topk_fp16.h"

int TopkFp16DescendCmp(const void *a, const void *b) {
  float16_t sub = ((const TopkFp16Node *)b)->element - ((const TopkFp16Node *)a)->element;
  if (sub > 0) {
    return 1;
  } else if (sub < 0) {
    return -1;
  }
  if (((const TopkFp16Node *)a)->index > ((const TopkFp16Node *)b)->index) {
    return 1;
  } else {
    return -1;
  }
}

int TopkFp16IndexSortCmp(const void *a, const void *b) {
  if (((const TopkFp16Node *)a)->index > ((const TopkFp16Node *)b)->index) {
    return 1;
  } else {
    return -1;
  }
}

void TopkFp16(void *input_data, void *output_data, int32_t *output_index, TopkParameter *parameter) {
  int last_dim_size = parameter->last_dim_size_;
  int loop_num = parameter->loop_num_;
  int k = parameter->k_;
  TopkFp16Node *top_map = (TopkFp16Node *)parameter->topk_node_list_;

  float16_t *cur_input_data = (float16_t *)input_data;
  float16_t *cur_output_data = (float16_t *)output_data;
  int32_t *cur_output_index = output_index;
  for (int i = 0; i < loop_num; i++) {
    for (int j = 0; j < last_dim_size; j++) {
      top_map[j].element = *(cur_input_data + j);
      top_map[j].index = j;
    }
    qsort(top_map, last_dim_size, sizeof(top_map[0]), TopkFp16DescendCmp);
    if (!parameter->sorted_) {
      qsort(top_map, k, sizeof(top_map[0]), TopkFp16IndexSortCmp);
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
