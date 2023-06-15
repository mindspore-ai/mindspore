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

#include "nnacl/fp32/topk_fp32.h"
#include "nnacl/errorcode.h"

int DescendCmp(const void *a, const void *b) {
  NNACL_CHECK_NULL_RETURN_ERR(a);
  NNACL_CHECK_NULL_RETURN_ERR(b);
  float sub = ((const TopkNode *)b)->element - ((const TopkNode *)a)->element;
  if (sub > 0) {
    return 1;
  } else if (sub < 0) {
    return -1;
  }
  if (((const TopkNode *)a)->index > ((const TopkNode *)b)->index) {
    return 1;
  } else {
    return -1;
  }
}

int IndexSortCmp(const void *a, const void *b) {
  if (((const TopkNode *)a)->index > ((const TopkNode *)b)->index) {
    return 1;
  } else {
    return -1;
  }
}

void Topk(void *input_data, void *output_data, int32_t *output_index, TopkParameter *parameter) {
  int dim_size = parameter->dim_size_;
  int outer_loop_num = parameter->outer_loop_num_;
  int inner_loop_num = parameter->inner_loop_num_;
  int k = parameter->k_;
  TopkNode *top_map = (TopkNode *)parameter->topk_node_list_;

  float *cur_input_data = (float *)input_data;
  float *cur_output_data = (float *)output_data;
  int32_t *cur_output_index = output_index;
  for (int i = 0; i < outer_loop_num; i++) {
    int in_offset = i * dim_size * inner_loop_num;
    int out_offset = i * k * inner_loop_num;
    for (int j = 0; j < inner_loop_num; j++) {
      for (int m = 0; m < dim_size; m++) {
        int offset = in_offset + m * inner_loop_num + j;
        top_map[m].element = *(cur_input_data + offset);
        top_map[m].index = m;
      }
      qsort(top_map, dim_size, sizeof(top_map[0]), DescendCmp);
      if (!parameter->sorted_) {
        qsort(top_map, k, sizeof(top_map[0]), IndexSortCmp);
      }
      for (int m = 0; m < k; m++) {
        int offset = out_offset + m * inner_loop_num + j;
        cur_output_data[offset] = top_map[m].element;
        cur_output_index[offset] = top_map[m].index;
      }
    }
  }
}

void TopkInt(void *input_data, void *output_data, int32_t *output_index, TopkParameter *parameter) {
  int dim_size = parameter->dim_size_;
  int outer_loop_num = parameter->outer_loop_num_;
  int inner_loop_num = parameter->inner_loop_num_;
  int k = parameter->k_;
  TopkNode *top_map = (TopkNode *)parameter->topk_node_list_;

  int32_t *cur_input_data = (int32_t *)input_data;
  int32_t *cur_output_data = (int32_t *)output_data;
  int32_t *cur_output_index = output_index;
  for (int i = 0; i < outer_loop_num; i++) {
    int in_offset = i * dim_size * inner_loop_num;
    int out_offset = i * k * inner_loop_num;
    for (int j = 0; j < inner_loop_num; j++) {
      for (int m = 0; m < dim_size; m++) {
        int offset = in_offset + m * inner_loop_num + j;
        top_map[m].element = (float)(*(cur_input_data + offset));
        top_map[m].index = m;
      }
      qsort(top_map, dim_size, sizeof(top_map[0]), DescendCmp);
      if (!parameter->sorted_) {
        qsort(top_map, k, sizeof(top_map[0]), IndexSortCmp);
      }
      for (int m = 0; m < k; m++) {
        int offset = out_offset + m * inner_loop_num + j;
        cur_output_data[offset] = (int)(top_map[m].element);
        cur_output_index[offset] = top_map[m].index;
      }
    }
  }
}
