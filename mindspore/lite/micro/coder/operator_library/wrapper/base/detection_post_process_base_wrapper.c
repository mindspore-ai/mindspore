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

#include "wrapper/base/detection_post_process_base_wrapper.h"
#include <stdbool.h>

static inline void swap_index(int *arr, int lhs, int rhs) {
  int temp = arr[lhs];
  arr[lhs] = arr[rhs];
  arr[rhs] = temp;
}

static inline bool compare(int i, int j, const float *scores) {
  if (scores[i] == scores[j]) {
    return i < j;
  }
  return scores[i] > scores[j];
}

static void heapify(const float *scores, int *indexes, int n, int i) {
  while (i < n) {
    int cur = i;
    int l = 2 * i + 1;
    const int r = 2 * i + 2;
    if (r < n && compare(indexes[cur], indexes[r], scores)) {
      cur = r;
    }
    if (l < n && compare(indexes[cur], indexes[l], scores)) {
      cur = l;
    }
    if (cur != i) {
      swap_index(indexes, i, cur);
      i = cur;
    } else {
      break;
    }
  }
}

void PartialArgSort(const float *scores, int *indexes, int num_to_sort, int num_values) {
  // make heap
  int start_index = num_to_sort / 2 - 1;
  for (int i = start_index; i >= 0; i--) {
    heapify(scores, indexes, num_to_sort, i);
  }
  // compare the rest elements with heap top
  for (int i = num_to_sort; i < num_values; ++i) {
    if (!compare(indexes[0], indexes[i], scores)) {
      swap_index(indexes, i, 0);
      heapify(scores, indexes, num_to_sort, 0);
    }
  }
  // heap sort
  for (int cur_length = num_to_sort - 1; cur_length > 0; cur_length--) {
    swap_index(indexes, 0, cur_length);
    heapify(scores, indexes, cur_length, 0);
  }
}
