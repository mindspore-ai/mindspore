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
#include <string>
#include <vector>
#include <iostream>
#include "minddata/dataset/core/tensor_helpers.h"

namespace mindspore {
namespace dataset {

void IndexGeneratorHelper(int8_t depth, std::vector<dsize_t> *numbers,
                          const std::vector<mindspore::dataset::SliceOption> &slice_list,
                          std::vector<std::vector<dsize_t>> *matrix) {
  // for loop changes if its an index instead of a slice object
  if (depth > 0) {
    dsize_t new_depth = depth - 1;
    dsize_t curr_ind = numbers->size() - depth;

    if (slice_list[curr_ind].slice_.valid()) {
      dsize_t increment = slice_list[curr_ind].slice_.step_;

      if (increment > 0) {
        for (int i = slice_list[curr_ind].slice_.start_; i < slice_list[curr_ind].slice_.stop_;
             i = i + slice_list[curr_ind].slice_.step_) {
          (*numbers)[curr_ind] = i;
          IndexGeneratorHelper(new_depth, numbers, slice_list, matrix);
        }
      } else {
        for (int i = slice_list[curr_ind].slice_.start_; i > slice_list[curr_ind].slice_.stop_;
             i = i + slice_list[curr_ind].slice_.step_) {
          (*numbers)[curr_ind] = i;
          IndexGeneratorHelper(new_depth, numbers, slice_list, matrix);
        }
      }
    } else {
      for (int i = 0; i < slice_list[curr_ind].indices_.size(); i++) {
        (*numbers)[curr_ind] = slice_list[curr_ind].indices_[i];
        IndexGeneratorHelper(new_depth, numbers, slice_list, matrix);
      }
    }

  } else {
    (*matrix).emplace_back((*numbers));
  }
}

// Used to generate slice indices
std::vector<std::vector<dsize_t>> IndexGenerator(const std::vector<mindspore::dataset::SliceOption> &slice_list) {
  int8_t depth = slice_list.size();
  std::vector<dsize_t> numbers(depth, 0);
  std::vector<std::vector<dsize_t>> matrix(0, std::vector<dsize_t>(depth, 0));

  IndexGeneratorHelper(depth, &numbers, slice_list, &matrix);

  return matrix;
}
}  // namespace dataset
}  // namespace mindspore
