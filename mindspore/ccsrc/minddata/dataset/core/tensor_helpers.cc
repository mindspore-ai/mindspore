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
#include "minddata/dataset/core/tensor_helpers.h"
#include <memory>

#include "include/dataset/constants.h"
#include "include/dataset/transforms.h"
#include "minddata/dataset/util/log_adapter.h"
namespace mindspore {
namespace dataset {

void IndexGeneratorHelper(int8_t depth, std::vector<dsize_t> *numbers,
                          const std::vector<mindspore::dataset::SliceOption> &slice_list,
                          std::vector<std::vector<dsize_t>> *matrix) {
  if (numbers == nullptr || matrix == nullptr) {
    MS_LOG(ERROR) << "Invalid input pointer, can't be NULL";
    return;
  }
  // for loop changes if its an index instead of a slice object
  if (depth > 0) {
    int8_t new_depth = depth - 1;
    // depth is always less than or equal to numbers->size() (based on the caller functions)
    size_t curr_ind = static_cast<size_t>(numbers->size() - static_cast<size_t>(depth));
    if (curr_ind >= slice_list.size()) {
      MS_LOG(ERROR) << "The index is out of range in slice_list.";
      return;
    }

    if (slice_list[curr_ind].slice_.valid()) {
      dsize_t increment = slice_list[curr_ind].slice_.step_;

      if (increment > 0) {
        for (dsize_t i = slice_list[curr_ind].slice_.start_; i < slice_list[curr_ind].slice_.stop_;
             i = i + slice_list[curr_ind].slice_.step_) {
          (*numbers)[curr_ind] = i;
          IndexGeneratorHelper(new_depth, numbers, slice_list, matrix);
        }
      } else {
        for (dsize_t j = slice_list[curr_ind].slice_.start_; j > slice_list[curr_ind].slice_.stop_;
             j = j + slice_list[curr_ind].slice_.step_) {
          (*numbers)[curr_ind] = j;
          IndexGeneratorHelper(new_depth, numbers, slice_list, matrix);
        }
      }
    } else {
      for (size_t k = 0; k < slice_list[curr_ind].indices_.size(); k++) {
        (*numbers)[curr_ind] = slice_list[curr_ind].indices_[k];
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
