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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_PERF_DATA_H
#define MINDSPORE_CCSRC_MINDDATA_DATASET_PERF_DATA_H

#include <vector>
#include "minddata/dataset/include/constants.h"

namespace mindspore {
namespace dataset {

// PerfData is a convenience class to record and store the data produced by Monitor
// and represents a 2D column major table with every column storing samples
// for an operator. The number of rows equals to the number of samples,
// the number of columns equals to the number of operators.
// The capacity is determined on construction and cannot be changed.
// ColumnType can be std::vector or CyclicArray. In case of the latter data can be added
// indefinitely without the risk of overflowing otherwise the capacity must not be exceeded.
// Given PerfData pd(n_rows, n_cols) an element in the column i and row j can be accessed as
// pd[i][j]

template <typename ColumnType>
class PerfData {
 public:
  PerfData() = default;
  ~PerfData() = default;
  PerfData(dsize_t max_rows, dsize_t n_cols) : counter_(0), max_rows_(max_rows), n_cols_(n_cols) {
    for (auto i = 0; i < n_cols_; i++) {
      data_.push_back(ColumnType(max_rows_));
    }
  }
  PerfData(const PerfData &rhs) = default;
  PerfData(PerfData &&rhs) = default;

  // Adds a row of data
  // T must be any container working with range based loops
  template <typename T>
  void AddSample(const T &row) {
    auto i = 0;
    for (const auto &e : row) {
      data_[i++].push_back(e);
    }
    counter_++;
  }

  // Fetches a row of data by copy
  template <typename V = typename ColumnType::value_type>
  auto Row(dsize_t idx) {
    std::vector<V> row(n_cols_);
    for (auto i = 0; i < n_cols_; i++) {
      row[i] = data_[i][idx];
    }
    return row;
  }

  // returns a column of data
  ColumnType &operator[](size_t idx) { return data_[idx]; }

  const ColumnType &operator[](size_t idx) const { return data_[idx]; }

  dsize_t size() { return counter_ < max_rows_ ? counter_ : max_rows_; }

  dsize_t capacity() { return max_rows_; }

 private:
  std::vector<ColumnType> data_;
  dsize_t counter_;
  dsize_t max_rows_;
  int n_cols_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_PERF_DATA_H
