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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_ITERATOR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_ITERATOR_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "minddata/dataset/include/status.h"

namespace mindspore {
namespace dataset {

// Forward declare
class ExecutionTree;
class DatasetIterator;
class DatasetOp;
class Tensor;

class NativeRuntimeContext;
class IteratorConsumer;

class Dataset;

using TensorMap = std::unordered_map<std::string, std::shared_ptr<Tensor>>;
using TensorVec = std::vector<std::shared_ptr<Tensor>>;

// Abstract class for iterating over the dataset.
class Iterator {
 public:
  /// \brief Constructor
  Iterator();

  /// \brief Destructor
  ~Iterator();

  /// \brief Method for building and launching the pipeline.
  /// \param[in] ops - a vector of DatasetOp in the data pipeline.
  /// \param[in] num_epochs Number of epochs passed down to EpochCtrlNode, default -1, infinite epochs
  /// \return - a Status error code, returns OK if no error encountered.
  Status BuildAndLaunchTree(std::shared_ptr<Dataset> ds, int32_t num_epochs);

  /// \brief Function to get the next row from the data pipeline.
  /// \note Type of return data is a map(with column name).
  /// \param[out] row - the output tensor row.
  /// \return Returns true if no error encountered else false.
  bool GetNextRow(TensorMap *row);

  /// \brief Function to get the next row from the data pipeline.
  /// \note Type of return data is a vector(without column name).
  /// \param[out] row - the output tensor row.
  /// \return Returns true if no error encountered else false.
  bool GetNextRow(TensorVec *row);

  /// \brief Function to shut down the data pipeline.
  void Stop();

  class _Iterator {
   public:
    explicit _Iterator(Iterator *lt) : lt_{lt}, cur_row_{nullptr} {
      if (lt_) {
        cur_row_ = new TensorMap();
        lt_->GetNextRow(cur_row_);
      }
    }

    // Destructor
    ~_Iterator() {
      if (cur_row_) {
        delete cur_row_;
      }
    }

    _Iterator &operator++() {
      if (lt_) {
        ++ind_;
        lt_->GetNextRow(cur_row_);
      }
      if (cur_row_ && cur_row_->size() == 0) {
        delete cur_row_;
        cur_row_ = nullptr;
      }
      return *this;
    }                                             // prefix ++ overload
    TensorMap &operator*() { return *cur_row_; }  // dereference operator
    TensorMap *operator->() { return cur_row_; }

    bool operator!=(const _Iterator &rhs) { return cur_row_ != rhs.cur_row_; }

   private:
    int ind_;  // the cur node our Iterator points to
    Iterator *lt_;
    TensorMap *cur_row_;
  };

  _Iterator begin() { return _Iterator(this); }

  _Iterator end() { return _Iterator(nullptr); }

 private:
  std::unique_ptr<NativeRuntimeContext> runtime_context_;
  IteratorConsumer *consumer_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_ITERATOR_H_
