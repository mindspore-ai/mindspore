/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_ITERATOR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_ITERATOR_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "include/api/dual_abi_helper.h"
#include "include/api/status.h"
#include "include/api/types.h"

namespace mindspore {
namespace dataset {
// Forward declare
class ExecutionTree;
class DatasetOp;
class Tensor;

class NativeRuntimeContext;
class IteratorConsumer;
class PullBasedIteratorConsumer;

class Dataset;

using MSTensorMap = std::unordered_map<std::string, mindspore::MSTensor>;
using MSTensorMapChar = std::map<std::vector<char>, mindspore::MSTensor>;
using MSTensorVec = std::vector<mindspore::MSTensor>;

// Abstract class for iterating over the dataset.
class DATASET_API Iterator {
 public:
  /// \brief Constructor.
  Iterator();

  /// \brief Destructor.
  virtual ~Iterator();

  /// \brief Method for building and launching the pipeline.
  /// \param[in] ds The last DatasetOp in the dataset pipeline.
  /// \param[in] num_epochs Number of epochs passed down to EpochCtrlNode (default=-1, which means infinite epochs).
  /// \return Status error code, returns OK if no error encountered.
  virtual Status BuildAndLaunchTree(const std::shared_ptr<Dataset> &ds, int32_t num_epochs);

  /// \brief Function to get the next row from the data pipeline.
  /// \note Type of return data is a unordered_map(with column name).
  /// \param[out] row The output tensor row.
  /// \return Status error code, returns OK if no error encountered.
  /// \par Example
  /// \code
  ///      /* dataset is an instance of Dataset object */
  ///      std::shared_ptr<Iterator> = dataset->CreateIterator();
  ///      std::unordered_map<std::string, mindspore::MSTensor> row;
  ///      iter->GetNextRow(&row);
  /// \endcode
  Status GetNextRow(MSTensorMap *row) {
    if (row == nullptr) {
      return Status(kMDUnexpectedError, "Got nullptr when GetNext row.");
    }
    MSTensorMapChar row_;
    row_.clear();
    row->clear();
    Status s = GetNextRowCharIF(&row_);
    TensorMapCharToString(&row_, row);
    return s;
  }

  /// \brief Char interface(CharIF) of GetNextRow.
  /// \note The reason for using this API is that std::string will be constrained by the
  ///    compiler option '_GLIBCXX_USE_CXX11_ABI' while char is free of this restriction.
  Status GetNextRowCharIF(MSTensorMapChar *row);

  /// \brief Function to get the next row from the data pipeline.
  /// \note Type of return data is a vector(without column name).
  /// \param[out] row The output tensor row.
  /// \return Status error code, returns OK if no error encountered.
  /// \par Example
  /// \code
  ///      /* dataset is an instance of Dataset object */
  ///      std::shared_ptr<Iterator> = dataset->CreateIterator();
  ///      std::vector<mindspore::MSTensor> row;
  ///      iter->GetNextRow(&row);
  /// \endcode
  virtual Status GetNextRow(MSTensorVec *row);

  /// \brief Function to shut down the data pipeline.
  void Stop();

  /// \brief Inter class as iterator of Iterator.
  class _Iterator {
   public:
    /// \brief Constructor
    explicit _Iterator(Iterator *lt);

    /// \brief Destructor
    ~_Iterator() {
      if (cur_row_ != nullptr) {
        delete cur_row_;
        cur_row_ = nullptr;
      }
    }

    /// \brief prefix ++ overload
    _Iterator &operator++();

    /// \brief dereference operator
    MSTensorMap &operator*() { return *cur_row_; }

    /// \brief dereference operator
    MSTensorMap *operator->() { return cur_row_; }

    /// \brief bool operator
    bool operator!=(const _Iterator &rhs) { return cur_row_ != rhs.cur_row_; }

   private:
    int ind_;  // the cur node our Iterator points to
    Iterator *lt_;
    MSTensorMap *cur_row_;
  };

  /// \brief Function to return the iterator points to the begin of Iterator.
  _Iterator begin() { return _Iterator(this); }

  /// \brief Function to return the iterator points to the end of Iterator.
  _Iterator end() { return _Iterator(nullptr); }

 private:
  std::unique_ptr<NativeRuntimeContext> runtime_context_;
  IteratorConsumer *consumer_;
};

class DATASET_API PullIterator : public Iterator {
 public:
  /// \brief Constructor.
  PullIterator();

  /// \brief Destructor.
  ~PullIterator() override;

  /// \brief Function to get next row from the data pipeline.
  /// \note Type of return data is a vector(without column name).
  /// \param[out] row The output tensor row.
  /// \return Status error code, returns OK if no error encountered else false.
  /// \par Example
  /// \code
  ///      /* dataset is an instance of Dataset object */
  ///      std::shared_ptr<Iterator> = dataset->CreatePullBasedIterator();
  ///      std::vector<mindspore::MSTensor> row;
  ///      iter->GetNextRow(&row);
  /// \endcode
  Status GetNextRow(MSTensorVec *const row) override;

  /// \brief Function to get specified rows from the data pipeline.
  /// \note Type of return data is a vector(without column name). This behavior is subject to change.
  /// \param[in] num_rows The number of rows to fetch.
  /// \param[out] row The output tensor row.
  /// \return Status error code, returns OK if no error encountered else false.
  /// \par Example
  /// \code
  ///      /* dataset is an instance of Dataset object */
  ///      std::shared_ptr<Iterator> = dataset->CreatePullBasedIterator();
  ///      std::vector<std::vector<mindspore::MSTensor>> rows;
  ///      iter->GetNextRow(5, &rows);
  /// \endcode
  Status GetRows(int32_t num_rows, std::vector<MSTensorVec> *const row);

  /// \brief Method for building and launching the pipeline.
  /// \note Consider making this function protected.
  /// \param[in] ds The root node that calls the function.
  /// \param[in] num_epochs Number of epochs passed down to EpochCtrlNode (default=-1, which means infinite epochs).
  /// \return Status error code, returns OK if no error encountered.
  Status BuildAndLaunchTree(const std::shared_ptr<Dataset> &ds, int32_t num_epochs) override;

 private:
  std::unique_ptr<PullBasedIteratorConsumer> pull_consumer_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_ITERATOR_H_
