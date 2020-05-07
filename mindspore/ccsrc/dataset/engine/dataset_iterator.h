/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef DATASET_ENGINE_DATASET_ITERATOR_H_
#define DATASET_ENGINE_DATASET_ITERATOR_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "dataset/util/status.h"
#include "dataset/core/tensor.h"
#include "dataset/engine/datasetops/dataset_op.h"
#include "dataset/engine/execution_tree.h"

namespace mindspore {
namespace dataset {
using TensorMap = std::unordered_map<std::string, std::shared_ptr<Tensor>>;

// forward declare
class ExecutionTree;

class DataBuffer;

// IteratorBase class is used to iterate data from an executionTree one row at a time.
// The base class provides the general interface, whereas derived classes provide slightly
// different implementations.
class IteratorBase {
 public:
  // Constructor of IteratorBase
  IteratorBase();

  // Destructor
  virtual ~IteratorBase();

  // Fetches one row of data from the iterator.
  // the base class version simply performs error handling and returns empty row. Actual
  // functionality exists in the derived versions of this function.
  // @param out_row - A TensorRow (vector of shared pointers to Tensors).  If any of the of data
  // messages are encountered (such as eoe or eof), then an empty TensorRow is returned back.
  // @return Status - The error code return
  // @note The position of a Tensor/column might be different from the initial column order
  // in the storageOp. User must be aware that MapOp, ZipOps, and others might change
  // the column ordering.
  virtual Status FetchNextTensorRow(TensorRow *out_row);

  // Fetches one row of data from the iterator as a column map.
  // @return A unordered map from column name to shared pointer to Tensor.
  Status GetNextAsMap(TensorMap *out_map);

  // Getter
  // @return T/F if this iterator is completely done after getting an eof
  bool eof_handled() const { return eof_handled_; }

  // Getter
  // @return The string to column id mapping.
  virtual std::unordered_map<std::string, int32_t> GetColumnNameMap() const = 0;

 protected:
  std::unique_ptr<DataBuffer> curr_buffer_;  // holds the current buffer
  bool eof_handled_;                         // T/F if this op got an eof
  bool first_row_;                           // internal tracking for first row case
  std::unordered_map<std::string, int32_t> col_name_id_map_;
};

// The DatasetIterator derived class is for fetching rows off the end/root of the execution tree.
class DatasetIterator : public IteratorBase {
 public:
  // Constructor of the DatasetIterator
  // @param exe_tree The execution tree we want to pull/iterate the data from using it's root node.
  explicit DatasetIterator(std::shared_ptr<ExecutionTree> exe_tree);

  // Destructor
  ~DatasetIterator();

  // Fetches one row of data from the iterator.  Overrides the base class.  This one fetches
  // from the tree root node directly.
  // @param out_row - A TensorRow (vector of shared pointers to Tensors).  If any of the of data
  // messages are encountered (such as eoe or eof), then an empty TensorRow is returned back.
  // @return Status - The error code return
  Status FetchNextTensorRow(TensorRow *out_row) override;

  // Fetches the next tensor row into device row, and returns it's shape.
  // @param out_shapes - A vector of tensor shapes (one shape per column)
  // @return Status - The error code return
  Status GetOutputShapes(std::vector<TensorShape> *out_shapes);

  // Fetches the next tensor row into device row, and returns it's shape.
  // @param outShapes - A vector of tensor shapes (one shape per column)
  // @return Status - The error code return
  Status GetOutputTypes(std::vector<DataType> *out_types);

  // Getter
  // @return The string to column id mapping.
  std::unordered_map<std::string, int32_t> GetColumnNameMap() const override;

 private:
  std::shared_ptr<DatasetOp> root_;  // saves the root of the executionTree
  TensorRow device_queue_row_;
};

// The ChildIterator derived class is for fetching rows from intermediate nodes of execution tree.
// This one should only be used by internal Dataset operators, rather than an end-user.
class ChildIterator : public IteratorBase {
 public:
  // Constructor of the DatasetIterator
  // @param current_op - The parent op from which we'll fetch from it's children.
  // @param worker_id - The worker id to use when fetching from the children.
  // @param child_idx - The index to the child to fetch from.
  ChildIterator(DatasetOp *current_op, int32_t worker_id, int32_t child_idx);

  // Destructor
  ~ChildIterator();

  // Fetches one row of data from the iterator.  Overrides the base class.  This one fetches
  // only from the child/worker id as given from the constructor.
  // @param out_row - A TensorRow (vector of shared pointers to Tensors).  If any of the of data
  // messages are encountered (such as eoe or eof), then an empty TensorRow is returned back.
  // @return Status - The error code return
  Status FetchNextTensorRow(TensorRow *out_row) override;

  // This function drains buffer until next eoe has been received.
  // It will be a no-op if the previous row returned is empty.
  // @return Status - The error code return
  Status Drain();

  // Getter
  // @return The string to column id mapping.
  std::unordered_map<std::string, int32_t> GetColumnNameMap() const override;

 private:
  DatasetOp *current_op_;  // The parent operator. We consume from it's children.
  int32_t child_idx_;      // The specific child this iterator will fetch from.
  int32_t worker_id_;      // The worker id uses for fetching the child data.
  bool end_epoch_;         // the flag used when an empty row has been returned.
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATASET_ITERATOR_H_
