/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_ZIP_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_ZIP_OP_H_

#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/engine/datasetops/pipeline_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// forward declare
class DataBuffer;

class ZipOp : public PipelineOp {
 public:
  //  The nested builder class inside of the ZipOp is used to help manage all of
  //  the arguments for constructing it.  Use the builder by setting each argument
  //  with the provided set methods, and then finally call the build method to execute
  //  the actual construction.
  //  NOTE: the rows per buffer with initial value 0 means to default to the number of rows from the first child

  class Builder {
   public:
    // Builder constructor.  Creates the builder object.
    // @note No default args
    // @return This is a constructor.
    Builder();

    // Default destructor
    ~Builder() = default;

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetRowsPerBuffer(int32_t rows_per_buffer) {
      builder_rows_per_buffer_ = rows_per_buffer;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetOpConnectorSize(int32_t op_connector_size) {
      builder_op_connector_size_ = op_connector_size;
      return *this;
    }

    // The builder "build" method creates the ZipOp dataset Operator.
    // @return shared_ptr to the new ZipOp object
    Status Build(std::shared_ptr<ZipOp> *);

   private:
    int32_t builder_rows_per_buffer_;
    int32_t builder_op_connector_size_;

    Status SanityCheck() const;
  };

  // Constructor for ZipOp
  // @param rows_per_buffer - number of rows in output buffer
  // @param op_connector_size - connector size
  ZipOp(int32_t rows_per_buffer, int32_t op_connector_size);

  // Destructor
  ~ZipOp();

  Status EofReceived(int32_t) override;

  Status EoeReceived(int32_t) override;

  // Print function for Zip
  // @param out - output stream to print to
  // @param show_all - if it should print everything
  void Print(std::ostream &out, bool show_all) const override;

  // Provide stream operator for displaying it
  friend std::ostream &operator<<(std::ostream &out, const ZipOp &zo) {
    zo.Print(out, false);
    return out;
  }

  // Class functor operator () override.
  // All dataset ops operate by launching a thread (see ExecutionTree). This class functor will
  // provide the master loop that drives the logic for performing the work
  // @return Status The status code returned
  Status operator()() override;

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return kZipOp; }

 private:
  // Handles preprocessing of the main loop, used when starting new epoch
  Status prepare(TensorQTable *const table);

  // This function calls takes a table repeatedly adds rows to it.
  // @param table a table of tensors to be moved into a buffer
  Status fillBuffer(TensorQTable *const table);

  // Special handle case where an empty row has been received from child iterator
  // @note - we need to drain eoe signals from all children connectors.
  // @details - when this function is called, then we encountered eoe at child iterator
  // we have to drain rows from other child iterators until we hit eoe from all other child iterators
  Status drainPipeline();

  // Merges 1 row from each childIterator together
  // @param new_zip_row - input and output, will be a non-empty row if all rows from childConnectors are non-empty
  // @param updateColumnMapping - generates a new column name to index mapping (mColNameIdMap) if set to true
  // @details merge rows from iterator together. This is the main functionality for ZipOp
  //          this function takes one row and fills it with tensors from rows fetched
  //          from childIterators.
  // @example:
  //   Zips multiple rows at a time, the output is store in newZipRow
  //       1    a     T
  //       \    |     /
  //         1, a, T
  Status getNextTensorRow(TensorRow *const new_zip_row);

  // Computing the assignment of the column name map.
  // @return - Status
  Status ComputeColMap() override;

  int32_t children_num_;
  int32_t rows_per_buffer_;
  int32_t buffer_id_;
  bool draining_;
  bool eof_;
  std::vector<std::unique_ptr<ChildIterator>> child_iterators_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_ZIP_OP_H_
