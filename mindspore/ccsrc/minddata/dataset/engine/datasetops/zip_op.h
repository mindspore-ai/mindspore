/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

class ZipOp : public PipelineOp {
 public:
  // Constructor for ZipOp
  // @param op_connector_size - connector size
  ZipOp();

  // Destructor
  ~ZipOp();

  Status EofReceived(int32_t) override;

  Status EoeReceived(int32_t) override;

  // Print function for Zip
  // \param[in] out - output stream to print to
  // \param[in] show_all - if it should print everything
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

  /// \brief Gets the next row
  /// \param row[out] - Fetched TensorRow
  /// \return Status The status code returned
  Status GetNextRow(TensorRow *row) override;

  /// \brief In pull mode, gets the next row
  /// \param row[out] - Fetched TensorRow
  /// \return Status The status code returned
  Status GetNextRowPullMode(TensorRow *const row) override;

 protected:
  /// \brief Gets the implementation status for operator in pull mode
  /// \return Implementation status
  ImplementedPullMode PullModeImplementationStatus() const override { return ImplementedPullMode::Implemented; }

 private:
  /// \brief Drain eoe signals from all children connectors.
  /// \notes Handle special handle case where an empty row has been received from child iterator.
  ///     When this function is called and encounters eoe at child iterator,
  ///     we need to drain rows from other child iterators until we hit eoe from all other child iterators.
  /// \param[in] skip_child - identifier for child to be skipped
  /// \param[in] is_pull_mode - an indicator to identify if in pull mode or not
  Status drainPipeline(int32_t skip_child, bool is_pull_mode) const;

  // Merges 1 row from each childIterator together
  // \param[in] new_zip_row - input and output, will be a non-empty row if all rows from childConnectors are non-empty
  // \param[in] skip_child - input and output, identifier for child to be skipped
  // \param[in] is_pull_mode - an indicator to identify if in pull mode or not
  // @details merge rows from iterator together. This is the main functionality for ZipOp
  //          this function takes one row and fills it with tensors from rows fetched
  //          from childIterators.
  // @example:
  //   Zips multiple rows at a time, the output is store in newZipRow
  //       1    a     T
  //       \    |     /
  //         1, a, T
  Status getNextZippedRow(TensorRow *const new_zip_row, int32_t *skip_child, bool is_pull_mode) const;

  // Computing the assignment of the column name map.
  // @return - Status
  Status ComputeColMap() override;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_ZIP_OP_H_
