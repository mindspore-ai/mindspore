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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_RENAME_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_RENAME_OP_H_

#include <memory>
#include <queue>
#include <string>
#include <vector>
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/engine/datasetops/pipeline_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class RenameOp : public PipelineOp {
 public:
  // Constructor for RenameOp
  // @param in_col_names names of columns to rename
  // @param out_col_names names of columns after rename
  // @param op_connector_size connector size
  RenameOp(const std::vector<std::string> &in_col_names, const std::vector<std::string> &out_col_names);

  // Destructor
  ~RenameOp();

  // Print function for Rename
  // @param out output stream to print to
  // @param show_all if it should print everything
  void Print(std::ostream &out, bool show_all) const override;

  // Provide stream operator for displaying it
  friend std::ostream &operator<<(std::ostream &out, const RenameOp &ro) {
    ro.Print(out, false);
    return out;
  }

  // Class functor operator () override.
  // All dataset ops operate by launching a thread (see ExecutionTree). This class functor will
  // provide the master loop that drives the logic for performing the work
  // @return Status The status code returned
  Status operator()() override;

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return kRenameOp; }

  // Gets a row from the child node and projects that row. The caller is typically our parent node.
  // @param row - output pointer to the projected row.
  // @param worker_id - The worker id
  Status GetNextRow(TensorRow *row, int32_t worker_id, bool retry_if_eoe) override;
  int32_t NumConsumers() const override;
  int32_t NumProducers() const override;

 protected:
  // Rename core functionality
  // Computing the assignment of the new column name map.
  // @return - Status
  Status ComputeColMap() override;

  // Variable to store the input column names
  std::vector<std::string> in_columns_;

  // Variable to store the output column names
  std::vector<std::string> out_columns_;

  std::unique_ptr<ChildIterator> child_iterator_;  // An iterator for fetching.
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_RENAME_OP_H_
