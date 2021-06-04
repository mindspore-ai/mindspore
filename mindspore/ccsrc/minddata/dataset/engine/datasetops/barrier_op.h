/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_BARRIER_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_BARRIER_OP_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/engine/datasetops/pipeline_op.h"
#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {
// Forward declare
class ExecutionTree;

// BarrierOp class implements the Barrier operator. It will block sending of rows until a signal has
// been received. This signal is given from python layer.

class BarrierOp : public PipelineOp {
 public:
  // Constructor for BarrierOp
  // @param op_connector_size - connector size
  // @param condition_name - the condition name associated with this operator
  // @param condition_func - the blocking condition check per row
  // The reason for this is having other values would complicate how the pipeline behaves with other operators
  // One example of such case is having batch after barrier.
  BarrierOp(int32_t op_connector_size, const std::string &condition_name, py::function condition_func);

  /// Destructor
  ~BarrierOp();

  Status EofReceived(int32_t) override;

  Status EoeReceived(int32_t) override;

  /// Print function for Barrier
  /// @param out - output stream to print to
  /// @param show_all - if it should print everything
  void Print(std::ostream &out, bool show_all) const override;

  /// Op name getter
  /// @return Name of the current Op
  std::string Name() const override { return kBarrierOp; }

  /// Provide stream operator for displaying it
  friend std::ostream &operator<<(std::ostream &out, const BarrierOp &bo) {
    bo.Print(out, false);
    return out;
  }

  /// Class functor operator () override.
  /// All dataset ops operate by launching a thread (see ExecutionTree). This class functor will
  /// provide the master loop that drives the logic for performing the work
  /// @return Status The status code returned
  Status operator()() override;

  // Handles preprocessing of the main loop, used when starting new epoch
  // @param table - a table of tensors to be moved into a row
  Status prepare();

  // Gets next tensor row and sets control signals
  Status getNextTensorRow(TensorRow *new_row);

  /// This function runs the wait function on condition
  Status blockCond();

 private:
  // clean up variable
  bool clean_up_;
  // end of file state, we stop reading data and shut down
  bool eof_;
  // iterator to pull new rows, we only have one child
  std::unique_ptr<ChildIterator> child_iterator_;
  // condition name, to support multiple barriers
  std::string condition_name_;
  // Function pointer of blocking function
  py::function condition_function_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_BARRIER_OP_H_
