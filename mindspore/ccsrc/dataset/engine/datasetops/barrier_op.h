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
#ifndef DATASET_ENGINE_DATASETOPS_BARRIER_OP_H_
#define DATASET_ENGINE_DATASETOPS_BARRIER_OP_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "dataset/core/tensor.h"
#include "dataset/engine/dataset_iterator.h"
#include "dataset/engine/datasetops/pipeline_op.h"
#include "dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {
// Forward declare
class DataBuffer;
class ExecutionTree;

// BarrierOp class implements the Barrier operator. It will block sending of rows until a signal has
// been received. This signal is given from python layer. The current barrier design respects the
// rows per buffer design and will only output a buffer with rows once it has received rows per buffer
// signals from python.

class BarrierOp : public PipelineOp {
 public:
  //  The nested builder class inside of the BarrierOp is used to help manage all of
  //  the arguments for constructing it.  Use the builder by setting each argument
  //  with the provided set methods, and then finally call the build method to execute
  //  the actual construction.

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
    // @param int32_t op_connector_size
    // @return Builder setter method returns reference to the builder.
    Builder &SetOpConnectorSize(int32_t op_connector_size) {
      builder_op_connector_size_ = op_connector_size;
      return *this;
    }

    // Setter method.
    // @param const std::string & condition_name
    // @return Builder setter method returns reference to the builder.
    Builder &SetConditionName(const std::string &condition_name) {
      builder_condition_name_ = condition_name;
      return *this;
    }

    // Setter method.
    // @param py::function condition_func - blocking condition function
    // @return Builder setter method returns reference to the builder.
    Builder &SetConditionFunc(py::function condition_func) {
      builder_condition_func_ = condition_func;
      return *this;
    }

    // The builder "build" method creates the BarrierOp dataset Operator.
    // @return shared_ptr to the new BarrierOp object
    Status Build(std::shared_ptr<BarrierOp> *);

   private:
    int32_t builder_rows_per_buffer_;
    int32_t builder_op_connector_size_;
    std::string builder_condition_name_;
    py::function builder_condition_func_;

    Status SanityCheck() const;
  };

  // Constructor for BarrierOp
  // @param rows_per_buffer - number of rows in output buffer
  // @param op_connector_size - connector size
  // @param condition_name - the condition name associated with this operator
  // @param condition_func - the blocking condition check per row
  // @note - currently rows_per_buffer should = 1 for barrier.
  // The reason for this is having other values would complicate how the pipeline behaves with other operators
  // One example of such case is having batch after barrier. Batch would be waiting for data and having
  // rows per buffer in this case can result in hanging
  BarrierOp(int32_t rows_per_buffer, int32_t op_connector_size, const std::string &condition_name,
            py::function condition_func);

  // Destructor
  ~BarrierOp();

  Status EofReceived(int32_t) override;

  Status EoeReceived(int32_t) override;

  // Print function for Barrier
  // @param out - output stream to print to
  // @param show_all - if it should print everything
  void Print(std::ostream &out, bool show_all) const override;

  // Provide stream operator for displaying it
  friend std::ostream &operator<<(std::ostream &out, const BarrierOp &bo) {
    bo.Print(out, false);
    return out;
  }

  // Class functor operator () override.
  // All dataset ops operate by launching a thread (see ExecutionTree). This class functor will
  // provide the master loop that drives the logic for performing the work
  // @return Status - The error code return
  Status operator()() override;

  // Handles preprocessing of the main loop, used when starting new epoch
  // @param table - a table of tensors to be moved into a buffer
  Status prepare(TensorQTable *const table);

  // This function calls takes a table repeatedly adds rows to it.
  // @param table - a table of tensors to be moved into a buffer
  Status fillBuffer(TensorQTable *const table);

  // Gets next tensor row and sets control signals
  Status getNextTensorRow(TensorRow *new_row);

  // This function runs the wait function on condition
  Status blockCond();

 private:
  // clean up variable to return imcomplete buffer
  bool clean_up_;
  // end of file state, we stop reading data and shut down
  bool eof_;
  // rows per buffer
  int32_t rows_per_buffer_;
  // buffer_id
  int32_t buffer_id_;
  // local variable to keep track of the buffer information
  std::unordered_map<std::string, int32_t> col_name_id_map_;
  // iterator to pull new rows, we only have one child
  std::unique_ptr<ChildIterator> child_iterator_;
  // condition name, to support multiple barriers
  std::string condition_name_;
  // Function pointer of blocking function
  py::function condition_function_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATASETOPS_BARRIER_OP_H_
