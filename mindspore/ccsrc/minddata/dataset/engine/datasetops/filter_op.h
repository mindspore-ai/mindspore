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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_FILTER_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_FILTER_OP_H_

#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/queue.h"

namespace mindspore {
namespace dataset {

class FilterOp : public ParallelOp {
 public:
  // The nested builder class inside of the FilterOp is used to help manage all of
  // the arguments for constructing it.  Use the builder by setting each argument
  // with the provided set methods, and then finally call the build method to execute
  // the actual construction.
  class Builder {
   public:
    // Builder constructor. Creates the builder object.
    // @note No default args.
    // @return This is a constructor.
    Builder();

    // Default destructor
    ~Builder() = default;

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetPredicateFunc(std::shared_ptr<TensorOp> func) {
      builder_predicate_func_ = std::move(func);
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetInColNames(const std::vector<std::string> &in_col_names) {
      build_in_col_names_ = in_col_names;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetNumWorkers(int32_t num_workers) {
      builder_num_workers_ = num_workers;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetOpConnectorSize(int32_t connector_size) {
      builder_op_connector_size_ = connector_size;
      return *this;
    }

    // The builder "build" method creates the final object.
    // @param ptr The shared_ptr to the new FilterOp object.
    // @return Status.
    Status Build(std::shared_ptr<FilterOp> *ptr);

   private:
    // Sanity check for builder class args.
    // @return Status The status code returned.
    Status SanityCheck();
    std::vector<std::string> build_in_col_names_;
    std::shared_ptr<TensorOp> builder_predicate_func_;
    int32_t builder_num_workers_;
    int32_t builder_op_connector_size_;
  };

  enum filterCtrl : int8_t { kFilterEmpty = 0, kFilterPartial = 1, kFilterFull = 2, kFilterEoe = 3, kFilterEof = 4 };

  // Constructor of FilterOp
  // @note The builder class should be used to call it.
  // @param in_col_names A list of input column names,when it is empty the predicate will be
  //     applied all columns in the dataset.
  // @param num_workers The number of worker threads.
  // @param op_connector_size The size of each queue in the connector.
  // @param predicate_func python callable which returns a boolean value.
  FilterOp(const std::vector<std::string> &in_col_names, int32_t num_workers, int32_t op_queue_size,
           std::shared_ptr<TensorOp> predicate_func);

  // Destructor
  ~FilterOp() = default;

  // Class functor operator () override.
  // All dataset ops operate by launching a thread (see ExecutionTree),This class functor will
  // provide the master loop that drives the logic for performing the work.
  // @return Status The status code returned
  Status operator()() override;

  // @param int32_t workerId.
  // @return Status The status code returned.
  Status EofReceived(int32_t) override;

  // @param int32_t workerId.
  // @return Status The status code returned.
  Status EoeReceived(int32_t) override;

  // A print method typically used for debugging.
  // @param out The output stream to write output to.
  // @param show_all A bool to control if you want to show all info or just a summary.
  void Print(std::ostream &out, bool show_all) const override;

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return kFilterOp; }

 private:
  // predicate_func python callable which returns a boolean value.
  std::shared_ptr<TensorOp> predicate_func_;

  // Variable to store the column name that will feed to predicate function.
  std::vector<std::string> in_columns_;

  // Internal queue for filter.
  QueueList<std::pair<std::unique_ptr<DataBuffer>, filterCtrl>> filter_queues_;

  // Private function for worker/thread to loop continuously. It comprises the main
  // logic of FilterOp, getting the data from previous Op, validating user specified column names,
  // applying predicate to each of the data, filter the data when predicate result is false.
  // @param worker_id The id assigned to this thread/worker upon creation.
  // @return Status The status code returned
  Status WorkerEntry(int32_t worker_id) override;  //  In: workerId assigned by tree_

  // Filter the data by  predicate function .
  // @param in_buffer input data buffer.
  // @param to_proess_indices Indices of columns to be processed.
  // @param out data buffer that are filtered by predicate.
  // @return Status The status code returned
  Status WorkerCompute(DataBuffer *in_buffer, std::unique_ptr<TensorQTable> *out);

  // Collector databuffer.
  // @return Status The status code returned
  Status Collector();

  // @param input tensor vector.
  // @return Status The status code returned.
  Status CheckInput(const TensorRow &input) const;

  // Invoke python func.
  // @param input tensor vector.
  // @param the result of predicate.
  // @return Status The status code returned.
  Status InvokePredicateFunc(const TensorRow &input, bool *out_predicate);

  // Private function for validating if each of the user specified input column names
  // exist in the DataBuffer.
  // @param input_columns The vector of input column names used in the current thread.
  // @return Status The status code returned
  Status ValidateInColumns(const std::vector<std::string> *input_columns);

  // Private function for checking the column legality
  // @param in_buf A raw pointer to the DataBuffer. A raw pointer is fine because this function does not manage memory
  //     and is not shared with other threads.
  // @param[out] to_process_indices Indices of columns that will feed to predicate.
  // @param input_columns The vector of input column names used in the current thread.
  Status CheckColumns(const DataBuffer *in_buf, const std::vector<std::string> *input_columns);
};

}  // namespace dataset
}  // namespace mindspore
#endif
