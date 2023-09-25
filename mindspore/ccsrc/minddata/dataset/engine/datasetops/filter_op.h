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
#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/queue.h"

namespace mindspore {
namespace dataset {

enum filterCtrl : int8_t { kFilterEmpty = 0, kFilterPartial = 1, kFilterFull = 2, kFilterEoe = 3, kFilterEof = 4 };

class FilterOp : public ParallelOp<TensorRow, TensorRow> {
 public:
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

  /// \brief Gets the next row
  /// \param row[out] - Fetched TensorRow
  /// \return Status The status code returned
  Status GetNextRowPullMode(TensorRow *const row) override;

 protected:
  /// \brief Gets the implementation status for operator in pull mode
  /// \return implementation status
  ImplementedPullMode PullModeImplementationStatus() const override { return ImplementedPullMode::Implemented; }

 private:
  // predicate_func python callable which returns a boolean value.
  std::shared_ptr<TensorOp> predicate_func_;

  // Variable to store the column name that will feed to predicate function.
  std::vector<std::string> in_columns_;

  std::unique_ptr<ChildIterator> child_iterator_;

  // Private function for worker/thread to loop continuously. It comprises the main
  // logic of FilterOp, getting the data from previous Op, validating user specified column names,
  // applying predicate to each of the data, filter the data when predicate result is false.
  // @param worker_id The id assigned to this thread/worker upon creation.
  // @return Status The status code returned
  Status WorkerEntry(int32_t worker_id) override;  //  In: workerId assigned by tree_

  // Filter the data by  predicate function .
  // @param in_row input row.
  // @param out_predicate result boolean to filter or not.
  // @return Status The status code returned
  Status WorkerCompute(const TensorRow &in_row, bool *out_predicate);

  // @param input tensor vector.
  // @return Status The status code returned.
  Status CheckInput(const TensorRow &input) const;

  // Invoke python func.
  // @param input tensor vector.
  // @param the result of predicate.
  // @return Status The status code returned.
  Status InvokePredicateFunc(const TensorRow &input, bool *out_predicate);

  // Private function for validating if each of the user specified input column names
  // exist in column_name_id_map_.
  // @param input_columns The vector of input column names used in the current thread.
  // @return Status The status code returned
  Status ValidateInColumns(const std::vector<std::string> &input_columns) const;
};

}  // namespace dataset
}  // namespace mindspore
#endif
