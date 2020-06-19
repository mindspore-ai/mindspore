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
#ifndef DATASET_ENGINE_DATASETOPS_MAP_OP_H_
#define DATASET_ENGINE_DATASETOPS_MAP_OP_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "dataset/engine/datasetops/parallel_op.h"
#include "dataset/kernels/tensor_op.h"
#include "dataset/util/queue.h"

namespace mindspore {
namespace dataset {
// Forward declare
class DataBuffer;
class ExecutionTree;

// MapOp class implements the Map operator. It will apply a list of operations to each record specified by column names.
// The column order behavior after MapOp is as follows.
// [Case 1] If the number of Input Columns == the number of Output Column, column ordering after MapOp
// is the same as the original column order where the Remainder Columns stay in the same position,
// and the Output Columns are placed the same position of the Input Columns.
// For example, initially if the dataset has column order |A, B, C, D, E|,
// and we apply MapOp() with Input Columns {B, C} and Output Columns {X, Y}.
// The column order after applying MapOp will be |A, X, Y, D, E|.
// Note that in this case, |X, Y| is the Output Columns and |A, D, E| which is the Remainder Columns stay in
// their original position, and column B is replaced by column X and column C is replace by column Y.
// [Case 2] If the number of Input Columns != the number of Output Column, column ordering after MapOp
// is Output Columns followed by Remainder Columns.
// For example, initially if the dataset has column order |A, B, C, D, E|,
// and we apply MapOp() with Input Columns {B, C, A} and Output Columns {X, Y}.
// The column order after applying MapOp will be |X, Y, D, E|.
// Note that in this case, |X, Y| is the Output Columns and |D, E| is the Remainder Columns,
// and the Input Columns are gone and replaced by the Output Columns.

// Keywords:
// Input Columns : a vector of column names (string) passed to MapOp specifying the column names from which
//     Tensors are taken and passed to the TensorOp Compute().
// Output Columns : a vector of column names (string) passed to MapOp specifying what are the column names
//     for the Tensors produced by TensorOp Compute().
// Remainder Columns : columns that exist in the dataset but are not mentioned in Input Columns.
//     These columns will not be passed to TensorOp Compute(), but will be appended to the end of the Output Columns.
class MapOp : public ParallelOp {
 public:
  // The nested builder class inside of the MapOp is used to help manage all of
  // the arguments for constructing it.  Use the builder by setting each argument
  // with the provided set methods, and then finally call the build method to execute
  // the actual construction.
  class Builder {
   public:
    // Builder constructor. Creates the builder object.
    // @note No default args
    // @return This is a constructor.
    Builder();

    // Default destructor
    ~Builder() = default;

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetInColNames(const std::vector<std::string> &in_col_names) {
      build_in_col_names_ = in_col_names;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetOutColNames(const std::vector<std::string> &out_col_names) {
      build_out_col_names_ = out_col_names;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetTensorFuncs(std::vector<std::shared_ptr<TensorOp>> funcs) {
      build_tensor_funcs_ = std::move(funcs);
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetColOrder(const std::vector<std::string> &col_order_) {
      build_col_order_ = col_order_;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetNumWorkers(int32_t num_workers) {
      build_num_workers_ = num_workers;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetOpConnectorSize(int32_t connector_size) {
      build_op_connector_size_ = connector_size;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetPerformanceMode(bool perf_mode) {
      build_perf_mode_ = perf_mode;
      return *this;
    }

    // The builder "build" method creates the final object.
    // @param ptr The shared_ptr to the new MapOp object
    // @return Status
    Status Build(std::shared_ptr<MapOp> *ptr);

   private:
    std::vector<std::string> build_in_col_names_;
    std::vector<std::string> build_out_col_names_;
    std::vector<std::shared_ptr<TensorOp>> build_tensor_funcs_;
    std::vector<std::string> build_col_order_;
    int32_t build_num_workers_;
    int32_t build_op_connector_size_;
    bool build_perf_mode_;  // Default true.

    // Check if the required parameters are set by the builder.
    // @return Status The error code return
    Status sanityCheck() const;
  };

  // Constructor of MapOp
  // @note The builder class should be used to call it.
  // @param in_col_names A list of input column names (should match the input/output \p tensorFuncs).
  // @param out_col_names A list of output column names (should match the input/output \p tensorFuncs).
  // @param tensor_funcs A list of TensorOp pointers for MapOp to apply to each data.
  // @param columns_order names A full list of column names (should match the whole dataset view post \p tensorFuncs).
  // @param num_workers The number of worker threads.
  // @param op_connector_size The size of each queue in the connector.
  MapOp(const std::vector<std::string> &in_col_names, const std::vector<std::string> &out_col_names,
        std::vector<std::shared_ptr<TensorOp>> tensor_funcs, const std::vector<std::string> &columns_order,
        int32_t num_workers, int32_t op_connector_size, bool perf_mode);

  // Destructor
  ~MapOp() = default;

  // A print method typically used for debugging
  // @param out The output stream to write output to
  // @param show_all A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;

  // << Stream output operator overload
  // @notes This allows you to write the debug print info using stream operators
  // @param out reference to the output stream being overloaded
  // @param mo reference to the MapOp to display
  // @return the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const MapOp &mo) {
    mo.Print(out, false);
    return out;
  }

  // Class functor operator () override.
  // All dataset ops operate by launching a thread (see ExecutionTree). This class functor will
  // provide the master loop that drives the logic for performing the work
  // @return Status The error code return
  Status operator()() override;

  // Getter
  // @return the number of threads consuming data from previous op's output Connector.
  int32_t num_consumers() const override;

  // Base-class override for NodePass visitor acceptor.
  // @param p - Pointer to the NodePass to be accepted.
  // @param modified - Whether this node visit modified the pipeline.
  // @return - Status of the node visit.
  Status Accept(NodePass *p, bool *modified) override;

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return "MapOp"; }

  // Columns order getter
  // @return The post map columns order
  std::vector<std::string> const &ColumnsOrder() const { return columns_order_; }

 private:
  // Local queues where worker threads can pop from.
  // Popping directly from the Connector can block if the previous designated threads haven't pop.
  // Setting the size of these queues to 0 is essentially the same as pulling directly from Connector.
  QueueList<std::unique_ptr<DataBuffer>> local_queues_;

  // Static variables to be ready by worker threads, no modification and readonly
  const std::vector<std::shared_ptr<TensorOp>> tfuncs_;

  // Variable to store the column name that the tensorOps are consuming
  std::vector<std::string> in_columns_;

  // Variable to store the column name that the tensorOps are producing
  std::vector<std::string> out_columns_;

  // Boolean mapping, true means to keep the column.
  std::vector<bool> keep_input_columns_;

  // Indices of the columns to process.
  std::vector<size_t> to_process_indices_;

  // Variable to store the column_order of all columns post tensorOps
  std::vector<std::string> columns_order_;

  // Performance mode is when the main thread creates local queues, pulls databuffers from the previous
  // op's Connector and distributes them to the local queues. Workers pull from the local queues.
  // If this flag is false, each worker pulls directly from the Connector. This use less resources
  // (thread and memory), but when the computation cost is heavy (e.g. DecodeOp) and fluctuating, it can
  // cause additional blocking because pop calls to Connector from the threads are synchronized to enforce the order.
  bool perf_mode_;

  // Private function for worker/thread to loop continuously. It comprises the main
  // logic of MapOp: getting the data from previous Op, validating user specified column names,
  // applying a list of TensorOps to each of the data, process the results and then
  // pushing them back to MapOp's output Connector to be fetched by the next Op.
  // @param worker_id The id assigned to this thread/worker upon creation.
  // @return Status The error code return
  Status WorkerEntry(int32_t worker_id) override;  //  In: workerId assigned by tree_

  // Private helper function for getting the next buffer
  // When PerformanceMode is enabled, workers pop from the local queue.
  // Otherwise, workers pop from the first child output Connector.
  // @param p_buffer - the buffer to return
  // @return Status return code
  Status FetchNextBuffer(std::unique_ptr<DataBuffer> *p_buffer, int32_t worker_id) {
    if (perf_mode_) {
      RETURN_IF_NOT_OK(local_queues_[worker_id]->PopFront(p_buffer));
    } else {
      RETURN_IF_NOT_OK(child_[0]->GetNextBuffer(p_buffer, worker_id));
    }
    return Status::OK();
  }

  // Private function for worker thread to perform TensorOp's compute function and get the result.
  // @param in_buffer A raw pointer to the DataBuffer. A raw pointer is fine because this function doesn't manage memory
  //     and is not shared with other threads.
  // @param[out] new_tensor_table A new Tensor Table to be populated in this function.
  Status WorkerCompute(DataBuffer *in_buffer, TensorQTable *new_tensor_table);

  // Private function that create the final column name to index mapping and
  // get indices of the columns this mapop does not use.
  // @param col_name_id_map The column name to index mapping obtained from child operator
  void CreateFinalColMap(std::unordered_map<std::string, int32_t> *col_name_id_map);

  // Validating if each of the input_columns exists in the DataBuffer.
  // @param - the column map to check
  // @return - status return code
  Status ValidateInColumns(const std::unordered_map<std::string, int32_t> &col_name_id_map);

  // Private function for computing the assignment of the column name map.
  // @return - Status
  Status ComputeColMap() override;

  // Private function for initializing private variables such as in_columns_, out_columns_.
  // @return - Status
  Status InitPrivateVariable(std::unordered_map<std::string, int32_t> *col_name_id_map);
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATASETOPS_MAP_OP_H_
