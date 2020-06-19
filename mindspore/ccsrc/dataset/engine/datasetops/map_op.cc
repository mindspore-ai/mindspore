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
#include "dataset/engine/datasetops/map_op.h"
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include "dataset/core/config_manager.h"

#include "dataset/core/constants.h"
#include "dataset/core/global_context.h"
#include "dataset/core/tensor.h"
#include "dataset/engine/data_buffer.h"
#include "dataset/engine/db_connector.h"
#include "dataset/engine/execution_tree.h"
#include "dataset/engine/opt/pass.h"
#include "dataset/kernels/tensor_op.h"
#include "utils/log_adapter.h"
#include "dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
// Builder constructor. Creates the builder object.
MapOp::Builder::Builder() : build_perf_mode_(true) {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  build_num_workers_ = cfg->num_parallel_workers();
  build_op_connector_size_ = cfg->op_connector_size();
}

// Check if the required parameters are set by the builder.
Status MapOp::Builder::sanityCheck() const {
  if (build_tensor_funcs_.empty()) {
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__,
                  "Building a MapOp that has not provided any function/operation to apply");
  }
  return Status::OK();
}

// The builder "build" method creates the final object.
Status MapOp::Builder::Build(std::shared_ptr<MapOp> *ptr) {
  RETURN_IF_NOT_OK(sanityCheck());
  *ptr = std::make_shared<MapOp>(std::move(build_in_col_names_), std::move(build_out_col_names_),
                                 std::move(build_tensor_funcs_), std::move(build_col_order_), build_num_workers_,
                                 build_op_connector_size_, build_perf_mode_);
  return Status::OK();
}

// Constructor of MapOp
MapOp::MapOp(const std::vector<std::string> &in_col_names, const std::vector<std::string> &out_col_names,
             std::vector<std::shared_ptr<TensorOp>> tensor_funcs, const std::vector<std::string> &columns_order,
             int32_t num_workers, int32_t op_connector_size, bool perf_mode)
    : ParallelOp(num_workers, op_connector_size),
      tfuncs_(std::move(tensor_funcs)),
      in_columns_(in_col_names),
      out_columns_(out_col_names),
      columns_order_(columns_order),
      perf_mode_(perf_mode) {
  // If caller didn't specify the out_col_names, assume they are same as the in_columns.
  if (out_columns_.empty() || out_columns_[0].empty()) {
    out_columns_ = in_columns_;
  }
  MS_LOG(DEBUG) << "Performance Mode in map operator is " << perf_mode_ << ".";
}

// The number of threads consuming data from previous op's output Connector.
int32_t MapOp::num_consumers() const {
  // When Performance Mode is on, there is only one thread consuming from the previous Connector.
  return perf_mode_ == true ? 1 : num_workers_;
}

// A print method typically used for debugging
void MapOp::Print(std::ostream &out, bool show_all) const {
  // Always show the id and name as first line regardless if this summary or detailed print
  out << "(" << std::setw(2) << operator_id_ << ") <MapOp>:";
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nInput column names:";
    for (size_t i = 0; i < in_columns_.size(); i++) {
      out << " " << in_columns_[i];
    }
    out << "\n  TensorOps:";
    for (size_t i = 0; i < tfuncs_.size(); i++) {
      out << " " << tfuncs_[i];
    }
    out << "\n\n";
  }
}

// This class functor will provide the master loop that drives the logic for performing the work
Status MapOp::operator()() {
  if (perf_mode_) {
    // Create and register the local queues.
    local_queues_.Init(num_workers_, oc_queue_size_);
    Status rc = local_queues_.Register(tree_->AllTasks());
    if (rc.IsError()) {
      TaskManager::FindMe()->Post();
      return rc;
    }
  }

  // The operator class just starts off threads by calling the tree_ function
  Status rc = tree_->LaunchWorkers(num_workers_, std::bind(&MapOp::WorkerEntry, this, std::placeholders::_1));
  // Synchronize with TaskManager
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(rc);

  if (perf_mode_) {
    int64_t que_id = 0;
    std::unique_ptr<DataBuffer> buff;
    bool is_eof = false;
    // Draining output connector of the previous op and distribute it to local queues.
    // Stop when all worker threads are finished (received EOF).
    while (!is_eof) {
      RETURN_IF_NOT_OK(child_[0]->GetNextBuffer(&buff, 0));
      is_eof = buff->eof();
      RETURN_IF_NOT_OK(local_queues_[que_id]->Add(std::move(buff)));
      que_id = (que_id + 1) % num_workers_;
    }
  }

  return Status::OK();
}

// Private function for worker/thread to loop continuously. It comprises the main
// logic of MapOp: getting the data from previous Op, validating user specified column names,
// applying a list of TensorOps to each of the data, process the results and then
// pushing them back to MapOp's output Connector to be fetched by the next Op.
Status MapOp::WorkerEntry(int32_t worker_id) {
  // Handshake with TaskManager that thread creation is successful.
  TaskManager::FindMe()->Post();
  std::unique_ptr<DataBuffer> in_buffer;

  // Getting a databuffer to work on.
  // Perform the first fetch here outside of the loop.  This allows us to execute one-time only
  // initializations that happen after the first fetch.
  RETURN_IF_NOT_OK(FetchNextBuffer(&in_buffer, worker_id));

  // Sanity check the databuffer.
  // Special case: if there's more threads than buffers, some threads simply get the final control
  // messages (eoe/eof), and so they will not perform the check.
  if (!in_buffer->eoe() && !in_buffer->eof()) {
    int32_t num_rows = in_buffer->NumRows();
    int32_t num_cols = in_buffer->NumCols();
    if (num_rows == 0 || num_cols == 0) {
      RETURN_STATUS_UNEXPECTED("MapOp is getting an empty DataBuffer.");
    }
  }

  // Now that init work is done, drop into the main fetching loop.
  // Map op does not use child iterator, and it needs to manually handle eoe and eof's itself
  // rather than use the base-class defaults.
  while (true) {
    // Handle EOE and EOF ourselves. Implicit eoe/eof handling in GetNextInput does not work
    // with Performance Mode design.
    if (in_buffer->eoe()) {
      // Calling base class EoeReceived to forward eoe buffer.
      RETURN_IF_NOT_OK(EoeReceived(worker_id));
      RETURN_IF_NOT_OK(FetchNextBuffer(&in_buffer, worker_id));
      continue;
    } else if (in_buffer->eof()) {
      // Calling base class EofReceived to forward eof buffer.
      RETURN_IF_NOT_OK(EofReceived(worker_id));
      break;
    }

    std::unique_ptr<TensorQTable> new_tensor_table(std::make_unique<TensorQTable>());
    // Perform the compute function of TensorOp(s) and store the result in new_tensor_table.
    RETURN_IF_NOT_OK(WorkerCompute(in_buffer.get(), new_tensor_table.get()));

    // Replace the TensorTable in DataBuffer with the new one.
    in_buffer->set_tensor_table(std::move(new_tensor_table));

    // Push the buffer onto the connector for next operator to consume.
    RETURN_IF_NOT_OK(out_connector_->Add(static_cast<int>(worker_id), std::move(in_buffer)));

    // Fetch the next buffer and loop back to the top.
    RETURN_IF_NOT_OK(FetchNextBuffer(&in_buffer, worker_id));
  }

  return Status::OK();
}

Status MapOp::WorkerCompute(DataBuffer *in_buffer, TensorQTable *new_tensor_table) {
  // Getting number of rows and cols in this buffer.
  int32_t num_rows = in_buffer->NumRows();
  int32_t num_cols = in_buffer->NumCols();

  for (int32_t r = 0; r < num_rows; r++) {
    // to_process   : A vector of Tensors only holding cols in input_columns.
    // result_row;  : A vector of Tensors to hold the result after Compute().
    // cur_row      : A vector of Tensors holding all the columns from DataBuffer.
    TensorRow to_process, result_row, cur_row;
    RETURN_IF_NOT_OK(in_buffer->PopRow(&cur_row));

    // Populate the Tensor from the current row to be processed by TensorOp
    for (const auto &idx : to_process_indices_) {
      to_process.push_back(std::move(cur_row[idx]));
    }

    // Looping over multiple TensorOps supplied in to MapOp.
    // The assumption is that the result of one TensorOp matches the required input to the next TensorOp.
    for (size_t i = 0; i < tfuncs_.size(); i++) {
      // TensorOp can operate on single col or multiple cols. MapOp always call compute for multiple cols.
      // TensorOp base class will call the single column Compute() depending on the ops.
      // Note: The columns of the result_row is not preallocated, the compute function of each tensor op are
      // required to resize/push back the result_row
      RETURN_IF_NOT_OK(tfuncs_[i]->Compute(to_process, &result_row));

      // Assign result_row to to_process for the next TensorOp processing, except for the last TensorOp in the list.
      if (i + 1 < tfuncs_.size()) {
        to_process = std::move(result_row);
      }
    }

    if (out_columns_.size() != result_row.size()) {
      return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__,
                    "Result of a tensorOp doesn't match output column names");
    }

    if (in_columns_.size() == out_columns_.size()) {
      for (size_t i = 0; i < result_row.size(); i++) {
        cur_row[to_process_indices_[i]] = std::move(result_row[i]);
      }
      new_tensor_table->push_back(std::move(cur_row));
    } else {
      // Add the columns we did not touch to the result_row.
      for (int32_t i = 0; i < num_cols; i++) {
        if (keep_input_columns_[i]) {
          result_row.push_back(std::move(cur_row[i]));
        }
      }

      // Add this final result_row to our new TensorTable.
      new_tensor_table->push_back(std::move(result_row));
    }
  }

  return Status::OK();
}

Status MapOp::ComputeColMap() {
  // If the map has not been set up yet in the base class, then set it up
  if (column_name_id_map_.empty()) {
    std::unordered_map<std::string, int32_t> current_name_id_map = child_[0]->column_name_id_map();
    // Initialize private variables
    RETURN_IF_NOT_OK(InitPrivateVariable(&current_name_id_map));
    // Create the final column name to index mapping in the base class field
    CreateFinalColMap(&current_name_id_map);
    MS_LOG(DEBUG) << "Column name map for map op set: " << this->ColumnNameMapAsString();
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

// Validating if each of the input_columns exists in the DataBuffer.
Status MapOp::ValidateInColumns(const std::unordered_map<std::string, int32_t> &col_name_id_map) {
  for (const auto &inCol : in_columns_) {
    bool found = col_name_id_map.find(inCol) != col_name_id_map.end() ? true : false;
    if (!found) {
      std::string err_msg = "input column name: " + inCol + " doesn't exist in the dataset columns.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  }
  return Status::OK();
}

Status MapOp::InitPrivateVariable(std::unordered_map<std::string, int32_t> *col_name_id_map) {
  // If input_columns is empty(), The col at index-0 will be picked.
  if (in_columns_.empty()) {
    for (const auto &pair : *col_name_id_map) {
      if (pair.second == 0) {
        MS_LOG(INFO) << "Input columns empty for map op, will apply to the first column in the current table.";
        in_columns_.push_back(pair.first);
        break;
      }
    }

    // If caller didn't specify the out_col_names, assume they are same as the input_columns.
    // This was done in the constructor, but if input columns was empty to start we have to redo it here.
    if (out_columns_.empty() || out_columns_[0].empty()) {
      out_columns_ = in_columns_;
    }
  }

  // Before we continue, issue a sanity check to make sure the input columns from user and the incoming
  // columns from child are correct
  RETURN_IF_NOT_OK(this->ValidateInColumns(*col_name_id_map));

  // initialize keep_input_columns, true means to keep the column.
  keep_input_columns_.resize(col_name_id_map->size(), true);
  for (const auto &col_name : in_columns_) {
    int32_t missed = (*col_name_id_map)[col_name];
    keep_input_columns_[missed] = false;
  }

  // initialize to_process_indices.
  for (const auto &col_name : in_columns_) {
    to_process_indices_.push_back((*col_name_id_map)[col_name]);
  }
  return Status::OK();
}

// Create the final column name to index mapping and get indices of the columns this mapop does not use.
void MapOp::CreateFinalColMap(std::unordered_map<std::string, int32_t> *col_name_id_map) {
  std::unordered_map<std::string, int32_t> final_col_name_id_map;
  size_t num_cols = col_name_id_map->size();
  std::vector<int32_t> new_ids(num_cols);
  if (in_columns_.size() == out_columns_.size()) {
    for (size_t i = 0; i < in_columns_.size(); i++) {
      int32_t loc = (*col_name_id_map)[in_columns_[i]];
      (void)col_name_id_map->erase(in_columns_[i]);
      (*col_name_id_map)[out_columns_[i]] = loc;
    }

    // Set the base class final column id map result
    column_name_id_map_ = *col_name_id_map;
  } else {
    int32_t fill_idx = 0;
    // First columns of the tables are occupied by the output columns from tensorOp.
    for (const auto &col_name : out_columns_) {
      final_col_name_id_map[col_name] = fill_idx++;
    }

    // Creating new_ids mapping for the columns we keep.
    for (size_t i = 0; i < num_cols; i++) {
      if (keep_input_columns_[i]) {
        new_ids[i] = fill_idx++;
      }
    }

    // Iterating through the old mapping to update the final mapping for the columns we kept.
    std::string name;
    for (const auto &pair : *col_name_id_map) {
      name = pair.first;
      int32_t old_id = pair.second;
      if (keep_input_columns_[old_id]) {
        final_col_name_id_map[name] = new_ids[old_id];
      }
    }

    // Set the base class final column id map result
    column_name_id_map_ = final_col_name_id_map;
  }
}

// Visitor accept method for NodePass
Status MapOp::Accept(NodePass *p, bool *modified) {
  // Downcast shared pointer then call visitor
  return p->RunOnNode(std::static_pointer_cast<MapOp>(shared_from_this()), modified);
}
}  // namespace dataset
}  // namespace mindspore
