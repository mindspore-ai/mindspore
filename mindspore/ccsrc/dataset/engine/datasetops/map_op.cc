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
                                 std::move(build_tensor_funcs_), build_num_workers_, build_op_connector_size_,
                                 build_perf_mode_);
  return Status::OK();
}

// Constructor of MapOp
MapOp::MapOp(const std::vector<std::string> &in_col_names, const std::vector<std::string> &out_col_names,
             std::vector<std::shared_ptr<TensorOp>> tensor_funcs, int32_t num_workers, int32_t op_connector_size,
             bool perf_mode)
    : ParallelOp(num_workers, op_connector_size),
      tfuncs_(std::move(tensor_funcs)),
      in_columns_(in_col_names),
      out_columns_(out_col_names),
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
    RETURN_IF_NOT_OK(local_queues_.Register(tree_->AllTasks()));
  }

  // The operator class just starts off threads by calling the tree_ function
  RETURN_IF_NOT_OK(tree_->LaunchWorkers(num_workers_, std::bind(&MapOp::WorkerEntry, this, std::placeholders::_1)));
  // Synchronize with TaskManager
  TaskManager::FindMe()->Post();

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

  // MapOp is not using the ChildIterator class to fetch rows because it needs to track
  // rows at a per-buffer level. ChildIterator abstracts the concept of buffers making it
  // less convenient to use that interface for fetching.
  std::unique_ptr<DataBuffer> in_buffer;

  // Loop until eof buffer is encountered
  while (true) {
    // Getting a databuffer to work on.
    // When PerformanceMode is enabled, workers pop from the local queue.
    // Otherwise, workers pop from the first child output Connector.
    if (perf_mode_) {
      RETURN_IF_NOT_OK(local_queues_[worker_id]->PopFront(&in_buffer));
    } else {
      RETURN_IF_NOT_OK(child_[0]->GetNextBuffer(&in_buffer, worker_id));
    }

    // Handle EOE and EOF ourselves. Implicit eoe/eof handling in GetNextInput does not work
    // with Performance Mode design.
    if (in_buffer->eoe()) {
      // Calling base class EoeReceived to forward eoe buffer.
      RETURN_IF_NOT_OK(EoeReceived(worker_id));
      continue;
    } else if (in_buffer->eof()) {
      // Calling base class EofReceived to forward eof buffer.
      RETURN_IF_NOT_OK(EofReceived(worker_id));
      break;
    }

    // Boolean mapping, true means to keep the column.
    std::vector<bool> keep_input_columns;
    // Indices of the columns to process.
    std::vector<size_t> to_process_indices;
    // The final column mapping after performing this map
    std::unordered_map<std::string, int32_t> final_col_name_id_map;

    // Thread local variables to avoid lock. When in_columns_ is empty and workers will write
    // the name of the first column into input_columns (thread local) instead of in_columns_ (thread global).
    std::vector<std::string> input_columns = in_columns_;
    std::vector<std::string> output_columns = out_columns_;

    // Initialize the above data structures
    RETURN_IF_NOT_OK(WorkerEntryInit(in_buffer.get(), &keep_input_columns, &to_process_indices, &final_col_name_id_map,
                                     &input_columns, &output_columns));

    std::unique_ptr<TensorQTable> new_tensor_table(std::make_unique<TensorQTable>());
    // Perform the compute function of TensorOp(s) and store the result in new_tensor_table.
    RETURN_IF_NOT_OK(WorkerCompute(in_buffer.get(), to_process_indices, new_tensor_table.get(), keep_input_columns,
                                   &input_columns, &output_columns));

    // Update column name to index mapping because tensorOp might add/remove column.
    in_buffer->set_column_name_map(final_col_name_id_map);
    // Replace the TensorTable in DataBuffer with the new one.
    in_buffer->set_tensor_table(std::move(new_tensor_table));

    // Push the buffer onto the connector for next operator to consume.
    RETURN_IF_NOT_OK(out_connector_->Add(static_cast<int>(worker_id), std::move(in_buffer)));
  }

  return Status::OK();
}

Status MapOp::WorkerCompute(DataBuffer *in_buffer, const std::vector<size_t> &to_process_indices,
                            TensorQTable *new_tensor_table, const std::vector<bool> &keep_input_columns,
                            std::vector<std::string> *input_columns, std::vector<std::string> *output_columns) {
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
    for (const auto &idx : to_process_indices) {
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

    if (output_columns->size() != result_row.size()) {
      return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__,
                    "Result of a tensorOp doesn't match output column names");
    }

    if (input_columns->size() == output_columns->size()) {
      for (size_t i = 0; i < result_row.size(); i++) {
        cur_row[to_process_indices[i]] = std::move(result_row[i]);
      }
      new_tensor_table->push_back(std::move(cur_row));
    } else {
      // Add the columns we did not touch to the result_row.
      for (int32_t i = 0; i < num_cols; i++) {
        if (keep_input_columns[i]) {
          result_row.push_back(std::move(cur_row[i]));
        }
      }

      // Add this final result_row to our new TensorTable.
      new_tensor_table->push_back(std::move(result_row));
    }
  }

  return Status::OK();
}

// Validating if each of the input_columns exists in the DataBuffer.
Status MapOp::ValidateInColumns(const std::unordered_map<std::string, int32_t> &col_name_id_map,
                                std::vector<std::string> *input_columns) {
  for (const auto &inCol : *input_columns) {
    bool found = col_name_id_map.find(inCol) != col_name_id_map.end() ? true : false;
    if (!found) {
      std::string err_msg = "input column name: " + inCol + " doesn't exist in the dataset columns.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  }
  return Status::OK();
}

// initialize some internal data structure used by WorkerEntry()
Status MapOp::WorkerEntryInit(const DataBuffer *in_buf, std::vector<bool> *keep_input_columns,
                              std::vector<size_t> *to_process_indices,
                              std::unordered_map<std::string, int32_t> *final_col_name_id_map,
                              std::vector<std::string> *input_columns, std::vector<std::string> *output_columns) {
  int32_t num_rows = in_buf->NumRows();
  int32_t num_cols = in_buf->NumCols();
  if (num_rows == 0 || num_cols == 0) {
    RETURN_STATUS_UNEXPECTED("MapOp is getting an empty DataBuffer.");
  }
  std::unordered_map<std::string, int32_t> col_name_id_map = in_buf->column_name_map();
  // Check if there is invalid column name in the inColumns.
  RETURN_IF_NOT_OK(ValidateInColumns(col_name_id_map, input_columns));

  // If input_columns is empty(), The col at index-0 will be picked.
  if (input_columns->empty()) {
    for (const auto &pair : col_name_id_map) {
      if (pair.second == 0) {
        MS_LOG(INFO) << "Input columns in map operator is empty, will apply to the first column in the current table.";
        input_columns->push_back(pair.first);
        break;
      }
    }

    // If caller didn't specify the out_col_names, assume they are same as the input_columns.
    if (output_columns->empty() || (*output_columns)[0].empty()) {
      *output_columns = *input_columns;
    }
  }

  // initialize keep_input_columns, true means to keep the column.
  keep_input_columns->resize(num_cols, true);
  for (const auto &col_name : *input_columns) {
    int32_t missed = col_name_id_map[col_name];
    (*keep_input_columns)[missed] = false;
  }

  // initialize to_process_indices.
  for (const auto &col_name : *input_columns) {
    to_process_indices->push_back(col_name_id_map[col_name]);
  }

  // Create the final column name to index mapping.
  *final_col_name_id_map = CreateFinalColMap(&col_name_id_map, *keep_input_columns, input_columns, output_columns);

  return Status::OK();
}

// Create the final column name to index mapping and get indices of the columns this mapop does not use.
std::unordered_map<std::string, int32_t> MapOp::CreateFinalColMap(
  std::unordered_map<std::string, int32_t> *col_name_id_map, const std::vector<bool> &keep_input_columns,
  std::vector<std::string> *input_columns, std::vector<std::string> *output_columns) {
  std::unordered_map<std::string, int32_t> final_col_name_id_map;
  size_t num_cols = col_name_id_map->size();
  std::vector<int32_t> new_ids(num_cols);
  if (input_columns->size() == output_columns->size()) {
    for (size_t i = 0; i < input_columns->size(); i++) {
      int32_t loc = (*col_name_id_map)[(*input_columns)[i]];
      (void)col_name_id_map->erase((*input_columns)[i]);
      (*col_name_id_map)[(*output_columns)[i]] = loc;
    }

    return *col_name_id_map;
  } else {
    int32_t fill_idx = 0;
    // First columns of the tables are occupied by the output columns from tensorOp.
    for (const auto &col_name : *output_columns) {
      final_col_name_id_map[col_name] = fill_idx++;
    }

    // Creating new_ids mapping for the columns we keep.
    for (size_t i = 0; i < num_cols; i++) {
      if (keep_input_columns[i]) {
        new_ids[i] = fill_idx++;
      }
    }

    // Iterating through the old mapping to update the final mapping for the columns we kept.
    std::string name;
    for (const auto &pair : *col_name_id_map) {
      name = pair.first;
      int32_t old_id = pair.second;
      if (keep_input_columns[old_id]) {
        final_col_name_id_map[name] = new_ids[old_id];
      }
    }

    return final_col_name_id_map;
  }
}
}  // namespace dataset
}  // namespace mindspore
