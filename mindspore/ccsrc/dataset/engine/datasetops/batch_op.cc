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
#include "dataset/engine/datasetops/batch_op.h"
#include <utility>
#include "common/utils.h"
#include "dataset/engine/data_buffer.h"
#include "dataset/engine/db_connector.h"

namespace mindspore {
namespace dataset {
BatchOp::Builder::Builder(int32_t batch_size) : builder_drop_(false) {
  builder_batch_size_ = batch_size;
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  builder_num_workers_ = cfg->num_parallel_workers();
  builder_op_connector_size_ = cfg->op_connector_size();
}

Status BatchOp::Builder::Build(std::shared_ptr<BatchOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  *ptr = std::make_shared<BatchOp>(builder_batch_size_, builder_drop_, builder_op_connector_size_, builder_num_workers_,
                                   builder_cols_to_map_, builder_batch_size_func_, builder_batch_map_func_);
  return Status::OK();
}

Status BatchOp::Builder::SanityCheck() {
  std::string err;
  err += builder_op_connector_size_ <= 0 ? "connector size <= 0\n" : "";
  err += builder_batch_size_ <= 0 ? "batch size <= 0\n" : "";
  err += builder_num_workers_ <= 0 ? "batch num_parallel_workers <= 0\n" : "";
  return err.empty() ? Status::OK() : Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, common::SafeCStr(err));
}

BatchOp::BatchOp(int32_t batch_size, bool drop, int32_t op_queue_size, int32_t num_workers,
                 const std::vector<std::string> &cols_to_map, py::function batch_size_func, py::function batch_map_func)
    : ParallelOp(num_workers, op_queue_size),
      start_batch_size_(batch_size),
      drop_(drop),
      input_column_names_(cols_to_map),
      batch_size_func_(batch_size_func),
      batch_map_func_(batch_map_func) {
  worker_queues_.Init(num_workers, op_queue_size);
}

Status BatchOp::operator()() {
  RETURN_IF_NOT_OK(LaunchThreadsAndInitOp());
  TaskManager::FindMe()->Post();
  int64_t epoch_num = 0, batch_num = 0, cnt = 0;
  TensorRow new_row;
  std::unique_ptr<TensorQTable> table = std::make_unique<TensorQTable>();
  child_iterator_ = std::make_unique<ChildIterator>(this, 0, 0);
  RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
  column_name_map_ = child_iterator_->col_name_id_map();
  int32_t cur_batch_size = 0;
  RETURN_IF_NOT_OK(GetBatchSize(&cur_batch_size, CBatchInfo(0, 0, 0)));
  while (child_iterator_->eof_handled() == false) {
    while (new_row.empty() == false) {
      table->emplace_back(new_row);
      // if # of rows is enough to make 1 batch (1 batch is buffer), send it to worker_queue
      if (table->size() == static_cast<size_t>(cur_batch_size)) {
        RETURN_IF_NOT_OK(worker_queues_[cnt++ % num_workers_]->EmplaceBack(
          std::make_pair(std::move(table), CBatchInfo(epoch_num, batch_num++, cnt - epoch_num))));
        table = std::make_unique<TensorQTable>();
        RETURN_IF_NOT_OK(GetBatchSize(&cur_batch_size, CBatchInfo(epoch_num, batch_num, cnt - epoch_num)));
      }
      RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
    }
    // Reminder logic, execute only when there is a remainder (table is non empty) and don't drop
    if (drop_ == false && table->empty() == false) {
      RETURN_IF_NOT_OK(worker_queues_[cnt++ % num_workers_]->EmplaceBack(
        std::make_pair(std::move(table), CBatchInfo(epoch_num, batch_num++, cnt - epoch_num))));
    }
    table = std::make_unique<TensorQTable>();  // this drops when drop == true
    // end of the current epoch, batch_num should start from 0 again
    batch_num = 0;
    epoch_num++;
    RETURN_IF_NOT_OK(
      worker_queues_[cnt++ % num_workers_]->EmplaceBack(std::make_pair(nullptr, CBatchInfo(batchCtrl::kEOE))));
    RETURN_IF_NOT_OK(GetBatchSize(&cur_batch_size, CBatchInfo(epoch_num, batch_num, cnt - epoch_num)));
    RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
  }  // end of eof_handled() == false
  RETURN_IF_NOT_OK(
    worker_queues_[cnt++ % num_workers_]->EmplaceBack(std::make_pair(nullptr, CBatchInfo(batchCtrl::kEOF))));
  // EOF received, send quit signal (an empty buffer) to all workers
  for (int32_t ind = 0; ind < num_workers_; ind++) {
    RETURN_IF_NOT_OK(
      worker_queues_[cnt++ % num_workers_]->EmplaceBack(std::make_pair(nullptr, CBatchInfo(batchCtrl::kQuit))));
  }
  return Status::OK();
}

void BatchOp::Print(std::ostream &out, bool show_all) const {
  ParallelOp::Print(out, show_all);
  out << "\nBatchOp:\n"
      << "number of parallel workers: " << num_workers_ << "\nBatch size: " << start_batch_size_
      << "\nDrop remainder: " << (drop_ ? "yes" : "no") << "\n\n";
}

Status BatchOp::BatchRows(const std::unique_ptr<TensorQTable> *source_table,
                          const std::unique_ptr<TensorQTable> *dest_table, size_t batch_size) {
  if ((*source_table)->size() < batch_size || (*source_table)->size() == 0) {
    RETURN_STATUS_UNEXPECTED("[Internal Batch ERROR] Insufficient rows in source_table\n");
  }
  TensorRow row = std::move((*source_table)->front());
  (*source_table)->pop_front();
  if (batch_size == 1) {
    for (std::shared_ptr<Tensor> tensor : row) {
      RETURN_IF_NOT_OK(tensor->ExpandDim(0));
    }
    (*dest_table)->push_back(row);
  } else {  // batch_size > 1
    std::vector<TensorShape> row_shapes;
    TensorRow batched_row;
    for (size_t i = 0; i < row.size(); i++) {  // Handle the first row popped
      row_shapes.push_back(row[i]->shape());
      std::shared_ptr<Tensor> ts;
      RETURN_IF_NOT_OK(Tensor::CreateTensor(
        &ts, TensorImpl::kFlexible, row[i]->shape().PrependDim(static_cast<int64_t>(batch_size)), row[i]->type()));
      batched_row.emplace_back(ts);
      RETURN_IF_NOT_OK(batched_row[i]->InsertTensor(std::vector<dsize_t>(1, 0), row[i]));  // {j} = 0
    }
    for (size_t j = 1; j < batch_size; j++) {  // Handle the rest of the rows
      row = std::move((*source_table)->front());
      (*source_table)->pop_front();
      for (size_t i = 0; i < row.size(); i++) {
        if (row[i]->shape() == row_shapes[i]) {  // check the newly popped rows have the same dim as the first
          RETURN_IF_NOT_OK(batched_row[i]->InsertTensor(std::vector<dsize_t>(1, j), row[i]));
        } else {
          RETURN_STATUS_UNEXPECTED("[Batch ERROR] Inconsistent TensorShapes\n");
        }
      }
    }
    (*dest_table)->emplace_back(batched_row);
  }
  return Status::OK();
}

Status BatchOp::WorkerEntry(int32_t workerId) {
  TaskManager::FindMe()->Post();
  std::pair<std::unique_ptr<TensorQTable>, CBatchInfo> table_pair;
  RETURN_IF_NOT_OK(worker_queues_[workerId]->PopFront(&table_pair));
  while (table_pair.second.ctrl_ != batchCtrl::kQuit) {
    if (table_pair.second.ctrl_ == batchCtrl::kEOE) {
      RETURN_IF_NOT_OK(out_connector_->Add(workerId, std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE)));
    } else if (table_pair.second.ctrl_ == batchCtrl::kEOF) {
      RETURN_IF_NOT_OK(out_connector_->Add(workerId, std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOF)));
    } else if (table_pair.second.ctrl_ == batchCtrl::kNoCtrl) {
      std::unique_ptr<DataBuffer> db = nullptr;
      RETURN_IF_NOT_OK(MakeBatchedBuffer(std::move(table_pair), &db));
      RETURN_IF_NOT_OK(out_connector_->Add(workerId, std::move(db)));
    }
    RETURN_IF_NOT_OK(worker_queues_[workerId]->PopFront(&table_pair));
  }
  return Status::OK();
}

Status BatchOp::MakeBatchedBuffer(std::pair<std::unique_ptr<TensorQTable>, CBatchInfo> table_pair,
                                  std::unique_ptr<DataBuffer> *db) {
  RETURN_UNEXPECTED_IF_NULL(table_pair.first);
  if (!input_column_names_.empty()) RETURN_IF_NOT_OK(MapColumns(&table_pair));  // pass it through pyfunc
  (*db) = std::make_unique<DataBuffer>(table_pair.second.batch_num_, DataBuffer::kDeBFlagNone);
  std::unique_ptr<TensorQTable> dest_table = std::make_unique<TensorQTable>();
  RETURN_IF_NOT_OK(BatchRows(&table_pair.first, &dest_table, table_pair.first->size()));
  (*db)->set_tensor_table(std::move(dest_table));
  (*db)->set_column_name_map(column_name_map_);
  return Status::OK();
}

Status BatchOp::LaunchThreadsAndInitOp() {
  RETURN_UNEXPECTED_IF_NULL(tree_);
  RETURN_IF_NOT_OK(worker_queues_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(tree_->LaunchWorkers(num_workers_, std::bind(&BatchOp::WorkerEntry, this, std::placeholders::_1)));
  return Status::OK();
}

Status BatchOp::EofReceived(int32_t) { return Status::OK(); }

Status BatchOp::EoeReceived(int32_t) {
  state_ = OpState::kDeOpIdle;
  return Status::OK();
}

Status BatchOp::MapColumns(std::pair<std::unique_ptr<TensorQTable>, CBatchInfo> *table_pair) {
  TensorBatchTable input_table;
  input_table.reserve(input_column_names_.size());
  for (std::string col_name : input_column_names_) {
    if (column_name_map_.find(col_name) == column_name_map_.end()) {
      RETURN_STATUS_UNEXPECTED("column : '" + col_name + "' does not exist\n");
    }
    TensorBatch tensor_batch;
    tensor_batch.reserve(table_pair->first->size());
    size_t col_idx = static_cast<size_t>(column_name_map_[col_name]);
    for (size_t row_idx = 0; row_idx < table_pair->first->size(); row_idx++) {
      tensor_batch.push_back(std::move(table_pair->first->at(row_idx)[col_idx]));
    }
    input_table.push_back(std::move(tensor_batch));
  }

  // Perform batch map
  TensorBatchTable output_table;
  RETURN_IF_NOT_OK(InvokeBatchMapFunc(&input_table, &output_table, table_pair->second));

  // Write back to TensorQTable
  for (size_t input_idx = 0; input_idx < input_column_names_.size(); input_idx++) {
    size_t col_idx = static_cast<size_t>(column_name_map_[input_column_names_[input_idx]]);
    size_t row_id = 0;
    for (TensorRow &row : *(table_pair->first)) {
      row[col_idx] = std::move(output_table[input_idx][row_id++]);
    }
  }
  return Status::OK();
}

Status BatchOp::GetBatchSize(int32_t *batch_size, CBatchInfo info) {
  if (batch_size_func_ != nullptr) {
    RETURN_IF_NOT_OK(InvokeBatchSizeFunc(batch_size, info));
  } else {
    (*batch_size) = start_batch_size_;
  }
  return Status::OK();
}

Status BatchOp::InvokeBatchSizeFunc(int32_t *batch_size, CBatchInfo info) {
  {
    // Acquire Python GIL
    py::gil_scoped_acquire gil_acquire;
    if (Py_IsInitialized() == 0) {
      return Status(StatusCode::kPythonInterpreterFailure, "Python Interpreter is finalized");
    }
    try {
      py::object size = batch_size_func_(info);
      *batch_size = size.cast<int32_t>();
      if (*batch_size <= 0) {
        return Status(StatusCode::kPyFuncException, "Batch size function should return an integer > 0");
      }
    } catch (const py::error_already_set &e) {
      return Status(StatusCode::kPyFuncException, e.what());
    } catch (const py::cast_error &e) {
      return Status(StatusCode::kPyFuncException, "Batch size function should return an integer > 0");
    }
  }
  return Status(StatusCode::kOK, "Batch size func call succeed");
}

Status BatchOp::InvokeBatchMapFunc(TensorBatchTable *input, TensorBatchTable *output, CBatchInfo info) {
  {
    // Acquire Python GIL
    py::gil_scoped_acquire gil_acquire;
    if (Py_IsInitialized() == 0) {
      return Status(StatusCode::kPythonInterpreterFailure, "Python Interpreter is finalized");
    }
    try {
      // Prepare batch map call back parameters
      py::tuple input_args(input->size() + 1);
      for (size_t i = 0; i < input->size(); i++) {
        std::vector<py::array> np_batch;
        for (std::shared_ptr<Tensor> t : input->at(i)) {
          py::array np_array;
          RETURN_IF_NOT_OK(t->GetDataAsNumpy(&np_array));
          np_batch.push_back(std::move(np_array));
        }
        input_args[i] = np_batch;
      }
      input_args[input->size()] = info;
      // Invoke batch map func
      py::object ret_py_obj = batch_map_func_(*input_args);
      // Parse batch map return value
      py::tuple ret_tuple = py::cast<py::tuple>(ret_py_obj);
      if (ret_tuple.size() != input_column_names_.size() || !py::isinstance<py::tuple>(ret_tuple)) {
        return Status(StatusCode::kPyFuncException, "Batch map function should return an tuple if size(input_columns)");
      }
      for (size_t i = 0; i < ret_tuple.size(); i++) {
        TensorBatch output_batch;
        py::list output_list = py::cast<py::list>(ret_tuple[i]);
        for (size_t j = 0; j < output_list.size(); j++) {
          std::shared_ptr<Tensor> out;
          RETURN_IF_NOT_OK(Tensor::CreateTensor(&out, py::cast<py::array>(output_list[j])));
          output_batch.push_back(std::move(out));
        }
        output->push_back(std::move(output_batch));
      }
    } catch (const py::error_already_set &e) {
      return Status(StatusCode::kPyFuncException, e.what());
    } catch (const py::cast_error &e) {
      return Status(StatusCode::kPyFuncException, "Batch map function should return an tuple of list of numpy array");
    }
  }
  return Status(StatusCode::kOK);
}
}  // namespace dataset
}  // namespace mindspore
