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
#include <iomanip>

#include "common/utils.h"
#include "dataset/core/pybind_support.h"
#include "dataset/engine/data_buffer.h"
#include "dataset/engine/db_connector.h"
#include "dataset/engine/opt/pass.h"
#include "dataset/kernels/data/data_utils.h"

using float16 = Eigen::half;

namespace mindspore {
namespace dataset {
BatchOp::Builder::Builder(int32_t batch_size) : builder_drop_(false), builder_pad_(false), builder_pad_map_({}) {
  builder_batch_size_ = batch_size;
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  builder_num_workers_ = cfg->num_parallel_workers();
  builder_op_connector_size_ = cfg->op_connector_size();
}

Status BatchOp::Builder::Build(std::shared_ptr<BatchOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  *ptr = std::make_shared<BatchOp>(builder_batch_size_, builder_drop_, builder_pad_, builder_op_connector_size_,
                                   builder_num_workers_, builder_cols_to_map_, builder_batch_size_func_,
                                   builder_batch_map_func_, builder_pad_map_);
  return Status::OK();
}

Status BatchOp::Builder::SanityCheck() {
  std::string err;
  err += builder_op_connector_size_ <= 0 ? "connector size <= 0\n" : "";
  err += builder_batch_size_ <= 0 ? "batch size <= 0\n" : "";
  err += builder_num_workers_ <= 0 ? "batch num_parallel_workers <= 0\n" : "";
  return err.empty() ? Status::OK() : Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, common::SafeCStr(err));
}

BatchOp::BatchOp(int32_t batch_size, bool drop, bool pad, int32_t op_queue_size, int32_t num_workers,
                 const std::vector<std::string> &cols_to_map, py::function batch_size_func, py::function batch_map_func,
                 PadInfo pad_map)
    : ParallelOp(num_workers, op_queue_size),
      start_batch_size_(batch_size),
      drop_(drop),
      pad_(pad),
      pyfunc_column_names_(cols_to_map),
      batch_size_func_(batch_size_func),
      batch_map_func_(batch_map_func),
      pad_info_(pad_map) {
  worker_queues_.Init(num_workers, op_queue_size);
}

Status BatchOp::operator()() {
  Status rc = LaunchThreadsAndInitOp();
  // Synchronize with TaskManager
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(rc);
  int64_t epoch_num = 0, batch_num = 0, cnt = 0;
  TensorRow new_row;
  std::unique_ptr<TensorQTable> table = std::make_unique<TensorQTable>();
  child_iterator_ = std::make_unique<ChildIterator>(this, 0, 0);
  RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
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
  // Always show the id and name as first line regardless if this summary or detailed print
  out << "(" << std::setw(2) << operator_id_ << ") <BatchOp>:";
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << " [batch size: " << start_batch_size_ << "]\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nStart batch size: " << start_batch_size_ << "\nDrop remainder: " << (drop_ ? "yes" : "no") << "\n\n";
  }
}

Status BatchOp::BatchRows(const std::unique_ptr<TensorQTable> *src, const std::unique_ptr<TensorQTable> *dest,
                          dsize_t batch_size) {
  if ((*src)->size() != batch_size) {
    RETURN_STATUS_UNEXPECTED("[Internal Batch ERROR] Source table size does not match the batch_size");
  }

  if (batch_size == 1) {
    TensorRow row = std::move((*src)->front());
    (*src)->pop_front();
    (*dest)->push_back(row);
    for (const auto &tensor : (*dest)->front()) {
      RETURN_IF_NOT_OK(tensor->ExpandDim(0));
    }
    return Status::OK();
  }

  TensorRow batched_row;
  auto num_columns = (*src)->front().size();
  for (size_t i = 0; i < num_columns; i++) {
    std::shared_ptr<Tensor> first_tensor = (*src)->at(0).at(i);  // first row, column i
    TensorShape first_shape = first_tensor->shape();
    DataType first_type = first_tensor->type();
    TensorShape new_shape = first_shape.PrependDim(static_cast<int64_t>(batch_size));

    std::shared_ptr<Tensor> new_tensor;
    if (first_type.IsNumeric()) {  // numeric tensor
      RETURN_IF_NOT_OK(Tensor::CreateTensor(&new_tensor, TensorImpl::kFlexible, new_shape, first_type));
      dsize_t j = 0;
      for (auto row : **src) {
        std::shared_ptr<Tensor> old_tensor = row.at(i);  // row j, column i
        if (old_tensor->shape() == first_shape) {        // check the newly popped rows have the same dim as the first
          RETURN_IF_NOT_OK(new_tensor->InsertTensor({j++}, old_tensor));
        } else {
          RETURN_STATUS_UNEXPECTED("[Batch ERROR] Inconsistent TensorShapes of Column " + std::to_string(i));
        }
      }
    } else {  // handle string column differently
      std::vector<std::string> strings;
      for (dsize_t j = 0; j < batch_size; j++) {
        std::shared_ptr<Tensor> old_tensor = (*src)->at(j).at(i);
        for (auto itr = old_tensor->begin<std::string_view>(); itr != old_tensor->end<std::string_view>(); itr++) {
          strings.emplace_back(*itr);
        }
      }
      RETURN_IF_NOT_OK(Tensor::CreateTensor(&new_tensor, strings, new_shape));
    }
    batched_row.emplace_back(new_tensor);
  }

  (*dest)->emplace_back(batched_row);

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
  if (!pyfunc_column_names_.empty()) RETURN_IF_NOT_OK(MapColumns(&table_pair));               // pass it through pyfunc
  if (pad_) RETURN_IF_NOT_OK(PadColumns(&table_pair.first, pad_info_, column_name_id_map_));  // do padding if needed
  (*db) = std::make_unique<DataBuffer>(table_pair.second.batch_num_, DataBuffer::kDeBFlagNone);
  std::unique_ptr<TensorQTable> dest_table = std::make_unique<TensorQTable>();
  RETURN_IF_NOT_OK(BatchRows(&table_pair.first, &dest_table, table_pair.first->size()));
  (*db)->set_tensor_table(std::move(dest_table));
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
  input_table.reserve(pyfunc_column_names_.size());
  for (std::string col_name : pyfunc_column_names_) {
    if (column_name_id_map_.find(col_name) == column_name_id_map_.end()) {
      RETURN_STATUS_UNEXPECTED("column : '" + col_name + "' does not exist\n");
    }
    TensorBatch tensor_batch;
    tensor_batch.reserve(table_pair->first->size());
    size_t col_idx = static_cast<size_t>(column_name_id_map_[col_name]);
    for (size_t row_idx = 0; row_idx < table_pair->first->size(); row_idx++) {
      tensor_batch.push_back(std::move(table_pair->first->at(row_idx)[col_idx]));
    }
    input_table.push_back(std::move(tensor_batch));
  }

  // Perform batch map
  TensorBatchTable output_table;
  RETURN_IF_NOT_OK(InvokeBatchMapFunc(&input_table, &output_table, table_pair->second));

  // Write back to TensorQTable
  for (size_t input_idx = 0; input_idx < pyfunc_column_names_.size(); input_idx++) {
    size_t col_idx = static_cast<size_t>(column_name_id_map_[pyfunc_column_names_[input_idx]]);
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
      if (ret_tuple.size() != pyfunc_column_names_.size() || !py::isinstance<py::tuple>(ret_tuple)) {
        return Status(StatusCode::kPyFuncException, "Batch map function should return a tuple");
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

Status BatchOp::PadColumns(std::unique_ptr<TensorQTable> *table, const PadInfo &pad_info,
                           const std::unordered_map<std::string, int32_t> &column_name_id_map) {
  RETURN_UNEXPECTED_IF_NULL(table);  // placeholder for now, might need this in the future
  CHECK_FAIL_RETURN_UNEXPECTED((*table)->front().size() == column_name_id_map.size(), "col_name_map mismatch");
  std::vector<std::shared_ptr<Tensor>> pad_vals(column_name_id_map.size(),
                                                0);  // value to pad each column's tensor with, default 0
  std::set<int32_t> pad_cols;
  // padded_shape provided by user, maximum shapes of current batch of tensors
  std::vector<std::vector<dsize_t>> pad_shapes(column_name_id_map.size()), max_shapes(column_name_id_map.size());
  RETURN_IF_NOT_OK(UnpackPadInfo(pad_info, column_name_id_map, &pad_cols, &pad_vals, &pad_shapes));

  // init each shape in max_shape to {-1,-1...} init each unspecified shape in pad_shape to -1 as well
  for (size_t col_id : pad_cols) {
    max_shapes[col_id] = std::vector<dsize_t>((*table)->front()[col_id]->Rank(), -1);
    if (pad_shapes[col_id].empty()) pad_shapes[col_id] = max_shapes[col_id];  // fill pad shape with -1
    CHECK_FAIL_RETURN_UNEXPECTED(pad_shapes[col_id].size() == max_shapes[col_id].size(), "wrong rank in pad_shape");
  }

  // calculate maximum shape for each column that needs to be padded
  for (const TensorRow &row : **table) {  // iterator each row in a batch
    for (size_t col_id : pad_cols) {      // iterator each tensor in a row
      CHECK_FAIL_RETURN_UNEXPECTED(row[col_id]->Rank() == max_shapes[col_id].size(),
                                   "Tensor to be padded together need to have the same rank");
      for (size_t dim = 0; dim < row[col_id]->Rank(); dim++) {  // pick the largest number in each dimension
        max_shapes[col_id][dim] = std::max(max_shapes[col_id][dim], row[col_id]->shape()[dim]);
      }
    }
  }

  // if user sets a dimension to -1 (None in python), use the max value for current dimension
  for (size_t col_id : pad_cols) {
    for (size_t dim = 0; dim < pad_shapes[col_id].size(); dim++) {
      if (pad_shapes[col_id][dim] < 0) pad_shapes[col_id][dim] = max_shapes[col_id][dim];
    }
  }

  // call pad on each tensor that needs to be padded
  for (TensorRow &row : **table) {
    for (size_t col_id : pad_cols) {
      std::shared_ptr<Tensor> pad_tensor;
      RETURN_IF_NOT_OK(PadEnd(row[col_id], &pad_tensor, pad_shapes[col_id], pad_vals[col_id]));
      row[col_id] = pad_tensor;
    }
  }
  return Status::OK();
}

Status BatchOp::UnpackPadInfo(const PadInfo &pad_info,
                              const std::unordered_map<std::string, int32_t> &column_name_id_map,
                              std::set<int32_t> *pad_cols, std::vector<std::shared_ptr<Tensor>> *pad_vals,
                              std::vector<std::vector<dsize_t>> *pad_shapes) {
  if (pad_info.empty()) {  // if pad_info empty, pad every columns automatically
    for (dsize_t col_id = 0; col_id < column_name_id_map.size(); col_id++) {
      pad_cols->insert(col_id);
    }
  } else {
    for (const auto &p : pad_info) {
      auto location = column_name_id_map.find(p.first);
      CHECK_FAIL_RETURN_UNEXPECTED(location != column_name_id_map.end(), "no column exists with name:" + p.first);
      auto col_id = static_cast<dsize_t>(location->second);
      CHECK_FAIL_RETURN_UNEXPECTED(col_id < pad_vals->size() && col_id < pad_shapes->size(), "col_id out of bound");
      pad_cols->insert(col_id);
      (*pad_vals)[col_id] = p.second.second;              // set pad values
      (*pad_shapes)[col_id] = p.second.first.AsVector();  // empty vector if shape is unknown
    }
  }
  return Status::OK();
}

// Visitor accept method for NodePass
Status BatchOp::Accept(NodePass *p, bool *modified) {
  // Downcast shared pointer then call visitor
  return p->RunOnNode(std::static_pointer_cast<BatchOp>(shared_from_this()), modified);
}

}  // namespace dataset
}  // namespace mindspore
