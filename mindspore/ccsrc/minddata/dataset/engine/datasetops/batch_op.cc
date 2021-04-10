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
#include "minddata/dataset/engine/datasetops/batch_op.h"

#include <utility>

#include "utils/ms_utils.h"
#ifdef ENABLE_PYTHON
#include "minddata/dataset/core/pybind_support.h"
#endif
#include "minddata/dataset/engine/data_buffer.h"
#include "minddata/dataset/engine/db_connector.h"
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/util/status.h"

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
#ifdef ENABLE_PYTHON
  *ptr = std::make_shared<BatchOp>(builder_batch_size_, builder_drop_, builder_pad_, builder_op_connector_size_,
                                   builder_num_workers_, builder_in_names_, builder_out_names_,
                                   builder_batch_size_func_, builder_batch_map_func_, builder_pad_map_);
#else
  *ptr = std::make_shared<BatchOp>(builder_batch_size_, builder_drop_, builder_pad_, builder_op_connector_size_,
                                   builder_num_workers_, builder_in_names_, builder_pad_map_);
#endif
  return Status::OK();
}

Status BatchOp::Builder::SanityCheck() {
  std::string err;
  err += builder_op_connector_size_ <= 0 ? "Invalid parameter, connector_size must be greater than 0, but got " +
                                             std::to_string(builder_op_connector_size_) + ".\n"
                                         : "";
  err += builder_batch_size_ <= 0 ? "Invalid parameter, batch_size must be greater than 0, but got " +
                                      std::to_string(builder_batch_size_) + ".\n"
                                  : "";
  err += builder_num_workers_ <= 0 ? "Invalid parameter, num_parallel_workers must be greater than 0, but got " +
                                       std::to_string(builder_num_workers_) + ".\n"
                                   : "";
  return err.empty() ? Status::OK() : Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, common::SafeCStr(err));
}

#ifdef ENABLE_PYTHON
BatchOp::BatchOp(int32_t batch_size, bool drop, bool pad, int32_t op_queue_size, int32_t num_workers,
                 const std::vector<std::string> &in_col, const std::vector<std::string> &out_col,
                 py::function batch_size_func, py::function batch_map_func, PadInfo pad_map)
    : ParallelOp(num_workers, op_queue_size),
      start_batch_size_(batch_size),
      drop_(drop),
      pad_(pad),
      in_col_names_(in_col),
      out_col_names_(out_col),
      batch_size_func_(batch_size_func),
      batch_map_func_(batch_map_func),
      pad_info_(pad_map),
      batch_num_(0),
      batch_cnt_(0) {
  // Adjust connector queue size.  After batch each row is batch_size times larger
  int32_t queue_size;
  queue_size = std::max(1, op_queue_size / start_batch_size_);
  if (num_workers == 1) {
    // ensure there is at least 2 queue slots for whole operation..  If only 1 worker, incrase it to 2
    queue_size = std::max(2, queue_size);
  }

  worker_queues_.Init(num_workers, queue_size);
}
// if PYTHON is disabled. per_batch_map can't be used
#else
BatchOp::BatchOp(int32_t batch_size, bool drop, bool pad, int32_t op_queue_size, int32_t num_workers,
                 const std::vector<std::string> &cols_to_map, PadInfo pad_map)
    : ParallelOp(num_workers, op_queue_size),
      start_batch_size_(batch_size),
      drop_(drop),
      pad_(pad),
      in_col_names_(cols_to_map),
      pad_info_(pad_map),
      batch_num_(0),
      batch_cnt_(0) {
  int32_t queue_size;
  queue_size = std::max(1, op_queue_size / start_batch_size_);
  if (num_workers == 1) {
    // ensure there is at least 2 queue slots for whole operation..  If only 1 worker, incrase it to 2
    queue_size = std::max(2, queue_size);
  }
  worker_queues_.Init(num_workers, queue_size);
}
#endif

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
        RETURN_IF_NOT_OK(worker_queues_[cnt % num_workers_]->EmplaceBack(
          std::make_pair(std::move(table), CBatchInfo(epoch_num, batch_num++, cnt + 1 - epoch_num))));
        cnt++;
        table = std::make_unique<TensorQTable>();
        RETURN_IF_NOT_OK(GetBatchSize(&cur_batch_size, CBatchInfo(epoch_num, batch_num, cnt - epoch_num)));
      }
      RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
    }
    // Reminder logic, execute only when there is a remainder (table is non empty) and don't drop
    if (drop_ == false && table->empty() == false) {
      RETURN_IF_NOT_OK(worker_queues_[cnt % num_workers_]->EmplaceBack(
        std::make_pair(std::move(table), CBatchInfo(epoch_num, batch_num++, cnt + 1 - epoch_num))));
      cnt++;
    }
    table = std::make_unique<TensorQTable>();  // this drops when drop == true
    // end of the current epoch, batch_num should start from 0 again
    batch_num = 0;
    epoch_num++;
    RETURN_IF_NOT_OK(
      worker_queues_[cnt++ % num_workers_]->EmplaceBack(std::make_pair(nullptr, CBatchInfo(batchCtrl::kEOE))));
    RETURN_IF_NOT_OK(GetBatchSize(&cur_batch_size, CBatchInfo(epoch_num, batch_num, cnt - epoch_num)));
    RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));

#if !defined(_WIN32) && !defined(_WIN64) && ENABLE_PYTHON
    if ((num_workers_ > 1 || batch_map_func_) && GetMemoryUsage() > MAX_MEMORY_USAGE_THRESHOLD) {
      MS_LOG(WARNING) << "Memory consumption is more than " << MAX_MEMORY_USAGE_THRESHOLD * 100 << "%, "
                      << "which may cause oom error. Please reduce num_parallel_workers size / "
                      << "optimize per_batch_map function / other python data preprocess function to "
                      << "reduce memory usage.";
    }
#endif
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
    row.setPath({});
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
      RETURN_IF_NOT_OK(Tensor::CreateEmpty(new_shape, first_type, &new_tensor));
      dsize_t j = 0;
      for (auto row : **src) {
        std::shared_ptr<Tensor> old_tensor = row.at(i);  // row j, column i
        if (old_tensor->shape() == first_shape) {        // check the newly popped rows have the same dim as the first
          if (new_shape.NumOfElements() != 0) {
            RETURN_IF_NOT_OK(new_tensor->InsertTensor({j++}, old_tensor));
          }
          // Don't do anything if the tensor has no data
        } else {
          RETURN_STATUS_UNEXPECTED(
            "Invalid data, expect same shape for each data row, but got inconsistent data shapes in column " +
            std::to_string(i));
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
      RETURN_IF_NOT_OK(Tensor::CreateFromVector(strings, new_shape, &new_tensor));
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
#ifdef ENABLE_PYTHON
  if (!in_col_names_.empty()) RETURN_IF_NOT_OK(MapColumns(&table_pair));  // pass it through pyfunc
#endif
  if (pad_) RETURN_IF_NOT_OK(PadColumns(&table_pair.first, pad_info_, column_name_id_map_));  // do padding if needed
  (*db) = std::make_unique<DataBuffer>(table_pair.second.batch_num_, DataBuffer::kDeBFlagNone);
  std::unique_ptr<TensorQTable> dest_table = std::make_unique<TensorQTable>();
  RETURN_IF_NOT_OK(BatchRows(&table_pair.first, &dest_table, table_pair.first->size()));
  (*db)->set_tensor_table(std::move(dest_table));
  return Status::OK();
}

Status BatchOp::LaunchThreadsAndInitOp() {
  if (tree_ == nullptr) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, "Pipeline init failed, Execution tree not set.");
  }
  RETURN_IF_NOT_OK(worker_queues_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(
    tree_->LaunchWorkers(num_workers_, std::bind(&BatchOp::WorkerEntry, this, std::placeholders::_1), Name(), id()));
  return Status::OK();
}

Status BatchOp::EofReceived(int32_t) { return Status::OK(); }

Status BatchOp::EoeReceived(int32_t) {
  state_ = OpState::kDeOpIdle;
  return Status::OK();
}

#ifdef ENABLE_PYTHON
Status BatchOp::MapColumns(std::pair<std::unique_ptr<TensorQTable>, CBatchInfo> *table_pair) {
  std::unique_ptr<TensorQTable> in_q_table = std::move(table_pair->first);
  size_t num_rows = in_q_table->size();
  auto out_q_table = std::make_unique<TensorQTable>(num_rows, TensorRow(column_name_id_map_.size(), nullptr));
  TensorTable in_cols(in_col_names_.size(), TensorRow(num_rows, nullptr)), out_cols;

  std::unordered_map<std::string, size_t> in_col_name_id;  // name of columns that need to be fed to per-batch_map
  for (size_t i = 0; i < in_col_names_.size(); i++) in_col_name_id.insert({in_col_names_[i], i});

  for (const auto &itr : child_map_) {
    auto col_itr = in_col_name_id.find(itr.first);
    if (col_itr != in_col_name_id.end()) {  // col needs to be prepared for per_batch_map
      for (size_t i = 0; i < num_rows; i++) {
        in_cols[col_itr->second][i] = std::move((*in_q_table)[i][itr.second]);
      }
    } else {  // col needs to be placed into the out table
      size_t col_id = column_name_id_map_[itr.first];
      for (size_t i = 0; i < num_rows; i++) {
        (*out_q_table)[i][col_id] = std::move((*in_q_table)[i][itr.second]);
      }
    }
  }

  in_q_table.reset();  // release the input table
  RETURN_IF_NOT_OK(InvokeBatchMapFunc(&in_cols, &out_cols, table_pair->second));

  for (size_t i = 0; i < out_cols.size(); i++) {
    size_t col_id = column_name_id_map_[out_col_names_[i]];
    size_t row_id = 0;
    CHECK_FAIL_RETURN_UNEXPECTED(num_rows == out_cols[i].size(),
                                 "column: " + out_col_names_[i] + " expects: " + std::to_string(num_rows) +
                                   " rows returned from per_batch_map, gets: " + std::to_string(out_cols[i].size()));
    for (auto &t_row : *out_q_table) {
      t_row[col_id] = out_cols[i][row_id++];
    }
  }

  table_pair->first = std::move(out_q_table);
  return Status::OK();
}
#endif

Status BatchOp::GetBatchSize(int32_t *batch_size, CBatchInfo info) {
#ifdef ENABLE_PYTHON
  if (batch_size_func_) {
    RETURN_IF_NOT_OK(InvokeBatchSizeFunc(batch_size, info));
  } else {
    (*batch_size) = start_batch_size_;
  }
#else
  (*batch_size) = start_batch_size_;
#endif
  return Status::OK();
}

#ifdef ENABLE_PYTHON
Status BatchOp::InvokeBatchSizeFunc(int32_t *batch_size, CBatchInfo info) {
  {
    // Acquire Python GIL
    py::gil_scoped_acquire gil_acquire;
    if (Py_IsInitialized() == 0) {
      return Status(StatusCode::kMDPythonInterpreterFailure, "Python Interpreter is finalized.");
    }
    try {
      py::object size = batch_size_func_(info);
      *batch_size = size.cast<int32_t>();
      if (*batch_size <= 0) {
        return Status(StatusCode::kMDPyFuncException,
                      "Invalid parameter, batch size function should return an integer greater than 0.");
      }
    } catch (const py::error_already_set &e) {
      return Status(StatusCode::kMDPyFuncException, e.what());
    } catch (const py::cast_error &e) {
      return Status(StatusCode::kMDPyFuncException,
                    "Invalid parameter, batch size function should return an integer greater than 0.");
    }
  }
  return Status(StatusCode::kSuccess, "Batch size func call succeed.");
}

Status BatchOp::InvokeBatchMapFunc(TensorTable *input, TensorTable *output, CBatchInfo info) {
  {
    // Acquire Python GIL
    py::gil_scoped_acquire gil_acquire;
    if (Py_IsInitialized() == 0) {
      return Status(StatusCode::kMDPythonInterpreterFailure, "Python Interpreter is finalized.");
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
      CHECK_FAIL_RETURN_UNEXPECTED(py::isinstance<py::tuple>(ret_tuple), "Batch map function should return a tuple.");
      CHECK_FAIL_RETURN_UNEXPECTED(
        ret_tuple.size() == out_col_names_.size(),
        "Incorrect number of columns returned. expects: " + std::to_string(out_col_names_.size()) +
          " gets: " + std::to_string(ret_tuple.size()));
      for (size_t i = 0; i < ret_tuple.size(); i++) {
        TensorRow output_batch;
        // If user returns a type that is neither a list nor an array, issue a error msg.
        if (!py::isinstance<py::list>(ret_tuple[i])) {
          MS_LOG(INFO) << "column: " << out_col_names_[i]
                       << " returned by per_batch_map is not a list, this could lead to conversion failure.";
        }

        py::list output_list = py::cast<py::list>(ret_tuple[i]);

        for (size_t j = 0; j < output_list.size(); j++) {
          std::shared_ptr<Tensor> out;
          RETURN_IF_NOT_OK(Tensor::CreateFromNpArray(py::cast<py::array>(output_list[j]), &out));
          output_batch.push_back(std::move(out));
        }
        output->push_back(std::move(output_batch));
      }
    } catch (const py::error_already_set &e) {
      return Status(StatusCode::kMDPyFuncException, e.what());
    } catch (const py::cast_error &e) {
      return Status(StatusCode::kMDPyFuncException,
                    "Invalid parameter, batch map function should return a tuple of list of numpy array.");
    }
  }
  return Status::OK();
}
#endif

Status BatchOp::PadColumns(std::unique_ptr<TensorQTable> *table, const PadInfo &pad_info,
                           const std::unordered_map<std::string, int32_t> &column_name_id_map) {
  RETURN_UNEXPECTED_IF_NULL(table);  // placeholder for now, might need this in the future
  CHECK_FAIL_RETURN_UNEXPECTED(
    (*table)->front().size() == column_name_id_map.size(),
    "Invalid parameter, size of column_name_id_map must be equal to num of data columns. map size: " +
      std::to_string(column_name_id_map.size()) + ", column nums: " + std::to_string((*table)->front().size()));
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
    CHECK_FAIL_RETURN_UNEXPECTED(
      pad_shapes[col_id].size() == max_shapes[col_id].size(),
      "Invalid data, rank of pad_shape must be equal to rank of specified column. pad_shapes rank:" +
        std::to_string(pad_shapes[col_id].size()) + ", column rank: " + std::to_string(max_shapes[col_id].size()));
  }

  // calculate maximum shape for each column that needs to be padded
  for (const TensorRow &row : **table) {  // iterator each row in a batch
    for (size_t col_id : pad_cols) {      // iterator each tensor in a row
      CHECK_FAIL_RETURN_UNEXPECTED(
        row[col_id]->Rank() == max_shapes[col_id].size(),
        "Invalid data, data to be padded together need to have the same rank, got shape 1: " +
          std::to_string(row[col_id]->Rank()) + ", shape 2: " + std::to_string(max_shapes[col_id].size()));
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
      CHECK_FAIL_RETURN_UNEXPECTED(location != column_name_id_map.end(),
                                   "Invalid parameter, column name: " + p.first + " does not exist.");
      auto col_id = static_cast<dsize_t>(location->second);
      CHECK_FAIL_RETURN_UNEXPECTED(
        col_id < pad_vals->size() && col_id < pad_shapes->size(),
        "Invalid parameter, column id must be less than the size of pad_val and pad_shape, but got: " +
          std::to_string(col_id));
      pad_cols->insert(col_id);
      (*pad_vals)[col_id] = p.second.second;              // set pad values
      (*pad_shapes)[col_id] = p.second.first.AsVector();  // empty vector if shape is unknown
    }
  }
  return Status::OK();
}

Status BatchOp::ComputeColMap() {
  CHECK_FAIL_RETURN_UNEXPECTED(child_.size() == 1,
                               "Batch has " + std::to_string(child_.size()) + " child/children, expects only 1 child.");
  CHECK_FAIL_RETURN_UNEXPECTED(!(child_[0]->column_name_id_map().empty()), "BatchOp child map is empty.");

  if (in_col_names_.empty()) {  // if per_batch_map is not set, do not need to deal with out_col_names
    column_name_id_map_ = child_[0]->column_name_id_map();
    return Status::OK();
  }

  // from this point onward, per_batch_map is needed, therefore, child_map_ must be set
  child_map_ = child_[0]->column_name_id_map();

  // check all input columns exist
  for (const auto &col : in_col_names_) {
    CHECK_FAIL_RETURN_UNEXPECTED(child_map_.find(col) != child_map_.end(), "col:" + col + " doesn't exist.");
  }

  // following logic deals with per_batch_map
  bool col_name_flag = (out_col_names_.empty() || out_col_names_ == in_col_names_);  // true if col name is unchanged

  // column names are unchanged
  if (col_name_flag) {
    if (out_col_names_.empty()) out_col_names_ = in_col_names_;
    column_name_id_map_ = child_map_;
    return Status::OK();
  }
  // column names are changed from this point onward, this map is the child_map without input cols for per_batch_map
  auto child_map_no_in_col = child_map_;

  for (const auto &col : in_col_names_) {
    child_map_no_in_col.erase(col);
  }

  // col names are changed
  if (out_col_names_.size() == in_col_names_.size()) {  // column names changed, but same number of columns
    // the following code rename the input keys to output keys. ["a","b"] -> ["b", "a"] is allowed
    column_name_id_map_ = child_map_no_in_col;
    for (auto i = 0; i < in_col_names_.size(); i++) {
      column_name_id_map_[out_col_names_[i]] = child_map_[in_col_names_[i]];
    }
  } else {  // number of columns are different, put the output column names first, then the original ones
    for (const std::string &col : out_col_names_) {
      column_name_id_map_.insert({col, column_name_id_map_.size()});
    }
    for (const auto &itr : child_map_no_in_col) {
      column_name_id_map_.insert({itr.first, column_name_id_map_.size()});
    }
  }

  CHECK_FAIL_RETURN_UNEXPECTED(column_name_id_map_.size() == (child_map_no_in_col.size() + out_col_names_.size()),
                               "Key error in column_name_id_map_. output_columns is NOT set correctly!");
  return Status::OK();
}

int64_t BatchOp::GetTreeBatchSize() {
#ifdef ENABLE_PYTHON
  if (batch_size_func_) {
    return -1;
  }
#endif
  return start_batch_size_;
}

Status BatchOp::GetNextRow(TensorRow *const row) {
  std::unique_ptr<TensorQTable> table = std::make_unique<TensorQTable>();
  child_iterator_ = std::make_unique<ChildIterator>(this, 0, 0);
  int32_t cur_batch_size = 0;
  RETURN_IF_NOT_OK(GetBatchSize(&cur_batch_size, CBatchInfo(0, batch_num_, batch_cnt_)));
  for (int i = 0; i < cur_batch_size; i++) {
    TensorRow new_row;
    RETURN_IF_NOT_OK(child_[0]->GetNextRow(&new_row));
    if (!new_row.empty()) {
      table->emplace_back(new_row);
      if (table->size() == static_cast<size_t>(cur_batch_size)) break;
    } else {
      if (drop_ || table->empty()) {
        table = std::make_unique<TensorQTable>();  // this drops when drop == true
      }
    }
  }
  std::unique_ptr<TensorQTable> out = std::make_unique<TensorQTable>();
  RETURN_UNEXPECTED_IF_NULL(table);
  if (pad_) RETURN_IF_NOT_OK(PadColumns(&table, pad_info_, column_name_id_map_));  // do padding if needed
  if (!table->empty()) {
    RETURN_IF_NOT_OK(BatchRows(&table, &out, table->size()));
    CHECK_FAIL_RETURN_UNEXPECTED(out->size() == 1, "Batch returned 2 rows while 1 row was expected.");
    *row = out->back();
    batch_cnt_++;
    batch_num_++;
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
