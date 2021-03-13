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
#include "minddata/dataset/engine/datasetops/source/mindrecord_op.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <utility>

#include "utils/ms_utils.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/engine/data_buffer.h"
#include "minddata/dataset/engine/datasetops/dataset_op.h"
#include "minddata/dataset/engine/db_connector.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/util/log_adapter.h"

namespace mindspore {
namespace dataset {

using mindrecord::kInt64Len;
using mindrecord::MSRStatus;
using mindrecord::Schema;
using mindrecord::ShardOperator;
using mindrecord::ShardReader;

// Builder constructor.  Creates the builder object.
MindRecordOp::Builder::Builder() : build_dataset_file_({}) {
  // Some arguments to the MindRecordOp constructor have a default argument that is taken
  // from the client config.
  // The user may choose to change these values for the construction of the MindRecordOp by
  // using the various builder set methods.

  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  build_num_mind_record_workers_ = kDefaultMindRecordWorkers;
  build_rows_per_buffer_ = cfg->rows_per_buffer();
  build_op_connector_queue_size_ = cfg->op_connector_size();
  builder_num_workers_ = 0;
  build_load_dataset_ = false;
  build_num_padded_ = 0;
  build_sample_ = nullptr;
}

// The builder "build" method creates the final object.
Status MindRecordOp::Builder::Build(std::shared_ptr<MindRecordOp> *ptr) {
  std::shared_ptr<MindRecordOp> new_mind_record_op;

  if (build_dataset_file_.empty()) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                  "Invalid file, MindRecord path is invalid or not set.");
  }
  mindrecord::json sample_json;
  if (build_num_padded_ > 0) {
    sample_json = ToJson(build_sample_);
  }
  new_mind_record_op =
    std::make_shared<MindRecordOp>(build_num_mind_record_workers_, build_rows_per_buffer_, build_dataset_file_,
                                   build_load_dataset_, build_op_connector_queue_size_, build_columns_to_load_,
                                   build_operators_, build_num_padded_, sample_json, build_sample_bytes_);

  RETURN_IF_NOT_OK(new_mind_record_op->Init());
  *ptr = std::move(new_mind_record_op);
  return Status::OK();
}

Status MindRecordOp::Builder::SanityCheck() const { return Status::OK(); }

mindrecord::json MindRecordOp::Builder::ToJson(const py::handle &obj) {
  if (obj.is_none()) {
    return nullptr;
  }
  if (py::isinstance<py::int_>(obj)) {
    return obj.cast<int64_t>();
  }
  if (py::isinstance<py::float_>(obj)) {
    return obj.cast<double>();
  }
  if (py::isinstance<py::str>(obj)) {  // also catch py::bytes
    return obj.cast<std::string>();
  }
  if (py::isinstance<py::dict>(obj)) {
    auto out = mindrecord::json::object();
    for (const py::handle &key : obj) {
      if (py::isinstance<py::bytes>(obj[key])) {
        build_sample_bytes_[py::str(key).cast<std::string>()] = obj[key].cast<std::string>();
      } else {
        out[py::str(key).cast<std::string>()] = ToJson(obj[key]);
      }
    }
    return out;
  }
  MS_LOG(ERROR) << "Python object convert to json failed, object is: " << py::cast<std::string>(obj);
  return mindrecord::json();
}

// Constructor of the MindRecordOp.
MindRecordOp::MindRecordOp(int32_t num_mind_record_workers, int32_t rows_per_buffer,
                           std::vector<std::string> dataset_file, bool load_dataset, int32_t op_connector_queue_size,
                           const std::vector<std::string> &columns_to_load,
                           const std::vector<std::shared_ptr<ShardOperator>> &operators, int64_t num_padded,
                           const mindrecord::json &sample_json, const std::map<std::string, std::string> &sample_bytes)
    : ParallelOp(num_mind_record_workers, op_connector_queue_size),
      rows_per_buffer_(rows_per_buffer),
      dataset_file_(dataset_file),
      load_dataset_(load_dataset),
      columns_to_load_(columns_to_load),
      operators_(operators),
      num_mind_record_workers_(num_mind_record_workers),
      num_rows_(0),
      buffers_needed_(0),
      buf_cnt_(0),
      ended_worker_(0),
      num_padded_(num_padded),
      sample_json_(sample_json),
      sample_bytes_(sample_bytes) {
  io_block_queues_.Init(num_workers_, op_connector_queue_size);
  epoch_sync_flag_ = true;  // MindRecordOp needs to turn this flag on, otherwise, calling ShuffleTask() before all
                            // tasks are consumed by the worker threads would cause problem.
}

// Private helper method to encapsulate some common construction/reset tasks
Status MindRecordOp::Init() {
  shard_reader_ = std::make_unique<ShardReader>();
  auto rc = shard_reader_->Open(dataset_file_, load_dataset_, num_mind_record_workers_, columns_to_load_, operators_,
                                num_padded_);

  CHECK_FAIL_RETURN_UNEXPECTED(rc == MSRStatus::SUCCESS, "MindRecordOp init failed, " + ErrnoToMessage(rc));

  data_schema_ = std::make_unique<DataSchema>();

  std::vector<std::string> col_names = shard_reader_->GetShardColumn()->GetColumnName();
  CHECK_FAIL_RETURN_UNEXPECTED(!col_names.empty(), "Invalid data, no column names are specified.");
  std::vector<mindrecord::ColumnDataType> col_data_types = shard_reader_->GetShardColumn()->GeColumnDataType();
  std::vector<std::vector<int64_t>> col_shapes = shard_reader_->GetShardColumn()->GetColumnShape();

  bool load_all_cols = columns_to_load_.empty();  // if columns_to_load_ is empty it means load everything
  std::map<std::string, int32_t> colname_to_ind;
  for (uint32_t i = 0; i < col_names.size(); i++) {
    std::string colname = col_names[i];
    ColDescriptor col_desc;

    TensorShape t_shape = TensorShape::CreateUnknownRankShape();  // shape of tensor, default unknown
    std::string type_str = mindrecord::ColumnDataTypeNameNormalized[col_data_types[i]];
    DataType t_dtype = DataType(type_str);  // valid types: {"bytes", "string", "int32", "int64", "float32", "float64"}

    if (col_data_types[i] == mindrecord::ColumnBytes) {  // rank = 1
      col_desc = ColDescriptor(colname, t_dtype, TensorImpl::kFlexible, 1);
    } else if (col_data_types[i] == mindrecord::ColumnString) {  // rank = 0
      col_desc = ColDescriptor(colname, t_dtype, TensorImpl::kFlexible, 0);
    } else if (col_shapes[i].size() > 0) {
      std::vector<dsize_t> vec(col_shapes[i].size());  // temporary vector to hold shape
      (void)std::copy(col_shapes[i].begin(), col_shapes[i].end(), vec.begin());
      t_shape = TensorShape(vec);
      col_desc = ColDescriptor(colname, t_dtype, TensorImpl::kFlexible, t_shape.Rank(), &t_shape);
    } else {  // unknown shape
      // create colDesc and add it to schema
      col_desc = ColDescriptor(colname, t_dtype, TensorImpl::kFlexible, t_shape.Rank(), &t_shape);
    }

    colname_to_ind[colname] = data_schema_->NumColumns();
    RETURN_IF_NOT_OK(data_schema_->AddColumn(col_desc));

    if (load_all_cols) {
      columns_to_load_.emplace_back(colname);
    }
  }

  if (!load_all_cols) {
    std::unique_ptr<DataSchema> tmp_schema = std::make_unique<DataSchema>();
    for (std::string colname : columns_to_load_) {
      CHECK_FAIL_RETURN_UNEXPECTED(colname_to_ind.find(colname) != colname_to_ind.end(),
                                   "Invalid parameter, column name: " + colname + " does not exist.");
      RETURN_IF_NOT_OK(tmp_schema->AddColumn(data_schema_->column(colname_to_ind[colname])));
    }
    data_schema_ = std::move(tmp_schema);
  }

  return Status::OK();
}

// Destructor
MindRecordOp::~MindRecordOp() {}

// A print method typically used for debugging
void MindRecordOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nDataset file : ";
    for (auto &file : dataset_file_) {
      out << file << " ";
    }
    out << "\nNumber of rows : " << num_rows_ << "\nRows per buffer : " << rows_per_buffer_
        << "\nNumber of buffers : " << buffers_needed_
        << "\nNumber of ShardReader workers : " << num_mind_record_workers_ << "\n\n";
  }
}

Status MindRecordOp::WorkerEntry(int32_t worker_id) {
  TaskManager::FindMe()->Post();
  std::unique_ptr<IOBlock> io_block;
  RETURN_IF_NOT_OK(io_block_queues_[worker_id]->PopFront(&io_block));
  while (io_block != nullptr) {
    if (io_block->wait()) {
      // Sync io_block is a signal that master thread wants us to pause and sync with other workers.
      // The last guy who comes to this sync point should reset the counter and wake up the master thread.
      if (++num_workers_paused_ == num_workers_) {
        wait_for_workers_post_.Set();
      }
      RETURN_IF_NOT_OK(io_block_queues_[worker_id]->PopFront(&io_block));
      continue;
    }
    if (io_block->eoe()) {
      RETURN_IF_NOT_OK(
        out_connector_->Add(worker_id, std::move(std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE))));
      RETURN_IF_NOT_OK(io_block_queues_[worker_id]->PopFront(&io_block));
      continue;
    }
    if (io_block->eof()) {
      RETURN_IF_NOT_OK(
        out_connector_->Add(worker_id, std::move(std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOF))));
      RETURN_IF_NOT_OK(io_block_queues_[worker_id]->PopFront(&io_block));
      continue;
    }

    // load data buffer
    std::vector<int64_t> keys;
    RETURN_IF_NOT_OK(io_block->GetKeys(&keys));
    if (keys.empty() == true) {
      {
        std::unique_lock<std::mutex> lock(ended_worker_mutex_);
        ended_worker_++;
        if (ended_worker_ == num_workers_) shard_reader_->Close();
      }
      return Status::OK();  // empty key is a quit signal for workers
    }

    const uint64_t buffer_id = keys[0];
    std::unique_ptr<DataBuffer> fetched_buffer;

    // Get the next buffer. Push it up to the output connector.
    if (buffer_id % LOG_INTERVAL == 0) {
      MS_LOG(DEBUG) << "MindRecord operator consumed buffer " << buffer_id << " by worker " << worker_id << ".";
    }
    RETURN_IF_NOT_OK(GetBufferFromReader(&fetched_buffer, buffer_id, worker_id));
    RETURN_IF_NOT_OK(out_connector_->Add(worker_id, std::move(fetched_buffer)));
    RETURN_IF_NOT_OK(io_block_queues_[worker_id]->PopFront(&io_block));
  }
  RETURN_STATUS_UNEXPECTED("Unexpected nullptr received in worker.");
}

Status MindRecordOp::GetBufferFromReader(std::unique_ptr<DataBuffer> *fetched_buffer, int64_t buffer_id,
                                         int32_t worker_id) {
  *fetched_buffer = std::make_unique<DataBuffer>(buffer_id, DataBuffer::kDeBFlagNone);
  std::unique_ptr<TensorQTable> tensor_table = std::make_unique<TensorQTable>();
  for (int32_t i = 0; i < rows_per_buffer_; ++i) {
    int32_t row_id = buffer_id * rows_per_buffer_ + i;
    auto rc = shard_reader_->GetNextById(row_id, worker_id);
    auto task_type = rc.first;
    auto tupled_buffer = rc.second;
    if (task_type == mindrecord::TaskType::kPaddedTask) {
      TensorRow tensor_row;
      RETURN_IF_NOT_OK(LoadTensorRow(&tensor_row, {}, mindrecord::json(), task_type));
      std::vector<std::string> file_path(tensor_row.size(), dataset_file_[0]);
      tensor_row.setPath(file_path);
      tensor_table->push_back(std::move(tensor_row));
    }
    if (tupled_buffer.empty()) break;
    if (task_type == mindrecord::TaskType::kCommonTask) {
      for (const auto &tupled_row : tupled_buffer) {
        std::vector<uint8_t> columns_blob = std::get<0>(tupled_row);
        mindrecord::json columns_json = std::get<1>(tupled_row);
        TensorRow tensor_row;
        RETURN_IF_NOT_OK(LoadTensorRow(&tensor_row, columns_blob, columns_json, task_type));
        std::vector<std::string> file_path(tensor_row.size(), dataset_file_[0]);
        tensor_row.setPath(file_path);
        tensor_table->push_back(std::move(tensor_row));
      }
    }
  }

  // Replace the TensorTable in DataBuffer with the new one.
  (*fetched_buffer)->set_tensor_table(std::move(tensor_table));
  return Status::OK();
}

Status MindRecordOp::LoadTensorRow(TensorRow *tensor_row, const std::vector<uint8_t> &columns_blob,
                                   const mindrecord::json &columns_json, const mindrecord::TaskType task_type) {
  for (uint32_t i_col = 0; i_col < columns_to_load_.size(); i_col++) {
    auto column_name = columns_to_load_[i_col];

    // Initialize column parameters
    const unsigned char *data = nullptr;
    std::unique_ptr<unsigned char[]> data_ptr;
    uint64_t n_bytes = 0;
    mindrecord::ColumnDataType column_data_type = mindrecord::ColumnNoDataType;
    uint64_t column_data_type_size = 1;
    std::vector<int64_t> column_shape;

    // Get column data
    auto shard_column = shard_reader_->GetShardColumn();
    if (num_padded_ > 0 && task_type == mindrecord::TaskType::kPaddedTask) {
      auto rc =
        shard_column->GetColumnTypeByName(column_name, &column_data_type, &column_data_type_size, &column_shape);
      if (rc.first != MSRStatus::SUCCESS) {
        RETURN_STATUS_UNEXPECTED("Invalid parameter, column_name: " + column_name + "does not exist in dataset.");
      }
      if (rc.second == mindrecord::ColumnInRaw) {
        auto has_column = shard_column->GetColumnFromJson(column_name, sample_json_, &data_ptr, &n_bytes);
        if (has_column == MSRStatus::FAILED) {
          RETURN_STATUS_UNEXPECTED("Invalid data, failed to retrieve raw data from padding sample.");
        }
      } else if (rc.second == mindrecord::ColumnInBlob) {
        if (sample_bytes_.find(column_name) == sample_bytes_.end()) {
          RETURN_STATUS_UNEXPECTED("Invalid data, failed to retrieve blob data from padding sample.");
        }
        std::string ss(sample_bytes_[column_name]);
        n_bytes = ss.size();
        data_ptr = std::make_unique<unsigned char[]>(n_bytes);
        std::copy(ss.begin(), ss.end(), data_ptr.get());
      } else {
        RETURN_STATUS_UNEXPECTED("Invalid data, retrieved data type is unknown.");
      }
      if (data == nullptr) {
        data = reinterpret_cast<const unsigned char *>(data_ptr.get());
      }
    } else {
      auto has_column =
        shard_column->GetColumnValueByName(column_name, columns_blob, columns_json, &data, &data_ptr, &n_bytes,
                                           &column_data_type, &column_data_type_size, &column_shape);
      if (has_column == MSRStatus::FAILED) {
        RETURN_STATUS_UNEXPECTED("Invalid data, failed to retrieve data from mindrecord reader.");
      }
    }

    std::shared_ptr<Tensor> tensor;
    const ColDescriptor &column = data_schema_->column(i_col);
    DataType type = column.type();

    // Set shape
    CHECK_FAIL_RETURN_UNEXPECTED(column_data_type_size != 0, "The divisor cannot be 0.");
    auto num_elements = n_bytes / column_data_type_size;
    if (type == DataType::DE_STRING) {
      std::string s{data, data + n_bytes};
      RETURN_IF_NOT_OK(Tensor::CreateScalar(s, &tensor));
    } else if (column.hasShape()) {
      auto new_shape = TensorShape(column.shape());
      RETURN_IF_NOT_OK(column.MaterializeTensorShape(static_cast<int32_t>(num_elements), &new_shape));
      RETURN_IF_NOT_OK(Tensor::CreateFromMemory(new_shape, type, data, &tensor));
    } else {
      std::vector<dsize_t> shapeDetails = {static_cast<dsize_t>(num_elements)};
      auto new_shape = TensorShape(shapeDetails);
      RETURN_IF_NOT_OK(Tensor::CreateFromMemory(new_shape, type, data, &tensor));
    }
    tensor_row->push_back(std::move(tensor));
  }
  return Status::OK();
}

// Class functor operator () override.
// All dataset ops operate by launching a thread (see ExecutionTree). This class functor will
// provide the master loop that drives the logic for performing the work
// Main logic, Register Queue with TaskGroup, launch all threads and do the functor's work
Status MindRecordOp::operator()() {
  RETURN_IF_NOT_OK(LaunchThreadAndInitOp());
  num_rows_ = shard_reader_->GetNumRows();
  // Compute how many buffers we would need to accomplish rowsPerBuffer
  buffers_needed_ = (num_rows_ + rows_per_buffer_ - 1) / rows_per_buffer_;

  while (true) {  // each iterator is 1 epoch
    for (int32_t i = 0; i < buffers_needed_; ++i) {
      std::vector<int64_t> keys(1, i);
      RETURN_IF_NOT_OK(io_block_queues_[buf_cnt_++ % num_workers_]->Add(
        std::make_unique<IOBlock>(IOBlock(keys, IOBlock::kDeIoBlockNone))));
    }
    if (IsLastIteration()) {
      RETURN_IF_NOT_OK(
        io_block_queues_[(buf_cnt_++) % num_workers_]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEoe)));
      RETURN_IF_NOT_OK(
        io_block_queues_[(buf_cnt_++) % num_workers_]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEof)));
      for (int32_t i = 0; i < num_workers_; i++) {
        RETURN_IF_NOT_OK(io_block_queues_[i]->Add(
          std::move(std::make_unique<IOBlock>(std::vector<int64_t>(), IOBlock::kDeIoBlockNone))));
      }
      return Status::OK();
    } else {
      RETURN_IF_NOT_OK(
        io_block_queues_[(buf_cnt_++) % num_workers_]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEoe)));
    }

    if (epoch_sync_flag_) {
      // If epoch_sync_flag_ is set, then master thread sleeps until all the worker threads have finished their job for
      // the current epoch.
      RETURN_IF_NOT_OK(WaitForWorkers());
    }
    // If not the last repeat, self-reset and go to loop again.
    if (!IsLastIteration()) RETURN_IF_NOT_OK(Reset());
    UpdateRepeatAndEpochCounter();
  }
}

// Overrides base class reset method.  When an operator does a reset, it cleans up any state
// info from it's previous execution and then initializes itself so that it can be executed
// again.
Status MindRecordOp::Reset() {
  MS_LOG(DEBUG) << Name() << " performing a self-reset.";
  RETURN_IF_NOT_OK(ParallelOp::Reset());  // Call our super class reset first.

  shard_reader_->ShuffleTask();

  return Status::OK();
}

Status MindRecordOp::LaunchThreadAndInitOp() {
  if (tree_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("Pipeline init failed, Execution tree not set.");
  }

  RETURN_IF_NOT_OK(io_block_queues_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(wait_for_workers_post_.Register(tree_->AllTasks()));
  if (shard_reader_->Launch(true) == MSRStatus::FAILED) {
    RETURN_STATUS_UNEXPECTED("MindRecordOp launch failed.");
  }
  // Launch main workers that load DataBuffers by reading all images
  RETURN_IF_NOT_OK(
    tree_->LaunchWorkers(num_workers_, std::bind(&MindRecordOp::WorkerEntry, this, std::placeholders::_1), "", id()));
  TaskManager::FindMe()->Post();
  return Status::OK();
}

Status MindRecordOp::CountTotalRows(const std::vector<std::string> dataset_path, bool load_dataset,
                                    const std::shared_ptr<ShardOperator> &op, int64_t *count, int64_t num_padded) {
  std::unique_ptr<ShardReader> shard_reader = std::make_unique<ShardReader>();
  MSRStatus rc = shard_reader->CountTotalRows(dataset_path, load_dataset, op, count, num_padded);
  if (rc == MSRStatus::FAILED) {
    RETURN_STATUS_UNEXPECTED("Invalid data, MindRecordOp failed to count total rows.");
  }
  return Status::OK();
}

Status MindRecordOp::ComputeColMap() {
  if (column_name_id_map_.empty()) {
    for (int i = 0; i < static_cast<int>(columns_to_load_.size()); i++) {
      column_name_id_map_[columns_to_load_[i]] = i;
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
