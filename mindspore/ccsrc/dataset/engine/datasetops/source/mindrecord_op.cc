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
#include "dataset/engine/datasetops/source/mindrecord_op.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <utility>

#include "common/utils.h"
#include "dataset/core/config_manager.h"
#include "dataset/core/constants.h"
#include "dataset/core/global_context.h"
#include "dataset/engine/data_buffer.h"
#include "dataset/engine/datasetops/dataset_op.h"
#include "dataset/engine/db_connector.h"
#include "dataset/engine/execution_tree.h"
#include "dataset/engine/opt/pass.h"
#include "utils/log_adapter.h"

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
  // The user may choose to change these values for the construction of the StorageOp by
  // using the various builder set methods.

  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  build_num_mind_record_workers_ = kDefaultMindRecordWorkers;
  build_rows_per_buffer_ = cfg->rows_per_buffer();
  build_op_connector_queue_size_ = cfg->op_connector_size();
  build_block_reader_ = false;
  builder_num_workers_ = 0;
  build_num_padded_ = 0;
  build_sample_ = nullptr;
}

// The builder "build" method creates the final object.
Status MindRecordOp::Builder::Build(std::shared_ptr<MindRecordOp> *ptr) {
  std::shared_ptr<MindRecordOp> new_mind_record_op;

  if (build_dataset_file_.empty()) {
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__,
                  "Building a MindRecordOp that has not provided a file.");
  }
  mindrecord::json sample_json;
  if (build_num_padded_ > 0) {
    sample_json = ToJson(build_sample_);
  }
  new_mind_record_op = std::make_shared<MindRecordOp>(
    build_num_mind_record_workers_, build_rows_per_buffer_, build_dataset_file_, build_load_dataset_,
    build_op_connector_queue_size_, build_columns_to_load_, build_operators_, build_block_reader_, build_num_padded_,
    sample_json, build_sample_bytes_);

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
                           const std::vector<std::shared_ptr<ShardOperator>> &operators, const bool &block_reader,
                           int64_t num_padded, const mindrecord::json &sample_json,
                           const std::map<std::string, std::string> &sample_bytes)
    : ParallelOp(num_mind_record_workers, op_connector_queue_size),
      rows_per_buffer_(rows_per_buffer),
      dataset_file_(dataset_file),
      load_dataset_(load_dataset),
      columns_to_load_(columns_to_load),
      operators_(operators),
      num_mind_record_workers_(num_mind_record_workers),
      block_reader_(block_reader),
      buffers_needed_(0),
      buf_cnt_(0),
      ended_worker_(0),
      buffer_water_mark_(0),
      num_padded_(num_padded),
      sample_json_(sample_json),
      sample_bytes_(sample_bytes) {
  io_blk_queues_.Init(num_workers_, op_connector_queue_size);
  if (!block_reader_) return;
  for (int32_t i = 0; i < num_workers_; ++i) {
    block_buffer_.emplace_back(std::make_unique<std::vector<ShardTuple>>(std::vector<ShardTuple>{}));
  }
}

// Private helper method to encapsulate some common construction/reset tasks
Status MindRecordOp::Init() {
  shard_reader_ = std::make_unique<ShardReader>();
  auto rc = shard_reader_->Open(dataset_file_, load_dataset_, num_mind_record_workers_, columns_to_load_, operators_,
                                block_reader_, num_padded_);

  CHECK_FAIL_RETURN_UNEXPECTED(rc == MSRStatus::SUCCESS,
                               "MindRecordOp init failed. Error message: " + ErrnoToMessage(rc));

  data_schema_ = std::make_unique<DataSchema>();

  std::vector<std::string> col_names = shard_reader_->GetShardColumn()->GetColumnName();
  CHECK_FAIL_RETURN_UNEXPECTED(!col_names.empty(), "No schema found");
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
      CHECK_FAIL_RETURN_UNEXPECTED(colname_to_ind.find(colname) != colname_to_ind.end(), colname + ": doesn't exist");
      RETURN_IF_NOT_OK(tmp_schema->AddColumn(data_schema_->column(colname_to_ind[colname])));
    }
    data_schema_ = std::move(tmp_schema);
  }

  for (int i = 0; i < static_cast<int>(columns_to_load_.size()); i++) {
    column_name_id_map_[columns_to_load_[i]] = i;
  }

  return Status::OK();
}

// Destructor
MindRecordOp::~MindRecordOp() {}

// A print method typically used for debugging
void MindRecordOp::Print(std::ostream &out, bool show_all) const {
  // Always show the id and name as first line regardless if this summary or detailed print
  out << "(" << std::setw(2) << operator_id_ << ") <MindRecordOp>:";
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\n Dataset file : ";
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
  RETURN_IF_NOT_OK(io_blk_queues_[worker_id]->PopFront(&io_block));
  while (io_block != nullptr) {
    if (io_block->eoe()) {
      RETURN_IF_NOT_OK(
        out_connector_->Add(worker_id, std::move(std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE))));
      RETURN_IF_NOT_OK(io_blk_queues_[worker_id]->PopFront(&io_block));
      continue;
    }
    if (io_block->eof()) {
      RETURN_IF_NOT_OK(
        out_connector_->Add(worker_id, std::move(std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOF))));
      RETURN_IF_NOT_OK(io_blk_queues_[worker_id]->PopFront(&io_block));
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
    if (!block_reader_) {
      RETURN_IF_NOT_OK(io_blk_queues_[worker_id]->PopFront(&io_block));
      continue;
    }

    // update block-reader buffer
    block_buffer_[buffer_id % num_workers_]->clear();
    {
      std::unique_lock<std::mutex> lck(mtx_block_reader_);
      if (buffer_id == buffer_water_mark_) {
        buffer_water_mark_++;
        while (block_set_.count(buffer_water_mark_) > 0) (void)block_set_.erase(buffer_water_mark_++);
      } else {
        (void)block_set_.insert(buffer_id);
      }
    }
    cv_reader_.notify_one();
    RETURN_IF_NOT_OK(io_blk_queues_[worker_id]->PopFront(&io_block));
  }
  RETURN_STATUS_UNEXPECTED("Unexpected nullptr received in worker");
}

Status MindRecordOp::GetBufferFromReader(std::unique_ptr<DataBuffer> *fetched_buffer, int64_t buffer_id,
                                         int32_t worker_id) {
  *fetched_buffer = std::make_unique<DataBuffer>(buffer_id, DataBuffer::kDeBFlagNone);
  std::unique_ptr<TensorQTable> tensor_table = std::make_unique<TensorQTable>();
  for (int32_t i = 0; i < rows_per_buffer_; ++i) {
    ShardTuple tupled_buffer;
    mindrecord::TaskType task_type = mindrecord::TaskType::kCommonTask;
    if (block_reader_) {
      if (i >= block_buffer_[buffer_id % num_workers_]->size()) break;
      tupled_buffer = block_buffer_[buffer_id % num_workers_]->at(i);
    } else {
      int32_t row_id = buffer_id * rows_per_buffer_ + i;
      auto rc = shard_reader_->GetNextById(row_id, worker_id);
      task_type = rc.first;
      tupled_buffer = rc.second;
      if (task_type == mindrecord::TaskType::kPaddedTask) {
        TensorRow tensor_row;
        RETURN_IF_NOT_OK(LoadTensorRow(&tensor_row, {}, mindrecord::json(), task_type));
        tensor_table->push_back(std::move(tensor_row));
      }
      if (tupled_buffer.empty()) break;
    }
    if (task_type == mindrecord::TaskType::kCommonTask) {
      for (const auto &tupled_row : tupled_buffer) {
        std::vector<uint8_t> columns_blob = std::get<0>(tupled_row);
        mindrecord::json columns_json = std::get<1>(tupled_row);
        TensorRow tensor_row;
        RETURN_IF_NOT_OK(LoadTensorRow(&tensor_row, columns_blob, columns_json, task_type));
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
        RETURN_STATUS_UNEXPECTED("Failed to retrieve data type.");
      }
      if (rc.second == mindrecord::ColumnInRaw) {
        auto has_column = shard_column->GetColumnFromJson(column_name, sample_json_, &data_ptr, &n_bytes);
        if (has_column == MSRStatus::FAILED) {
          RETURN_STATUS_UNEXPECTED("Failed to retrieve raw data from padding sample.");
        }
      } else if (rc.second == mindrecord::ColumnInBlob) {
        if (sample_bytes_.find(column_name) == sample_bytes_.end()) {
          RETURN_STATUS_UNEXPECTED("Failed to retrieve blob data from padding sample.");
        }
        std::string ss(sample_bytes_[column_name]);
        n_bytes = ss.size();
        data_ptr = std::make_unique<unsigned char[]>(n_bytes);
        std::copy(ss.begin(), ss.end(), data_ptr.get());
      } else {
        RETURN_STATUS_UNEXPECTED("Retrieved data type is unknown.");
      }
      if (data == nullptr) {
        data = reinterpret_cast<const unsigned char *>(data_ptr.get());
      }
    } else {
      auto has_column =
        shard_column->GetColumnValueByName(column_name, columns_blob, columns_json, &data, &data_ptr, &n_bytes,
                                           &column_data_type, &column_data_type_size, &column_shape);
      if (has_column == MSRStatus::FAILED) {
        RETURN_STATUS_UNEXPECTED("Failed to retrieve data from mindrecord reader.");
      }
    }

    std::shared_ptr<Tensor> tensor;
    const ColDescriptor &column = data_schema_->column(i_col);
    DataType type = column.type();

    // Set shape
    auto num_elements = n_bytes / column_data_type_size;
    if (type == DataType::DE_STRING) {
      std::string s{data, data + n_bytes};
      RETURN_IF_NOT_OK(Tensor::CreateTensor(&tensor, {s}, TensorShape::CreateScalar()));
    } else if (column.hasShape()) {
      auto new_shape = TensorShape(column.shape());
      RETURN_IF_NOT_OK(column.MaterializeTensorShape(static_cast<int32_t>(num_elements), &new_shape));
      RETURN_IF_NOT_OK(Tensor::CreateTensor(&tensor, column.tensorImpl(), new_shape, type, data));
    } else {
      std::vector<dsize_t> shapeDetails = {static_cast<dsize_t>(num_elements)};
      auto new_shape = TensorShape(shapeDetails);
      RETURN_IF_NOT_OK(Tensor::CreateTensor(&tensor, column.tensorImpl(), new_shape, type, data));
    }
    tensor_row->push_back(std::move(tensor));
  }
  return Status::OK();
}

Status MindRecordOp::FetchBlockBuffer(const int32_t &buffer_id) {
  {
    std::unique_lock<std::mutex> lck(mtx_block_reader_);
    cv_reader_.wait(lck, [buffer_id, this] { return buffer_id < buffer_water_mark_ + num_workers_; });
  }
  for (int32_t i = 0; i < rows_per_buffer_; i++) {
    // Block reader does NOT care about argument
    auto rc = shard_reader_->GetNextById(i, i);
    ShardTuple tuple_buffer = rc.second;
    if (tuple_buffer.empty()) break;
    block_buffer_[buffer_id % num_workers_]->push_back(std::move(tuple_buffer));
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
      if (block_reader_) RETURN_IF_NOT_OK(FetchBlockBuffer(i));
      std::vector<int64_t> keys(1, i);
      RETURN_IF_NOT_OK(io_blk_queues_[buf_cnt_++ % num_workers_]->Add(
        std::make_unique<IOBlock>(IOBlock(keys, IOBlock::kDeIoBlockNone))));
    }
    if (!BitTest(op_ctrl_flags_, kDeOpRepeated) || BitTest(op_ctrl_flags_, kDeOpLastRepeat)) {
      RETURN_IF_NOT_OK(
        io_blk_queues_[(buf_cnt_++) % num_workers_]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEoe)));
      RETURN_IF_NOT_OK(
        io_blk_queues_[(buf_cnt_++) % num_workers_]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEof)));
      for (int32_t i = 0; i < num_workers_; i++) {
        RETURN_IF_NOT_OK(io_blk_queues_[i]->Add(
          std::move(std::make_unique<IOBlock>(std::vector<int64_t>(), IOBlock::kDeIoBlockNone))));
      }
      return Status::OK();
    } else {  // not the last repeat. Acquire lock, sleeps master thread, wait for the wake-up from reset
      RETURN_IF_NOT_OK(
        io_blk_queues_[(buf_cnt_++) % num_workers_]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEoe)));

      // reset our buffer count and go to loop again.
      RETURN_IF_NOT_OK(shard_reader_wait_post_.Wait());
      shard_reader_wait_post_.Clear();
    }
  }
}

// Overrides base class reset method.  When an operator does a reset, it cleans up any state
// info from it's previous execution and then initializes itself so that it can be executed
// again.
Status MindRecordOp::Reset() {
  RETURN_IF_NOT_OK(ParallelOp::Reset());  // Call our super class reset first.

  if (block_reader_) {
    shard_reader_->Reset();
    buffer_water_mark_ = 0;
  } else {
    shard_reader_->ShuffleTask();
  }
  shard_reader_wait_post_.Set();

  return Status::OK();
}

Status MindRecordOp::LaunchThreadAndInitOp() {
  if (tree_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("tree_ not set");
  }

  RETURN_IF_NOT_OK(io_blk_queues_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(shard_reader_wait_post_.Register(tree_->AllTasks()));
  if (shard_reader_->Launch(!block_reader_) == MSRStatus::FAILED) {
    RETURN_STATUS_UNEXPECTED("MindRecordOp launch failed.");
  }
  // Launch main workers that load DataBuffers by reading all images
  RETURN_IF_NOT_OK(
    tree_->LaunchWorkers(num_workers_, std::bind(&MindRecordOp::WorkerEntry, this, std::placeholders::_1)));
  TaskManager::FindMe()->Post();
  return Status::OK();
}

Status MindRecordOp::CountTotalRows(const std::vector<std::string> dataset_path, bool load_dataset,
                                    const std::shared_ptr<ShardOperator> &op, int64_t *count, int64_t num_padded) {
  std::unique_ptr<ShardReader> shard_reader = std::make_unique<ShardReader>();
  MSRStatus rc = shard_reader->CountTotalRows(dataset_path, load_dataset, op, count, num_padded);
  if (rc == MSRStatus::FAILED) {
    RETURN_STATUS_UNEXPECTED("MindRecordOp count total rows failed.");
  }
  return Status::OK();
}

// Visitor accept method for NodePass
Status MindRecordOp::Accept(NodePass *p, bool *modified) {
  // Downcast shared pointer then call visitor
  return p->RunOnNode(std::static_pointer_cast<MindRecordOp>(shared_from_this()), modified);
}
}  // namespace dataset
}  // namespace mindspore
