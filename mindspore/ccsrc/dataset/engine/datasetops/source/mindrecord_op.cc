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
#include "utils/log_adapter.h"

namespace mindspore {
namespace dataset {
using mindrecord::kInt64Len;
using mindrecord::MSRStatus;
using mindrecord::Schema;
using mindrecord::ShardOperator;
using mindrecord::ShardReader;

// Builder constructor.  Creates the builder object.
MindRecordOp::Builder::Builder() : build_dataset_file_("") {
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
}

// The builder "build" method creates the final object.
Status MindRecordOp::Builder::Build(std::shared_ptr<MindRecordOp> *ptr) {
  std::shared_ptr<MindRecordOp> new_mind_record_op;

  if (build_dataset_file_.empty()) {
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__,
                  "Building a MindRecordOp that has not provided a file.");
  }

  new_mind_record_op = std::make_shared<MindRecordOp>(build_num_mind_record_workers_, build_rows_per_buffer_,
                                                      build_dataset_file_, build_op_connector_queue_size_,
                                                      build_columns_to_load_, build_operators_, build_block_reader_);

  RETURN_IF_NOT_OK(new_mind_record_op->Init());

  *ptr = std::move(new_mind_record_op);
  return Status::OK();
}

Status MindRecordOp::Builder::SanityCheck() const { return Status::OK(); }

// Constructor of the MindRecordOp.
MindRecordOp::MindRecordOp(int32_t num_mind_record_workers, int32_t rows_per_buffer, std::string dataset_file,
                           int32_t op_connector_queue_size, const std::vector<std::string> &columns_to_load,
                           const std::vector<std::shared_ptr<ShardOperator>> &operators, const bool &block_reader)
    : ParallelOp(num_mind_record_workers, op_connector_queue_size),
      rows_per_buffer_(rows_per_buffer),
      dataset_file_(dataset_file),
      columns_to_load_(columns_to_load),
      operators_(operators),
      num_mind_record_workers_(num_mind_record_workers),
      block_reader_(block_reader),
      buffers_needed_(0),
      buf_cnt_(0),
      num_rows_(0),
      ended_worker_(0),
      buffer_water_mark_(0) {
  io_blk_queues_.Init(num_workers_, op_connector_queue_size);
  if (!block_reader_) return;
  for (int32_t i = 0; i < num_workers_; ++i) {
    block_buffer_.emplace_back(std::make_unique<std::vector<ShardTuple>>(std::vector<ShardTuple>{}));
  }
}

// Private helper method to encapsulate some common construction/reset tasks
Status MindRecordOp::Init() {
  shard_reader_ = std::make_unique<ShardReader>();
  auto rc = shard_reader_->Open(dataset_file_, num_mind_record_workers_, columns_to_load_, operators_, block_reader_);

  CHECK_FAIL_RETURN_UNEXPECTED(rc != MSRStatus::FAILED,
                               "MindRecordOp init failed. Error message: " + ErrnoToMessage(rc));

  data_schema_ = std::make_unique<DataSchema>();

  std::vector<std::shared_ptr<Schema>> schema_vec = shard_reader_->get_shard_header()->get_schemas();
  // check whether schema exists, if so use the first one
  CHECK_FAIL_RETURN_UNEXPECTED(!schema_vec.empty(), "No schema found");
  mindrecord::json mr_schema = schema_vec[0]->GetSchema()["schema"];

  bool load_all_cols = columns_to_load_.empty();  // if columns_to_load_ is empty it means load everything
  std::map<std::string, int32_t> colname_to_ind;
  for (mindrecord::json::iterator it = mr_schema.begin(); it != mr_schema.end(); ++it) {
    std::string colname = it.key();          // key of the json, column name
    mindrecord::json it_value = it.value();  // value, which contains type info and may contain shape
    ColDescriptor col_desc;
    TensorShape t_shape = TensorShape::CreateUnknownRankShape();  // shape of tensor, default unknown
    std::string type_str = (it_value["type"] == "bytes" || it_value["type"] == "string") ? "uint8" : it_value["type"];
    DataType t_dtype = DataType(type_str);  // valid types: {"bytes", "string", "int32", "int64", "float32", "float64"}
    if (it_value["type"] == "bytes") {      // rank = 1
      col_desc = ColDescriptor(colname, t_dtype, TensorImpl::kFlexible, 1);
    } else if (it_value.find("shape") != it_value.end()) {
      std::vector<dsize_t> vec(it_value["shape"].size());  // temporary vector to hold shape
      (void)std::copy(it_value["shape"].begin(), it_value["shape"].end(), vec.begin());
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
    column_name_mapping_[columns_to_load_[i]] = i;
  }

  num_rows_ = shard_reader_->get_num_rows();
  // Compute how many buffers we would need to accomplish rowsPerBuffer
  buffers_needed_ = (num_rows_ + rows_per_buffer_ - 1) / rows_per_buffer_;
  RETURN_IF_NOT_OK(SetColumnsBlob());

  return Status::OK();
}

Status MindRecordOp::SetColumnsBlob() {
  columns_blob_ = shard_reader_->get_blob_fields().second;
  columns_blob_index_ = std::vector<int32_t>(columns_to_load_.size(), -1);
  int32_t iBlob = 0;
  for (uint32_t i = 0; i < columns_blob_.size(); ++i) {
    if (column_name_mapping_.count(columns_blob_[i])) {
      columns_blob_index_[column_name_mapping_[columns_blob_[i]]] = iBlob++;
    }
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
    out << "\n1 Dataset file : " << dataset_file_ << "\nNumber of rows : " << num_rows_
        << "\nRows per buffer : " << rows_per_buffer_ << "\nNumber of buffers : " << buffers_needed_
        << "\nNumber of ShardReader workers : " << num_mind_record_workers_ << "\n\n";
  }
}

template <typename T>
Status MindRecordOp::LoadFeature(std::shared_ptr<Tensor> *tensor, int32_t i_col,
                                 const std::vector<uint8_t> &columns_blob, const mindrecord::json &columns_json) const {
  TensorShape new_shape = TensorShape::CreateUnknownRankShape();
  const unsigned char *data = nullptr;

  std::unique_ptr<T[]> array_data;
  std::string string_data;

  const ColDescriptor &cur_column = data_schema_->column(i_col);
  std::string column_name = columns_to_load_[i_col];
  DataType type = cur_column.type();

  // load blob column
  if (columns_blob_index_[i_col] >= 0 && columns_blob.size() > 0) {
    int32_t pos = columns_blob_.size() == 1 ? -1 : columns_blob_index_[i_col];
    RETURN_IF_NOT_OK(LoadBlob(&new_shape, &data, columns_blob, pos, cur_column));
  } else {
    switch (type.value()) {
      case DataType::DE_UINT8: {
        // For strings (Assume DE_UINT8 is reserved for strings)
        RETURN_IF_NOT_OK(LoadByte(&new_shape, &string_data, column_name, columns_json));
        data = reinterpret_cast<const unsigned char *>(common::SafeCStr(string_data));
        break;
      }
      case DataType::DE_FLOAT32: {
        // For both float scalars and arrays
        RETURN_IF_NOT_OK(LoadFloat(&new_shape, &array_data, column_name, columns_json, cur_column, false));
        data = reinterpret_cast<const unsigned char *>(array_data.get());
        break;
      }
      case DataType::DE_FLOAT64: {
        // For both double scalars and arrays
        RETURN_IF_NOT_OK(LoadFloat(&new_shape, &array_data, column_name, columns_json, cur_column, true));
        data = reinterpret_cast<const unsigned char *>(array_data.get());
        break;
      }
      default: {
        // For both integers scalars and arrays
        RETURN_IF_NOT_OK(LoadInt(&new_shape, &array_data, column_name, columns_json, cur_column));
        data = reinterpret_cast<const unsigned char *>(array_data.get());
        break;
      }
    }
  }
  // Create Tensor with given details
  RETURN_IF_NOT_OK(Tensor::CreateTensor(tensor, cur_column.tensorImpl(), new_shape, type, data));

  return Status::OK();
}

Status MindRecordOp::LoadBlob(TensorShape *new_shape, const unsigned char **data,
                              const std::vector<uint8_t> &columns_blob, const int32_t pos,
                              const ColDescriptor &column) {
  const auto kColumnSize = column.type().SizeInBytes();
  if (kColumnSize == 0) {
    RETURN_STATUS_UNEXPECTED("column size is null");
  }
  if (pos == -1) {
    if (column.hasShape()) {
      *new_shape = TensorShape::CreateUnknownRankShape();
      RETURN_IF_NOT_OK(
        column.MaterializeTensorShape(static_cast<int32_t>(columns_blob.size() / kColumnSize), new_shape));
    } else {
      std::vector<dsize_t> shapeDetails = {static_cast<dsize_t>(columns_blob.size() / kColumnSize)};
      *new_shape = TensorShape(shapeDetails);
    }
    *data = reinterpret_cast<const uint8_t *>(&(columns_blob[0]));
    return Status::OK();
  }
  auto uint64_from_bytes = [&](int64_t pos) {
    uint64_t result = 0;
    for (uint64_t n = 0; n < kInt64Len; n++) {
      result = (result << 8) + columns_blob[pos + n];
    }
    return result;
  };
  uint64_t iStart = 0;
  for (int32_t i = 0; i < pos; i++) {
    uint64_t num_bytes = uint64_from_bytes(iStart);
    iStart += kInt64Len + num_bytes;
  }
  uint64_t num_bytes = uint64_from_bytes(iStart);
  iStart += kInt64Len;
  if (column.hasShape()) {
    *new_shape = TensorShape::CreateUnknownRankShape();
    RETURN_IF_NOT_OK(column.MaterializeTensorShape(static_cast<int32_t>(num_bytes / kColumnSize), new_shape));
  } else {
    std::vector<dsize_t> shapeDetails = {static_cast<dsize_t>(num_bytes / kColumnSize)};
    *new_shape = TensorShape(shapeDetails);
  }
  *data = reinterpret_cast<const uint8_t *>(&(columns_blob[iStart]));
  return Status::OK();
}

template <typename T>
Status MindRecordOp::LoadFloat(TensorShape *new_shape, std::unique_ptr<T[]> *array_data, const std::string &column_name,
                               const mindrecord::json &columns_json, const ColDescriptor &column, bool use_double) {
  if (!columns_json[column_name].is_array()) {
    T value = 0;
    RETURN_IF_NOT_OK(GetFloat(&value, columns_json[column_name], use_double));

    *new_shape = TensorShape::CreateScalar();
    *array_data = std::make_unique<T[]>(1);
    (*array_data)[0] = value;
  } else {
    if (column.hasShape()) {
      *new_shape = TensorShape(column.shape());
    } else {
      std::vector<dsize_t> shapeDetails = {static_cast<dsize_t>(columns_json[column_name].size())};
      *new_shape = TensorShape(shapeDetails);
    }

    int idx = 0;
    *array_data = std::make_unique<T[]>(new_shape->NumOfElements());
    for (auto &element : columns_json[column_name]) {
      T value = 0;
      RETURN_IF_NOT_OK(GetFloat(&value, element, use_double));

      (*array_data)[idx++] = value;
    }
  }

  return Status::OK();
}

template <typename T>
Status MindRecordOp::GetFloat(T *value, const mindrecord::json &data, bool use_double) {
  if (data.is_number()) {
    *value = data;
  } else if (data.is_string()) {
    try {
      if (use_double) {
        *value = data.get<double>();
      } else {
        *value = data.get<float>();
      }
    } catch (mindrecord::json::exception &e) {
      RETURN_STATUS_UNEXPECTED("Conversion to float failed.");
    }
  } else {
    RETURN_STATUS_UNEXPECTED("Conversion to float failed.");
  }

  return Status::OK();
}

template <typename T>
Status MindRecordOp::LoadInt(TensorShape *new_shape, std::unique_ptr<T[]> *array_data, const std::string &column_name,
                             const mindrecord::json &columns_json, const ColDescriptor &column) {
  if (!columns_json[column_name].is_array()) {
    T value = 0;
    RETURN_IF_NOT_OK(GetInt(&value, columns_json[column_name]));

    *new_shape = TensorShape::CreateScalar();
    *array_data = std::make_unique<T[]>(1);
    (*array_data)[0] = value;
  } else {
    if (column.hasShape()) {
      *new_shape = TensorShape(column.shape());
    } else {
      std::vector<dsize_t> shapeDetails = {static_cast<dsize_t>(columns_json[column_name].size())};
      *new_shape = TensorShape(shapeDetails);
    }

    int idx = 0;
    *array_data = std::make_unique<T[]>(new_shape->NumOfElements());
    for (auto &element : columns_json[column_name]) {
      T value = 0;
      RETURN_IF_NOT_OK(GetInt(&value, element));

      (*array_data)[idx++] = value;
    }
  }

  return Status::OK();
}

template <typename T>
Status MindRecordOp::GetInt(T *value, const mindrecord::json &data) {
  int64_t temp_value = 0;
  bool less_than_zero = false;

  if (data.is_number_integer()) {
    const mindrecord::json json_zero = 0;
    if (data < json_zero) less_than_zero = true;
    temp_value = data;
  } else if (data.is_string()) {
    std::string string_value = data;

    if (!string_value.empty() && string_value[0] == '-') {
      try {
        temp_value = std::stoll(string_value);
        less_than_zero = true;
      } catch (std::invalid_argument &e) {
        RETURN_STATUS_UNEXPECTED("Conversion to int failed, invalid argument.");
      } catch (std::out_of_range &e) {
        RETURN_STATUS_UNEXPECTED("Conversion to int failed, out of range.");
      }
    } else {
      try {
        temp_value = static_cast<int64_t>(std::stoull(string_value));
      } catch (std::invalid_argument &e) {
        RETURN_STATUS_UNEXPECTED("Conversion to int failed, invalid argument.");
      } catch (std::out_of_range &e) {
        RETURN_STATUS_UNEXPECTED("Conversion to int failed, out of range.");
      }
    }
  } else {
    RETURN_STATUS_UNEXPECTED("Conversion to int failed.");
  }

  if ((less_than_zero && temp_value < static_cast<int64_t>(std::numeric_limits<T>::min())) ||
      (!less_than_zero && static_cast<uint64_t>(temp_value) > static_cast<uint64_t>(std::numeric_limits<T>::max()))) {
    RETURN_STATUS_UNEXPECTED("Conversion to int failed. Out of range");
  }
  *value = static_cast<T>(temp_value);

  return Status::OK();
}

Status MindRecordOp::LoadByte(TensorShape *new_shape, std::string *string_data, const std::string &column_name,
                              const mindrecord::json &columns_json) {
  *string_data = columns_json[column_name];
  std::vector<dsize_t> shape_details = {static_cast<dsize_t>(string_data->size())};
  *new_shape = TensorShape(shape_details);

  return Status::OK();
}

Status MindRecordOp::WorkerEntry(int32_t worker_id) {
  TaskManager::FindMe()->Post();
  std::unique_ptr<IOBlock> io_block;
  RETURN_IF_NOT_OK(io_blk_queues_[worker_id]->PopFront(&io_block));
  while (io_block != nullptr) {
    if (io_block->eoe() == true) {
      RETURN_IF_NOT_OK(
        out_connector_->Add(worker_id, std::move(std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE))));
      RETURN_IF_NOT_OK(io_blk_queues_[worker_id]->PopFront(&io_block));
      continue;
    }
    if (io_block->eof() == true) {
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
  (*fetched_buffer)->set_column_name_map(column_name_mapping_);
  std::unique_ptr<TensorQTable> tensor_table = std::make_unique<TensorQTable>();
  for (int32_t i = 0; i < rows_per_buffer_; ++i) {
    ShardTuple tupled_buffer;
    if (block_reader_) {
      if (i >= block_buffer_[buffer_id % num_workers_]->size()) break;
      tupled_buffer = block_buffer_[buffer_id % num_workers_]->at(i);
    } else {
      int32_t row_id = buffer_id * rows_per_buffer_ + i;
      tupled_buffer = shard_reader_->GetNextById(row_id, worker_id);
      if (tupled_buffer.empty()) break;
    }
    for (const auto &tupled_row : tupled_buffer) {
      std::vector<uint8_t> columnsBlob = std::get<0>(tupled_row);
      mindrecord::json columns_json = std::get<1>(tupled_row);
      TensorRow tensor_row;
      for (uint32_t j = 0; j < columns_to_load_.size(); ++j) {
        std::shared_ptr<Tensor> tensor;

        const ColDescriptor &cur_column = data_schema_->column(j);
        DataType type = cur_column.type();
        RETURN_IF_NOT_OK(SwitchLoadFeature(type, &tensor, j, columnsBlob, columns_json));

        tensor_row.push_back(std::move(tensor));
      }

      tensor_table->push_back(std::move(tensor_row));
    }
  }

  // Replace the TensorTable in DataBuffer with the new one.
  (*fetched_buffer)->set_tensor_table(std::move(tensor_table));
  return Status::OK();
}

Status MindRecordOp::SwitchLoadFeature(const DataType &type, std::shared_ptr<Tensor> *tensor, int32_t i_col,
                                       const std::vector<uint8_t> &columns_blob,
                                       const mindrecord::json &columns_json) const {
  switch (type.value()) {
    case DataType::DE_BOOL: {
      return LoadFeature<bool>(tensor, i_col, columns_blob, columns_json);
    }
    case DataType::DE_INT8: {
      return LoadFeature<int8_t>(tensor, i_col, columns_blob, columns_json);
    }
    case DataType::DE_UINT8: {
      return LoadFeature<uint8_t>(tensor, i_col, columns_blob, columns_json);
    }
    case DataType::DE_INT16: {
      return LoadFeature<int16_t>(tensor, i_col, columns_blob, columns_json);
    }
    case DataType::DE_UINT16: {
      return LoadFeature<uint16_t>(tensor, i_col, columns_blob, columns_json);
    }
    case DataType::DE_INT32: {
      return LoadFeature<int32_t>(tensor, i_col, columns_blob, columns_json);
    }
    case DataType::DE_UINT32: {
      return LoadFeature<uint32_t>(tensor, i_col, columns_blob, columns_json);
    }
    case DataType::DE_INT64: {
      return LoadFeature<int64_t>(tensor, i_col, columns_blob, columns_json);
    }
    case DataType::DE_UINT64: {
      return LoadFeature<uint64_t>(tensor, i_col, columns_blob, columns_json);
    }
    case DataType::DE_FLOAT32: {
      return LoadFeature<float>(tensor, i_col, columns_blob, columns_json);
    }
    case DataType::DE_FLOAT64: {
      return LoadFeature<double>(tensor, i_col, columns_blob, columns_json);
    }
    default: {
      return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__,
                    "mindrecord column list type does not match any known types");
    }
  }
}

Status MindRecordOp::FetchBlockBuffer(const int32_t &buffer_id) {
  {
    std::unique_lock<std::mutex> lck(mtx_block_reader_);
    cv_reader_.wait(lck, [buffer_id, this] { return buffer_id < buffer_water_mark_ + num_workers_; });
  }
  for (int32_t i = 0; i < rows_per_buffer_; i++) {
    // Block reader does NOT care about argument
    ShardTuple tuple_buffer = shard_reader_->GetNextById(i, i);
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
  num_rows_ = shard_reader_->get_num_rows();

  buffers_needed_ = num_rows_ / rows_per_buffer_;
  if (num_rows_ % rows_per_buffer_ != 0) {
    buffers_needed_++;
  }

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

Status MindRecordOp::CountTotalRows(const std::string dataset_path, const std::shared_ptr<ShardOperator> &op,
                                    int64_t *count) {
  std::unique_ptr<ShardReader> shard_reader = std::make_unique<ShardReader>();
  MSRStatus rc = shard_reader->CountTotalRows(dataset_path, op, count);
  if (rc == MSRStatus::FAILED) {
    RETURN_STATUS_UNEXPECTED("MindRecordOp count total rows failed.");
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
