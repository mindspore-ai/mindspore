/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include <utility>

#include "utils/ms_utils.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/engine/datasetops/source/sampler/mind_record_sampler.h"
#include "minddata/mindrecord/include/shard_column.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/util/log_adapter.h"

namespace mindspore {
namespace dataset {

using mindrecord::kInt64Len;
using mindrecord::MSRStatus;
using mindrecord::Schema;
using mindrecord::ShardOperator;
using mindrecord::ShardReader;

// Constructor of the MindRecordOp.
MindRecordOp::MindRecordOp(int32_t num_mind_record_workers, std::vector<std::string> dataset_file, bool load_dataset,
                           int32_t op_connector_queue_size, const std::vector<std::string> &columns_to_load,
                           const std::vector<std::shared_ptr<ShardOperator>> &operators, int64_t num_padded,
                           const mindrecord::json &sample_json, const std::map<std::string, std::string> &sample_bytes,
                           const ShuffleMode shuffle_mode, std::unique_ptr<ShardReader> shard_reader,
                           std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_mind_record_workers, op_connector_queue_size, std::move(sampler)),
      dataset_file_(std::move(dataset_file)),
      load_dataset_(load_dataset),
      columns_to_load_(columns_to_load),
      operators_(operators),
      num_mind_record_workers_(num_mind_record_workers),
      ended_worker_(0),
      num_padded_(num_padded),
      sample_json_(sample_json),
      sample_bytes_(sample_bytes),
      shuffle_mode_(shuffle_mode),
      shard_reader_(std::move(shard_reader)) {
  epoch_sync_flag_ = true;  // MindRecordOp needs to turn this flag on, otherwise, calling ShuffleTask() before all
                            // tasks are consumed by the worker threads would cause problem.
}

// Private helper method to encapsulate some common construction/reset tasks
Status MindRecordOp::Init() {
  RETURN_IF_NOT_OK(shard_reader_->Open(dataset_file_, load_dataset_, num_mind_record_workers_, columns_to_load_,
                                       operators_, num_padded_));

  data_schema_ = std::make_unique<DataSchema>();

  std::vector<std::string> col_names = shard_reader_->GetShardColumn()->GetColumnName();
  CHECK_FAIL_RETURN_UNEXPECTED(!col_names.empty(),
                               "Invalid column, no column names are specified, check mindrecord file.");
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
                                   "Invalid column, " + colname + " does not exist in data file.");
      RETURN_IF_NOT_OK(tmp_schema->AddColumn(data_schema_->Column(colname_to_ind[colname])));
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
    out << "\nNumber of rows : " << num_rows_ << "\nNumber of ShardReader workers : " << num_mind_record_workers_
        << "\n\n";
  }
}

Status MindRecordOp::WorkerEntry(int32_t worker_id) {
  TaskManager::FindMe()->Post();
  std::unique_ptr<IOBlock> io_block;
  RETURN_IF_NOT_OK(worker_in_queues_[worker_id]->PopFront(&io_block));
  while (io_block != nullptr) {
    if (io_block->wait()) {
      RETURN_IF_NOT_OK(worker_out_queues_[worker_id]->EmplaceBack(TensorRow(TensorRow::TensorRowFlags::kFlagWait)));
      TaskManager::FindMe()->Wait();  // wait for auto tune update workers successful
      TaskManager::FindMe()->Clear();
    } else if (io_block->eoe()) {
      RETURN_IF_NOT_OK(worker_out_queues_[worker_id]->EmplaceBack(TensorRow(TensorRow::TensorRowFlags::kFlagEOE)));
    } else if (io_block->eof()) {
      RETURN_IF_NOT_OK(worker_out_queues_[worker_id]->EmplaceBack(TensorRow(TensorRow::TensorRowFlags::kFlagEOF)));
    } else {
      // load TensorRow
      std::vector<int64_t> keys;
      RETURN_IF_NOT_OK(io_block->GetKeys(&keys));
      if (keys.empty() == true) {
        {
          std::unique_lock<std::mutex> lock(ended_worker_mutex_);
          ended_worker_++;
          if (ended_worker_ == num_workers_) {
            shard_reader_->Close();
          }
        }
        return Status::OK();  // empty key is a quit signal for workers
      }

      const uint64_t row_id = keys[0];
      TensorRow fetched_row;

      // Get the next row. Push it up to the output connector.
      if (row_id % LOG_INTERVAL == 0) {
        MS_LOG(DEBUG) << "MindRecord operator consumed row " << row_id << " by worker " << worker_id << ".";
      }
      RETURN_IF_NOT_OK(GetRowFromReader(&fetched_row, row_id, worker_id));
      RETURN_IF_NOT_OK(worker_out_queues_[worker_id]->EmplaceBack(std::move(fetched_row)));
    }
    RETURN_IF_NOT_OK(worker_in_queues_[worker_id]->PopFront(&io_block));
  }
  RETURN_STATUS_UNEXPECTED("[Internal ERROR] Unexpected nullptr received in worker.");
}

Status MindRecordOp::GetRowFromReader(TensorRow *fetched_row, uint64_t row_id, int32_t worker_id) {
  RETURN_UNEXPECTED_IF_NULL(fetched_row);
  *fetched_row = {};
  auto rc = shard_reader_->GetNextById(row_id, worker_id);
  auto task_type = rc.first;
  auto tupled_buffer = rc.second;
  if (task_type == mindrecord::TaskType::kPaddedTask) {
    RETURN_IF_NOT_OK(LoadTensorRow(fetched_row, {}, mindrecord::json(), task_type));
    std::vector<std::string> file_path(fetched_row->size(), dataset_file_[0]);
    fetched_row->setPath(file_path);
    fetched_row->setId(row_id);
  }
  if (tupled_buffer.empty()) {
    return Status::OK();
  }
  if (task_type == mindrecord::TaskType::kCommonTask) {
    for (const auto &tupled_row : tupled_buffer) {
      std::vector<uint8_t> columns_blob = std::get<0>(tupled_row);
      mindrecord::json columns_json = std::get<1>(tupled_row);
      RETURN_IF_NOT_OK(LoadTensorRow(fetched_row, columns_blob, columns_json, task_type));
      std::vector<std::string> file_path(fetched_row->size(), dataset_file_[0]);
      fetched_row->setPath(file_path);
      fetched_row->setId(row_id);
    }
  }

  return Status::OK();
}

Status MindRecordOp::LoadTensorRow(TensorRow *tensor_row, const std::vector<uint8_t> &columns_blob,
                                   const mindrecord::json &columns_json, const mindrecord::TaskType task_type) {
  for (int32_t i_col = 0; i_col < columns_to_load_.size(); i_col++) {
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
      mindrecord::ColumnCategory category;
      RETURN_IF_NOT_OK(shard_column->GetColumnTypeByName(column_name, &column_data_type, &column_data_type_size,
                                                         &column_shape, &category));
      if (category == mindrecord::ColumnInRaw) {
        RETURN_IF_NOT_OK(shard_column->GetColumnFromJson(column_name, sample_json_, &data_ptr, &n_bytes));
      } else if (category == mindrecord::ColumnInBlob) {
        CHECK_FAIL_RETURN_UNEXPECTED(sample_bytes_.find(column_name) != sample_bytes_.end(),
                                     "Invalid padded_sample, failed to retrieve blob data from padding sample, "
                                     "check 'padded_sample'.");

        std::string ss(sample_bytes_[column_name]);
        n_bytes = ss.size();
        data_ptr = std::make_unique<unsigned char[]>(n_bytes);
        std::copy(ss.begin(), ss.end(), data_ptr.get());
      } else {
        RETURN_STATUS_UNEXPECTED("Invalid datatype, retrieved data type is unknown.");
      }
      if (data == nullptr) {
        data = reinterpret_cast<const unsigned char *>(data_ptr.get());
      }
    } else {
      RETURN_IF_NOT_OK(shard_column->GetColumnValueByName(column_name, columns_blob, columns_json, &data, &data_ptr,
                                                          &n_bytes, &column_data_type, &column_data_type_size,
                                                          &column_shape));
    }

    std::shared_ptr<Tensor> tensor;
    const ColDescriptor &column = data_schema_->Column(i_col);
    DataType type = column.Type();

    // Set shape
    CHECK_FAIL_RETURN_UNEXPECTED(column_data_type_size != 0,
                                 "[Internal ERROR] Found memory size of column data type is 0.");
    auto num_elements = n_bytes / column_data_type_size;
    if (type == DataType::DE_STRING) {
      std::string s{data, data + n_bytes};
      RETURN_IF_NOT_OK(Tensor::CreateScalar(s, &tensor));
    } else if (column.HasShape()) {
      auto new_shape = TensorShape(column.Shape());
      // if the numpy is null, create empty tensor shape
      if (num_elements == 0) {
        new_shape = TensorShape({});
      } else {
        RETURN_IF_NOT_OK(column.MaterializeTensorShape(static_cast<int32_t>(num_elements), &new_shape));
      }
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

// Overrides base class reset method.  When an operator does a reset, it cleans up any state
// info from it's previous execution and then initializes itself so that it can be executed
// again.
Status MindRecordOp::Reset() {
  MS_LOG(DEBUG) << Name() << " performing a self-reset.";
  RETURN_IF_NOT_OK(WaitForWorkers());
  RETURN_IF_NOT_OK(MappableLeafOp::Reset());  // Call our super class reset first.

  // wakeup workers
  for (auto &item : worker_tasks_) {
    item->Post();
  }

  // wakeup the collector thread
  wait_for_collector_.Set();

  return Status::OK();
}

Status MindRecordOp::PrepareData() {
  num_rows_ = shard_reader_->GetNumRows();
  return Status::OK();
}

Status MindRecordOp::RegisterAndLaunchThreads() {
  RETURN_IF_NOT_OK(ParallelOp::RegisterAndLaunchThreads());
  RETURN_IF_NOT_OK(shard_reader_->Launch(true));
  return Status::OK();
}

Status MindRecordOp::CountTotalRows(const std::vector<std::string> dataset_path, bool load_dataset,
                                    const std::shared_ptr<ShardOperator> &op, int64_t *count, int64_t num_padded) {
  RETURN_UNEXPECTED_IF_NULL(op);
  RETURN_UNEXPECTED_IF_NULL(count);
  std::unique_ptr<ShardReader> shard_reader = std::make_unique<ShardReader>();
  RETURN_IF_NOT_OK(shard_reader->CountTotalRows(dataset_path, load_dataset, op, count, num_padded));
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

Status MindRecordOp::AddNewWorkers(int32_t num_new_workers) {
  // wait for workers to process the current rows
  RETURN_IF_NOT_OK(WaitForWorkers());

  RETURN_IF_NOT_OK(shard_reader_->ExtendRandomFileStreams(num_new_workers));
  num_mind_record_workers_ += num_new_workers;

  // create queue first
  for (int32_t i = 0; i < num_new_workers; i++) {
    RETURN_IF_NOT_OK(worker_in_queues_.AddQueue(tree_->AllTasks()));
    RETURN_IF_NOT_OK(worker_out_queues_.AddQueue(tree_->AllTasks()));
  }

  // launch new workers
  for (int32_t i = 0; i < num_new_workers; i++) {
    Task *new_task;
    RETURN_IF_NOT_OK(tree_->AllTasks()->CreateAsyncTask(
      Name() + "::WorkerEntry", std::bind(&MindRecordOp::WorkerEntry, this, num_workers_), &new_task, id()));
    CHECK_FAIL_RETURN_UNEXPECTED(new_task != nullptr, "Cannot create a new worker.");
    worker_tasks_.push_back(new_task);
    num_workers_++;
    MS_LOG(INFO) << "A new worker has been added to op: " << Name() << "::" << id() << " num_workers=" << num_workers_;
  }

  // wakeup old workers
  for (auto &item : worker_tasks_) {
    item->Post();
  }

  // wakeup the collector thread
  wait_for_collector_.Set();

  return Status::OK();
}

Status MindRecordOp::RemoveWorkers(int32_t num_workers) {
  // wait for workers to process the current rows
  RETURN_IF_NOT_OK(WaitForWorkers());

  num_mind_record_workers_ -= num_workers;
  RETURN_IF_NOT_OK(shard_reader_->ShrinkRandomFileStreams(num_workers));

  for (int32_t i = 0; i < num_workers; i++) {
    RETURN_IF_NOT_OK(SendQuitFlagToWorker(num_workers_ - 1));
    worker_tasks_[num_workers_ - 1]->Post();
    RETURN_IF_NOT_OK(worker_tasks_[num_workers_ - 1]->Join());
    RETURN_IF_NOT_OK(worker_in_queues_.RemoveLastQueue());
    worker_tasks_.pop_back();
    num_workers_--;
    MS_LOG(INFO) << "Worker ID " << num_workers_ << " is requested to be removed in operator: " << NameWithID()
                 << " num_workers=" << num_workers_;
  }

  // wakeup left workers
  for (auto &item : worker_tasks_) {
    item->Post();
  }

  // wakeup the collector thread
  wait_for_collector_.Set();

  return Status::OK();
}

Status MindRecordOp::InitPullMode() {
  RETURN_IF_NOT_OK(Init());
  RETURN_IF_NOT_OK(shard_reader_->Launch(true));
  return this->PrepareData();
}

Status MindRecordOp::LoadTensorRowPullMode(row_id_type row_id, TensorRow *row) {
  return GetRowFromReader(row, row_id, id());
}

}  // namespace dataset
}  // namespace mindspore
