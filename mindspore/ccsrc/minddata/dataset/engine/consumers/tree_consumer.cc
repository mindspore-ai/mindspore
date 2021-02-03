/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#include "minddata/dataset/engine/consumers/tree_consumer.h"
#include "minddata/dataset/engine/datasetops/device_queue_op.h"
#include "minddata/dataset/engine/opt/pre/getter_pass.h"
#include "minddata/dataset/engine/tree_adapter.h"

#ifndef ENABLE_ANDROID
#include "minddata/mindrecord/include/shard_index_generator.h"
#include "minddata/mindrecord/include/shard_header.h"
#include "minddata/mindrecord/include/shard_writer.h"
#endif

namespace mindspore {
namespace dataset {
// TreeConsumer
TreeConsumer::TreeConsumer() { tree_adapter_ = std::make_unique<TreeAdapter>(); }

Status TreeConsumer::Init(std::shared_ptr<DatasetNode> d) { return tree_adapter_->Compile(std::move(d)); }
Status TreeConsumer::Terminate() {
  CHECK_FAIL_RETURN_UNEXPECTED(tree_adapter_->AllTasks() != nullptr, " Execution tree has not been built");
  return tree_adapter_->AllTasks()->ServiceStop();
}

// IteratorConsumer
Status IteratorConsumer::Init(std::shared_ptr<DatasetNode> d) {
  return tree_adapter_->Compile(std::move(d), num_epochs_);
}

Status IteratorConsumer::GetNextAsVector(std::vector<TensorPtr> *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  out->clear();

  TensorRow res;
  RETURN_IF_NOT_OK(tree_adapter_->GetNext(&res));

  // Return empty vector if there's no data
  RETURN_OK_IF_TRUE(res.empty());

  std::copy(res.begin(), res.end(), std::back_inserter(*out));
  return Status::OK();
}

Status IteratorConsumer::GetNextAsMap(std::unordered_map<std::string, TensorPtr> *const out_map) {
  RETURN_UNEXPECTED_IF_NULL(out_map);
  out_map->clear();

  TensorRow res;
  RETURN_IF_NOT_OK(tree_adapter_->GetNext(&res));

  // Return empty map if there's no data
  RETURN_OK_IF_TRUE(res.empty());

  // Populate the out map from the row and return it
  for (const auto &colMap : tree_adapter_->GetColumnNameMap()) {
    (*out_map)[colMap.first] = std::move(res[colMap.second]);
  }
  return Status::OK();
}

Status IteratorConsumer::GetNextAsOrderedPair(std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> *const vec) {
  CHECK_FAIL_RETURN_UNEXPECTED(vec != nullptr && vec->empty(), "vec is null or non-empty.");

  TensorRow curr_row;

  RETURN_IF_NOT_OK(tree_adapter_->GetNext(&curr_row));
  RETURN_OK_IF_TRUE(curr_row.empty());

  size_t num_cols = curr_row.size();  // num_cols is non-empty.
  // order the column names according to their ids
  if (column_order_.empty()) {
    const int32_t invalid_col_id = -1;
    column_order_.resize(num_cols, {std::string(), invalid_col_id});
    for (const auto &itr : tree_adapter_->GetColumnNameMap()) {
      int32_t ind = itr.second;
      CHECK_FAIL_RETURN_UNEXPECTED(ind < num_cols && ind >= 0, "column id out of bounds.");
      column_order_[ind] = std::make_pair(itr.first, ind);
    }
    // error check, make sure the ids in col_name_id_map are continuous and starts from 0
    for (const auto &col : column_order_) {
      CHECK_FAIL_RETURN_UNEXPECTED(col.second != invalid_col_id, "column ids are not continuous.");
    }
  }

  vec->reserve(num_cols);

  std::transform(column_order_.begin(), column_order_.end(), std::back_inserter(*vec),
                 [curr_row](const auto &col) { return std::make_pair(col.first, curr_row[col.second]); });

  return Status::OK();
}

// ToDevice
Status ToDevice::Init(std::shared_ptr<DatasetNode> d) { return tree_adapter_->Compile(std::move(d), num_epochs_); }

Status ToDevice::Send() {
  std::unique_ptr<DataBuffer> db;
  RETURN_IF_NOT_OK(tree_adapter_->Launch());
  std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_->GetRoot());
  CHECK_FAIL_RETURN_UNEXPECTED(root != nullptr, "Root is a nullptr.");
  return Status::OK();
}

Status ToDevice::Continue() {
  // tree_.root() must be DeviceQueueOp
  std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_->GetRoot());
  CHECK_FAIL_RETURN_UNEXPECTED(root != nullptr, "Root is a nullptr.");
  DeviceQueueOp *op = dynamic_cast<DeviceQueueOp *>(root.get());
  CHECK_FAIL_RETURN_UNEXPECTED(op != nullptr, "ContinueSend only supported by DeviceQueueOp");
  op->ContinueSend();
  return Status::OK();
}

Status ToDevice::Stop() {
  std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_->GetRoot());
  CHECK_FAIL_RETURN_UNEXPECTED(root != nullptr, "Root is a nullptr.");
  DeviceQueueOp *op = dynamic_cast<DeviceQueueOp *>(root.get());
  CHECK_FAIL_RETURN_UNEXPECTED(op != nullptr, "StopSend only supported by DeviceQueueOp");
  op->StopSend();

  return Status::OK();
}

Status ToDevice::GetDataInfo(std::vector<DataType> *const types, std::vector<TensorShape> *const shapes) {
  // tree_.root() must be DeviceQueueOp
  std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_->GetRoot());
  CHECK_FAIL_RETURN_UNEXPECTED(root != nullptr, "Root is a nullptr.");
  DeviceQueueOp *op = dynamic_cast<DeviceQueueOp *>(root.get());
  CHECK_FAIL_RETURN_UNEXPECTED(op != nullptr, "GetDataInfo only supported by DeviceQueueOp");
  DATA_INFO data_info;
  RETURN_IF_NOT_OK(op->GetDataInfo(&data_info));
  for (auto el : data_info) {
    types->push_back(el.first);
    shapes->push_back(el.second);
  }
  return Status::OK();
}

Status ToDevice::Terminate() {
#ifdef ENABLE_TDTQUE
  std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_->GetRoot());
  CHECK_FAIL_RETURN_UNEXPECTED(root != nullptr, "Root is a nullptr.");
  DeviceQueueOp *op = dynamic_cast<DeviceQueueOp *>(root.get());
  CHECK_FAIL_RETURN_UNEXPECTED(op != nullptr, "StopSend only supported by DeviceQueueOp");
  op->StopWaiting();
#endif
  return TreeConsumer::Terminate();
}

#ifndef ENABLE_ANDROID
// SaveToDisk
Status SaveToDisk::ValidateParams() {
  if (dataset_path_.empty()) {
    std::string err = "CreateSaver failed, dataset_path must not be empty";
    MS_LOG(ERROR) << err;
    RETURN_STATUS_SYNTAX_ERROR(err);
  }
  Path dir(dataset_path_);
  if (dir.IsDirectory()) {
    std::string err = "CreateSaver failed, dataset_path must not be a directory";
    MS_LOG(ERROR) << err;
    RETURN_STATUS_SYNTAX_ERROR(err);
  }
  if (access(dir.ParentPath().c_str(), R_OK) == -1) {
    std::string err_msg = "CreateSaver failed, no access to specified dataset path: " + dataset_path_;
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (num_files_ <= 0 || num_files_ > 1000) {
    std::string err = "CreateSaver failed, num_files must between 1 and 1000, but got " + std::to_string(num_files_);
    MS_LOG(ERROR) << err;
    RETURN_STATUS_SYNTAX_ERROR(err);
  }
  if (dataset_type_ != "mindrecord") {
    std::string err = "CreateSaver failed, only \"mindrecord\" dataset format is supported, but got " + dataset_type_;
    MS_LOG(ERROR) << err;
    RETURN_STATUS_SYNTAX_ERROR(err);
  }
  return Status::OK();
}

Status SaveToDisk::Save() {
  std::vector<std::string> file_names;
  if (num_files_ == 1) {
    file_names.push_back(dataset_path_);
  } else {
    for (int32_t i = 0; i < num_files_; i++) {
      file_names.push_back(dataset_path_ + std::to_string(i));
    }
  }

  auto mr_header = std::make_shared<mindrecord::ShardHeader>();
  auto mr_writer = std::make_unique<mindrecord::ShardWriter>();
  std::vector<std::string> blob_fields;
  if (mindrecord::SUCCESS != mindrecord::ShardWriter::initialize(&mr_writer, file_names)) {
    RETURN_STATUS_UNEXPECTED("Error: failed to initialize ShardWriter.");
  }

  std::unordered_map<std::string, int32_t> column_name_id_map;
  for (auto el : tree_adapter_->GetColumnNameMap()) {
    std::string column_name = el.first;
    std::transform(column_name.begin(), column_name.end(), column_name.begin(),
                   [](unsigned char c) { return ispunct(c) ? '_' : c; });
    column_name_id_map[column_name] = el.second;
  }

  TensorRow row;
  uint64_t mr_schema_id = 0;
  bool first_loop = true;  // build schema in first loop
  do {
    nlohmann::json row_raw_data;
    std::map<std::string, std::unique_ptr<std::vector<uint8_t>>> row_bin_data;
    RETURN_IF_NOT_OK(tree_adapter_->GetNext(&row));
    if (row.empty()) {
      break;
    }
    if (first_loop) {
      nlohmann::json mr_json;
      std::vector<std::string> index_fields;
      RETURN_IF_NOT_OK(FetchMetaFromTensorRow(column_name_id_map, row, &mr_json, &index_fields));
      MS_LOG(DEBUG) << "Schema of saved mindrecord: " << mr_json.dump();
      if (mindrecord::SUCCESS !=
          mindrecord::ShardHeader::initialize(&mr_header, mr_json, index_fields, blob_fields, mr_schema_id)) {
        RETURN_STATUS_UNEXPECTED("Error: failed to initialize ShardHeader.");
      }
      mr_writer->SetShardHeader(mr_header);
      first_loop = false;
    }
    // construct data
    if (!row.empty()) {  // write data
      RETURN_IF_NOT_OK(FetchDataFromTensorRow(row, column_name_id_map, &row_raw_data, &row_bin_data));
      std::shared_ptr<std::vector<uint8_t>> output_bin_data;
      mr_writer->MergeBlobData(blob_fields, row_bin_data, &output_bin_data);
      std::map<std::uint64_t, std::vector<nlohmann::json>> raw_data;
      raw_data.insert(
        std::pair<uint64_t, std::vector<nlohmann::json>>(mr_schema_id, std::vector<nlohmann::json>{row_raw_data}));
      std::vector<std::vector<uint8_t>> bin_data;
      if (output_bin_data != nullptr) {
        bin_data.emplace_back(*output_bin_data);
      }
      mr_writer->WriteRawData(raw_data, bin_data);
    }
  } while (!row.empty());

  mr_writer->Commit();
  if (mindrecord::SUCCESS != mindrecord::ShardIndexGenerator::finalize(file_names)) {
    RETURN_STATUS_UNEXPECTED("Error: failed to finalize ShardIndexGenerator.");
  }
  return Status::OK();
}

Status SaveToDisk::FetchMetaFromTensorRow(const std::unordered_map<std::string, int32_t> &column_name_id_map,
                                          const TensorRow &row, nlohmann::json *schema,
                                          std::vector<std::string> *index_fields) {
  if (schema == nullptr) {
    RETURN_STATUS_UNEXPECTED("Error: schema is NULL.");
  }
  if (index_fields == nullptr) {
    RETURN_STATUS_UNEXPECTED("Error: index fields is NULL.");
  }
  if (column_name_id_map.empty()) {
    RETURN_STATUS_UNEXPECTED("Error: column not found.");
  }
  nlohmann::json dataset_schema;
  for (auto &col : column_name_id_map) {
    auto idx = col.second;
    auto column_name = col.first;
    auto &tensor = row[idx];
    auto column_type = tensor->type();
    auto column_shape = tensor->shape();

    std::string mr_type;
    auto shapes = column_shape.AsVector();
    std::vector<int> mr_shape(shapes.begin(), shapes.end());
    std::string el = column_type.ToString();
    dataset_schema[column_name] = el;
    if (mindrecord::kTypesMap.find(el) == mindrecord::kTypesMap.end()) {
      std::string err_msg("Error: can not support data type: " + el);
      RETURN_STATUS_UNEXPECTED(err_msg);
    } else {
      mr_type = mindrecord::kTypesMap.at(el);
    }
    if (mr_shape.empty()) {
      if (mr_type == "bytes") {  // map to int32 when bytes without shape.
        mr_type = "int32";
      }
      (*schema)[column_name] = {{"type", mr_type}};
    } else {
      if (mr_type == "string") {  // mindrecord can not support string with shape.
        std::string err_msg("Error: mindrecord can not support multi-dimensional string tensor.");
        RETURN_STATUS_UNEXPECTED(err_msg);
      }
      if (mr_type == "bytes") {  // ignore shape of bytes in minrecord
        (*schema)[column_name] = {{"type", mr_type}};
      } else {
        (*schema)[column_name] = {{"type", mr_type}, {"shape", mr_shape}};
      }
    }
    if (mr_type == "bytes" || !mr_shape.empty()) continue;
    index_fields->emplace_back(column_name);  // candidate of index fields
  }
  MS_LOG(DEBUG) << "Schema of dataset: " << dataset_schema.dump();
  return Status::OK();
}

static Status ValidateInputParams(nlohmann::json *row_raw_data,
                                  std::map<std::string, std::unique_ptr<std::vector<uint8_t>>> *row_bin_data,
                                  const std::unordered_map<std::string, int32_t> &column_name_id_map) {
  if (row_raw_data == nullptr) {
    RETURN_STATUS_UNEXPECTED("Error: row raw data is NULL.");
  }
  if (row_bin_data == nullptr) {
    RETURN_STATUS_UNEXPECTED("Error: row bin data is NULL.");
  }
  if (column_name_id_map.empty()) {
    RETURN_STATUS_UNEXPECTED("Error: column not found");
  }
  return Status::OK();
}

Status SaveToDisk::FetchFloatData(std::shared_ptr<Tensor> tensor, std::string column_name, nlohmann::json *row_raw_data,
                                  std::unique_ptr<std::vector<uint8_t>> *data_ptr) {
  auto column_type = tensor->type();
  Status s;
  if (column_type == DataType::DE_FLOAT32) {
    std::unique_ptr<float> data, dummy;
    s = TransformTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, data_ptr, &dummy);
    RETURN_IF_NOT_OK(s);
    if (data != nullptr) (*row_raw_data)[column_name] = std::move(*data);
  } else if (column_type == DataType::DE_FLOAT64) {
    std::unique_ptr<double> data, dummy;
    s = TransformTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, data_ptr, &dummy);
    RETURN_IF_NOT_OK(s);
    if (data != nullptr) (*row_raw_data)[column_name] = std::move(*data);
  }
  return Status::OK();
}

Status SaveToDisk::FetchItemData(std::shared_ptr<Tensor> tensor, std::string column_name, nlohmann::json *row_raw_data,
                                 std::map<std::string, std::unique_ptr<std::vector<uint8_t>>> *row_bin_data) {
  auto column_type = tensor->type();
  Status s;
  std::unique_ptr<std::vector<uint8_t>> data_ptr;
  if (column_type == DataType::DE_INT8) {
    std::unique_ptr<int32_t> data;
    std::unique_ptr<int8_t> dummy;
    s = TransformTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, &data_ptr, &dummy, true);
    RETURN_IF_NOT_OK(s);
    if (data != nullptr) (*row_raw_data)[column_name] = std::move(*data);
  } else if (column_type == DataType::DE_INT16) {
    std::unique_ptr<int32_t> data;
    std::unique_ptr<int16_t> dummy;
    s = TransformTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, &data_ptr, &dummy, true);
    RETURN_IF_NOT_OK(s);
    if (data != nullptr) (*row_raw_data)[column_name] = std::move(*data);
  } else if (column_type == DataType::DE_UINT16) {
    std::unique_ptr<int32_t> data;
    std::unique_ptr<uint16_t> dummy;
    s = TransformTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, &data_ptr, &dummy, true);
    RETURN_IF_NOT_OK(s);
    if (data != nullptr) (*row_raw_data)[column_name] = std::move(*data);
  } else if (column_type == DataType::DE_UINT8) {
    std::unique_ptr<uint8_t> data, dummy;
    s = TransformTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, &data_ptr, &dummy);
    RETURN_IF_NOT_OK(s);
    if (data != nullptr) (*row_raw_data)[column_name] = std::move(*data);
  } else if (column_type == DataType::DE_INT32) {
    std::unique_ptr<int32_t> data, dummy;
    s = TransformTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, &data_ptr, &dummy);
    RETURN_IF_NOT_OK(s);
    if (data != nullptr) (*row_raw_data)[column_name] = std::move(*data);
  } else if (column_type == DataType::DE_UINT32) {
    std::unique_ptr<int64_t> data;
    std::unique_ptr<uint32_t> dummy;
    s = TransformTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, &data_ptr, &dummy, true);
    RETURN_IF_NOT_OK(s);
    if (data != nullptr) (*row_raw_data)[column_name] = std::move(*data);
  } else if (column_type == DataType::DE_INT64) {
    std::unique_ptr<int64_t> data, dummy;
    s = TransformTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, &data_ptr, &dummy);
    RETURN_IF_NOT_OK(s);
    if (data != nullptr) (*row_raw_data)[column_name] = std::move(*data);
  } else if (column_type == DataType::DE_FLOAT32 || column_type == DataType::DE_FLOAT64) {
    s = FetchFloatData(tensor, column_name, row_raw_data, &data_ptr);
    RETURN_IF_NOT_OK(s);
  } else if (column_type == DataType::DE_STRING) {
    std::string_view sv;
    RETURN_IF_NOT_OK(tensor->GetItemAt(&sv, {0}));  // assume scalar string tensor
    std::string ss(sv);
    (*row_raw_data)[column_name] = std::move(ss);
    return Status::OK();
  } else {
    RETURN_STATUS_UNEXPECTED("Got unexpected type when casting data.");
  }
  if (data_ptr != nullptr) {
    (*row_bin_data)[column_name] = std::move(data_ptr);
  }
  return Status::OK();
}

Status SaveToDisk::FetchDataFromTensorRow(const TensorRow &row,
                                          const std::unordered_map<std::string, int32_t> &column_name_id_map,
                                          nlohmann::json *row_raw_data,
                                          std::map<std::string, std::unique_ptr<std::vector<uint8_t>>> *row_bin_data) {
  Status s;
  s = ValidateInputParams(row_raw_data, row_bin_data, column_name_id_map);
  if (s.IsError()) {
    return s;
  }
  for (auto &col : column_name_id_map) {
    auto idx = col.second;
    auto column_name = col.first;
    auto &tensor = row[idx];
    s = FetchItemData(tensor, column_name, row_raw_data, row_bin_data);
    RETURN_IF_NOT_OK(s);
  }
  return Status::OK();
}

template <typename T, typename S>
Status SaveToDisk::TransformTensor(const unsigned char *src, const TensorShape &shape, const int64_t num_of_elements,
                                   std::unique_ptr<T> *data, std::unique_ptr<std::vector<uint8_t>> *data_ptr,
                                   std::unique_ptr<S> *s, bool need_convert) {
  if (nullptr == src) {
    RETURN_STATUS_UNEXPECTED("Error: buffer of Tensor is NULL.");
  }
  *data_ptr = std::make_unique<std::vector<uint8_t>>(num_of_elements * sizeof(T));
  if (need_convert) {
    auto tmp_ptr = std::make_unique<std::vector<uint8_t>>(num_of_elements * sizeof(S));
    std::copy(src, src + sizeof(S) * num_of_elements, tmp_ptr->begin());
    auto s_ptr = reinterpret_cast<S *>(&(*(tmp_ptr->begin())));
    auto el = std::make_unique<T>();
    for (uint32_t i = 0; i < num_of_elements; ++i) {
      *el = *(s_ptr + i);
      auto t_ptr = reinterpret_cast<uint8_t *>(el.get());
      for (uint32_t j = 0; j < sizeof(T); ++j) {
        *((*data_ptr)->begin() + i * sizeof(T) + j) = *(t_ptr + j);
      }
    }
  } else {
    std::copy(src, src + sizeof(T) * num_of_elements, (*data_ptr)->begin());
  }
  if (shape.empty()) {
    *data = std::make_unique<T>();
    auto t_ptr = reinterpret_cast<uint8_t *>((*data).get());
    for (uint32_t i = 0; i < sizeof(T); ++i) {
      *(t_ptr + i) = *((*data_ptr)->begin() + i);
    }
  }
  return Status::OK();
}
#endif

TreeGetters::TreeGetters() : dataset_size_(-1), init_flag_(false), first_row_obtained_(false) {
  tree_adapter_ = std::make_unique<TreeAdapter>(TreeAdapter::UsageFlag::kDeGetter);
}

Status TreeGetters::Init(std::shared_ptr<DatasetNode> d) {
  root_ = std::move(d);
  return Status::OK();
}

Status TreeGetters::GetRow(TensorRow *row) { return tree_adapter_->GetNext(row); }

Status TreeGetters::GetOutputTypes(std::vector<DataType> *types) {
  RETURN_IF_NOT_OK(GetFirstRowShapeAndType());
  *types = first_row_type_;
  return Status::OK();
}

Status TreeGetters::GetOutputShapes(std::vector<TensorShape> *shapes) {
  RETURN_IF_NOT_OK(GetFirstRowShapeAndType());
  *shapes = first_row_shape_;
  return Status::OK();
}

Status TreeGetters::GetBatchSize(int64_t *batch_size) {
  RETURN_IF_NOT_OK(InternalInit());
  std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_->GetRoot());
  RETURN_UNEXPECTED_IF_NULL(root);
  *batch_size = root->GetTreeBatchSize();
  CHECK_FAIL_RETURN_UNEXPECTED(*batch_size != -1, "Error in finding the batch size.");
  return Status::OK();
}

Status TreeGetters::GetRepeatCount(int64_t *repeat_count) {
  RETURN_IF_NOT_OK(InternalInit());
  std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_->GetRoot());
  RETURN_UNEXPECTED_IF_NULL(root);
  *repeat_count = root->GetTreeRepeatCount();
  return Status::OK();
}

Status TreeGetters::GetNumClasses(int64_t *num_classes) {
  RETURN_IF_NOT_OK(InternalInit());
  std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_->GetRoot());
  RETURN_UNEXPECTED_IF_NULL(root);
  RETURN_IF_NOT_OK(root->GetNumClasses(num_classes));
  return Status::OK();
}

Status TreeGetters::GetColumnNames(std::vector<std::string> *output) {
  RETURN_IF_NOT_OK(InternalInit());
  std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_->GetRoot());
  RETURN_UNEXPECTED_IF_NULL(root);
  std::unordered_map<std::string, int32_t> column_name_id_map = root->column_name_id_map();
  CHECK_FAIL_RETURN_UNEXPECTED(!column_name_id_map.empty(), "GetColumnNames: column_name_id map is empty.");
  std::vector<std::pair<std::string, int32_t>> col_name_id_vec(column_name_id_map.begin(), column_name_id_map.end());
  std::sort(col_name_id_vec.begin(), col_name_id_vec.end(),
            [](const std::pair<std::string, int32_t> &a, const std::pair<std::string, int32_t> &b) {
              return a.second < b.second;
            });
  std::transform(col_name_id_vec.begin(), col_name_id_vec.end(), std::back_inserter(*output),
                 [](const std::pair<std::string, int32_t> &p) { return p.first; });
  return Status::OK();
}

Status TreeGetters::GetClassIndexing(std::vector<std::pair<std::string, std::vector<int32_t>>> *output_class_indexing) {
  RETURN_IF_NOT_OK(InternalInit());
  std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_->GetRoot());
  RETURN_UNEXPECTED_IF_NULL(root);
  RETURN_IF_NOT_OK(root->GetClassIndexing(output_class_indexing));
  return Status::OK();
}

Status TreeGetters::InternalInit() {
  if (init_flag_) return Status::OK();
  Status s = tree_adapter_->Compile(std::move(root_), 1);
  if (s.IsOk()) init_flag_ = true;
  return s;
}

Status TreeGetters::GetFirstRowShapeAndType() {
  RETURN_OK_IF_TRUE(first_row_obtained_);
  RETURN_IF_NOT_OK(InternalInit());
  TensorRow first_row;
  RETURN_IF_NOT_OK(GetRow(&first_row));
  std::transform(first_row.begin(), first_row.end(), std::back_inserter(first_row_type_),
                 [](const TensorPtr &t) { return t->type(); });
  std::transform(first_row.begin(), first_row.end(), std::back_inserter(first_row_shape_),
                 [](const TensorPtr &t) { return t->shape(); });
  first_row_obtained_ = true;
  return Status::OK();
}

Status BuildVocabConsumer::Init(std::shared_ptr<DatasetNode> d) { return tree_adapter_->Compile(std::move(d), 1); }

Status BuildVocabConsumer::Start() {
  // Getting one row would trigger building the vocab
  TensorRow row;
  RETURN_IF_NOT_OK(tree_adapter_->GetNext(&row));
  // The returned row would EOE which is an empty row
  CHECK_FAIL_RETURN_UNEXPECTED(row.empty(), "The fetched row from BuildVocab should be an EOE.");
  return Status::OK();
}
Status DatasetSizeGetter::GetDatasetSize(int64_t *size, bool estimate) {
  if (dataset_size_ == -1) {
    RETURN_IF_NOT_OK(root_->GetDatasetSize(shared_from_this(), estimate, size));
    dataset_size_ = *size;  // save the previous result
  }

  *size = dataset_size_;
  return Status::OK();
}
Status DatasetSizeGetter::Init(std::shared_ptr<DatasetNode> d) {
  root_ = std::move(d);
  return Status::OK();
}
Status DatasetSizeGetter::DryRun(std::shared_ptr<DatasetNode> ir_node, int64_t *dataset_size) {
  std::shared_ptr<TreeAdapter> tree_adapter = std::make_shared<TreeAdapter>(TreeAdapter::UsageFlag::kDeGetter);
  tree_adapters_.push_back(tree_adapter);
  RETURN_IF_NOT_OK(tree_adapter->Compile(ir_node, 1));
  TensorRow row;
  RETURN_IF_NOT_OK(GetRow(tree_adapter, &row));
  int64_t row_cnt = 0;
  while (!row.empty()) {
    ++row_cnt;
    RETURN_IF_NOT_OK(GetRow(tree_adapter, &row));
  }
  *dataset_size = row_cnt;
  return Status::OK();
}
Status DatasetSizeGetter::GetRow(const std::shared_ptr<TreeAdapter> &tree_adapter, TensorRow *row) {
  return tree_adapter->GetNext(row);
}
Status DatasetSizeGetter::Terminate() {
  for (const auto &tree : tree_adapters_) {
    RETURN_UNEXPECTED_IF_NULL(tree);
    RETURN_UNEXPECTED_IF_NULL(tree->AllTasks());
    RETURN_IF_NOT_OK(tree->AllTasks()->ServiceStop());
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
