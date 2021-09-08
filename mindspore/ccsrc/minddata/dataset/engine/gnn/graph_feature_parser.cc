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

#include "minddata/dataset/engine/gnn/graph_feature_parser.h"

#include <memory>
#include <utility>

#include "mindspore/ccsrc/minddata/mindrecord/include/shard_error.h"

namespace mindspore {
namespace dataset {
namespace gnn {

using mindrecord::MSRStatus;

GraphFeatureParser::GraphFeatureParser(const ShardColumn &shard_column) {
  shard_column_ = std::make_unique<ShardColumn>(shard_column);
}

Status GraphFeatureParser::LoadFeatureTensor(const std::string &key, const std::vector<uint8_t> &col_blob,
                                             std::shared_ptr<Tensor> *tensor) {
  const unsigned char *data = nullptr;
  std::unique_ptr<unsigned char[]> data_ptr;
  uint64_t n_bytes = 0, col_type_size = 1;
  mindrecord::ColumnDataType col_type = mindrecord::ColumnNoDataType;
  std::vector<int64_t> column_shape;
  RETURN_IF_NOT_OK(shard_column_->GetColumnValueByName(key, col_blob, {}, &data, &data_ptr, &n_bytes, &col_type,
                                                       &col_type_size, &column_shape));
  if (data == nullptr) {
    data = reinterpret_cast<const unsigned char *>(&data_ptr[0]);
  }
  RETURN_IF_NOT_OK(Tensor::CreateFromMemory(std::move(TensorShape({static_cast<dsize_t>(n_bytes / col_type_size)})),
                                            std::move(DataType(mindrecord::ColumnDataTypeNameNormalized[col_type])),
                                            data, tensor));
  return Status::OK();
}

#if !defined(_WIN32) && !defined(_WIN64)
Status GraphFeatureParser::LoadFeatureToSharedMemory(const std::string &key, const std::vector<uint8_t> &col_blob,
                                                     GraphSharedMemory *shared_memory,
                                                     std::shared_ptr<Tensor> *out_tensor) {
  const unsigned char *data = nullptr;
  std::unique_ptr<unsigned char[]> data_ptr;
  uint64_t n_bytes = 0, col_type_size = 1;
  mindrecord::ColumnDataType col_type = mindrecord::ColumnNoDataType;
  std::vector<int64_t> column_shape;
  RETURN_IF_NOT_OK(shard_column_->GetColumnValueByName(key, col_blob, {}, &data, &data_ptr, &n_bytes, &col_type,
                                                       &col_type_size, &column_shape));
  if (data == nullptr) {
    data = reinterpret_cast<const unsigned char *>(&data_ptr[0]);
  }
  std::shared_ptr<Tensor> tensor;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(std::move(TensorShape({2})), std::move(DataType(DataType::DE_INT64)), &tensor));
  auto fea_itr = tensor->begin<int64_t>();
  int64_t offset = 0;
  RETURN_IF_NOT_OK(shared_memory->InsertData(data, n_bytes, &offset));
  *fea_itr = offset;
  ++fea_itr;
  *fea_itr = n_bytes;
  *out_tensor = std::move(tensor);
  return Status::OK();
}
#endif

Status GraphFeatureParser::LoadFeatureIndex(const std::string &key, const std::vector<uint8_t> &col_blob,
                                            std::vector<int32_t> *indices) {
  const unsigned char *data = nullptr;
  std::unique_ptr<unsigned char[]> data_ptr;
  uint64_t n_bytes = 0, col_type_size = 1;
  mindrecord::ColumnDataType col_type = mindrecord::ColumnNoDataType;
  std::vector<int64_t> column_shape;
  RETURN_IF_NOT_OK(shard_column_->GetColumnValueByName(key, col_blob, {}, &data, &data_ptr, &n_bytes, &col_type,
                                                       &col_type_size, &column_shape));

  if (data == nullptr) {
    data = reinterpret_cast<const unsigned char *>(&data_ptr[0]);
  }

  for (int i = 0; i < n_bytes; i += col_type_size) {
    int32_t feature_ind = -1;
    if (col_type == mindrecord::ColumnInt32) {
      feature_ind = *(reinterpret_cast<const int32_t *>(data + i));
    } else if (col_type == mindrecord::ColumnInt64) {
      feature_ind = *(reinterpret_cast<const int64_t *>(data + i));
    } else {
      RETURN_STATUS_UNEXPECTED("Feature Index needs to be int32/int64 type!");
    }
    if (feature_ind >= 0) {
      indices->push_back(feature_ind);
    }
  }
  return Status::OK();
}

}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
