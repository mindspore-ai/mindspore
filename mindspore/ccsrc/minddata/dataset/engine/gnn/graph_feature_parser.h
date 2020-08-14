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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_FEATURE_PARSER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_FEATURE_PARSER_H_
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/tensor.h"
#if !defined(_WIN32) && !defined(_WIN64)
#include "minddata/dataset/engine/gnn/graph_shared_memory.h"
#endif
#include "minddata/dataset/engine/gnn/feature.h"
#include "minddata/dataset/util/status.h"
#include "minddata/mindrecord/include/shard_column.h"

namespace mindspore {
namespace dataset {
namespace gnn {

using mindrecord::ShardColumn;

class GraphFeatureParser {
 public:
  explicit GraphFeatureParser(const ShardColumn &shard_column);

  ~GraphFeatureParser() = default;

  // @param std::string key - column name
  // @param std::vector<uint8_t> &blob - contains data in blob field in mindrecord
  // @param std::vector<int32_t> *ind - return value, list of feature index in int32_t
  // @return Status - the status code
  Status LoadFeatureIndex(const std::string &key, const std::vector<uint8_t> &blob, std::vector<int32_t> *ind);

  // @param std::string &key - column name
  // @param std::vector<uint8_t> &blob - contains data in blob field in mindrecord
  // @param std::shared_ptr<Tensor> *tensor - return value feature tensor
  // @return Status - the status code
  Status LoadFeatureTensor(const std::string &key, const std::vector<uint8_t> &blob, std::shared_ptr<Tensor> *tensor);
#if !defined(_WIN32) && !defined(_WIN64)
  Status LoadFeatureToSharedMemory(const std::string &key, const std::vector<uint8_t> &col_blob,
                                   GraphSharedMemory *shared_memory, std::shared_ptr<Tensor> *out_tensor);
#endif
 private:
  std::unique_ptr<ShardColumn> shard_column_;
};
}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_FEATURE_PARSER_H_
