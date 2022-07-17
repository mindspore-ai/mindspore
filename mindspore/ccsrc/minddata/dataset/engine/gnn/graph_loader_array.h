/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_LOADER_ARRAY_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_LOADER_ARRAY_H_

#include <deque>
#include <memory>
#include <queue>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/gnn/edge.h"
#include "minddata/dataset/engine/gnn/feature.h"
#include "minddata/dataset/engine/gnn/graph_feature_parser.h"
#if !defined(_WIN32) && !defined(_WIN64)
#include "minddata/dataset/engine/gnn/graph_shared_memory.h"
#endif
#include "minddata/dataset/engine/gnn/node.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/engine/gnn/graph_data_impl.h"
#include "minddata/dataset/engine/gnn/graph_loader.h"

namespace mindspore {
namespace dataset {
namespace gnn {
class GraphLoaderFromArray : public GraphLoader {
 public:
  // Create graph with loading numpy array.
  GraphLoaderFromArray(GraphDataImpl *graph_impl, int32_t num_nodes, const std::shared_ptr<Tensor> &edge,
                       const std::unordered_map<std::int16_t, std::shared_ptr<Tensor>> &node_feat,
                       const std::unordered_map<std::int16_t, std::shared_ptr<Tensor>> &edge_feat,
                       const std::unordered_map<std::int16_t, std::shared_ptr<Tensor>> &graph_feat,
                       const std::shared_ptr<Tensor> &node_type, const std::shared_ptr<Tensor> &edge_type,
                       int32_t num_workers = 4, bool server_mode = false);

  /// \brief default destructor
  ~GraphLoaderFromArray() = default;

  // Init array and load everything into memory multi-threaded
  // @return Status - the status code
  Status InitAndLoad() override;

#if !defined(_WIN32) && !defined(_WIN64)
  // load feature into shared memory
  // @param int32_t i - feature index
  // @param std::pair<int16_t, std::shared_ptr<Tensor>> item - contain feature type and feature value
  // @param std::shared_ptr<Tensor> *out_tensor, Tensor that convert from corresponding feature
  // @return Status - the status code
  Status LoadFeatureToSharedMemory(int32_t i, std::pair<int16_t, std::shared_ptr<Tensor>> item,
                                   std::shared_ptr<Tensor> *out_tensor);
#endif

  // load feature item
  // @param int32_t i - feature index
  // @param std::pair<int16_t, std::shared_ptr<Tensor>> item - contain feature type and feature value
  // @param std::shared_ptr<Tensor> *tensor, Tensor that convert from corresponding feature
  // @return Status - the status code
  Status LoadFeatureTensor(int32_t i, std::pair<int16_t, std::shared_ptr<Tensor>> item,
                           std::shared_ptr<Tensor> *tensor);

 private:
  // worker thread that reads array data
  // @param int32_t worker_id - id of each worker
  // @return Status - the status code
  Status WorkerEntry(int32_t worker_id);

  // Load node into memory, returns a shared_ptr<Node>
  // @return Status - the status code
  Status LoadNode(int32_t worker_id);

  // Load edge into memory, returns a shared_ptr<Edge>
  // @return Status - the status code
  Status LoadEdge(int32_t worker_id);

  int32_t num_nodes_;
  const int32_t num_workers_;
  std::unordered_map<std::int16_t, std::shared_ptr<Tensor>> node_feat_;
  std::unordered_map<std::int16_t, std::shared_ptr<Tensor>> edge_feat_;
  std::unordered_map<std::int16_t, std::shared_ptr<Tensor>> graph_feat_;
  std::shared_ptr<Tensor> edge_ = nullptr;
  std::shared_ptr<Tensor> node_type_ = nullptr;
  std::shared_ptr<Tensor> edge_type_ = nullptr;
  std::shared_ptr<Tensor> node_weight_ = nullptr;
  std::shared_ptr<Tensor> edge_weight_ = nullptr;
};
}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_LOADER_ARRAY_H_
