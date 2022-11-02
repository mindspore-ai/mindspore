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
#include "minddata/dataset/engine/gnn/graph_loader_array.h"

#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/ipc.h>
#endif

#include <future>
#include <tuple>
#include <utility>

#include "minddata/dataset/engine/gnn/graph_data_impl.h"
#include "minddata/dataset/engine/gnn/local_edge.h"
#include "minddata/dataset/engine/gnn/local_node.h"
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
namespace gnn {
const FeatureType weight_feature_type = -1;

GraphLoaderFromArray::GraphLoaderFromArray(GraphDataImpl *graph_impl, int32_t num_nodes,
                                           const std::shared_ptr<Tensor> &edge,
                                           const std::unordered_map<FeatureType, std::shared_ptr<Tensor>> &node_feat,
                                           const std::unordered_map<FeatureType, std::shared_ptr<Tensor>> &edge_feat,
                                           const std::unordered_map<FeatureType, std::shared_ptr<Tensor>> &graph_feat,
                                           const std::shared_ptr<Tensor> &node_type,
                                           const std::shared_ptr<Tensor> &edge_type, int32_t num_workers,
                                           bool server_mode)
    : GraphLoader(graph_impl, "", num_workers),
      num_nodes_(num_nodes),
      edge_(edge),
      node_feat_(node_feat),
      edge_feat_(edge_feat),
      graph_feat_(graph_feat),
      node_type_(node_type),
      edge_type_(edge_type),
      num_workers_(num_workers) {}

Status GraphLoaderFromArray::InitAndLoad() {
  CHECK_FAIL_RETURN_UNEXPECTED(num_workers_ > 0, "num_workers should be equal or great than 1.");
  n_deques_.resize(num_workers_);
  e_deques_.resize(num_workers_);
  n_feature_maps_.resize(num_workers_);
  e_feature_maps_.resize(num_workers_);
  default_node_feature_maps_.resize(num_workers_);
  default_edge_feature_maps_.resize(num_workers_);
  TaskGroup vg;

  if (graph_impl_->server_mode_) {
#if !defined(_WIN32) && !defined(_WIN64)
    // obtain the size that required for store feature, if add_node or add_edge later, this should be larger initially
    int64_t total_feature_size = 0;
    total_feature_size = std::accumulate(node_feat_.begin(), node_feat_.end(), total_feature_size,
                                         [](int64_t temp_size, std::pair<FeatureType, std::shared_ptr<Tensor>> item) {
                                           return temp_size + item.second->SizeInBytes();
                                         });
    total_feature_size = std::accumulate(edge_feat_.begin(), edge_feat_.end(), total_feature_size,
                                         [](int64_t temp_size, std::pair<FeatureType, std::shared_ptr<Tensor>> item) {
                                           return temp_size + item.second->SizeInBytes();
                                         });

    MS_LOG(INFO) << "Total feature size in input data is(byte):" << total_feature_size;

    // generate memory_key
    char file_name[] = "/tmp/tempfile_XXXXXX";
    int fd = mkstemp(file_name);
    CHECK_FAIL_RETURN_UNEXPECTED(fd != -1, "create temp file failed when create graph with loading array data.");
    auto memory_key = ftok(file_name, kGnnSharedMemoryId);
    auto err = unlink(file_name);
    std::string err_msg = "unable to delete file:";
    CHECK_FAIL_RETURN_UNEXPECTED(err != -1, err_msg + file_name);

    close(fd);
    graph_impl_->graph_shared_memory_ = std::make_unique<GraphSharedMemory>(total_feature_size, memory_key);
    RETURN_IF_NOT_OK(graph_impl_->graph_shared_memory_->CreateSharedMemory());
#else
    RETURN_STATUS_UNEXPECTED("Server mode is not supported in Windows OS.");
#endif
  }

  // load graph feature into memory firstly
  for (const auto &item : graph_feat_) {
    graph_impl_->graph_feature_map_[item.first] = std::make_shared<Feature>(item.first, item.second);
  }
  graph_feat_.clear();

  // deal with weight in node and edge firstly
  auto weight_itr = node_feat_.find(weight_feature_type);
  if (weight_itr != node_feat_.end()) {
    node_weight_ = weight_itr->second;
    node_feat_.erase(weight_feature_type);
  }
  weight_itr = edge_feat_.find(weight_feature_type);
  if (weight_itr != edge_feat_.end()) {
    edge_weight_ = weight_itr->second;
    edge_feat_.erase(weight_feature_type);
  }

  for (int wkr_id = 0; wkr_id < num_workers_; ++wkr_id) {
    RETURN_IF_NOT_OK(
      vg.CreateAsyncTask("GraphLoaderFromArray", std::bind(&GraphLoaderFromArray::WorkerEntry, this, wkr_id)));
  }

  // wait for threads to finish and check its return code
  RETURN_IF_NOT_OK(vg.join_all(Task::WaitFlag::kBlocking));
  RETURN_IF_NOT_OK(vg.GetTaskErrorIfAny());

  return Status::OK();
}

Status GraphLoaderFromArray::WorkerEntry(int32_t worker_id) {
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(LoadNode(worker_id));
  RETURN_IF_NOT_OK(LoadEdge(worker_id));
  return Status::OK();
}

Status GraphLoaderFromArray::LoadNode(int32_t worker_id) {
  MS_LOG(INFO) << "start Load Node, worker id is:" << worker_id;
  for (NodeIdType i = worker_id; i < num_nodes_; i = i + num_workers_) {
    WeightType weight = 1.0;
    NodeType node_type;
    if (node_weight_ != nullptr) {
      RETURN_IF_NOT_OK(node_weight_->GetItemAt<WeightType>(&weight, {i}));
    }
    RETURN_IF_NOT_OK(node_type_->GetItemAt<NodeType>(&node_type, {i}));
    std::shared_ptr<Node> node_ptr = std::make_shared<LocalNode>(i, node_type, weight);

    if (graph_impl_->server_mode_) {
#if !defined(_WIN32) && !defined(_WIN64)
      for (const auto &item : node_feat_) {
        std::shared_ptr<Tensor> tensor_sm;
        RETURN_IF_NOT_OK(LoadFeatureToSharedMemory(i, item, &tensor_sm));
        RETURN_IF_NOT_OK(node_ptr->UpdateFeature(std::make_shared<Feature>(item.first, tensor_sm, true)));
        n_feature_maps_[worker_id][node_type].insert(item.first);

        // this may only need execute once, as all node has the same feature type
        if (default_node_feature_maps_[worker_id][item.first] == nullptr) {
          std::shared_ptr<Tensor> tensor = nullptr;
          std::shared_ptr<Tensor> zero_tensor;
          RETURN_IF_NOT_OK(LoadFeatureTensor(i, item, &tensor));
          RETURN_IF_NOT_OK(Tensor::CreateEmpty(tensor->shape(), tensor->type(), &zero_tensor));
          RETURN_IF_NOT_OK(zero_tensor->Zero());
          default_node_feature_maps_[worker_id][item.first] = std::make_shared<Feature>(item.first, zero_tensor);
        }
      }
#else
      RETURN_STATUS_UNEXPECTED("Server mode is not supported in Windows OS.");
#endif
    } else {
      for (const auto &item : node_feat_) {
        // get one row in corresponding node_feature
        std::shared_ptr<Tensor> feature_item;
        RETURN_IF_NOT_OK(LoadFeatureTensor(i, item, &feature_item));

        RETURN_IF_NOT_OK(node_ptr->UpdateFeature(std::make_shared<Feature>(item.first, feature_item)));
        n_feature_maps_[worker_id][node_type].insert(item.first);
        // this may only need execute once, as all node has the same feature type
        if (default_node_feature_maps_[worker_id][item.first] == nullptr) {
          std::shared_ptr<Tensor> zero_tensor;
          RETURN_IF_NOT_OK(Tensor::CreateEmpty(feature_item->shape(), feature_item->type(), &zero_tensor));
          RETURN_IF_NOT_OK(zero_tensor->Zero());
          default_node_feature_maps_[worker_id][item.first] = std::make_shared<Feature>(item.first, zero_tensor);
        }
      }
    }
    n_deques_[worker_id].emplace_back(node_ptr);
  }
  return Status::OK();
}

Status GraphLoaderFromArray::LoadEdge(int32_t worker_id) {
  MS_LOG(INFO) << "Start Load Edge, worker id is:" << worker_id;
  RETURN_UNEXPECTED_IF_NULL(edge_);
  auto num_edges = edge_->shape()[1];
  for (EdgeIdType i = worker_id; i < num_edges; i = i + num_workers_) {
    // if weight exist in feature, then update it
    WeightType weight = 1.0;
    if (edge_weight_ != nullptr) {
      RETURN_IF_NOT_OK(edge_weight_->GetItemAt<WeightType>(&weight, {i}));
    }
    NodeIdType src_id, dst_id;
    EdgeType edge_type;
    RETURN_IF_NOT_OK(edge_->GetItemAt<NodeIdType>(&src_id, {0, i}));
    RETURN_IF_NOT_OK(edge_->GetItemAt<NodeIdType>(&dst_id, {1, i}));
    RETURN_IF_NOT_OK(edge_type_->GetItemAt<EdgeType>(&edge_type, {i}));

    std::shared_ptr<Edge> edge_ptr = std::make_shared<LocalEdge>(i, edge_type, weight, src_id, dst_id);
    if (graph_impl_->server_mode_) {
#if !defined(_WIN32) && !defined(_WIN64)
      for (const auto &item : edge_feat_) {
        std::shared_ptr<Tensor> tensor_sm;
        RETURN_IF_NOT_OK(LoadFeatureToSharedMemory(i, item, &tensor_sm));
        RETURN_IF_NOT_OK(edge_ptr->UpdateFeature(std::make_shared<Feature>(item.first, tensor_sm, true)));
        e_feature_maps_[worker_id][edge_type].insert(item.first);

        // this may only need execute once, as all node has the same feature type
        if (default_edge_feature_maps_[worker_id][item.first] == nullptr) {
          std::shared_ptr<Tensor> tensor = nullptr;
          std::shared_ptr<Tensor> zero_tensor;
          RETURN_IF_NOT_OK(LoadFeatureTensor(i, item, &tensor));
          RETURN_IF_NOT_OK(Tensor::CreateEmpty(tensor->shape(), tensor->type(), &zero_tensor));
          RETURN_IF_NOT_OK(zero_tensor->Zero());
          default_edge_feature_maps_[worker_id][item.first] = std::make_shared<Feature>(item.first, zero_tensor);
        }
      }
#else
      RETURN_STATUS_UNEXPECTED("Server mode is not supported in Windows OS.");
#endif
    } else {
      for (const auto &item : edge_feat_) {
        std::shared_ptr<Tensor> feature_item;
        RETURN_IF_NOT_OK(LoadFeatureTensor(i, item, &feature_item));

        RETURN_IF_NOT_OK(edge_ptr->UpdateFeature(std::make_shared<Feature>(item.first, feature_item)));
        e_feature_maps_[worker_id][edge_type].insert(item.first);
        // this may only need execute once, as all node has the same feature type
        if (default_edge_feature_maps_[worker_id][item.first] == nullptr) {
          std::shared_ptr<Tensor> zero_tensor;
          RETURN_IF_NOT_OK(Tensor::CreateEmpty(feature_item->shape(), feature_item->type(), &zero_tensor));
          RETURN_IF_NOT_OK(zero_tensor->Zero());
          default_edge_feature_maps_[worker_id][item.first] = std::make_shared<Feature>(item.first, zero_tensor);
        }
      }
    }
    e_deques_[worker_id].emplace_back(edge_ptr);
  }
  return Status::OK();
}

#if !defined(_WIN32) && !defined(_WIN64)
Status GraphLoaderFromArray::LoadFeatureToSharedMemory(int32_t i, std::pair<int16_t, std::shared_ptr<Tensor>> item,
                                                       std::shared_ptr<Tensor> *out_tensor) {
  auto feature_num = item.second->shape()[1];
  uint8_t type_size = item.second->type().SizeInBytes();
  dsize_t src_flat_ind = 0;
  RETURN_IF_NOT_OK(item.second->shape().ToFlatIndex({i, 0}, &src_flat_ind));
  auto start_ptr = item.second->GetBuffer() + src_flat_ind * type_size;

  dsize_t n_bytes = feature_num * type_size;
  int64_t offset = 0;
  auto shared_memory = graph_impl_->graph_shared_memory_.get();
  RETURN_IF_NOT_OK(shared_memory->InsertData(start_ptr, n_bytes, &offset));

  std::shared_ptr<Tensor> tensor;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(std::move(TensorShape({2})), std::move(DataType(DataType::DE_INT64)), &tensor));
  auto fea_itr = tensor->begin<int64_t>();
  *fea_itr = offset;
  ++fea_itr;
  *fea_itr = n_bytes;
  *out_tensor = std::move(tensor);
  return Status::OK();
}
#endif

Status GraphLoaderFromArray::LoadFeatureTensor(int32_t i, std::pair<int16_t, std::shared_ptr<Tensor>> item,
                                               std::shared_ptr<Tensor> *tensor) {
  RETURN_UNEXPECTED_IF_NULL(tensor);
  std::shared_ptr<Tensor> feature_item;
  auto feature_num = item.second->shape()[1];
  uint8_t type_size = item.second->type().SizeInBytes();
  dsize_t src_flat_ind = 0;
  RETURN_IF_NOT_OK(item.second->shape().ToFlatIndex({i, 0}, &src_flat_ind));
  auto start_ptr = item.second->GetBuffer() + src_flat_ind * type_size;
  RETURN_IF_NOT_OK(Tensor::CreateFromMemory(TensorShape({feature_num}), item.second->type(), start_ptr, &feature_item));

  *tensor = std::move(feature_item);
  return Status::OK();
}
}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
