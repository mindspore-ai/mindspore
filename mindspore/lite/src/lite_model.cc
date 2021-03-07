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

#include "src/lite_model.h"
#include <vector>
#include <set>
#include <unordered_map>
#include "src/common/prim_util.h"
#ifdef ENABLE_V0
#include "src/ops/compat/compat_register.h"
#endif

namespace mindspore::lite {
#ifdef ENABLE_V0
int LiteModel::ConvertAttrs(Model::Node *node, std::vector<schema::Tensor *> *dst_tensor) {
  if (node == nullptr || dst_tensor == nullptr) {
    MS_LOG(ERROR) << "node or tensor_vec is nullptr.";
    return RET_ERROR;
  }
  auto primitive = node->primitive_;
  MS_ASSERT(primitive != nullptr);
  auto prim = reinterpret_cast<const schema::v0::Primitive *>(primitive);
  int primitive_type = prim->value_type();
  auto creator = CompatRegistry::GetInstance()->GetTransferAttrFunc(SCHEMA_VERSION::SCHEMA_V0, primitive_type);
  if (creator == nullptr) {
    MS_LOG(DEBUG) << "the node don't need to convert attr to tensor.";
    return RET_OK;
  }
  int status = creator(node, dst_tensor, &this->attr_tensor_bufs_);
  if (status != RET_OK && status != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "translate attr to tensor failed.";
    return status;
  }
  return RET_OK;
}

int LiteModel::ConvertAttrToTensors() {
  int schema_version = VersionManager::GetInstance()->GetSchemaVersion();
  if (schema_version != SCHEMA_VERSION::SCHEMA_V0) {
    MS_LOG(DEBUG) << "no need to convert attr to tensor.";
    return RET_OK;
  }
  std::unordered_map<int, std::set<int>> subgraph_node_indexes;
  for (size_t subgraph_index = 0; subgraph_index < this->sub_graphs_.size(); ++subgraph_index) {
    for (size_t node_index = 0; node_index < this->sub_graphs_[subgraph_index]->node_indices_.size(); ++node_index) {
      subgraph_node_indexes[subgraph_index].insert(this->sub_graphs_[subgraph_index]->node_indices_[node_index]);
    }
  }
  int cur_all_tensors_size = this->all_tensors_.size();
  for (size_t index = 0; index < this->all_nodes_.size(); ++index) {
    std::vector<schema::Tensor *> dst_tensors;
    int status = ConvertAttrs(this->all_nodes_[index], &dst_tensors);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "fail to convert attr to tensor.";
      return RET_ERROR;
    }
    if (dst_tensors.empty()) {
      continue;
    }
    std::vector<int> subgraphs_with_node;
    for (size_t subgraph_index = 0; subgraph_index < this->sub_graphs_.size(); ++subgraph_index) {
      if (subgraph_node_indexes[subgraph_index].find(index) == subgraph_node_indexes[subgraph_index].end()) {
        continue;
      }
      subgraphs_with_node.push_back(subgraph_index);
    }
    for (auto tensor : dst_tensors) {
      for (auto subgraph_index : subgraphs_with_node) {
        this->sub_graphs_[subgraph_index]->tensor_indices_.push_back(cur_all_tensors_size);
      }
      this->all_nodes_[index]->input_indices_.push_back(cur_all_tensors_size++);
      this->all_tensors_.push_back(tensor);
    }
  }
  return RET_OK;
}
#endif

void LiteModel::Free() {
  if (this->buf != nullptr) {
    free(this->buf);
    this->buf = nullptr;
  }
  auto nodes_size = this->all_nodes_.size();
  for (size_t i = 0; i < nodes_size; ++i) {
    auto node = this->all_nodes_[i];
    node->primitive_ = nullptr;
  }
  for (auto &tensor_buf : attr_tensor_bufs_) {
    free(tensor_buf);
    tensor_buf = nullptr;
  }
  attr_tensor_bufs_.resize(0);
}

void LiteModel::Destroy() {
  Free();
  auto nodes_size = this->all_nodes_.size();
  for (size_t i = 0; i < nodes_size; ++i) {
    auto node = this->all_nodes_[i];
    MS_ASSERT(node != nullptr);
    delete node;
  }
  this->all_nodes_.clear();

  auto sub_graph_size = this->sub_graphs_.size();
  for (size_t i = 0; i < sub_graph_size; ++i) {
    auto sub_graph = this->sub_graphs_[i];
    delete sub_graph;
  }
}

int LiteModel::ConvertSubGraph(const schema::SubGraph &sub_graph) {
  if (sub_graph.name() == nullptr || sub_graph.inputIndices() == nullptr || sub_graph.outputIndices() == nullptr ||
      sub_graph.nodeIndices() == nullptr || sub_graph.tensorIndices() == nullptr) {
    MS_LOG(ERROR) << "sub_graph is invalid";
    return RET_ERROR;
  }

  auto *subgraph = new (std::nothrow) Model::SubGraph();
  if (subgraph == nullptr) {
    MS_LOG(ERROR) << "new subGraph fail!";
    return RET_ERROR;
  }

  subgraph->name_ = sub_graph.name()->c_str();
  auto in_count = sub_graph.inputIndices()->size();
  for (uint32_t i = 0; i < in_count; ++i) {
    subgraph->input_indices_.push_back(sub_graph.inputIndices()->Get(i));
  }
  auto out_count = sub_graph.outputIndices()->size();
  for (uint32_t i = 0; i < out_count; ++i) {
    subgraph->output_indices_.push_back(sub_graph.outputIndices()->Get(i));
  }
  auto node_count = sub_graph.nodeIndices()->size();
  for (uint32_t i = 0; i < node_count; ++i) {
    subgraph->node_indices_.push_back(sub_graph.nodeIndices()->Get(i));
  }
  auto tensor_count = sub_graph.tensorIndices()->size();
  for (uint32_t i = 0; i < tensor_count; ++i) {
    subgraph->tensor_indices_.push_back(sub_graph.tensorIndices()->Get(i));
  }
  this->sub_graphs_.push_back(subgraph);
  return RET_OK;
}

int LiteModel::VersionVerify(flatbuffers::Verifier *verify) const {
  if (verify == nullptr) {
    MS_LOG(ERROR) << "verify is null.";
    return RET_ERROR;
  }
  if (schema::VerifyMetaGraphBuffer(*verify)) {
    return SCHEMA_VERSION::SCHEMA_CUR;
  }
#ifdef ENABLE_V0
  if (schema::v0::VerifyMetaGraphBuffer(*verify)) {
    return SCHEMA_VERSION::SCHEMA_V0;
  }
#endif
  return SCHEMA_VERSION::SCHEMA_INVALID;
}

int LiteModel::NodeVerify() const {
  auto tensor_size = this->all_tensors_.size();
  uint32_t subGraph_size = this->sub_graphs_.size();

  for (auto &node : this->all_nodes_) {
    if (node == nullptr || node->primitive_ == nullptr) {
      MS_LOG(ERROR) << "node or its primitive_ is null.";
      return RET_ERROR;
    }
    if (std::any_of(node->input_indices_.begin(), node->input_indices_.end(),
                    [&tensor_size](const uint32_t &idx) { return idx >= tensor_size; })) {
      MS_LOG(ERROR) << "Index of node->input_indices_ is beyond size.";
      return RET_ERROR;
    }
    if (std::any_of(node->output_indices_.begin(), node->output_indices_.end(),
                    [&tensor_size](const uint32_t &idx) { return idx >= tensor_size; })) {
      MS_LOG(ERROR) << "Index of node->output_indices_ is beyond size.";
      return RET_ERROR;
    }

    if (IsWhileNode(node->primitive_)) {
      auto body_index = GetWhileBodySubgraphIndex(node->primitive_);
      auto cond_index = GetWhileCondSubgraphIndex(node->primitive_);
      if (static_cast<uint32_t>(body_index) >= subGraph_size || static_cast<uint32_t>(cond_index) >= subGraph_size) {
        MS_LOG(ERROR) << "index of subGraph is beyond subGraph_size.";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

int LiteModel::SubGraphVerify() const {
  auto tensor_size = this->all_tensors_.size();
  auto node_size = this->all_nodes_.size();

  for (auto &graph : this->sub_graphs_) {
    if (graph == nullptr) {
      MS_LOG(ERROR) << "graph is null.";
      return RET_ERROR;
    }
    if (std::any_of(graph->input_indices_.begin(), graph->input_indices_.end(),
                    [&tensor_size](const uint32_t &idx) { return idx >= tensor_size; })) {
      MS_LOG(ERROR) << "Index of graph->input_indices_ is beyond tensor_size.";
      return RET_ERROR;
    }
    if (std::any_of(graph->output_indices_.begin(), graph->output_indices_.end(),
                    [&tensor_size](const uint32_t &idx) { return idx >= tensor_size; })) {
      MS_LOG(ERROR) << "Index of graph->output_indices_ is beyond tensor_size.";
      return RET_ERROR;
    }
    if (std::any_of(graph->tensor_indices_.begin(), graph->tensor_indices_.end(),
                    [&tensor_size](const uint32_t &idx) { return idx >= tensor_size; })) {
      MS_LOG(ERROR) << "Index of graph->tensor_indices_ is beyond tensor_size.";
      return RET_ERROR;
    }
    if (std::any_of(graph->node_indices_.begin(), graph->node_indices_.end(),
                    [&node_size](const uint32_t &idx) { return idx >= node_size; })) {
      MS_LOG(ERROR) << "Index of graph->node_indices_ is beyond node_size.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

bool LiteModel::ModelVerify() const { return NodeVerify() == RET_OK && SubGraphVerify() == RET_OK; }

const void *LiteModel::GetMetaGraphByVerison() {
  MS_ASSERT(this->buf != nullptr);
  auto schema_version = VersionManager::GetInstance()->GetSchemaVersion();
  if (schema_version == SCHEMA_VERSION::SCHEMA_CUR) {
    return reinterpret_cast<const void *>(schema::GetMetaGraph(this->buf));
  }
#ifdef ENABLE_V0
  if (schema_version == SCHEMA_VERSION::SCHEMA_V0) {
    return reinterpret_cast<const void *>(schema::v0::GetMetaGraph(buf));
  }
#endif
  return nullptr;
}

int LiteModel::GenerateModelByVersion(const void *meta_graph) {
  MS_ASSERT(meta_graph != nullptr);
  auto schema_version = VersionManager::GetInstance()->GetSchemaVersion();
  int status = RET_ERROR;
  if (schema_version == SCHEMA_VERSION::SCHEMA_CUR) {
    status = GenerateModel<schema::MetaGraph, schema::CNode>(*reinterpret_cast<const schema::MetaGraph *>(meta_graph));
  }
#ifdef ENABLE_V0
  if (schema_version == SCHEMA_VERSION::SCHEMA_V0) {
    status = GenerateModel<schema::v0::MetaGraph, schema::v0::CNode>(
      *reinterpret_cast<const schema::v0::MetaGraph *>(meta_graph));
  }
#endif
  return status;
}

int LiteModel::ConstructModel() {
  if (this->buf == nullptr || this->buf_size_ <= 0) {
    MS_LOG(ERROR) << "cannot construct model.";
    return RET_NULL_PTR;
  }
  flatbuffers::Verifier verify((const uint8_t *)this->buf, this->buf_size_);
  int schema_version = VersionVerify(&verify);
  if (schema_version == SCHEMA_INVALID) {
    MS_LOG(ERROR) << "The buffer is invalid and fail to create graph.";
    return RET_ERROR;
  }
  VersionManager::GetInstance()->SetSchemaVersion(schema_version);
  const void *meta_graph = GetMetaGraphByVerison();
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "meta_graph is nullptr!";
    return RET_NULL_PTR;
  }

  int status = GenerateModelByVersion(meta_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "fail to generate model";
    return status;
  }

  if (this->version_ != Version()) {
    MS_LOG(WARNING) << "model version is " << this->version_ << ", inference version is " << Version() << " not equal";
  }
  if (this->sub_graphs_.empty()) {
    return RET_ERROR;
  }

  return ModelVerify() ? RET_OK : RET_ERROR;
}

Model *ImportFromBuffer(const char *model_buf, size_t size, bool take_buf) {
  if (model_buf == nullptr) {
    MS_LOG(ERROR) << "The model buf is nullptr";
    return nullptr;
  }
  auto *model = new (std::nothrow) LiteModel();
  if (model == nullptr) {
    MS_LOG(ERROR) << "new model fail!";
    return nullptr;
  }
  if (take_buf) {
    model->buf = const_cast<char *>(model_buf);
  } else {
    if (size == 0) {
      MS_LOG(ERROR) << "malloc size is equal to 0";
      delete (model);
      return nullptr;
    }
    model->buf = reinterpret_cast<char *>(malloc(size));
    if (model->buf == nullptr) {
      MS_LOG(ERROR) << "new inner model buf fail!";
      delete (model);
      return nullptr;
    }
    memcpy(model->buf, model_buf, size);
  }
  model->buf_size_ = size;
  auto status = model->ConstructModel();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "construct model failed.";
    delete model;
    return nullptr;
  }
  return model;
}

Model *Model::Import(const char *model_buf, size_t size) { return ImportFromBuffer(model_buf, size, false); }
}  // namespace mindspore::lite
