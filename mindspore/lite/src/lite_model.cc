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
#include <sys/stat.h>
#include <iostream>
#include <functional>
#include <vector>
#include <algorithm>
#include <set>
#include <unordered_map>
#include <memory>
#include <numeric>
#include "src/common/prim_util.h"
#include "src/common/graph_util.h"
#include "src/common/file_utils.h"
#include "src/tensor.h"
#ifdef ENABLE_V0
#include "src/ops/compat/compat_register.h"
#endif

namespace mindspore::lite {
namespace {
constexpr size_t kMaxModelBufferSize = static_cast<size_t>(1024) * 1024 * 1024 * 2;
}

#ifdef ENABLE_V0
int LiteModel::ConvertAttrs(Model::Node *node, std::vector<schema::Tensor *> *dst_tensor) {
  if (node == nullptr || dst_tensor == nullptr) {
    MS_LOG(ERROR) << "node or tensor_vec is nullptr.";
    return RET_ERROR;
  }
  auto primitive = node->primitive_;
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr.";
    return RET_ERROR;
  }
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
  if (schema_version_ != SCHEMA_VERSION::SCHEMA_V0) {
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
    delete[](this->buf);
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

  for (auto &node_buf : node_bufs_) {
    free(node_buf);
    node_buf = nullptr;
  }
  node_bufs_.resize(0);

  for (auto *schema_tensor_wrapper : inner_all_tensors_) {
    delete schema_tensor_wrapper;
  }
  inner_all_tensors_.clear();

#ifdef ENABLE_MODEL_OBF
  for (auto &prim : deobf_prims_) {
    free(prim);
  }
  deobf_prims_.resize(0);
#endif
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
      sub_graph.tensorIndices() == nullptr) {
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
  if (sub_graph.nodeIndices() != nullptr) {
    auto node_count = sub_graph.nodeIndices()->size();
    for (uint32_t i = 0; i < node_count; ++i) {
      subgraph->node_indices_.push_back(sub_graph.nodeIndices()->Get(i));
    }
  }
  auto tensor_count = sub_graph.tensorIndices()->size();
  for (uint32_t i = 0; i < tensor_count; ++i) {
    subgraph->tensor_indices_.push_back(sub_graph.tensorIndices()->Get(i));
  }
  this->sub_graphs_.push_back(subgraph);
  return RET_OK;
}

int LiteModel::VersionVerify(flatbuffers::Verifier *verify) {
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
  uint32_t subgraph_size = static_cast<uint32_t>(this->sub_graphs_.size());

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
    if (std::any_of(node->output_indices_.begin(), node->output_indices_.end(), [&](const uint32_t &idx) {
          return this->all_tensors_[idx]->nodeType() == NodeType_ValueNode &&
                 this->all_tensors_[idx]->data() != nullptr;
        })) {
      MS_LOG(ERROR) << "node output tensor node type is ValueNode, node name: " << node->name_;
      return RET_ERROR;
    }
    if (IsPartialNode(node->primitive_, schema_version_)) {
      auto subgraph_index = GetPartialGraphIndex(node->primitive_, schema_version_);
      if (static_cast<uint32_t>(subgraph_index) >= subgraph_size) {
        MS_LOG(ERROR) << "subgraph index：" << subgraph_index << " is beyond subgraph_size: " << subgraph_size;
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

int LiteModel::SubGraphVerify() const {
  auto tensor_size = this->all_tensors_.size();
  auto node_size = this->all_nodes_.size();

  if (sub_graphs_[0]->input_indices_.size() == 0 || sub_graphs_[0]->output_indices_.size() == 0) {
    MS_LOG(ERROR) << "The model has invalid input and output, please check";
    return RET_ERROR;
  }

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
    if (std::any_of(graph->node_indices_.begin(), graph->node_indices_.end(), [&](const uint32_t &idx) {
          bool repeated = std::count_if(graph->node_indices_.begin(), graph->node_indices_.end(),
                                        [&idx](const uint32_t &index) { return index == idx; }) != 1;
          return repeated || idx >= node_size;
        })) {
      MS_LOG(ERROR) << "The subgraph contains repeated nodes or the node index is beyond node_size.";
      return RET_ERROR;
    }
    auto ret = SubGraphInOutVerify(graph);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Fail to pass the subgraph input output verification.";
      return ret;
    }
  }
  return RET_OK;
}

int LiteModel::SubGraphInOutVerify(const Model::SubGraph *graph) const {
  auto from_node = [&](uint32_t cur_idx) -> bool {
    for (auto node_idx : graph->node_indices_) {
      auto node = this->all_nodes_.at(node_idx);
      if (std::any_of(node->output_indices_.begin(), node->output_indices_.end(),
                      [&cur_idx](uint32_t idx) { return cur_idx == idx; })) {
        return true;
      }
    }
    return false;
  };
  for (auto in_idx : graph->input_indices_) {
    auto in_tensor = this->all_tensors_.at(in_idx);
    bool is_from_node = from_node(in_idx);
    bool has_data = in_tensor->data() != nullptr && in_tensor->data()->data() != nullptr;
    if (is_from_node || (in_tensor->dataType() != kObjectTypeTensorType && has_data)) {
      MS_LOG(ERROR) << "The graph input is not valid.";
      return RET_ERROR;
    }
  }
  for (auto out_idx : graph->output_indices_) {
    auto tensor = this->all_tensors_.at(out_idx);
    bool is_from_node = from_node(out_idx);
    bool is_input = std::any_of(graph->input_indices_.begin(), graph->input_indices_.end(),
                                [&out_idx](uint32_t idx) { return out_idx == idx; });
    bool from_node_and_has_data = is_from_node && (tensor->data() != nullptr && tensor->data()->data() != nullptr);
    bool isolated_and_no_data = !is_from_node && (tensor->data() == nullptr || tensor->data()->data() == nullptr);
    if (!is_input && (from_node_and_has_data || isolated_and_no_data)) {
      MS_LOG(ERROR) << "The graph output is not valid.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

bool LiteModel::ModelVerify() const {
  if (this->sub_graphs_.empty()) {
    MS_LOG(ERROR) << "Model does not have a main graph.";
    return false;
  }

  if (this->input_indices_.empty()) {
    MS_LOG(ERROR) << "Model does not have inputs.";
    return false;
  }

  if (this->output_indices_.empty()) {
    MS_LOG(ERROR) << "Model does not have outputs.";
    return false;
  }

  auto all_tensors_size = this->all_tensors_.size();
  for (auto input_index : this->input_indices_) {
    if (input_index >= all_tensors_size) {
      MS_LOG(ERROR) << "Graph input indices is beyond tensor_size.";
      return false;
    }
    auto *tensor = this->all_tensors_.at(input_index);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Tensor in all tensors is nullptr.";
      return false;
    }
    // check the input data type
    if ((static_cast<const TypeId>(tensor->dataType()) <= kNumberTypeBegin ||
         static_cast<const TypeId>(tensor->dataType()) >= kNumberTypeEnd) &&
        static_cast<const TypeId>(tensor->dataType()) != kObjectTypeString) {
      MS_LOG(ERROR) << "The data type is not supported to malloc.";
      return false;
    }
  }

  if (std::any_of(output_indices_.begin(), output_indices_.end(),
                  [&all_tensors_size](const uint32_t &idx) { return idx >= all_tensors_size; })) {
    MS_LOG(ERROR) << "Graph output indices is beyond tensor_size.";
    return false;
  }

  return NodeVerify() == RET_OK && SubGraphVerify() == RET_OK;
}

int LiteModel::GenerateModelByVersion() {
  if (this->buf == nullptr) {
    MS_LOG(ERROR) << "Model buffer not inited";
    return RET_ERROR;
  }
  const void *meta_graph = nullptr;
  if (schema_version_ == SCHEMA_VERSION::SCHEMA_CUR) {
    meta_graph = reinterpret_cast<const void *>(schema::GetMetaGraph(this->buf));
  }
#ifdef ENABLE_V0
  if (schema_version_ == SCHEMA_VERSION::SCHEMA_V0) {
    meta_graph = reinterpret_cast<const void *>(schema::v0::GetMetaGraph(buf));
  }
#endif
  MS_ASSERT(meta_graph != nullptr);
  int status = RET_ERROR;
#ifdef ENABLE_MODEL_OBF
  DeObfuscator *model_deobf = nullptr;
#endif
  if (schema_version_ == SCHEMA_VERSION::SCHEMA_CUR) {
#ifdef ENABLE_MODEL_OBF
    if (IsMetaGraphObfuscated<schema::MetaGraph>(*reinterpret_cast<const schema::MetaGraph *>(meta_graph))) {
      model_deobf =
        GetModelDeObfuscator<schema::MetaGraph>(*reinterpret_cast<const schema::MetaGraph *>(meta_graph), this);
      this->model_obfuscated_ = true;
      if (model_deobf == nullptr) {
        return RET_ERROR;
      }
    }
#endif
    status = GenerateModel<schema::MetaGraph, schema::CNode>(*reinterpret_cast<const schema::MetaGraph *>(meta_graph));
  }
#ifdef ENABLE_V0
  if (schema_version_ == SCHEMA_VERSION::SCHEMA_V0) {
    status = GenerateModel<schema::v0::MetaGraph, schema::v0::CNode>(
      *reinterpret_cast<const schema::v0::MetaGraph *>(meta_graph));
  }
#endif
#ifdef ENABLE_MODEL_OBF
  if (this->model_obfuscated_) {
    MS_ASSERT(model_deobf != nullptr);
    status = DeObfuscateModel(this, model_deobf);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "deobfuscate model wrong.";
      std::cerr << "deobfuscate model wrong." << std::endl;
    }
    delete (model_deobf);
  }
#endif
  if (this->version_ != Version()) {
    MS_LOG(WARNING) << "model version is " << this->version_ << ", inference version is " << Version() << " not equal";
  }
  MS_LOG(INFO) << "MindSpore Lite inference version: " << Version();
  return status;
}

namespace {
int InitModelBuffer(LiteModel *model, const char *model_buf, size_t size, bool take_buf) {
  if (model_buf == nullptr || size == 0) {
    MS_LOG(ERROR) << "Input model buffer is nullptr.";
    return RET_INPUT_PARAM_INVALID;
  }
  MS_ASSERT(model != nullptr);
  if (take_buf) {
    model->buf = const_cast<char *>(model_buf);
  } else {
    if (size > kMaxModelBufferSize) {
      MS_LOG(ERROR) << "Input model buffer size invalid, require (0, 2GB].";
      return RET_ERROR;
    }
    model->buf = new char[size];
    if (model->buf == nullptr) {
      MS_LOG(ERROR) << "new inner model buf fail!";
      return RET_NULL_PTR;
    }
    memcpy(model->buf, model_buf, size);
  }
  model->buf_size_ = size;
  return RET_OK;
}
}  // namespace

int LiteModel::ConstructModel(const char *model_buf, size_t size, bool take_buf) {
  auto ret = InitModelBuffer(this, model_buf, size, take_buf);
  if (ret != RET_OK) {
    return ret;
  }

  flatbuffers::Verifier verify((const uint8_t *)this->buf, this->buf_size_);
  schema_version_ = VersionVerify(&verify);
  if (schema_version_ == SCHEMA_INVALID) {
    MS_LOG(ERROR) << "The model buffer is invalid and fail to create graph.";
#ifndef ENABLE_V0
    MS_LOG(ERROR) << "Maybe this is a model transferred out using the conversion tool before 1.1.0";
    MS_LOG(ERROR) << unsupport_v0_log;
#endif
    if (take_buf) {
      this->buf = nullptr;
    }
    return RET_ERROR;
  }
  int status = GenerateModelByVersion();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "fail to generate model";
    if (take_buf) {
      this->buf = nullptr;
    }
    return status;
  }
  if (!ModelVerify()) {
    MS_LOG(ERROR) << "ModelVerify failed.";
    if (take_buf) {
      this->buf = nullptr;
    }
    return RET_ERROR;
  }
  if (!PrepareInnerTensors()) {
    MS_LOG(ERROR) << "PrepareInnerTensors failed.";
    if (take_buf) {
      this->buf = nullptr;
    }
    return RET_ERROR;
  }

  return RET_OK;
}

bool LiteModel::PrepareInnerTensors() {
  if (!this->inner_all_tensors_.empty()) {
    MS_LOG(ERROR) << "Already prepared tensors";
    return false;
  }
  auto dir = GetDirectory(this->model_path_);
  this->inner_all_tensors_.resize(all_tensors_.size());
  for (size_t i = 0; i < all_tensors_.size(); i++) {
    auto tensor_wrapper = new (std::nothrow) SchemaTensorWrapper();
    if (tensor_wrapper == nullptr) {
      MS_LOG(ERROR) << "Create SchemaTensorWrapper return nullptr";
      return false;
    }
    if (!tensor_wrapper->Init(*(all_tensors_.at(i)), static_cast<SCHEMA_VERSION>(schema_version_), dir)) {
      delete tensor_wrapper;
      return false;
    }
    this->inner_all_tensors_[i] = tensor_wrapper;
  }
  return true;
}

SchemaTensorWrapper *LiteModel::GetSchemaTensor(const size_t &tensor_index) const {
  if (tensor_index >= this->inner_all_tensors_.size()) {
    return nullptr;
  }
  return this->inner_all_tensors_.at(tensor_index);
}

LiteModel *LiteImportFromPath(const char *model_path) {
  if (model_path == nullptr) {
    MS_LOG(ERROR) << "The model path is nullptr";
    return nullptr;
  }
  size_t size = 0;
  auto buf = ReadFile(model_path, &size);
  if (buf == nullptr) {
    return nullptr;
  }
  auto *model = new (std::nothrow) LiteModel(model_path);
  if (model == nullptr) {
    MS_LOG(ERROR) << "new model fail!";
    return nullptr;
  }

  auto status = model->ConstructModel(buf, size, true);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "construct model failed.";
    delete model;
    return nullptr;
  }
  return model;
}

bool LiteModel::CheckQuantAllInit(
  const flatbuffers::Vector<flatbuffers::Offset<mindspore::schema::QuantParam>> *quant_params) {
  if (quant_params == nullptr) {
    return false;
  }
  for (size_t i = 0; i < quant_params->size(); i++) {
    auto quant_param = quant_params->Get(i);
    if (quant_param != nullptr && quant_param->inited() == false) {
      return false;
    }
  }
  return true;
}

Model *ImportFromPath(const char *model_path) { return LiteImportFromPath(model_path); }

Model *ImportFromBuffer(const char *model_buf, size_t size, bool take_buf) {
  auto *model = new (std::nothrow) LiteModel();
  if (model == nullptr) {
    MS_LOG(ERROR) << "new model fail!";
    return nullptr;
  }

  auto status = model->ConstructModel(model_buf, size, take_buf);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "construct model failed.";
    delete model;
    return nullptr;
  }
  return model;
}

Model *Model::Import(const char *model_buf, size_t size) { return ImportFromBuffer(model_buf, size, false); }

Model *Model::Import(const char *filename) { return ImportFromPath(filename); }

int Model::Export(Model *model, char *buffer, size_t *len) {
  if (len == nullptr) {
    MS_LOG(ERROR) << "len is nullptr";
    return RET_ERROR;
  }
  auto *liteModel = reinterpret_cast<LiteModel *>(model);

  if (liteModel->buf_size_ == 0 || liteModel->buf == nullptr) {
    MS_LOG(ERROR) << "model buffer is invalid";
    return RET_ERROR;
  }
  if (*len < liteModel->buf_size_ && buffer != nullptr) {
    MS_LOG(ERROR) << "Buffer is too small, Export Failed";
    return RET_ERROR;
  }
  if (buffer == nullptr) {
    buffer = reinterpret_cast<char *>(malloc(liteModel->buf_size_));
    if (buffer == nullptr) {
      MS_LOG(ERROR) << "allocated model buf fail!";
      return RET_ERROR;
    }
  }
  memcpy(buffer, liteModel->buf, liteModel->buf_size_);
  *len = liteModel->buf_size_;
  return RET_OK;
}

int Model::Export(Model *model, const char *filename) {
  auto *liteModel = reinterpret_cast<LiteModel *>(model);
  if (liteModel->buf_size_ == 0 || liteModel->buf == nullptr) {
    MS_LOG(ERROR) << "model buf is invalid";
    return RET_ERROR;
  }

  std::ofstream ofs(filename);
  if (!ofs.good() || !ofs.is_open()) {
    MS_LOG(ERROR) << "Could not open file \"" << filename << "\" for writing";
    return RET_ERROR;
  }

  ofs.seekp(0, std::ios::beg);
  ofs.write(liteModel->buf, liteModel->buf_size_);
  ofs.close();
#ifdef SUPPORT_MSVC
  return RET_OK;
#else
  return chmod(filename, S_IRUSR);
#endif
}
}  // namespace mindspore::lite
