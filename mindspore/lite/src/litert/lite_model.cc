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

#include "src/litert/lite_model.h"
#include <sys/stat.h>
#include <iostream>
#include <sstream>
#include <functional>
#include <vector>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <numeric>
#include "src/common/prim_util.h"
#include "src/common/graph_util.h"
#include "src/common/file_utils.h"
#include "src/tensor.h"
#include "extendrt/mindir_loader/model_loader.h"
#include "src/common/mmap_utils.h"

namespace mindspore::lite {
namespace {
constexpr size_t kMaxModelBufferSize = static_cast<size_t>(1024) * 1024 * 1024 * 2;
}

void LiteModel::Free() {
  if (this->model_buf_by_mmap_) {
    UnmapMmapBuffer(static_cast<void *>(this->buf), this->buf_size_);
    this->buf = nullptr;
  }
  if (this->buf != nullptr && !this->model_buf_by_mmap_) {
    delete[](this->buf);
    this->buf = nullptr;
  }
  auto nodes_size = this->graph_.all_nodes_.size();
  for (size_t i = 0; i < nodes_size; ++i) {
    auto node = this->graph_.all_nodes_[i];
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
  for (auto &prim : graph_.deobf_prims_) {
    free(prim);
  }
  graph_.deobf_prims_.resize(0);
#endif
}

void LiteModel::Destroy() {
  Free();
  auto nodes_size = this->graph_.all_nodes_.size();
  for (size_t i = 0; i < nodes_size; ++i) {
    auto node = this->graph_.all_nodes_[i];
    MS_ASSERT(node != nullptr);
    delete node;
  }
  this->graph_.all_nodes_.clear();

  auto sub_graph_size = this->graph_.sub_graphs_.size();
  for (size_t i = 0; i < sub_graph_size; ++i) {
    auto sub_graph = this->graph_.sub_graphs_[i];
    delete sub_graph;
  }
}

int LiteModel::ConvertSubGraph(const schema::SubGraph &sub_graph) {
  if (sub_graph.name() == nullptr || sub_graph.inputIndices() == nullptr || sub_graph.outputIndices() == nullptr ||
      sub_graph.tensorIndices() == nullptr) {
    MS_LOG(ERROR) << "sub_graph is invalid";
    return RET_ERROR;
  }

  auto *subgraph = new (std::nothrow) LiteGraph::SubGraph();
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
  this->graph_.sub_graphs_.push_back(subgraph);
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
  return SCHEMA_VERSION::SCHEMA_INVALID;
}

int LiteModel::NodeVerify() const {
  auto tensor_size = this->graph_.all_tensors_.size();
  uint32_t node_size = this->graph_.all_nodes_.size();
  uint32_t subgraph_size = static_cast<uint32_t>(this->graph_.sub_graphs_.size());

  for (uint32_t node_index = 0; node_index < node_size; node_index++) {
    auto &node = this->graph_.all_nodes_.at(node_index);
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
    if (std::any_of(node->output_indices_.begin(), node->output_indices_.end(), [&, this](const uint32_t &idx) {
          return this->graph_.all_tensors_[idx]->nodeType() == static_cast<int>(NodeType_ValueNode) &&
                 this->graph_.all_tensors_[idx]->data() != nullptr;
        })) {
      MS_LOG(ERROR) << "node output tensor node type is ValueNode, node name: " << node->name_;
      return RET_ERROR;
    }
    if (node->output_indices_.size() !=
        std::unordered_set<uint32_t>(node->output_indices_.begin(), node->output_indices_.end()).size()) {
      MS_LOG(ERROR) << "node output indices contain duplicate.";
      return RET_ERROR;
    }

    if (IsPartialNode(node->primitive_, schema_version_)) {
      auto partial_fusion = reinterpret_cast<const schema::Primitive *>(node->primitive_)->value_as_PartialFusion();
      MS_CHECK_FALSE(partial_fusion == nullptr, RET_ERROR);
      int64_t subgraph_index = partial_fusion->sub_graph_index();
      if (subgraph_index < 0) {
        MS_LOG(ERROR) << "invalid subgraph index：" << subgraph_index;
        return RET_ERROR;
      }
      if (subgraph_index >= static_cast<int64_t>(subgraph_size)) {
        MS_LOG(ERROR) << "subgraph index：" << subgraph_index << " is beyond subgraph_size: " << subgraph_size;
        return RET_ERROR;
      }
      for (uint32_t graph_index = 0; graph_index < subgraph_size; graph_index++) {
        auto &graph = this->graph_.sub_graphs_.at(graph_index);
        if (IsContain(graph->node_indices_, node_index) && graph_index == static_cast<uint32_t>(subgraph_index)) {
          MS_LOG(ERROR) << "The subgraph called by PartialNode is the subgraph where it is located, subgraph index: "
                        << subgraph_index;
          return RET_ERROR;
        }
      }
    }
    if ((!IsTensorListNode(node->primitive_, schema_version_)) && (!IsPartialNode(node->primitive_, schema_version_))) {
      if (std::any_of(node->input_indices_.begin(), node->input_indices_.end(), [this](const uint32_t &idx) {
            return TypeId(this->graph_.all_tensors_[idx]->dataType()) == kObjectTypeTensorType;
          })) {
        MS_LOG(ERROR) << "node input tensor type can't be object type, node name: " << node->name_;
        return RET_ERROR;
      }
      if (std::any_of(node->output_indices_.begin(), node->output_indices_.end(), [this](const uint32_t &idx) {
            return TypeId(this->graph_.all_tensors_[idx]->dataType()) == kObjectTypeTensorType;
          })) {
        MS_LOG(ERROR) << "node output tensor type can't be object type, node name: " << node->name_;
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

int LiteModel::SubGraphVerify() const {
  auto tensor_size = this->graph_.all_tensors_.size();
  auto node_size = this->graph_.all_nodes_.size();

  if (graph_.sub_graphs_[0]->input_indices_.size() == 0 || graph_.sub_graphs_[0]->output_indices_.size() == 0) {
    MS_LOG(ERROR) << "The model has invalid input and output, please check";
    return RET_ERROR;
  }

  for (auto &graph : this->graph_.sub_graphs_) {
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

int LiteModel::GraphInOutVerify() const {
  std::unordered_set<uint32_t> all_subgraphs_inputs;
  std::unordered_set<uint32_t> all_subgraphs_outputs;
  for (auto subgraph : this->graph_.sub_graphs_) {
    for (auto input_idx : subgraph->input_indices_) {
      all_subgraphs_inputs.emplace(input_idx);
    }
    for (auto output_idx : subgraph->output_indices_) {
      all_subgraphs_outputs.emplace(output_idx);
    }
  }

  for (auto input_idx : this->graph_.input_indices_) {
    if (all_subgraphs_inputs.count(input_idx) == 0) {
      MS_LOG(ERROR) << "The graph input is not valid.";
      return RET_ERROR;
    }
  }

  for (auto output_idx : this->graph_.output_indices_) {
    if (all_subgraphs_outputs.count(output_idx) == 0) {
      MS_LOG(ERROR) << "The graph output is not valid.";
      return RET_ERROR;
    }
  }

  return RET_OK;
}

int LiteModel::SubGraphInOutVerify(const LiteGraph::SubGraph *graph) const {
  auto from_node = [&, this](uint32_t cur_idx) -> bool {
    for (auto node_idx : graph->node_indices_) {
      auto node = this->graph_.all_nodes_.at(node_idx);
      if (std::any_of(node->output_indices_.begin(), node->output_indices_.end(),
                      [&cur_idx](uint32_t idx) { return cur_idx == idx; })) {
        return true;
      }
    }
    return false;
  };
  for (auto in_idx : graph->input_indices_) {
    auto in_tensor = this->graph_.all_tensors_.at(in_idx);
    bool is_from_node = from_node(in_idx);
    bool has_data = in_tensor->data() != nullptr && in_tensor->data()->data() != nullptr;
    if (is_from_node || (in_tensor->dataType() != kObjectTypeTensorType && has_data)) {
      MS_LOG(ERROR) << "The graph input is not valid.";
      return RET_ERROR;
    }
  }
  for (auto out_idx : graph->output_indices_) {
    auto tensor = this->graph_.all_tensors_.at(out_idx);
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
  if (this->graph_.sub_graphs_.empty()) {
    MS_LOG(ERROR) << "Model does not have a main graph.";
    return false;
  }

  if (this->graph_.input_indices_.empty()) {
    MS_LOG(ERROR) << "Model does not have inputs.";
    return false;
  }

  if (this->graph_.output_indices_.empty()) {
    MS_LOG(ERROR) << "Model does not have outputs.";
    return false;
  }

  if (this->graph_.input_indices_ == this->graph_.output_indices_) {
    MS_LOG(ERROR) << "Model outputs can not be totally same as the inputs.";
    return false;
  }

  auto all_tensors_size = this->graph_.all_tensors_.size();
  for (auto input_index : this->graph_.input_indices_) {
    if (input_index >= all_tensors_size) {
      MS_LOG(ERROR) << "Graph input indices is beyond tensor_size.";
      return false;
    }
    auto *tensor = this->graph_.all_tensors_.at(input_index);
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
  if (this->graph_.output_indices_.size() == 1 &&
      graph_.sub_graphs_[0]->output_indices_.size() != graph_.output_indices_.size()) {
    MS_LOG(ERROR) << "should be equal";
    return false;
  }

  if (std::any_of(graph_.output_indices_.begin(), graph_.output_indices_.end(),
                  [&all_tensors_size](const uint32_t &idx) { return idx >= all_tensors_size; })) {
    MS_LOG(ERROR) << "Graph output indices is beyond tensor_size.";
    return false;
  }

  if (GraphInOutVerify() != RET_OK) {
    MS_LOG(ERROR) << "The model has invalid input and output.";
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
  MS_ASSERT(meta_graph != nullptr);
  int status = RET_ERROR;
#ifdef ENABLE_MODEL_OBF
  DeObfuscator *model_deobf = nullptr;
#endif
  if (schema_version_ == SCHEMA_VERSION::SCHEMA_CUR) {
#ifdef ENABLE_MODEL_OBF
    if (IsMetaGraphObfuscated<schema::MetaGraph>(*reinterpret_cast<const schema::MetaGraph *>(meta_graph))) {
      model_deobf = GetModelDeObfuscator<schema::MetaGraph>(*reinterpret_cast<const schema::MetaGraph *>(meta_graph),
                                                            this, this->buf_size_);
      this->graph_.model_obfuscated_ = true;
      if (model_deobf == nullptr) {
        return RET_ERROR;
      }
    }
#endif
    status = GenerateModel<schema::MetaGraph, schema::CNode>(*reinterpret_cast<const schema::MetaGraph *>(meta_graph));
  }
#ifdef ENABLE_MODEL_OBF
  if (this->graph_.model_obfuscated_) {
    MS_ASSERT(model_deobf != nullptr);
    status = DeObfuscateModel(this, model_deobf);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "deobfuscate model wrong.";
      std::cerr << "deobfuscate model wrong." << std::endl;
    }
    delete (model_deobf);
  }
#endif
  if (this->graph_.version_ != Version()) {
    MS_LOG(INFO) << "model version is " << this->graph_.version_ << ", inference version is " << Version()
                 << " not equal";
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

#ifdef ENABLE_LITE_HELPER
int LiteModel::ConstructModel(const char *model_buf, size_t size, bool take_buf,
                              mindspore::infer::helper::InferHelpers *infer_helpers) {
#else
int LiteModel::ConstructModel(const char *model_buf, size_t size, bool take_buf) {
#endif
  auto ret = InitModelBuffer(this, model_buf, size, take_buf);
  if (ret != RET_OK) {
    return ret;
  }

  flatbuffers::Verifier verify((const uint8_t *)this->buf, this->buf_size_, INT32_MAX, INT32_MAX);
  schema_version_ = VersionVerify(&verify);
  if (schema_version_ == SCHEMA_INVALID) {
    MS_LOG(ERROR) << "The model buffer is invalid and fail to create graph.";
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
#ifdef ENABLE_LITE_HELPER
  if (!PrepareInnerTensors(infer_helpers)) {
#else
  if (!PrepareInnerTensors()) {
#endif
    MS_LOG(ERROR) << "PrepareInnerTensors failed.";
    if (take_buf) {
      this->buf = nullptr;
    }
    return RET_ERROR;
  }

  return RET_OK;
}

#ifdef ENABLE_LITE_HELPER
bool LiteModel::PrepareInnerTensors(mindspore::infer::helper::InferHelpers *infer_helpers) {
#else
bool LiteModel::PrepareInnerTensors() {
#endif
  if (!this->inner_all_tensors_.empty()) {
    MS_LOG(ERROR) << "Already prepared tensors";
    return false;
  }
  auto dir = GetDirectory(this->model_path_);
  this->inner_all_tensors_.resize(graph_.all_tensors_.size());
  for (size_t i = 0; i < graph_.all_tensors_.size(); i++) {
    auto tensor_wrapper = new (std::nothrow) SchemaTensorWrapper();
    if (tensor_wrapper == nullptr) {
      MS_LOG(ERROR) << "Create SchemaTensorWrapper return nullptr";
      return false;
    }
#ifdef ENABLE_LITE_HELPER
    if (!tensor_wrapper->Init(*(graph_.all_tensors_.at(i)), static_cast<SCHEMA_VERSION>(schema_version_), dir,
                              infer_helpers)) {
#else
    if (!tensor_wrapper->Init(*(graph_.all_tensors_.at(i)), static_cast<SCHEMA_VERSION>(schema_version_), dir)) {
#endif
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

#ifdef ENABLE_LITE_HELPER
Model *ImportFromBuffer(const char *model_buf, size_t size, bool take_buf, mindspore::ModelType model_type,
                        const std::string &path, mindspore::infer::helper::InferHelpers *infer_helpers) {
#else
Model *ImportFromBuffer(const char *model_buf, size_t size, bool take_buf, mindspore::ModelType model_type,
                        const std::string &path) {
#endif
  auto model_loader = mindspore::infer::ModelLoaderRegistry::GetInstance()->GetModelLoader(model_type);
  if (model_loader != nullptr) {
    MS_LOG(INFO) << "import model from model loader";
    auto model = model_loader->ImportModel(model_buf, size, true);
    if (model != nullptr) {
      return model;
    }
  }

  MS_LOG(INFO) << "import model from lite model";
  auto *model = new (std::nothrow) LiteModel(path);
  if (model == nullptr) {
    MS_LOG(ERROR) << "new model fail!";
    return nullptr;
  }
#ifdef ENABLE_LITE_HELPER
  auto status = model->ConstructModel(model_buf, size, take_buf, infer_helpers);
#else
  auto status = model->ConstructModel(model_buf, size, take_buf);
#endif
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
#ifdef _MSC_VER
  return RET_OK;
#else
  return chmod(filename, S_IRUSR);
#endif
}

std::string ModelDebugString(Model *model) {
  if (model == nullptr) {
    return "";
  }
  std::ostringstream oss;
  std::string deli = "\n";
  oss << "{" << deli;
  oss << "model_type: " << model->model_type_ << deli;

  // debug graph
  oss << "graph: {" << deli;
  oss << "name: " << model->graph_.name_ << deli;
  oss << "version: " << model->graph_.version_;

  // input indices
  oss << "input_indices: [" << deli;
  for (auto i : model->graph_.input_indices_) {
    oss << i << ", " << deli;
  }
  oss << "]" << deli;

  // output indices
  oss << "output_indices: [" << deli;
  for (auto i : model->graph_.output_indices_) {
    oss << i << ", " << deli;
  }
  oss << "]" << deli;

  // all tensors
  oss << "all_tensors: [" << deli;
  for (auto tensor : model->graph_.all_tensors_) {
    oss << "{" << tensor->name() << "}";
  }
  oss << "]" << deli;

  // all nodes
  oss << "all_nodes: [" << deli;
  for (auto node : model->graph_.all_nodes_) {
    oss << "{" << deli;
    oss << "name: " << node->name_ << deli;
    oss << "op_type: " << node->op_type_ << deli;
    oss << "node_type: " << node->node_type_ << deli;
    oss << "input: [";
    for (auto i : node->input_indices_) {
      oss << i << ", ";
    }
    oss << "]" << deli;
    oss << "output: [";
    for (auto i : node->output_indices_) {
      oss << i << ", ";
    }
    oss << "]" << deli;

    // // primitive
    // auto *primitive = reinterpret_cast<schema

    oss << "}" << deli;
  }
  oss << "]" << deli;

  oss << "}" << deli;
  oss << "}" << deli;

  // dump
  return oss.str();
}
}  // namespace mindspore::lite
