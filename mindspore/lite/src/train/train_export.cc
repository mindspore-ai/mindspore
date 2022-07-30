/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "src/train/train_export.h"
#include <unistd.h>
#include <sys/stat.h>
#include <fstream>
#include <utility>
#include <queue>
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include "schema/inner/model_generated.h"
#include "src/train/train_utils.h"
#include "src/common/quant_utils.h"
#include "src/common/storage.h"
#include "src/train/graph_fusion.h"
#include "src/train/graph_dropout.h"
#include "src/litert/weight_decoder.h"

namespace mindspore {
namespace lite {
namespace {
constexpr static int kFmkVal = 3;
constexpr static int kTransformTensorDim = 4;
std::vector<size_t> GetLinkedPostIdx(const schema::MetaGraphT &graphT, const size_t &tensorIdx) {
  std::vector<size_t> postNodeIdx;
  for (size_t i = 0; i < graphT.nodes.size(); i++) {
    auto &oldNode = graphT.nodes.at(i);
    if (oldNode == nullptr) {
      continue;
    }
    auto inputIndexes = oldNode->inputIndex;
    if (IsContain<uint32_t>(inputIndexes, tensorIdx)) {
      postNodeIdx.emplace_back(i);
    }
  }
  return postNodeIdx;
}

std::vector<size_t> GetOutputNodeIdx(const schema::MetaGraphT &graphT, const schema::CNodeT &node,
                                     const int outputIndexIdx = -1) {
  std::vector<uint32_t> outputIndexes;
  if (outputIndexIdx == -1) {
    outputIndexes = node.outputIndex;
  } else {
    outputIndexes.emplace_back(node.outputIndex.at(outputIndexIdx));
  }
  std::set<size_t> outputNodeIdx;
  for (uint32_t outputIdx : outputIndexes) {
    auto linkedPostIdx = GetLinkedPostIdx(graphT, outputIdx);
    outputNodeIdx.insert(linkedPostIdx.begin(), linkedPostIdx.end());
  }
  std::vector<size_t> ret;
  ret.insert(ret.end(), outputNodeIdx.begin(), outputNodeIdx.end());
  return ret;
}
}  // namespace

std::vector<uint8_t> TrainExport::CreateData(const lite::Tensor *tensor) {
  uint8_t *tensor_data = reinterpret_cast<uint8_t *>(tensor->data());
  auto size = tensor->Size();
  std::vector<uint8_t> data(tensor_data, tensor_data + size);
  return data;
}

bool TrainExport::NeedQuantization(const lite::Tensor *t, const int tensor_quant_type) {
  return ((quant_type_ == QT_WEIGHT && t->shape().size() > 1) ||
          ((quant_type_ == QT_DEFAULT) && (tensor_quant_type == schema::QuantType_QUANT_WEIGHT)));
}

schema::QuantType TrainExport::GetNodeQuantType(const mindspore::kernel::KernelExec *kernel) {
  return static_cast<schema::QuantType>(kernel->op_parameter()->quant_type_);
}

void TrainExport::TagQuantizedNodes() {
  if (quant_type_ == QT_WEIGHT) {
    for (auto &node : meta_graph_->nodes) {
      if (node->quantType != schema::QuantType_QUANT_WEIGHT) {
        for (auto t_idx : node->inputIndex) {
          if ((meta_graph_->allTensors.at(t_idx)->nodeType == NodeType_ValueNode) &&
              (meta_graph_->allTensors.at(t_idx)->quantParams.size() > 0)) {
            node->quantType = schema::QuantType_QUANT_WEIGHT;
          }
        }
      }
    }
  }
}

int TrainExport::QuantTensorData(schema::TensorT *dest_tensor, const lite::Tensor *src_tensor, int preferred_dim) {
  int channels = 1;
  int bit_num = 8;

  if (src_tensor->quant_params().size() > 0) {
    channels = src_tensor->quant_params().size();
    bit_num = src_tensor->quant_params().at(0).bitNum;
  }
  if (channels < 1) {
    MS_LOG(ERROR) << "Quant Params is empty";
    return RET_ERROR;
  }
  int quant_max = QuantMax(bit_num, false);
  int quant_min = QuantMin(bit_num, false);
  std::vector<int8_t> data(src_tensor->ElementsNum());
  std::vector<schema::QuantParamT> quant_params;

  STATUS ret = RET_OK;
  if (channels == kPerTensor) {
    ret = DoPerLayerQuant<int8_t>(reinterpret_cast<float *>(src_tensor->data()), src_tensor->ElementsNum(),
                                  &(quant_params), quant_max, quant_min, bit_num, &data, false, false);
  } else {
    ret = DoPerChannelQuant<int8_t>(reinterpret_cast<float *>(src_tensor->data()), src_tensor->ElementsNum(),
                                    schema::QuantType_QUANT_WEIGHT, &(quant_params), quant_max, quant_min, bit_num,
                                    &data, dest_tensor->dims, preferred_dim, false, false);
  }
  if (ret == RET_NO_CHANGE) {
    MS_LOG(DEBUG) << "No Need to quant per channel";
    return RET_OK;
  }
  if (ret == RET_ERROR) {
    MS_LOG(ERROR) << "QuantTensorData error,  channels = " << channels;
    return ret;
  }
  if (quant_params.empty()) {
    MS_LOG(ERROR) << "quant_params empty";
    return RET_ERROR;
  }
  dest_tensor->data = std::vector<uint8_t>(data.data(), data.data() + data.size());
  dest_tensor->dataType = kNumberTypeInt8;
  dest_tensor->quantParams.clear();
  for (auto quant_param : quant_params) {
    dest_tensor->quantParams.emplace_back(std::make_unique<schema::QuantParamT>(quant_param));
  }

  return RET_OK;
}

std::unique_ptr<schema::TensorT> TrainExport::CreateTensor(const mindspore::lite::Tensor *tensor,
                                                           schema::Tensor *scTensor, int preferred_dim,
                                                           const int tensor_quant_type) {
  auto tensorT = std::make_unique<schema::TensorT>();
  tensorT->nodeType = scTensor->nodeType();
  tensorT->dims = tensor->shape();
  tensorT->format = static_cast<schema::Format>(tensor->format());
  tensorT->name = tensor->tensor_name();
  tensorT->refCount = 0;
  tensorT->offset = 0;
  tensorT->dataType = tensor->data_type();
  tensorT->enableHuffmanCode = false;
  if ((tensorT->nodeType == NodeType_ValueNode) && (scTensor->data() != nullptr) && (scTensor->data()->size() > 0)) {
    if (NeedQuantization(tensor, tensor_quant_type)) {
      auto ret = QuantTensorData(tensorT.get(), tensor, preferred_dim);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "QuantTensorData failed.";
        return nullptr;
      }
    } else {
      tensorT->data = CreateData(tensor);
    }
  }
  tensorT->quantClusters = tensor->quant_clusters();
  return tensorT;
}

LiteGraph::Node *TrainExport::FindNode(const mindspore::kernel::KernelExec *kernel, const Model *model) {
  auto nodes = model->graph_.all_nodes_;
  auto it = std::find_if(nodes.begin(), nodes.end(),
                         [&kernel](mindspore::lite::LiteGraph::Node *n) { return (kernel->name() == n->name_); });
  if (it == nodes.end()) {
    return nullptr;
  }
  return *it;
}

int TrainExport::CreateAndAddCNode(const mindspore::kernel::KernelExec *kernel, std::vector<uint32_t> inputIndex,
                                   std::vector<uint32_t> outputIndex, const Model *model) {
  auto cnode = CreateCNode(kernel, inputIndex, outputIndex, model);
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "failed to create cnode";
    return RET_ERROR;
  }
  meta_graph_->nodes.emplace_back(std::move(cnode));
  if (!meta_graph_->subGraph.empty()) {
    meta_graph_->subGraph[0]->nodeIndices.push_back(meta_graph_->nodes.size() - 1);
  }
  return RET_OK;
}

std::unique_ptr<schema::CNodeT> TrainExport::CreateCNode(const mindspore::kernel::KernelExec *kernel,
                                                         std::vector<uint32_t> inputIndex,
                                                         std::vector<uint32_t> outputIndex, const Model *model) {
  auto cnodeT = std::make_unique<schema::CNodeT>();
  if (cnodeT == nullptr) {
    MS_LOG(ERROR) << " cannot allocate node";
    return nullptr;
  }
  cnodeT->inputIndex = inputIndex;
  cnodeT->outputIndex = outputIndex;
  cnodeT->name = kernel->name();
  cnodeT->quantType = GetNodeQuantType(kernel);
  // find kernel in model
  auto *node = FindNode(kernel, model);
  if (node == nullptr) {
    MS_LOG(ERROR) << "cannot find kernel " + kernel->name() + " in model";
    return nullptr;
  }
  auto primitive = reinterpret_cast<schema::Primitive *>(const_cast<void *>(node->primitive_));
  cnodeT->primitive = std::unique_ptr<schema::PrimitiveT>(primitive->UnPack());
  return cnodeT;
}

int TrainExport::LoadModel(void *buf, size_t buf_size) {
  flatbuffers::Verifier verify((const uint8_t *)buf, buf_size);
  if (!schema::VerifyMetaGraphBuffer(verify)) {
    MS_LOG(ERROR) << "model flatbuffer verify fail";
    return RET_ERROR;
  }
  meta_graph_ = schema::GetMetaGraph(buf)->UnPack();
  meta_graph_->outputIndex.clear();
  if (!meta_graph_->subGraph.empty()) {
    meta_graph_->subGraph[0]->outputIndices.clear();
  }
  return RET_OK;
}

std::unique_ptr<schema::TensorT> TrainExport::CreateTransformTensor(size_t id) {
  auto &scTensor = meta_graph_->allTensors.at(id);
  auto tensorT = std::make_unique<schema::TensorT>();
  if (tensorT == nullptr) {
    MS_LOG(ERROR) << "Could not create tensor ";
    return nullptr;
  }
  tensorT->nodeType = scTensor->nodeType;
  tensorT->dataType = scTensor->dataType;
  std::vector<int32_t> dims;
  std::vector<int32_t> val = {0, 2, 3, 1};
  if (scTensor->dims.size() == kTransformTensorDim) {
    for (size_t i = 0; i < val.size(); i++) {
      dims.push_back(scTensor->dims.at(val[i]));
    }
    tensorT->dims = dims;
  } else {
    tensorT->dims = scTensor->dims;
  }
  tensorT->format = schema::Format_NHWC;
  tensorT->name = scTensor->name + "_post";
  tensorT->refCount = 0;
  tensorT->offset = 0;
  tensorT->enableHuffmanCode = false;
  return tensorT;
}

std::unique_ptr<schema::TensorT> TrainExport::CreateTransformConst(size_t last_id) {
  auto tensorT = std::make_unique<schema::TensorT>();
  if (tensorT == nullptr) {
    MS_LOG(ERROR) << "Could not create tensor ";
    return nullptr;
  }
  tensorT->nodeType = lite::NodeType_ValueNode;
  tensorT->dataType = TypeId::kNumberTypeInt32;
  tensorT->dims = {kTransformTensorDim};
  tensorT->format = schema::Format_NCHW;
  tensorT->name = "const-" + std::to_string(last_id);
  tensorT->refCount = 0;
  tensorT->offset = 0;
  tensorT->enableHuffmanCode = false;
  int32_t val[] = {0, 2, 3, 1};
  uint8_t *valp = reinterpret_cast<uint8_t *>(val);
  tensorT->data = std::vector<uint8_t>(valp, valp + sizeof(val));
  return tensorT;
}

std::unique_ptr<schema::CNodeT> TrainExport::CreateTransformNode(std::vector<uint32_t> inputIndex,
                                                                 std::vector<uint32_t> outputIndex, size_t id) {
  auto cnodeT = std::make_unique<schema::CNodeT>();
  if (cnodeT == nullptr) {
    MS_LOG(ERROR) << "cannot allocate node";
    return nullptr;
  }
  cnodeT->inputIndex = inputIndex;
  cnodeT->outputIndex = outputIndex;
  cnodeT->name = "transpose-" + std::to_string(id);
  cnodeT->quantType = schema::QuantType_QUANT_NONE;
  cnodeT->primitive = std::make_unique<schema::PrimitiveT>();
  cnodeT->primitive->value.type = schema::PrimitiveType_Transpose;
  return cnodeT;
}

int TrainExport::AddTransformNode() {
  std::unordered_map<size_t, size_t> reconnect;
  size_t last_id = meta_graph_->allTensors.size();
  size_t last_node = meta_graph_->nodes.size();
  for (auto it : connect_) {
    auto tensorConst = CreateTransformConst(last_id);
    if (tensorConst == nullptr) {
      MS_LOG(ERROR) << "error in create tensor";
      return RET_ERROR;
    }
    meta_graph_->allTensors.emplace_back(std::move(tensorConst));  // last_id
    if (!meta_graph_->subGraph.empty()) {
      meta_graph_->subGraph[0]->tensorIndices.push_back(meta_graph_->allTensors.size() - 1);
    }
    auto tensorT = CreateTransformTensor(it.second);
    if (tensorT == nullptr) {
      MS_LOG(ERROR) << "error in create tensor";
      return RET_ERROR;
    }
    meta_graph_->allTensors.emplace_back(std::move(tensorT));  // last_id + 1
    if (!meta_graph_->subGraph.empty()) {
      meta_graph_->subGraph[0]->tensorIndices.push_back(meta_graph_->allTensors.size() - 1);
    }
    std::vector<uint32_t> in_idx = {static_cast<uint32_t>(it.second), static_cast<uint32_t>(last_id)};
    std::vector<uint32_t> out_idx = {static_cast<uint32_t>(last_id + 1)};
    reconnect[it.first] = last_id + 1;
    auto cnode = CreateTransformNode(in_idx, out_idx, last_node);
    if (cnode == nullptr) {
      MS_LOG(ERROR) << "error in node creation";
      return RET_ERROR;
    }
    meta_graph_->nodes.emplace_back(std::move(cnode));
    if (!meta_graph_->subGraph.empty()) {
      meta_graph_->subGraph[0]->nodeIndices.push_back(meta_graph_->nodes.size() - 1);
    }
  }
  connect_ = reconnect;
  return RET_OK;
}

void TrainExport::PrepareRemap(int offset) {
  for (auto it : connect_) {
    remap_[it.first + offset] = it.second;
  }
}

int TrainExport::FindSchemaTensorByName(const std::vector<uint32_t> &search_indices, const std::string &search_name,
                                        size_t *target_index) {
  MS_CHECK_TRUE_MSG(target_index != nullptr, RET_ERROR, "input param target_index is nullptr.");
  auto total_size = meta_graph_->allTensors.size();
  for (auto index : search_indices) {
    MS_CHECK_TRUE_MSG(index < total_size, RET_ERROR, "index is out of range.");
    if (meta_graph_->allTensors[index]->name == search_name) {
      *target_index = index;
      return RET_OK;
    }
  }
  return RET_NO_CHANGE;
}

int TrainExport::KeepGraphInputsInOrder(const Model *model) {
  MS_CHECK_TRUE_MSG(model != nullptr, RET_ERROR, "input param model is nullptr.");
  MS_CHECK_TRUE_MSG(meta_graph_->inputIndex.size() <= model->graph_.input_indices_.size(), RET_ERROR,
                    "export model input indices size is large than origin input indices size.");
  std::vector<uint32_t> origin_inputs_order;
  for (auto index : model->graph_.input_indices_) {
    MS_CHECK_TRUE_MSG(index < model->graph_.all_tensors_.size(), RET_ERROR, "input index out of range.");
    auto ori_input_tensor = model->graph_.all_tensors_[index];
    size_t meta_graph_input_index;
    auto status =
      FindSchemaTensorByName(meta_graph_->inputIndex, ori_input_tensor->name()->str(), &meta_graph_input_index);
    if (status == RET_NO_CHANGE) {
      MS_LOG(DEBUG) << "can't find tensor: " << ori_input_tensor->name()->str() << " in exported graph.";
      continue;
    } else if (status != RET_OK) {
      MS_LOG(ERROR) << "find schema tensor failed.";
      return RET_ERROR;
    }
    MS_CHECK_TRUE_MSG(status != RET_ERROR, RET_ERROR, "find graph input tensor failed.");
    origin_inputs_order.emplace_back(meta_graph_input_index);
  }
  meta_graph_->inputIndex = origin_inputs_order;
  if (!meta_graph_->subGraph.empty()) {
    MS_CHECK_TRUE_MSG(meta_graph_->subGraph[0]->inputIndices.size() == origin_inputs_order.size(), RET_ERROR,
                      "metagraph's subgraph input indices size is invalid.");
    meta_graph_->subGraph[0]->inputIndices = origin_inputs_order;
  }
  return RET_OK;
}
int TrainExport::ExportTensor(const Model *model, const std::vector<mindspore::lite::Tensor *> &tensors, int offset,
                              const std::vector<std::pair<size_t, tensor_info>> &map_index,
                              const std::vector<std::string> &output_names, const std::set<size_t> &out_set) {
  std::vector<mindspore::lite::Tensor *> in_tensors;
  for (auto index : map_index) {
    auto id = index.first;
    size_t pid = id - static_cast<size_t>(offset);
    mindspore::lite::Tensor *tensor = tensors.at(pid);
    in_tensors.push_back(tensor);
  }
  for (auto index : map_index) {
    auto id = index.first;
    size_t pid = id - static_cast<size_t>(offset);
    mindspore::lite::Tensor *tensor = tensors.at(pid);
    schema::Tensor *scTensor = model->graph_.all_tensors_.at(pid);
    auto preferred_dim = WeightDecoder::GetPreferredDim(in_tensors, index.second.op_parameter, index.second.input_index,
                                                        tensor->shape(), model->graph_.version_);
    auto tensorT = CreateTensor(tensor, scTensor, preferred_dim, index.second.op_parameter->quant_type_);
    if (tensorT == nullptr) {
      MS_LOG(ERROR) << "error in tensor creation";
      return RET_ERROR;
    }
    if (out_set.find(remap_[id]) == out_set.end()) {
      if (IsInputTensor(*tensorT)) {
        meta_graph_->inputIndex.push_back(remap_[id]);
        if (!meta_graph_->subGraph.empty()) {
          meta_graph_->subGraph[0]->inputIndices.push_back(remap_[id]);
        }
      }
    }
    // find output tensor
    if (std::find(output_names.begin(), output_names.end(), tensor->tensor_name()) != output_names.end()) {
      meta_graph_->outputIndex.push_back(remap_[id]);
      if (!meta_graph_->subGraph.empty()) {
        meta_graph_->subGraph[0]->outputIndices.push_back(remap_[id]);
      }
    }
    meta_graph_->allTensors.emplace_back(std::move(tensorT));
    if (!meta_graph_->subGraph.empty()) {
      meta_graph_->subGraph[0]->tensorIndices.push_back(meta_graph_->allTensors.size() - 1);
    }
  }
  return RET_OK;
}

int TrainExport::ExportNet(const std::vector<mindspore::kernel::KernelExec *> &kernels,
                           const std::vector<mindspore::lite::Tensor *> &tensors,
                           const std::vector<std::string> &output_names, const Model *model,
                           QuantizationType quant_type, const Model *bb_model) {
  std::vector<std::pair<size_t, tensor_info>> map_index;
  std::set<size_t> out_set;
  if (meta_graph_ == nullptr) {
    int status = ExportInit(model->graph_.name_, model->graph_.version_);
    if (status != RET_OK) {
      return status;
    }
  }
  int offset = meta_graph_->allTensors.size();
  int tensor_idx = offset;
  quant_type_ = quant_type;
  PrepareRemap(offset);

  for (const auto kernel : kernels) {
    std::vector<uint32_t> in_idx, out_idx;
    size_t input_index = 0;
    for (const auto tensor : kernel->in_tensors()) {
      size_t id = TSFindTensor(tensors, tensor) + static_cast<size_t>(offset);
      if (id == tensors.size()) {
        MS_LOG(ERROR) << "cannot find tensor " + tensor->ToString() + " in model";
        return RET_ERROR;
      }
      auto it = remap_.find(id);
      if (it == remap_.end()) {
        remap_[id] = tensor_idx;
        in_idx.push_back(tensor_idx);
        map_index.push_back({id, {input_index++, kernel->op_parameter()}});
        tensor_idx++;
      } else {
        in_idx.push_back(it->second);
      }
    }
    size_t output_index = 0;
    for (const auto tensor : kernel->out_tensors()) {
      size_t id = TSFindTensor(tensors, tensor) + offset;
      if (id == tensors.size()) {
        MS_LOG(ERROR) << "cannot find tensor " + tensor->ToString() + " in model";
        return RET_ERROR;
      }
      auto it = remap_.find(id);
      if (it == remap_.end()) {
        remap_[id] = tensor_idx;
        map_index.push_back({id, {output_index++, kernel->op_parameter()}});
        out_idx.push_back(tensor_idx);
        out_set.insert(tensor_idx);
        tensor_idx++;
      } else {
        out_idx.push_back(it->second);
        out_set.insert(it->second);
      }
    }
    auto ret = CreateAndAddCNode(kernel, in_idx, out_idx, model);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "failed to create cnode";
      return ret;
    }
  }

  auto status = ExportTensor(model, tensors, offset, map_index, output_names, out_set);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "ExportTensor failed.";
    return RET_ERROR;
  }
  auto origin_input_model = bb_model == nullptr ? model : bb_model;
  status = KeepGraphInputsInOrder(origin_input_model);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "keep graph inputs in order failed.";
    return RET_ERROR;
  }
  TagQuantizedNodes();  // do another loop to mark QUANT_WEIGHT_NODES
  status = TopologicalSort();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "TopologicalSort failed.";
    return RET_ERROR;
  }

  return RET_OK;
}

int TrainExport::TopologicalSort() {
  MS_ASSERT(meta_graph_ != nullptr);
  std::vector<std::unique_ptr<schema::CNodeT>> new_nodes;
  std::vector<size_t> sinked_tensor_idxes;
  for (auto &subgraph : meta_graph_->subGraph) {
    std::copy(subgraph->inputIndices.begin(), subgraph->inputIndices.end(), std::back_inserter(sinked_tensor_idxes));
  }
  // put all const tensor index into sinked_tensor_idxes
  for (size_t i = 0; i < meta_graph_->allTensors.size(); i++) {
    if (meta_graph_->allTensors.at(i)->nodeType == NodeType_ValueNode) {
      sinked_tensor_idxes.push_back(i);
    }
  }
  auto &old_nodes = meta_graph_->nodes;
  std::queue<std::unique_ptr<schema::CNodeT>> op_queue;
  // put all none depend node into queue
  for (size_t i = 0; i < meta_graph_->subGraph.size(); i++) {
    std::vector<unsigned int> new_subgraph_node_indices = {};
    auto subgraph_node_indices = meta_graph_->subGraph[i]->nodeIndices;

    for (size_t j = 0; j < subgraph_node_indices.size(); j++) {
      auto &node = old_nodes[subgraph_node_indices[j]];
      if (IsNodeNonDepend(node, sinked_tensor_idxes)) {
        sinked_tensor_idxes.insert(sinked_tensor_idxes.end(), node->outputIndex.begin(), node->outputIndex.end());
        op_queue.push(std::move(node));
      }
    }
    while (!op_queue.empty()) {
      auto &node = op_queue.front();
      auto post_node_idxes = GetOutputNodeIdx(*meta_graph_, *(node.get()));
      sinked_tensor_idxes.insert(sinked_tensor_idxes.end(), node->outputIndex.begin(), node->outputIndex.end());
      for (auto post_node_idx : post_node_idxes) {
        if (IsContain(subgraph_node_indices, (unsigned int)(post_node_idx))) {
          auto &post_node = old_nodes.at(post_node_idx);
          // check if post_node is non-depended
          if (IsNodeNonDepend(post_node, sinked_tensor_idxes)) {
            op_queue.push(std::move(post_node));
          }
        }
      }
      new_nodes.emplace_back(std::move(node));
      new_subgraph_node_indices.push_back(new_nodes.size() - 1);
      op_queue.pop();
    }
    meta_graph_->subGraph[i]->nodeIndices.swap(new_subgraph_node_indices);
  }
  if (new_nodes.size() != old_nodes.size()) {
    MS_LOG(ERROR) << "Unknown error in TopologicalSort, old_nodes size: " << old_nodes.size()
                  << ", new_nodes size: " << new_nodes.size();
    return RET_ERROR;
  }
  meta_graph_->nodes.swap(new_nodes);
  return RET_OK;
}

bool TrainExport::IsNodeNonDepend(const std::unique_ptr<schema::CNodeT> &node,
                                  const std::vector<size_t> &sinked_tensor_idxes) {
  MS_ASSERT(node != nullptr);
  return std::all_of(node->inputIndex.begin(), node->inputIndex.end(),
                     [&](size_t input_idx) { return IsContain(sinked_tensor_idxes, size_t(input_idx)); });
}

int TrainExport::ExportInit(const std::string model_name, std::string version) {
  meta_graph_ = new (std::nothrow) schema::MetaGraphT();
  if (meta_graph_ == nullptr) {
    MS_LOG(ERROR) << "cannot allocate meta_graph";
    return RET_ERROR;
  }
  auto sub_graph = std::make_unique<schema::SubGraphT>();
  if (sub_graph == nullptr) {
    MS_LOG(ERROR) << "cannot allocate SubGraphT";
    return RET_ERROR;
  }
  sub_graph->name = model_name + "_subgraph";
  meta_graph_->subGraph.emplace_back(std::move(sub_graph));
  meta_graph_->fmkType = kFmkVal;
  meta_graph_->name = model_name;
  meta_graph_->version = version;
  return RET_OK;
}

int TrainExport::SaveModel(lite::Model *model, const std::string &file_name) {
  std::string filename = file_name;
  if (filename.substr(filename.find_last_of(".") + 1) != "ms") {
    filename = filename + ".ms";
  }
#ifndef _MSC_VER
  if (access(filename.c_str(), F_OK) == 0) {
    chmod(filename.c_str(), S_IWUSR);
  }
#endif
  int status = mindspore::lite::Model::Export(model, filename.c_str());
  return status;
}

int TrainExport::SaveToFile() { return Storage::Save(*meta_graph_, file_name_); }

bool TrainExport::IsInputTensor(const schema::TensorT &t) {
  int total_dims = std::accumulate(t.dims.begin(), t.dims.end(), 1, std::multiplies<int>());
  return ((t.data.size() == 0) && (total_dims != 0));
}

int TrainExport::TrainModelFusion() {
  GraphFusion graph_fusion;
  auto status = graph_fusion.Run(meta_graph_);
  if (status != RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
}

int TrainExport::TrainModelDrop() {
  GraphDropout graph_dropout;
  auto status = graph_dropout.Run(meta_graph_);
  if (status != RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
}

TrainExport::~TrainExport() { delete meta_graph_; }
}  // namespace lite
}  // namespace mindspore
