/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "tools/converter/legacy_optimizer/graph/infershape_pass.h"
#include <vector>
#include <deque>
#include <set>
#include "src/common/common.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"
#include "src/tensor.h"
#include "src/tensorlist.h"
#include "src/common/prim_util.h"
#include "src/ops/populate/populate_register.h"
#include "src/runtime/infer_manager.h"
#include "tools/common/node_util.h"
#include "tools/converter/converter_flags.h"
#include "src/common/string_util.h"
#include "src/common/log_util.h"
#include "nnacl/op_base.h"

using mindspore::converter::kFmkTypeTf;
namespace mindspore {
namespace lite {
namespace {
constexpr int DEFAULT_DIM_VALUE = -1;
constexpr size_t kInitialSize = 1024;
constexpr int kMainGraphIndex = 0;
constexpr int kCallInputMinSize = 1;
constexpr int kSwitchInputMinSize = 3;
constexpr int kTypeIndex = 0;
constexpr int kElementShapeIndex = 1;
constexpr int kFirstElementShapeIndex = 2;
constexpr int kTensorListDatasize = 3;

void FreeTensors(std::vector<Tensor *> *input_tensors, std::vector<Tensor *> *output_tensors) {
  if (input_tensors == nullptr) {
    return;
  }
  for (auto &tensor : *input_tensors) {
    if (tensor == nullptr) {
      continue;
    }
    if (tensor->data_type() != kObjectTypeString && tensor->data_type() != kObjectTypeTensorType) {
      tensor->set_data(nullptr);
    }
    delete tensor;
    tensor = nullptr;
  }
  if (output_tensors == nullptr) {
    return;
  }
  for (auto &tensor : *output_tensors) {
    if (tensor == nullptr) {
      continue;
    }
    if (tensor->data_type() != kObjectTypeString && tensor->data_type() != kObjectTypeTensorType) {
      tensor->set_data(nullptr);
    }
    delete tensor;
    tensor = nullptr;
  }
  input_tensors->resize(0);
  output_tensors->resize(0);
}

void ConvertTensorList(MetaGraphT *graph, uint32_t index, bool *convert_succ, std::vector<Tensor *> *lite_tensors) {
  std::unique_ptr<Tensor> lite_tensor = nullptr;
  auto &tensorT = graph->allTensors.at(index);
  std::vector<int32_t> tensor_shape{};
  TypeId type = kTypeUnknown;
  std::vector<int> element_shape;
  if (!tensorT->data.empty()) {
    auto data_len = tensorT->data.size();
    int *data = reinterpret_cast<int *>(tensorT->data.data());
    type = TypeId(data[kTypeIndex]);
    if (data_len < kTensorDataSize ||
        (data[kElementShapeIndex] != 0 && static_cast<int>((data[kElementShapeIndex] + kTensorListDatasize) *
                                                           sizeof(int)) != static_cast<int>(tensorT->data.size()))) {
      MS_LOG(ERROR) << "tensorlist data length illegal, tensorT name: " << tensorT->name;
      MS_LOG(ERROR) << "(data[1] + 3) * sizeof(int): "
                    << ((data[kElementShapeIndex] + kTensorListDatasize) * sizeof(int));
      MS_LOG(ERROR) << "static_cast<int>(tensorT->data.size()): " << static_cast<int>(tensorT->data.size());
      *convert_succ = false;
      return;
    }
    for (int j = 0; j < data[kElementShapeIndex]; ++j) {
      element_shape.push_back(data[j + kFirstElementShapeIndex]);
    }
    if (INT_ADD_OVERFLOW(data[kElementShapeIndex], kFirstElementShapeIndex)) {
      MS_LOG(ERROR) << "int add overflow";
      *convert_succ = false;
      return;
    }
    tensor_shape = {data[data[kElementShapeIndex] + kFirstElementShapeIndex]};
  }
  lite_tensor = std::make_unique<TensorList>(tensor_shape, element_shape);
  if (lite_tensor == nullptr) {
    MS_LOG(ERROR) << "lite tensorlist is nullptr";
    *convert_succ = false;
    return;
  }

  auto lite_tensor_list = reinterpret_cast<TensorList *>(lite_tensor.get());
  std::vector<Tensor *> tensors{};
  if (!tensor_shape.empty() && tensor_shape.front() == -1) {
    MS_LOG(INFO) << "tensor_shape is -1, tensor name: " << lite_tensor->tensor_name();
  }
  if (!tensor_shape.empty() && tensor_shape.front() != -1) {
    for (int32_t i = 0; i < tensor_shape.front(); ++i) {
      auto tensor = new (std::nothrow) Tensor(type, element_shape);
      tensors.emplace_back(tensor);
    }
  }

  lite_tensor_list->set_tensors_data_type(type);
  lite_tensor_list->set_element_shape(element_shape);
  lite_tensor_list->set_tensors(tensors);
  lite_tensors->emplace_back(lite_tensor.release());
}

void ConvertString(MetaGraphT *graph, uint32_t index, bool *convert_succ, std::vector<Tensor *> *lite_tensors) {
  std::unique_ptr<Tensor> lite_tensor = nullptr;
  auto &tensorT = graph->allTensors.at(index);
  auto tensor_shape = tensorT->dims;
  lite_tensor = std::make_unique<Tensor>(
    TypeId(tensorT->dataType), tensor_shape, static_cast<mindspore::Format>(tensorT->format),
    TensorCategory(tensorT->nodeType, tensorT->dims.size(), TypeId(tensorT->dataType), tensorT->data.size()));
  if (lite_tensor == nullptr) {
    MS_LOG(ERROR) << "lite tensor is nullptr";
    *convert_succ = false;
    return;
  }
  auto lite_tensor_size = tensorT->data.size() * sizeof(uint8_t);
  // when tensorT as param input
  if (lite_tensor_size == 0) {
    lite_tensors->emplace_back(lite_tensor.release());
    return;
  }
  auto string_buffer = ParseStringBuffer(tensorT->data.data());
  auto ret = WriteStringsToTensor(lite_tensor.get(), string_buffer);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "WriteStringsToTensor failed";
    *convert_succ = false;
    return;
  }
  lite_tensors->emplace_back(lite_tensor.release());
}

void ConvertOtherTensor(MetaGraphT *graph, uint32_t index, bool *convert_succ, std::vector<Tensor *> *lite_tensors) {
  std::unique_ptr<Tensor> lite_tensor = nullptr;
  auto &tensorT = graph->allTensors.at(index);
  auto tensor_shape = tensorT->dims;
  lite_tensor = std::make_unique<Tensor>(
    TypeId(tensorT->dataType), tensor_shape, static_cast<mindspore::Format>(tensorT->format),
    TensorCategory(tensorT->nodeType, tensorT->dims.size(), TypeId(tensorT->dataType), tensorT->data.size()));
  if (lite_tensor == nullptr) {
    MS_LOG(ERROR) << "lite tensor is nullptr";
    *convert_succ = false;
    return;
  }
  auto lite_tensor_size = tensorT->data.size() * sizeof(uint8_t);
  // when tensorT as param input
  if (lite_tensor_size == 0) {
    lite_tensors->emplace_back(lite_tensor.release());
    return;
  }
  lite_tensor->set_data(tensorT->data.data());
  lite_tensors->emplace_back(lite_tensor.release());
}

std::vector<Tensor *> ConvertTensorToLiteTensor(MetaGraphT *graph, const std::vector<uint32_t> &tensor_indexs) {
  MS_ASSERT(graph != nullptr);
  std::vector<Tensor *> lite_tensors;
  bool convert_succ = true;
  for (size_t i = 0; i < tensor_indexs.size(); i++) {
    auto &tensorT = graph->allTensors.at(tensor_indexs[i]);
    switch (tensorT->dataType) {
      case kObjectTypeTensorType:
        ConvertTensorList(graph, tensor_indexs[i], &convert_succ, &lite_tensors);
        break;
      case kObjectTypeString:
        ConvertString(graph, tensor_indexs[i], &convert_succ, &lite_tensors);
        break;
      default:
        ConvertOtherTensor(graph, tensor_indexs[i], &convert_succ, &lite_tensors);
        break;
    }
  }
  if (!convert_succ) {
    FreeTensors(&lite_tensors, {});
    return {};
  }
  return lite_tensors;
}

STATUS NodeInferShape(const std::unique_ptr<schema::CNodeT> &node, const std::vector<Tensor *> &inputs,
                      std::vector<Tensor *> *outputs) {
  flatbuffers::FlatBufferBuilder fbb(kInitialSize);
  auto prim = ConvertToPrimitive(node->primitive.get(), &fbb);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "get primitive failed.";
    fbb.Clear();
    return RET_ERROR;
  }

  auto ret = KernelInferShape(inputs, *outputs, prim, {}, SCHEMA_CUR);
  if (ret == lite::RET_NOT_SUPPORT) {
    auto parameter_gen =
      lite::PopulateRegistry::GetInstance()->GetParameterCreator(static_cast<int>(prim->value_type()), SCHEMA_CUR);
    if (parameter_gen == nullptr) {
      fbb.Clear();
      MS_LOG(ERROR) << "PopulateParameter return nullptr, type: " << schema::EnumNamePrimitiveType(prim->value_type());
      return RET_ERROR;
    }
    auto parameter = parameter_gen(prim);
    if (parameter == nullptr) {
      fbb.Clear();
      MS_LOG(ERROR) << "parameter is nullptr.";
      return RET_ERROR;
    }
    parameter->quant_type_ = static_cast<int>(node->quantType);
    ret = KernelInferShape(inputs, *outputs, parameter);
    if (parameter->destroy_func_ != nullptr) {
      parameter->destroy_func_(parameter);
    }
    free(parameter);
    parameter = nullptr;
  }

  fbb.Clear();
  return ret;
}

#ifdef Debug
void PrintTensorShape(const std::vector<Tensor *> &input_tensors, const std::vector<Tensor *> &output_tensors) {
  int i = 0;
  for (auto input_tensor : input_tensors) {
    std::ostringstream oss;
    for (auto &dim : input_tensor->shape()) {
      oss << " " << dim;
    }
    MS_LOG(DEBUG) << "input shape " << i++ << ":" << oss.str();
  }
  i = 0;
  for (auto output_tensor : output_tensors) {
    std::ostringstream oss;
    for (auto &dim : output_tensor->shape()) {
      oss << " " << dim;
    }
    MS_LOG(DEBUG) << "output shape" << i++ << ":" << oss.str();
  }
}
#endif

int SetDataType(MetaGraphT *graph, const std::vector<Tensor *> &output_tensors, std::vector<InferTensor> *tensors,
                uint32_t i, uint32_t infer_node_index) {
  auto &node = graph->nodes.at(infer_node_index);
  auto &output_tensor = graph->allTensors.at(node->outputIndex[i]);
  output_tensor->format = static_cast<schema::Format>(output_tensors[i]->format());
  output_tensor->dataType = output_tensors[i]->data_type();
  if (output_tensors[i]->data_type() == kObjectTypeTensorType) {
    auto tensor_list = reinterpret_cast<TensorList *>(output_tensors[i]);
    int tensor_shape_dims = 0;
    if (!tensor_list->tensors().empty()) {
      tensor_shape_dims = static_cast<int>(tensor_list->tensors().front()->shape().size());
    }
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW((tensor_shape_dims + kTensorListDatasize), static_cast<int>(sizeof(int))),
                       RET_ERROR, "int mul overflow");
    auto total_size = (tensor_shape_dims + kTensorListDatasize) * sizeof(int);
    output_tensor->data.resize(total_size, 0);
    auto output_tensor_data = reinterpret_cast<int *>(output_tensor->data.data());
    if (tensor_list->tensors_data_type() == kTypeUnknown) {
      if (!tensor_list->tensors().empty()) {
        tensor_list->set_tensors_data_type(tensor_list->tensors().front()->data_type());
      }
    }
    output_tensor_data[kTypeIndex] = tensor_list->tensors_data_type();
    if (tensor_list->element_shape().empty() && !tensor_list->tensors().empty()) {
      tensor_list->set_element_shape(tensor_list->tensors().front()->shape());
    }
    output_tensor_data[kElementShapeIndex] = static_cast<int>(tensor_list->element_shape().size());
    for (size_t j = 0; j < tensor_list->element_shape().size(); ++j) {
      output_tensor_data[j + kFirstElementShapeIndex] = tensor_list->element_shape().at(j);
    }
    output_tensor_data[kFirstElementShapeIndex + output_tensor_data[kElementShapeIndex]] =
      static_cast<int>(tensor_list->tensors().size());
  } else if (output_tensors[i]->data_type() == kTypeUnknown) {
    tensors->at(node->outputIndex[i]).is_inferred_ = false;
    return RET_OK;
  }
  tensors->at(node->outputIndex[i]).is_inferred_ = true;
  return RET_OK;
}

int PartialGraphIndex(const CNodeT *partial_node) {
  return partial_node->primitive->value.AsPartialFusion()->sub_graph_index;
}
}  // namespace

int InferShapePass::CopyPartialShapeToSubGraph(const CNodeT *partial_node, MetaGraphT *graph) {
  auto subgraph_index = PartialGraphIndex(partial_node);
  auto &subgraph = graph->subGraph.at(subgraph_index);

  if (subgraph->inputIndices.size() != partial_node->inputIndex.size()) {
    MS_LOG(ERROR) << "partial node " << partial_node->name << " inputs size: " << partial_node->inputIndex.size()
                  << " vs "
                  << " subgraph " << subgraph_index << " input size: " << subgraph->inputIndices.size();
    return RET_PARAM_INVALID;
  }

  for (size_t i = 0; i < partial_node->inputIndex.size(); ++i) {
    auto &subgraph_input = graph->allTensors.at(subgraph->inputIndices[i]);
    auto &partial_input = graph->allTensors.at(partial_node->inputIndex[i]);
    subgraph_input->dataType = partial_input->dataType;
    subgraph_input->dims = partial_input->dims;
    subgraph_input->format = partial_input->format;
    subgraph_input->data.resize(partial_input->data.size(), 0);
    if (partial_input->data.empty()) {
      continue;
    }
    auto ret = memcpy_s(subgraph_input->data.data(), subgraph_input->data.size(), partial_input->data.data(),
                        partial_input->data.size());
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy failed, ret: " << ret;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int InferShapePass::RestoreSubGraphInput(const CNodeT *partial_node, MetaGraphT *graph) {
  auto subgraph_index = PartialGraphIndex(partial_node);
  auto &subgraph = graph->subGraph.at(subgraph_index);
  for (size_t i = 0; i < subgraph->inputIndices.size(); ++i) {
    auto &subgraph_input = graph->allTensors.at(subgraph->inputIndices[i]);
    if (subgraph_input->dataType != kObjectTypeTensorType) {
      subgraph_input->data = {};
    }
  }
  return RET_OK;
}

int InferShapePass::InferPartialNode(const CNodeT *partial_node, MetaGraphT *graph) {
  int subgraph_index = PartialGraphIndex(partial_node);
  int ret = CopyPartialShapeToSubGraph(partial_node, graph);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CopyPartialShapeToSubGraph failed, ret: " << ret;
    return ret;
  }

  ret = InferSubgraph(subgraph_index, graph);
  if (ret != RET_OK) {
    // not return ret here to infer the following part of graph
    MS_LOG(WARNING) << "InferSubgraph index: " << subgraph_index << " failed, ret: " << ret;
  }

  ret = RestoreSubGraphInput(partial_node, graph);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "RestoreSubGraphInput failed, ret: " << ret;
  }
  return ret;
}

void InferShapePass::InitInferTensor(MetaGraphT *graph) {
  tensors_.resize(graph->allTensors.size());
  for (size_t i = 0; i < graph->nodes.size(); i++) {
    auto &node = graph->nodes.at(i);
    auto node_input_indexes = node->inputIndex;
    //  init in_nodes index
    for (unsigned int node_input_indexe : node_input_indexes) {
      tensors_[node_input_indexe].next_nodes_.push_back(i);
    }
    auto node_output_indexes = node->outputIndex;
    for (unsigned int node_output_indexe : node_output_indexes) {
      tensors_[node_output_indexe].prev_nodes_.push_back(i);
    }
  }

  for (auto input_idx : graph->inputIndex) {
    auto input_tensor = graph->allTensors[input_idx].get();
    for (auto &dim : input_tensor->dims) {
      if (dim == 0) {
        MS_LOG(WARNING) << "One dimension of the input shape is 0, which would be set to -1 as a default value.";
        dim = DEFAULT_DIM_VALUE;
      }
    }
  }
}

int InferShapePass::InferSwitchNode(const std::unique_ptr<CNodeT> &switch_node, MetaGraphT *graph) {
  if (switch_node->inputIndex.size() < kSwitchInputMinSize) {
    MS_LOG(ERROR) << "switch node input size: " << switch_node->inputIndex.size() << " is less than three.";
    return RET_PARAM_INVALID;
  }

  static std::set<CNodeT *> partial_cnode_inferred{};
  std::deque<CNodeT *> to_process{};
  auto true_branch_output_index = switch_node->inputIndex.at(kSwitchTrueIndex);
  auto false_branch_output_index = switch_node->inputIndex.at(kSwitchFalseIndex);
  for (auto &node : graph->nodes) {
    if (node->primitive->value.type != PrimitiveType_PartialFusion) {
      continue;
    }
    if (IsContain(node->outputIndex, true_branch_output_index) &&
        partial_cnode_inferred.find(node.get()) == partial_cnode_inferred.end()) {
      to_process.push_back(node.get());
      partial_cnode_inferred.insert(node.get());
      break;
    }
  }
  for (auto &node : graph->nodes) {
    if (node->primitive->value.type != PrimitiveType_PartialFusion) {
      continue;
    }
    if (IsContain(node->outputIndex, false_branch_output_index) &&
        partial_cnode_inferred.find(node.get()) == partial_cnode_inferred.end()) {
      to_process.push_back(node.get());
      partial_cnode_inferred.insert(node.get());
      break;
    }
  }

  while (!to_process.empty()) {
    auto node = to_process.front();
    to_process.pop_front();
    int ret = InferPartialNode(node, graph);
    if (ret != RET_OK) {
      MS_LOG(WARNING) << "not support partial infer.";
      return ret;
    }
  }

  return RET_OK;
}

int InferShapePass::InferCallNode(const std::unique_ptr<CNodeT> &call_node, MetaGraphT *graph) {
  if (call_node->inputIndex.size() < kCallInputMinSize) {
    MS_LOG(ERROR) << "call node input size: " << call_node->inputIndex.size() << " is less than one.";
    return RET_PARAM_INVALID;
  }
  auto call_first_input_index = call_node->inputIndex.front();
  bool find_partial = false;
  bool find_switch = false;
  for (auto &node : graph->nodes) {
    if (IsContain(node->outputIndex, call_first_input_index) &&
        node->primitive->value.type == PrimitiveType_PartialFusion) {
      find_partial = true;
      int ret = InferPartialNode(node.get(), graph);
      if (ret != RET_OK) {
        MS_LOG(WARNING) << "not support partial infer.";
        return ret;
      }
      break;
    }
    if (IsContain(node->outputIndex, call_first_input_index) && node->primitive->value.type == PrimitiveType_Switch) {
      find_switch = true;
      int ret = InferSwitchNode(node, graph);
      if (ret != RET_OK) {
        MS_LOG(WARNING) << "not support partial infer.";
        return ret;
      }
      break;
    }
  }
  if (!find_partial && !find_switch) {
    MS_LOG(ERROR) << "not able to call partial or call switch.";
    return RET_ERROR;
  }
  return RET_OK;
}

int InferShapePass::InferSubgraph(const int &subgraph_index, MetaGraphT *graph) {
  std::vector<uint32_t> infer_node_indexes{};
  int ret = InitSearchTensor(subgraph_index, graph, &infer_node_indexes);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitSearchTensor failed.";
    return ret;
  }
  if (infer_node_indexes.empty()) {
    MS_LOG(DEBUG) << "no need to infer.";
    return RET_OK;
  }

  while (!infer_node_indexes.empty()) {
    auto infer_node_index = infer_node_indexes.front();
    auto &node = graph->nodes.at(infer_node_index);
    auto node_type = node->primitive->value.type;
    if (node_type == PrimitiveType_Call) {
      ret = InferCallNode(node, graph);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "infer call node failed.";
        return ret;
      }
    }

    infer_node_indexes.erase(infer_node_indexes.begin());
    auto input_tensors = ConvertTensorToLiteTensor(graph, node->inputIndex);
    auto output_tensors = ConvertTensorToLiteTensor(graph, node->outputIndex);
    if (output_tensors.empty() || output_tensors.size() != node->outputIndex.size() || input_tensors.empty() ||
        input_tensors.size() != node->inputIndex.size()) {
      MS_LOG(ERROR) << "convert lite tensor error";
      FreeTensors(&input_tensors, &output_tensors);
      return RET_INFER_ERR;
    }
    auto status = NodeInferShape(node, input_tensors, &output_tensors);
    MS_LOG(DEBUG) << "cur node:" << node->name;
    if (status == RET_OK || status == RET_INFER_INVALID) {
#ifdef Debug
      PrintTensorShape(input_tensors, output_tensors);
#endif
      // copy output shape to tensorT
      for (size_t i = 0; i < output_tensors.size(); i++) {
        auto output_dims = output_tensors[i]->shape();
        auto &output_tensorT = graph->allTensors.at(node->outputIndex[i]);
        output_tensorT->dims.swap(output_dims);
        SetDataType(graph, output_tensors, &tensors_, i, infer_node_index);
      }
    } else {
      MS_LOG(WARNING) << "InferShape failed, name: " << node->name
                      << ", type: " << schema::EnumNamePrimitiveType(node->primitive->value.type);
      FreeTensors(&input_tensors, &output_tensors);
      return RET_INFER_ERR;
    }
    FreeTensors(&input_tensors, &output_tensors);
    AddOutputNodes(graph, &infer_node_indexes, infer_node_index);
  }
  return RET_OK;
}

STATUS InferShapePass::Run(MetaGraphT *graph) {
  CHECK_NULL_RETURN(graph);
  InitInferTensor(graph);

  int ret = InferSubgraph(kMainGraphIndex, graph);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InferSubgraph index: " << kMainGraphIndex << " failed, ret: " << ret;
    return ret;
  }

  ResetIncorrectTensorShape(graph);
  return RET_OK;
}

int InferShapePass::InitSearchTensor(const int &subgraph_index, MetaGraphT *graph,
                                     std::vector<uint32_t> *infer_node_indexes) {
  if (static_cast<size_t>(subgraph_index) >= graph->subGraph.size()) {
    MS_LOG(ERROR) << "subgraph_index: " << subgraph_index
                  << " is larger than graph->subGraph.size(): " << graph->subGraph.size();
    return RET_ERROR;
  }
  auto &subgraph = graph->subGraph.at(subgraph_index);
  for (uint32_t i = 0; i < tensors_.size(); i++) {
    if (IsContain(subgraph->inputIndices, i) || !graph->allTensors.at(i)->data.empty()) {
      tensors_[i].is_inferred_ = true;
    }
  }
  for (size_t i = 0; i < subgraph->nodeIndices.size(); i++) {
    auto &node = graph->nodes.at(subgraph->nodeIndices.at(i));
    if (std::all_of(node->inputIndex.begin(), node->inputIndex.end(),
                    [&](uint32_t idx) { return tensors_[idx].is_inferred_; })) {
      infer_node_indexes->push_back(subgraph->nodeIndices.at(i));
    }
  }
  return RET_OK;
}

void InferShapePass::AddOutputNodes(MetaGraphT *graph, std::vector<uint32_t> *infer_node_indexes,
                                    uint32_t infer_node_index) {
  auto &node = graph->nodes.at(infer_node_index);
  for (size_t i = 0; i < node->outputIndex.size(); i++) {
    auto next_nodes_indexes = tensors_[node->outputIndex[i]].next_nodes_;
    for (size_t j = 0; j < next_nodes_indexes.size(); j++) {
      auto &next_node = graph->nodes.at(next_nodes_indexes[j]);
      if (std::any_of(next_node->outputIndex.begin(), next_node->outputIndex.end(),
                      [&](uint32_t idx) { return !tensors_[idx].is_inferred_; })) {
        AddNextInferShapeNode(graph, infer_node_indexes, next_nodes_indexes, j);
      }
    }
  }
}

void InferShapePass::AddNextInferShapeNode(MetaGraphT *graph, std::vector<uint32_t> *infer_node_indexes,
                                           std::vector<uint32_t> next_nodes_indexes, size_t index) {
  auto &next_node = graph->nodes.at(next_nodes_indexes[index]);
  if (find(infer_node_indexes->begin(), infer_node_indexes->end(), next_nodes_indexes[index]) ==
      infer_node_indexes->end()) {
    if (std::all_of(next_node->inputIndex.begin(), next_node->inputIndex.end(),
                    [&](uint32_t i) { return tensors_[i].is_inferred_; })) {
      infer_node_indexes->push_back(next_nodes_indexes[index]);
    }
  }
}

void InferShapePass::ResetIncorrectTensorShape(MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  for (auto &node : graph->nodes) {
    auto out_tensors_index = node->outputIndex;
    for (auto index : out_tensors_index) {
      auto &tensor = graph->allTensors.at(index);
      auto shape = tensor->dims;
      if (shape == std::vector{-1}) {
        tensor->dims = {};
      }
    }
  }
}
}  // namespace lite
}  // namespace mindspore
