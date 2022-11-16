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

#define USE_DEPRECATED_API
#include "tools/converter/legacy_optimizer/graph/infershape_pass.h"
#include <vector>
#include <deque>
#include "src/common/common.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"
#include "src/tensor.h"
#include "src/tensorlist.h"
#include "src/common/prim_util.h"
#include "src/common/ops/populate/populate_register.h"
#include "src/litert/infer_manager.h"
#include "src/common/primitive_t_utils.h"
#include "tools/common/node_util.h"
#include "src/common/string_utils.h"
#include "src/common/log_util.h"
#include "nnacl/op_base.h"

using mindspore::converter::kFmkTypeTf;
namespace {
constexpr int DEFAULT_DIM_VALUE = -1;
constexpr size_t kInitialSize = 1024;
constexpr int kMainGraphIndex = 0;
constexpr int kCallInputMinSize = 1;
constexpr int kSwitchInputMinSize = 3;
constexpr int kTypeIndex = 0;
constexpr int kElementShapeIndex = 1;
constexpr int kFirstElementShapeIndex = 2;
constexpr int kTensorListDataSize = 3;
}  // namespace
namespace mindspore {
namespace lite {
namespace {
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

namespace {
constexpr int kBytesPerInt = 4;
}  // namespace

void ConvertTensorList(const MetaGraphT *graph, uint32_t index, bool *convert_succ,
                       std::vector<Tensor *> *lite_tensors) {
  if (graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr";
    return;
  }
  std::unique_ptr<Tensor> lite_tensor = nullptr;
  auto &tensorT = graph->allTensors.at(index);
  std::vector<int32_t> tensor_shape{};
  TypeId type = kTypeUnknown;
  std::vector<int> element_shape;
  if (tensorT->data.size() >= kBytesPerInt) {
    int *data = reinterpret_cast<int *>(tensorT->data.data());
    type = TypeId(data[kTypeIndex]);
    auto basic_data_size = tensorT->data.size() / sizeof(int);
    if (basic_data_size < static_cast<size_t>(kTensorListDataSize)) {
      MS_LOG(ERROR) << "tensorlist data length illegal, which should be at least 3, now is " << basic_data_size;
      *convert_succ = false;
      return;
    }
    if (data[kElementShapeIndex] < 0 || INT_ADD_OVERFLOW(data[kElementShapeIndex], kTensorListDataSize)) {
      MS_LOG(ERROR) << "int add overflow.";
      *convert_succ = false;
      return;
    }
    if (static_cast<size_t>((data[kElementShapeIndex] + kTensorListDataSize)) > basic_data_size) {
      MS_LOG(ERROR) << "tensorlist data length illegal. current tensorlist data length should be at least "
                    << (data[kElementShapeIndex] + kTensorListDataSize) << ", but now is " << basic_data_size;
      *convert_succ = false;
      return;
    }
    auto element_num = data[data[kElementShapeIndex] + kFirstElementShapeIndex];
    if (element_num > 0 && INT_ADD_OVERFLOW(element_num, 1)) {
      MS_LOG(ERROR) << "int add overflow.";
      *convert_succ = false;
      return;
    }
    auto shape_once = data[kElementShapeIndex] + 1;
    auto shape_group_num = element_num < 0 ? 1 : element_num + 1;
    if (INT_MUL_OVERFLOW(shape_once, shape_group_num)) {
      MS_LOG(ERROR) << "int mul overflow.";
      *convert_succ = false;
      return;
    }
    tensor_shape = {element_num};
    auto shape_info_size = shape_once * shape_group_num;
    if (INT_ADD_OVERFLOW(shape_info_size, kFirstElementShapeIndex)) {
      MS_LOG(ERROR) << "int add overflow.";
      *convert_succ = false;
      return;
    }
    int real_data_size = shape_info_size + kFirstElementShapeIndex;
    if (real_data_size <= 0 || static_cast<uint32_t>(real_data_size) != basic_data_size) {
      MS_LOG(ERROR) << "current tensorlist data length should be " << real_data_size << ", but now is "
                    << basic_data_size;
      *convert_succ = false;
      return;
    }
    for (int j = 0; j < data[kElementShapeIndex]; ++j) {
      element_shape.push_back(data[j + kFirstElementShapeIndex]);
    }
  }
  lite_tensor = std::make_unique<TensorList>(tensor_shape, element_shape);
  if (lite_tensor == nullptr) {
    MS_LOG(ERROR) << "lite tensorlist is nullptr";
    *convert_succ = false;
    return;
  }

  auto lite_tensor_list = reinterpret_cast<TensorList *>(lite_tensor.get());
  std::vector<Tensor *> tensors{};
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

namespace {
std::unique_ptr<Tensor> CreateRuntimeTensor(const std::unique_ptr<TensorT> &src_tensor) {
  if (src_tensor == nullptr) {
    MS_LOG(ERROR) << "src tensor is nullptr";
    return nullptr;
  }
  std::unique_ptr<Tensor> runtime_tensor = nullptr;
  auto tensor_shape = src_tensor->dims;
  runtime_tensor = std::make_unique<Tensor>(TypeId(src_tensor->dataType), tensor_shape,
                                            static_cast<mindspore::Format>(src_tensor->format),
                                            TensorCategory(src_tensor->nodeType, src_tensor->dims.size(),
                                                           TypeId(src_tensor->dataType), src_tensor->data.size()));
  if (runtime_tensor == nullptr) {
    MS_LOG(ERROR) << "Create runtime tensor failed";
    return nullptr;
  }
  return runtime_tensor;
}
}  // namespace

void ConvertString(const MetaGraphT *graph, uint32_t index, bool *convert_succ, std::vector<Tensor *> *lite_tensors) {
  auto &tensorT = graph->allTensors.at(index);
  auto runtime_tensor = CreateRuntimeTensor(tensorT);
  if (runtime_tensor == nullptr) {
    *convert_succ = false;
    return;
  }
  // when tensorT as param input
  if (tensorT->data.empty()) {
    lite_tensors->emplace_back(runtime_tensor.release());
    return;
  }
  auto string_buffer = ParseStringBuffer(tensorT->data.data());
  auto ret = WriteStringsToTensor(runtime_tensor.get(), string_buffer);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "WriteStringsToTensor failed";
    *convert_succ = false;
    return;
  }
  lite_tensors->emplace_back(runtime_tensor.release());
}

void ConvertOtherTensor(const MetaGraphT *graph, uint32_t index, bool *convert_succ,
                        std::vector<Tensor *> *lite_tensors) {
  CHECK_NULL_RETURN_VOID(graph);
  auto &tensorT = graph->allTensors.at(index);
  auto runtime_tensor = CreateRuntimeTensor(tensorT);
  if (runtime_tensor == nullptr) {
    *convert_succ = false;
    return;
  }
  // when tensorT as param input
  if (tensorT->data.empty()) {
    lite_tensors->emplace_back(runtime_tensor.release());
    return;
  }
  runtime_tensor->set_data(tensorT->data.data());
  lite_tensors->emplace_back(runtime_tensor.release());
}

std::vector<Tensor *> ConvertTensorToLiteTensor(const MetaGraphT *graph, const std::vector<uint32_t> &tensor_indexs) {
  MS_ASSERT(graph != nullptr);
  std::vector<Tensor *> lite_tensors;
  bool convert_succ = true;
  for (unsigned int tensor_index : tensor_indexs) {
    auto &tensorT = graph->allTensors.at(tensor_index);
    switch (tensorT->dataType) {
      case kObjectTypeTensorType:
        ConvertTensorList(graph, tensor_index, &convert_succ, &lite_tensors);
        break;
      case kObjectTypeString:
        MS_CHECK_TRUE_MSG(tensorT->dims.size() <= 1, {}, "String type tensor dims should be less than or equal to 1.");
        ConvertString(graph, tensor_index, &convert_succ, &lite_tensors);
        break;
      default:
        ConvertOtherTensor(graph, tensor_index, &convert_succ, &lite_tensors);
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

  auto ret = KernelInferShape(inputs, *outputs, prim, {}, static_cast<int>(SCHEMA_CUR));
  if (ret == lite::RET_NOT_SUPPORT) {
    auto parameter_gen = lite::PopulateRegistry::GetInstance()->GetParameterCreator(
      static_cast<int>(prim->value_type()), static_cast<int>(SCHEMA_CUR));
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

int SetDataType(MetaGraphT *graph, const std::vector<Tensor *> &output_tensors,
                const std::unique_ptr<mindspore::schema::CNodeT> &node, std::vector<InferTensor> *tensors, size_t i) {
  auto &output_tensor = graph->allTensors.at(node->outputIndex[i]);
  output_tensor->format = static_cast<schema::Format>(output_tensors[i]->format());
  output_tensor->dataType = output_tensors[i]->data_type();
  if (output_tensors[i]->data_type() == kObjectTypeTensorType) {
    auto tensor_list = reinterpret_cast<TensorList *>(output_tensors[i]);
    MSLITE_CHECK_PTR(tensor_list);
    int tensor_shape_dims = 0;
    if (!tensor_list->tensors().empty()) {
      tensor_shape_dims = static_cast<int>(tensor_list->tensors().front()->shape().size());
    }
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW((tensor_shape_dims + kTensorListDataSize), static_cast<int>(sizeof(int))),
                       RET_ERROR, "int mul overflow");
    if (tensor_list->tensors_data_type() == kTypeUnknown) {
      if (!tensor_list->tensors().empty()) {
        tensor_list->set_tensors_data_type(tensor_list->tensors().front()->data_type());
      }
    }
    std::vector<int> basic_data;
    basic_data.push_back(tensor_list->tensors_data_type());
    if (tensor_list->element_shape().empty() && !tensor_list->tensors().empty()) {
      tensor_list->set_element_shape(tensor_list->tensors().front()->shape());
    }
    basic_data.push_back(tensor_list->element_shape().size());
    for (size_t j = 0; j < tensor_list->element_shape().size(); ++j) {
      basic_data.push_back(tensor_list->element_shape().at(j));
    }
    basic_data.push_back(tensor_list->tensors().size());
    for (size_t index = 0; index < tensor_list->tensors().size(); ++index) {
      auto tensor_shape = tensor_list->GetTensor(static_cast<int>(index))->shape();
      basic_data.push_back(tensor_shape.size());
      for (size_t j = 0; j < tensor_shape.size(); ++j) {
        basic_data.push_back(tensor_shape[j]);
      }
    }
    output_tensor->data.resize(basic_data.size() * sizeof(int));
    if (memcpy_s(output_tensor->data.data(), output_tensor->data.size(), basic_data.data(),
                 basic_data.size() * sizeof(int)) != EOK) {
      MS_LOG(ERROR) << "memcpy data failed.";
      return RET_ERROR;
    }
  } else if (output_tensors[i]->data_type() == kTypeUnknown) {
    tensors->at(node->outputIndex[i]).is_inferred_ = false;
    return RET_OK;
  }
  tensors->at(node->outputIndex[i]).is_inferred_ = true;
  return RET_OK;
}

int CopyOutputInfoToTensorT(MetaGraphT *graph, const std::vector<Tensor *> &output_tensors,
                            const std::unique_ptr<mindspore::schema::CNodeT> &node, std::vector<InferTensor> *tensors) {
  for (uint32_t i = 0; i < output_tensors.size(); i++) {
    auto output_dims = output_tensors[i]->shape();
    auto &output_tensorT = graph->allTensors.at(node->outputIndex[i]);
    MSLITE_CHECK_PTR(output_tensorT);
    output_tensorT->dims.swap(output_dims);
    if (SetDataType(graph, output_tensors, node, tensors, i) != RET_OK) {
      MS_LOG(ERROR) << "SetDataType failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int64_t PartialGraphIndex(const CNodeT *partial_node) {
  MSLITE_CHECK_PTR(partial_node);
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
    MSLITE_CHECK_PTR(subgraph_input);
    auto &partial_input = graph->allTensors.at(partial_node->inputIndex[i]);
    MSLITE_CHECK_PTR(partial_input);
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

void InferShapePass::RestoreSubGraphInput(const CNodeT *partial_node, MetaGraphT *graph) {
  auto subgraph_index = PartialGraphIndex(partial_node);
  auto &subgraph = graph->subGraph.at(subgraph_index);
  for (size_t i = 0; i < subgraph->inputIndices.size(); ++i) {
    auto &subgraph_input = graph->allTensors.at(subgraph->inputIndices[i]);
    if (subgraph_input->dataType != kObjectTypeTensorType) {
      subgraph_input->data = {};
    }
  }
}

int InferShapePass::SetNonTailCallOutputShape(const std::unique_ptr<CNodeT> &call_node, const CNodeT *partial_node,
                                              MetaGraphT *graph) {
  auto subgraph_index = PartialGraphIndex(partial_node);
  auto &subgraph = graph->subGraph.at(subgraph_index);
  size_t call_node_output_size = call_node->outputIndex.size();
  size_t subgraph_output_size = subgraph->outputIndices.size();
  if (subgraph_output_size != call_node_output_size) {
    MS_LOG(ERROR) << "call node output size: " << call_node_output_size
                  << " is same as corresponding subgraph output size: " << subgraph_output_size;
    return RET_ERROR;
  }
  for (size_t i = 0; i < subgraph_output_size; ++i) {
    auto &subgraph_output_tensor = graph->allTensors.at(subgraph->outputIndices[i]);
    auto &call_output_tensor = graph->allTensors.at(call_node->outputIndex[i]);
    call_output_tensor->format = subgraph_output_tensor->format;
    call_output_tensor->dims = subgraph_output_tensor->dims;
    call_output_tensor->dataType = subgraph_output_tensor->dataType;
  }
  return RET_OK;
}

int InferShapePass::InferPartialNode(const bool &is_tail_call, const std::unique_ptr<CNodeT> &call_node,
                                     const CNodeT *partial_node, MetaGraphT *graph) {
  int64_t subgraph_index = PartialGraphIndex(partial_node);
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

  RestoreSubGraphInput(partial_node, graph);

  if (!is_tail_call) {
    ret = SetNonTailCallOutputShape(call_node, partial_node, graph);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "SetNonTailCallOutputShape failed.";
      return ret;
    }
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
    CHECK_NULL_RETURN_VOID(input_tensor);
    for (auto &dim : input_tensor->dims) {
      if (dim == 0) {
        MS_LOG(WARNING) << "One dimension of the input shape is 0, which would be set to -1 as a default value.";
        dim = DEFAULT_DIM_VALUE;
      }
    }
  }
}

int InferShapePass::InferSwitchOrSwitchLayerNode(const bool &is_tail_call, const std::unique_ptr<CNodeT> &call_node,
                                                 const std::unique_ptr<CNodeT> &aim_node, MetaGraphT *graph) {
  if (aim_node->inputIndex.size() < kSwitchInputMinSize) {
    MS_LOG(ERROR) << "switch or switch_layer node input size: " << aim_node->inputIndex.size() << " is less than 3.";
    return RET_PARAM_INVALID;
  }

  size_t aim_node_input_size = aim_node->inputIndex.size();
  std::vector<uint32_t> all_partial_index{};
  for (size_t i = 1; i < aim_node_input_size; ++i) {
    all_partial_index.push_back(aim_node->inputIndex.at(i));
  }

  std::vector<CNodeT *> all_partial_nodes{};
  for (auto &partial_index : all_partial_index) {
    for (auto &node : graph->nodes) {
      MSLITE_CHECK_PTR(node);
      if (node->primitive->value.type != PrimitiveType_PartialFusion) {
        continue;
      }
      if (IsContain(node->outputIndex, partial_index)) {
        all_partial_nodes.push_back(node.get());
        break;
      }
    }
  }

  std::deque<CNodeT *> to_process{};
  for (auto &partial_node : all_partial_nodes) {
    if (partial_cnode_inferred_.find(partial_node) == partial_cnode_inferred_.end()) {
      to_process.push_back(partial_node);
      (void)partial_cnode_inferred_.insert(partial_node);
    }
  }

  while (!to_process.empty()) {
    auto node = to_process.front();
    to_process.pop_front();
    int ret = InferPartialNode(is_tail_call, call_node, node, graph);
    if (ret != RET_OK) {
      MS_LOG(WARNING) << "not support partial infer.";
      return ret;
    }
  }

  return RET_OK;
}

int InferShapePass::InferCallNode(const std::unique_ptr<CNodeT> &call_node, MetaGraphT *graph) {
  MSLITE_CHECK_PTR(call_node);
  if (call_node->inputIndex.size() < kCallInputMinSize) {
    MS_LOG(ERROR) << "call node input size: " << call_node->inputIndex.size() << " is less than one.";
    return RET_PARAM_INVALID;
  }
  auto call_first_input_index = call_node->inputIndex.front();
  bool is_tail_call = call_node->primitive->value.AsCall()->is_tail_call;
  for (auto &node : graph->nodes) {
    if (!IsContain(node->outputIndex, call_first_input_index)) {
      continue;
    }
    switch (node->primitive->value.type) {
      case PrimitiveType_PartialFusion:
        return InferPartialNode(is_tail_call, call_node, node.get(), graph);
      case PrimitiveType_Switch:
      case PrimitiveType_SwitchLayer:
        return InferSwitchOrSwitchLayerNode(is_tail_call, call_node, node, graph);
      default:
        MS_LOG(ERROR) << "not able to call partial or call switch.";
        return RET_ERROR;
    }
  }
  return RET_OK;
}

int InferShapePass::InferSubgraph(const int64_t &subgraph_index, MetaGraphT *graph) {
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
    MSLITE_CHECK_PTR(node);
    infer_node_indexes.erase(infer_node_indexes.begin());
    auto node_type = node->primitive->value.type;
    if (node_type == PrimitiveType_Call) {
      ret = InferCallNode(node, graph);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "infer call node failed.";
        return ret;
      }
    }

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
      ret = CopyOutputInfoToTensorT(graph, output_tensors, node, &tensors_);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "SetDataType failed: " << ret;
        FreeTensors(&input_tensors, &output_tensors);
        return RET_INFER_ERR;
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

int InferShapePass::InitSearchTensor(const int64_t &subgraph_index, MetaGraphT *graph,
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
