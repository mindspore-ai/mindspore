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

#include "tools/anf_exporter/anf_exporter.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "abstract/abstract_value.h"
#include "mindspore/core/ir/primitive.h"
#include "src/ir/tensor.h"
#include "src/param_value_lite.h"
#include "src/common/utils.h"

namespace mindspore::lite {
void AnfExporter::RemoveIfMakeTuple(const CNodePtr &cnode) {
  bool has_make_tuple = false;
  std::vector<AnfNodePtr> inputs;
  inputs.clear();

  inputs.emplace_back(cnode->input(0));
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    AnfNodePtr inputNode = cnode->input(i);
    if (!inputNode->isa<CNode>()) {
      inputs.emplace_back(cnode->input(i));
      continue;
    }
    auto make_tuple_node = utils::cast<CNodePtr>(inputNode);
    if (IsPrimitiveCNode(make_tuple_node, schema::PrimitiveType_MakeTuple)) {
      has_make_tuple = true;
      for (size_t j = 1; j < make_tuple_node->inputs().size(); ++j) {
        inputs.emplace_back(make_tuple_node->input(j));
      }
    } else {
      inputs.emplace_back(cnode->input(i));
    }
  }
  if (has_make_tuple) {
    cnode->set_inputs(inputs);
  }
}

bool AnfExporter::RemoveIfTupleGetItem(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  bool has_tuple_get_item = false;
  std::vector<AnfNodePtr> inputs;
  inputs.clear();
  inputs.emplace_back(cnode->input(0));
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    AnfNodePtr inputNode = cnode->input(i);
    if (!inputNode->isa<CNode>()) {
      inputs.emplace_back(cnode->input(i));
      continue;
    }
    auto tuple_get_item_node = utils::cast<CNodePtr>(inputNode);
    if (IsPrimitiveCNode(tuple_get_item_node, schema::PrimitiveType_TupleGetItem)) {
      has_tuple_get_item = true;
      inputs.emplace_back(tuple_get_item_node->input(1));
      AnfNodePtr indexNode = tuple_get_item_node->input(2);
      if (!utils::isa<ValueNode>(indexNode)) {
        MS_LOG(ERROR) << "TupleGetItem's input 2 is not valuenode";
        return false;
      }
      ValueNodePtr value_node = utils::cast<ValueNodePtr>(indexNode);
      map_remove_get_item_[tuple_get_item_node->input(1)->fullname_with_scope()] = GetValue<int>(value_node->value());
    } else {
      inputs.emplace_back(cnode->input(i));
    }
  }
  if (has_tuple_get_item) {
    cnode->set_inputs(inputs);
  }
  return true;
}

bool AnfExporter::AddOutPutIfReturn(const std::unique_ptr<schema::MetaGraphT> &meta_graphT, const CNodePtr &cnode) {
  MS_ASSERT(meta_graphT != nullptr);
  MS_ASSERT(cnode != nullptr);
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    auto inputNode = cnode->input(i);
    if (!inputNode->isa<CNode>()) {
      MS_LOG(ERROR) << "Node of Return's input is not CNode";
      return false;
    }
    auto inputCNode = utils::cast<CNodePtr>(inputNode);
    std::string inputName = inputNode->fullname_with_scope();
    auto graphOutput = node_id_map_[inputName];
    meta_graphT->outputIndex.emplace_back(graphOutput);
  }
  return true;
}

int AnfExporter::ConvertQuantParam(const std::unique_ptr<schema::MetaGraphT> &meta_graph,
                                   const std::shared_ptr<PrimitiveTValue> primitive,
                                   const std::unique_ptr<schema::CNodeT> &dst_node) {
  MS_ASSERT(meta_graph != nullptr);
  MS_ASSERT(primitive != nullptr);
  MS_ASSERT(dst_node != nullptr);
  // add quant param
  dst_node->quantType = primitive->GetQuantType();
  if (dst_node->quantType == schema::QuantType_PostTraining || dst_node->quantType == schema::QuantType_AwareTraining) {
    MS_LOG(DEBUG) << "node: " << dst_node->name << " add QuantParam";
    // activation
    auto input_quant_params = primitive->GetInputQuantParams();
    auto node_type = primitive->GetPrimitiveT()->value.type;
    for (size_t i = 0; i < input_quant_params.size(); i++) {
      if (i >= dst_node->inputIndex.size()) {
        MS_LOG(ERROR) << "node: " << dst_node->name << " input has " << input_quant_params.size()
                      << " quant_params; but only " << dst_node->inputIndex.size() << " input";
        break;
      }
      auto activate_index = dst_node->inputIndex[i];
      auto tensor_input = meta_graph->allTensors[activate_index].get();
      if (tensor_input->quantParams.empty()) {
        for (auto input_quant_param : input_quant_params[i]) {
          std::unique_ptr<schema::QuantParamT> input_quant_param_ptr =
            std::make_unique<schema::QuantParamT>(input_quant_param);
          MS_LOG(DEBUG) << "[input]node: " << dst_node->name << " scale: " << input_quant_param_ptr->scale
                        << " zp: " << input_quant_param_ptr->zeroPoint;
          tensor_input->quantParams.emplace_back(std::move(input_quant_param_ptr));
        }
      }
    }

    // output
    auto output_index = dst_node->outputIndex[0];
    auto tensor_output = meta_graph->allTensors[output_index].get();
    auto output_quant_params = primitive->GetOutputQuantParams();
    if (output_quant_params.empty()) {
      MS_LOG(WARNING) << "node: " << dst_node->name << " output quant params is empty";
    } else {
      for (auto output_quant_param : output_quant_params[0]) {
        if (tensor_output->quantParams.empty()) {
          std::unique_ptr<schema::QuantParamT> output_quant_param_ptr =
            std::make_unique<schema::QuantParamT>(output_quant_param);
          MS_LOG(DEBUG) << "[input]node: " << dst_node->name << " scale: " << output_quant_param_ptr->scale
                        << " zp: " << output_quant_param_ptr->zeroPoint;
          tensor_output->quantParams.emplace_back(std::move(output_quant_param_ptr));
        }
      }
    }
    if (dst_node->quantType != schema::QuantType_AwareTraining &&
        !(node_type == schema::PrimitiveType_QuantDTypeCast &&
          primitive->GetPrimitiveT()->value.AsQuantDTypeCast()->dstT == kNumberTypeFloat32)) {
      tensor_output->dataType = kNumberTypeInt8;
    }
  }
  return RET_OK;
}

void AnfExporter::SetGraphInputIndex(const std::unique_ptr<schema::MetaGraphT> &meta_graphT) {
  for (auto node : graph_input_nodes_) {
    for (auto input : node->inputIndex) {
      auto tensor = meta_graphT->allTensors[input].get();
      if (tensor->data.empty()) {
        tensor->nodeType = schema::NodeType_ValueNode;
        tensor->format = schema::Format_NHWC;
        if (!IsContain(meta_graphT->inputIndex, input)) {
          meta_graphT->inputIndex.emplace_back(input);
        }
      }
    }
  }
}

schema::MetaGraphT *AnfExporter::Export(const FuncGraphPtr &func_graph) {
  auto cnodes = func_graph->GetOrderedCnodes();
  auto meta_graphT = std::make_unique<schema::MetaGraphT>();
  for (const auto &cnode : cnodes) {
    auto primitiveT_value = GetValueNode<std::shared_ptr<PrimitiveTValue>>(cnode->input(0));
    if (primitiveT_value == nullptr) {
      MS_LOG(ERROR) << "PrimitiveT_value is nullptr";
      return nullptr;
    }
    auto primT = primitiveT_value->GetPrimitiveT();
    if (primT == nullptr) {
      MS_LOG(ERROR) << "PrimitiveT is nullptr";
      return nullptr;
    }
    if (primT->value.type == schema::PrimitiveType_TupleGetItem ||
        primT->value.type == schema::PrimitiveType_MakeTuple) {
      continue;
    }
    map_remove_get_item_.clear();
    RemoveIfMakeTuple(cnode);
    if (!RemoveIfTupleGetItem(cnode)) {
      MS_LOG(ERROR) << "RemoveIfTupleGetItem failed";
      return nullptr;
    }

    if (primT->value.type == schema::PrimitiveType_Return) {
      AddOutPutIfReturn(meta_graphT, cnode);
      continue;
    }

    auto node = std::make_unique<schema::CNodeT>();
    node->name = cnode->fullname_with_scope();
    node->nodeType = schema::NodeType_CNode;

    node->primitive = std::unique_ptr<schema::PrimitiveT>(primT);
    auto ret = SetOpInputNode(cnode, meta_graphT, node.get());
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "SetOpInputNode failed";
      return nullptr;
    }

    SetOpOutputNode(cnode, meta_graphT, node.get());

    ret = ConvertQuantParam(meta_graphT, primitiveT_value, node);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ConvertQuantParam failed";
      return nullptr;
    }

    meta_graphT->nodes.emplace_back(std::move(node));
  }
  // set graph input tensors
  SetGraphInputIndex(meta_graphT);
  return meta_graphT.release();
}

void AnfExporter::ConvertInputCNode(const std::shared_ptr<AnfNode> input_anode, schema::CNodeT *output_cnode) {
  std::string input_name = input_anode->fullname_with_scope();
  if (!map_remove_get_item_.empty()) {
    for (auto name : map_remove_get_item_) {
      if (name.first == input_name) {
        input_name = input_name + "_o:" + std::to_string(name.second);
      }
    }
  }
  if (node_id_map_.find(input_name) != node_id_map_.end()) {
    output_cnode->inputIndex.emplace_back(node_id_map_[input_name]);
  }
}

int AnfExporter::ConvertInputParameter(const std::shared_ptr<AnfNode> input_anode, size_t anode_index,
                                       const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                       schema::CNodeT *output_cnode) {
  std::string input_name = input_anode->fullname_with_scope();
  auto paramNode = input_anode->cast<ParameterPtr>();
  if (paramNode->name().empty()) {
    paramNode->set_name(input_name + "_i:" + std::to_string(anode_index - 1));
  }
  if (node_id_map_.find(paramNode->name()) != node_id_map_.end()) {
    output_cnode->inputIndex.emplace_back(node_id_map_[paramNode->name()]);
    return RET_OK;
  }
  auto paramTensor = std::make_unique<schema::TensorT>();
  auto abstractBase = paramNode->abstract();
  if (abstractBase == nullptr) {
    MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << paramNode->name();
    return RET_ERROR;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstractBase)) {
    MS_LOG(ERROR) << "Abstract of parameter should be anstract tensor, " << paramNode->name();
    return RET_ERROR;
  }
  auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(abstractBase);
  auto typePtr = abstractTensor->element()->GetTypeTrack();
  MS_ASSERT(typePtr != nullptr);
  paramTensor->dataType = typePtr->type_id();
  paramTensor->format = schema::Format(abstractTensor->format());
  if (!utils::isa<abstract::ShapePtr>(abstractTensor->BuildShape())) {
    MS_LOG(ERROR) << "Shape of Abstract of parameter should be ShapePtr, " << paramNode->name();
    return RET_ERROR;
  }
  paramTensor->dims = utils::cast<abstract::ShapePtr>(abstractTensor->BuildShape())->shape();
  auto paramValue = std::dynamic_pointer_cast<ParamValueLite>(paramNode->default_param());
  if (paramValue != nullptr) {
    paramTensor->nodeType = schema::NodeType_ValueNode;
    paramTensor->data.resize(paramValue->tensor_size());
    memcpy(paramTensor->data.data(), paramValue->tensor_addr(), paramValue->tensor_size());
  }
  node_id_map_[paramNode->fullname_with_scope()] = meta_graphT->allTensors.size();
  output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
  meta_graphT->allTensors.emplace_back(std::move(paramTensor));
  return RET_OK;
}

int AnfExporter::ConvertInputValueNode(std::shared_ptr<AnfNode> input_anode,
                                       const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                       schema::CNodeT *output_cnode) {
  auto valueNode = input_anode->cast<ValueNodePtr>();
  auto paramTensor = std::make_unique<schema::TensorT>();
  auto value = valueNode->value();
  if (value->isa<lite::tensor::Tensor>()) {
    auto valueAbstract = valueNode->abstract();
    auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(valueAbstract);
    auto typePtr = abstractTensor->element()->GetTypeTrack();
    paramTensor->dataType = typePtr->type_id();
    paramTensor->dims = utils::cast<abstract::ShapePtr>(abstractTensor->BuildShape())->shape();
    paramTensor->nodeType = schema::NodeType_ValueNode;
    auto data = value->cast<lite::tensor::TensorPtr>();
    paramTensor->data.resize(data->Size());
    memcpy(paramTensor->data.data(), data->Data(), data->Size());
    node_id_map_[valueNode->fullname_with_scope()] = meta_graphT->allTensors.size();
    output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
    meta_graphT->allTensors.emplace_back(std::move(paramTensor));
  } else if (value->isa<mindspore::Int32Imm>()) {
    auto valueAbstract = valueNode->abstract();
    auto abstractScalar = utils::cast<abstract::AbstractScalarPtr>(valueAbstract);
    auto typePtr = abstractScalar->GetTypeTrack();
    paramTensor->dataType = typePtr->type_id();
    paramTensor->dims = {1};
    paramTensor->nodeType = schema::NodeType_ValueNode;
    auto data = value->cast<mindspore::Int32ImmPtr>();
    paramTensor->data.emplace_back(data->value());
    node_id_map_[valueNode->fullname_with_scope()] = meta_graphT->allTensors.size();
    output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
    meta_graphT->allTensors.emplace_back(std::move(paramTensor));
  } else if (value->isa<mindspore::ValueSequeue>()) {
    MS_LOG(DEBUG) << "Value type is ValueSequence.";
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "Not support value type , need add support.";
    return RET_ERROR;
  }
  return RET_OK;
}

int AnfExporter::SetOpInputNode(const CNodePtr &cnode, const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                schema::CNodeT *fb_node) {
  MS_ASSERT(nullptr != meta_graph);
  MS_ASSERT(nullptr != fb_node);
  if (cnode->inputs().size() <= 1) {
    return RET_OK;
  }
  bool is_graph_input = true;
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto input_node = cnode->input(i);
    if (input_node->isa<CNode>()) {
      is_graph_input = false;
      ConvertInputCNode(input_node, fb_node);
    } else if (input_node->isa<Parameter>()) {
      auto ret = ConvertInputParameter(input_node, i, meta_graphT, fb_node);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "ConvertInputParameter failed";
        return RET_ERROR;
      }
    } else if (input_node->isa<ValueNode>()) {
      auto ret = ConvertInputValueNode(input_node, meta_graphT, fb_node);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "ConvertInputValueNode failed";
        return RET_ERROR;
      }
    }
  }
  fb_node->name = cnode->fullname_with_scope();
  if (is_graph_input) {
    graph_input_nodes_.emplace_back(fb_node);
  }
  return RET_OK;
}

void AnfExporter::SetOpOutputNode(const CNodePtr &cnode, const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                  schema::CNodeT *fb_node) {
  MS_ASSERT(nullptr != graph);
  MS_ASSERT(nullptr != fb_node);
  std::string cnode_name = fb_node->name;

  if (utils::isa<abstract::AbstractTuple>(cnode->abstract())) {
    auto tuple = std::reinterpret_pointer_cast<abstract::AbstractTuple>(cnode->abstract());
    for (size_t i = 0; i < tuple->size(); i++) {
      auto msTensor = new schema::TensorT();
      msTensor->nodeType = schema::NodeType_Parameter;
      fb_node->outputIndex.emplace_back(meta_graphT->allTensors.size());
      if (tuple->size() == 1) {
        node_id_map_[cnode_name] = meta_graphT->allTensors.size();
      } else {
        std::string name = cnode_name + "_o:" + std::to_string(i);
        node_id_map_[name] = meta_graphT->allTensors.size();
      }
      meta_graphT->allTensors.emplace_back(msTensor);
      if (IsPrimitiveCNode(cnode, schema::PrimitiveType_Conv2D)
          || IsPrimitiveCNode(cnode, schema::PrimitiveType_DepthwiseConv2D)
          || IsPrimitiveCNode(cnode, schema::PrimitiveType_FusedBatchNorm)) {
        break;
      }
    }
  } else {
    auto ms_tensor = new schema::TensorT();
    ms_tensor->nodeType = schema::NodeType_Parameter;
    fb_node->outputIndex.emplace_back(meta_graphT->allTensors.size());
    node_id_map_[cnode_name] = meta_graphT->allTensors.size();
    meta_graphT->allTensors.emplace_back(ms_tensor);
  }
}

bool AnfExporter::IsPrimitiveCNode(const AnfNodePtr &node, schema::PrimitiveType type) {
  MS_ASSERT(node != nullptr);
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return false;
  }

  const auto &prim = GetValueNode<std::shared_ptr<PrimitiveTValue>>(cnode->input(0));
  if (prim == nullptr) {
    return false;
  }
  auto *primitiveT = prim->GetPrimitiveT();
  if (primitiveT == nullptr) {
    return false;
  }
  return primitiveT->value.type == type;
}

schema::MetaGraphT *Export(const FuncGraphPtr &func_graph) {
  AnfExporter anf_exporter;
  return anf_exporter.Export(func_graph);
}
}  // namespace mindspore::lite
