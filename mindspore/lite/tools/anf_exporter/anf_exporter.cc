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
#include <algorithm>

#include "src/ops/quant_dtype_cast.h"
#include "abstract/abstract_value.h"
#include "mindspore/core/ir/primitive.h"
#include "src/tensor.h"
#include "src/param_value_lite.h"
#include "src/common/utils.h"

namespace mindspore::lite {
void AnfExporter::RemoveIfMakeTuple(const CNodePtr &cnode) {
  bool has_make_tuple = false;
  std::vector<AnfNodePtr> inputs;
  inputs.clear();

  inputs.emplace_back(cnode->input(0));
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    AnfNodePtr input_node = cnode->input(i);
    if (!input_node->isa<CNode>()) {
      inputs.emplace_back(cnode->input(i));
      continue;
    }
    auto make_tuple_node = utils::cast<CNodePtr>(input_node);
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

void AnfExporter::RemoveIfDepend(const CNodePtr &cnode) {
  bool hasDepend = false;
  std::vector<AnfNodePtr> inputs;
  inputs.clear();

  inputs.emplace_back(cnode->input(0));
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    AnfNodePtr inputNode = cnode->input(i);
    if (!inputNode->isa<CNode>()) {
      inputs.emplace_back(cnode->input(i));
      continue;
    }
    auto dependNode = utils::cast<CNodePtr>(inputNode);
    if (IsPrimitiveCNode(dependNode, schema::PrimitiveType_Depend) ||
        IsPrimitiveCNode(dependNode, schema::PrimitiveType_ControlDepend)) {
      hasDepend = true;
      for (size_t j = 1; j < dependNode->inputs().size(); ++j) {
        AnfNodePtr dependInputNode = dependNode->input(j);
        if (dependInputNode->isa<CNode>()) {
          inputs.emplace_back(dependInputNode);
        }
      }
    } else {
      inputs.emplace_back(cnode->input(i));
    }
  }
  if (hasDepend) {
    cnode->set_inputs(inputs);
  }
}

int AnfExporter::ConvertQuantParam(const std::unique_ptr<schema::MetaGraphT> &meta_graph,
                                   const std::shared_ptr<PrimitiveC> &primitive,
                                   const std::unique_ptr<schema::CNodeT> &dst_node) {
  MS_ASSERT(meta_graph != nullptr);
  MS_ASSERT(primitive != nullptr);
  MS_ASSERT(dst_node != nullptr);
  // add quant param
  dst_node->quantType = primitive->GetQuantType();
  MS_LOG(DEBUG) << "node: " << dst_node->name << " add QuantParam";
  // activation
  auto input_quant_params = primitive->GetInputQuantParams();
  auto node_type = (schema::PrimitiveType)primitive->Type();
  if (!input_quant_params.empty()) {
    for (size_t i = 0; i < input_quant_params.size(); i++) {
      if (i >= dst_node->inputIndex.size()) {
        MS_LOG(INFO) << "node: " << dst_node->name << " input has " << input_quant_params.size()
                     << " quant_params; but only " << dst_node->inputIndex.size() << " input";
        break;
      }
      auto activate_index = dst_node->inputIndex[i];
      auto tensor_input = meta_graph->allTensors[activate_index].get();
      if (tensor_input->quantParams.empty()) {
        for (auto input_quant_param : input_quant_params[i]) {
          std::unique_ptr<schema::QuantParamT> input_quant_param_ptr =
            std::make_unique<schema::QuantParamT>(input_quant_param);
          MS_LOG(DEBUG) << "[input][" << i << "]node: " << dst_node->name << " scale: " << input_quant_param_ptr->scale
                        << " zp: " << input_quant_param_ptr->zeroPoint;
          tensor_input->quantParams.emplace_back(std::move(input_quant_param_ptr));
        }
      }
    }
  } else {
    MS_LOG(DEBUG) << "node: " << dst_node->name << " input quant params is empty";
  }
  // output
  auto output_index = dst_node->outputIndex[0];
  auto tensor_output = meta_graph->allTensors[output_index].get();
  auto output_quant_params = primitive->GetOutputQuantParams();
  if (output_quant_params.empty()) {
    if (node_type != schema::PrimitiveType_QuantDTypeCast) {
      MS_LOG(DEBUG) << "node: " << dst_node->name << " output quant params is empty";
    }
  } else {
    for (auto output_quant_param : output_quant_params[0]) {
      if (tensor_output->quantParams.empty() && dst_node->quantType != schema::QuantType_WeightQuant) {
        std::unique_ptr<schema::QuantParamT> output_quant_param_ptr =
          std::make_unique<schema::QuantParamT>(output_quant_param);
        MS_LOG(DEBUG) << "[output]node: " << dst_node->name << " scale: " << output_quant_param_ptr->scale
                      << " zp: " << output_quant_param_ptr->zeroPoint;
        tensor_output->quantParams.emplace_back(std::move(output_quant_param_ptr));
      }
    }
  }
  if (dst_node->quantType == schema::QuantType_PostTraining) {
    if (node_type != schema::PrimitiveType_QuantDTypeCast) {
      tensor_output->dataType = kNumberTypeInt8;
    } else {
      MS_ASSERT(utils::isa<std::shared_ptr<QuantDTypeCast>>(primitive));
      auto primc = utils::cast<std::shared_ptr<QuantDTypeCast>>(primitive);
      MS_ASSERT(primc != nullptr);
      if (primc->GetDstT() != kNumberTypeFloat32) {
        tensor_output->dataType = kNumberTypeInt8;
      }
    }
  }
  return RET_OK;
}

void AnfExporter::SetGraphInputIndex(const std::unique_ptr<schema::MetaGraphT> &meta_graphT) {
  for (auto node : graph_input_nodes_) {
    for (auto input : node->inputIndex) {
      auto tensor = meta_graphT->allTensors[input].get();
      if (tensor->nodeType != schema::NodeType_CNode && tensor->data.empty()) {
        tensor->nodeType = schema::NodeType_ValueNode;
        tensor->format = schema::Format_NHWC;
        if (!IsContain(meta_graphT->inputIndex, input)) {
          meta_graphT->inputIndex.emplace_back(input);
        }
      }
    }
  }
}

int AnfExporter::SetGraphoutputIndex(const CNodePtr &cnode, const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                     schema::CNodeT *return_node) {
  MS_ASSERT(nullptr != meta_graphT);
  MS_ASSERT(nullptr != return_node);
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto input_node = cnode->input(i);
    if (input_node == nullptr) {
      MS_LOG(ERROR) << "output node is nullptr";
      return RET_NULL_PTR;
    } else if (input_node->isa<CNode>()) {
      auto ret = ConvertInputCNode(input_node, return_node);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "obtain outputs failed";
        return ret;
      }
    } else {
      MS_LOG(ERROR) << "the node " << input_node->fullname_with_scope().c_str() << "is not output node";
      return RET_ERROR;
    }
  }
  for (unsigned int &i : return_node->inputIndex) {
    meta_graphT->outputIndex.push_back(i);
  }
  return RET_OK;
}

schema::MetaGraphT *AnfExporter::Export(const FuncGraphPtr &func_graph, bool keep_graph, bool copy_primitive) {
  auto cnodes = func_graph->GetOrderedCnodes();
  auto meta_graphT = std::make_unique<schema::MetaGraphT>();
  int ret = RET_OK;
  for (const auto &cnode : cnodes) {
    auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "primitive_c is nullptr";
      ret = RET_MEMORY_FAILED;
      break;
    }
    if ((primitive_c->Type() == schema::PrimitiveType_TupleGetItem) ||
#ifdef SUPPORT_TRAIN
        (primitive_c->Type() == schema::PrimitiveType_Depend) ||
        (primitive_c->Type() == schema::PrimitiveType_ControlDepend) ||
#endif
        (primitive_c->Type() == schema::PrimitiveType_MakeTuple)) {
      continue;
    }
    RemoveIfMakeTuple(cnode);
#ifdef SUPPORT_TRAIN
    RemoveIfDepend(cnode);
#endif
    auto primT = primitive_c->GetPrimitiveT();
    auto node = std::make_unique<schema::CNodeT>();
    if (node == nullptr) {
      MS_LOG(ERROR) << "object failed to be constructed";
      ret = RET_MEMORY_FAILED;
      break;
    }
    if (primT->value.type == schema::PrimitiveType_Return) {
      node->name = "return_node";
      ret = SetGraphoutputIndex(cnode, meta_graphT, node.get());
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "SetOpOutputN failed";
        break;
      }
      continue;
    }
    node->nodeType = schema::NodeType_CNode;
    node->name = cnode->fullname_with_scope();
    if (copy_primitive) {
      auto primitive = new (std::nothrow) schema::PrimitiveT();
      if (primitive != nullptr) {
        *primitive = *primT;
        node->primitive = std::unique_ptr<schema::PrimitiveT>(primitive);
      }
    } else {
      node->primitive = std::unique_ptr<schema::PrimitiveT>(primT);
    }
    ret = SetOpInputNode(cnode, meta_graphT, node.get());
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "SetOpInputNode failed";
      break;
    }
    SetOpOutputNode(cnode, meta_graphT, node.get());
    ret = ConvertQuantParam(meta_graphT, primitive_c, node);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ConvertQuantParam failed";
      break;
    }
    if (!keep_graph) {
      primitive_c->ClearPrimitiveT();
    }
    meta_graphT->nodes.emplace_back(std::move(node));
  }
  if (ret != RET_OK) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
    return nullptr;
  }
  // set graph input tensors
  SetGraphInputIndex(meta_graphT);
  return meta_graphT.release();
}

int AnfExporter::ConvertInputCNode(const std::shared_ptr<AnfNode> &input_anode, schema::CNodeT *output_cnode) {
  std::string input_name = input_anode->fullname_with_scope();
  auto input_cnode = utils::cast<CNodePtr>(input_anode);

  if (!IsPrimitiveCNode(input_cnode, schema::PrimitiveType_TupleGetItem)) {
#ifndef SUPPORT_TRAIN
    if (node_id_map_.find(input_name) != node_id_map_.end()) {
      output_cnode->inputIndex.emplace_back(node_id_map_[input_name]);
    }
#else
    bool found = false;
    if (node_id_map_.find(input_name) != node_id_map_.end()) {
      output_cnode->inputIndex.emplace_back(node_id_map_[input_name]);
      found = true;
    }

    if (found == false) {
      auto input_index_key = input_name + "_o:" + std::to_string(0);
      if (node_id_map_.find(input_index_key) != node_id_map_.end()) {
        output_cnode->inputIndex.emplace_back(node_id_map_[input_index_key]);
      }
    }
#endif
  } else {
    auto inputs = input_cnode->inputs();

    if (inputs.size() != 3) {
      MS_LOG(ERROR) << "TupleGetItem should have 3 inputs, got " << inputs.size();
      return RET_ERROR;
    }
    auto get_item_input_cnode = inputs.at(1);
    auto index_vnode = inputs.at(2);
    if (!utils::isa<ValueNode>(index_vnode)) {
      MS_LOG(ERROR) << "TupleGetItem's input 2 is not valuenode";
      return RET_ERROR;
    }
    auto value_node = utils::cast<ValueNodePtr>(index_vnode);
    if (value_node == nullptr) {
      MS_LOG(ERROR) << "cast to ValueNode failed";
      return RET_ERROR;
    }
    auto input_index_key = get_item_input_cnode->fullname_with_scope() + "_o:" +
                           std::to_string(value_node->value()->type()->number_type() == kNumberTypeInt64
                                            ? GetValue<int64_t>(value_node->value())
                                            : GetValue<int>(value_node->value()));
    auto iter = node_id_map_.find(input_index_key);
    if (iter == node_id_map_.end()) {
#ifdef SUPPORT_TRAIN
      input_index_key = get_item_input_cnode->fullname_with_scope() + "_o:" + std::to_string(0);  // try name with 0
      iter = node_id_map_.find(input_index_key);
      if (iter == node_id_map_.end()) {
        MS_LOG(ERROR) << "Can not find get_item output tensor" << input_index_key;
        return RET_ERROR;
      }
#else
      MS_LOG(ERROR) << "Can not find get_item output tensor" << input_index_key;
      return RET_ERROR;
#endif
    }
    output_cnode->inputIndex.emplace_back(iter->second);
  }
  return RET_OK;
}

int AnfExporter::ConvertInputParameter(const std::shared_ptr<AnfNode> &input_anode,
                                       const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                       schema::CNodeT *output_cnode) {
  auto paramNode = input_anode->cast<ParameterPtr>();
  std::string input_name = paramNode->fullname_with_scope();
  if (node_id_map_.find(input_name) != node_id_map_.end()) {
    output_cnode->inputIndex.emplace_back(node_id_map_[paramNode->name()]);
    return RET_OK;
  }
  auto paramTensor = std::make_unique<schema::TensorT>();
  paramTensor->format = schema::Format_NHWC;
  auto abstractBase = paramNode->abstract();
  if (abstractBase == nullptr) {
    MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << paramNode->name();
    return RET_PARAM_INVALID;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstractBase)) {
    MS_LOG(ERROR) << "Abstract of parameter should be anstract tensor, " << paramNode->name();
    return RET_INPUT_TENSOR_ERROR;
  }
  auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(abstractBase);
  auto typePtr = abstractTensor->element()->GetTypeTrack();
  MS_ASSERT(typePtr != nullptr);
  paramTensor->dataType = typePtr->type_id();
  if (!utils::isa<abstract::ShapePtr>(abstractTensor->BuildShape())) {
    MS_LOG(ERROR) << "Shape of Abstract of parameter should be ShapePtr, " << paramNode->name();
    return RET_PARAM_INVALID;
  }
  auto shape_vector = utils::cast<abstract::ShapePtr>(abstractTensor->BuildShape())->shape();
  std::vector<int32_t> dims;
  (void)std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(dims),
                       [](const int64_t &value) { return static_cast<int32_t>(value); });
  paramTensor->dims = dims;
  auto paramValue = std::dynamic_pointer_cast<ParamValueLite>(paramNode->default_param());
  if (paramValue != nullptr) {
    paramTensor->data.resize(paramValue->tensor_size());
    paramTensor->format = schema::Format(paramValue->format());
    memcpy(paramTensor->data.data(), paramValue->tensor_addr(), paramValue->tensor_size());
  }

  node_id_map_[input_name] = meta_graphT->allTensors.size();
  output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
  meta_graphT->allTensors.emplace_back(std::move(paramTensor));
  return RET_OK;
}

int AnfExporter::ConvertInputValueNode(const std::shared_ptr<AnfNode> &input_anode,
                                       const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                       schema::CNodeT *output_cnode) {
  auto valueNode = input_anode->cast<ValueNodePtr>();
  auto paramTensor = std::make_unique<schema::TensorT>();
  auto value = valueNode->value();
  if (value->isa<tensor::Tensor>()) {
    auto valueAbstract = valueNode->abstract();
    auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(valueAbstract);
    auto typePtr = abstractTensor->element()->GetTypeTrack();
    paramTensor->dataType = typePtr->type_id();
    auto shape_vector = utils::cast<abstract::ShapePtr>(abstractTensor->BuildShape())->shape();
    std::vector<int32_t> dims;
    (void)std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(dims),
                         [](const int64_t &value) { return static_cast<int32_t>(value); });
    paramTensor->dims = dims;
#ifdef SUPPORT_TRAIN
    if (paramTensor->dims.size() == 0) paramTensor->dims = {1};
#endif
    paramTensor->nodeType = schema::NodeType::NodeType_ValueNode;
    auto data = value->cast<tensor::TensorPtr>();
    paramTensor->data.resize(data->Size());
    memcpy(paramTensor->data.data(), data->data_c(), data->Size());
    node_id_map_[valueNode->fullname_with_scope()] = meta_graphT->allTensors.size();
    output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
    meta_graphT->allTensors.emplace_back(std::move(paramTensor));
  } else if (value->isa<mindspore::Int32Imm>()) {
    auto valueAbstract = valueNode->abstract();
    auto abstractScalar = utils::cast<abstract::AbstractScalarPtr>(valueAbstract);
    auto typePtr = abstractScalar->GetTypeTrack();
    paramTensor->dataType = typePtr->type_id();
    paramTensor->dims = {1};
    paramTensor->nodeType = schema::NodeType::NodeType_ValueNode;
    int real_data = CastToInt(value, false).front();
    paramTensor->data.resize(sizeof(int32_t));
    memcpy(paramTensor->data.data(), &real_data, sizeof(int32_t));
    node_id_map_[valueNode->fullname_with_scope()] = meta_graphT->allTensors.size();
    output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
    meta_graphT->allTensors.emplace_back(std::move(paramTensor));
  } else if (value->isa<mindspore::BoolImm>()) {
    auto valueAbstract = valueNode->abstract();
    auto abstractScalar = utils::cast<abstract::AbstractScalarPtr>(valueAbstract);
    auto typePtr = abstractScalar->GetTypeTrack();
    paramTensor->dataType = typePtr->type_id();
    paramTensor->dims = {1};
    paramTensor->nodeType = schema::NodeType_ValueNode;
    auto data = value->cast<mindspore::BoolImmPtr>();
    paramTensor->data.emplace_back(data->value());
    node_id_map_[valueNode->fullname_with_scope()] = meta_graphT->allTensors.size();
    output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
    meta_graphT->allTensors.emplace_back(std::move(paramTensor));
  } else if (value->isa<mindspore::Int>()) {
    paramTensor->dataType = kNumberTypeInt32;
    paramTensor->dims = {1};
    paramTensor->nodeType = schema::NodeType_ValueNode;
    paramTensor->data.emplace_back(kNumberTypeInt32);
    node_id_map_[valueNode->fullname_with_scope()] = meta_graphT->allTensors.size();
    output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
    meta_graphT->allTensors.emplace_back(std::move(paramTensor));
  } else if (value->isa<mindspore::ValueSequeue>()) {
    MS_LOG(DEBUG) << "Value type is ValueSequence.";
    return RET_OK;
  } else if (value->isa<mindspore::BoolImm>()) {
    auto valueAbstract = valueNode->abstract();
    auto abstractScalar = utils::cast<abstract::AbstractScalarPtr>(valueAbstract);
    auto typePtr = abstractScalar->GetTypeTrack();
    paramTensor->dataType = typePtr->type_id();
    paramTensor->dims = {1};
    paramTensor->nodeType = schema::NodeType_ValueNode;
    auto data = value->cast<mindspore::BoolImmPtr>();
    paramTensor->data.emplace_back(data->value());
    node_id_map_[valueNode->fullname_with_scope()] = meta_graphT->allTensors.size();
    output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
    meta_graphT->allTensors.emplace_back(std::move(paramTensor));
  } else if (value->isa<Number>()) {
    MS_LOG(INFO) << "Value is a number.";
    return RET_OK;
  } else if (value->isa<mindspore::ParamValueLite>()) {
    auto valueLite = std::dynamic_pointer_cast<ParamValueLite>(value);
    paramTensor->data.resize(valueLite->tensor_size());
    paramTensor->format = schema::Format(valueLite->format());
    paramTensor->dataType = valueLite->tensor_type();
    paramTensor->dims = valueLite->tensor_shape();
    memcpy(paramTensor->data.data(), valueLite->tensor_addr(), valueLite->tensor_size());
    node_id_map_[valueNode->fullname_with_scope()] = meta_graphT->allTensors.size();
    output_cnode->inputIndex.emplace_back(meta_graphT->allTensors.size());
    meta_graphT->allTensors.emplace_back(std::move(paramTensor));
  } else {
    MS_LOG(ERROR) << "Not support value type , need add support.";
    return RET_ERROR;
  }
  return RET_OK;
}

int AnfExporter::SetOpInputNode(const CNodePtr &cnode, const std::unique_ptr<schema::MetaGraphT> &meta_graphT,
                                schema::CNodeT *fb_node) {
  MS_ASSERT(nullptr != meta_graphT);
  MS_ASSERT(nullptr != fb_node);
  if (cnode->inputs().size() <= 1) {
    return RET_OK;
  }
  bool is_graph_input = false;
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto input_node = cnode->input(i);
    if (input_node->isa<CNode>()) {
      auto ret = ConvertInputCNode(input_node, fb_node);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "ConvertInputCNode failed";
        return ret;
      }
    } else if (input_node->isa<Parameter>()) {
      auto ret = ConvertInputParameter(input_node, meta_graphT, fb_node);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "ConvertInputParameter failed";
        return ret;
      }
      if (!input_node->cast<ParameterPtr>()->has_default()) {
        is_graph_input = true;
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
  MS_ASSERT(nullptr != meta_graphT);
  MS_ASSERT(nullptr != fb_node);
  std::string cnode_name = fb_node->name;

  if (utils::isa<abstract::AbstractTuple>(cnode->abstract())) {
    auto tuple = std::reinterpret_pointer_cast<abstract::AbstractTuple>(cnode->abstract());
    for (size_t i = 0; i < tuple->size(); i++) {
      auto msTensor = new (std::nothrow) schema::TensorT();
      msTensor->nodeType = schema::NodeType_CNode;
      fb_node->outputIndex.emplace_back(meta_graphT->allTensors.size());
#ifdef SUPPORT_TRAIN
      std::string name = cnode_name + "_o:" + std::to_string(i);
      node_id_map_[name] = meta_graphT->allTensors.size();
      meta_graphT->allTensors.emplace_back(msTensor);
      if (IsPrimitiveCNode(cnode, schema::PrimitiveType_Conv2D) ||
          IsPrimitiveCNode(cnode, schema::PrimitiveType_DepthwiseConv2D) ||
          IsPrimitiveCNode(cnode, schema::PrimitiveType_Adam))
        break;
#else
      if (tuple->size() == 1) {
        node_id_map_[cnode_name] = meta_graphT->allTensors.size();
      } else {
        std::string name = cnode_name + "_o:" + std::to_string(i);
        node_id_map_[name] = meta_graphT->allTensors.size();
      }
      meta_graphT->allTensors.emplace_back(msTensor);
      if (IsPrimitiveCNode(cnode, schema::PrimitiveType_Conv2D) ||
          IsPrimitiveCNode(cnode, schema::PrimitiveType_DepthwiseConv2D) ||
          IsPrimitiveCNode(cnode, schema::PrimitiveType_FusedBatchNorm)) {
        break;
      }
#endif
    }
  } else {
    auto ms_tensor = new (std::nothrow) schema::TensorT();
    ms_tensor->nodeType = schema::NodeType_CNode;
    ms_tensor->dataType = TypeId::kNumberTypeFloat32;
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

  auto prim = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
  if (prim == nullptr) {
    return false;
  }
  return (schema::PrimitiveType)(prim->Type()) == type;
}

schema::MetaGraphT *Export(const FuncGraphPtr &func_graph, bool keep_graph, bool copy_primitive) {
  AnfExporter anf_exporter;
  return anf_exporter.Export(func_graph, keep_graph, copy_primitive);
}
}  // namespace mindspore::lite
