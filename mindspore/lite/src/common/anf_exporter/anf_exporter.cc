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

#include "src/common/anf_exporter/anf_exporter.h"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "abstract/abstract_value.h"
#include "base/core_ops.h"
#include "mindspore/core/ir/primitive.h"
// #include "src/common/anf_exporter/anf_populater/anf_node_populater_registry.h"
#include "src/ir/primitive_t_value.h"
#include "src/ir/tensor.h"
#include "src/param_value_lite.h"
#include "src/common/utils.h"

namespace mindspore::lite {
std::set<std::string> RemoveNodeInAnfExporter{"tuple_getitem", "make_tuple"};

void AnfExporter::RemoveIfMakeTuple(const CNodePtr &cnode) {
  bool hasMakeTuple = false;
  std::vector<AnfNodePtr> inputs;
  inputs.clear();

  inputs.emplace_back(cnode->input(0));
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    AnfNodePtr inputNode = cnode->input(i);
    if (!inputNode->isa<CNode>()) {
      inputs.emplace_back(cnode->input(i));
      continue;
    }
    auto makeTupleNode = utils::cast<CNodePtr>(inputNode);
    if (IsPrimitiveCNode(makeTupleNode, prim::kPrimMakeTuple)) {
      hasMakeTuple = true;
      for (size_t j = 1; j < makeTupleNode->inputs().size(); ++j) {
        inputs.emplace_back(makeTupleNode->input(j));
      }
    } else {
      inputs.emplace_back(cnode->input(i));
    }
  }
  if (hasMakeTuple) {
    cnode->set_inputs(inputs);
  }
}

bool AnfExporter::RemoveIfTupleGetItem(const CNodePtr &cnode) {
  bool hasTupleGetItem = false;
  std::vector<AnfNodePtr> inputs;
  inputs.clear();
  inputs.emplace_back(cnode->input(0));
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    AnfNodePtr inputNode = cnode->input(i);
    if (!inputNode->isa<CNode>()) {
      inputs.emplace_back(cnode->input(i));
      continue;
    }
    auto tupleGetItemNode = utils::cast<CNodePtr>(inputNode);
    if (IsPrimitiveCNode(tupleGetItemNode, prim::kPrimTupleGetItem)) {
      hasTupleGetItem = true;
      inputs.emplace_back(tupleGetItemNode->input(1));
      AnfNodePtr indexNode = tupleGetItemNode->input(2);
      if (!utils::isa<ValueNode>(indexNode)) {
        MS_LOG(ERROR) << "TupleGetItem's input 2 is not valuenode";
        return false;
      }
      ValueNodePtr valueNode = utils::cast<ValueNodePtr>(indexNode);
      mapRemoveGetItem_[tupleGetItemNode->input(1)->fullname_with_scope()] = GetValue<int>(valueNode->value());
    } else {
      inputs.emplace_back(cnode->input(i));
    }
  }
  if (hasTupleGetItem) {
    cnode->set_inputs(inputs);
  }
  return true;
}

bool AnfExporter::AddOutPutIfReturn(const std::unique_ptr<schema::MetaGraphT> &metaGraphT, const CNodePtr &cnode) {
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    auto inputNode = cnode->input(i);
    if (!inputNode->isa<CNode>()) {
      MS_LOG(ERROR) << "Node of Return's input is not CNode";
      return false;
    }
    auto inputCNode = utils::cast<CNodePtr>(inputNode);
    auto inputPrimitive = GetValueNode<PrimitivePtr>(inputCNode->input(0));
    std::string inputName = inputNode->fullname_with_scope();
    auto graphOutput = nodeIdMap[inputName];
    metaGraphT->outputIndex.emplace_back(graphOutput);
  }
  return true;
}

schema::MetaGraphT *AnfExporter::Export(const FuncGraphPtr &funcGraph) {
  auto cnodes = funcGraph->GetOrderedCnodes();
  auto metaGraphT = std::make_unique<schema::MetaGraphT>();
  for (const auto &cnode : cnodes) {
    auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (primitive != nullptr) {
      if (RemoveNodeInAnfExporter.count(primitive->name()) != 0) {
        continue;
      }
    } else {
      auto primitiveT_value = GetValueNode<std::shared_ptr<PrimitiveTValue>>(cnode->input(0));
      auto primT = primitiveT_value->GetPrimitiveT();
      if (primT->value.type == schema::PrimitiveType_TupleGetItem ||
          primT->value.type == schema::PrimitiveType_MakeTuple) {
        continue;
      }
    }
    mapRemoveGetItem_.clear();
    RemoveIfMakeTuple(cnode);
    RemoveIfTupleGetItem(cnode);

    if (primitive != nullptr) {
      if (primitive->name() == prim::kPrimReturn->name()) {
        AddOutPutIfReturn(metaGraphT, cnode);
        continue;
      }
    } else {
      auto primitiveT_value = GetValueNode<std::shared_ptr<PrimitiveTValue>>(cnode->input(0));
      auto primT = primitiveT_value->GetPrimitiveT();
      if (primT->value.type == schema::PrimitiveType_Return) {
        AddOutPutIfReturn(metaGraphT, cnode);
        continue;
      }
    }

    auto node = std::make_unique<schema::CNodeT>();
    node->name = cnode->fullname_with_scope();
    node->nodeType = schema::NodeType_CNode;
    // populate primitive
    // if (primitive != nullptr) {
    //   primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
    //   MS_ASSERT(primitive != nullptr);
    //   std::string opType = primitive->name();
    //   auto nodeParser = AnfNodePopulaterRegistry::GetInstance()->GetNodePopulater(opType);
    //   if (nodeParser == nullptr) {
    //     MS_LOG(ERROR) << "Find op parser failed, opType: " << opType;
    //     return nullptr;
    //   }
    //   std::vector<schema::TensorT *> outputs;
    //   if (utils::isa<abstract::AbstractSequeue>(cnode->abstract())) {
    //     auto abstract_cnode = utils::cast<abstract::AbstractSequeuePtr>(cnode->abstract());
    //     outputs.resize(abstract_cnode->size());
    //   }
    //
    //   nodeParser->Parse(cnode, node.get(), &outputs);
    //   SetOpInputNode(cnode, metaGraphT.get(), node.get());
    //   SetOpOutputNode(cnode, outputs, metaGraphT.get(), node.get());
    //   metaGraphT->nodes.emplace_back(std::move(node));
    //   continue;
    // }
    auto primitiveT_value = GetValueNode<std::shared_ptr<PrimitiveTValue>>(cnode->input(0));
    if (primitiveT_value == nullptr) {
      MS_LOG(ERROR) << "PrimitiveT_value is nullptr";
      return nullptr;
    }

    auto *lite_primitive = primitiveT_value->GetPrimitiveT();
    if (lite_primitive == nullptr) {
      MS_LOG(ERROR) << "Primitive in primitiveT_value is nullptr";
      return nullptr;
    }

    node->primitive = std::unique_ptr<schema::PrimitiveT>(primitiveT_value->GetPrimitiveT());
    std::vector<schema::TensorT *> outputs;
    SetOpInputNode(cnode, metaGraphT.get(), node.get());
    SetOpOutputNode(cnode, outputs, metaGraphT.get(), node.get());

    // add quant param
    node->quantType = primitiveT_value->GetQuantType();
    if (node->quantType == schema::QuantType_PostTraining || node->quantType == schema::QuantType_AwareTraining) {
      MS_LOG(INFO) << "node: " << node->name << " add QuantParam";
      // activation
      auto input_quant_params = primitiveT_value->GetInputQuantParams();
      auto node_type = primitiveT_value->GetPrimitiveT()->value.type;
      for (int i = 0; i < input_quant_params.size(); i++) {
        if (i >= node->inputIndex.size()) {
          MS_LOG(ERROR) << "node: " << node->name << " input has " << input_quant_params.size()
                        << " quant_params; but only " << node->inputIndex.size() << " input";
          break;
        }
        auto activate_index = node->inputIndex[i];
        auto tensor_input = metaGraphT->allTensors[activate_index].get();
        if (tensor_input->quantParams.empty()) {
          for (auto input_quant_param : input_quant_params[i]) {
            std::unique_ptr<schema::QuantParamT> input_quant_param_ptr =
              std::make_unique<schema::QuantParamT>(input_quant_param);
            MS_LOG(DEBUG) << "[input]node: " << node->name << " scale: " << input_quant_param_ptr->scale
                          << " zp: " << input_quant_param_ptr->zeroPoint;
            tensor_input->quantParams.emplace_back(std::move(input_quant_param_ptr));
          }
        }
      }

      // output
      auto output_index = node->outputIndex[0];
      auto tensor_output = metaGraphT->allTensors[output_index].get();
      auto output_quant_params = primitiveT_value->GetOutputQuantParams();
      if (output_quant_params.empty()) {
        MS_LOG(WARNING) << "node: " << node->name << " output quant params is empty";
      } else {
        for (auto output_quant_param : output_quant_params[0]) {
          if (tensor_output->quantParams.empty()) {
            std::unique_ptr<schema::QuantParamT> output_quant_param_ptr =
              std::make_unique<schema::QuantParamT>(output_quant_param);
            MS_LOG(DEBUG) << "[input]node: " << node->name << " scale: " << output_quant_param_ptr->scale
                          << " zp: " << output_quant_param_ptr->zeroPoint;
            tensor_output->quantParams.emplace_back(std::move(output_quant_param_ptr));
          }
        }
      }
      if (node->quantType != schema::QuantType_AwareTraining &&
          !(node_type == schema::PrimitiveType_QuantDTypeCast &&
            primitiveT_value->GetPrimitiveT()->value.AsQuantDTypeCast()->dstT == kNumberTypeFloat32)) {
        tensor_output->dataType = kNumberTypeInt8;
      }
      //      // TensorType
      //      valuePtr = primitive->GetAttr(kInputTensorDataType);
      //      if (valuePtr != nullptr) {
      //        MS_LOG(INFO) << "node: " << node->name << " input tensor data
      //        type: " << GetValue<int>(valuePtr); for (auto input :
      //        node->inputIndex) {
      //          auto tensor = subGraph->allTensors[input].get();
      //          tensor->dataType = kNumberTypeUInt8;
      //        }
      //      }
    }

    metaGraphT->nodes.emplace_back(std::move(node));
  }
  // set graph input tensors
  for (auto node : graphInputNodes) {
    for (auto input : node->inputIndex) {
      auto tensor = metaGraphT->allTensors[input].get();
      if (tensor->data.empty()) {
        tensor->nodeType = schema::NodeType_ValueNode;
        tensor->format = schema::Format_NHWC;
        if (!IsContain(metaGraphT->inputIndex, input)) {
          metaGraphT->inputIndex.emplace_back(input);
        }
      }
    }
  }
  return metaGraphT.release();
}

void AnfExporter::SetOpInputNode(const CNodePtr &cnode, schema::MetaGraphT *meta_graph, schema::CNodeT *fbNode) {
  MS_ASSERT(nullptr != meta_graph);
  MS_ASSERT(nullptr != fbNode);
  if (cnode->inputs().size() <= 1) {
    return;
  }
  std::string cNodeName = cnode->fullname_with_scope();
  bool isGraphInput = true;
  for (int i = 1; i < static_cast<int>(cnode->inputs().size()); i++) {
    auto inputNode = cnode->input(i);
    if (inputNode->isa<CNode>()) {
      isGraphInput = false;
      std::string inputName = inputNode->fullname_with_scope();
      if (!mapRemoveGetItem_.empty()) {
        for (auto name : mapRemoveGetItem_) {
          if (name.first == inputName) {
            inputName = inputName + "_o:" + std::to_string(name.second);
          }
        }
      }
      if (nodeIdMap.find(inputName) != nodeIdMap.end()) {
        fbNode->inputIndex.emplace_back(nodeIdMap[inputName]);
      }
    } else if (inputNode->isa<Parameter>()) {
      auto paramNode = inputNode->cast<ParameterPtr>();
      if (paramNode->name().empty()) {
        paramNode->set_name(cNodeName + "_i:" + std::to_string(i - 1));
      }
      if (nodeIdMap.find(paramNode->name()) != nodeIdMap.end()) {
        fbNode->inputIndex.emplace_back(nodeIdMap[paramNode->name()]);
        continue;
      }
      auto paramTensor = std::make_unique<schema::TensorT>();
      auto abstractBase = paramNode->abstract();
      if (abstractBase == nullptr) {
        MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << paramNode->name();
        MS_ASSERT(false);
        return;
      }
      if (!utils::isa<abstract::AbstractTensorPtr>(abstractBase)) {
        MS_LOG(ERROR) << "Abstract of parameter should be anstract tensor, " << paramNode->name();
        MS_ASSERT(false);
        return;
      }
      auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(abstractBase);
      auto typePtr = abstractTensor->element()->GetTypeTrack();
      MS_ASSERT(typePtr != nullptr);
      paramTensor->dataType = typePtr->type_id();
      if (!utils::isa<abstract::ShapePtr>(abstractTensor->BuildShape())) {
        MS_LOG(ERROR) << "Shape of Abstract of parameter should be ShapePtr, " << paramNode->name();
        MS_ASSERT(false);
        return;
      }
      paramTensor->dims = utils::cast<abstract::ShapePtr>(abstractTensor->BuildShape())->shape();
      auto paramValue = std::dynamic_pointer_cast<ParamValueLite>(paramNode->default_param());
      if (paramValue != nullptr) {
        paramTensor->nodeType = schema::NodeType_ValueNode;
        paramTensor->data.resize(paramValue->tensor_size());
        memcpy(paramTensor->data.data(), paramValue->tensor_addr(), paramValue->tensor_size());
      }
      nodeIdMap[paramNode->fullname_with_scope()] = meta_graph->allTensors.size();
      fbNode->inputIndex.emplace_back(meta_graph->allTensors.size());
      meta_graph->allTensors.emplace_back(std::move(paramTensor));
    } else if (inputNode->isa<ValueNode>()) {
      auto valueNode = inputNode->cast<ValueNodePtr>();
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
        nodeIdMap[valueNode->fullname_with_scope()] = meta_graph->allTensors.size();
        fbNode->inputIndex.emplace_back(meta_graph->allTensors.size());
        meta_graph->allTensors.emplace_back(std::move(paramTensor));
      } else if (value->isa<mindspore::Int32Imm>()) {
        auto valueAbstract = valueNode->abstract();
        auto abstractScalar = utils::cast<abstract::AbstractScalarPtr>(valueAbstract);
        auto typePtr = abstractScalar->GetTypeTrack();
        paramTensor->dataType = typePtr->type_id();
        paramTensor->dims = {1};
        paramTensor->nodeType = schema::NodeType_ValueNode;
        auto data = value->cast<mindspore::Int32ImmPtr>();
        paramTensor->data.emplace_back(data->value());
        nodeIdMap[valueNode->fullname_with_scope()] = meta_graph->allTensors.size();
        fbNode->inputIndex.emplace_back(meta_graph->allTensors.size());
        meta_graph->allTensors.emplace_back(std::move(paramTensor));
      } else if (value->isa<mindspore::ValueSequeue>()) {
        MS_LOG(INFO) << "Value type is ValueSequence.";
        break;
      } else {
        MS_LOG(ERROR) << "Not support value type , need add support.";
      }
    }
  }
  if (isGraphInput) {
    graphInputNodes.emplace_back(fbNode);
  }
}

void AnfExporter::SetOpOutputNode(const CNodePtr &cnode, const std::vector<schema::TensorT *> &outputTensors,
                                  schema::MetaGraphT *graph, schema::CNodeT *fbnode) {
  MS_ASSERT(nullptr != graph);
  MS_ASSERT(nullptr != fbnode);
  std::string cnodeName = fbnode->name;
  if (!outputTensors.empty()) {
    int i = 0;
    for (auto outputTensor : outputTensors) {
      std::string name = cnodeName + "_o:" + std::to_string(i);
      auto msTensor = new schema::TensorT();
      msTensor->nodeType = schema::NodeType_Parameter;
      nodeIdMap[name] = graph->allTensors.size();
      fbnode->outputIndex.emplace_back(graph->allTensors.size());
      graph->allTensors.emplace_back(msTensor);
      i++;
    }
    return;
  }

  if (utils::isa<abstract::AbstractTuple>(cnode->abstract())) {
    auto tuple = std::reinterpret_pointer_cast<abstract::AbstractTuple>(cnode->abstract());
    for (int i = 0; i < tuple->size(); i++) {
      auto msTensor = new schema::TensorT();
      msTensor->nodeType = schema::NodeType_Parameter;
      fbnode->outputIndex.emplace_back(graph->allTensors.size());
      if (tuple->size() == 1) {
        nodeIdMap[cnodeName] = graph->allTensors.size();
      } else {
        std::string name = cnodeName + "_o:" + std::to_string(i);
        nodeIdMap[name] = graph->allTensors.size();
      }
      graph->allTensors.emplace_back(msTensor);
    }
  } else {
    auto msTensor = new schema::TensorT();
    msTensor->nodeType = schema::NodeType_Parameter;
    fbnode->outputIndex.emplace_back(graph->allTensors.size());
    nodeIdMap[cnodeName] = graph->allTensors.size();
    graph->allTensors.emplace_back(msTensor);
  }
}

schema::MetaGraphT *Export(const FuncGraphPtr &funcGraph) {
  AnfExporter anfExporter;
  return anfExporter.Export(funcGraph);
}
}  // namespace mindspore::lite
