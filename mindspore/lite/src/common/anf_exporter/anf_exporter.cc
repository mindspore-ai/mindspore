/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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
#include <utility>
#include <vector>
#include <string>
#include "abstract/abstract_value.h"
#include "src/common/anf_exporter/anf_populater/anf_node_populater_registry.h"
#include "src/param_value_lite.h"
#include "mindspore/core/ir/primitive.h"
#include "src/ir/primitive_t_value.h"
#include "base/core_ops.h"
#include "src/ir/tensor.h"

namespace mindspore::lite {
schema::MetaGraphT *AnfExporter::Export(const FuncGraphPtr &funcGraph) {
  auto cnodes = funcGraph->GetOrderedCnodes();
  auto metaGraphT = std::make_unique<schema::MetaGraphT>();
  for (const auto &cnode : cnodes) {
    auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (primitive != nullptr && primitive == prim::kPrimReturn) {
      // set graph outputs tensors
      auto inputNode = cnode->input(1);
      if (!inputNode->isa<CNode>()) {
        continue;
      }
      auto inputCNode = utils::cast<CNodePtr>(inputNode);
      auto inputPrimitive = GetValueNode<PrimitivePtr>(inputCNode->input(0));
      if (inputPrimitive == prim::kPrimMakeTuple) {
        continue;
      } else {
        std::string inputName = inputNode->fullname_with_scope();
        auto graphOutput = nodeIdMap[inputName];
        metaGraphT->outputIndex.emplace_back(graphOutput);
      }
      continue;
    }
    if (primitive != nullptr && primitive == prim::kPrimMakeTuple) {
      for (size_t i = 1; i < cnode->inputs().size(); i++) {
        auto graphOutNode = cnode->input(i);
        if (!graphOutNode->isa<CNode>()) {
          MS_LOG(ERROR) << "Inputs of MakeTuple should be cNode";
          return nullptr;
        }
        std::string graphOutNodeName = graphOutNode->fullname_with_scope();
        auto graphOutIndex = nodeIdMap[graphOutNodeName];
        metaGraphT->outputIndex.emplace_back(graphOutIndex);
      }
      continue;
    }

    auto node = std::make_unique<schema::CNodeT>();
    node->name = cnode->fullname_with_scope();
    node->nodeType = schema::NodeType_CNode;
    // populate primitive
    if (primitive != nullptr) {
      primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
      MS_ASSERT(primitive != nullptr);
      std::string opType = primitive->name();
      auto nodeParser = AnfNodePopulaterRegistry::GetInstance()->GetNodePopulater(opType);
      if (nodeParser == nullptr) {
        MS_LOG(ERROR) << "Find op parser failed, opType: " << opType;
        return nullptr;
      }
      std::vector<schema::TensorT *> outputs;
      nodeParser->Parse(cnode, node.get(), &outputs);
      SetOpInputNode(cnode, metaGraphT.get(), node.get());
      SetOpOutputNode(outputs, metaGraphT.get(), node.get());
      metaGraphT->nodes.emplace_back(std::move(node));
      continue;
    }
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
    SetOpOutputNode(outputs, metaGraphT.get(), node.get());

    // add quant param
    node->quantType = primitiveT_value->GetQuantType();
    if (node->quantType == schema::QuantType_PostTraining) {
      MS_LOG(INFO) << "node: " << node->name << " add QuantParam";
      // activation
      auto activate_index = node->inputIndex[0];
      auto tensor_input = metaGraphT->allTensors[activate_index].get();
      auto input_quant_params = primitiveT_value->GetInputQuantParams();
      if (input_quant_params.empty()) {
        MS_LOG(WARNING) << "node: " << node->name << " input quant params is empty";
      } else {
        std::unique_ptr<schema::QuantParamT> input_quant_param =
          std::make_unique<schema::QuantParamT>(input_quant_params[0]);
        tensor_input->quantParams.emplace_back(std::move(input_quant_param));
      }
      tensor_input->dataType = kNumberTypeInt8;
      // output
      auto output_index = node->outputIndex[0];
      auto tensor_output = metaGraphT->allTensors[output_index].get();
      auto output_quant_params = primitiveT_value->GetOutputQuantParams();
      if (output_quant_params.empty()) {
        MS_LOG(WARNING) << "node: " << node->name << " output quant params is empty";
      } else {
        std::unique_ptr<schema::QuantParamT> output_quant_param =
          std::make_unique<schema::QuantParamT>(output_quant_params[0]);
        tensor_output->quantParams.emplace_back(std::move(output_quant_param));
      }
      tensor_output->dataType = kNumberTypeInt8;
      //      // TensorType
      //      valuePtr = primitive->GetAttr(kInputTensorDataType);
      //      if (valuePtr != nullptr) {
      //        MS_LOG(INFO) << "node: " << node->name << " input tensor data type: " << GetValue<int>(valuePtr);
      //        for (auto input : node->inputIndex) {
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
        // tensor->refCount = lite::MSCONST_WEIGHT_REFCOUNT;
        metaGraphT->inputIndex.emplace_back(input);
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
      for (auto &ite : paramValue->quant_param()) {
        auto quantPar = std::make_unique<schema::QuantParamT>();
        quantPar->scale = ite->scale;
        quantPar->zeroPoint = ite->zeroPoint;
        quantPar->min = ite->min;
        quantPar->max = ite->max;
        quantPar->narrowRange = ite->narrowRange;
        quantPar->inited = ite->inited;
        quantPar->numBits = ite->numBits;
        paramTensor->quantParams.emplace_back(std::move(quantPar));
        paramTensor->dataType = paramValue->tensor_type();
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

void AnfExporter::SetOpOutputNode(const std::vector<schema::TensorT *> &outputTensors, schema::MetaGraphT *graph,
                                  schema::CNodeT *cnode) {
  MS_ASSERT(nullptr != graph);
  MS_ASSERT(nullptr != cnode);
  std::string cnodeName = cnode->name;
  if (!outputTensors.empty()) {
    int i = 0;
    for (auto outputTensor : outputTensors) {
      std::string name = cnodeName + "_o:" + std::to_string(i);
      nodeIdMap[name] = graph->allTensors.size();
      cnode->outputIndex.emplace_back(graph->allTensors.size());
      graph->allTensors.emplace_back(outputTensor);
      i++;
    }
    return;
  }
  auto msTensor = new schema::TensorT();
  msTensor->nodeType = schema::NodeType_Parameter;
  cnode->outputIndex.emplace_back(graph->allTensors.size());
  nodeIdMap[cnodeName] = graph->allTensors.size();
  graph->allTensors.emplace_back(msTensor);
}

schema::MetaGraphT *Export(const FuncGraphPtr &funcGraph) {
  AnfExporter anfExporter;
  return anfExporter.Export(funcGraph);
}
}  // namespace mindspore::lite
