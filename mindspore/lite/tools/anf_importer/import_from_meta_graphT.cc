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

#include <vector>
#include <algorithm>
#include "schema/inner/model_generated.h"
#include "frontend/operator/ops.h"
#include "src/param_value_lite.h"
#include "import_from_meta_graphT.h"
#include "utils/log_adapter.h"
#include "include/errorcode.h"


namespace mindspore::lite {
int AnfImporterFromMetaGraphT::ConverterConstTensor() {
  MS_ASSERT(nullptr != meta_graph_);
  MS_ASSERT(nullptr != func_graph_);
  for (size_t i = 0; i < meta_graph_->allTensors.size(); i++) {
    auto &tensor = meta_graph_->allTensors.at(i);
    MS_ASSERT(tensor != nullptr);
    // converter weight and graph input into parameter node
    if (tensor->nodeType != schema::NodeType_ValueNode) {
      continue;
    }
    MS_ASSERT(tensor->dims() != nullptr);
    auto parameter = func_graph_->add_parameter();
    std::vector<int> shape(tensor->dims.size());
    std::copy(tensor->dims.begin(), tensor->dims.end(), shape.begin());
    auto type_id = static_cast<TypeId>(tensor->dataType);
    auto type_ptr = TypeIdToType(type_id);
    auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape);
    abstract_tensor->set_format(tensor->format);
    parameter->set_abstract(abstract_tensor);
    parameter->set_name("const_" + std::to_string(i));

    ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
    MS_ASSERT(param_value != nullptr);
    param_value->set_tensor_shape(shape);
    param_value->set_tensor_type(type_id);
    if (!tensor->data.empty()) {
      auto size = tensor->data.size();
      char *tensor_data = new (std::nothrow) char[size];
      if (tensor_data == nullptr) {
        MS_LOG(ERROR) << "new char[] failed";
        return RET_ERROR;
      }
      std::memcpy(tensor_data, tensor->data.data(), size);
      param_value->set_tensor_addr(tensor_data);
      param_value->set_tensor_size(size);
    }
//    if (!tensor->quantParams.empty()) {
//      std::unique_ptr<AnfQuantParam> quantParam = std::make_unique<AnfQuantParam>();
//      quantParam->scale = tensor->quantParams[0]->scale;
//      quantParam->zeroPoint = tensor->quantParams[0]->zeroPoint;
//      quantParam->min = tensor->quantParams[0]->min;
//      quantParam->max = tensor->quantParams[0]->max;
//      quantParam->narrowRange = tensor->quantParams[0]->narrowRange;
//      quantParam->numBits = tensor->quantParams[0]->numBits;
//      quantParam->inited = tensor->quantParams[0]->inited;
//      param_value->set_quant_param(quantParam);
//    }
    parameter->set_default_param(param_value);
    AddNode(i, parameter);
  }
  return RET_OK;
}

ValueNodePtr AnfImporterFromMetaGraphT::ConvertPrimitive(const std::unique_ptr<schema::CNodeT> &cNode) {
  MS_ASSERT(nullptr != meta_graph_);
  MS_ASSERT(nullptr != cNode);
  auto primTValue = std::make_shared<PrimitiveTValue>(cNode->primitive.release());
  cNode->primitive = nullptr;
  // add quant parameter
  if (cNode->quantType == schema::QuantType_AwareTraining) {
    primTValue->SetQuantType(cNode->quantType);
    for (int index : cNode->inputIndex) {
      std::vector<schema::QuantParamT> quant_params = {*(meta_graph_->allTensors[index]->quantParams[0])};
      primTValue->AddInputQuantParam(quant_params);
    }
    for (int index : cNode->outputIndex) {
      std::vector<schema::QuantParamT> quant_params = {*(meta_graph_->allTensors[index]->quantParams[0])};
      primTValue->AddOutputQuantParam(quant_params);
    }
  }
  auto value_node = NewValueNode(primTValue);
  return value_node;
}

abstract::AbstractTensorPtr AnfImporterFromMetaGraphT::ConvertTensorToAbstractTensor(
  const std::unique_ptr<schema::TensorT> &tensor) {
  MS_ASSERT(nullptr != tensor);
  std::vector<int> shape(tensor->dims.size());
  std::copy(tensor->dims.begin(), tensor->dims.end(), shape.begin());
  auto type_id = static_cast<TypeId>(tensor->dataType);
  auto type_ptr = TypeIdToType(type_id);
  return std::make_shared<abstract::AbstractTensor>(type_ptr, shape);
}

void AnfImporterFromMetaGraphT::ConvertAbstract(const std::unique_ptr<schema::CNodeT> &src_cnode,
                                                const CNodePtr &dst_cnode) {
  MS_ASSERT(nullptr != meta_graph_);
  MS_ASSERT(nullptr != src_cnode);
  MS_ASSERT(nullptr != dst_cnode);
  std::vector<uint32_t> out_tensor_ids = src_cnode->outputIndex;
  if (out_tensor_ids.size() == 1) {
    auto out_tensor_id = out_tensor_ids.front();
    MS_ASSERT(meta_graph_->allTensors.size() > out_tensor_id);
    auto &tensor = meta_graph_->allTensors.at(out_tensor_id);
    MS_ASSERT(nullptr != tensor);
    dst_cnode->set_abstract(ConvertTensorToAbstractTensor(tensor));
    AddNode(out_tensor_id, dst_cnode);
  } else {
    AbstractBasePtrList abstract_list;
    for (size_t i = 0; i < out_tensor_ids.size(); i++) {
      auto out_tensor_id = out_tensor_ids.at(i);
      MS_ASSERT(meta_graph_->allTensors.size() > out_tensor_id);
      auto &tensor = meta_graph_->allTensors.at(out_tensor_id);
      MS_ASSERT(nullptr != tensor);
      abstract_list.emplace_back(ConvertTensorToAbstractTensor(tensor));
      auto tuple_get_item_prim = NewValueNode(GetTupleGetItemPrim());
      auto get_item_value = NewValueNode(MakeValue<int>(i));
      std::vector<AnfNodePtr> inputs{tuple_get_item_prim, dst_cnode, get_item_value};
      CNodePtr get_item_cnode = func_graph_->NewCNode(inputs);
      AddNode(out_tensor_id, get_item_cnode);
    }
    dst_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  }
}

int AnfImporterFromMetaGraphT::ConverterCNode() {
  MS_ASSERT(nullptr != meta_graph_);
  MS_ASSERT(nullptr != func_graph_);
  for (const auto &cNode : meta_graph_->nodes) {
    MS_ASSERT(nullptr != cNode);

    std::vector<AnfNodePtr> op_inputs = {ConvertPrimitive(cNode)};
    for (unsigned int j : cNode->inputIndex) {
      auto node = GetNode(j);
      if (nullptr == node) {
        MS_LOG(ERROR) << "Can't find input node.";
        return RET_ERROR;
      }
      // todo: CheckInputNodeType, the first node should be op;
      op_inputs.push_back(node);
    }
    auto new_cnode = func_graph_->NewCNode(op_inputs);
    new_cnode->set_fullname_with_scope(cNode->name);
    ConvertAbstract(cNode, new_cnode);
  }
  return RET_OK;
}

int AnfImporterFromMetaGraphT::AddReturnCNode() {
  MS_EXCEPTION_IF_NULL(meta_graph_);
  MS_EXCEPTION_IF_NULL(func_graph_);
  std::vector<AnfNodePtr> make_tuple_inputs;
  auto make_tuple_prim = NewValueNode(GetMakeTuplePrim());
  make_tuple_inputs.emplace_back(make_tuple_prim);
  for (auto tensor_id : meta_graph_->outputIndex) {
    auto cNode = GetNode(tensor_id);
    if (nullptr == cNode) {
      MS_LOG(ERROR) << "Can't find input node.";
      return RET_ERROR;
    }
    make_tuple_inputs.emplace_back(cNode);
  }
  auto make_tuple_cnode = func_graph_->NewCNode(make_tuple_inputs);
  make_tuple_cnode->set_fullname_with_scope("return tuple");

  std::vector<AnfNodePtr> op_inputs;
  auto value_node = NewValueNode(GetReturnPrim());
  op_inputs.emplace_back(value_node);
  op_inputs.emplace_back(make_tuple_cnode);
  auto cnode = func_graph_->NewCNode(op_inputs);
  cnode->set_fullname_with_scope("return");
  func_graph_->set_return(cnode);
  return RET_OK;
}

FuncGraphPtr AnfImporterFromMetaGraphT::GetResult() { return this->func_graph_; }
}  // namespace mindspore::lite
