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
#include <memory>
#include "schema/inner/model_generated.h"
#include "frontend/operator/ops.h"
#include "src/param_value_lite.h"
#include "import_from_meta_graphT.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"
#include "src/ir/primitive_value.h"
#include "src/ir/primitive_t_value.h"
#include "include/errorcode.h"
#include "src/ops/ops.h"

namespace mindspore::lite {
void AnfImporterFromMetaGraphT::ConverterConstTensor() {
  MS_EXCEPTION_IF_NULL(meta_graph_);
  MS_EXCEPTION_IF_NULL(func_graph_);
  for (size_t i = 0; i < meta_graph_->allTensors.size(); i++) {
    auto &tensor = meta_graph_->allTensors.at(i);
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->nodeType != schema::NodeType_ValueNode) {
      continue;
    }
    MS_ASSERT(tensor->dims() != nullptr);
    auto parameter = func_graph_->add_parameter();
    std::vector<int> shape;
    for (int &dim : tensor->dims) {
      shape.push_back(dim);
    }
    auto type_id = static_cast<TypeId>(tensor->dataType);
    auto type_ptr = TypeIdToType(type_id);
    auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape);
    parameter->set_abstract(abstract_tensor);

    ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
    MS_EXCEPTION_IF_NULL(param_value);
    param_value->set_tensor_shape(shape);
    param_value->set_tensor_type(type_id);
    if (!tensor->data.empty()) {
      auto size = tensor->data.size();
      char *tensor_data = new char[size];
      std::memcpy(tensor_data, tensor->data.data(), size);
      MS_EXCEPTION_IF_NULL(tensor_data);
      param_value->set_tensor_addr(tensor_data);
      param_value->set_tensor_size(size);
    }
    if (tensor->quantParams.size() > 0) {
      std::unique_ptr<AnfQuantParam> quantParam = std::make_unique<AnfQuantParam>();
      quantParam->scale = tensor->quantParams[0]->scale;
      quantParam->zeroPoint = tensor->quantParams[0]->zeroPoint;
      quantParam->min = tensor->quantParams[0]->min;
      quantParam->max = tensor->quantParams[0]->max;
      quantParam->narrowRange = tensor->quantParams[0]->narrowRange;
      quantParam->numBits = tensor->quantParams[0]->numBits;
      quantParam->inited = tensor->quantParams[0]->inited;
      param_value->set_quant_param(quantParam);
    }
    parameter->set_default_param(param_value);
    AddNode(i, parameter);
  }
}

int AnfImporterFromMetaGraphT::ConverterCNode() {
  MS_EXCEPTION_IF_NULL(meta_graph_);
  MS_EXCEPTION_IF_NULL(func_graph_);
  for (size_t i = 0; i < meta_graph_->nodes.size(); i++) {
    auto &cNode = meta_graph_->nodes.at(i);
    MS_EXCEPTION_IF_NULL(cNode);

    bool flag = false;
    if (cNode->outputIndex.size() > 1) {
      flag = true;
    }
    auto primTValue = std::make_shared<PrimitiveTValue>(cNode->primitive.release());
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
    cNode->primitive = nullptr;
    auto value_node = NewValueNode(primTValue);

    std::vector<AnfNodePtr> op_inputs = {value_node};
    for (size_t j = 0; j < cNode->inputIndex.size(); j++) {
      auto node = GetNode(cNode->inputIndex.at(j));
      if (nullptr == node) {
        MS_LOG(ERROR) << "Can't find input node.";
        return RET_ERROR;
      }
      // todo: CheckInputNodeType, the first node should be op;
      op_inputs.push_back(node);
    }

    auto new_cnode = func_graph_->NewCNode(op_inputs);
    new_cnode->set_fullname_with_scope(cNode->name);

    std::vector<uint32_t> out_tensor_ids = cNode->outputIndex;

    AbstractBasePtrList ptr_list;
    int total = 0;
    for (auto out_tensor_id : out_tensor_ids) {
      if (nullptr != GetNode(out_tensor_id)) {
        ptr_list.push_back(GetNode(out_tensor_id)->abstract());
        continue;
      }
      std::vector<int> shape;
      auto &tensor = meta_graph_->allTensors.at(out_tensor_id);
      for (int &dim : tensor->dims) {
        shape.push_back(dim);
      }
      auto type_id = static_cast<TypeId>(tensor->dataType);
      auto type_ptr = TypeIdToType(type_id);
      auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape);
      auto getItemPrim = NewValueNode(prim::kPrimTupleGetItem);
      if (flag) {
        auto getItemIndex = NewValueNode(MakeValue<int>(total++));
        std::vector<AnfNodePtr> inputs{getItemPrim, new_cnode, getItemIndex};
        CNodePtr new_item_cnode = func_graph_->NewCNode(inputs);
        AddNode(out_tensor_id, new_item_cnode);
      } else {
        AddNode(out_tensor_id, new_cnode);
      }
      ptr_list.push_back(std::move(abstract_tensor));
    }
    new_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(ptr_list));
  }
  return RET_OK;
}

void AnfImporterFromMetaGraphT::AddReturnCNode() {
  MS_EXCEPTION_IF_NULL(meta_graph_);
  MS_EXCEPTION_IF_NULL(func_graph_);
  std::vector<AnfNodePtr> make_tuple_inputs;
  auto make_tuple_value_node = NewValueNode(prim::kPrimMakeTuple);
  make_tuple_inputs.emplace_back(make_tuple_value_node);
  for (auto tensor_id : meta_graph_->outputIndex) {
    make_tuple_inputs.emplace_back(GetNode(tensor_id));
  }
  auto make_tuple_cnode = func_graph_->NewCNode(make_tuple_inputs);
  make_tuple_cnode->set_fullname_with_scope("return tuple");

  std::vector<AnfNodePtr> op_inputs;
  auto value_node = NewValueNode(prim::kPrimReturn);
  op_inputs.emplace_back(value_node);
  op_inputs.emplace_back(make_tuple_cnode);
  auto cnode = func_graph_->NewCNode(op_inputs);
  cnode->set_fullname_with_scope("return");
  func_graph_->set_return(cnode);
}

FuncGraphPtr AnfImporterFromMetaGraphT::GetResult() { return this->func_graph_; }
}  // namespace mindspore::lite
