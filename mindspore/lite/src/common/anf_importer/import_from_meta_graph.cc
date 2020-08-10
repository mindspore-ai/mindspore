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

#include "src/common/anf_importer/import_from_meta_graph.h"
#include <string>
#include <vector>
#include <memory>
#include "frontend/operator/ops.h"
#include "src/param_value_lite.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"
#include "src/ir/primitive_value.h"
#include "include/errorcode.h"

namespace mindspore::lite {
void AnfImporterFromMetaGraph::ConverterConstTensor() {
  MS_EXCEPTION_IF_NULL(model_);
  auto *meta_graph = model_->GetMetaGraph();
  MS_EXCEPTION_IF_NULL(meta_graph);
  num_of_tensors_ = meta_graph->allTensors()->size();
  for (size_t i = 0; i < num_of_tensors_; i++) {
    auto *tensor = meta_graph->allTensors()->GetAs<schema::Tensor>(i);
    MS_EXCEPTION_IF_NULL(tensor);
    if ((tensor->nodeType() != schema::NodeType_ValueNode) && (tensor->nodeType() != schema::NodeType_Parameter)) {
      continue;
    }
    MS_ASSERT(tensor->dims() != nullptr);
    auto parameter = model_->add_parameter();
    std::vector<int> shape;
    for (size_t j = 0; j < tensor->dims()->size(); ++j) {
      shape.push_back(tensor->dims()->data()[j]);
    }
    auto type_id = static_cast<TypeId>(tensor->dataType());  // todo: check error
    auto type_ptr = TypeIdToType(type_id);
    auto abstractBase = std::make_shared<abstract::AbstractTensor>(type_ptr, shape);
    // XXX TODO copy format
    parameter->set_abstract(abstractBase);
    parameter->set_name(std::string("Parameter"));

    if (tensor->nodeType() == schema::NodeType_ValueNode) {
      ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
      MS_EXCEPTION_IF_NULL(param_value);
      param_value->set_tensor_shape(shape);
      param_value->set_tensor_type(type_id);
      if (tensor->data() != nullptr) {
        auto size = tensor->data()->size();
        char *tensor_data = new char[size]();
        std::memcpy(tensor_data, tensor->data()->data(), size);
        MS_EXCEPTION_IF_NULL(tensor_data);
        param_value->set_tensor_addr(tensor_data);
        param_value->set_tensor_size(size);
      }
      parameter->set_default_param(param_value);
    }
    AddNode(i, parameter);
    model_->AddAnfNode(i, parameter);
  }
}

int AnfImporterFromMetaGraph::ConverterCNode() {
  MS_EXCEPTION_IF_NULL(model_);
  auto *meta_graph = model_->GetMetaGraph();
  MS_EXCEPTION_IF_NULL(meta_graph);

  // Crate CNode -- Order of inputs is as follows
  // First input should be the Primitive
  // Then we have CNodes that contribute to this CNode
  // Finally we Have the parameters

  // first itteration -- create CNode with primitive, create originator map
  for (size_t i = 0; i < meta_graph->nodes()->size(); i++) {
    auto cNode = meta_graph->nodes()->GetAs<schema::CNode>(i);
    MS_EXCEPTION_IF_NULL(cNode);
    auto prim = std::make_shared<PrimitiveValue>(model_->GetOp(cNode->name()->str()));
    if (prim == nullptr) {
      MS_LOG(ERROR) << "th tensorDef in subGraphDef is nullptr";
      return RET_ERROR;
    }
    auto value_node = NewValueNode(prim);
    // auto prim_name = std::string("PrimitivePy: ") + std::string(cNode->name()->c_str());
    // value_node->set_fullname_with_scope(prim_name);
    std::vector<AnfNodePtr> op_inputs = {value_node};

    auto cnode = model_->NewCNode(op_inputs);
    auto node_name = std::string(cNode->name()->c_str()) + std::to_string(i);
    cnode->set_fullname_with_scope(node_name);
    AddNode(num_of_tensors_ + i, cnode);

    for (size_t j = 0; j < cNode->outputIndex()->size(); j++) {
      int tensor_id = cNode->outputIndex()->data()[j];
      originator_[tensor_id] = cnode;
    }
  }
  // second itteration -- fill in input CNodes and Parameters
  // populate map
  for (size_t i = 0; i < meta_graph->nodes()->size(); i++) {
    std::vector<int> input;
    std::vector<int> output;
    int tensor_id;
    auto cNode = meta_graph->nodes()->GetAs<schema::CNode>(i);
    MS_EXCEPTION_IF_NULL(cNode);
    auto cnode = std::dynamic_pointer_cast<CNode>(GetNode(num_of_tensors_ + i));

    for (size_t j = 0; j < cNode->outputIndex()->size(); j++) {
      tensor_id = cNode->outputIndex()->data()[j];
      output.push_back(tensor_id);
    }

    MS_EXCEPTION_IF_NULL(cNode->inputIndex());
    for (size_t j = 0; j < cNode->inputIndex()->size(); j++) {
      tensor_id = cNode->inputIndex()->data()[j];
      input.push_back(tensor_id);
      auto *tensor = meta_graph->allTensors()->GetAs<schema::Tensor>(tensor_id);
      MS_EXCEPTION_IF_NULL(tensor);
      if ((tensor->nodeType() == schema::NodeType_Parameter) && (originator_[tensor_id] != nullptr)) {
        cnode->add_input(originator_[tensor_id]);
      }
    }
    // finally add all the Parameters (which are ValueNodes)
    for (size_t j = 0; j < cNode->inputIndex()->size(); j++) {
      tensor_id = cNode->inputIndex()->data()[j];
      auto *tensor = meta_graph->allTensors()->GetAs<schema::Tensor>(tensor_id);
      MS_EXCEPTION_IF_NULL(tensor);
      if ((tensor->nodeType() == schema::NodeType_ValueNode) && (GetNode(tensor_id) != nullptr)) {
        cnode->add_input(GetNode(tensor_id));
      }
    }

    model_->AddCNodeInputOutput(cnode->fullname_with_scope(), input, output);
  }

  return RET_OK;
}

void AnfImporterFromMetaGraph::AddReturnCNode() {
  MS_EXCEPTION_IF_NULL(model_);
  auto *meta_graph = model_->GetMetaGraph();
  MS_EXCEPTION_IF_NULL(meta_graph);
  std::vector<int> input;
  std::vector<int> output;
  std::vector<AnfNodePtr> op_inputs;
  auto value_node = NewValueNode(prim::kPrimReturn);
  // value_node->set_fullname_with_scope("Primitive");
  op_inputs.push_back(value_node);
  for (int i = 0; i < meta_graph->outputIndex()->size(); i++) {
    auto prev_cnode = originator_[meta_graph->outputIndex()->data()[i]];
    if (prev_cnode != nullptr) op_inputs.push_back(prev_cnode);
    input.push_back(meta_graph->outputIndex()->data()[i]);
  }
  auto cnode = model_->NewCNode(op_inputs);
  cnode->set_fullname_with_scope("return");
  model_->set_return(cnode);
  model_->AddCNodeInputOutput(cnode->fullname_with_scope(), input, output);
}
FuncGraphPtr AnfImporterFromMetaGraph::GetResult() { return this->model_; }
}  // namespace mindspore::lite
