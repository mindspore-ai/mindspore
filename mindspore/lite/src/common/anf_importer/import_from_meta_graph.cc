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
  MS_EXCEPTION_IF_NULL(model);
  auto *meta_graph = model->GetMetaGraph();
  MS_EXCEPTION_IF_NULL(meta_graph);
  for (size_t i = 0; i < meta_graph->allTensors()->size(); i++) {
    auto *tensor = meta_graph->allTensors()->GetAs<schema::Tensor>(i);
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->nodeType() != schema::NodeType_ValueNode) {
      continue;
    }
    MS_ASSERT(tensor->dims() != nullptr);
    auto parameter = model->add_parameter();
    std::vector<int> shape;
    for (size_t j = 0; j < tensor->dims()->size(); ++j) {
      shape.push_back(tensor->dims()->data()[j]);
    }
    auto type_id = static_cast<TypeId>(tensor->dataType());
    auto type_ptr = TypeIdToType(type_id);
    auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape);
    parameter->set_abstract(abstract_tensor);

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
    AddNode(i, parameter);
  }
}

int AnfImporterFromMetaGraph::ConverterCNode() {
  MS_EXCEPTION_IF_NULL(model);
  auto *meta_graph = model->GetMetaGraph();
  MS_EXCEPTION_IF_NULL(meta_graph);
  auto cNodes = meta_graph->nodes();
  for (size_t i = 0; i < cNodes->size(); i++) {
    auto cNode = cNodes->GetAs<schema::CNode>(i);
    MS_EXCEPTION_IF_NULL(cNode);
    auto tensor_id = cNode->outputIndex()->data()[0];
    if (GetNode(tensor_id)) {
      continue;
    }

    auto prim = std::make_shared<PrimitiveValue>(model->GetOp(cNode->name()->str()));
    if (prim == nullptr) {
      MS_LOG(ERROR) << "th tensorDef in subGraphDef is nullptr";
      return RET_ERROR;
    }
    auto value_node = NewValueNode(prim);
    AddNode(tensor_id, value_node);

    std::vector<AnfNodePtr> op_inputs = {value_node};
    MS_EXCEPTION_IF_NULL(cNode->inputIndex());
    for (size_t j = 0; j < cNode->inputIndex()->size(); j++) {
      auto node = GetNode(*(cNode->inputIndex()->GetAs<uint32_t>(j)));
      if (nullptr == node) {
        MS_LOG(ERROR) << "Can't find input node.";
        return RET_ERROR;
      }
      // todo: CheckInputNodeType, the first node should be op;
      op_inputs.push_back(node);
    }
    auto cnode = model->NewCNode(op_inputs);
    auto node_name = std::string(cNode->name()->c_str());
    cnode->set_fullname_with_scope(node_name);
    AddNode(tensor_id, cnode);
  }
  return RET_OK;
}

void AnfImporterFromMetaGraph::AddReturnCNode() {
  MS_EXCEPTION_IF_NULL(model);
  auto *meta_graph = model->GetMetaGraph();
  MS_EXCEPTION_IF_NULL(meta_graph);
  std::vector<AnfNodePtr> op_inputs;
  auto value_node = NewValueNode(prim::kPrimReturn);
  op_inputs.push_back(value_node);
  auto tensor_id = meta_graph->outputIndex()->data()[0];
  op_inputs.push_back(GetNode(tensor_id));
  auto cnode = model->NewCNode(op_inputs);
  cnode->set_fullname_with_scope("return");
  model->set_return(cnode);
}
FuncGraphPtr AnfImporterFromMetaGraph::GetResult() { return this->model; }
}  // namespace mindspore::lite

