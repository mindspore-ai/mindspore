/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/graph/output_variable_pass.h"
#include <memory>
#include <utility>
#include <vector>
#include <string>
#include "mindspore/core/ops/lite_ops.h"
#include "ops/fusion/conv2d_fusion.h"
#include "mindspore/lite/include/errorcode.h"
#include "ops/op_utils.h"
#include "ops/assign.h"
#include "ops/depend.h"
#include "ops/sequence_ops.h"
#include "tools/common/func_graph_utils.h"

namespace mindspore::opt {
namespace {
constexpr size_t kInputNumber = 2;
}

bool OutputVariablePass::Run(const FuncGraphPtr &graph) {
  MS_LOG(INFO) << "Start to run output variable pass";
  MS_ASSERT(graph != nullptr);
  auto return_node = graph->get_return();
  if (return_node == nullptr) {
    MS_LOG(ERROR) << "Return node is nullptr";
    return false;
  }
  if (return_node->inputs().size() < kInputNumber) {
    MS_LOG(ERROR) << "return node input size less than 2";
    return false;
  }
  auto make_tuple = return_node->input(1)->cast<CNodePtr>();
  if (make_tuple == nullptr) {
    MS_LOG(ERROR) << "Make tuple node is nullptr";
    return false;
  }

  if (make_tuple->inputs().size() - 1 < static_cast<size_t>(outputs_variable_index_.back()) + 1) {
    MS_LOG(ERROR) << "The output number  " << make_tuple->inputs().size() - 1 << " is less than "
                  << outputs_variable_index_.back() + 1;
    return false;
  }
  auto outputs_name = FuncGraphUtils::GetFuncGraphOutputNames(graph);
  for (size_t i = 0; i < outputs_variable_index_.size(); ++i) {
    int64_t output_index = outputs_variable_index_[i];
    auto make_tuple_input = make_tuple->input(output_index);
    if (make_tuple_input == nullptr) {
      MS_LOG(ERROR) << "The " << output_index << "output  is nullptr";
      return false;
    }
    if (!utils::isa<abstract::AbstractTensorPtr>(make_tuple_input->abstract())) {
      MS_LOG(ERROR) << "abstract base should be abstract tensor";
      return false;
    }
    auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(make_tuple_input->abstract());
    auto type_ptr = abstract_tensor->element()->GetTypeTrack();
    if (type_ptr == nullptr) {
      MS_LOG(ERROR) << "type ptr is nullptr";
      return false;
    }
    abstract::ShapePtr shape = dyn_cast<abstract::Shape>(make_tuple_input->Shape());
    tensor::TensorPtr tensor_data = std::make_shared<tensor::Tensor>(type_ptr->type_id(), shape->shape());

    float *data_addr = static_cast<float *>(tensor_data->data_c());
    for (size_t j = 0; i < tensor_data->DataSize(); ++j) {
      *(data_addr + j) = 0;
    }
    auto full_encoder_parameter = graph->add_parameter();
    if (full_encoder_parameter == nullptr) {
      MS_LOG(ERROR) << "full decoder parameter is nullptr";
      return false;
    }
    std::string full_encoder_parameter_name = outputs_name[output_index];
    full_encoder_parameter->set_name(full_encoder_parameter_name);
    full_encoder_parameter->set_default_param(tensor_data);
    full_encoder_parameter->set_abstract(tensor_data->ToAbstract()->Broaden());
    auto assign_node = std::make_shared<ops::Assign>();
    if (assign_node == nullptr) {
      MS_LOG(ERROR) << "assign node is nullptr";
      return false;
    }
    auto assign_primitive = assign_node->GetPrim();
    if (assign_primitive == nullptr) {
      MS_LOG(ERROR) << "assign primitive is nullptr";
      return false;
    }
    auto cnode = graph->NewCNode(assign_primitive, {full_encoder_parameter, make_tuple_input});
    if (cnode == nullptr) {
      MS_LOG(ERROR) << "new assign cnode is nullptr";
      return false;
    }
    cnode->set_abstract(tensor_data->ToAbstract()->Broaden());
    assign_nodes_.emplace_back(cnode);
  }
  if (!CreateDependNode(graph)) {
    MS_LOG(ERROR) << "Add depend node to graph failed";
    return false;
  }
  MS_LOG(INFO) << "Run output variable pass success";
  return true;
}

bool OutputVariablePass::CreateDependNode(const FuncGraphPtr &graph) {
  auto depend_op = std::make_shared<ops::Depend>();
  if (depend_op == nullptr) {
    MS_LOG(ERROR) << "depend op is nullptr";
    return false;
  }
  auto depend_primitive = depend_op->GetPrim();
  if (depend_primitive == nullptr) {
    MS_LOG(ERROR) << "depend primitive is nullptr";
    return false;
  }
  AnfNodePtrList new_make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  AbstractBasePtrList abstract_list;
  for (size_t i = 0; i < assign_nodes_.size(); ++i) {
    new_make_tuple_inputs.emplace_back(assign_nodes_[i]);
    abstract_list.emplace_back(assign_nodes_[i]->abstract());
  }
  auto new_make_tuple = graph->NewCNode(new_make_tuple_inputs);
  if (new_make_tuple == nullptr) {
    MS_LOG(ERROR) << "new make tuple is nullptr";
    return false;
  }
  new_make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  auto output_nodes = opt::GetNodeInputs(graph->get_return());
  if (output_nodes.empty()) {
    MS_LOG(ERROR) << "output nodes is empty";
    return false;
  }
  auto depend_node = graph->NewCNode(depend_primitive, {output_nodes[0].first, new_make_tuple});
  if (depend_node == nullptr) {
    MS_LOG(ERROR) << "new depend cnode failed";
    return false;
  }
  depend_node->set_abstract(output_nodes[0].first->abstract());
  graph->set_output(depend_node);
  return true;
}
}  // namespace mindspore::opt
