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
#include "tools/optimizer/graph/input_and_output_variable_pass.h"
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

bool InputAndOutputVariablePass::Run(const FuncGraphPtr &graph) {
  MS_LOG(INFO) << "Start to run input and output variable pass";
  MS_ASSERT(graph != nullptr);
  auto parameters = graph->parameters();
  if (parameters.size() < static_cast<size_t>(inputs_variable_index_.back()) + 1) {
    MS_LOG(ERROR) << "the size of parameters is" << parameters.size() << " is less than "
                  << inputs_variable_index_.back() + 1;
    return false;
  }
  auto outputs = opt::GetNodeInputs(graph->get_return());
  if (outputs.size() < static_cast<size_t>(outputs_variable_index_.back()) + 1) {
    MS_LOG(ERROR) << "The output number " << outputs.size() << " is less than " << outputs_variable_index_.back() + 1;
    return false;
  }
  if (inputs_variable_index_.size() != outputs_variable_index_.size()) {
    MS_LOG(ERROR) << "The num of input variable " << inputs_variable_index_.size() << " is not equal"
                  << " the number of output variable" << outputs_variable_index_.size();
    return false;
  }
  auto output_names = FuncGraphUtils::GetFuncGraphOutputNames(graph);
  std::vector<CNodePtr> assign_nodes;
  for (size_t i = 0; i < outputs_variable_index_.size(); ++i) {
    size_t output_index = static_cast<size_t>(outputs_variable_index_[i]);
    auto inc_decoder_output = outputs[output_index].first;
    size_t parameter_index = static_cast<size_t>(inputs_variable_index_[i]);
    auto inc_decoder_parameter = parameters[parameter_index]->cast<ParameterPtr>();
    if (inc_decoder_parameter == nullptr) {
      MS_LOG(ERROR) << "Parameter is nullptr";
      return false;
    }
    std::string inc_decoder_parameter_name = output_names[output_index];
    inc_decoder_parameter->set_name(inc_decoder_parameter_name);
    auto assign = CreateAssign(inc_decoder_output, inc_decoder_parameter, graph);
    if (assign == nullptr) {
      MS_LOG(ERROR) << "Create assign node failed";
      return false;
    }
    assign_nodes.emplace_back(assign);
  }

  auto depend_node = std::make_shared<ops::Depend>();
  if (depend_node == nullptr) {
    MS_LOG(ERROR) << "depend node is nullptr";
    return false;
  }
  auto depend_prim = depend_node->GetPrim();
  if (depend_prim == nullptr) {
    MS_LOG(ERROR) << "depend prim is nullptr";
    return false;
  }
  auto output = graph->output();
  if (output == nullptr) {
    MS_LOG(ERROR) << "output node is nullptr";
    return false;
  }
  if (!utils::isa<CNode>(output)) {
    MS_LOG(ERROR) << "output node is not cnode";
    return false;
  }
  auto old_output_node = output->cast<CNodePtr>();
  auto make_tuple = old_output_node->input(1);
  if (make_tuple == nullptr) {
    MS_LOG(ERROR) << "make tuple input1 is nullptr";
    return false;
  }
  if (!utils::isa<CNode>(make_tuple)) {
    MS_LOG(ERROR) << "make tuple is not cnode";
    return false;
  }
  auto make_tuple_node = make_tuple->cast<CNodePtr>();
  auto output_1 = make_tuple_node->input(1);

  AnfNodePtrList new_make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  AbstractBasePtrList abstract_list;
  for (size_t i = 0; i < assign_nodes.size(); ++i) {
    new_make_tuple_inputs.emplace_back(assign_nodes[i]);
    abstract_list.emplace_back(assign_nodes[i]->abstract());
  }
  auto new_make_tuple = graph->NewCNode(new_make_tuple_inputs);
  new_make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  auto depend_cnode = graph->NewCNode(depend_prim, {output_1, new_make_tuple});
  depend_cnode->set_abstract(output_1->abstract());
  graph->set_output(depend_cnode);
  MS_LOG(INFO) << "Run input and output variable pass success";
  return true;
}

CNodePtr InputAndOutputVariablePass::CreateAssign(const AnfNodePtr &anf_node, const ParameterPtr &parameter,
                                                  const FuncGraphPtr &graph) {
  MS_CHECK_TRUE_MSG(parameter != nullptr, nullptr, "parameter is nullptr");
  auto assign_node = std::make_shared<ops::Assign>();
  if (assign_node == nullptr) {
    MS_LOG(ERROR) << "assign_node is nullptr";
    return nullptr;
  }
  auto assign_prim = assign_node->GetPrim();
  if (assign_prim == nullptr) {
    MS_LOG(ERROR) << "assign_prim is nullptr";
    return nullptr;
  }
  auto abstract = parameter->abstract();
  MS_CHECK_TRUE_MSG(abstract != nullptr, nullptr, "abstract is nullptr");

  ShapeVector shape;
  if (FetchShapeFromAbstract(abstract, &shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "Fetch shape from abstract failed";
    return nullptr;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstract)) {
    MS_LOG(ERROR) << "abstract of node is not valid";
    return nullptr;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract);
  auto type_ptr = abstract_tensor->element()->GetTypeTrack();
  if (type_ptr == nullptr) {
    MS_LOG(ERROR) << "type ptr is nullptr";
    return nullptr;
  }
  tensor::TensorPtr tensor_data = std::make_shared<tensor::Tensor>(type_ptr->type_id(), shape);
  float *val = static_cast<float *>(tensor_data->data_c());
  for (size_t i = 0; i < tensor_data->DataSize(); ++i) {
    *(val + i) = 0;
  }
  parameter->set_default_param(tensor_data);
  auto cnode = graph->NewCNode(assign_prim, {parameter, anf_node});
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "cnode is nullptr";
    return nullptr;
  }
  cnode->set_abstract(tensor_data->ToAbstract()->Broaden());
  return cnode;
}

}  // namespace mindspore::opt
