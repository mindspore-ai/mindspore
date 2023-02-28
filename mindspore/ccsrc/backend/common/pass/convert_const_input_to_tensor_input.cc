/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/convert_const_input_to_tensor_input.h"

#include <vector>
#include <memory>
#include <set>

#include "ir/graph_utils.h"
#include "backend/common/optimizer/helper.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/kernel_graph.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace opt {
namespace {
AnfNodePtr CreateTensorInput(const KernelGraphPtr &kernel_graph, const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  auto value_node = input_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  tensor::TensorPtr tensor_ptr = nullptr;
  if (value->isa<Scalar>()) {
    tensor_ptr = ScalarToTensor(value->cast<ScalarPtr>());
  } else if (value->isa<ValueTuple>()) {
    tensor_ptr = CreateTupleTensor(value->cast<ValueTuplePtr>());
  } else if (value->isa<ValueList>()) {
    tensor_ptr = CreateTupleTensor(std::make_shared<ValueTuple>(value->cast<ValueListPtr>()->value()));
  } else {
    MS_LOG(EXCEPTION) << "The value should be a scalar or value tuple";
  }
  if (tensor_ptr == nullptr) {
    MS_LOG(DEBUG) << "Create tensor failed";
    return nullptr;
  }
  auto tensor_input = std::make_shared<ValueNode>(tensor_ptr);
  MS_EXCEPTION_IF_NULL(tensor_input);
  tensor_input->set_abstract(tensor_ptr->ToAbstract());
  if (kernel_graph != nullptr) {
    tensor_input = kernel_graph->NewValueNode(tensor_input);
    kernel_graph->AddValueNodeToGraph(tensor_input);
  } else {
    tensor_input = MakeValueNode(tensor_input);
  }
  tensor_input->set_scope(input_node->scope());
  return tensor_input;
}
}  // namespace

AnfNodePtr ConvertConstInputToTensorInput::ConstInputToTensorInput(const FuncGraphPtr &func_graph,
                                                                   const CNodePtr &cnode) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  const std::set<std::string> no_need_to_convert_nodes = {kStackOpName};
  auto node_type = common::AnfAlgo::GetCNodeName(cnode);
  if (no_need_to_convert_nodes.find(node_type) != no_need_to_convert_nodes.end()) {
    return nullptr;
  }
  std::vector<AnfNodePtr> new_inputs;
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  auto inputs = cnode->inputs();
  new_inputs.push_back(inputs[0]);
  bool need_update = false;
  std::vector<int64_t> fake_tensor_pos;
  // the first input is primitive node which is not the real input
  for (size_t i = 0; i < inputs.size() - 1; ++i) {
    auto input_node = inputs[i + 1];
    if (AnfAlgo::IsScalarConvertToTensor(input_node, cnode) || IsValueNode<ValueSequence>(input_node)) {
      auto tensor_input = CreateTensorInput(kernel_graph, input_node);
      if (tensor_input == nullptr) {
        new_inputs.push_back(input_node);
        continue;
      }
      new_inputs.push_back(tensor_input);
      fake_tensor_pos.push_back(i);
      need_update = true;
    } else {
      new_inputs.push_back(input_node);
    }
  }
  if (need_update) {
    MS_EXCEPTION_IF_NULL(func_graph);
    auto new_cnode = NewCNode(new_inputs, func_graph);
    MS_EXCEPTION_IF_NULL(new_cnode);
    new_cnode->set_primal_attrs(cnode->primal_attrs());
    new_cnode->set_attrs(cnode->attrs());
    if (common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimDepend)) {
      new_cnode->set_abstract(new_inputs[1]->abstract());
    } else {
      new_cnode->set_abstract(cnode->abstract());
    }
    new_cnode->set_scope(cnode->scope());
    common::AnfAlgo::CopyNodeAttrs(cnode, new_cnode);
    if (kernel_graph != nullptr) {
      kernel_graph->FrontBackendlMapUpdate(cnode, new_cnode);
    }
    if (common::AnfAlgo::CheckPrimitiveType(new_cnode, prim::kPrimPrint) && fake_tensor_pos.size() > 0) {
      common::AnfAlgo::SetNodeAttr(kFakeTensorPos, MakeValue<std::vector<int64_t>>(fake_tensor_pos), new_cnode);
    }
    return new_cnode;
  }
  return nullptr;
}

const AnfNodePtr ConvertConstInputToTensorInput::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                         const EquivPtr &) const {
  // The virtual node maybe the output node of graph and can't miss the attribute of value.
  if (node == nullptr || func_graph == nullptr || common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimTupleGetItem) ||
      common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimMakeTuple) ||
      common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimDepend)) {
    return nullptr;
  }
  if (!node->isa<CNode>()) {
    return nullptr;
  }

  return ConstInputToTensorInput(func_graph, node->cast<CNodePtr>());
}
}  // namespace opt
}  // namespace mindspore
