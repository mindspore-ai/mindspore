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
#include "backend/optimizer/pass/convert_const_input_to_tensor_input.h"

#include <vector>
#include <memory>
#include <utility>

#include "ir/graph_utils.h"
#include "backend/optimizer/common/helper.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/session/kernel_graph.h"
#include "backend/kernel_compiler/common_utils.h"

namespace mindspore {
namespace opt {
namespace {
AnfNodePtr CreateTensorInput(const AnfNodePtr &node, const KernelGraphPtr &kernel_graph, const AnfNodePtr &input_node) {
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
  } else {
    MS_LOG(EXCEPTION) << "The value should be a scalar or value tuple";
  }
  if (tensor_ptr == nullptr) {
    MS_LOG(WARNING) << "Create tensor failed";
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

AnfNodePtr ConstInputToTensorInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<AnfNodePtr> new_inputs;
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  auto inputs = cnode->inputs();
  new_inputs.push_back(inputs[0]);
  bool need_update = false;
  // the first input is primitive node which is not the real input
  for (size_t i = 0; i < inputs.size() - 1; ++i) {
    auto input_node = inputs[i + 1];
    if (IsValueNode<Scalar>(input_node) || IsValueNode<ValueTuple>(input_node)) {
      auto tensor_input = CreateTensorInput(cnode, kernel_graph, input_node);
      if (tensor_input == nullptr) {
        new_inputs.push_back(input_node);
        continue;
      }
      new_inputs.push_back(tensor_input);
      need_update = true;
    } else {
      new_inputs.push_back(input_node);
    }
  }
  if (need_update) {
    MS_EXCEPTION_IF_NULL(func_graph);
    auto new_cnode = func_graph->NewCNode(new_inputs);
    MS_EXCEPTION_IF_NULL(new_cnode);
    if (AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimDepend)) {
      new_cnode->set_abstract(new_inputs[1]->abstract());
    } else {
      new_cnode->set_abstract(cnode->abstract());
    }
    new_cnode->set_scope(cnode->scope());
    AnfAlgo::CopyNodeAttrs(cnode, new_cnode);
    if (kernel_graph != nullptr) {
      kernel_graph->FrontBackendlMapUpdate(cnode, new_cnode);
    }
    return new_cnode;
  }
  return nullptr;
}

AnfNodePtr ProcessGraphKernelOp(const AnfNodePtr &node) {
  auto sub_graph = AnfAlgo::GetCNodeFuncGraphPtr(node);
  MS_EXCEPTION_IF_NULL(sub_graph);
  auto mng = sub_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  std::vector<AnfNodePtr> todo;
  kernel::GetValidKernelNodes(sub_graph, &todo);

  for (auto &t : todo) {
    auto t_new_node = ConstInputToTensorInput(sub_graph, t->cast<CNodePtr>());
    if (t_new_node != nullptr && t_new_node != t) {
      (void)mng->Replace(t, t_new_node);
    }
  }

  return node;
}
}  // namespace

const AnfNodePtr ConvertConstInputToTensorInput::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                         const EquivPtr &) const {
  if (node == nullptr || func_graph == nullptr || AnfAlgo::CheckPrimitiveType(node, prim::kPrimTupleGetItem) ||
      AnfAlgo::CheckPrimitiveType(node, prim::kPrimMakeTuple)) {
    return nullptr;
  }
  if (!node->isa<CNode>()) {
    return nullptr;
  }
  if (AnfAlgo::IsGraphKernel(node)) {
    return ProcessGraphKernelOp(node);
  } else {
    return ConstInputToTensorInput(func_graph, node->cast<CNodePtr>());
  }
}
}  // namespace opt
}  // namespace mindspore
