/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ge/add_noop_to_es_grad.h"
#include <vector>
#include <memory>
#include "ops/nn_ops.h"
#include "ops/array_ops.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "mindspore/core/ops/framework_ops.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
bool IsESGradOps(const BaseRef &ref) {
  if (utils::isa<AnfNodePtr>(ref)) {
    auto node = utils::cast<AnfNodePtr>(ref);
    MS_EXCEPTION_IF_NULL(node);
    const mindspore::HashSet<PrimitivePtr, PrimitiveHasher, PrimitiveEqual> es_grad_prims = {
      prim::kPrimEmbeddingApplyAdam, prim::kPrimEmbeddingApplyAdamW, prim::kPrimEmbeddingApplyAdaGrad,
      prim::kPrimEmbeddingApplyFtrl};
    if (IsOneOfPrimitiveCNode(node, es_grad_prims)) {
      return true;
    }
  }
  return false;
}

AnfNodePtr GetRealInput(const CNodePtr &node, const AnfNodePtr &input) {
  if (input == nullptr || node == nullptr) {
    return nullptr;
  }
  AnfNodePtr pred = nullptr;
  auto input_cnode = input->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(input_cnode);
  pred = input_cnode->input(1);
  return pred;
}

void ChangeValueNode(const KernelGraphPtr &kernel_graph, const CNodePtr &cnode) {
  auto inputs = cnode->inputs();
  for (size_t i = 1; i < inputs.size(); i++) {
    if (inputs[i]->isa<CNode>()) {
      continue;
    }
    auto abs = inputs[i]->abstract();
    auto value_ptr = abs->BuildValue();
    auto new_value_node = kernel_graph->NewValueNode(value_ptr);
    MS_EXCEPTION_IF_NULL(new_value_node);
    cnode->set_input(i, new_value_node);
  }
}

void RemoveOptimizerNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto cast_users = manager->node_users()[node];
  if (!cast_users.empty()) {
    auto opt = cast_users.begin();
    auto opt_node = opt->first;
    MS_EXCEPTION_IF_NULL(opt_node);
    auto opt_users = manager->node_users()[opt_node];
    for (const auto &node_index : opt_users) {
      AnfNodePtr output = node_index.first;
      MS_EXCEPTION_IF_NULL(output);
      manager->SetEdge(output, node_index.second, node);
    }
  }
}

AnfNodePtr InsertNoOpForOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  CNodePtr cast = node->cast<CNodePtr>();
  auto adam = cast->input(1);
  MS_EXCEPTION_IF_NULL(adam);
  auto adam_cnode = adam->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(adam_cnode);
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // There is an issue of inconsistent input addresses that will be resolved in the future
  ChangeValueNode(kernel_graph, adam_cnode);
  RemoveOptimizerNode(func_graph, cast);
  auto cast_value = kernel_graph->NewValueNode(MakeValue(std::make_shared<tensor::Tensor>(1)));
  cast->set_input(1, cast_value);
  auto return_node = func_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  std::vector<AnfNodePtr> new_node_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimUpdateState->name())),
                                             adam};
  auto no_op = NewCNode(new_node_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(no_op);
  if (!common::AnfAlgo::HasNodeAttr(kAttrVisited, return_node)) {
    std::vector<AnfNodePtr> tuple_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name())),
                                            no_op};
    auto tuple_op = NewCNode(tuple_inputs, func_graph);
    MS_EXCEPTION_IF_NULL(tuple_op);
    auto ret_input = return_node->input(1);
    MS_EXCEPTION_IF_NULL(ret_input);
    auto real_input = GetRealInput(return_node, ret_input);
    MS_EXCEPTION_IF_NULL(real_input);
    std::vector<AnfNodePtr> depend_node_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                                  real_input, tuple_op};
    auto depend_op = NewCNode(depend_node_inputs, func_graph);
    MS_EXCEPTION_IF_NULL(depend_op);
    depend_op->set_abstract(real_input->abstract());
    if (ret_input != node) {
      kernel_graph->ReplaceNode(ret_input, depend_op);
    }
    common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), return_node);
  } else {
    auto depend = return_node->input(1);
    MS_EXCEPTION_IF_NULL(depend);
    if (!IsPrimitiveCNode(depend, prim::kPrimDepend)) {
      MS_LOG(ERROR) << "Need Depend ops, but get " << depend->fullname_with_scope();
      return nullptr;
    }
    auto depend_cnode = depend->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(depend_cnode);
    auto tuple = depend_cnode->input(2);
    MS_EXCEPTION_IF_NULL(tuple);
    if (!IsPrimitiveCNode(tuple, prim::kPrimMakeTuple)) {
      MS_LOG(ERROR) << "Need MakeTuple ops, but get " << tuple->fullname_with_scope();
      return nullptr;
    }
    auto tuple_cnode = tuple->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(tuple_cnode);
    tuple_cnode->add_input(no_op);
  }
  common::AnfAlgo::SetNodeAttr(kAttrNotRemove, MakeValue(true), no_op);
  return no_op;
}
}  // namespace

const BaseRef AddNoOpToESGrad::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr is_es_grad_node = std::make_shared<CondVar>(IsESGradOps);
  return VectorRef({prim::kPrimCast, is_es_grad_node, X});
}

const AnfNodePtr AddNoOpToESGrad::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                          const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (InsertNoOpForOutput(func_graph, node) == nullptr) {
    MS_LOG(EXCEPTION) << "Insert NoOp for node: " << node->fullname_with_scope() << " failed.";
  }
  return node;
}
}  // namespace opt
}  // namespace mindspore
