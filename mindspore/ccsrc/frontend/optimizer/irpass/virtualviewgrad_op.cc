/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/irpass/virtualviewgrad_op.h"

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "frontend/optimizer/irpass/view_inplace_utils.h"
#include "primitive/other_ops.h"
#include "ir/graph_utils.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace {
void InsertVirtualViewGradAfterInplaceCNodeInner(const FuncGraphPtr &func_graph, const CNodePtr &view_cnode,
                                                 const AnfNodePtr &umonad, const FuncGraphManagerPtr &manager) {
  // Insert VirtualViewGrad op recursively
  // eg:
  // CNode1 = PrimFunc_InplaceAddExt(x_view_output2, 1, U1)
  // U2 = UpdateState(U1, CNode1)
  // ==>
  // ...
  // CNode2 = VirtualViewGrad(x_view_input2, x_view_output2, U2)
  // U3 = UpdateState(U2, CNode2)
  // CNode3 = VirtualViewGrad(x_view_input1, x_view_input2(x_view_output1), U3)
  // U4 = UpdateState(U3, CNode3)
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(view_cnode);
  CNodePtr view_output = view_cnode;
  AnfNodePtr last_umonad = umonad;
  AnfNodePtr first_virtual_view_grad_node = nullptr;
  AnfNodePtr first_new_umonad = nullptr;
  while (true) {
    const auto &view_output_node_inputs = view_output->inputs();
    auto view_input = view_output_node_inputs[1];
    const auto &ori_view_name = GetCNodePrimitive(view_output)->ToString();
    MS_LOG(DEBUG) << "The name of view operator is: " << ori_view_name;
    auto view_op_node = NewValueNode(ori_view_name);

    // To calculate dout for view_input and view_output, insert origin view cnode inputs:
    // ==> view_output = {kPrimViewOp, view_input, other_view_arg1, other_view_arg2, ..., U_for_view}
    // ==> From: VirtualViewGrad(view_input, view_output, U_for_virtual_view_grad)
    // ==> To: VirtualViewGrad(view_input, view_output, kPrimViewOp, other_view_arg1, other_view_arg2, ...,
    // U_for_virtual_view_grad)
    AnfNodePtrList vvg_node_inputs = {NewValueNode(prim::kPrimVirtualViewGrad), view_input, view_output, view_op_node};
    for (size_t i = kIndex2; i < view_output_node_inputs.size() - 1; ++i) {
      (void)vvg_node_inputs.emplace_back(view_output_node_inputs[i]);
    }
    (void)vvg_node_inputs.emplace_back(last_umonad);
    auto vvg_node = func_graph->NewCNodeInOrder(vvg_node_inputs);
    vvg_node->set_abstract(view_input->abstract());
    auto new_umonad = func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimUpdateState), last_umonad, vvg_node});
    new_umonad->set_abstract(last_umonad->abstract());
    if (first_virtual_view_grad_node == nullptr) {
      first_virtual_view_grad_node = vvg_node;
      first_new_umonad = new_umonad;
    }
    last_umonad = new_umonad;
    auto result = IsCreatedByViewOp(view_input);
    if (result.first == nullptr) {
      break;
    }
    auto view_func_graph = result.first->func_graph();
    const auto &used_func_graphs = func_graph->func_graphs_used_total();
    if (std::find(used_func_graphs.begin(), used_func_graphs.end(), view_func_graph) != used_func_graphs.end()) {
      MS_LOG(INFO) << "Current_func_graph: " << func_graph->ToString()
                   << ", view_input node: " << view_input->DebugString();
      break;
    }

    view_output = result.first;
  }
  // Set edge for original umonad users to last_umonad
  auto updatastate_users = manager->node_users()[umonad];
  for (const auto &node_index : updatastate_users) {
    auto used_node = node_index.first;
    MS_EXCEPTION_IF_NULL(used_node);
    if (used_node == first_virtual_view_grad_node || used_node == first_new_umonad) {
      continue;
    }
    auto used_cnode = used_node->cast<CNodePtr>();
    manager->SetEdge(used_cnode, node_index.second, last_umonad);
  }
}

void InsertVirtualViewGradAfterInplaceCNode(const CNodePtr &inplace_cnode, const CNodePtr &view_output_cnode,
                                            const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(inplace_cnode);
  MS_EXCEPTION_IF_NULL(view_output_cnode);
  MS_EXCEPTION_IF_NULL(func_graph);
  // CNode1 = PrimFunc_Inplace(x, y, inplace_umonad)
  // inplace_next_updatestate = UpdateState(inplace_umonad, CNode1)
  AnfNodePtr inplace_next_updatestate = nullptr;
  AnfNodePtr inplace_umonad = inplace_cnode->inputs().back();
  (void)CheckUMonad(inplace_umonad);
  for (const auto &node_index : manager->node_users()[inplace_cnode]) {
    const auto &used_node = node_index.first;
    MS_EXCEPTION_IF_NULL(used_node);
    if (!IsPrimitiveCNode(used_node, prim::kPrimUpdateState)) {
      continue;
    }
    auto update_cnode = used_node->cast<CNodePtr>();
    if (update_cnode->input(kIndex1) == inplace_umonad && update_cnode->input(kIndex2) == inplace_cnode) {
      inplace_next_updatestate = used_node;
      break;
    }
  }
  MS_EXCEPTION_IF_NULL(inplace_next_updatestate);
  InsertVirtualViewGradAfterInplaceCNodeInner(func_graph, view_output_cnode, inplace_next_updatestate, manager);
}

void VirtualViewGradInsertInner(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(manager);
  for (const auto &node : TopoSort(root->get_return())) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    auto prim = GetCNodePrimitive(cnode);
    if (prim == nullptr || !prim->inplace_prim()) {
      continue;
    }
    CNodePtr view_node;
    bool is_view_output;
    const auto &inplace_indexes = prim->rw_write_input_indexes();
    for (size_t index = 0; index < inplace_indexes.size(); ++index) {
      auto input_node = cnode->input(inplace_indexes[index] + 1);
      std::tie(view_node, is_view_output) = IsCreatedByViewOp(input_node);
      // 1. If view_node not nullptr, do insert VirtualViewGrad
      // 2. If view_node is nullptr, but is_view_output is true, throw exception, not support control flow
      // 3. If view_node is nullptr, and is_view_output is false, inplace input is not a view output, just ignore
      if (view_node == nullptr) {
        if (is_view_output) {
          MS_LOG(EXCEPTION) << "In backpropagation, inplace modification of the output of view operations within "
                               "control flow is not supported.\nThe view operator information is ambiguous, "
                               "and you can avoid this problem by writing the view operator and the inplace operator "
                               "in the same control flow branch.\nThe node is: "
                            << cnode->DebugString() << ".\nPlease check your codes which location is as follows:"
                            << trace::GetDebugInfoStr(cnode->debug_info());
        }
        continue;
      }
      InsertVirtualViewGradAfterInplaceCNode(cnode, view_node, root, manager);
    }
  }
}

bool CheckControlFlow(const PrimitivePtr &prim, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(cnode);
  for (auto index : prim->rw_write_input_indexes()) {
    const auto &input = cnode->input(prim->rw_write_input_indexes()[index] + 1);
    const auto &input_abs = input->abstract();
    if (!input_abs->isa<abstract::AbstractRefTensor>()) {
      MS_LOG(INTERNAL_EXCEPTION) << "The abstract of rw_write input of inplace op is not ref:" << input_abs->ToString()
                                 << ", inplace operation is: " << cnode->DebugString();
    }
    auto input_ref = input_abs->cast<abstract::AbstractRefPtr>();
    if (input_ref->is_view_output()) {
      auto view_op = input_ref->user_data<CNode>(kOriginalViewOp);
      if (view_op == nullptr || view_op->func_graph() != cnode->func_graph()) {
        return true;
      }
    }
  }
  return false;
}

void MarkViewOp(const AnfNodePtr &node, bool *control_flow_scene) {
  if (IsViewNode(node)) {
    const auto &abs = node->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    if (!abs->isa<abstract::AbstractRefTensor>() && !abs->isa<abstract::AbstractTuple>()) {
      MS_LOG(EXCEPTION) << "The abstract of view operation is exception: " << abs->ToString();
    }
    if (abs->isa<abstract::AbstractTuple>()) {
      MS_LOG(EXCEPTION) << "In backpropagation, in-place modification operations are not supported for view operators "
                           "with multiple outputs.\nThe node is: "
                        << node->DebugString() << ".\nPlease check your codes which location is as follows:"
                        << trace::GetDebugInfoStr(node->debug_info());
    }
    auto ref = abs->cast<abstract::AbstractRefPtr>();
    MS_EXCEPTION_IF_NULL(ref);
    auto cnode = node->cast<CNodePtr>();
    ref->set_user_data<CNode>(kOriginalViewOp, cnode);
    return;
  }
  if (!IsInplaceNode(node)) {
    return;
  }
  auto cnode = node->cast<CNodePtr>();
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  // Check if is control flow scene with view inplace.
  if (CheckControlFlow(prim, cnode)) {
    *control_flow_scene = true;
    return;
  }

  const auto &inplace_node_abs = node->abstract();
  // Currently, only consider the case where the inplace operator has only
  // one output and one inplace input.
  if (!inplace_node_abs->isa<abstract::AbstractRefTensor>() || prim->rw_write_input_indexes().size() != 1) {
    MS_LOG(DEBUG) << "The inplace node is: " << node->DebugString();
    return;
  }
  const auto &rw_write_input = cnode->input(prim->rw_write_input_indexes()[0] + 1);
  const auto &rw_write_input_abs = rw_write_input->abstract();
  auto input_ref = rw_write_input_abs->cast<abstract::AbstractRefPtr>();
  MS_EXCEPTION_IF_NULL(input_ref);
  if (input_ref->is_view_output()) {
    auto view_op = input_ref->user_data<CNode>(kOriginalViewOp);
    if (view_op != nullptr) {
      inplace_node_abs->set_user_data<CNode>(kOriginalViewOp, view_op);
      MS_LOG(DEBUG) << "Mark view operator to inplace abstract, view_op: " << view_op->DebugString()
                    << " abstract: " << inplace_node_abs->ToString();
    }
  }
}

void MarkViewOpToAbstract(const FuncGraphPtr &func_graph, bool *control_flow_scene) {
  const auto &nodes = TopoSort(func_graph->get_return());
  for (size_t i = 0; i < nodes.size(); ++i) {
    MarkViewOp(nodes[i], control_flow_scene);
    if (*control_flow_scene) {
      return;
    }
  }
}

std::string GetRealOpName(const std::string &str) {
  const std::string prefix = "PrimFunc_";
  if (str.rfind(prefix, 0) == 0) {
    return str.substr(prefix.length());
  }
  return str;
}
}  // namespace

bool VirtualViewGradInsert(const FuncGraphPtr &root, const opt::OptimizerPtr &opt) {
  MS_EXCEPTION_IF_NULL(root);
  auto manager = opt->manager();
  MS_EXCEPTION_IF_NULL(manager);

  // Insert VirtualViewGrad op for func_graph and sub_graphs
  VirtualViewGradInsertInner(root, manager);
  auto sub_graphs = root->func_graphs_used_total();
  for (const auto &sub_graph : sub_graphs) {
    VirtualViewGradInsertInner(sub_graph, manager);
  }

  return false;
}

bool RemoveRedundantVirtualOps(const FuncGraphPtr &root, const opt::OptimizerPtr &opt) {
  auto manager = opt->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_users = manager->node_users();
  constexpr size_t kMinUsersSizeScene1 = 1;
  constexpr size_t kMinUsersSizeScene2 = 2;
  auto &all_nodes = manager->all_nodes();
  auto nodes = TopoSort(root->get_return(), SuccDeeperSimple);
  for (auto &node : nodes) {
    if (node == nullptr || !all_nodes.contains(node)) {
      continue;
    }
    if (IsPrimitiveCNode(node, prim::kPrimVirtualViewGrad)) {
      // Remove virtualviewgrad op only used by updatestate, replace %2 to U
      // %1 = VirtualViewGrad(%0, x, ..., U)
      // %2 = UpdateState(U, %1)
      auto &cur_node_users = node_users[node];
      if (cur_node_users.size() == kMinUsersSizeScene1 &&
          IsPrimitiveCNode(cur_node_users.front().first, prim::kPrimUpdateState)) {
        const auto &use_node = cur_node_users.front().first;
        MS_LOG(DEBUG) << "Need remove redundant virtual view grad op: " << node->DebugString();
        manager->Replace(use_node, use_node->cast<CNodePtr>()->input(kIndex1));
      }
    } else if (IsVirtualViewCNode(node)) {
      // Remove virtualview which take virtualviewgrad as input
      // and virtualview's original view_output is virtualviewgrad's first input
      // ====> For example:
      // %1 = View(y, U)
      // %2 = UpdateState(U, %1)
      // %3 = Inplace(%1, %2)
      // %4 = UpdateState(%2, %3)
      // %5 = VirtualViewGrad(y, %3, %4)   ==> Depend(y, %4)
      // %6 = UpdateState(%4, %5)
      // %7 = VirtualView(%5, %6)[original_node: %1] ==> Depend(%3, %6)
      // %8 = UpdateState(%6, %7)
      // return Depend(%7, %8)
      auto vv_node = node->cast<CNodePtr>();
      const auto &vv_inputs = vv_node->inputs();
      if (!IsPrimitiveCNode(vv_inputs[kIndex1], prim::kPrimVirtualViewGrad)) {
        continue;
      }
      auto vvg_node = vv_inputs[kIndex1]->cast<CNodePtr>();
      // VirtualViewGrad only used by updatestate and virtualview
      if (node_users[vvg_node].size() > kMinUsersSizeScene2) {
        continue;
      }
      const auto &vvg_inputs = vvg_node->inputs();
      // ==> vvg_view_output: %3 = VirtualViewGrad(y, %2)  ==> get second arg ==> %2
      // ==> vv_view_output:  %4 = VirtualView(%3)[original_node: %1]  ==> get original_node ==> %1
      const auto vvg_view_output = vvg_inputs[kIndex2];
      const auto &vv_view_output = vv_node->user_data<AnfNode>(kIsVirtualViewOp);
      // Temp: Use refkey to check whether virtual_view_ori_view and virtual_view_grad_ori_view are the same (%2 and %1
      // are same node)
      auto refkey1 = GetRefKey(vv_view_output);
      if (!refkey1.empty() && refkey1 == GetRefKey(vvg_view_output)) {
        MS_LOG(DEBUG) << "Need remove redundant virtual view op: " << vv_node->DebugString()
                      << " , and virtual view grad op: " << vvg_node->DebugString();
        // VirtualViewGrad(y, %3, %4)   ==> Depend(y, %4)
        auto depend_for_view_input =
          root->NewCNode({NewValueNode(prim::kPrimDepend), vvg_inputs[kIndex1], CheckUMonad(vvg_inputs.back())});
        manager->Replace(vvg_node, depend_for_view_input);
        // VirtualView(%5, %6)[original_node: %1] ==> Depend(%3, %6)
        auto depend_for_view_output =
          root->NewCNode({NewValueNode(prim::kPrimDepend), vvg_view_output, CheckUMonad(vv_inputs.back())});
        manager->Replace(vv_node, depend_for_view_output);
      }
    }
  }
  return false;
}

// {prim::kPrimVirtualViewGrad, X, Y, ..., U} ==> {prim::kPrimDepend, X, U}
AnfNodePtr VirtualViewGradEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  auto func_graph = node->func_graph();
  if (!IsPrimitiveCNode(node, prim::kPrimVirtualViewGrad) || func_graph == nullptr) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &inputs = cnode->inputs();
  const auto &view_input_node = inputs[kIndex1];
  auto new_depend_node =
    func_graph->NewCNode({NewValueNode(prim::kPrimDepend), view_input_node, CheckUMonad(inputs.back())});
  new_depend_node->set_abstract(view_input_node->abstract());
  return new_depend_node;
}

bool PreprocessForVirtualViewGradInsert(const FuncGraphPtr &root, const opt::OptimizerPtr &opt) {
  // mark view operator to abstract.
  bool control_flow_scene = false;
  MarkViewOpToAbstract(root, &control_flow_scene);
  if (control_flow_scene) {
    return true;
  }
  const auto &fg_used_total = root->func_graphs_used_total();
  for (const auto &fg : fg_used_total) {
    MarkViewOpToAbstract(fg, &control_flow_scene);
    if (control_flow_scene) {
      return true;
    }
  }
  return false;
}

void ConvertViewOpNameInVirtualViewGrad(const FuncGraphPtr &func_graph, const opt::OptimizerPtr &optimizer) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(optimizer);
  const auto &nodes = TopoSort(func_graph->get_return(), SuccDeeperSimple);
  auto manager = optimizer->manager();
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (!IsPrimitiveCNode(nodes[i], prim::kPrimVirtualViewGrad)) {
      continue;
    }
    constexpr size_t view_op_index = 3;
    auto cnode = nodes[i]->cast<CNodePtr>();
    const auto inputs = cnode->inputs();
    if (inputs.size() <= view_op_index) {
      MS_LOG(INTERNAL_EXCEPTION) << "The VirtualViewGrad operator is wrong: " << cnode->DebugString();
    }
    auto view_op_name = cnode->input(view_op_index);
    auto view_value = view_op_name->cast<ValueNodePtr>();
    std::string real_name = GetRealOpName(view_value->value()->ToString());
    auto prim = std::make_shared<Primitive>(real_name);
    auto prim_node = NewValueNode(prim);
    prim_node->set_abstract(std::make_shared<abstract::PrimitiveAbstractClosure>(prim));
    manager->Replace(view_op_name, prim_node);
  }
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
