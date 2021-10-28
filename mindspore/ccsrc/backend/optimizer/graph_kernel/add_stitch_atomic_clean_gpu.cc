/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/optimizer/graph_kernel/add_stitch_atomic_clean_gpu.h"

#include <algorithm>
#include <string>
#include "base/core_ops.h"
#include "ir/tensor.h"
#include "utils/utils.h"
#include "utils/log_adapter.h"
#include "backend/kernel_compiler/kernel.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/session/kernel_graph.h"

namespace mindspore {
namespace opt {
CNodePtr StitchAtomicCleanInsertter::CreateInplaceAssignNode(const FuncGraphPtr &sub_graph,
                                                             const AnfNodePtr &new_parameter) const {
  // add inplaceassign
  AnfNodePtr out_node = atomic_add_node_;  // Use result data itself, and set attr "fake_out" true.
  auto inplace_assign_node =
    CreateCNode({NewValueNode(prim::kPrimInplaceAssign), new_parameter, atomic_add_node_, out_node}, sub_graph,
                {.format = GetFormat(out_node), .shape = GetShape(out_node), .type = GetType(out_node)});
  SetNodeAttrSafely("fake_output", MakeValue(true), inplace_assign_node);
  AnfAlgo::EraseNodeAttr(kAttrStitch, atomic_add_node_);
  SetNodeAttrSafely(kAttrStitch, MakeValue("common"), inplace_assign_node);
  return inplace_assign_node;
}

void StitchAtomicCleanInsertter::ProcessOriginCNode(const AnfNodePtr &composite_node, const AnfNodePtr &new_input) {
  auto sub_graph = AnfAlgo::GetCNodeFuncGraphPtr(composite_node);
  auto mng_sub = sub_graph->manager();
  if (mng_sub == nullptr) {
    mng_sub = Manage(sub_graph, false);
    sub_graph->set_manager(mng_sub);
  }

  // add input
  auto inputs = composite_node->cast<CNodePtr>()->inputs();
  inputs.push_back(new_input);
  composite_node->cast<CNodePtr>()->set_inputs(inputs);

  // add parameter
  auto parameter = sub_graph->add_parameter();
  parameter->set_abstract(new_input->abstract());
  parameter->set_kernel_info(new_input->kernel_info_ptr());

  auto inplace_assign = CreateInplaceAssignNode(sub_graph, parameter);

  // Replace atomic ReduceSum's user with atomic clean output, and add depend op after inplaceassign to avoid
  // elimination.
  std::vector<std::pair<AnfNodePtr, int>> reduce_user_nodes = FindInnerCNodeUsers(stitch_node_, atomic_add_node_);
  bool connected = false;
  for (const auto &[user_node, index] : reduce_user_nodes) {
    auto user_cnode = user_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(user_cnode);
    user_cnode->set_input(static_cast<size_t>(index), parameter);
    if (!connected) {
      std::vector<std::pair<AnfNodePtr, int>> user_user = FindInnerCNodeUsers(stitch_node_, user_cnode);
      if (!user_user.empty()) {
        auto pair = user_user[0];
        AddDepend(sub_graph, user_cnode, inplace_assign, pair.first, pair.second);
      }
      connected = true;
    }
    CorrectKernelBuildInfo(composite_node, new_input, false);
  }

  auto old_graph_name = GetValue<std::string>(sub_graph->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL));
  auto new_graph_name = ExtractGraphKernelName(TopoSort(sub_graph->get_return()), "", "atomic_add");
  sub_graph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(new_graph_name));
  MS_LOG(INFO) << "Convert " << old_graph_name << " to atomic add graph " << new_graph_name;
}

std::vector<std::pair<AnfNodePtr, int>> StitchAtomicCleanInsertter::FindInnerCNodeUsers(const AnfNodePtr &inner_node,
                                                                                        const CNodePtr &target) const {
  auto node = inner_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(node);
  auto sub_graph = AnfAlgo::GetCNodeFuncGraphPtr(node);
  auto mng_sub = sub_graph->manager();
  if (mng_sub == nullptr) {
    mng_sub = Manage(sub_graph, false);
    sub_graph->set_manager(mng_sub);
  }
  std::vector<std::pair<AnfNodePtr, int>> inner_user_nodes;
  auto users = mng_sub->node_users()[target];
  std::transform(users.cbegin(), users.cend(), std::back_inserter(inner_user_nodes),
                 [](const std::pair<AnfNodePtr, int> &pair) { return pair; });
  return inner_user_nodes;
}

bool StitchAtomicCleanInsertter::IsStitchWithAtomic(const AnfNodePtr &anf_node) {
  if (!AnfAlgo::IsGraphKernel(anf_node)) return false;
  auto node = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(node);
  auto sub_graph = AnfAlgo::GetCNodeFuncGraphPtr(node);
  AnfNodePtrList kernel_nodes;
  kernel::GetValidKernelNodes(sub_graph, &kernel_nodes);
  for (auto &n : kernel_nodes) {
    if (AnfAlgo::HasNodeAttr(kAttrStitch, n->cast<CNodePtr>()) &&
        AnfAlgo::GetNodeAttr<std::string>(n, kAttrStitch) == "atomic" && IsPrimitiveCNode(n, prim::kPrimReduceSum)) {
      MS_LOG(INFO) << "GOT STITCH WITH ATOMIC!!!";
      atomic_add_node_ = n->cast<CNodePtr>();
      stitch_node_ = anf_node;
      return true;
    }
  }
  return false;
}

bool StitchAtomicCleanInsertter::Run(const FuncGraphPtr &func_graph) {
  auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(func_graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto mng = kernel_graph->manager();
  if (mng == nullptr) {
    mng = Manage(kernel_graph, true);
    kernel_graph->set_manager(mng);
  }

  bool changed = false;
  auto topo_nodes = TopoSort(kernel_graph->get_return());
  for (const auto &node : topo_nodes) {
    // if stitch attr exists, add atomic clean op depends on the attr
    if (IsStitchWithAtomic(node)) {
      InsertAtomicClean(kernel_graph, node, mng);
      changed = true;
    }
  }

  if (changed) {
    UpdateMng(mng, func_graph);
  }

  return changed;
}
}  // namespace opt
}  // namespace mindspore
