/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "common/graph_kernel/floatstatus_addn_fusion.h"

#include <vector>
#include <string>
#include <unordered_set>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/common_utils.h"
#include "common/graph_kernel/graph_kernel_flags.h"
#include "common/graph_kernel/graph_kernel_helper.h"
#include "common/graph_kernel/adapter/expander.h"
#include "common/graph_kernel/core/graph_builder.h"
#include "common/graph_kernel/core/graph_kernel_utils.h"

namespace mindspore::graphkernel {
namespace {
constexpr auto kNameAddN = "AddN";
constexpr auto kNameFloatStatus = "FloatStatus";

bool CanConvert() {
  const auto &flags = GraphKernelFlags::GetInstance();
  if (!flags.enable_expand_ops_only.empty()) {
    std::unordered_set<std::string> all_ops(flags.enable_expand_ops_only.begin(), flags.enable_expand_ops_only.end());
    return all_ops.find(kNameAddN) != all_ops.end() && all_ops.find(kNameFloatStatus) != all_ops.end();
  }
  if (!flags.disable_expand_ops.empty()) {
    auto find_target = std::find_if(flags.disable_expand_ops.begin(), flags.disable_expand_ops.end(),
                                    [](const std::string &op) { return op == kNameAddN || op == kNameFloatStatus; });
    return find_target == flags.disable_expand_ops.end();
  }
  return true;
}

CleanZeroUserInfo SubGraphSignleOutput(const AnfNodePtr &anf_node) {
  CleanZeroUserInfo new_op_info;
  auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(anf_node);
  auto output = sub_graph->output();
  if (IsPrimitiveCNode(output, kPrimElemAny)) {
    new_op_info.op_node = output->cast<CNodePtr>();
  }
  return new_op_info;
}

CNodePtr InsertLoad(const FuncGraphPtr &main_graph, const CNodePtr &broadcast_to_node, const CNodePtr &rely) {
  AnfNodePtrList load_inputs = {NewValueNode(prim::kPrimLoad), broadcast_to_node, rely};
  auto load_node = main_graph->NewCNode(load_inputs);
  load_node->set_abstract(broadcast_to_node->abstract());
  main_graph->AddNode(load_node);
  return load_node;
}
}  // namespace

void FloatStatusAddNFusion::ProcessFloatStatusAddN(const FuncGraphPtr &main_graph, const CNodePtr &addn,
                                                   const FuncGraphManagerPtr &mng) {
  // Expand floatstatus to subgraph
  for (size_t i = 1; i < addn->inputs().size(); i++) {
    auto floatstatus = addn->input(i)->cast<CNodePtr>();
    auto expand_fg = GetCNodeFuncGraph(graphkernel::GetExpander(floatstatus, false)->Run(floatstatus));
    MS_EXCEPTION_IF_NULL(expand_fg);
    expand_fg->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(AnfUtils::GetCNodeName(floatstatus)));
    std::vector<AnfNodePtr> inputs(floatstatus->inputs().begin() + 1, floatstatus->inputs().end());
    auto graph_kernel_node = CreateNewFuseCNode(main_graph, expand_fg, inputs);
    (void)mng->Replace(floatstatus, graph_kernel_node);
  }

  // Create broadcast node.
  CleanZeroUserInfo op_info = SubGraphSignleOutput(addn->input(1));
  auto out_type = GetType(op_info.op_node)->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(out_type);
  auto broadcast_to_node = CreateCleanCompositeNode(op_info, main_graph, out_type->element()->type_id());

  // Insert extra input(broadcast node output) to composite node, and make elemany inplace-assign to it.
  for (size_t i = 1; i < addn->inputs().size(); i++) {
    op_info = SubGraphSignleOutput(addn->input(i));
    ProcessOriginCNode(addn->input(i), {{op_info, broadcast_to_node}});
  }

  // Insert UpdateState
  AnfNodePtrList updatestate_inputs(addn->inputs().begin() + 1, addn->inputs().end());
  auto updatestate_node = InsertUpdateState(main_graph, updatestate_inputs);

  // Insert Load
  auto load_node = InsertLoad(main_graph, broadcast_to_node, updatestate_node);

  // Remove AddN
  (void)mng->Replace(addn, load_node);
}

bool FloatStatusAddNFusion::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto changed = false;
  if (!CanConvert()) return changed;
  auto nodes = TopoSort(func_graph->get_return());
  for (auto node : nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimAddN)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    bool pattern_match =
      std::all_of(cnode->inputs().begin() + 1, cnode->inputs().end(),
                  [](const AnfNodePtr &anf_node) { return IsPrimitiveCNode(anf_node, kPrimFloatStatus); });
    if (!pattern_match) continue;
    ProcessFloatStatusAddN(func_graph, cnode, mng);
    changed = true;
  }

  if (changed) {
    GkUtils::UpdateFuncGraphManager(mng, func_graph);
  }

  return changed;
}
}  // namespace mindspore::graphkernel
