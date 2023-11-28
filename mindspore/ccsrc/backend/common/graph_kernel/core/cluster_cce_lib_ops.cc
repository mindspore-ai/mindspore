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
#include "backend/common/graph_kernel/core/cluster_cce_lib_ops.h"
#include <string>
#include <unordered_map>
#include <algorithm>
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "backend/common/graph_kernel/core/graph_kernel_callback.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/graph_kernel/core/graph_builder.h"

namespace mindspore::graphkernel {
std::vector<PrimitivePtr> ClusterCceLibOps::GetClusterableOpList() {
  std::vector<OpWithLevel> cce_lib_ops_with_level = {
    {kAllTarget, OpLevel_0, prim::kPrimPagedAttention},
    {kAllTarget, OpLevel_0, prim::kPrimPagedAttentionMask},
    {kAllTarget, OpLevel_0, prim::kPrimReshapeAndCache},
    {kAllTarget, OpLevel_0, prim::kPrimMatMul},
  };
  const auto &flags = GraphKernelFlags::GetInstance();
  return GkUtils::GetValidOps(cce_lib_ops_with_level, flags.fusion_ops_level, flags.enable_cce_lib_ops,
                              flags.enable_cce_lib_ops_only, flags.disable_cce_lib_ops);
}

bool ClusterCceLibOps::IsClusterableOp(const AnfNodePtr &node) {
  if (AnfUtils::IsGraphKernel(node)) {
    // only fuse itself.
    return false;
  }
  if (GkUtils::IsKeepBasicNode(node)) {
    return false;
  }
  bool node_in_oplist = std::any_of(op_list_.begin(), op_list_.end(),
                                    [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
  if (!node_in_oplist) {
    return false;
  }
  return true;
}
void ClusterCceLibOps::CreateFuncGraph(const FuncGraphPtr &func_graph, const std::vector<size_t> &nodes_id) {
  AnfNodePtrList old_nodes;
  (void)std::transform(nodes_id.begin(), nodes_id.end(), std::back_inserter(old_nodes),
                       [this](size_t id) { return this->nodes_[id]; });
  auto new_node = ReplaceNodesWithGraphKernelNode(old_nodes, func_graph, "fusion");
  if (new_node->isa<CNode>()) {
    auto cnode = new_node->cast<CNodePtr>();
    cnode->AddAttr("use_akg_cce", MakeValue(true));
  }
  if (GraphKernelFlags::GetInstance().dump_as_text) {
    DumpClusterInfo(old_nodes, new_node);
  }
}
}  // namespace mindspore::graphkernel
