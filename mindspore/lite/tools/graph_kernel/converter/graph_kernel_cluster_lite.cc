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

#include "tools/graph_kernel/converter/graph_kernel_cluster_lite.h"

#include <utility>
#include <vector>

#include "backend/common/graph_kernel/core/graph_kernel_callback.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "utils/ms_context.h"

namespace mindspore::graphkernel {
std::vector<PrimitivePtr> GraphKernelClusterLite::GetClusterableOpList() {
  std::vector<OpWithLevel> clusterable_ops_with_level = {
    {kAllTarget, OpLevel_0, prim::kPrimAdd},          {kAllTarget, OpLevel_0, prim::kPrimMul},
    {kAllTarget, OpLevel_0, prim::kPrimSub},          {kAllTarget, OpLevel_0, prim::kPrimRealDiv},
    {kAllTarget, OpLevel_0, prim::kPrimLog},          {kAllTarget, OpLevel_0, prim::kPrimExp},
    {kAllTarget, OpLevel_0, prim::kPrimPow},          {kAllTarget, OpLevel_0, prim::kPrimNeg},
    {kAllTarget, OpLevel_0, prim::kPrimRsqrt},        {kAllTarget, OpLevel_0, prim::kPrimSqrt},
    {kAllTarget, OpLevel_0, prim::kPrimSin},          {kAllTarget, OpLevel_0, prim::kPrimTanh},
    {kAllTarget, OpLevel_0, prim::kPrimCos},          {kAllTarget, OpLevel_0, prim::kPrimGreater},
    {kAllTarget, OpLevel_0, prim::kPrimGreaterEqual}, {kAllTarget, OpLevel_0, prim::kPrimLess},
    {kAllTarget, OpLevel_0, prim::kPrimLessEqual},    {kAllTarget, OpLevel_0, prim::kPrimLogicalAnd},
    {kAllTarget, OpLevel_0, prim::kPrimLogicalOr},    {kAllTarget, OpLevel_0, prim::kPrimLogicalNot},
  };
  const auto &flags = GraphKernelFlags::GetInstance();
  return GkUtils::GetValidOps(clusterable_ops_with_level, flags.fusion_ops_level, flags.enable_cluster_ops_only,
                              flags.enable_cluster_ops, flags.disable_cluster_ops);
}

bool GraphKernelClusterLite::IsClusterableOp(const AnfNodePtr &node) {
  if (!GraphKernelCluster::IsClusterableOp(node)) {
    return false;
  }
  // check if the node has dynamic shape
  auto cb = Callback::Instance();
  auto cnode = node->cast<CNodePtr>();
  for (size_t i = 0; i < cnode->size() - 1; i++) {
    if (!cnode->input(i + 1)->isa<Parameter>() && !cnode->input(i + 1)->isa<ValueNode>() &&
        cb->GetInputShape(cnode, i).size() == 0) {
      return false;
    }
  }
  return true;
}
}  // namespace mindspore::graphkernel
