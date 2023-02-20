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
#include "plugin/device/ascend/optimizer/buffer_fusion/batchmatmul_dropoutdomaskv3_fusion_pass.h"
#include <memory>
#include <string>
#include <vector>
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/core_ops.h"
#include "backend/common/optimizer/fusion_id_allocator.h"
#include "plugin/device/ascend/optimizer/platform.h"

namespace mindspore {
namespace opt {
void BatchMatmulDropoutDoMaskV3FusionPass::MatchBatchMatmulDropoutDoMaskV3(const CNodePtr &cnode,
                                                                           FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  auto batch_matmul = cnode->input(1);
  MS_EXCEPTION_IF_NULL(batch_matmul);
  if (batch_matmul->isa<CNode>() && common::AnfAlgo::CheckPrimitiveType(batch_matmul, prim::kPrimBatchMatMul)) {
    mindspore::HashSet<AnfNodePtr> record{cnode, batch_matmul};
    candidate_fusion->push_back(record);
    SetRecordFusionId(record);
  }
}

void BatchMatmulDropoutDoMaskV3FusionPass::MatchSingleFusionPattern(const session::KernelGraph &kernel_graph,
                                                                    FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  MS_CHECK_CUBE_VECTOR_SPLIT();
  const auto &node_list = TopoSort(kernel_graph.get_return());
  for (auto &node : node_list) {
    if (!AnfUtils::IsRealCNodeKernel(node) || fusion_id_allocator->HasFusionIdAttr(node) ||
        common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimReturn)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);

    if (common::AnfAlgo::GetCNodeName(cnode) == kDropOutDoMaskV3DOpName) {
      MatchBatchMatmulDropoutDoMaskV3(cnode, candidate_fusion);
    }
  }
}
}  // namespace opt
}  // namespace mindspore
