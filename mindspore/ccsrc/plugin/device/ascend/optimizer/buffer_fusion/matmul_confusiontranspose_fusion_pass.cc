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
#include "plugin/device/ascend/optimizer/buffer_fusion/matmul_confusiontranspose_fusion_pass.h"
#include "kernel/kernel_fusion.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/core_ops.h"
#include "utils/ms_context.h"
#include "backend/common/optimizer/fusion_id_allocator.h"
#include "plugin/device/ascend/hal/common/platform_info_util.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kAttrTransposeX2 = "transpose_x2";
}  // namespace

void MatmulConfusionTranposeFusionPass::MatchMatmulConfusionTranpose(const CNodePtr &cnode,
                                                                     FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  auto matmul = cnode->input(kIndex1);
  MS_EXCEPTION_IF_NULL(matmul);
  if (matmul->isa<CNode>() && (common::AnfAlgo::CheckPrimitiveType(matmul, prim::kPrimMatMul) ||
                               common::AnfAlgo::CheckPrimitiveType(matmul, prim::kPrimBatchMatMul))) {
    if (common::AnfAlgo::GetBooleanAttr(matmul, kAttrTransposeX2) == true) {
      return;
    }
    mindspore::HashSet<AnfNodePtr> record{cnode, matmul};
    candidate_fusion->push_back(record);
    SetRecordFusionId(record);
  }
}

void MatmulConfusionTranposeFusionPass::MatchSingleFusionPattern(const session::KernelGraph &kernel_graph,
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

    if (common::AnfAlgo::GetCNodeName(cnode) == kConfusionTransposeDOpName) {
      MatchMatmulConfusionTranpose(cnode, candidate_fusion);
    }
  }
}
}  // namespace opt
}  // namespace mindspore
