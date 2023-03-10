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
#include "plugin/device/ascend/optimizer/buffer_fusion/matmul_eltwise_fusion_pass.h"
#include "kernel/kernel_fusion.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/core_ops.h"
#include "plugin/device/ascend/optimizer/fusion_id_allocator.h"
#include "plugin/device/ascend/hal/common/platform_info_util.h"

namespace mindspore {
namespace opt {
void MatmulEltwiseFusionPass::MatchMatmulEltwise(const CNodePtr &cnode, const AnfNodePtr &relu_input,
                                                 FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  if (fusion_id_allocator->HasFusionIdAttr(relu_input)) {
    return;
  }
  mindspore::HashSet<AnfNodePtr> record{cnode, relu_input};
  candidate_fusion->push_back(record);
  SetRecordFusionId(record);
}

void MatmulEltwiseFusionPass::MatchSingleFusionPattern(const session::KernelGraph &kernel_graph,
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
    if (AnfAlgo::GetKernelType(cnode) == KernelType::TBE_KERNEL &&
        (AnfAlgo::GetFusionType(cnode) == kernel::kPatternElemWise ||
         AnfAlgo::GetFusionType(cnode) == kernel::kPatternBroadcast) &&
        AnfAlgo::GetOutputTensorNum(cnode) == ELTWISE_SINGLE_OUTPUT_SIZE) {
      auto eltwise_input = cnode->input(kIndex1);
      MS_EXCEPTION_IF_NULL(eltwise_input);
      if (eltwise_input->isa<CNode>() && common::AnfAlgo::CheckPrimitiveType(eltwise_input, prim::kPrimMatMul)) {
        MatchMatmulEltwise(cnode, eltwise_input, candidate_fusion);
      }
    }
  }
}
}  // namespace opt
}  // namespace mindspore
