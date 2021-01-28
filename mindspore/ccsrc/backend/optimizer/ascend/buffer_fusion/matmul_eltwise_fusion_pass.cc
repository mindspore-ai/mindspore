/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/ascend/buffer_fusion/matmul_eltwise_fusion_pass.h"
#include <vector>
#include <unordered_set>
#include <memory>
#include "backend/kernel_compiler/kernel_fusion.h"
#include "debug/anf_ir_dump.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "base/core_ops.h"
#include "utils/ms_context.h"
#include "backend/optimizer/common/fusion_id_allocator.h"

namespace mindspore {
namespace opt {
void MatmulEltwiseFusionPass::MatchMatmulEltwise(const CNodePtr &cnode, const AnfNodePtr &relu_input,
                                                 const session::KernelGraph &kernel_graph,
                                                 FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (fusion_id_allocator->HasFusionIdAttr(relu_input)) {
    return;
  }
  std::vector<int64_t> output_used_num{SizeToLong(manager->node_users()[relu_input].size())};
  AnfAlgo::SetNodeAttr(kAttrOutputUsedNum, MakeValue(output_used_num), relu_input);
  std::unordered_set<AnfNodePtr> record{cnode, relu_input};
  candidate_fusion->push_back(record);
  SetRecordFusionId(record);
}

void MatmulEltwiseFusionPass::MatchSingleFusionPattern(const session::KernelGraph &kernel_graph,
                                                       FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  std::vector<AnfNodePtr> node_list = TopoSort(kernel_graph.get_return());
  for (auto &node : node_list) {
    if (!AnfAlgo::IsRealCNodeKernel(node) || fusion_id_allocator->HasFusionIdAttr(node) ||
        AnfAlgo::CheckPrimitiveType(node, prim::kPrimReturn)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (AnfAlgo::GetKernelType(cnode) == KernelType::TBE_KERNEL &&
        AnfAlgo::GetFusionType(cnode) == kernel::FusionType::ELEMWISE) {
      auto eltwise_input = cnode->input(1);
      MS_EXCEPTION_IF_NULL(eltwise_input);
      if (eltwise_input->isa<CNode>() && AnfAlgo::CheckPrimitiveType(eltwise_input, prim::kPrimMatMul)) {
        MatchMatmulEltwise(cnode, eltwise_input, kernel_graph, candidate_fusion);
      }
    }
  }
}
}  // namespace opt
}  // namespace mindspore
