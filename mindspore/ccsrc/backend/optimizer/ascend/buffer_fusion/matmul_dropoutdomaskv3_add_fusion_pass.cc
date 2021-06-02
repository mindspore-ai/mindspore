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
#include "backend/optimizer/ascend/buffer_fusion/matmul_dropoutdomaskv3_add_fusion_pass.h"
#include <vector>
#include <unordered_set>
#include <memory>
#include <string>
#include "backend/kernel_compiler/kernel_fusion.h"
#include "debug/anf_ir_dump.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "base/core_ops.h"
#include "utils/ms_context.h"
#include "backend/optimizer/common/fusion_id_allocator.h"

namespace mindspore {
namespace opt {
void MatmulDropoutDoMaskV3AddFusionPass::MatchMatmulDropoutDoMaskV3Add(const CNodePtr &cnode,
                                                                       const session::KernelGraph &kernel_graph,
                                                                       FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto add_input = cnode->input(2);
  MS_EXCEPTION_IF_NULL(add_input);
  if (!add_input->isa<CNode>() || !AnfAlgo::CheckPrimitiveType(add_input, prim::kPrimDropoutDoMaskV3)) {
      return;
  }
  auto dropout_do_mask_v3 = add_input->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dropout_do_mask_v3);
  auto matmul = dropout_do_mask_v3->input(1);
  MS_EXCEPTION_IF_NULL(matmul);
  if (!matmul->isa<CNode>() || !AnfAlgo::CheckPrimitiveType(matmul, prim::kPrimMatMul)) {
    return;
  }
  std::vector<int64_t> dropout_output_used_num{SizeToLong(manager->node_users()[dropout_do_mask_v3].size())};
  AnfAlgo::SetNodeAttr(kAttrOutputUsedNum, MakeValue(dropout_output_used_num), dropout_do_mask_v3);
  std::vector<int64_t> matmul_output_used_num{SizeToLong(manager->node_users()[matmul].size())};
  AnfAlgo::SetNodeAttr(kAttrOutputUsedNum, MakeValue(matmul_output_used_num), matmul);
  std::unordered_set<AnfNodePtr> record{cnode, dropout_do_mask_v3, matmul};
  candidate_fusion->push_back(record);
  SetRecordFusionId(record);
}

void MatmulDropoutDoMaskV3AddFusionPass::MatchSingleFusionPattern(const session::KernelGraph &kernel_graph,
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

    if (AnfAlgo::GetCNodeName(cnode) == kAddOpName) {
      MatchMatmulDropoutDoMaskV3Add(cnode, kernel_graph, candidate_fusion);
    }
  }
}
}  // namespace opt
}  // namespace mindspore
