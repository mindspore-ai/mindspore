/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "pre_activate/ascend/buffer_fusion/conv_bnreduce_fusion_pass.h"

#include <vector>
#include <unordered_set>
#include <memory>
#include <string>
#include "kernel/kernel_fusion.h"
#include "debug/anf_ir_dump.h"
#include "session/anf_runtime_algorithm.h"
#include "operator/ops.h"
#include "utils/context/ms_context.h"
#include "pre_activate/common/fusion_id_allocator.h"

namespace mindspore {
namespace opt {
void ConvBnReduceFusionPass::MatchConvBnreduce(const CNodePtr &cnode, const session::KernelGraph &kernel_graph,
                                               FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto conv = cnode->input(1);
  MS_EXCEPTION_IF_NULL(conv);
  if (conv->isa<CNode>() && AnfAlgo::GetCNodeName(conv) == prim::kPrimConv2D->name()) {
    std::vector<int> output_used_num{SizeToInt(manager->node_users()[conv].size())};
    AnfAlgo::SetNodeAttr(kAttrOutputUsedNum, MakeValue(output_used_num), conv);
    std::unordered_set<AnfNodePtr> record{cnode, conv};
    candidate_fusion->push_back(record);
    SetRecordFusionId(record);
  }
}

void ConvBnReduceFusionPass::MatchSingleFusionPattern(const session::KernelGraph &kernel_graph,
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
    if (AnfAlgo::GetCNodeName(cnode) == kBNTrainingReduceOpName) {
      MatchConvBnreduce(cnode, kernel_graph, candidate_fusion);
    }
  }
}
}  // namespace opt
}  // namespace mindspore
