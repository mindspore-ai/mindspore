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
#include "plugin/device/ascend/optimizer/buffer_fusion/eltwise_fusion_pass.h"
#include <vector>
#include "kernel/kernel_fusion.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/core_ops.h"
#include "utils/ms_context.h"
#include "plugin/device/ascend/optimizer/fusion_id_allocator.h"

namespace mindspore {
namespace opt {
void EltwiseFusionPass::MatchEltwise(const CNodePtr &cnode, const session::KernelGraph &kernel_graph,
                                     FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  mindspore::HashSet<AnfNodePtr> record{cnode};
  auto eltwise_input = cnode->input(kIndex1);
  MS_EXCEPTION_IF_NULL(eltwise_input);
  while (CheckEltWiseOrBroadCastNode(kernel_graph, eltwise_input, ELTWISE_INPUT_SIZE)) {
    (void)record.insert(eltwise_input);
    if (record.size() == MAX_ELTWISE_SIZE) {
      break;
    }
    auto input_cnode = eltwise_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(input_cnode);
    eltwise_input = input_cnode->input(kIndex1);
  }
  if (CheckEltWiseOrBroadCastNode(kernel_graph, eltwise_input, ELTWISE_DOUBLE_IN_INPUT_SIZE)) {
    (void)record.insert(eltwise_input);
  }
  if (record.size() < MIN_ELTWISE_SIZE) {
    return;
  }
  candidate_fusion->push_back(record);
  SetRecordFusionId(record);
}

void EltwiseFusionPass::MatchSingleFusionPattern(const session::KernelGraph &kernel_graph,
                                                 FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  std::vector<AnfNodePtr> node_list = TopoSort(kernel_graph.get_return());
  std::reverse(node_list.begin(), node_list.end());
  for (auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (!AnfUtils::IsRealCNodeKernel(node) || fusion_id_allocator->HasFusionIdAttr(node) ||
        common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimReturn)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (AnfAlgo::GetKernelType(cnode) == KernelType::TBE_KERNEL &&
        (AnfAlgo::GetFusionType(cnode) == kernel::kPatternElemWise ||
         AnfAlgo::GetFusionType(cnode) == kernel::kPatternBroadcast) &&
        cnode->inputs().size() == ELTWISE_INPUT_SIZE) {
      MatchEltwise(cnode, kernel_graph, candidate_fusion);
    }
  }
}

bool EltwiseFusionPass::CheckEltWiseOrBroadCastNode(const session::KernelGraph &kernel_graph, const AnfNodePtr &node,
                                                    size_t input_size) {
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>() || !AnfUtils::IsRealCNodeKernel(node) || fusion_id_allocator->HasFusionIdAttr(node)) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  size_t not_updatestate_nums = GetNotUpdateStateUserNums(kernel_graph, node);
  return AnfAlgo::GetKernelType(node) == KernelType::TBE_KERNEL &&
         (AnfAlgo::GetFusionType(node) == kernel::kPatternElemWise ||
          AnfAlgo::GetFusionType(node) == kernel::kPatternBroadcast) &&
         not_updatestate_nums == ELTWISE_USE && cnode->inputs().size() == input_size;
}
}  // namespace opt
}  // namespace mindspore
