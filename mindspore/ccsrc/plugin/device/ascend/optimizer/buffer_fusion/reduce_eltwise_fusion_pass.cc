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
#include "plugin/device/ascend/optimizer/buffer_fusion/reduce_eltwise_fusion_pass.h"
#include <vector>
#include "kernel/kernel_fusion.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/core_ops.h"
#include "utils/ms_context.h"
#include "plugin/device/ascend/optimizer/fusion_id_allocator.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace opt {
void ReduceEltwiseFusionPass::MatchReduceEltwise(const CNodePtr &cnode, const session::KernelGraph &kernel_graph,
                                                 FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  mindspore::HashSet<AnfNodePtr> record{cnode};
  auto eltwise_input = cnode->input(kIndex1);
  while (CheckSingleInEltWiseNode(kernel_graph, eltwise_input)) {
    (void)record.insert(eltwise_input);
    auto input_cnode = eltwise_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(input_cnode);
    eltwise_input = input_cnode->input(kIndex1);
    if (record.size() == MAX_ELTWISE_NUM) {
      break;
    }
  }
  MS_EXCEPTION_IF_NULL(eltwise_input);
  if (!eltwise_input->isa<CNode>() || !AnfUtils::IsRealCNodeKernel(eltwise_input) ||
      fusion_id_allocator->HasFusionIdAttr(eltwise_input)) {
    return;
  }
  if (AnfAlgo::GetKernelType(eltwise_input) == KernelType::TBE_KERNEL &&
      AnfAlgo::GetFusionType(eltwise_input) == kernel::kPatternCommReduce &&
      GetNodeOutputTotalUsedNum(kernel_graph, eltwise_input) == 1) {
    (void)record.insert(eltwise_input);
    auto previous_input_cnode = eltwise_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(previous_input_cnode);
    auto previous_eltwise_input = previous_input_cnode->input(kIndex1);
    auto previous_size = record.size();
    while (CheckSingleInEltWiseNode(kernel_graph, previous_eltwise_input)) {
      (void)record.insert(previous_eltwise_input);
      auto previous_node = previous_eltwise_input->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(previous_node);
      previous_eltwise_input = previous_node->input(kIndex1);
      if (record.size() - previous_size == MAX_ELTWISE_NUM) {
        break;
      }
    }
    candidate_fusion->push_back(record);
    SetRecordFusionId(record);
  }
}

void ReduceEltwiseFusionPass::MatchSingleFusionPattern(const session::KernelGraph &kernel_graph,
                                                       FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  std::vector<AnfNodePtr> node_list = TopoSort(kernel_graph.get_return());
  std::reverse(node_list.begin(), node_list.end());
  for (auto &node : node_list) {
    if (!AnfUtils::IsRealCNodeKernel(node) || fusion_id_allocator->HasFusionIdAttr(node) ||
        common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimReturn)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    // Fusion squaresumv1 and sqrt will get worse performance in bert
    if (AnfAlgo::GetKernelType(cnode) == KernelType::TBE_KERNEL &&
        AnfAlgo::GetFusionType(cnode) == kernel::kPatternElemWise && cnode->inputs().size() == ELTWISE_INPUT_SIZE &&
        common::AnfAlgo::GetCNodeName(cnode) != kCastOpName && common::AnfAlgo::GetCNodeName(cnode) != kSqrtOpName) {
      MatchReduceEltwise(cnode, kernel_graph, candidate_fusion);
    }
  }
}
}  // namespace opt
}  // namespace mindspore
