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
#include "plugin/device/ascend/optimizer/buffer_fusion/bnupdate_eltwise_eltwise_fusion_pass.h"
#include "utils/hash_set.h"
#include "kernel/kernel_fusion.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/core_ops.h"
#include "utils/ms_context.h"
#include "backend/common/optimizer/fusion_id_allocator.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
void BnupdateEltwiseEltwiseFusionPass::MatchBnupdateAddRelu(const CNodePtr &cnode, const AnfNodePtr &relu_input,
                                                            const session::KernelGraph &kernel_graph,
                                                            FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  MS_EXCEPTION_IF_NULL(relu_input);
  auto add = relu_input->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(add);
  if (common::AnfAlgo::GetInputTensorNum(add) != (ELTWISE_DOUBLE_IN_INPUT_SIZE - 1)) {
    return;
  }
  auto tuple_getitem = add->input(kIndex1);
  MS_EXCEPTION_IF_NULL(tuple_getitem);
  if (tuple_getitem->isa<CNode>() && common::AnfAlgo::GetCNodeName(tuple_getitem) == prim::kPrimTupleGetItem->name()) {
    auto getitem = tuple_getitem->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(getitem);
    auto bnupdate = getitem->input(kRealInputNodeIndexInTupleGetItem);
    MS_EXCEPTION_IF_NULL(bnupdate);
    if (bnupdate->isa<CNode>() && common::AnfAlgo::GetCNodeName(bnupdate) == kBNTrainingUpdateOpName &&
        GetNodeOutputTotalUsedNum(kernel_graph, bnupdate) == kBNTrainingUpdateOutputUsedTotalNum) {
      mindspore::HashSet<AnfNodePtr> record{cnode, relu_input, bnupdate};
      candidate_fusion->push_back(record);
      SetRecordFusionId(record);
    }
  }
}

void BnupdateEltwiseEltwiseFusionPass::MatchSingleFusionPattern(const session::KernelGraph &kernel_graph,
                                                                FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  const auto &node_list = TopoSort(kernel_graph.get_return());
  for (auto &node : node_list) {
    if (!AnfUtils::IsRealCNodeKernel(node) || fusion_id_allocator->HasFusionIdAttr(node) ||
        common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimReturn)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (AnfAlgo::GetKernelType(cnode) == KernelType::TBE_KERNEL &&
        AnfAlgo::GetFusionType(cnode) == kernel::kPatternElemWise &&
        AnfAlgo::GetOutputTensorNum(cnode) == ELTWISE_DOUBLE_OUTPUT_SIZE &&
        common::AnfAlgo::GetInputTensorNum(cnode) == (ELTWISE_INPUT_SIZE - 1)) {
      auto eltwise_input = cnode->input(kIndex1);
      MS_EXCEPTION_IF_NULL(eltwise_input);
      if (eltwise_input->isa<CNode>() && common::AnfAlgo::CheckPrimitiveType(eltwise_input, prim::kPrimAdd)) {
        MatchBnupdateAddRelu(cnode, eltwise_input, kernel_graph, candidate_fusion);
      }
    }
  }
}
}  // namespace opt
}  // namespace mindspore
