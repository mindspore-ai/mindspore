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
#include "backend/optimizer/ascend/buffer_fusion/bnupdate_eltwise_fusion_pass.h"
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
void BnupdateEltwiseFusionPass::MatchBnupdateDoubleOutputEltwise(const CNodePtr &cnode, const AnfNodePtr &eltwise_input,
                                                                 const session::KernelGraph &kernel_graph,
                                                                 FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(eltwise_input);
  auto getitem = eltwise_input->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(getitem);
  auto bnupdate = getitem->input(1);
  MS_EXCEPTION_IF_NULL(bnupdate);
  if (bnupdate->isa<CNode>() && AnfAlgo::GetCNodeName(bnupdate) == kBNTrainingUpdateOpName) {
    std::vector<int64_t> output_used_num(AnfAlgo::GetOutputTensorNum(bnupdate), 0);
    for (auto out_getitem : manager->node_users()[bnupdate]) {
      MS_EXCEPTION_IF_NULL(out_getitem.first);
      auto out_getitem_ptr = out_getitem.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(out_getitem_ptr);
      auto input2 = out_getitem_ptr->input(2);
      auto output_idx = GetValue<int64_t>(GetValueNode(input2));
      output_used_num[output_idx] = SizeToLong(manager->node_users()[out_getitem.first].size());
    }
    AnfAlgo::SetNodeAttr(kAttrOutputUsedNum, MakeValue(output_used_num), bnupdate);
    std::unordered_set<AnfNodePtr> record{cnode, bnupdate};
    candidate_fusion->push_back(record);
    SetRecordFusionId(record);
  }
}

void BnupdateEltwiseFusionPass::MatchSingleFusionPattern(const session::KernelGraph &kernel_graph,
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
        AnfAlgo::GetFusionType(cnode) == kernel::FusionType::ELEMWISE &&
        AnfAlgo::GetOutputTensorNum(cnode) == ELTWISE_DOUBLE_OUTPUT_SIZE) {
      auto eltwise_input = cnode->input(1);
      if (eltwise_input->isa<CNode>() && AnfAlgo::CheckPrimitiveType(eltwise_input, prim::kPrimTupleGetItem)) {
        MatchBnupdateDoubleOutputEltwise(cnode, eltwise_input, kernel_graph, candidate_fusion);
      }
    }
  }
}
}  // namespace opt
}  // namespace mindspore
