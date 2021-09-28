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
#include "backend/optimizer/ascend/buffer_fusion/stridedread_conv_stridedwrite_fusion_pass.h"
#include "backend/kernel_compiler/kernel_fusion.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "base/core_ops.h"
#include "utils/ms_context.h"
#include "backend/optimizer/common/fusion_id_allocator.h"

namespace mindspore {
namespace opt {
void StridedReadConvStridedWriteFusionPass::MatchStridedReadConvStridedWrite(const CNodePtr &cnode,
                                                                             const session::KernelGraph &kernel_graph,
                                                                             FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  std::unordered_set<AnfNodePtr> record{cnode};
  auto write_input = cnode->input(kIndex1);
  if (CheckEltWiseNode(kernel_graph, write_input)) {
    (void)record.insert(write_input);
    auto input_cnode = write_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(input_cnode);
    write_input = input_cnode->input(kIndex1);
  }
  MS_EXCEPTION_IF_NULL(write_input);
  if (!write_input->isa<CNode>() || !AnfAlgo::IsRealCNodeKernel(write_input) ||
      fusion_id_allocator->HasFusionIdAttr(write_input)) {
    return;
  }
  auto conv_cnode = write_input->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(conv_cnode);
  if (AnfAlgo::GetKernelType(conv_cnode) == KernelType::TBE_KERNEL &&
      AnfAlgo::GetFusionType(conv_cnode) == kernel::FusionType::CONV &&
      conv_cnode->inputs().size() >= CONV_DOUBLE_IN_INPUT_SIZE &&
      conv_cnode->inputs().size() <= CONV_QUART_IN_INPUT_SIZE) {
    (void)record.insert(write_input);
    auto conv_input = conv_cnode->input(kIndex1);
    MS_EXCEPTION_IF_NULL(conv_input);
    if (!conv_input->isa<CNode>() || !AnfAlgo::IsRealCNodeKernel(conv_input) ||
        fusion_id_allocator->HasFusionIdAttr(conv_input)) {
      return;
    }
    if (AnfAlgo::GetCNodeName(conv_input) == kStridedReadOpName) {
      (void)record.insert(conv_input);
      candidate_fusion->push_back(record);
      SetRecordFusionId(record);
    }
  }
}

void StridedReadConvStridedWriteFusionPass::MatchSingleFusionPattern(const session::KernelGraph &kernel_graph,
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
    if (AnfAlgo::GetCNodeName(cnode) == kStridedWriteOpName) {
      MatchStridedReadConvStridedWrite(cnode, kernel_graph, candidate_fusion);
    }
  }
}
}  // namespace opt
}  // namespace mindspore
