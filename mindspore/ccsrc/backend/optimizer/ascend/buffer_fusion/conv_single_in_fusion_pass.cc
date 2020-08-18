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
#include "backend/optimizer/ascend/buffer_fusion/conv_single_in_fusion_pass.h"
#include <vector>
#include <unordered_set>
#include <memory>
#include "backend/kernel_compiler/kernel_fusion.h"
#include "debug/anf_ir_dump.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "frontend/operator/ops.h"
#include "utils/ms_context.h"
#include "backend/optimizer/common/fusion_id_allocator.h"

namespace mindspore {
namespace opt {
void ConvSingleInFusionPass::MatchConvSingleInEltwise(const CNodePtr &cnode, const session::KernelGraph &kernel_graph,
                                                      FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::unordered_set<AnfNodePtr> record{cnode};
  auto eltwise_input = cnode->input(1);
  while (CheckEltWiseNode(manager.get(), eltwise_input)) {
    (void)record.insert(eltwise_input);
    auto input_cnode = eltwise_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(input_cnode);
    eltwise_input = input_cnode->input(1);
    if (record.size() == MAX_ELTWISE_NUM) {
      break;
    }
  }
  MS_EXCEPTION_IF_NULL(eltwise_input);
  if (!eltwise_input->isa<CNode>() || !AnfAlgo::IsRealCNodeKernel(eltwise_input) ||
      fusion_id_allocator->HasFusionIdAttr(eltwise_input)) {
    return;
  }
  if (AnfAlgo::GetKernelType(eltwise_input) == KernelType::TBE_KERNEL &&
      AnfAlgo::GetFusionType(eltwise_input) == kernel::FusionType::CONVLUTION) {
    (void)record.insert(eltwise_input);
    candidate_fusion->push_back(record);
    SetRecordFusionId(record);
  }
}

void ConvSingleInFusionPass::MatchSingleFusionPattern(const session::KernelGraph &kernel_graph,
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
        AnfAlgo::GetFusionType(cnode) == kernel::FusionType::ELEMWISE && cnode->inputs().size() == ELTWISE_INPUT_SIZE) {
      MatchConvSingleInEltwise(cnode, kernel_graph, candidate_fusion);
    }
  }
}
}  // namespace opt
}  // namespace mindspore
