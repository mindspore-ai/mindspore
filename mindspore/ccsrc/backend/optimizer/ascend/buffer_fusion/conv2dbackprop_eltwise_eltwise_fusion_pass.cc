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
#include "backend/optimizer/ascend/buffer_fusion/conv2dbackprop_eltwise_eltwise_fusion_pass.h"
#include "backend/kernel_compiler/kernel_fusion.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "base/core_ops.h"
#include "utils/ms_context.h"
#include "backend/optimizer/common/fusion_id_allocator.h"

namespace mindspore {
namespace opt {
void Conv2DBackpropEltwiseEltwiseFusionPass::MatchConv2DBackpropInputEltwiseEltwise(
  const CNodePtr &cnode, const session::KernelGraph &kernel_graph, FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  std::unordered_set<AnfNodePtr> record{cnode};
  auto eltwise_input = cnode->input(kIndex1);
  MS_EXCEPTION_IF_NULL(eltwise_input);
  if (CheckDoubleInEltWiseNode(kernel_graph, eltwise_input)) {
    (void)record.insert(eltwise_input);
  } else {
    return;
  }
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto input_cnode = eltwise_input->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(input_cnode);
  auto double_in_eltwise_input = input_cnode->input(kIndex2);
  MS_EXCEPTION_IF_NULL(double_in_eltwise_input);
  if (!double_in_eltwise_input->isa<CNode>() || !AnfAlgo::IsRealCNodeKernel(double_in_eltwise_input)) {
    return;
  }
  if (AnfAlgo::CheckPrimitiveType(double_in_eltwise_input, prim::kPrimConv2DBackpropInput) &&
      !fusion_id_allocator->HasFusionIdAttr(double_in_eltwise_input)) {
    (void)record.insert(double_in_eltwise_input);
    candidate_fusion->push_back(record);
    SetRecordFusionId(record);
  } else {
    auto double_in_eltwise_input_1 = input_cnode->input(kIndex1);
    MS_EXCEPTION_IF_NULL(double_in_eltwise_input_1);
    if (!double_in_eltwise_input_1->isa<CNode>() || !AnfAlgo::IsRealCNodeKernel(double_in_eltwise_input_1)) {
      return;
    }
    if (AnfAlgo::CheckPrimitiveType(double_in_eltwise_input_1, prim::kPrimConv2DBackpropInput) &&
        !fusion_id_allocator->HasFusionIdAttr(double_in_eltwise_input_1)) {
      (void)record.insert(double_in_eltwise_input_1);
      candidate_fusion->push_back(record);
      SetRecordFusionId(record);
    }
  }
}

void Conv2DBackpropEltwiseEltwiseFusionPass::MatchSingleFusionPattern(const session::KernelGraph &kernel_graph,
                                                                      FusedNodeRecord *candidate_fusion) {
  if (!LicManager::GetInstance().GetPassSwitch(OptPassEnum::Conv2DBackpropEltwiseFusionPass)) {
    return;
  }
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
        (cnode->inputs().size() == ELTWISE_INPUT_SIZE || cnode->inputs().size() == ELTWISE_DOUBLE_IN_INPUT_SIZE)) {
      MatchConv2DBackpropInputEltwiseEltwise(cnode, kernel_graph, candidate_fusion);
    }
  }
}
}  // namespace opt
}  // namespace mindspore
