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
#include <unordered_set>
#include "plugin/device/ascend/optimizer/buffer_fusion/conv2dbackprop_eltwise_eltwise_fusion_pass.h"
#include "kernel/kernel_fusion.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/core_ops.h"
#include "utils/ms_context.h"
#include "backend/common/optimizer/fusion_id_allocator.h"
#include "plugin/device/ascend/hal/common/platform_info_util.h"

namespace mindspore {
namespace opt {
void Conv2DBackpropEltwiseEltwiseFusionPass::MatchConv2DBackpropInputEltwiseEltwise(
  const CNodePtr &cnode, const session::KernelGraph &kernel_graph, FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  mindspore::HashSet<AnfNodePtr> record{cnode};
  auto eltwise_input = cnode->input(kIndex1);
  MS_EXCEPTION_IF_NULL(eltwise_input);
  const std::unordered_set<std::string> support_node_names{kAddNOpName, kAddOpName};
  if (CheckDoubleInEltWiseNode(kernel_graph, eltwise_input, {kernel::kPatternElemWise, kernel::kPatternBroadcast}) &&
      support_node_names.find(common::AnfAlgo::GetCNodeName(eltwise_input)) != support_node_names.cend()) {
    (void)record.insert(eltwise_input);
  } else {
    return;
  }
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto input_cnode = eltwise_input->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(input_cnode);
  std::vector candidate_cb_nodes{input_cnode->input(kIndex2), input_cnode->input(kIndex1)};
  for (const auto &cb_node : candidate_cb_nodes) {
    MS_EXCEPTION_IF_NULL(cb_node);
    if (!cb_node->isa<CNode>() || !AnfUtils::IsRealCNodeKernel(cb_node)) {
      return;
    }
    // skip when output0 of conv2dbackpropinputd is fp32, it may be slower
    const std::unordered_set<TypeId> fp32_types{TypeId::kNumberTypeFloat32, TypeId::kNumberTypeFloat};
    if (common::AnfAlgo::CheckPrimitiveType(cb_node, prim::kPrimConv2DBackpropInputD) &&
        fp32_types.find(AnfAlgo::GetOutputDeviceDataType(cb_node, kIndex0)) == fp32_types.cend() &&
        !fusion_id_allocator->HasFusionIdAttr(cb_node)) {
      (void)record.insert(cb_node);
      candidate_fusion->push_back(record);
      SetRecordFusionId(record);
      return;
    }
  }
}

void Conv2DBackpropEltwiseEltwiseFusionPass::MatchSingleFusionPattern(const session::KernelGraph &kernel_graph,
                                                                      FusedNodeRecord *candidate_fusion) {
  MS_EXCEPTION_IF_NULL(candidate_fusion);
  MS_CHECK_CUBE_VECTOR_SPLIT();

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
        common::AnfAlgo::GetCNodeName(cnode) == kReluGradV2OpName) {
      MatchConv2DBackpropInputEltwiseEltwise(cnode, kernel_graph, candidate_fusion);
    }
  }
}
}  // namespace opt
}  // namespace mindspore
