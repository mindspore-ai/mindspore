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
#include "pre_activate/ascend/buffer_fusion/fusion_base_pass.h"
#include <unordered_set>
#include <memory>
#include "debug/anf_ir_dump.h"
#include "utils/context/ms_context.h"
#include "pre_activate/common/fusion_id_allocator.h"
#include "session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
bool FusionBasePass::CheckEltWiseNode(FuncGraphManager *manager, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(manager);
  if (!node->isa<CNode>() || !AnfAlgo::IsRealCNodeKernel(node) || fusion_id_allocator->HasFusionIdAttr(node)) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto user_nodes = manager->node_users()[node];
  return AnfAlgo::GetKernelType(node) == KernelType::TBE_KERNEL &&
         AnfAlgo::GetFusionType(node) == kernel::FusionType::ELEMWISE && user_nodes.size() == ELTWISE_USE &&
         cnode->inputs().size() == ELTWISE_INPUT_SIZE;
}

bool FusionBasePass::CheckDoubleInEltWiseNode(FuncGraphManager *manager, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(manager);
  if (!node->isa<CNode>() || !AnfAlgo::IsRealCNodeKernel(node) || fusion_id_allocator->HasFusionIdAttr(node)) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto user_nodes = manager->node_users()[node];
  return AnfAlgo::GetKernelType(node) == KernelType::TBE_KERNEL &&
         AnfAlgo::GetFusionType(node) == kernel::FusionType::ELEMWISE && user_nodes.size() == ELTWISE_USE &&
         cnode->inputs().size() == ELTWISE_DOUBLE_IN_INPUT_SIZE;
}

void FusionBasePass::SetRecordFusionId(const std::unordered_set<AnfNodePtr> &record) {
  auto id = fusion_id_allocator->AllocateFusionId();
  for (auto node : record) {
    fusion_id_allocator->SetFusionId(node, id);
  }
}

bool FusionBasePass::MatchUBFusionPattern(const session::KernelGraph &kernel_graph) {
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto return_node = kernel_graph.get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  if (return_node->inputs().size() <= 1) {
    return false;
  }
  MS_LOG(DEBUG) << "MatchBufferFusionPattern start...";
  FusedNodeRecord candidate_fusion;
  MatchSingleFusionPattern(kernel_graph, &candidate_fusion);
  if (candidate_fusion.empty()) {
    return false;
  }
  MS_LOG(DEBUG) << "MatchBufferFusionPattern Success...";
  return true;
}

bool FusionBasePass::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<std::shared_ptr<session::KernelGraph>>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  return MatchUBFusionPattern(*kernel_graph);
}
}  // namespace opt
}  // namespace mindspore
