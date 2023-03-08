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
#include "plugin/device/ascend/optimizer/buffer_fusion/fusion_base_pass.h"
#include <memory>
#include "utils/ms_context.h"
#include "backend/common/optimizer/fusion_id_allocator.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
bool FusionBasePass::CheckEltWiseNode(const session::KernelGraph &kernel_graph, const AnfNodePtr &node,
                                      const std::unordered_set<std::string> &fusion_types, size_t input_size,
                                      size_t not_updatestate_size) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>() || !AnfUtils::IsRealCNodeKernel(node) || fusion_id_allocator->HasFusionIdAttr(node)) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  return AnfAlgo::GetKernelType(node) == KernelType::TBE_KERNEL &&
         fusion_types.find(AnfAlgo::GetFusionType(node)) != fusion_types.cend() &&
         GetNotUpdateStateUserNums(kernel_graph, node) == not_updatestate_size && cnode->inputs().size() == input_size;
}

bool FusionBasePass::CheckSingleInEltWiseNode(const session::KernelGraph &kernel_graph, const AnfNodePtr &node,
                                              const std::unordered_set<std::string> &fusion_types) {
  return CheckEltWiseNode(kernel_graph, node, fusion_types, ELTWISE_INPUT_SIZE, ELTWISE_USE);
}

bool FusionBasePass::CheckDoubleInEltWiseNode(const session::KernelGraph &kernel_graph, const AnfNodePtr &node,
                                              const std::unordered_set<std::string> &fusion_types) {
  return CheckEltWiseNode(kernel_graph, node, fusion_types, ELTWISE_DOUBLE_IN_INPUT_SIZE, ELTWISE_USE);
}

bool FusionBasePass::CheckMultiOutputEltWiseNode(const session::KernelGraph &kernel_graph, const AnfNodePtr &node,
                                                 const std::unordered_set<std::string> &fusion_types) {
  return CheckEltWiseNode(kernel_graph, node, fusion_types, ELTWISE_INPUT_SIZE, ELTWISE_MULTI_USE);
}

size_t FusionBasePass::GetNotUpdateStateUserNums(const session::KernelGraph &kernel_graph,
                                                 const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  auto manager = kernel_graph.manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto user_nodes = manager->node_users()[node];
  size_t not_updatestate_users = 0;
  for (auto &user : user_nodes) {
    auto user_node = user.first;
    if (!common::AnfAlgo::CheckPrimitiveType(user_node, prim::kPrimUpdateState)) {
      not_updatestate_users++;
    }
  }
  return not_updatestate_users;
}

void FusionBasePass::SetRecordFusionId(const mindspore::HashSet<AnfNodePtr> &record) {
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

bool FusionBasePass::RunPass(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<std::shared_ptr<session::KernelGraph>>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  return MatchUBFusionPattern(*kernel_graph);
}
}  // namespace opt
}  // namespace mindspore
