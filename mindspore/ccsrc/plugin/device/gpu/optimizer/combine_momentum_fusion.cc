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
#include "plugin/device/gpu/optimizer/combine_momentum_fusion.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
bool GetDealList(const std::vector<AnfNodePtr> &node_list, std::vector<std::vector<AnfNodePtr>> *deal_list) {
  MS_EXCEPTION_IF_NULL(deal_list);
  std::vector<AnfNodePtr> momentum;
  std::vector<AnfNodePtr> momentum_decay;
  for (auto &momentum_node : node_list) {
    if (momentum_node != nullptr && momentum_node->isa<CNode>()) {
      if (common::AnfAlgo::GetCNodeName(momentum_node) == kFusedScaleApplyMomentum) {
        momentum.push_back(momentum_node);
      } else if (common::AnfAlgo::GetCNodeName(momentum_node) == kFusedWeightScaleApplyMomentum) {
        momentum_decay.push_back(momentum_node);
      }
    }
  }
  if (momentum.size() <= 1 && momentum_decay.size() <= 1) {
    return false;
  }
  if (momentum.size() > 1) {
    deal_list->push_back(momentum);
  }
  if (momentum_decay.size() > 1) {
    deal_list->push_back(momentum_decay);
  }
  return true;
}
bool CombineMomentumFusion::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  // 1 get all the cast node
  std::vector<std::vector<AnfNodePtr>> deal_list;
  if (!GetDealList(node_list, &deal_list)) {
    return false;
  }
  for (auto momentums : deal_list) {
    if (momentums.size() == 0) {
      MS_LOG(EXCEPTION) << "The size of momentums is zero.";
    }
    // 2 create node momentum
    std::vector<AnfNodePtr> inputs = {};
    if (common::AnfAlgo::GetCNodeName(momentums[0]) == kFusedScaleApplyMomentum) {
      auto prim = std::make_shared<Primitive>("CombineMomentum");
      MS_EXCEPTION_IF_NULL(prim);
      inputs.push_back(NewValueNode(prim));
    } else {
      auto prim = std::make_shared<Primitive>("CombineMomentumWeight");
      MS_EXCEPTION_IF_NULL(prim);
      inputs.push_back(NewValueNode(prim));
    }
    // set inputs for momentum
    size_t input_num = common::AnfAlgo::GetInputTensorNum(momentums[0]);
    for (auto mom : momentums) {
      for (size_t i = 0; i < input_num; i++) {
        auto cnode = utils::cast<CNodePtr>(mom);
        MS_EXCEPTION_IF_NULL(cnode);
        inputs.push_back(common::AnfAlgo::GetInputNode(cnode, i));
      }
    }
    TraceGuard guard(std::make_shared<TraceOpt>(momentums[0]->debug_info()));
    auto combine_mom = graph->NewCNode(inputs);
    auto kernel_info = std::make_shared<device::KernelInfo>();
    MS_EXCEPTION_IF_NULL(combine_mom);
    MS_EXCEPTION_IF_NULL(kernel_info);
    combine_mom->set_kernel_info(kernel_info);
    AbstractBasePtrList abstract_list;
    for (size_t idx = 0; idx < momentums.size(); ++idx) {
      auto cnode = utils::cast<CNodePtr>(momentums[idx]);
      MS_EXCEPTION_IF_NULL(cnode);
      abstract_list.push_back(cnode->abstract());
    }
    auto kernel_build_info = GenerateKernelBuildInfo(momentums);
    AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info, combine_mom.get());
    auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
    MS_EXCEPTION_IF_NULL(abstract_tuple);
    combine_mom->set_abstract(abstract_tuple);
    common::AnfAlgo::SetNodeAttr("n", MakeValue(momentums.size()), combine_mom);
    // 3 replace all the cast by momentum
    for (size_t idx = 0; idx < momentums.size(); ++idx) {
      if (!manager->Replace(momentums[idx], combine_mom)) {
        MS_LOG(EXCEPTION) << "manager replace node failed";
      }
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
