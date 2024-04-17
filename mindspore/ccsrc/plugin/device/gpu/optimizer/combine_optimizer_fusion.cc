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

#include <unordered_map>
#include "plugin/device/gpu/optimizer/combine_optimizer_fusion.h"
#include "ops/ascend_op_name.h"
#include "ops/nn_optimizer_op_name.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/distributed/ps/ps_context.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
#define REGISTER_COMBINE_OPTIMIZER_TYPE(M, T, S) M[T] = S
std::unordered_map<std::string, std::string> kOptimizerMap;

void CombineOptimizerFusion::InitCombineOptimizer() {
  REGISTER_COMBINE_OPTIMIZER_TYPE(kOptimizerMap, kApplyMomentumOpName, kCombineMomentumOpName);
  REGISTER_COMBINE_OPTIMIZER_TYPE(kOptimizerMap, kFusedScaleApplyMomentumOpName, kCombineScaleMomentumOpName);
  REGISTER_COMBINE_OPTIMIZER_TYPE(kOptimizerMap, kFusedWeightScaleApplyMomentumOpName,
                                  kCombineWeightDecayScaleMomentumOpName);
}

bool CombineOptimizerFusion::CheckFuncGraph(const FuncGraphPtr &graph) {
  std::unordered_map<std::string, std::vector<TypeId>> base_optimizer_input_types;
  for (auto &node : graph->nodes()) {
    if (node == nullptr || !node->isa<CNode>()) {
      continue;
    }
    std::string node_name = common::AnfAlgo::GetCNodeName(node);
    if (kOptimizerMap.find(node_name) == kOptimizerMap.end()) {
      continue;
    }
    size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
    std::vector<TypeId> type_vec;
    for (size_t i = 0; i < input_num; i++) {
      type_vec.push_back(common::AnfAlgo::GetPrevNodeOutputInferDataType(node, i));
    }
    if (base_optimizer_input_types.find(node_name) == base_optimizer_input_types.end()) {
      base_optimizer_input_types[node_name] = type_vec;
    } else if (type_vec != base_optimizer_input_types[node_name]) {
      return false;
    }
  }
  return true;
}

bool CombineOptimizerFusion::CheckCondition() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto enable_ps = ps::PSContext::instance()->is_ps_mode();
  auto server_mode = ps::PSContext::instance()->server_mode();
  if (enable_ps || server_mode == mindspore::ps::kServerModePS) {
    return false;
  }
  auto ms_role = common::GetEnv(mindspore::ps::kEnvRole);
  if (ms_role == mindspore::ps::kEnvRoleOfWorker || ms_role == mindspore::ps::kEnvRoleOfPServer ||
      ms_role == mindspore::ps::kEnvRoleOfScheduler) {
    return false;
  }
  return true;
}

bool CombineOptimizerFusion::TransformOptimizerList(const std::vector<AnfNodePtr> &node_list,
                                                    std::vector<std::vector<AnfNodePtr>> *optimizer_node_lists) {
  MS_EXCEPTION_IF_NULL(optimizer_node_lists);

  std::unordered_map<std::string, std::vector<AnfNodePtr>> optimizer_anf_map;
  for (auto item : kOptimizerMap) {
    std::vector<AnfNodePtr> vec;
    optimizer_anf_map[item.first] = vec;
  }
  for (auto &node : node_list) {
    if (node == nullptr || !node->isa<CNode>()) {
      continue;
    }
    std::string node_name = common::AnfAlgo::GetCNodeName(node);
    if (kOptimizerMap.find(node_name) != kOptimizerMap.end()) {
      optimizer_anf_map[node_name].push_back(node);
    }
  }
  for (auto item : optimizer_anf_map) {
    auto optimizer_node_list = item.second;
    if (optimizer_node_list.size() > 1) {
      optimizer_node_lists->push_back(optimizer_node_list);
    }
  }
  return optimizer_node_lists->size() >= 1;
}

AnfNodePtr CombineOptimizerFusion::FindFirstMonadInput(
  const std::vector<AnfNodePtr> &optimizer_node_list,
  const mindspore::HashMap<AnfNodePtr, size_t> &nodes_to_topo_orders) {
  if (optimizer_node_list.empty()) {
    MS_LOG(EXCEPTION) << "The size of optimizer node list is zero.";
  }

  size_t first_topo_order = SIZE_MAX;
  AnfNodePtr first_topo_order_node = nullptr;
  for (const auto &optimizer_node : optimizer_node_list) {
    const auto &iter = nodes_to_topo_orders.find(optimizer_node);
    if (iter == nodes_to_topo_orders.end()) {
      MS_LOG(EXCEPTION) << "Can not find topo order of node: " << optimizer_node->fullname_with_scope();
    }
    if (iter->second < first_topo_order) {
      first_topo_order = iter->second;
      first_topo_order_node = optimizer_node;
    }
  }

  MS_EXCEPTION_IF_NULL(first_topo_order_node);
  auto first_optimizer_cnode = first_topo_order_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(first_optimizer_cnode);
  const auto &input_nodes = first_optimizer_cnode->inputs();
  if (input_nodes.empty()) {
    MS_LOG(EXCEPTION) << "The optimizer: " << first_optimizer_cnode->fullname_with_scope() << " has no input";
  }
  if (!HasAbstractMonad(input_nodes.back())) {
    MS_LOG(EXCEPTION) << "The last input of " << first_optimizer_cnode->fullname_with_scope() << " is not Monad node.";
  }
  return input_nodes.back();
}

bool CombineOptimizerFusion::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  if (!CheckCondition()) {
    return false;
  }

  if (!CheckFuncGraph(graph)) {
    return false;
  }
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  // 1 get all the cast node
  std::vector<std::vector<AnfNodePtr>> optimizer_node_lists;
  if (!TransformOptimizerList(node_list, &optimizer_node_lists)) {
    return false;
  }

  // Record all optimizer nodes topo order.
  mindspore::HashMap<AnfNodePtr, size_t> nodes_to_topo_orders;
  for (const std::vector<AnfNodePtr> &optimizer_nodes : optimizer_node_lists) {
    for (size_t i = 0; i < optimizer_nodes.size(); i++) {
      nodes_to_topo_orders[optimizer_nodes[i]] = i;
    }
  }

  for (auto optimizer_node_list : optimizer_node_lists) {
    if (optimizer_node_list.size() == 0) {
      MS_LOG(EXCEPTION) << "The size of optimizer node list is zero.";
    }
    // 2 create node combine optimizer node
    std::vector<AnfNodePtr> inputs = {};
    std::string node_name = common::AnfAlgo::GetCNodeName(optimizer_node_list[0]);
    if (kOptimizerMap.find(node_name) == kOptimizerMap.end()) {
      MS_LOG(EXCEPTION) << "The node name: " << node_name << " is invalid.";
    }

    auto combine_node_name = kOptimizerMap[node_name];
    auto prim = std::make_shared<Primitive>(combine_node_name);
    MS_EXCEPTION_IF_NULL(prim);
    inputs.push_back(NewValueNode(prim));
    // set inputs for combine optimizer node
    size_t input_num = common::AnfAlgo::GetInputTensorNum(optimizer_node_list[0]);
    for (auto optimizer_node : optimizer_node_list) {
      for (size_t i = 0; i < input_num; i++) {
        auto cnode = utils::cast<CNodePtr>(optimizer_node);
        MS_EXCEPTION_IF_NULL(cnode);
        inputs.push_back(common::AnfAlgo::GetInputNode(cnode, i));
      }
    }
    // Add monad input.
    const auto first_monad_input_node = FindFirstMonadInput(optimizer_node_list, nodes_to_topo_orders);
    MS_EXCEPTION_IF_NULL(first_monad_input_node);
    inputs.push_back(first_monad_input_node);

    TraceGuard guard(std::make_shared<TraceOpt>(optimizer_node_list[0]->debug_info()));
    auto combine_optimizer_node = graph->NewCNode(inputs);
    auto kernel_info = std::make_shared<device::KernelInfo>();
    MS_EXCEPTION_IF_NULL(combine_optimizer_node);
    MS_EXCEPTION_IF_NULL(kernel_info);
    combine_optimizer_node->set_kernel_info(kernel_info);
    AbstractBasePtrList abstract_list;
    for (size_t idx = 0; idx < optimizer_node_list.size(); ++idx) {
      auto cnode = utils::cast<CNodePtr>(optimizer_node_list[idx]);
      MS_EXCEPTION_IF_NULL(cnode);
      abstract_list.push_back(cnode->abstract());
    }
    auto kernel_build_info = GenerateKernelBuildInfo(optimizer_node_list);
    AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info, combine_optimizer_node.get());
    auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
    MS_EXCEPTION_IF_NULL(abstract_tuple);
    combine_optimizer_node->set_abstract(abstract_tuple);
    common::AnfAlgo::SetNodeAttr("combine_num", MakeValue(optimizer_node_list.size()), combine_optimizer_node);
    // 3 replace all the cast by combine optimizer node
    for (size_t idx = 0; idx < optimizer_node_list.size(); ++idx) {
      if (!manager->Replace(optimizer_node_list[idx], combine_optimizer_node)) {
        MS_LOG(EXCEPTION) << "manager replace node failed";
      }
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
