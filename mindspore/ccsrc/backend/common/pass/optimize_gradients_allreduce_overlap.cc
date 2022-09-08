/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "backend/common/pass/optimize_gradients_allreduce_overlap.h"
#include <algorithm>
#include <vector>
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/parallel_context.h"

namespace mindspore {
namespace opt {
constexpr const int64_t kFusionGap = 2;
constexpr auto kGradientsFlag = "Gradients";
constexpr auto depend_key = "allreduce_dependent_node";
constexpr auto downstream_key = "allreduce_downstream_node";
namespace {
std::string FusionGroupKey(const std::string &comm_group, const std::string &backward_comm_name, int64_t fusion_id) {
  std::string fusion_key = backward_comm_name + "_" + comm_group + "_" + std::to_string(fusion_id);
  return fusion_key;
}

bool IsBpropNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  return node->fullname_with_scope().find(kGradientsFlag) == 0;
}

bool CheckFusionIdMatch(mindspore::HashMap<std::string, std::vector<CNodePtr>> fusion_allreduce_list_map,
                        mindspore::HashMap<std::string, std::vector<CNodePtr>> fusion_grads_node_list_map) {
  std::vector<std::string> fusion_id_allreduce;
  std::vector<std::string> fusion_id_grads;
  (void)std::transform(fusion_allreduce_list_map.begin(), fusion_allreduce_list_map.end(),
                       std::back_inserter(fusion_id_allreduce), [](auto pair) { return pair.first; });
  (void)std::transform(fusion_grads_node_list_map.begin(), fusion_grads_node_list_map.end(),
                       std::back_inserter(fusion_id_grads), [](auto pair) { return pair.first; });
  std::sort(fusion_id_allreduce.begin(), fusion_id_allreduce.end());
  std::sort(fusion_id_grads.begin(), fusion_id_grads.end());
  MS_LOG(INFO) << "fusion_id_allreduce: " << fusion_id_allreduce << ", fusion_id_grads: " << fusion_id_grads;
  return fusion_id_allreduce == fusion_id_grads;
}

// set nodes on which allreduce dependents with allreduce_dependent_tag
void SpreadDependLabel(const std::string &fusion_key, const CNodePtr &cnode) {
  for (size_t i = 1; i < cnode->size(); ++i) {
    if (IsPrimitiveCNode(cnode->input(i))) {
      auto input_cnode = cnode->input(i)->cast<CNodePtr>();
      if (input_cnode->HasPrimalAttr(fusion_key)) {
        std::vector<std::string> dependent_node_fusion_key_list;
        bool is_spread = false;
        if (!input_cnode->HasPrimalAttr(depend_key)) {
          dependent_node_fusion_key_list = {fusion_key};
          input_cnode->AddPrimalAttr(depend_key, MakeValue(dependent_node_fusion_key_list));
          is_spread = true;
        } else {
          dependent_node_fusion_key_list = GetValue<std::vector<std::string>>(input_cnode->GetPrimalAttr(depend_key));
          if (std::find(dependent_node_fusion_key_list.begin(), dependent_node_fusion_key_list.end(), fusion_key) ==
              dependent_node_fusion_key_list.end()) {
            dependent_node_fusion_key_list.push_back(fusion_key);
            input_cnode->AddPrimalAttr(depend_key, MakeValue(dependent_node_fusion_key_list));
            is_spread = true;
          }
        }
        if (is_spread) {
          SpreadDependLabel(fusion_key, input_cnode);
        }
      }
    }
  }
}

// set allreduce downstream nodes with allreduce_downstream_tag
void SpreadOutputLabel(const std::string &fusion_key, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto func_graph = cnode->func_graph();
  auto manager = func_graph->manager();
  auto node_users = manager->node_users()[cnode];
  for (auto &node_pair : node_users) {
    if (!IsPrimitiveCNode(node_pair.first)) {
      continue;
    }
    auto output_cnode = node_pair.first->cast<CNodePtr>();
    std::vector<std::string> output_node_fusion_key_list;
    bool is_spread = false;
    if (!output_cnode->HasPrimalAttr(downstream_key)) {
      output_node_fusion_key_list = {fusion_key};
      output_cnode->AddPrimalAttr(downstream_key, MakeValue(output_node_fusion_key_list));
      is_spread = true;
    } else {
      output_node_fusion_key_list = GetValue<std::vector<std::string>>(output_cnode->GetPrimalAttr(downstream_key));
      if (std::find(output_node_fusion_key_list.begin(), output_node_fusion_key_list.end(), fusion_key) ==
          output_node_fusion_key_list.end()) {
        output_node_fusion_key_list.push_back(fusion_key);
        output_cnode->AddPrimalAttr(downstream_key, MakeValue(output_node_fusion_key_list));
        is_spread = true;
      }
    }
    if (is_spread) {
      SpreadOutputLabel(fusion_key, output_cnode);
    }
  }
}

mindspore::HashMap<std::string, size_t> GradNodeRelatedForwardNodeOrderId(const std::vector<AnfNodePtr> &node_list) {
  mindspore::HashMap<std::string, size_t> forward_order_id;
  for (size_t i = 0; i < node_list.size(); ++i) {
    auto node = node_list[i];
    if (!node->cast<CNodePtr>() || !AnfUtils::IsRealKernel(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (IsBpropNode(cnode) || !cnode->HasPrimalAttr(parallel::kRelatedFusionKey) ||
        !cnode->HasPrimalAttr(parallel::kRelatedNodeId)) {
      continue;
    }
    auto related_node_id = GetValue<std::string>(cnode->GetPrimalAttr(parallel::kRelatedNodeId));
    forward_order_id[related_node_id] = i;
  }
  return forward_order_id;
}

void ExtractFusionCommNodes(const std::vector<AnfNodePtr> &node_list,
                            mindspore::HashMap<std::string, std::vector<CNodePtr>> *fusion_allreduce_list_map) {
  for (auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->cast<CNodePtr>() || !AnfUtils::IsRealKernel(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!common::AnfAlgo::IsFusion(cnode)) {
      continue;
    }
    if ((IsPrimitiveCNode(cnode, prim::kPrimAllReduce) &&
         GetCNodePrimitive(cnode)->instance_name().find("grad_mirror") != std::string::npos) ||
        (IsPrimitiveCNode(cnode, prim::kPrimReduceScatter) &&
         GetCNodePrimitive(cnode)->instance_name().find("grad_parallel_optimizer") != std::string::npos)) {
      auto fusion_id = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion);
      auto group_name = common::AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrGroup);
      auto comm_name = IsPrimitiveCNode(cnode, prim::kPrimAllReduce) ? "all_reduce" : "reduce_scatter";
      auto key_name = FusionGroupKey(group_name, comm_name, fusion_id);
      if ((*fusion_allreduce_list_map).find(key_name) == (*fusion_allreduce_list_map).end()) {
        std::vector<CNodePtr> allreduce_node_list = {cnode};
        (*fusion_allreduce_list_map)[key_name] = allreduce_node_list;
      } else {
        (*fusion_allreduce_list_map)[key_name].push_back(cnode);
      }
    }
  }
}

void ExtractGradRelatedNodes(const std::vector<AnfNodePtr> &node_list,
                             mindspore::HashMap<std::string, std::vector<CNodePtr>> *fusion_grads_node_list_map) {
  for (auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->cast<CNodePtr>() || !AnfUtils::IsRealKernel(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!IsPrimitiveCNode(cnode) || !IsBpropNode(cnode) || !cnode->HasPrimalAttr(parallel::kRelatedFusionKey)) {
      continue;
    }
    auto grad_key_name = GetValue<std::string>(cnode->GetPrimalAttr(parallel::kRelatedFusionKey));
    if ((*fusion_grads_node_list_map).find(grad_key_name) == (*fusion_grads_node_list_map).end()) {
      std::vector<CNodePtr> fusion_grads_node_list = {cnode};
      (*fusion_grads_node_list_map)[grad_key_name] = fusion_grads_node_list;
    } else {
      (*fusion_grads_node_list_map)[grad_key_name].push_back(cnode);
    }
  }
}

// tag nodes between first backward node and last allreduce node with the same fusion id, and construct to the searching
// space of the allreduce dependence searching.
void TagAllReduceDependentsNodesSearchSpace(
  const std::vector<AnfNodePtr> &node_list,
  mindspore::HashMap<std::string, std::vector<CNodePtr>> *fusion_allreduce_list_map,
  mindspore::HashMap<std::string, std::vector<CNodePtr>> *fusion_grads_node_list_map) {
  for (auto iter = (*fusion_allreduce_list_map).begin(); iter != (*fusion_allreduce_list_map).end(); ++iter) {
    std::string fusion_key = (*iter).first;
    auto first_grad_node = (*fusion_grads_node_list_map)[fusion_key].front();
    auto first_grad_node_idx_in_orders = std::find(node_list.begin(), node_list.end(), first_grad_node);
    std::vector<CNodePtr> allreduce_list = (*iter).second;
    auto last_allreduce_node = allreduce_list.back();
    auto last_allreduce_idx_in_orders = std::find(node_list.begin(), node_list.end(), last_allreduce_node);
    if (last_allreduce_idx_in_orders - first_grad_node_idx_in_orders < 0) {
      MS_LOG(EXCEPTION) << "The allreduce node dose not has any backward node with the same fusion id before it.";
    }
    for (auto anf_iter = first_grad_node_idx_in_orders; anf_iter != last_allreduce_idx_in_orders; ++anf_iter) {
      if (!(*anf_iter)->cast<CNodePtr>()) {
        continue;
      }
      auto tag_cnode = (*anf_iter)->cast<CNodePtr>();
      tag_cnode->AddPrimalAttr(fusion_key, MakeValue(true));
    }
  }
}

void LabelingAllReduceDependentsNodes(
  const mindspore::HashMap<std::string, std::vector<CNodePtr>> &fusion_allreduce_list_map) {
  for (auto iter = fusion_allreduce_list_map.begin(); iter != fusion_allreduce_list_map.end(); ++iter) {
    std::string fusion_key = (*iter).first;
    std::vector<CNodePtr> allreduce_list = (*iter).second;
    for (auto &allreduce_node : allreduce_list) {
      SpreadDependLabel(fusion_key, allreduce_node);
    }
    for (auto &allreduce_node : allreduce_list) {
      SpreadOutputLabel(fusion_key, allreduce_node);
    }
  }
}

mindspore::HashMap<std::string, CNodePtr> LastNodeNotDownStream(
  const std::vector<AnfNodePtr> &node_list,
  mindspore::HashMap<std::string, std::vector<CNodePtr>> *fusion_allreduce_list_map,
  mindspore::HashMap<std::string, size_t> *forward_node_order_id_map) {
  mindspore::HashMap<std::string, CNodePtr> last_node_not_downstream;
  for (auto &pair_iter : (*fusion_allreduce_list_map)) {
    auto fusion_key = pair_iter.first;
    auto comm_list = pair_iter.second;
    auto last_comm_node = comm_list.back();
    auto last_comm_node_idx_in_orders = std::find(node_list.begin(), node_list.end(), last_comm_node);
    std::vector<CNodePtr> candidate_node_list;
    for (auto anf_iter = last_comm_node_idx_in_orders; anf_iter != node_list.end(); ++anf_iter) {
      auto anf_node = *anf_iter;
      MS_EXCEPTION_IF_NULL(anf_node);
      if (!anf_node->cast<CNodePtr>() || !IsBpropNode(anf_node) || common::AnfAlgo::IsCommunicationOp(anf_node)) {
        continue;
      }
      auto cnode = anf_node->cast<CNodePtr>();
      if (!cnode->HasPrimalAttr(parallel::kRelatedFusionKey) || !cnode->HasPrimalAttr(parallel::kRelatedNodeId)) {
        continue;
      }
      auto grad_key_name = GetValue<std::string>(cnode->GetPrimalAttr(parallel::kRelatedFusionKey));
      if (grad_key_name == fusion_key) {
        continue;
      }
      // remove those branch nodes.
      if (cnode->HasPrimalAttr(depend_key)) {
        auto dependent_node_fusion_key_list = GetValue<std::vector<std::string>>(cnode->GetPrimalAttr(depend_key));
        if (std::find(dependent_node_fusion_key_list.begin(), dependent_node_fusion_key_list.end(), grad_key_name) !=
            dependent_node_fusion_key_list.end()) {
          continue;
        }
      }
      if (cnode->HasPrimalAttr(downstream_key)) {
        auto downstream_fusion_key_list = GetValue<std::vector<std::string>>(cnode->GetPrimalAttr(downstream_key));
        if (std::find(downstream_fusion_key_list.begin(), downstream_fusion_key_list.end(), fusion_key) !=
            downstream_fusion_key_list.end()) {
          continue;
        }
      }
      candidate_node_list.push_back(cnode);
    }
    if (candidate_node_list.empty()) {
      continue;
    }
    size_t min_id = node_list.size();
    for (auto &candidate_node : candidate_node_list) {
      auto candidate_node_id = GetValue<std::string>(candidate_node->GetPrimalAttr(parallel::kRelatedNodeId));
      if ((*forward_node_order_id_map).find(candidate_node_id) != (*forward_node_order_id_map).end()) {
        auto order_id = (*forward_node_order_id_map)[candidate_node_id];
        if (order_id < min_id) {
          min_id = order_id;
          last_node_not_downstream[fusion_key] = candidate_node;
          MS_LOG(INFO) << "final last_node_not_downstream node:" << candidate_node->fullname_with_scope()
                       << ", fusion_key:" << fusion_key;
        }
      }
    }
  }
  return last_node_not_downstream;
}

void InsertFirstDependNode(const FuncGraphPtr &graph, const CNodePtr &first_grad_node_allreduce_no_dependent,
                           const std::vector<CNodePtr> &allreduce_node_list) {
  auto manager = graph->manager();
  // insert the first depend:  allreduce_next_cnode -> depend -> first_grad_node_allreduce_no_dependent
  if (first_grad_node_allreduce_no_dependent == nullptr || allreduce_node_list.empty()) {
    return;
  }
  for (auto &allreduce_node : allreduce_node_list) {
    std::vector<AnfNodePtr> first_depend_inputs = {
      NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
      common::AnfAlgo::GetInputNode(first_grad_node_allreduce_no_dependent, 0), allreduce_node};
    auto first_depend = graph->NewCNode(first_depend_inputs);
    first_depend->AddPrimalAttr("first_depend", MakeValue(true));
    manager->SetEdge(first_grad_node_allreduce_no_dependent, 1, first_depend);
  }
}

void RemoveFirstDependNode(const std::vector<AnfNodePtr> &node_list) {
  for (auto &node : node_list) {
    if (!IsPrimitiveCNode(node, prim::kPrimDepend)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!cnode->HasPrimalAttr("first_depend")) {
      continue;
    }
    auto input_cnode = common::AnfAlgo::GetInputNode(cnode, 0);
    auto manager = node->func_graph()->manager();
    (void)manager->Replace(cnode, input_cnode);
  }
}

void InsertSecondDependNode(const FuncGraphPtr &graph, const CNodePtr &last_node_not_downstream,
                            const std::vector<CNodePtr> &allreduce_next_cnode_list) {
  auto manager = graph->manager();
  // last fusion id nodes no need the second depend
  if (last_node_not_downstream == nullptr) {
    return;
  }
  // insert the second depend: allreduce_next_cnode_list[i]-> depend -> last_node_not_downstream
  std::vector<AnfNodePtr> allreduce_next_cnode_input_list;
  for (auto &allreduce_next_cnode : allreduce_next_cnode_list) {
    auto allreduce_next_cnode_input = common::AnfAlgo::GetInputNode(allreduce_next_cnode, 0);
    allreduce_next_cnode_input_list.push_back(allreduce_next_cnode_input);
  }
  for (size_t i = 0; i < allreduce_next_cnode_list.size(); ++i) {
    auto allreduce_next_cnode = allreduce_next_cnode_list[i];
    auto allreduce_next_cnode_input = allreduce_next_cnode_input_list[i];
    std::vector<AnfNodePtr> second_depend_inputs = {
      NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())), allreduce_next_cnode_input,
      last_node_not_downstream};
    auto second_depend = graph->NewCNode(second_depend_inputs);
    second_depend->AddPrimalAttr("second_depend", MakeValue(true));
    manager->SetEdge(allreduce_next_cnode, 1, second_depend);
  }
}

// first_grad_node_allreduce_no_dependent -> first_depend -> allreduce_next_cnode
// reconstruct topo sort order, than remove first depend
// last_node_not_downstream -> second_depend -> allreduce_next_cnode_list[i]
bool OverLapGradientsAllReduceAndCompute(
  const FuncGraphPtr &graph, mindspore::HashMap<std::string, std::vector<CNodePtr>> *fusion_allreduce_list_map,
  mindspore::HashMap<std::string, std::vector<CNodePtr>> *fusion_grads_node_list_map) {
  auto manager = graph->manager();
  bool changed = false;
  for (auto grad_iter = (*fusion_grads_node_list_map).begin(); grad_iter != (*fusion_grads_node_list_map).end();
       ++grad_iter) {
    auto fusion_key = (*grad_iter).first;
    std::vector<CNodePtr> grads_list = (*grad_iter).second;
    CNodePtr first_grad_node_allreduce_no_dependent = nullptr;
    for (auto &grad_node : grads_list) {
      if (grad_node->HasPrimalAttr(depend_key)) {
        auto dependent_node_fusion_key_list = GetValue<std::vector<std::string>>(grad_node->GetPrimalAttr(depend_key));
        if (std::find(dependent_node_fusion_key_list.begin(), dependent_node_fusion_key_list.end(), fusion_key) !=
            dependent_node_fusion_key_list.end()) {
          continue;
        }
      }
      first_grad_node_allreduce_no_dependent = grad_node;
      break;
    }

    InsertFirstDependNode(graph, first_grad_node_allreduce_no_dependent, (*fusion_allreduce_list_map)[fusion_key]);
    changed = true;
  }
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  auto forward_node_order_id_map = GradNodeRelatedForwardNodeOrderId(node_list);
  for (auto &comm_pair : (*fusion_allreduce_list_map)) {
    comm_pair.second.clear();
  }

  ExtractFusionCommNodes(node_list, fusion_allreduce_list_map);
  auto last_node_not_downstream_map =
    LastNodeNotDownStream(node_list, fusion_allreduce_list_map, &forward_node_order_id_map);
  // remove first depend
  RemoveFirstDependNode(node_list);

  for (auto grad_iter = (*fusion_grads_node_list_map).begin(); grad_iter != (*fusion_grads_node_list_map).end();
       ++grad_iter) {
    auto fusion_key = (*grad_iter).first;
    std::vector<CNodePtr> allreduce_next_cnode_list;
    for (auto &allreduce_node : (*fusion_allreduce_list_map)[fusion_key]) {
      auto node_users = manager->node_users()[allreduce_node];
      for (auto &node_pair : node_users) {
        if (IsPrimitiveCNode(node_pair.first)) {
          allreduce_next_cnode_list.push_back(node_pair.first->cast<CNodePtr>());
        }
      }
    }
    if (last_node_not_downstream_map.find(fusion_key) == last_node_not_downstream_map.end()) {
      continue;
    }
    CNodePtr last_node_not_downstream = last_node_not_downstream_map[fusion_key];
    InsertSecondDependNode(graph, last_node_not_downstream, allreduce_next_cnode_list);
    changed = true;
  }
  return changed;
}
}  // namespace

bool OptimizeGradientsAllReduceOverlap::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode != parallel::kSemiAutoParallel && parallel_mode != parallel::kAutoParallel) {
    return false;
  }
  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() > 1) {
    return false;
  }
  mindspore::HashMap<std::string, std::vector<CNodePtr>> fusion_allreduce_list_map;
  mindspore::HashMap<std::string, std::vector<CNodePtr>> fusion_grads_node_list_map;
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  ExtractFusionCommNodes(node_list, &fusion_allreduce_list_map);
  ExtractGradRelatedNodes(node_list, &fusion_grads_node_list_map);
  if (!CheckFusionIdMatch(fusion_allreduce_list_map, fusion_grads_node_list_map)) {
    MS_LOG(INFO) << "The fusion id is not match for allreduce and backward nodes.";
    return false;
  }

  TagAllReduceDependentsNodesSearchSpace(node_list, &fusion_allreduce_list_map, &fusion_grads_node_list_map);

  LabelingAllReduceDependentsNodes(fusion_allreduce_list_map);

  return OverLapGradientsAllReduceAndCompute(graph, &fusion_allreduce_list_map, &fusion_grads_node_list_map);
}
}  // namespace opt
}  // namespace mindspore
