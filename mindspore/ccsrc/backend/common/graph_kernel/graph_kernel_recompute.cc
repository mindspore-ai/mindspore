/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "backend/common/graph_kernel/graph_kernel_recompute.h"

#include <algorithm>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <tuple>
#include <utility>
#include <vector>
#include "kernel/common_utils.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "backend/common/graph_kernel/core/graph_builder.h"
#include "ir/func_graph_cloner.h"

namespace mindspore::graphkernel {
namespace {
int64_t GetGetitemIndex(const AnfNodePtr &getitem) {
  auto vnode = GetValueNode(getitem->cast<CNodePtr>()->input(kInputNodeOutputIndexInTupleGetItem));
  return GetValue<int64_t>(vnode);
}

AnfNodePtr GetOutput(const FuncGraphPtr &func_graph, size_t i) {
  auto output_node = func_graph->output()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(output_node);
  if (IsPrimitiveCNode(output_node, prim::kPrimMakeTuple)) {
    if (i + 1 >= output_node->size()) {
      MS_LOG(EXCEPTION) << i << " is out of range of MakeTuple's size " << output_node->size();
    }
    return output_node->input(i + 1);
  } else {
    if (i > 0) {
      MS_LOG(EXCEPTION) << "the graph is single output but i is not 0. it's " << i;
    }
    return output_node->cast<AnfNodePtr>();
  }
}

bool IsExclude(const AnfNodePtr &node) {
  static std::vector<PrimitivePtr> excludes = {prim::kPrimReturn, prim::kPrimUpdateState, prim::kPrimLoad,
                                               prim::kPrimMakeTuple, prim::kPrimDepend};
  return std::any_of(excludes.begin(), excludes.end(),
                     [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
}

enum class VisitType : char { FOLLOW, STOP };
using VisitFunc = std::function<VisitType(const AnfNodePtr &)>;
using NextFunc = std::function<AnfNodePtrList(const AnfNodePtr &)>;
using ProcessFunc = std::function<void(const AnfNodePtr &)>;

void Dfs(const AnfNodePtr &current, const VisitFunc &visit_func, const NextFunc &next_func,
         const ProcessFunc &before_func, const ProcessFunc &after_func, std::set<AnfNodePtr> *visited) {
  if (visited->count(current) > 0) {
    return;
  }
  (void)visited->insert(current);
  if (visit_func(current) != VisitType::FOLLOW) {
    return;
  }

  for (const auto &next : next_func(current)) {
    before_func(next);
    Dfs(next, visit_func, next_func, before_func, after_func, visited);
    after_func(next);
  }
}

OrderedMap<AnfNodePtr, AnfNodePtrList> CollectLinkPaths(const std::map<AnfNodePtr, MemorySize> &topo_indice,
                                                        const OrderedSet<AnfNodePtr> &direct_users,
                                                        MemorySize max_topo_user_index,
                                                        const FuncGraphManagerPtr &mng) {
  std::stack<AnfNodePtr> cur_stack;
  OrderedMap<AnfNodePtr, AnfNodePtrList> link_paths;
  auto TmpVisitFunc = [&topo_indice, max_topo_user_index](const AnfNodePtr &n) -> VisitType {
    if (IsExclude(n)) {
      return VisitType::STOP;
    }

    auto iter = topo_indice.find(n);
    if (iter == topo_indice.end()) {
      MS_LOG(EXCEPTION) << "Cannot find " << n->fullname_with_scope() << " in topo indices!";
    }
    if (iter->second > max_topo_user_index) {
      return VisitType::STOP;
    }
    return VisitType::FOLLOW;
  };

  auto TmpNextFunc = [&mng](const AnfNodePtr &n) -> AnfNodePtrList {
    auto users = mng->node_users()[n];
    AnfNodePtrList nexts;
    (void)std::transform(users.cbegin(), users.cend(), std::back_inserter(nexts),
                         [](const std::pair<AnfNodePtr, int> &user) { return user.first; });
    return nexts;
  };

  auto TmpBeforeFunc = [&link_paths, &cur_stack, &direct_users](const AnfNodePtr &next) -> void {
    if (direct_users.count(next) == 0) {
      return;
    }
    auto cur_node = cur_stack.top();
    if (link_paths.find(cur_node) == link_paths.end()) {
      (void)link_paths.emplace(cur_node, AnfNodePtrList());
    }
    link_paths[cur_node].push_back(next);
    cur_stack.push(next);
  };

  auto TmpAfterFunc = [&cur_stack, &direct_users](const AnfNodePtr &next) -> void {
    if (direct_users.count(next) == 0) {
      return;
    }
    cur_stack.push(next);
  };

  std::set<AnfNodePtr> visited;
  for (auto user : direct_users) {
    cur_stack.push(user);
    Dfs(user, TmpVisitFunc, TmpNextFunc, TmpBeforeFunc, TmpAfterFunc, &visited);
    cur_stack.pop();
  }

  return link_paths;
}

OrderedSet<AnfNodePtr> GetLongTermNodes(const AnfNodePtrList &nodes, const AnfNodePtr &end_node,
                                        const std::map<AnfNodePtr, MemorySize> &topo_indices,
                                        const FuncGraphManagerPtr &mng) {
  OrderedSet<AnfNodePtr> long_term_nodes;
  for (auto node : nodes) {
    auto real_node = common::AnfAlgo::VisitKernelWithReturnType(node, 0).first;
    // Parameter or value have long term tensors.
    if (!utils::isa<CNodePtr>(real_node)) {
      (void)long_term_nodes.insert(node);
      continue;
    }

    auto users = mng->node_users()[node];
    if (std::any_of(users.cbegin(), users.cend(), [&topo_indices, &end_node](const std::pair<AnfNodePtr, int> &user) {
          auto user_topo = topo_indices.find(user.first);
          auto end_topo = topo_indices.find(end_node);
          return user_topo->second >= end_topo->second;
        })) {
      (void)long_term_nodes.insert(node);
    }
  }
  return long_term_nodes;
}

/**
 * @brief Remove real input which is not used and change the related graph parameters.
 *
 * @param func_graph Graph.
 * @param inputs Real inputs for graph cnode.
 */
void ElimRedundantInputsAndGraphParameters(const FuncGraphPtr &func_graph, AnfNodePtrList *inputs) {
  MS_EXCEPTION_IF_NULL(inputs);
  const auto &ori_parameter = func_graph->parameters();
  auto nodes = TopoSort(func_graph->get_return());
  std::set<AnfNodePtr> used_param;
  for (auto node : nodes) {
    if (node->isa<Parameter>()) {
      (void)used_param.insert(node);
    }
  }
  if (used_param.size() == ori_parameter.size()) {
    return;
  }
  AnfNodePtrList new_parameter, new_inputs;
  for (size_t i = 0; i < ori_parameter.size(); ++i) {
    if (used_param.count(ori_parameter[i]) != 0) {
      new_parameter.push_back(ori_parameter[i]);
      new_inputs.push_back((*inputs)[i]);
    }
  }
  func_graph->set_parameters(new_parameter);
  *inputs = std::move(new_inputs);
}
}  // namespace

std::vector<Candidate> AutoRecompute::Run(const FuncGraphPtr &func_graph) {
  lifetime_threshold_ = GraphKernelFlags::GetInstance().recompute_increment_threshold;
  local_peak_threshold_ = GraphKernelFlags::GetInstance().recompute_peak_threshold;
  if (!IsThresholdDefaultValue()) {
    FindCandidates(func_graph);
  }
  return candidates_;
}

/**
 * @brief Filter the input tensor(that live longer than end node) out and return valid inputs for memory calculation. \n
 *        If the topo indices of the input's user is at least one greater than end_node,                              \n
 *        it will retain when after end_node's execution.
 *
 * @param source_node
 * @param end_node
 * @param edge_pos
 * @param mng
 * @return AnfNodePtrList
 */
AnfNodePtrList AutoRecompute::Filter(const AnfNodePtr &source_node, const AnfNodePtr &end_node, int edge_pos,
                                     const FuncGraphManagerPtr &mng) {
  auto source_cnode = source_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(source_cnode);
  AnfNodePtrList node_inputs(source_cnode->inputs().begin() + 1, source_cnode->inputs().end());
  OrderedSet<AnfNodePtr> long_term_inputs = GetLongTermNodes(node_inputs, end_node, topo_indice_, mng);

  AnfNodePtrList check_inputs;
  if (IsPrimitiveCNode(end_node->cast<CNodePtr>()->input(IntToSize(edge_pos)), prim::kPrimTupleGetItem)) {
    auto out_index = GetSourceLinkOutPos(end_node, edge_pos);
    auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(source_node);
    auto out = sub_graph->output();
    if (!IsPrimitiveCNode(out, prim::kPrimMakeTuple)) {
      MS_LOG(EXCEPTION) << "Expect MakeTuple node, but got " << common::AnfAlgo::GetCNodeName(out);
    }

    // Find subgraph's input according to edge node.
    auto start_node = out->cast<CNodePtr>()->input(IntToSize(out_index + 1));
    AnfNodePtrList sub_input_parameters;
    std::queue<AnfNodePtr> node_q;
    node_q.push(start_node);
    while (!node_q.empty()) {
      auto cur = node_q.front();
      node_q.pop();
      if (utils::isa<ParameterPtr>(cur)) {
        sub_input_parameters.push_back(cur);
      }
      auto cur_cnode = cur->cast<CNodePtr>();
      if (cur_cnode) {
        for (size_t i = 1; i < cur_cnode->inputs().size(); ++i) {
          node_q.push(cur_cnode->input(i));
        }
      }
    }

    // Filte input that user's topo index is great than source graph.
    for (auto para : sub_input_parameters) {
      for (size_t i = 0; i < sub_graph->parameters().size(); ++i) {
        if (para == sub_graph->parameters()[i]) {
          check_inputs.push_back(node_inputs[i]);
        }
      }
    }
  } else {
    check_inputs = node_inputs;
  }

  AnfNodePtrList res;
  for (auto input : check_inputs) {
    if (long_term_inputs.count(input) == 0) {
      res.push_back(input);
    }
  }

  return res;
}

/**
 * @brief Get valid users information by giving node, excluding TupleGetItem, Load and so on.
 */
std::tuple<OrderedSet<AnfNodePtr>, OutPosLinkMap, MemorySize> AutoRecompute::GetValidUsers(
  const AnfNodePtr &node, const FuncGraphManagerPtr &mng) {
  auto &user_map = mng->node_users();
  auto users = user_map[node];
  MemorySize max_topo_user_index = 0;
  std::queue<std::pair<AnfNodePtr, int>> users_queue;
  for (auto user_index : users) {
    users_queue.push(user_index);
  }
  OrderedSet<AnfNodePtr> direct_users;
  OutPosLinkMap user_edge_pos;
  while (!users_queue.empty()) {
    auto [user, index] = users_queue.front();
    users_queue.pop();
    if (IsPrimitiveCNode(user, prim::kPrimTupleGetItem)) {
      for (auto get_item_user : user_map[user]) {
        users_queue.push(get_item_user);
      }
      continue;
    } else if (IsExclude(user)) {
      continue;
    }
    user_edge_pos[user].push_back(index);
    (void)direct_users.insert(user);
    // Update maximum topo value.
    if (topo_indice_[user] > max_topo_user_index) {
      max_topo_user_index = topo_indice_[user];
    }
  }

  return {direct_users, user_edge_pos, max_topo_user_index};
}

/**
 * @brief Judege target node for recompute according to current node, and capture source node information when find   \n
 *        target. There two type for tensor of the edge between source node and target node, example:                 \n
 *          source ──[Short-Term]── A ── other                                                                        \n
 *             │                           │                                                                          \n
 *             └───────[Long-Term]────── target                                                                       \n
 *          For this example,                                                                                         \n
 *          1. There are two path from source node to target node, and target is directly user for source node,       \n
 *             so the tensor of their edge is a long-term tensor.                                                     \n
 *          2. From source node to A, there is only one path, and A is directly user for source node,                 \n
 *             so the tensor of their edge is a short-term tensor.
 *
 * @param node Source node.
 * @param mng Graph manager.
 * @return OutPosLinkList Vector[Tuple(target node, input positions of target node for edge, edge type)].
 */
OutPosLinkList AutoRecompute::JudegeTargetAndCaptureSource(const AnfNodePtr &node, const FuncGraphManagerPtr &mng) {
  auto [direct_users, user_edge_pos, max_topo_user_index] = GetValidUsers(node, mng);
  OutPosLinkList target_link_infos;
  OrderedSet<AnfNodePtr> long_term_users;
  // If the number of direct users is less than 2, there will no side way to its user....
  if (direct_users.size() >= 2) {
    OrderedMap<AnfNodePtr, AnfNodePtrList> link_paths =
      CollectLinkPaths(topo_indice_, direct_users, max_topo_user_index, mng);
    for (const auto &[source, paths] : link_paths) {
      for (auto target : paths) {
        if (target != source) {
          (void)target_link_infos.emplace_back(target, user_edge_pos[target], EdgeLifeTimeType::LongTerm);
          (void)long_term_users.insert(target);
        }
      }
    }
  }

  // Direct users include long term users and short term users.
  // If the short term user is graph kernel composite node, it may be absorb and reduce the local peak memory.
  for (const auto &user : direct_users) {
    if (long_term_users.count(user) == 0 && common::AnfAlgo::IsGraphKernel(user)) {
      (void)target_link_infos.emplace_back(user, user_edge_pos[user], EdgeLifeTimeType::ShortTerm);
    }
  }

  RecomputeLinkEdgeLog(node, direct_users, target_link_infos);
  return target_link_infos;
}

/**
 * @brief Get position of edge tensor between source node and target node.     \n
 *        For example, giving target node and edge position 0, will return 1:  \n
 *          source node                                                        \n
 *          [0] [1] [2]  <- output position                                    \n
 *               |                                                             \n
 *               |                                                             \n
 *              /                                                              \n
 *            [0] [1]    <- input position                                     \n
 *          target node
 *
 * @param target Target node.
 * @param pos The input position of target node for edge.
 * @return int The output position of source node for edge.
 */
int AutoRecompute::GetSourceLinkOutPos(const AnfNodePtr &target, int pos) const {
  // If the input is get-item, than use get-item's index, otherwise zero.
  auto cnode = target->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto prenode = cnode->input(IntToSize(pos));
  if (!IsPrimitiveCNode(prenode, prim::kPrimTupleGetItem)) {
    return 0;
  }

  auto get_item_cnode = prenode->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(get_item_cnode);
  auto value_input = get_item_cnode->input(kInputNodeOutputIndexInTupleGetItem);
  MS_EXCEPTION_IF_NULL(value_input);
  auto value_node = value_input->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  return static_cast<int>(GetValue<int64_t>(value_node->value()));
}

MemorySize AutoRecompute::SelectThreshold(EdgeLifeTimeType type) const {
  MemorySize threshold = 0;
  auto local_peak_th = local_peak_threshold_ == 0 ? std::numeric_limits<MemorySize>::max() : local_peak_threshold_;
  auto lifetime_th = lifetime_threshold_ == 0 ? std::numeric_limits<MemorySize>::max() : lifetime_threshold_;
  if (type == EdgeLifeTimeType::ShortTerm) {
    threshold = local_peak_th;
  } else if (type == EdgeLifeTimeType::LongTerm) {
    threshold = std::min(local_peak_th, lifetime_th);
  }

  return threshold;
}

bool AutoRecompute::IsThresholdDefaultValue() const {
  if (local_peak_threshold_ == 0 && lifetime_threshold_ == 0) {
    return true;
  }
  return false;
}

/**
 * @brief Find recompute candidates(source node, target node, edge and its type) in func_graph. \n
 *        Result will be add to candidates_.
 *
 * @param func_graph
 */
void AutoRecompute::FindCandidates(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  candidates_.clear();

  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);

  auto topo_nodes = TopoSort(func_graph->get_return());
  // Topo indice is use to early stop in predecessor check.
  for (size_t i = 0; i < topo_nodes.size(); ++i) {
    (void)topo_indice_.emplace(topo_nodes[i], i);
  }

  // Candidate condition:
  // 1. Judge current node can see its graph_kernel input with other input's backward path.
  // 2. Memory variety between split out and origin more than threshold:
  //    `Size(gs_direct_outs_to_gt) - filter(gs_inputs, its) > threshold`.
  for (auto node : topo_nodes) {
    if (!common::AnfAlgo::IsGraphKernel(node)) {
      continue;
    }
    auto target_graphs = JudegeTargetAndCaptureSource(node, mng);
    if (target_graphs.empty()) {
      continue;
    }
    auto node_candidates = FindNodeRecomputeCandidates(node, target_graphs, mng);
    // Delete duplicated link.
    for (const auto &[source, target_and_link] : node_candidates) {
      for (const auto &[target, link] : target_and_link) {
        candidates_.push_back({source, target, link.first, link.second});
      }
    }
  }

  RecomputeCandidatesLog(candidates_);
}

/**
 * @brief Find recompute candidates for node as source graph.
 *
 * @param node Source graph node.
 * @param target_graphs Vector of [AnfNodePtr, std::vector<int>, EdgeLifeTimeType].
 * @param mng Manager of main graph(which contains this node).
 * @return AutoRecompute::NodeRecomputeCandidates
 */
AutoRecompute::NodeRecomputeCandidates AutoRecompute::FindNodeRecomputeCandidates(const AnfNodePtr &node,
                                                                                  const OutPosLinkList &target_graphs,
                                                                                  const FuncGraphManagerPtr &mng) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(mng);
  NodeRecomputeCandidates node_candidates;
  auto graph_node = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
  MS_EXCEPTION_IF_NULL(graph_node);
  auto nodes = graph_node->nodes();
  if (std::any_of(nodes.cbegin(), nodes.cend(),
                  [](const AnfNodePtr n) { return IsPrimitiveCNode(n, prim::kPrimReduceSum); })) {
    return node_candidates;
  }
  for (auto [gt, gt_in_pos_vec, edge_life_time_type] : target_graphs) {
    MemorySize threshold = SelectThreshold(edge_life_time_type);
    for (auto gt_in_pos : gt_in_pos_vec) {
      MemorySize out_tensor_size =
        static_cast<MemorySize>(AnfAlgo::GetOutputTensorMemSize(node, IntToSize(GetSourceLinkOutPos(gt, gt_in_pos))));
      MemorySize absorb_input_tensor_size = 0;
      for (auto input : Filter(node, gt, gt_in_pos, mng)) {
        absorb_input_tensor_size += static_cast<MemorySize>(AnfAlgo::GetOutputTensorMemSize(input, 0));
      }
      auto gt_cnode = gt->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(gt_cnode);
      auto edge = gt_cnode->input(IntToSize(gt_in_pos));

      MS_LOG(DEBUG) << "Recompute case: GS(" << node->fullname_with_scope() << ") -> GT(" << gt->fullname_with_scope()
                    << ") with Edge(" << edge->fullname_with_scope() << "<" << edge_life_time_type << ">.";

      if (out_tensor_size < absorb_input_tensor_size) {
        MS_LOG(DEBUG) << " ==> Skip this case because memory reduction.";
        continue;
      }

      auto memory_increment = out_tensor_size - absorb_input_tensor_size;
      MS_LOG(DEBUG) << " ==> Threshold: " << threshold << ", Out Tensor[" << out_tensor_size << "] - Absort Tensor["
                    << absorb_input_tensor_size << "] = " << memory_increment;

      if (memory_increment > threshold) {
        if (node_candidates[node].find(gt) == node_candidates[node].end()) {
          node_candidates[node][gt] = {edge_life_time_type, AnfNodePtrList{}};
        }
        // Only add getitem node as edge, if GS is single output node, there will be no edges.
        if (IsPrimitiveCNode(edge, prim::kPrimTupleGetItem)) {
          node_candidates[node][gt].second.push_back(edge);
        }
      }
    }
  }
  return node_candidates;
}

void AutoRecompute::RecomputeLinkEdgeLog(const AnfNodePtr &node, const OrderedSet<AnfNodePtr> &direct_users,
                                         const OutPosLinkList &target_link_infos) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Recompute users for node: " << node->fullname_with_scope();
  for (const auto &direct_user : direct_users) {
    MS_LOG(DEBUG) << "  └─ " << direct_user->fullname_with_scope();
  }

  MS_LOG(DEBUG) << "Edge Link relation: ";
  for (const auto &[target, tartget_in_index, life_type] : target_link_infos) {
    MS_EXCEPTION_IF_NULL(target);
    MS_LOG(DEBUG) << "  └[" << tartget_in_index << "|<" << life_type
                  << ">]─> Link to: " << target->fullname_with_scope();
  }
}

void AutoRecompute::RecomputeCandidatesLog(const std::vector<Candidate> &candidates) const {
  MS_LOG(INFO) << "Recompute candidates: ";
  for (auto candidate : candidates) {
    MS_LOG(INFO) << "  └─ GS: " << candidate.source_graph->fullname_with_scope();
    MS_LOG(INFO) << "  └─ GT: " << candidate.target_graph->fullname_with_scope();
    for (auto edge : candidate.recompute_edges) {
      MS_LOG(INFO) << "    └─[Edge]─> " << edge->fullname_with_scope();
    }
  }
}

std::vector<Candidate> CSRRecompute::Run(const FuncGraphPtr &func_graph) {
  FindCandidates(func_graph);
  return candidates_;
}

bool CSRRecompute::CheckPrimitiveInput(AnfNodePtr base, const PrimitivePtr &prim_type) const {
  std::deque<AnfNodePtr> q{base};
  std::set<AnfNodePtr> visited;
  while (!q.empty()) {
    auto node = q.front();
    q.pop_front();
    if (visited.count(node) > 0) {
      continue;
    }
    (void)visited.insert(node);
    if (IsPrimitiveCNode(node, prim_type)) {
      return true;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    auto inputs = cnode->inputs();
    (void)q.insert(q.begin(), inputs.begin(), inputs.end());
  }
  return false;
}

AutoRecompute::NodeRecomputeCandidates CSRRecompute::FindNodeRecomputeCandidates(const AnfNodePtr &node,
                                                                                 const OutPosLinkList &target_graphs,
                                                                                 const FuncGraphManagerPtr &mng) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(mng);
  NodeRecomputeCandidates node_candidates;
  auto graph_node = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
  MS_EXCEPTION_IF_NULL(graph_node);
  // subgraphs outputting UnsortedSegmentSum or CSRReduceSum along with other ops
  // (likely the result of Gather), or containing CSRDiv without outputting
  // UnsortedSegmentSum or CSRReduceSum, are selected as candidates for recompute.
  auto TargetTail = [](const AnfNodePtr n) {
    return IsPrimitiveCNode(n, prim::kPrimUnsortedSegmentSum) || IsPrimitiveCNode(n, prim::kPrimCSRReduceSum);
  };
  auto TargetHead = [](const AnfNodePtr n) { return IsPrimitiveCNode(n, prim::kPrimCSRDiv); };
  auto return_node = graph_node->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  auto return_cnode = return_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(return_cnode);
  auto return_inputs = return_cnode->inputs();
  auto return_tup = return_inputs[1]->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(return_tup);
  auto tuple_inputs = return_tup->inputs();
  std::set<size_t> candidate_idx;
  if (std::any_of(tuple_inputs.cbegin(), tuple_inputs.cend(), TargetTail)) {
    for (size_t i = 1; i < tuple_inputs.size(); ++i) {
      if (!TargetTail(tuple_inputs[i])) {
        (void)candidate_idx.insert(i - 1);
      }
    }
  } else if (std::any_of(tuple_inputs.cbegin(), tuple_inputs.cend(), TargetHead)) {
    for (size_t i = 1; i < tuple_inputs.size(); ++i) {
      if (CheckPrimitiveInput(tuple_inputs[i], prim::kPrimCSRDiv)) {
        (void)candidate_idx.insert(i - 1);
      }
    }
  }
  if (candidate_idx.empty()) {
    return node_candidates;
  }
  for (size_t i = 0; i < target_graphs.size(); ++i) {
    AnfNodePtr gt;
    std::vector<int> gt_in_pos_vec;
    std::tie(gt, gt_in_pos_vec, std::ignore) = target_graphs[i];
    for (auto gt_in_pos : gt_in_pos_vec) {
      auto gt_cnode = gt->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(gt_cnode);
      auto edge = gt_cnode->input(IntToSize(gt_in_pos));
      if (!IsPrimitiveCNode(edge, prim::kPrimTupleGetItem)) {
        continue;
      }
      auto edge_cnode = edge->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(edge_cnode);
      auto tuple_idx = common::AnfAlgo::GetTupleGetItemOutIndex(edge_cnode);
      if (candidate_idx.count(tuple_idx) > 0) {
        node_candidates[node][gt].second.push_back(edge);
      }
    }
  }
  return node_candidates;
}

std::pair<FuncGraphPtr, AnfNodePtrList> GraphKernelRecompute::CloneGraph(const CNodePtr &source_graph,
                                                                         const AnfNodePtrList &recompute_edges) const {
  MS_EXCEPTION_IF_NULL(source_graph);
  auto gs = common::AnfAlgo::GetCNodeFuncGraphPtr(source_graph);
  MS_EXCEPTION_IF_NULL(gs);
  AnfNodePtrList inputs(source_graph->inputs().begin() + 1, source_graph->inputs().end());
  auto new_funcgraph = BasicClone(gs);
  auto output_node = new_funcgraph->output()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(output_node);
  if (!IsPrimitiveCNode(output_node, prim::kPrimMakeTuple)) {
    return {new_funcgraph, inputs};
  }
  // remove outputs that not in recompute edges.
  AnfNodePtrList new_outputs;
  for (auto &edge : recompute_edges) {
    auto idx = GetGetitemIndex(edge);
    new_outputs.push_back(GetOutput(new_funcgraph, LongToSize(idx)));
  }
  if (new_outputs.size() + 1 == output_node->size()) {
    return {new_funcgraph, inputs};
  }
  (void)new_outputs.insert(new_outputs.cbegin(), output_node->input(0));
  auto new_output_node = new_funcgraph->NewCNode(new_outputs);
  // use the old abstract, since the new_funcgraph will be deleted in later process.
  new_output_node->set_abstract(output_node->abstract());
  new_output_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  new_funcgraph->set_output(new_output_node);
  ElimRedundantInputsAndGraphParameters(new_funcgraph, &inputs);
  return {new_funcgraph, inputs};
}

void GraphKernelRecompute::LinkIntoTargetFuncGraph(
  const Candidate &candidate, const FuncGraphPtr &cloned_func, const AnfNodePtrList &cloned_inputs,
  const std::function<std::pair<bool, size_t>(const Candidate &, const AnfNodePtr &)> &edge_match_func) const {
  auto cloned_nodes = TopoSort(cloned_func->get_return());
  auto gt = common::AnfAlgo::GetCNodeFuncGraphPtr(candidate.target_graph);
  MS_EXCEPTION_IF_NULL(gt);
  auto mng = gt->manager();
  if (mng == nullptr) {
    mng = Manage(gt, true);
    gt->set_manager(mng);
  }

  // link the outputs to gt
  auto gt_node = candidate.target_graph->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(gt_node);
  AnfNodePtrList new_parameters;
  AnfNodePtrList new_inputs;
  auto &params = gt->parameters();
  for (size_t i = 0; i < params.size(); i++) {
    // if the parameter is a recompute edge, then links the param to the cloned_func's output.
    auto [is_match, out_index] = edge_match_func(candidate, gt_node->input(i + 1));
    if (is_match) {
      (void)mng->Replace(params[i], GetOutput(cloned_func, out_index));
    } else {
      new_parameters.push_back(params[i]);
      new_inputs.push_back(gt_node->input(i + 1));
    }
  }

  // add new parameters
  auto &cloned_func_params = cloned_func->parameters();
  for (size_t i = 0; i < cloned_func_params.size(); i++) {
    auto iter = std::find(new_inputs.begin(), new_inputs.end(), cloned_inputs[i]);
    if (iter != new_inputs.end()) {
      auto idx = iter - new_inputs.begin();
      (void)cloned_func->manager()->Replace(cloned_func_params[i], new_parameters[LongToSize(idx)]);
    } else {
      new_parameters.push_back(gt->add_parameter());
      new_inputs.push_back(cloned_inputs[i]);
      (void)cloned_func->manager()->Replace(cloned_func_params[i], new_parameters.back());
    }
  }

  // reset the func_graph for cloned_nodes.
  for (auto &node : cloned_nodes) {
    if (node->isa<CNode>()) {
      node->set_func_graph(gt);
    }
  }
  AnfNodePtrList new_node_inputs = {gt_node->input(0)};
  (void)new_node_inputs.insert(new_node_inputs.cend(), new_inputs.cbegin(), new_inputs.cend());
  gt->set_parameters(new_parameters);
  gt_node->set_inputs(new_node_inputs);
  AnfNodePtrList outputs;
  kernel::GetFuncGraphOutputNodes(gt, &outputs);
  gt_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  SetNewKernelInfo(gt_node, gt, new_inputs, outputs);
  mng->RemoveRoots();
  mng->KeepRoots({gt});
}

void GraphKernelRecompute::Process(const Candidate &candidate) const {
  FuncGraphPtr new_funcgraph;
  AnfNodePtrList inputs;
  std::function<std::pair<bool, size_t>(const Candidate &, const AnfNodePtr &)> edge_match_func;
  if (candidate.recompute_edges.empty()) {
    // single output, clone the whole source_graph.
    auto gs = common::AnfAlgo::GetCNodeFuncGraphPtr(candidate.source_graph);
    MS_EXCEPTION_IF_NULL(gs);
    new_funcgraph = BasicClone(gs);
    auto source_cnode = candidate.source_graph->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(source_cnode);
    (void)inputs.insert(inputs.cend(), source_cnode->inputs().cbegin() + 1, source_cnode->inputs().cend());
    edge_match_func = [](const Candidate &match_candidate, const AnfNodePtr &to_match) -> std::pair<bool, size_t> {
      if (match_candidate.source_graph == to_match) {
        return std::make_pair(true, 0);
      }
      return std::make_pair(false, 0);
    };
  } else {
    std::tie(new_funcgraph, inputs) = CloneGraph(candidate.source_graph->cast<CNodePtr>(), candidate.recompute_edges);
    edge_match_func = [](const Candidate &match_candidate, const AnfNodePtr &to_match) -> std::pair<bool, size_t> {
      auto iter = std::find(match_candidate.recompute_edges.begin(), match_candidate.recompute_edges.end(), to_match);
      if (iter != match_candidate.recompute_edges.end()) {
        auto out_index = iter - match_candidate.recompute_edges.begin();
        return std::make_pair(true, LongToSize(out_index));
      }
      return std::make_pair(false, 0);
    };
  }

  auto mng = new_funcgraph->manager();
  if (mng == nullptr) {
    mng = Manage(new_funcgraph, true);
    new_funcgraph->set_manager(mng);
  }

  if (common::AnfAlgo::IsGraphKernel(candidate.target_graph)) {
    // the target graph is a GraphKernel, push the new_funcgraph into the target graph.
    LinkIntoTargetFuncGraph(candidate, new_funcgraph, inputs, edge_match_func);
  } else {
    // The target graph is not a GraphKernel, build the new_funcgraph to a CNode.
    MS_LOG(WARNING) << "Target node " << candidate.target_graph->fullname_with_scope()
                    << " is not a graph kernel node, cannot absort the link edge!";
    return;
  }
}

bool GraphKernelRecompute::DoRun(const FuncGraphPtr &func_graph, bool use_csr) {
  int repeat_times = 2;
  while ((repeat_times--) != 0) {
    if (use_csr) {
      CSRRecompute csr_recompute;
      candidates_ = csr_recompute.Run(func_graph);
    } else {
      AutoRecompute auto_recompute;
      candidates_ = auto_recompute.Run(func_graph);
    }
    if (candidates_.empty()) {
      return false;
    }
    auto mng = func_graph->manager();
    MS_EXCEPTION_IF_NULL(mng);
    for (auto &c : candidates_) {
      if (!common::AnfAlgo::IsGraphKernel(c.target_graph)) {
        continue;
      }
      std::ostringstream oss;
      for (auto &e : c.recompute_edges) {
        if (!IsPrimitiveCNode(e, prim::kPrimTupleGetItem)) {
          MS_LOG(EXCEPTION) << "The edge should be GetItem but got " << e->fullname_with_scope();
        }
        oss << e->fullname_with_scope() << ", ";
      }
      MS_LOG(INFO) << "Clone " << c.source_graph->fullname_with_scope() << " to "
                   << c.target_graph->fullname_with_scope() << ", edges [" << oss.str() << "]";
      Process(c);
    }
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return true;
}

bool GraphKernelRecompute::Run(const FuncGraphPtr &func_graph) {
  bool status = DoRun(func_graph);
  if (GraphKernelFlags::GetInstance().enable_csr_fusion) {
    status = DoRun(func_graph, true) || status;
  }
  return status;
}
}  // namespace mindspore::graphkernel
