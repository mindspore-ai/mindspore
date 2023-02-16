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

#include "common/graph_kernel/parallel_fusion.h"

#include <algorithm>
#include <list>
#include <queue>
#include <unordered_map>
#include <utility>
#include "common/graph_kernel/graph_kernel_flags.h"
#include "kernel/kernel.h"
#include "common/graph_kernel/graph_kernel_helper.h"
#include "kernel/common_utils.h"
#include "frontend/operator/ops.h"
#include "ir/func_graph_cloner.h"
#include "common/graph_kernel/core/update_state_formatter.h"
#include "common/graph_kernel/core/graph_builder.h"

namespace mindspore::graphkernel {
namespace {
// Cuda's parameter table can accept maximum 4KB, so the number of parameters should be less than 512.
constexpr size_t CUDA_PARA_LIMIT = 512;

void ProcessThroughPassCNode(OrderedMap<AnfNodePtr, NodeRelation> *node_rels) {
  std::set<AnfNodePtr> to_be_erased;
  for (const auto &[node, node_rel] : (*node_rels)) {
    if (!IsOneOfPrimitiveCNode(
          node, {prim::kPrimReshape, prim::kPrimExpandDims, prim::kPrimSqueeze, prim::kPrimTupleGetItem})) {
      continue;
    }
    to_be_erased.insert(node);
    for (const auto &pre : node_rel.pres) {
      auto &pre_nexts = (*node_rels)[pre].nexts;
      (void)pre_nexts.erase(node);
      for (const auto &next : node_rel.nexts) {
        (void)pre_nexts.insert(next);
        auto &next_pres = (*node_rels)[next].pres;
        (void)next_pres.erase(node);
        (void)next_pres.insert(pre);
      }
    }
  }
  for (const auto &node : to_be_erased) {
    (void)node_rels->erase(node);
  }
}

void ProcessTailMakeTupleCNode(OrderedMap<AnfNodePtr, NodeRelation> *node_rels) {
  AnfNodePtrList latter_to_be_erased;
  for (auto &[node, node_rel] : (*node_rels)) {
    if (!IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
      continue;
    }

    AnfNodePtrList check_next_list;
    check_next_list.push_back(node);

    bool disinterested = false;
    for (auto &successor : node_rel.nexts) {
      if (!IsPrimitiveCNode(successor, prim::kPrimTupleGetItem)) {
        disinterested = true;
        break;
      }
      check_next_list.push_back(successor);
    }
    if (disinterested) {
      continue;
    }

    if (!std::all_of(check_next_list.cbegin(), check_next_list.cend(),
                     [&node_rels](const AnfNodePtr &n) -> bool { return (*node_rels)[n].nexts.empty(); })) {
      continue;
    }

    latter_to_be_erased.push_back(node);
  }

  // Delete Tail MakeTuple(including its getitem nodes).
  for (const auto &node : latter_to_be_erased) {
    for (auto &pre : (*node_rels)[node].pres) {
      (void)(*node_rels)[pre].nexts.erase(node);
    }

    // Tail MakeTuple is just be consumed by nothing or invalid getitem node.
    for (auto &getitem : (*node_rels)[node].nexts) {
      (void)node_rels->erase(getitem);
    }

    (void)node_rels->erase(node);
  }
}

bool IsSingleInputNode(const OrderedMap<AnfNodePtr, NodeRelation> &node_rels, const AnfNodePtr &node) {
  if (auto iter = node_rels.find(node); iter != node_rels.end() && iter->second.pres.size() == 1) {
    return true;
  }
  return false;
}

bool IsSingleOutputNode(const OrderedMap<AnfNodePtr, NodeRelation> &node_rels, const AnfNodePtr &node) {
  if (auto iter = node_rels.find(node); iter != node_rels.end() && iter->second.nexts.size() == 1) {
    return true;
  }
  return false;
}

bool IsMultiInputsNode(const OrderedMap<AnfNodePtr, NodeRelation> &node_rels, const AnfNodePtr &node) {
  if (auto iter = node_rels.find(node); iter != node_rels.end() && iter->second.pres.size() > 1) {
    return true;
  }
  return false;
}

bool IsMultiOutputsNode(const OrderedMap<AnfNodePtr, NodeRelation> &node_rels, const AnfNodePtr &node) {
  if (auto iter = node_rels.find(node); iter != node_rels.end() && iter->second.nexts.size() > 1) {
    return true;
  }
  return false;
}

bool IsNoInputsNode(const OrderedMap<AnfNodePtr, NodeRelation> &node_rels, const AnfNodePtr &node) {
  if (auto iter = node_rels.find(node); iter != node_rels.end() && iter->second.pres.size() == 0) {
    return true;
  }
  return false;
}

bool IsNoOutputsNode(const OrderedMap<AnfNodePtr, NodeRelation> &node_rels, const AnfNodePtr &node) {
  if (auto iter = node_rels.find(node); iter != node_rels.end() && iter->second.nexts.size() == 0) {
    return true;
  }
  return false;
}

void ProcessLocalStructure(OrderedMap<AnfNodePtr, NodeRelation> *node_rels, std::set<AnfNodePtr> *virtual_noout_nodes,
                           std::set<AnfNodePtr> *ignore_noin_nodes) {
  // 1. Local relation
  // Graph as following left part, relation D->B and D->E(D is a no input node)
  // will make B and E to be multiply inputs node.
  // But for parallel, this local relation can ignore for B and E, which make
  // them be able to be paralleled.
  //
  // ************************************
  // *                                  *
  // * |                    |           *
  // * A   D                A      D    *
  // * |  /|                |     / \   *
  // * | C |                |    C   F  *
  // * |/  /                |    |   |  *
  // * B  F      ====>      B    x   x  *
  // * | /                  |           *
  // * |/                   |           *
  // * E                    E           *
  // * |                    |           *
  // *                                  *
  // ************************************
  AnfNodePtrList no_input_nodes;
  for (const auto &node_rel : *node_rels) {
    auto &node = node_rel.first;
    if (IsNoInputsNode(*node_rels, node)) {
      no_input_nodes.push_back(node);
    }
  }

  std::vector<std::pair<AnfNodePtr, AnfNodePtr>> latter_delete;

  for (const auto &ninode : no_input_nodes) {
    AnfNodePtrList cnexts((*node_rels)[ninode].nexts.cbegin(), (*node_rels)[ninode].nexts.cend());
    for (const auto &n : cnexts) {
      AnfNodePtr serial_tail = ninode;
      AnfNodePtr cur_node = n;
      while (IsSingleInputNode(*node_rels, cur_node) && IsSingleOutputNode(*node_rels, cur_node)) {
        serial_tail = cur_node;
        cur_node = *((*node_rels)[cur_node].nexts.begin());
      }
      (void)latter_delete.emplace_back(serial_tail, cur_node);
    }
  }

  // Delete relation.
  for (const auto &[serial_tail, cur_node] : latter_delete) {
    (void)virtual_noout_nodes->insert(serial_tail);
    (void)ignore_noin_nodes->insert(cur_node);
    (void)(*node_rels)[serial_tail].nexts.erase(cur_node);
    (void)(*node_rels)[cur_node].pres.erase(serial_tail);
    MS_LOG(INFO) << "Process local relation delete relation: " << serial_tail->fullname_with_scope() << " -> "
                 << cur_node->fullname_with_scope();
  }
}

std::tuple<AnfNodePtrList, AnfNodePtrList, AnfNodePtrList, AnfNodePtrList> GetInterestNodeIds(
  const OrderedMap<AnfNodePtr, NodeRelation> &node_rels, const std::set<AnfNodePtr> &virtual_noout_nodes,
  const std::set<AnfNodePtr> &ignore_noin_nodes) {
  AnfNodePtrList multi_inputs_nodes, multi_outputs_nodes, no_input_nodes, no_output_nodes;
  std::list<std::function<void(const AnfNodePtr &)>> func_list = {
    [&node_rels, &multi_inputs_nodes](const AnfNodePtr &node) {
      if (IsMultiInputsNode(node_rels, node)) {
        multi_inputs_nodes.push_back(node);
      }
    },
    [&node_rels, &multi_outputs_nodes](const AnfNodePtr &node) {
      if (IsMultiOutputsNode(node_rels, node)) {
        multi_outputs_nodes.push_back(node);
      }
    },
    [&node_rels, &no_input_nodes, &ignore_noin_nodes](const AnfNodePtr &node) {
      if (IsNoInputsNode(node_rels, node) && ignore_noin_nodes.count(node) == 0) {
        no_input_nodes.push_back(node);
      }
    },
    [&node_rels, &no_output_nodes, &virtual_noout_nodes](const AnfNodePtr &node) {
      if (IsNoOutputsNode(node_rels, node) && virtual_noout_nodes.count(node) == 0) {
        no_output_nodes.push_back(node);
      }
    }};

  for (const auto &node_rel : node_rels) {
    for (const auto &func : func_list) {
      func(node_rel.first);
    }
  }

  return std::make_tuple(multi_inputs_nodes, multi_outputs_nodes, no_input_nodes, no_output_nodes);
}

bool WhiteOpsFilter(const AnfNodePtr &node) { return common::AnfAlgo::IsGraphKernel(node); }

// Parallel cannot work with stitching for now.
bool Parallelizable(const AnfNodePtr &node) { return WhiteOpsFilter(node) && !IsBufferStitchNode(node); }

std::vector<AnfNodePtrList> SearchFromNodes(const AnfNodePtrList &nodes,
                                            const std::function<bool(const AnfNodePtr &)> &filter_func,
                                            const OrderedMap<AnfNodePtr, NodeRelation> &node_rels, bool is_backward,
                                            std::set<AnfNodePtr> *seen) {
  // Start from multi-inputs node, stop on seen node or multi-inputs or multi-outputs nodes.
  // For backward search, the other multi-inputs node can be contained in.
  // For forward search, the other multi-outputs node can be contained in.
  auto get_contain_node_set = [is_backward](const NodeRelation &info) { return is_backward ? info.pres : info.nexts; };
  auto get_exclude_node_set = [is_backward](const NodeRelation &info) { return is_backward ? info.nexts : info.pres; };

  std::vector<AnfNodePtrList> group;
  for (const auto &node : nodes) {
    AnfNodePtrList stream;
    AnfNodePtr n = node;
    for (auto iter = node_rels.find(n);
         seen->count(n) == 0 && iter != node_rels.end() && get_exclude_node_set(iter->second).size() <= 1;
         iter = node_rels.find(n)) {
      if (filter_func(n)) {
        stream.push_back(n);
        (void)seen->insert(n);
      }
      if (get_contain_node_set(iter->second).size() != 1) {
        break;
      }
      n = *(get_contain_node_set(iter->second).cbegin());
    }
    if (stream.size() > 0) {
      group.push_back(stream);
    }
  }

  if (group.size() == 1) {
    for (const auto &drop : group[0]) {
      (void)seen->erase(drop);
    }
    group.clear();
  }

  return group;
}

void SearchStreamFromMultiRelationNode(const AnfNodePtrList &multi_nodes,
                                       const OrderedMap<AnfNodePtr, NodeRelation> &node_rels, bool is_backward,
                                       std::vector<std::vector<AnfNodePtrList>> *groups, std::set<AnfNodePtr> *seen) {
  auto get_related_nodes = [is_backward](const NodeRelation &info) { return is_backward ? info.pres : info.nexts; };
  for (const auto &node : multi_nodes) {
    if (auto iter = node_rels.find(node); iter != node_rels.end()) {
      const auto &pre_nodes = get_related_nodes(iter->second);
      AnfNodePtrList related_nodes(pre_nodes.begin(), pre_nodes.end());
      groups->push_back(SearchFromNodes(related_nodes, Parallelizable, node_rels, is_backward, seen));
    }
  }

  // Erase empty groups.
  for (auto iter = groups->begin(); iter != groups->end();) {
    if (iter->size() == 0) {
      iter = groups->erase(iter);
    } else {
      ++iter;
    }
  }
}

void SearchStreamFromUnidirectionalNode(const AnfNodePtrList &ud_nodes,
                                        const OrderedMap<AnfNodePtr, NodeRelation> &node_rels, bool is_backward,
                                        std::vector<std::vector<AnfNodePtrList>> *groups, std::set<AnfNodePtr> *seen) {
  groups->push_back(SearchFromNodes(ud_nodes, Parallelizable, node_rels, is_backward, seen));

  // Erase empty groups.
  for (auto iter = groups->begin(); iter != groups->end();) {
    if (iter->size() == 0) {
      iter = groups->erase(iter);
    } else {
      ++iter;
    }
  }
}

std::string DumpNode(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  std::stringstream buf;
  buf << (common::AnfAlgo::IsGraphKernel(cnode) ? "[graph]" : "[primitive]") << cnode->fullname_with_scope() << "|"
      << cnode->ToString();
  return buf.str();
}

void DumpParallelGroups(const std::vector<std::vector<AnfNodePtrList>> &groups, const std::string &title = "") {
  MS_LOG(INFO) << "[" << title << "]"
               << "There are " << groups.size() << " parallel groups, their detail is: ";
  int i = 0;
  for (const auto &group : groups) {
    std::stringstream buf;
    buf << "[" << i << " group] " << group.size() << ":\n";
    for (const auto &nodes : group) {
      buf << "  " << nodes.size() << ": [<";
      for (const auto &node : nodes) {
        buf << "(" << DumpNode(node) << ") -> ";
      }
      buf << ">]\n";
    }
    i++;
    MS_LOG(INFO) << buf.str();
  }
}

void DumpParallelFusionDetail(const AnfNodePtrList &source, const AnfNodePtr &target) {
  std::stringstream buf;
  buf << "Parallel fusion detail: ";
  for (const auto &node : source) {
    buf << "(" << DumpNode(node) << ") + ";
  }
  buf << "==>"
      << "(" << DumpNode(target) << ")";
  MS_LOG(INFO) << buf.str();
}

inline bool ParameterLimit(const AnfNodePtrList &nodes) {
  if (nodes.empty()) {
    MS_LOG(EXCEPTION) << "Nodes is empty, can not check condition.";
  }

  bool res = true;
  auto processor_type = AnfAlgo::GetProcessor(nodes[0]);
  if (processor_type == kernel::Processor::CUDA) {
    // The number of inputs and outputs for a valid kernel should be less than cuda's limit.
    size_t para_count = 0;
    for (const auto &node : nodes) {
      para_count += common::AnfAlgo::GetInputTensorNum(node);
      para_count += AnfAlgo::GetOutputTensorNum(node);
    }
    res = para_count <= CUDA_PARA_LIMIT;
  }

  return res;
}

bool ExtraFusionCondition(const AnfNodePtrList &nodes) { return ParameterLimit(nodes); }
}  // namespace

OrderedMap<AnfNodePtr, NodeRelation> ParallelOpFusion::GenAnalysisGraph(const AnfNodePtrList &nodes) {
  // Based on anf node input information, build a simple graph for latter analyzation.
  OrderedMap<AnfNodePtr, NodeRelation> node_rels;
  auto get_info = [&node_rels](const AnfNodePtr &node) {
    if (node_rels.count(node) == 0) {
      (void)node_rels.emplace(node, NodeRelation());
    }
    return &(node_rels[node]);
  };

  for (const auto &node : nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }

    auto prior_node = get_info(node);
    for (const auto &input : (node->cast<CNodePtr>())->inputs()) {
      if (!input->isa<CNode>()) {
        continue;
      }
      auto behind_node = get_info(input);
      (void)prior_node->pres.insert(input);
      (void)behind_node->nexts.insert(node);
    }
  }

  ProcessThroughPassCNode(&node_rels);
  ProcessTailMakeTupleCNode(&node_rels);
  ProcessLocalStructure(&node_rels, &virtual_noout_nodes_, &ignore_noin_nodes_);

  return node_rels;
}

std::vector<std::vector<AnfNodePtrList>> ParallelOpFusion::SearchParallelGroups(
  const OrderedMap<AnfNodePtr, NodeRelation> &node_rels) {
  // Get interesting nodes: multi-inputs nodes, multi-outputs nodes, no input nodes and no output nodes.
  auto [mul_ins_nodes, mul_outs_nodes, no_in_nodes, no_out_nodes] =
    GetInterestNodeIds(node_rels, virtual_noout_nodes_, ignore_noin_nodes_);

  // Get streams and group them
  std::set<AnfNodePtr> seen;
  std::vector<std::vector<AnfNodePtrList>> groups;

  SearchStreamFromMultiRelationNode(mul_ins_nodes, node_rels, true, &groups, &seen);
  SearchStreamFromUnidirectionalNode(no_out_nodes, node_rels, true, &groups, &seen);
  SearchStreamFromMultiRelationNode(mul_outs_nodes, node_rels, false, &groups, &seen);
  SearchStreamFromUnidirectionalNode(no_in_nodes, node_rels, false, &groups, &seen);

  DumpParallelGroups(groups, "Dependency Analyze");
  return groups;
}

std::tuple<AnfNodePtrList, std::vector<int>> ParallelOpFusion::GetAvaliableNodesByOffset(
  int start, const std::vector<size_t> &offsets, const std::vector<bool> &used, const AnfNodePtrList &nodes,
  const std::set<int> &excludes) const {
  // Get unused nodes by offset index, the result will contain the node with start index.
  int node_limit = static_cast<int>(nodes.size());
  if (start >= node_limit) {
    MS_LOG(EXCEPTION) << "Index offset should be less than the limit of given nodes " << node_limit << ", but got "
                      << start;
  }
  AnfNodePtrList target_nodes = {nodes[IntToSize(start)]};
  std::vector<int> valid_indices;
  std::vector<size_t> unused;
  for (size_t i = IntToSize(start); i < used.size(); ++i) {
    if (!used[i] && excludes.count(i) == 0) {
      unused.push_back(i);
    }
  }
  size_t limit = unused.size();
  for (auto offset : offsets) {
    if (offset >= limit) {
      MS_LOG(EXCEPTION) << "Index offset should be less than the limit of unused nodes " << limit << ", but got "
                        << offset;
    }
    if (SizeToInt(unused[offset]) >= node_limit) {
      MS_LOG(EXCEPTION) << "Index offset should be less than the limit of nodes " << node_limit << ", but got "
                        << unused[offset];
    }
    valid_indices.push_back(unused[offset]);
    target_nodes.push_back(nodes[unused[offset]]);
  }

  return std::make_tuple(target_nodes, valid_indices);
}

std::tuple<std::vector<bool>, std::vector<ParallelInfo>> ParallelOpFusion::DoSearchInSortedCandidates(
  size_t origin_size, const AnfNodePtrList &candidates, std::map<AnfNodePtr, int> *origin_indices,
  std::map<AnfNodePtr, int> *sorted_indices) {
  auto get_index = [](std::map<AnfNodePtr, int> *indices, const AnfNodePtr &node) -> int {
    MS_EXCEPTION_IF_NULL(node);
    if (indices->find(node) == indices->end()) {
      MS_LOG(EXCEPTION) << "There is no index record for node " << node->ToString();
    }
    return (*indices)[node];
  };

  std::vector<ParallelInfo> parallel_infos;
  std::vector<bool> origin_candidates_used(origin_size, false);
  std::vector<bool> sorted_candidates_used(candidates.size(), false);
  size_t offset;
  for (size_t i = 0; i < candidates.size(); i += offset + 1) {
    offset = 0;
    if (sorted_candidates_used[i]) {
      continue;
    }

    int max_benefit = 0;
    ParallelInfo best_parallel_info;
    size_t unused_num = 0;
    for (size_t j = i + 1; j < sorted_candidates_used.size(); ++j) {
      unused_num += sorted_candidates_used[j] ? 0 : 1;
    }
    if (unused_num < 1) {
      break;
    }

    unused_num = std::min(unused_num, config_.max_num_for_fuse() - 1);

    size_t begin = 1, end = unused_num;
    while (begin <= end) {
      size_t mid = (begin + end) / 2;
      std::vector<size_t> tc(mid);
      for (size_t idx = 0; idx < mid; idx++) {
        tc[idx] = idx + 1;
      }
      AnfNodePtrList other_candidates;
      std::tie(other_candidates, std::ignore) =
        GetAvaliableNodesByOffset(SizeToInt(i), tc, sorted_candidates_used, candidates, std::set<int>());
      if (ExtraFusionCondition(other_candidates)) {
        int benefit;
        std::tie(std::ignore, benefit, std::ignore) = cost_model_ptr_->CalFuseInfo(other_candidates);
        if (benefit > 0) {
          begin = mid + 1;
          continue;
        }
      }
      end = mid - 1;
    }

    if (begin > 1) {
      std::vector<size_t> tc(begin - 1);
      for (size_t idx = 0; idx < begin - 1; idx++) {
        tc[idx] = idx + 1;
      }
      AnfNodePtrList other_candidates;
      std::tie(other_candidates, std::ignore) =
        GetAvaliableNodesByOffset(SizeToInt(i), tc, sorted_candidates_used, candidates, std::set<int>());
      auto [dim_infos, benefit, fusion_info] = cost_model_ptr_->CalFuseInfo(other_candidates);
      if (benefit <= 0) {
        MS_LOG(EXCEPTION) << "Internal error in candidate search! benefit should be greater than 0, but got "
                          << benefit;
      }
      max_benefit = benefit;
      best_parallel_info = ParallelInfo(other_candidates, dim_infos, fusion_info);
      offset = begin - 1;
    }

    if (max_benefit > 0) {
      parallel_infos.push_back(best_parallel_info);
      for (const auto &node : best_parallel_info.nodes()) {
        sorted_candidates_used[IntToSize(get_index(sorted_indices, node))] = true;
        origin_candidates_used[IntToSize(get_index(origin_indices, node))] = true;
      }
    }
  }

  // Current nodes is not suitable to fuse, so pop first node to try other fusion possibility.
  if (parallel_infos.size() == 0) {
    origin_candidates_used[IntToSize(get_index(origin_indices, candidates[parallel_infos.size()]))] = true;
  }

  return std::make_tuple(origin_candidates_used, parallel_infos);
}

std::tuple<std::vector<bool>, std::vector<ParallelInfo>> ParallelOpFusion::SearchFuseNodesInCandidates(
  const AnfNodePtrList &cs) {
  std::map<AnfNodePtr, int> origin_indices;
  std::vector<size_t> indices;
  for (size_t i = 0; i < cs.size(); ++i) {
    if (cs[i]) {
      origin_indices[cs[i]] = SizeToInt(i);
      indices.push_back(i);
    }
  }

  // A calculated heavy node can cover more lighter nodes' cost, so sort them first.
  std::map<size_t, int64_t> cal_amounts;
  for (auto id : indices) {
    cal_amounts[id] = cost_model_ptr_->GetNodeCalAmount(cs[id]);
  }
  std::sort(indices.begin(), indices.end(),
            [&cal_amounts](size_t a, size_t b) { return cal_amounts[a] > cal_amounts[b]; });

  AnfNodePtrList candidates;
  for (size_t i = 0; i < indices.size(); ++i) {
    candidates.push_back(cs[indices[i]]);
  }

  std::map<AnfNodePtr, int> sorted_indices;
  for (size_t i = 0; i < candidates.size(); ++i) {
    sorted_indices[candidates[i]] = SizeToInt(i);
  }

  return DoSearchInSortedCandidates(cs.size(), candidates, &origin_indices, &sorted_indices);
}

void ParallelOpFusion::SearchFuseNodesInParallelGroup(const std::vector<AnfNodePtrList> &group,
                                                      std::vector<ParallelInfo> *parallel_infos) {
  std::vector<AnfNodePtrList::const_iterator> tails;
  std::vector<AnfNodePtrList::const_iterator> ended;
  for (const auto &node_list : group) {
    tails.push_back(node_list.begin());
    ended.push_back(node_list.end());
  }
  auto get_candidates = [&tails, &ended]() {
    AnfNodePtrList candidates;
    for (size_t id = 0; id < tails.size(); ++id) {
      candidates.push_back(tails[id] != ended[id] ? *tails[id] : AnfNodePtr());
    }
    return candidates;
  };
  auto update_tails = [&tails](const std::vector<bool> &used) {
    if (used.size() != tails.size()) {
      MS_LOG(EXCEPTION) << "Judged nodes size is different from left ones size: " << used.size() << " vs "
                        << tails.size();
    }
    for (size_t id = 0; id < used.size(); ++id) {
      if (used[id]) {
        ++tails[id];
      }
    }
  };
  auto valid_candidate_num = [](const AnfNodePtrList &cs) {
    return std::count_if(cs.begin(), cs.end(), [](const AnfNodePtr &n) { return n != nullptr; });
  };

  auto candidates = get_candidates();
  while (valid_candidate_num(candidates) > 1) {
    auto [used, fnds] = SearchFuseNodesInCandidates(candidates);
    (void)std::transform(fnds.cbegin(), fnds.cend(), std::back_insert_iterator(*parallel_infos),
                         [](const ParallelInfo &pi) { return pi; });
    update_tails(used);
    candidates = get_candidates();
  }
}

std::vector<ParallelInfo> ParallelOpFusion::SearchFusableParallelCNodes(
  const std::vector<std::vector<AnfNodePtrList>> &groups) {
  // Find core-fusable groups with cost model.
  std::vector<ParallelInfo> parallel_infos;
  for (const auto &group : groups) {
    SearchFuseNodesInParallelGroup(group, &parallel_infos);
  }

  return parallel_infos;
}

void ParallelOpFusion::SetFusedParallelOpAttrToReturnNode(const ParallelInfo &parallel_info) {
  AnfNodePtr attach_node;
  // Dim info should be attach to each segment's output.
  for (size_t i = 0; i < parallel_info.GetSize(); ++i) {
    const auto &fuse_nodes = parallel_info.nodes();
    std::vector<size_t> info = {i, std::dynamic_pointer_cast<CommonDimInfo>(parallel_info.dims()[i])->dim_info()};
    if (!common::AnfAlgo::IsGraphKernel(fuse_nodes[i])) {
      attach_node = fuse_nodes[i];
      SetNodeAttrSafely(kAttrParallelDimInfo, MakeValue<std::vector<size_t>>(info), fuse_nodes[i]);
    } else {
      auto node_g = GetValueNode<FuncGraphPtr>((fuse_nodes[i]->cast<CNodePtr>())->input(0));
      auto out_node = node_g->output();
      if (IsPrimitiveCNode(out_node, prim::kPrimMakeTuple)) {
        auto inputs = out_node->cast<CNodePtr>()->inputs();
        for (size_t j = 1; j < inputs.size(); ++j) {
          SetNodeAttrSafely(kAttrParallelDimInfo, MakeValue<std::vector<size_t>>(info), inputs[j]);
        }
        attach_node = inputs[1];
      } else {
        attach_node = out_node;
        SetNodeAttrSafely(kAttrParallelDimInfo, MakeValue<std::vector<size_t>>(info), out_node);
      }
    }
  }

  // Fusion info is ok to attach to one of the segments.
  SetFusionInfoAttrToNode(attach_node, parallel_info);
}

void ParallelOpFusion::SetFusionInfoAttrToNode(const AnfNodePtr &node, const ParallelInfo &parallel_info) {
  auto fusion_type = parallel_info.fusion_info()->FusionType();
  common::AnfAlgo::SetNodeAttr(kAttrParallelFusionType, MakeValue<std::string>(fusion_type), node);
  if (parallel_info.fusion_info()->ExistTypeInfo()) {
    if (auto pipeline_fusion = std::dynamic_pointer_cast<BlockPipelineFusionInfo>(parallel_info.fusion_info())) {
      common::AnfAlgo::SetNodeAttr(kAttrParallelTypeInfo,
                                   MakeValue<std::vector<std::vector<int>>>(pipeline_fusion->PipelineIds()), node);
    }
  }
}

bool ParallelOpFusion::CreateParallelOpSubGraphs(const std::vector<ParallelInfo> &parallel_infos,
                                                 const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  bool changed = false;

  for (size_t i = 0; i < parallel_infos.size(); ++i) {
    const auto &fuse_nodes = parallel_infos[i].nodes();
    if (fuse_nodes.size() <= 1) {
      continue;
    }
    changed = true;
    SetFusedParallelOpAttrToReturnNode(parallel_infos[i]);
    auto sg_node = ReplaceNodesWithGraphKernelNode(fuse_nodes, kernel_graph, "parallel");
    common::AnfAlgo::SetNodeAttr(kAttrCompositeType, MakeValue("parallel_fusion"), sg_node);
    DumpParallelFusionDetail(fuse_nodes, sg_node);
  }

  return changed;
}

std::set<AnfNodePtr> CollectCapturedNodes(const std::vector<ParallelInfo> &infos) {
  std::set<AnfNodePtr> captured;
  (void)std::for_each(infos.cbegin(), infos.cend(), [&captured](const ParallelInfo &info) {
    captured.insert(info.nodes().cbegin(), info.nodes().cend());
  });
  return captured;
}

std::vector<std::vector<AnfNodePtrList>> GetParallelGroupsByBfs(const OrderedMap<AnfNodePtr, NodeRelation> &node_rels,
                                                                const std::set<AnfNodePtr> &exclude) {
  std::vector<std::vector<AnfNodePtrList>> groups;
  // BFS
  std::queue<AnfNodePtr> node_que;
  std::unordered_map<AnfNodePtr, int> outdegrees;
  for (const auto &[node, ref] : node_rels) {
    outdegrees[node] = SizeToInt(ref.nexts.size());
    if (outdegrees[node] == 0) {
      node_que.push(node);
    }
  }

  int total_node_num = SizeToInt(node_rels.size());
  while (!node_que.empty()) {
    std::vector<AnfNodePtrList> group;
    int node_size = SizeToInt(node_que.size());
    while (node_size != 0) {
      node_size--;
      auto node = node_que.front();
      node_que.pop();
      if (exclude.count(node) == 0 && Parallelizable(node)) {
        (void)group.emplace_back(AnfNodePtrList({node}));
      }
      --total_node_num;
      auto iter = node_rels.find(node);
      if (iter == node_rels.end()) {
        MS_LOG(EXCEPTION) << "Internal error in node relationship!";
      }
      for (const auto &pre : iter->second.pres) {
        if (--outdegrees[pre] == 0) {
          node_que.push(pre);
        }
      }
    }
    if (!group.empty()) {
      groups.push_back(group);
    }
  }

  if (total_node_num > 0) {
    MS_LOG(EXCEPTION) << "There is circle in analyze graph!";
  }
  DumpParallelGroups(groups, "BFS");
  return groups;
}

bool ParallelOpFusion::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  parallel_level_ = GraphKernelFlags::GetInstance().parallel_ops_level;

  (void)std::make_shared<ShrinkUpdateState>()->Run(graph);
  auto kernel_graph = graph->cast<std::shared_ptr<session::KernelGraph>>();
  MS_EXCEPTION_IF_NULL(kernel_graph);

  cost_model_ptr_ = ParellelCostModelWarehouse::Instance().GetParallelCostModel(target_);
  MS_EXCEPTION_IF_NULL(cost_model_ptr_);

  auto nodes = TopoSort(kernel_graph->get_return());
  std::reverse(nodes.begin(), nodes.end());

  auto node_rels = GenAnalysisGraph(nodes);
  auto groups = SearchParallelGroups(node_rels);
  auto parallel_infos = SearchFusableParallelCNodes(groups);

  // Search in BFS for left nodes.
  if (parallel_level_ > 0) {
    auto exclued_nodes = CollectCapturedNodes(parallel_infos);
    auto groups_bfs = GetParallelGroupsByBfs(node_rels, exclued_nodes);
    auto bfs_parallel_infos = SearchFusableParallelCNodes(groups_bfs);

    (void)parallel_infos.insert(parallel_infos.cend(), bfs_parallel_infos.cbegin(), bfs_parallel_infos.cend());
  }

  // Create core-fuse subgraph and change origin graph.
  bool changed = CreateParallelOpSubGraphs(parallel_infos, kernel_graph);
  (void)std::make_shared<SpreadUpdateState>()->Run(graph);
  return changed;
}
}  // namespace mindspore::graphkernel
