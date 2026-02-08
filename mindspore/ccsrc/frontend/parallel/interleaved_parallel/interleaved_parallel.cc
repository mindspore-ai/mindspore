/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/interleaved_parallel/interleaved_parallel.h"

#include <set>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <queue>
#include <utility>

#include "frontend/operator/ops.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/graph_util/pipeline_split_utils.h"
#include "frontend/parallel/node_check.h"
#include "frontend/parallel/parameter_manager.h"
#include "include/utils/comm_manager.h"
#include "include/utils/parallel_context.h"
#include "include/utils/anfalgo.h"
#include "ir/param_info.h"
#include "ir/tensor.h"
#include "ir/graph_utils.h"
#include "primitive/array_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/nn_ops.h"
#include "primitive/other_ops.h"
#include "primitive/sequence_ops.h"
#include "frontend/parallel/parallel_node_check.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"
#include "primitive/auto_generate/gen_ops_primitive_v.h"

namespace mindspore {
namespace parallel {
std::vector<int64_t> fine_grain_concat_used;

void ChangeAllGatherGroup(const CNodePtr &ag_cnode, const RankList &new_group_ranks) {
  Group new_group;
  if (g_device_manager->CreateGroup(new_group_ranks, &new_group) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, ag_cnode)
      << ": Create communication group failed, the rank_list is: " << new_group_ranks;
  }
  auto ag_prim = GetCNodePrimitive(ag_cnode);
  MS_EXCEPTION_IF_NULL(ag_prim);
  ag_prim->AddAttr(GROUP, MakeValue(new_group.name()));
  ag_prim->AddAttr(GROUP_RANKS, MakeValue(g_device_manager->FindRankListNameByHashName(new_group.name())));
  ag_prim->AddAttr(RANK_SIZE, MakeValue<int64_t>(new_group_ranks.size()));
}

std::vector<CNodePtr> InterleavedReplacedConcatNodes(const std::vector<CNodePtr> &ag_vector) {
  std::vector<CNodePtr> replace_nodes;
  for (const auto &ag : ag_vector) {
    auto ag_next_nodes = GetOutputNodesWithFilter(ag, [&](const AnfNodePtr &anode) {
      return IsPrimitiveCNode(anode, prim::kPrimSplit) || IsPrimitiveCNode(anode, prim::kPrimTupleGetItem) ||
             IsPrimitiveCNode(anode, prim::kPrimMakeTuple);
    });
    std::set<AnfNodePtr> next_nodes_set;
    std::transform(ag_next_nodes.begin(), ag_next_nodes.end(), std::inserter(next_nodes_set, next_nodes_set.begin()),
                   [](auto pair) { return pair.first; });
    if (!(next_nodes_set.size() == kSizeOne && IsPrimitiveCNode(ag_next_nodes.front().first, prim::kPrimConcat))) {
      continue;
    }
    auto concat_cnode = ag_next_nodes.front().first->cast<CNodePtr>();
    auto concat_prim = GetCNodePrimitive(concat_cnode);
    MS_EXCEPTION_IF_NULL(concat_prim);
    if (concat_prim->instance_name().find(REDISTRIBUTION_OP) != std::string::npos) {
      replace_nodes.push_back(concat_cnode);
    }
  }
  return replace_nodes;
}

std::vector<std::vector<CNodePtr>> CreateInterleavedNeedReplaceOpLists(const CNodePtr &virtual_converter_end,
                                                                       const PrimitivePtr &r_prim) {
  std::vector<std::vector<CNodePtr>> need_replace_op_lists;
  for (size_t j = 1; j < virtual_converter_end->size(); ++j) {
    auto current_node = virtual_converter_end->input(j)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(current_node);
    std::vector<CNodePtr> need_replace_op_list;
    while (!IsPrimitiveCNode(current_node, prim::kPrimVirtualConverterBegin)) {
      if (IsPrimitiveCNode(current_node, r_prim)) {
        need_replace_op_list.push_back(current_node);
      }
      current_node = current_node->input(kIndex1)->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(current_node);
    }
    need_replace_op_lists.push_back(need_replace_op_list);
  }
  return need_replace_op_lists;
}

CNodePtr ReplaceInterleavedAllGatherToConcat(const FuncGraphPtr &func_graph, const std::vector<CNodePtr> &ag_vector,
                                             const std::vector<std::vector<int64_t>> &new_group_ranks_vector,
                                             size_t independent_size) {
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple->Clone())};
  std::transform(ag_vector.begin(), ag_vector.end(), std::back_inserter(make_tuple_inputs),
                 [&](auto node) { return independent_size == 1 ? node->input(kIndex1) : node; });
  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  auto replace_nodes = InterleavedReplacedConcatNodes(ag_vector);
  bool replace_concat = (!replace_nodes.empty() && independent_size == 1);
  AnfNodePtr axis = NewValueNode(MakeValue<int64_t>(0));
  if (replace_concat) {
    axis = replace_nodes.front()->input(kIndex2);
  }
  std::vector<AnfNodePtr> concat_inputs = {NewValueNode(prim::kPrimConcat->Clone()), make_tuple, axis};
  auto concat = func_graph->NewCNode(concat_inputs);
  concat->AddAttr(INTERLEAVED_PARALLEL, MakeValue(true));
  auto manager = func_graph->manager();

  for (size_t i = 0; i < ag_vector.size(); ++i) {
    auto ag = ag_vector[i];
    if (independent_size != 1) {
      // set allgather attrs
      ChangeAllGatherGroup(ag, new_group_ranks_vector[i]);
    }
    if (!replace_concat) {
      (void)manager->Replace(ag, concat);
    }
  }
  if (!replace_concat) {
    return concat;
  }
  for (size_t i = 0; i < replace_nodes.size(); ++i) {
    (void)manager->Replace(replace_nodes[i], concat);
  }
  return concat;
}

void MergeOpBeforeInterleaveSlice(const FuncGraphPtr &func_graph, const CNodePtr &virtual_converter_end) {
  auto virtual_converter_end_prim = GetCNodePrimitive(virtual_converter_end);
  MS_EXCEPTION_IF_NULL(virtual_converter_end_prim);
  std::vector<std::vector<CNodePtr>> need_replace_op_lists =
    CreateInterleavedNeedReplaceOpLists(virtual_converter_end, prim::kPrimStridedSlice);
  auto manager = func_graph->manager();
  if (need_replace_op_lists.empty()) {
    return;
  }
  auto col_size = need_replace_op_lists.front().size();
  for (size_t i = 0; i < need_replace_op_lists.size(); ++i) {
    if (need_replace_op_lists[i].size() != col_size) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, virtual_converter_end) << "Slice redistribution infer failed.";
    }
  }
  for (size_t col = 0; col < col_size; ++col) {
    std::set<std::vector<std::vector<int64_t>>> slice_value_list_set;
    for (size_t row = 0; row < need_replace_op_lists.size(); ++row) {
      auto slice_cnode = need_replace_op_lists[row][col];
      std::vector<std::vector<int64_t>> slice_value_list;
      for (size_t i = 2; i < kSizeFive; ++i) {
        ValuePtr slice_value = GetValueNode(slice_cnode->input(i));
        MS_EXCEPTION_IF_NULL(slice_value);
        auto value_vector = GetValue<std::vector<int64_t>>(slice_value);
        slice_value_list.push_back(value_vector);
      }
      slice_value_list_set.insert(slice_value_list);
    }
    if (slice_value_list_set.size() != need_replace_op_lists.size()) {
      continue;
    }
    // merge nodes before multi slice
    auto slice_input = need_replace_op_lists[kIndex0][col]->input(kIndex1);
    need_replace_op_lists[kIndex0][col]->AddAttr(INTERLEAVED_PARALLEL, MakeValue(true));
    bool has_fine_grain_index = virtual_converter_end_prim->HasAttr(kAttrFineGrainedInterleavedBlockIndex);
    if (has_fine_grain_index) {
      auto need_replace_op_lists_prim = GetCNodePrimitive(need_replace_op_lists[kIndex0][col]);
      MS_EXCEPTION_IF_NULL(need_replace_op_lists_prim);
      need_replace_op_lists_prim->AddAttr(kAttrFineGrainedInterleavedBlockIndex,
                                          virtual_converter_end_prim->GetAttr(kAttrFineGrainedInterleavedBlockIndex));
    }
    for (size_t row = 1; row < need_replace_op_lists.size(); ++row) {
      auto slice_cnode = need_replace_op_lists[row][col];
      auto slice_cnode_prim = GetCNodePrimitive(slice_cnode);
      MS_EXCEPTION_IF_NULL(slice_cnode_prim);
      slice_cnode->AddAttr(INTERLEAVED_PARALLEL, MakeValue(true));
      if (has_fine_grain_index) {
        slice_cnode_prim->AddAttr(kAttrFineGrainedInterleavedBlockIndex,
                                  virtual_converter_end_prim->GetAttr(kAttrFineGrainedInterleavedBlockIndex));
      }
      (void)manager->SetEdge(slice_cnode, kIndex1, slice_input);
    }
  }
}

void TagFineGrainedInterleavedBlockIndex(const CNodePtr &virtual_converter_end, const CNodePtr &replaced_concat) {
  auto virtual_converter_end_prim = GetCNodePrimitive(virtual_converter_end);
  if (virtual_converter_end_prim != nullptr &&
      virtual_converter_end_prim->HasAttr(kAttrFineGrainedInterleavedBlockIndex)) {
    auto block_index = GetValue<int64_t>(virtual_converter_end_prim->GetAttr(kAttrFineGrainedInterleavedBlockIndex));
    if (std::find(fine_grain_concat_used.begin(), fine_grain_concat_used.end(), block_index) ==
        fine_grain_concat_used.end()) {
      auto replace_concat_prim = GetCNodePrimitive(replaced_concat);
      MS_EXCEPTION_IF_NULL(replace_concat_prim);
      replace_concat_prim->AddAttr(kAttrFineGrainedInterleavedBlockIndex,
                                   virtual_converter_end_prim->GetAttr(kAttrFineGrainedInterleavedBlockIndex));
      fine_grain_concat_used.push_back(block_index);
    }
  }
}

void ConvertInterleaveAllGatherToConcat(const FuncGraphPtr &func_graph, const CNodePtr &virtual_converter_end,
                                        const std::vector<std::vector<std::vector<int64_t>>> &ag_group_ranks_vectors) {
  // Change communication rank_list && Create communication group
  // Replace AllConcat to Concat
  auto manager = func_graph->manager();
  bool merge_virtual_end = false;
  std::vector<std::vector<CNodePtr>> need_replace_op_lists =
    CreateInterleavedNeedReplaceOpLists(virtual_converter_end, prim::kPrimAllGather);
  MergeOpBeforeInterleaveSlice(func_graph, virtual_converter_end);
  if (need_replace_op_lists.size() != ag_group_ranks_vectors.size()) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, virtual_converter_end) << "AllGather redistribution infer failed.";
  }
  if (need_replace_op_lists.empty()) {
    return;
  }
  auto col_size = need_replace_op_lists.front().size();
  for (size_t i = 0; i < need_replace_op_lists.size(); ++i) {
    if (need_replace_op_lists[i].size() != col_size || ag_group_ranks_vectors[i].size() != col_size) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, virtual_converter_end) << "AllGather redistribution infer failed.";
    }
  }
  auto interleaved_num = ParallelContext::GetInstance()->fine_grained_micro_interleaved_size();
  auto stage_begin_rank = g_device_manager->stage_device_num() * g_device_manager->stage_id() * interleaved_num;
  for (size_t col = 0; col < col_size; ++col) {
    std::vector<std::vector<int64_t>> new_group_ranks_vector;
    std::unordered_map<std::string, std::vector<CNodePtr>> ag_vector_map;
    size_t independent_size = 0;
    for (size_t row = 0; row < need_replace_op_lists.size(); ++row) {
      auto group_ranks = ag_group_ranks_vectors[row][col];
      std::vector<int64_t> new_group_ranks;
      std::set<int64_t> new_group_ranks_set;
      for (const auto &g_rank : group_ranks) {
        new_group_ranks_set.insert(int64_t((g_rank - stage_begin_rank) / interleaved_num) +
                                   stage_begin_rank / interleaved_num);
        new_group_ranks.push_back(int64_t((g_rank - stage_begin_rank) / interleaved_num) +
                                  stage_begin_rank / interleaved_num);
      }
      if (new_group_ranks_set.size() == new_group_ranks.size()) {
        // set allgather attrs
        ChangeAllGatherGroup(need_replace_op_lists[row][col], new_group_ranks);
        continue;
      }
      std::vector<int64_t> new_group_ranks_no_repeat;
      std::copy(new_group_ranks_set.begin(), new_group_ranks_set.end(), std::back_inserter(new_group_ranks_no_repeat));
      std::sort(new_group_ranks_no_repeat.begin(), new_group_ranks_no_repeat.end());
      new_group_ranks_vector.push_back(new_group_ranks_no_repeat);
      if (independent_size > 0 && new_group_ranks_no_repeat.size() != independent_size) {
        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, virtual_converter_end)
          << "The concat group in micro interleaved is wrong!";
      }
      independent_size = new_group_ranks_no_repeat.size();
      auto group_str = g_device_manager->GenerateGroupNameByRanks(group_ranks);
      ag_vector_map[group_str].push_back(need_replace_op_lists[row][col]);
    }
    if (new_group_ranks_vector.empty()) {
      continue;
    }

    // Check whether all branch needing be replace
    if (new_group_ranks_vector.size() < need_replace_op_lists.size()) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, virtual_converter_end) << "The concat group in micro interleaved is wrong!";
    }

    // replace allgathers to one concat.
    for (const auto &ag_vector_pair : ag_vector_map) {
      auto replaced_concat = ReplaceInterleavedAllGatherToConcat(func_graph, ag_vector_pair.second,
                                                                 new_group_ranks_vector, independent_size);
      TagFineGrainedInterleavedBlockIndex(virtual_converter_end, replaced_concat);
      auto replaced_concat_users =
        GetOutputNodesWithFilter(replaced_concat, [&](const AnfNodePtr &anode) { return false; });
      if (replaced_concat_users.size() == kSizeOne) {
        merge_virtual_end = false;
        continue;
      }
      if (std::all_of(replaced_concat_users.begin(), replaced_concat_users.end(),
                      [](const std::pair<AnfNodePtr, int> &pair) {
                        return IsPrimitiveCNode(pair.first, prim::kPrimStridedSlice) &&
                               pair.first->cast<CNodePtr>()->HasAttr(INTERLEAVED_PARALLEL);
                      })) {
        merge_virtual_end = false;
        continue;
      }
      merge_virtual_end = true;
    }
  }
  if (!merge_virtual_end) {
    return;
  }
  // merge the nodes afer the interleaved parallel concat.
  auto virtual_end_input1 = virtual_converter_end->input(kIndex1)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(virtual_end_input1);
  auto new_virtual_converter_end = CreateVirtualConverterEndNode(func_graph, {virtual_end_input1});
  (void)manager->Replace(virtual_converter_end, new_virtual_converter_end);
}

bool IsDuplicatedVirtualConverterBegin(const CNodePtr &virtual_converter_begin) {
  auto virtual_converter_begin_input = virtual_converter_begin->input(kSizeOne);
  if (IsPrimitiveCNode(virtual_converter_begin_input, prim::kPrimVirtualConverterEnd)) {
    return false;
  }
  if (!IsPrimitiveCNode(virtual_converter_begin_input) ||
      IsPrimitiveCNode(virtual_converter_begin_input, prim::kPrimUpdateState)) {
    return false;
  }
  auto virtual_converter_begin_input_cnode = virtual_converter_begin_input->cast<CNodePtr>();
  if (IsParallelCareNode(virtual_converter_begin_input_cnode)) {
    return false;
  }
  auto virtual_converter_begin_users = GetOutputNodesWithFilter(
    virtual_converter_begin, [&](const AnfNodePtr &anode) { return IsPrimitiveCNode(anode, prim::kPrimTupleGetItem); });
  if (virtual_converter_begin_users.size() <= kSizeOne) {
    return false;
  }
  std::set<std::vector<std::vector<int64_t>>> slice_value_list_set;
  for (const auto &user_pair : virtual_converter_begin_users) {
    if (!IsPrimitiveCNode(user_pair.first, prim::kPrimStridedSlice)) {
      continue;
    }
    auto slice = user_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(slice);
    std::vector<std::vector<int64_t>> slice_value_list;
    for (size_t i = 2; i < kSizeFive; ++i) {
      ValuePtr slice_value = GetValueNode(slice->input(i));
      MS_EXCEPTION_IF_NULL(slice_value);
      auto value_vector = GetValue<std::vector<int64_t>>(slice_value);
      slice_value_list.push_back(value_vector);
    }
    slice_value_list_set.insert(slice_value_list);
  }
  if (slice_value_list_set.size() == virtual_converter_begin_users.size()) {
    return false;
  }
  return true;
}

bool GetOrderOfTwoAnode(const std::pair<AnfNodePtr, int> &pair1, const std::pair<AnfNodePtr, int> &pair2) {
  int number1 = pair1.second;
  int number2 = pair2.second;
  auto pair1_cast = pair1.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(pair1_cast);
  auto pair1_input_node = pair1_cast->input(pair1.second);
  auto pair2_cast = pair2.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(pair2_cast);
  auto pair2_input_node = pair2_cast->input(pair2.second);
  if (IsPrimitiveCNode(pair1_input_node, prim::kPrimTupleGetItem)) {
    number1 = LongToInt(GetTupleGetItemIndex(pair1_input_node->cast<CNodePtr>()));
  }
  if (IsPrimitiveCNode(pair2_input_node, prim::kPrimTupleGetItem)) {
    number2 = LongToInt(GetTupleGetItemIndex(pair2_input_node->cast<CNodePtr>()));
  }
  return number1 < number2;
}

bool IsCallFuncInputParam(const AnfNodePtr &node) {
  if (!node->isa<Parameter>()) {
    return false;
  }
  auto node_param_ptr = node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(node_param_ptr);
  if (node_param_ptr->has_default()) {
    return false;
  }
  if (!RefParameterToActualParameter(node)) {
    return true;
  }
  return false;
}

std::vector<CNodePtr> DoSplitForNotParallelCareOpsInterleaved(const FuncGraphManagerPtr &manager,
                                                              const CNodePtr &virtual_converter_begin) {
  auto virtual_converter_begin_input = virtual_converter_begin->input(kSizeOne);
  auto virtual_converter_begin_users = GetOutputNodesWithFilter(
    virtual_converter_begin, [&](const AnfNodePtr &anode) { return IsPrimitiveCNode(anode, prim::kPrimTupleGetItem); });
  std::sort(virtual_converter_begin_users.begin(), virtual_converter_begin_users.end(),
            [](const auto &pair1, const auto &pair2) { return GetOrderOfTwoAnode(pair1, pair2); });
  auto virtual_converter_begin_input_cnode = virtual_converter_begin_input->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(virtual_converter_begin_input_cnode);
  std::vector<AnfNodePtr> new_inputs;
  std::vector<CNodePtr> new_virtual_converter_begin_vector;
  for (size_t i = 1; i < virtual_converter_begin_input_cnode->size(); ++i) {
    auto v_input_node = virtual_converter_begin_input_cnode->input(i);
    if ((!IsPrimitiveCNode(v_input_node) && !IsCallFuncInputParam(v_input_node) &&
         !(v_input_node->isa<CNode>() && v_input_node->cast<CNodePtr>() != nullptr &&
           IsValueNode<FuncGraph>(v_input_node->cast<CNodePtr>()->input(kIndex0)))) ||
        IsPrimitiveCNode(v_input_node, prim::kPrimUpdateState)) {
      new_inputs.push_back(v_input_node);
      continue;
    }
    auto new_virtual_converter_begin = CreateVirtualConverterBeginNode(virtual_converter_begin_input_cnode->input(i),
                                                                       virtual_converter_begin_users.size());
    new_inputs.push_back(new_virtual_converter_begin);
    new_virtual_converter_begin_vector.push_back(new_virtual_converter_begin);
  }

  for (size_t interleveaved_index = 0; interleveaved_index < virtual_converter_begin_users.size();
       ++interleveaved_index) {
    std::vector<AnfNodePtr> splited_node_inputs = {virtual_converter_begin_input_cnode->input(kIndex0)};
    for (size_t i = 0; i < new_inputs.size(); ++i) {
      if (!IsPrimitiveCNode(new_inputs[i]) || IsPrimitiveCNode(new_inputs[i], prim::kPrimUpdateState)) {
        splited_node_inputs.push_back(new_inputs[i]);
        continue;
      }
      std::vector<AnfNodePtr> tuple_get_item_inputs{NewValueNode(prim::kPrimTupleGetItem), new_inputs[i],
                                                    CreatInt64Imm(UlongToLong(interleveaved_index))};
      auto tuple_get_item_cnode = virtual_converter_begin_input_cnode->func_graph()->NewCNode(tuple_get_item_inputs);
      splited_node_inputs.push_back(tuple_get_item_cnode);
    }
    auto splited_node = virtual_converter_begin_input_cnode->func_graph()->NewCNode(splited_node_inputs);
    (void)manager->SetEdge(virtual_converter_begin_users[interleveaved_index].first,
                           virtual_converter_begin_users[interleveaved_index].second, splited_node);
  }
  return new_virtual_converter_begin_vector;
}

void SplitNotParallelCareOpsInterleaved(const FuncGraphPtr &root) {
  AnfNodePtr ret_after = root->get_return();
  MS_EXCEPTION_IF_NULL(ret_after);
  auto all_nodes = TopoSort(ret_after, SuccDeeperSimple);
  auto manager = root->manager();
  auto node_users = manager->node_users();
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimVirtualConverterBegin)) {
      continue;
    }
    std::queue<CNodePtr> visited;
    visited.push(node->cast<CNodePtr>());
    while (!visited.empty()) {
      auto virtual_converter_begin = visited.front();
      visited.pop();
      if (!IsDuplicatedVirtualConverterBegin(virtual_converter_begin)) {
        continue;
      }
      // Need to split the input
      auto new_virtual_converter_begins = DoSplitForNotParallelCareOpsInterleaved(manager, virtual_converter_begin);
      for (auto &new_virtual_converter_begin : new_virtual_converter_begins) {
        visited.push(new_virtual_converter_begin);
      }
    }
  }
  auto new_all_nodes = TopoSort(ret_after, SuccDeeperSimple);
  auto new_node_users = manager->node_users();
  for (const auto &node : new_all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimVirtualConverterEnd)) {
      continue;
    }
    auto end_cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(end_cnode);
    auto end_users = new_node_users.at(node);
    auto func_graph = node->func_graph();
    for (const auto &end_user : end_users) {
      if (!IsPrimitiveCNode(end_user.first, prim::kPrimCast)) {
        continue;
      }
      auto cast_cnode = end_user.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cast_cnode);
      std::vector<AnfNodePtr> new_end_inputs = {end_cnode->input(kIndex0)};
      for (size_t i = 1; i < end_cnode->size(); ++i) {
        std::vector<AnfNodePtr> new_cast_inputs = {cast_cnode->input(kIndex0), end_cnode->input(i),
                                                   cast_cnode->input(kIndex2)};
        auto new_cast = func_graph->NewCNode(new_cast_inputs);
        new_end_inputs.push_back(new_cast);
      }
      auto new_end = func_graph->NewCNode(new_end_inputs);
      (void)manager->Replace(cast_cnode, new_end);
    }
  }
}

int64_t SendRecvInterleavedAxis(const CNodePtr &send_recv) {
  MS_EXCEPTION_IF_NULL(send_recv);
  if (send_recv->has_user_data<TensorLayout>()) {
    auto layout = send_recv->user_data<TensorLayout>();
    MS_EXCEPTION_IF_NULL(layout);
    if (layout->IsInterleavedParallel()) {
      auto inter_layout = layout->LayoutForRedistribution();
      auto new_slice_shape = inter_layout.base_slice_shape().array();
      auto slice_shape = layout->base_slice_shape().array();
      if (new_slice_shape.size() != slice_shape.size()) {
        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, send_recv)
          << "The size of shape between interleaved and no interleaved is not equal.";
      }
      for (size_t i = 0; i < new_slice_shape.size(); ++i) {
        if (new_slice_shape[i] != slice_shape[i]) {
          return SizeToLong(i);
        }
      }
    }
  }
  return 0;
}

int64_t UserIsSend(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->size() <= kSizeOne) {
    return -1;
  }
  auto end_users = GetOutputNodesWithFilter(cnode, [&](const AnfNodePtr &anode) {
    return IsPrimitiveCNode(anode, prim::kPrimMakeTuple) || IsPrimitiveCNode(anode, prim::kPrimDepend);
  });
  if (end_users.size() == 1 && IsPrimitiveCNode(end_users.front().first, prim::kPrimSend)) {
    return SendRecvInterleavedAxis(end_users.front().first->cast<CNodePtr>());
  }
  if (end_users.size() == 1 && IsPrimitiveCNode(end_users.front().first, prim::kPrimReturn)) {
    auto func_graph = cnode->func_graph();
    auto fg_map = func_graph->func_graph_cnodes_index();
    for (auto &fg_use : fg_map) {
      auto fg_node = fg_use.first->first->cast<CNodePtr>();
      auto fg_users = GetOutputNodesWithFilter(fg_node, [&](const AnfNodePtr &anode) {
        return IsPrimitiveCNode(anode, prim::kPrimLoad) || IsPrimitiveCNode(anode, prim::kPrimDepend) ||
               IsPrimitiveCNode(anode, prim::kPrimTupleGetItem);
      });
      int64_t axis = -1;
      for (const auto &fg_user_pair : fg_users) {
        if (IsPrimitiveCNode(fg_user_pair.first, prim::kPrimUpdateState)) {
          continue;
        }
        if (!IsPrimitiveCNode(fg_user_pair.first, prim::kPrimSend)) {
          MS_LOG(INFO) << "The user of call func in cell reuse is not send.";
          return -1;
        }
        axis = SendRecvInterleavedAxis(fg_user_pair.first->cast<CNodePtr>());
      }
      return axis;
    }
  }
  return -1;
}

void MoveVirtualConverterEndInsideCallFunc(const FuncGraphPtr &root) {
  auto all_nodes = TopoSort(root->get_return(), SuccDeeperSimple);
  auto manager = root->manager();
  auto node_users = manager->node_users();
  for (const auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto call_cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(call_cnode);
    if (!IsValueNode<FuncGraph>(call_cnode->input(0))) {
      continue;
    }
    auto sub_func_graph = GetValueNode<FuncGraphPtr>(call_cnode->input(0));
    MS_EXCEPTION_IF_NULL(sub_func_graph);
    auto call_inputs(call_cnode->inputs());
    size_t inserted_num = 0;
    auto sub_graph_parameters = sub_func_graph->parameters();
    auto new_user_graph_parameters(sub_graph_parameters);
    std::vector<std::vector<AnfNodePtr>> new_virtual_end_inputs_list;
    std::vector<AnfNodeIndexSet> replaced_users_list;
    for (size_t i = 1; i < call_cnode->size(); ++i) {
      auto call_input = call_cnode->input(i);
      if (!IsPrimitiveCNode(call_input, prim::kPrimVirtualConverterEnd)) {
        continue;
      }
      auto virtual_converter_end = call_input->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(virtual_converter_end);
      auto call_cnodes_map = sub_func_graph->func_graph_cnodes_index();
      if (call_cnodes_map.size() > 1) {
        MS_LOG_WITH_NODE(EXCEPTION, call_input)
          << "Func graph :" << sub_func_graph->ToString()
          << " has been called more than once, but its input has different sharding strategy.";
      }
      call_inputs.erase(call_inputs.begin() + inserted_num + i);
      for (size_t j = 0; j < virtual_converter_end->size() - 1; ++j) {
        call_inputs.insert(call_inputs.begin() + inserted_num + i + j, virtual_converter_end->input(j + 1));
      }

      size_t curr_param_index = i - 1;
      auto origin_param_users = node_users[new_user_graph_parameters[inserted_num + curr_param_index]];
      replaced_users_list.push_back(origin_param_users);
      new_user_graph_parameters.erase(new_user_graph_parameters.begin() + inserted_num + curr_param_index);
      std::vector<AnfNodePtr> virtual_end_inputs{NewValueNode(prim::kPrimVirtualConverterEnd)};
      for (size_t j = 0; j < virtual_converter_end->size() - 1; ++j) {
        auto new_parameter = sub_func_graph->add_parameter();
        new_user_graph_parameters.insert(new_user_graph_parameters.begin() + inserted_num + curr_param_index + j,
                                         new_parameter);
        virtual_end_inputs.push_back(new_parameter);
      }
      new_virtual_end_inputs_list.push_back(virtual_end_inputs);
      inserted_num += virtual_converter_end->size() - kSizeTwo;
    }
    if (!new_virtual_end_inputs_list.empty()) {
      auto new_call_cnode = call_cnode->func_graph()->NewCNode(call_inputs);
      (void)manager->Replace(call_cnode, new_call_cnode);
      sub_func_graph->set_parameters(new_user_graph_parameters);
    }
    for (size_t j = 0; j < new_virtual_end_inputs_list.size(); ++j) {
      auto virtual_converter_end = sub_func_graph->NewCNode(new_virtual_end_inputs_list[j]);
      auto param_users = replaced_users_list[j];
      for (const auto &user_pair : param_users) {
        (void)manager->SetEdge(user_pair.first, user_pair.second, virtual_converter_end);
      }
    }
  }
}

int64_t RemoveVirtualConverterEndInCall(const FuncGraphManagerPtr &manager, const CNodePtr &virtual_end) {
  auto cur_cnode = virtual_end;
  auto func_graph = virtual_end->func_graph();
  int64_t tuple_index = -1;
  while (true) {
    auto node_users = manager->node_users();
    auto next_node_pair = node_users[cur_cnode].front();
    auto next_node = next_node_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(next_node);
    if (IsPrimitiveCNode(next_node, prim::kPrimMakeTuple) || IsPrimitiveCNode(next_node, prim::kPrimReturn)) {
      auto is_tuple_output = IsPrimitiveCNode(next_node, prim::kPrimMakeTuple);
      std::vector<AnfNodePtr> new_return_inputs = next_node->inputs();
      new_return_inputs[kIndex0] = NewValueNode(prim::kPrimMakeTuple);
      new_return_inputs.erase(new_return_inputs.begin() + next_node_pair.second);
      for (size_t i = 1; i < cur_cnode->size(); ++i) {
        new_return_inputs.insert(new_return_inputs.begin() + next_node_pair.second + i - 1, cur_cnode->input(i));
      }
      auto new_return_cnode = func_graph->NewCNode(new_return_inputs);
      if (is_tuple_output) {
        tuple_index = int64_t(next_node_pair.second) - 1;
        (void)manager->Replace(next_node, new_return_cnode);
      } else {
        (void)manager->SetEdge(next_node, next_node_pair.second, new_return_cnode);
      }
      break;
    }
    std::vector<AnfNodePtr> virtual_end_inputs{NewValueNode(prim::kPrimVirtualConverterEnd)};
    for (size_t i = 1; i < cur_cnode->size(); ++i) {
      auto new_next_node_inputs = next_node->inputs();
      new_next_node_inputs[next_node_pair.second] = cur_cnode->input(i);
      auto new_next_node = func_graph->NewCNode(new_next_node_inputs);
      virtual_end_inputs.push_back(new_next_node);
    }
    cur_cnode = func_graph->NewCNode(virtual_end_inputs);
    (void)manager->Replace(next_node, cur_cnode);
  }
  return tuple_index;
}

void InsertVirtualConverterEnd(const FuncGraphPtr &root, const FuncGraphPtr &func_graph, const CNodePtr &fg_node,
                               int64_t tuple_index, size_t size) {
  auto manager = root->manager();
  auto parent_graph = fg_node->func_graph();
  auto node_users = manager->node_users();
  auto fg_node_users = node_users.at(fg_node);
  for (const auto &fg_user_pair : fg_node_users) {
    if (!IsPrimitiveCNode(fg_user_pair.first, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto get_item_node = fg_user_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(get_item_node);
    auto get_item_node_index2 = get_item_node->input(kIndex2);
    MS_EXCEPTION_IF_NULL(get_item_node_index2);
    auto get_item_node_index2_cast = get_item_node_index2->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(get_item_node_index2_cast);
    auto index_value = get_item_node_index2_cast->value();
    auto index = GetValue<int64_t>(index_value);
    if (index == tuple_index) {
      std::vector<AnfNodePtr> virtual_end_inputs{NewValueNode(prim::kPrimVirtualConverterEnd)};
      for (size_t i = 0; i < size; ++i) {
        auto new_get_item_node_inputs = get_item_node->inputs();
        auto new_index = index + int64_t(i);
        new_get_item_node_inputs[kIndex2] = NewValueNode(MakeValue<int64_t>(new_index));
        auto new_get_item_node = func_graph->NewCNode(new_get_item_node_inputs);
        virtual_end_inputs.push_back(new_get_item_node);
      }
      auto new_virtual_end = func_graph->NewCNode(virtual_end_inputs);
      (void)manager->Replace(get_item_node, new_virtual_end);
    } else if (index > tuple_index) {
      auto new_index = index + int64_t(size);
      auto new_tuple_getitem_inputs = get_item_node->inputs();
      new_tuple_getitem_inputs[kIndex2] = NewValueNode(MakeValue<int64_t>(new_index));
      auto new_tuple_getitem = parent_graph->NewCNode(new_tuple_getitem_inputs);
      (void)manager->Replace(get_item_node, new_tuple_getitem);
    }
  }
}

void MoveVirtualConverterEndOutsideCallFunc(const FuncGraphPtr &root) {
  auto all_nodes = TopoSort(root->get_return(), SuccDeeperSimple);
  auto manager = root->manager();
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimVirtualConverterEnd)) {
      continue;
    }
    auto virtual_end = node->cast<CNodePtr>();
    if (UserIsSend(virtual_end) >= 0) {
      continue;
    }
    auto end_users = GetOutputNodesWithFilter(virtual_end, [&](const AnfNodePtr &anode) {
      return IsPrimitiveCNode(anode, prim::kPrimMakeTuple) || IsPrimitiveCNode(anode, prim::kPrimDepend);
    });

    if (end_users.size() != 1 || !IsPrimitiveCNode(end_users.front().first, prim::kPrimReturn)) {
      continue;
    }
    // Move down virtual_converter_end
    auto tuple_index = RemoveVirtualConverterEndInCall(manager, virtual_end);
    auto is_tuple_output = tuple_index >= 0;
    auto func_graph = virtual_end->func_graph();
    auto fg_map = func_graph->func_graph_cnodes_index();
    auto interleave_size = virtual_end->size() - 1;
    for (auto &fg_use : fg_map) {
      auto fg_node = fg_use.first->first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(fg_node);
      auto parent_graph = fg_node->func_graph();
      auto node_users = manager->node_users();
      auto fg_node_users = node_users.at(fg_node);
      // insert virtual_converter_end at call
      if (is_tuple_output) {
        InsertVirtualConverterEnd(root, func_graph, fg_node, tuple_index, interleave_size);
        continue;
      }

      std::vector<AnfNodePtr> virtual_end_inputs{NewValueNode(prim::kPrimVirtualConverterEnd)};
      for (size_t i = 0; i < interleave_size; ++i) {
        std::vector<AnfNodePtr> get_item_inputs{NewValueNode(prim::kPrimTupleGetItem), fg_node,
                                                NewValueNode(MakeValue<int64_t>(i))};
        auto get_item_node = parent_graph->NewCNode(get_item_inputs);
        virtual_end_inputs.push_back(get_item_node);
      }
      auto virtual_end_new = parent_graph->NewCNode(virtual_end_inputs);
      for (const auto &fg_user_pair : fg_node_users) {
        if (IsPrimitiveCNode(fg_user_pair.first, prim::kPrimUpdateState)) {
          manager->SetEdge(fg_user_pair.first, fg_user_pair.second, virtual_end_new->input(kIndex1));
          continue;
        }
        manager->SetEdge(fg_user_pair.first, fg_user_pair.second, virtual_end_new);
      }
    }
  }
}

bool CheckMonadUsers(const std::vector<std::pair<AnfNodePtr, int>> &end_users) {
  if (end_users.size() == 1 && IsPrimitiveCNode(end_users.front().first, prim::kPrimUpdateState)) {
    return true;
  }
  constexpr size_t monad_user_size = 2;
  if (end_users.size() == monad_user_size) {
    const auto &first_user = end_users.front().first;
    const auto &second_user = end_users.back().first;
    if (IsPrimitiveCNode(first_user, prim::kPrimUpdateState) && IsPrimitiveCNode(second_user, prim::kPrimLoad)) {
      return true;
    }
    if (IsPrimitiveCNode(second_user, prim::kPrimUpdateState) && IsPrimitiveCNode(first_user, prim::kPrimLoad)) {
      return true;
    }
  }
  return false;
}

void EraseResVirtualConverterEnd(const FuncGraphPtr &root, bool is_fine_grained) {
  AnfNodePtr new_ret_after = root->get_return();
  auto manager = root->manager();
  MS_EXCEPTION_IF_NULL(new_ret_after);
  auto new_all_nodes = TopoSort(new_ret_after, SuccDeeperSimple);
  for (const auto &node : new_all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimVirtualConverterEnd)) {
      auto virtual_converter_end_cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(virtual_converter_end_cnode);
      if (virtual_converter_end_cnode->size() != kSizeTwo) {
        auto end_users = GetOutputNodesWithFilter(virtual_converter_end_cnode, [&](const AnfNodePtr &anode) {
          return IsPrimitiveCNode(anode, prim::kPrimMakeTuple) || IsPrimitiveCNode(anode, prim::kPrimDepend);
        });
        if (CheckMonadUsers(end_users)) {
          auto make_tuple_cnode = MakeMakeTupleByCNode(virtual_converter_end_cnode);
          (void)manager->Replace(virtual_converter_end_cnode, make_tuple_cnode);
          continue;
        }
        auto concat_axis = UserIsSend(virtual_converter_end_cnode);
        if (concat_axis >= 0) {
          auto make_tuple_cnode = MakeMakeTupleByCNode(virtual_converter_end_cnode);
          AnfNodePtr axis = NewValueNode(MakeValue<int64_t>(concat_axis));
          std::vector<AnfNodePtr> concat_inputs{NewValueNode(prim::kPrimConcat->Clone()), make_tuple_cnode, axis};
          auto concat = virtual_converter_end_cnode->func_graph()->NewCNode(concat_inputs);
          (void)manager->Replace(virtual_converter_end_cnode, concat);
          if (is_fine_grained) {
            auto concat_prim = GetCNodePrimitive(concat);
            MS_EXCEPTION_IF_NULL(concat_prim);
            concat_prim->AddAttr(kAttrFineGrainedInterleavedBlockIndex,
                                 MakeValue<int64_t>(kFineGrainedInterleavedBlockIndexMax));
          }
          continue;
        }

        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, virtual_converter_end_cnode)
          << "The VirtualConverterEnd nums is not equal to VirtualConverterBegin nums. "
             "Currently not support the last node of network sharding interleaved_parallel";
      }
      auto virtual_converter_end_input = virtual_converter_end_cnode->input(kIndex1);
      (void)manager->Replace(virtual_converter_end_cnode, virtual_converter_end_input);
    }
  }
}

void EraseVirtualConverter(const FuncGraphPtr &root) {
  MoveVirtualConverterEndOutsideCallFunc(root);
  MoveVirtualConverterEndInsideCallFunc(root);
  AnfNodePtr ret_after = root->get_return();
  auto all_nodes = TopoSort(ret_after, SuccDeeperSimple);
  auto manager = root->manager();
  auto node_users = manager->node_users();
  bool is_fine_grained = false;
  for (const auto &node : all_nodes) {
    if (IsPrimitiveCNode(node)) {
      auto node_cast_prim = GetCNodePrimitive(node->cast<CNodePtr>());
      MS_EXCEPTION_IF_NULL(node_cast_prim);
      if (node_cast_prim->HasAttr(kAttrFineGrainedInterleavedBlockIndex)) {
        is_fine_grained = true;
      }
    }

    if (!IsPrimitiveCNode(node, prim::kPrimVirtualConverterBegin)) {
      continue;
    }
    auto virtual_converter_begin = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(virtual_converter_begin);
    if (!IsPrimitiveCNode(virtual_converter_begin->input(kIndex1), prim::kPrimVirtualConverterEnd)) {
      MS_LOG(INFO) << "The VirtualConverterBegin input is not VirtualConverterEnd, it is "
                   << virtual_converter_begin->input(kIndex1)->fullname_with_scope();
      auto virtual_converter_begin_input_node = virtual_converter_begin->input(kIndex1);
      auto real_node = RefParameterToActualNode(virtual_converter_begin_input_node,
                                                [&](const CNodePtr &cnode) { return std::make_pair(false, 1); });
      if (real_node && real_node->isa<Parameter>()) {
        real_node =
          RefParameterToActualNode(real_node, [&](const CNodePtr &cnode) { return std::make_pair(false, 1); });
      }
      if (real_node && IsPrimitiveCNode(real_node, prim::kPrimReceive) &&
          node_users.at(virtual_converter_begin).size() > kSizeOne) {
        // Create Split op
        auto split_axis = SendRecvInterleavedAxis(real_node->cast<CNodePtr>());
        AnfNodePtr axis = NewValueNode(MakeValue<int64_t>(split_axis));
        auto v_begin_prim = GetCNodePrimitive(virtual_converter_begin);
        MS_EXCEPTION_IF_NULL(v_begin_prim);
        auto output_num = v_begin_prim->GetAttr("output_nums");
        AnfNodePtr split_size = NewValueNode(output_num);
        std::vector<AnfNodePtr> split_inputs{NewValueNode(prim::kPrimSplit->Clone()),
                                             virtual_converter_begin_input_node, axis, split_size};
        auto split = virtual_converter_begin->func_graph()->NewCNode(split_inputs);
        (void)manager->Replace(virtual_converter_begin, split);
        continue;
      }
      for (const auto &v_user_pair : node_users.at(virtual_converter_begin)) {
        (void)manager->Replace(v_user_pair.first, virtual_converter_begin_input_node);
      }
      continue;
    }
    auto virtual_converter_end = virtual_converter_begin->input(kIndex1)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(virtual_converter_end);
    auto virtual_converter_begin_users = manager->node_users()[virtual_converter_begin];
    if (virtual_converter_begin_users.size() != virtual_converter_end->size() - 1) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, virtual_converter_end)
        << "The VirtualConverterBegin users nums is not equal to VirtualConverterEnd inputs nums";
    }
    for (const auto &node_pair : virtual_converter_begin_users) {
      if (!IsPrimitiveCNode(node_pair.first, prim::kPrimTupleGetItem)) {
        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, node_pair.first)
          << "The VirtualConverterBegin user should be tuple_get_item.";
      }
      auto tuple_get_item = node_pair.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(tuple_get_item);
      auto tuple_get_item_index_value = GetValueNode(tuple_get_item->input(kIndex2));
      MS_EXCEPTION_IF_NULL(tuple_get_item_index_value);
      auto get_item_index = GetValue<int64_t>(tuple_get_item_index_value);
      (void)manager->Replace(tuple_get_item, virtual_converter_end->input(get_item_index + 1));
    }
  }
  EraseResVirtualConverterEnd(root, is_fine_grained);
}
}  // namespace parallel
}  // namespace mindspore
