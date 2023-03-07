/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/grouped_pairwise_exchange_alltoall.h"
#include <memory>
#include <queue>
#include <utility>
#include <list>
#include <vector>
#include <string>
#include <algorithm>

#include "mindspore/core/ops/core_ops.h"
#include "utils/anf_utils.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/tensor_layout/tensor_info.h"
#include "frontend/parallel/device_matrix.h"
#include "pipeline/jit/action.h"

namespace mindspore {
namespace opt {
namespace {
using CNodePtrPair = std::pair<CNodePtr, CNodePtr>;
using GpeaInfo = GroupedPairwiseExchangeAllToAllInfo;

CNodePtr FindFrontAlltoall(const CNodePtr &marked_node, std::vector<CNodePtr> *visited_marked_nodes) {
  MS_EXCEPTION_IF_NULL(marked_node);
  auto input_node = marked_node->input(1);
  auto input_cnode = input_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(input_cnode);
  std::queue<CNodePtr> node_queue;
  node_queue.push(input_cnode);

  CNodePtr alltoall_node = nullptr;
  while (!node_queue.empty()) {
    auto cnode = node_queue.front();
    node_queue.pop();
    if (IsPrimitiveCNode(cnode, prim::kPrimAllToAll)) {
      alltoall_node = cnode;
      break;
    }

    if (cnode->HasAttr("gpea_label")) {
      visited_marked_nodes->push_back(cnode);
    }

    auto input = cnode->input(1);
    MS_EXCEPTION_IF_NULL(input);
    if (!input->isa<CNode>()) {
      break;
    }
    auto in_cnode = input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(in_cnode);
    node_queue.push(in_cnode);
  }

  if (alltoall_node == nullptr) {
    MS_LOG(WARNING) << "Can't find alltoall node before " << GetCNodePrimitive(marked_node)->name();
  }
  return alltoall_node;
}

CNodePtr FindBackAlltoall(const FuncGraphManagerPtr &manager, const CNodePtr &marked_node,
                          std::vector<CNodePtr> *visited_marked_nodes) {
  MS_EXCEPTION_IF_NULL(marked_node);
  auto node_users_map = manager->node_users();
  auto node_users = node_users_map[marked_node];
  auto first_user = node_users.front().first;
  auto first_user_cnode = first_user->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(first_user_cnode);
  std::queue<CNodePtr> node_queue;
  node_queue.push(first_user_cnode);

  CNodePtr alltoall_node = nullptr;
  while (!node_queue.empty()) {
    auto cnode = node_queue.front();
    node_queue.pop();
    if (IsPrimitiveCNode(cnode, prim::kPrimAllToAll)) {
      alltoall_node = cnode;
      break;
    }

    if (GetCNodePrimitive(cnode)->HasAttr("gpea_label")) {
      visited_marked_nodes->push_back(cnode);
    }

    auto cnode_users = node_users_map[cnode];
    if (cnode_users.empty()) {  // last cnode, exit while
      break;
    }
    auto first_node = cnode_users.front().first;
    MS_EXCEPTION_IF_NULL(first_node);
    auto first_cnode = first_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(first_cnode);
    node_queue.push(first_cnode);
  }

  if (alltoall_node == nullptr) {
    MS_LOG(WARNING) << "Can't find alltoall node after " << GetCNodePrimitive(marked_node)->name();
  }
  return alltoall_node;
}

CNodePtrPair FindAlltoallPair(const FuncGraphManagerPtr &manager, const CNodePtr &marked_node,
                              std::vector<CNodePtr> *visited_marked_nodes) {
  auto front_alltoall = FindFrontAlltoall(marked_node, visited_marked_nodes);
  if (front_alltoall == nullptr) {
    CNodePtrPair null_alltoall_pair(nullptr, nullptr);
    return null_alltoall_pair;
  }

  auto back_alltoall = FindBackAlltoall(manager, marked_node, visited_marked_nodes);
  if (back_alltoall == nullptr) {
    CNodePtrPair null_alltoall_pair(nullptr, nullptr);
    return null_alltoall_pair;
  }

  CNodePtrPair alltoall_pair(front_alltoall, back_alltoall);
  return alltoall_pair;
}

void FindAlltoallNodePairs(const FuncGraphManagerPtr &manager, const std::vector<CNodePtr> &origin_nodes_topological,
                           std::vector<CNodePtrPair> *alltoall_pairs) {
  std::vector<CNodePtr> visited_marked_nodes;
  for (size_t i = 0; i < origin_nodes_topological.size(); i++) {
    auto cnode = origin_nodes_topological[i];
    if (!IsPrimitiveCNode(cnode)) {
      continue;
    }

    if (!GetCNodePrimitive(cnode)->HasAttr("gpea_label")) {
      continue;
    }

    if (std::find(visited_marked_nodes.begin(), visited_marked_nodes.end(), cnode) != visited_marked_nodes.end()) {
      continue;
    }

    visited_marked_nodes.push_back(cnode);
    auto alltoall_pair = FindAlltoallPair(manager, cnode, &visited_marked_nodes);
    if (alltoall_pair.first == nullptr || alltoall_pair.second == nullptr) {
      MS_LOG(WARNING) << "not find alltoall_pair around cnode: " << GetCNodePrimitive(cnode)->name();
      continue;
    }
    alltoall_pairs->push_back(alltoall_pair);
  }
}

size_t GetSplitDimFromAlltoall(const AnfNodePtr &alltoall) {
  size_t split_dim = LongToSize(common::AnfAlgo::GetNodeAttr<int64_t>(alltoall, kAttrSplitDim));
  return split_dim;
}

size_t GetConcatDimFromAlltoall(const AnfNodePtr &alltoall) {
  size_t concat_dim = LongToSize(common::AnfAlgo::GetNodeAttr<int64_t>(alltoall, kAttrConcatDim));
  return concat_dim;
}

CNodePtr NewSplitNode(const AnfNodePtr &input_node, size_t split_dim, size_t split_num) {
  if (split_num == 0) {
    MS_LOG(EXCEPTION) << "split_num should not be zero.";
  }
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> split_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplit->name())),
                                          input_node};
  auto split = input_node->func_graph()->NewCNode(split_inputs);
  MS_EXCEPTION_IF_NULL(split);

  auto dtype = common::AnfAlgo::GetOutputInferDataType(input_node, 0);
  std::vector<TypeId> dtypes(split_num, dtype);
  auto shape = common::AnfAlgo::GetOutputInferShape(input_node, 0);
  shape[split_dim] /= split_num;
  std::vector<ShapeVector> shapes(split_num, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, split.get());

  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue<int64_t>(split_dim), split);
  common::AnfAlgo::SetNodeAttr(kAttrOutputNum, MakeValue<int64_t>(split_num), split);
  split->set_scope(input_node->scope());
  return split;
}

CNodePtr NewSplitNode(const AnfNodePtr &input_node, size_t split_dim, size_t split_num, const ShapeVector &input_shape,
                      const TypeId &input_dtype) {
  if (split_num == 0) {
    MS_LOG(EXCEPTION) << "split_num should not be zero.";
  }
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> split_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplit->name())),
                                          input_node};
  auto split = input_node->func_graph()->NewCNode(split_inputs);
  MS_EXCEPTION_IF_NULL(split);

  std::vector<TypeId> dtypes(split_num, input_dtype);
  ShapeVector shape;
  for (size_t i = 0; i < input_shape.size(); i++) {
    shape.push_back(input_shape[i]);
  }
  shape[split_dim] /= split_num;
  std::vector<ShapeVector> shapes(split_num, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, split.get());

  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue<int64_t>(split_dim), split);
  common::AnfAlgo::SetNodeAttr(kAttrOutputNum, MakeValue<int64_t>(split_num), split);
  split->set_scope(input_node->scope());
  return split;
}

CNodePtr NewConcatNode(const AnfNodePtr &input_node, size_t concat_dim, size_t input_num) {
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> concat_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name())),
                                           input_node};
  auto concat = input_node->func_graph()->NewCNode(concat_inputs);
  MS_EXCEPTION_IF_NULL(concat);

  std::vector<TypeId> dtypes = {common::AnfAlgo::GetOutputInferDataType(input_node, 0)};
  auto shape = common::AnfAlgo::GetOutputInferShape(input_node, 0);
  shape[concat_dim] *= input_num;
  std::vector<ShapeVector> shapes(1, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, concat.get());

  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(static_cast<int64_t>(concat_dim)), concat);
  common::AnfAlgo::SetNodeAttr(kAttrInputNums, MakeValue(input_num), concat);
  common::AnfAlgo::SetNodeAttr(kAttrN, MakeValue(input_num), concat);
  concat->set_scope(input_node->scope());
  return concat;
}

CNodePtr NewMakeTupleNode(const std::vector<AnfNodePtr> &input_nodes) {
  // input_nodes are getitem nodes
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  for (size_t i = 0; i < input_nodes.size(); i++) {
    make_tuple_inputs.push_back(input_nodes[i]);
  }
  auto make_tuple = input_nodes[0]->func_graph()->NewCNode(make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple);

  auto dtype = common::AnfAlgo::GetOutputInferDataType(input_nodes[0], 0);
  std::vector<TypeId> dtypes(input_nodes.size(), dtype);
  auto shape = common::AnfAlgo::GetOutputInferShape(input_nodes[0], 0);
  std::vector<ShapeVector> shapes(input_nodes.size(), shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, make_tuple.get());
  make_tuple->set_scope(input_nodes[0]->scope());
  return make_tuple;
}

CNodePtr NewTupleGetItemNode(const AnfNodePtr &input_node, size_t output_index) {
  MS_EXCEPTION_IF_NULL(input_node);
  auto idx = NewValueNode(SizeToLong(output_index));
  MS_EXCEPTION_IF_NULL(idx);
  auto getitem = input_node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), input_node, idx});
  MS_EXCEPTION_IF_NULL(getitem);

  std::vector<TypeId> dtypes = {common::AnfAlgo::GetOutputInferDataType(input_node, output_index)};
  auto shapes = {common::AnfAlgo::GetOutputInferShape(input_node, output_index)};
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, getitem.get());
  getitem->set_scope(input_node->scope());
  return getitem;
}

void MakeSortedSplitGetItemNodes(const AnfNodePtr &input_node, const std::vector<int64_t> &sort_idx,
                                 std::vector<AnfNodePtr> *getitem_nodes) {
  if (AnfUtils::GetOutputTensorNum(input_node) != sort_idx.size()) {
    MS_LOG(EXCEPTION) << "The number of MakeTuple inputs is not equal to sort index number";
  }

  for (size_t i = 0; i < sort_idx.size(); i++) {
    auto getitem = NewTupleGetItemNode(input_node, LongToSize(sort_idx[i]));
    getitem_nodes->push_back(getitem);
  }
}

void NewTupleGetItemNodes(const AnfNodePtr &input_node, size_t split_num, std::vector<AnfNodePtr> *getitem_nodes) {
  // input_node is a node such as split node or neighbor exchange node
  for (size_t i = 0; i < split_num; i++) {
    auto getitem = NewTupleGetItemNode(input_node, i);
    getitem_nodes->push_back(getitem);
  }
}

CNodePtr NewNeighborExchangeNode(const AnfNodePtr &input_node, const std::vector<int64_t> &send_rank_ids,
                                 const std::vector<int64_t> &recv_rank_ids) {
  // input_node is maketuple node
  std::vector<AnfNodePtr> ne_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimNeighborExchange->name())),
                                       input_node};
  auto neighbor_exchange = input_node->func_graph()->NewCNode(ne_inputs);
  MS_EXCEPTION_IF_NULL(neighbor_exchange);
  auto input_cnode = input_node->cast<CNodePtr>();

  // RECV_TYPE
  auto dtype = common::AnfAlgo::GetOutputInferDataType(input_node, 0);
  common::AnfAlgo::SetNodeAttr(parallel::RECV_TYPE, TypeIdToType(dtype), neighbor_exchange);

  // GROUP
  std::string group = parallel::g_device_manager->world_group();
  common::AnfAlgo::SetNodeAttr(parallel::GROUP, MakeValue<std::string>(group), neighbor_exchange);

  // SEND_RANK_IDS, RECV_RANK_IDS
  common::AnfAlgo::SetNodeAttr(parallel::SEND_RANK_IDS, parallel::MakeListValue(send_rank_ids), neighbor_exchange);
  common::AnfAlgo::SetNodeAttr(parallel::RECV_RANK_IDS, parallel::MakeListValue(recv_rank_ids), neighbor_exchange);

  // SEND_SHAPES, RECV_SHAPES
  auto maketuple_input = input_cnode->inputs()[1];
  parallel::Shape shape = common::AnfAlgo::GetOutputInferShape(maketuple_input, 0);
  parallel::Shapes send_shapes;
  parallel::Shapes recv_shapes;
  for (size_t i = 0; i < send_rank_ids.size(); i++) {
    send_shapes.push_back(shape);
    recv_shapes.push_back(shape);
  }
  common::AnfAlgo::SetNodeAttr(parallel::SEND_SHAPES, parallel::MakeTupleListValue(send_shapes), neighbor_exchange);
  common::AnfAlgo::SetNodeAttr(parallel::RECV_SHAPES, parallel::MakeTupleListValue(recv_shapes), neighbor_exchange);

  // set dtypes and shapes
  std::vector<TypeId> dtypes(recv_shapes.size(), dtype);
  std::vector<ShapeVector> shapes(recv_shapes.size(), shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, neighbor_exchange.get());

  neighbor_exchange->set_scope(input_node->scope());
  return neighbor_exchange;
}

void CreateNeighborExchangeNodes(const AnfNodePtr &input_node, size_t split_dim, size_t concat_dim,
                                 const std::vector<int64_t> &send_rank_ids, const std::vector<int64_t> &recv_rank_ids,
                                 std::vector<AnfNodePtr> *neighbor_exchange_nodes) {
  CNodePtr split = nullptr;
  size_t send_num = send_rank_ids.size();
  std::vector<AnfNodePtr> getitem_nodes;
  if (IsPrimitiveCNode(input_node, prim::kPrimSplit)) {
    NewTupleGetItemNodes(input_node, send_num, &getitem_nodes);
  } else {
    split = NewSplitNode(input_node, split_dim, send_num);
    NewTupleGetItemNodes(split, send_num, &getitem_nodes);
  }

  auto maketuple = NewMakeTupleNode(getitem_nodes);
  auto neighbor_exchange = NewNeighborExchangeNode(maketuple, send_rank_ids, recv_rank_ids);
  std::vector<AnfNodePtr> getitem_nodes_after;
  size_t recv_num = recv_rank_ids.size();
  NewTupleGetItemNodes(neighbor_exchange, recv_num, &getitem_nodes_after);
  auto maketuple_after = NewMakeTupleNode(getitem_nodes_after);
  auto concat = NewConcatNode(maketuple_after, concat_dim, recv_num);

  if (split != nullptr) {
    neighbor_exchange_nodes->push_back(split);
  }
  neighbor_exchange_nodes->insert(neighbor_exchange_nodes->end(), getitem_nodes.begin(), getitem_nodes.end());
  neighbor_exchange_nodes->push_back(maketuple);
  neighbor_exchange_nodes->push_back(neighbor_exchange);
  neighbor_exchange_nodes->insert(neighbor_exchange_nodes->end(), getitem_nodes_after.begin(),
                                  getitem_nodes_after.end());
  neighbor_exchange_nodes->push_back(maketuple_after);
  neighbor_exchange_nodes->push_back(concat);
}

int64_t FindNodeIndex(const std::vector<CNodePtr> &node_vector, const CNodePtr &target_node) {
  auto iter = std::find(node_vector.begin(), node_vector.end(), target_node);
  if (iter == node_vector.end()) {
    return -1;
  } else {
    return std::distance(node_vector.begin(), iter);
  }
}

size_t FindAlltoallIndex(const std::vector<CNodePtr> &origin_nodes_topological, const CNodePtr &alltoall) {
  int64_t idx = FindNodeIndex(origin_nodes_topological, alltoall);
  if (idx == -1) {
    MS_LOG(EXCEPTION) << "Can not find alltoall node in origin_nodes_topological";
  }
  return LongToSize(idx);
}

ValueNodePtr ScaleShapeValueNode(const AnfNodePtr &old_shape_node, size_t scale_dim, int64_t scale_factor) {
  if (scale_factor == 0) {
    MS_LOG(EXCEPTION) << "scale_factor should not be zero.";
  }
  auto shape_value_node = old_shape_node->cast<ValueNodePtr>();
  auto value_ptr = shape_value_node->value();
  std::vector<ValuePtr> value_ptr_vec = value_ptr->cast<ValueTuplePtr>()->value();
  ShapeVector new_shape;
  for (size_t i = 0; i < value_ptr_vec.size(); i++) {
    auto shape_value = GetValue<int64_t>(value_ptr_vec[i]);
    if (i == scale_dim) {
      shape_value /= scale_factor;
    }
    new_shape.push_back(shape_value);
  }
  return NewValueNode(MakeValue(new_shape));
}

const std::vector<CNodePtr> FindCNodesAmongAlltoall(const std::vector<CNodePtr> &origin_nodes_topological,
                                                    const CNodePtrPair &alltoall_pair) {
  auto front_alltoall = alltoall_pair.first;
  auto back_alltoall = alltoall_pair.second;
  size_t front_alltoall_idx = FindAlltoallIndex(origin_nodes_topological, front_alltoall);
  size_t back_alltoall_idx = FindAlltoallIndex(origin_nodes_topological, back_alltoall);
  std::vector<CNodePtr> cnodes;
  for (size_t i = front_alltoall_idx + 1; i < back_alltoall_idx; i++) {
    cnodes.push_back(origin_nodes_topological[i]);
  }
  return cnodes;
}

void CloneScaledGraph(const std::vector<CNodePtr> &old_cnodes, const AnfNodePtr &input_node, size_t scale_factor,
                      GpeaInfo *gpea_info, std::vector<AnfNodePtr> *new_nodes) {
  mindspore::HashMap<CNodePtr, CNodePtr> cnode_map;
  auto input_cnode = input_node->cast<CNodePtr>();
  auto old_input_node = old_cnodes[0]->input(1);
  auto old_input_cnode = old_input_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(old_input_cnode);
  cnode_map[old_input_cnode] = input_cnode;

  size_t reshape_cnt = 0;
  std::vector<uint32_t> reshape_scale_axis = gpea_info->GetReshapeScaleAxisVec();
  for (size_t i = 0; i < old_cnodes.size(); i++) {
    auto cnode = old_cnodes[i];
    MS_LOG(DEBUG) << "node in " << i << " " << GetCNodePrimitive(cnode)->name();
    if (IsPrimitiveCNode(cnode, prim::kPrimLoad)) {
      new_nodes->push_back(cnode);  // reuse old Load node to not increase device memory
      continue;
    }

    // clone inputs
    std::vector<AnfNodePtr> new_inputs;
    auto inputs = cnode->inputs();
    for (size_t j = 0; j < inputs.size(); j++) {
      auto input = inputs[j];
      if (input->isa<CNode>()) {
        auto curr_input_cnode = input->cast<CNodePtr>();
        CNodePtr new_cnode;
        if (IsPrimitiveCNode(curr_input_cnode, prim::kPrimLoad)) {
          new_cnode = curr_input_cnode;
        } else {
          new_cnode = cnode_map[curr_input_cnode];
        }
        auto new_anf_node = new_cnode->cast<AnfNodePtr>();
        new_inputs.push_back(new_anf_node);
      } else if (input->isa<ValueNode>()) {
        ValueNodePtr new_value_node = NewValueNode(GetValueNode(input));
        new_inputs.push_back(new_value_node);
      } else if (input->isa<Parameter>()) {
        new_inputs.push_back(input);
      }
    }

    // scale reshape shape value
    if (IsPrimitiveCNode(cnode, prim::kPrimReshape)) {
      new_inputs[kIndex2] = ScaleShapeValueNode(new_inputs[kIndex2], reshape_scale_axis[reshape_cnt], scale_factor);
      reshape_cnt += 1;
    }

    // create CNode
    auto new_cnode = input_node->func_graph()->NewCNode(new_inputs);
    MS_EXCEPTION_IF_NULL(new_cnode);
    new_cnode->set_scope(cnode->scope());
    cnode_map[cnode] = new_cnode;
    new_nodes->push_back(new_cnode);
  }
}

void InsertDependOnBranches(const FuncGraphManagerPtr &manager, const std::vector<std::vector<AnfNodePtr>> &front_nodes,
                            const std::vector<std::vector<AnfNodePtr>> &back_nodes) {
  size_t comm_node_idx = 0;
  for (size_t i = 0; i < front_nodes[0].size(); i++) {
    auto cnode = front_nodes[0][i]->cast<CNodePtr>();
    if (IsPrimitiveCNode(cnode, prim::kPrimNeighborExchange)) {
      comm_node_idx = i;
      break;
    }
  }

  for (size_t branch_idx = 0; branch_idx < front_nodes.size() - 1; branch_idx++) {
    auto prev_node = front_nodes[branch_idx][comm_node_idx];
    auto node = front_nodes[branch_idx][comm_node_idx + 1];
    auto add_node = back_nodes[branch_idx + 1].front();
    // graph branch execution is reverse, so former branch depends on latter branch
    std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), prev_node, add_node};
    auto depend = node->func_graph()->NewCNode(depend_inputs);
    MS_EXCEPTION_IF_NULL(depend);
    depend->set_abstract(prev_node->abstract()->Clone());
    manager->SetEdge(node, 1, depend);
  }
}

CNodePtr CreateReplaceGraph(const FuncGraphManagerPtr &manager, const std::vector<CNodePtr> &origin_nodes_topological,
                            const CNodePtrPair &alltoall_pair, GpeaInfo *gpea_info) {
  auto front_alltoall = alltoall_pair.first;
  auto back_alltoall = alltoall_pair.second;
  auto graph_input = front_alltoall->input(1);

  // split input into several branch
  size_t front_split_dim = GetSplitDimFromAlltoall(front_alltoall);
  size_t front_concat_dim = GetConcatDimFromAlltoall(front_alltoall);
  size_t split_num = LongToSize(gpea_info->GetGroupNum());
  auto split = NewSplitNode(graph_input, front_split_dim, split_num);

  // clone several branch calculation graph
  size_t back_split_dim = GetSplitDimFromAlltoall(back_alltoall);
  size_t back_concat_dim = GetConcatDimFromAlltoall(back_alltoall);
  auto back_input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(back_alltoall, 0);
  if (split_num == 0) {
    MS_LOG(EXCEPTION) << "split_num should not be zero.";
  }
  back_input_shape[back_split_dim] /= split_num;
  auto back_input_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(back_alltoall, 0);
  std::vector<int64_t> send_group_ranks = gpea_info->GetSendGroupRanks();
  std::vector<std::vector<AnfNodePtr>> front_comm_branches;
  std::vector<std::vector<AnfNodePtr>> back_comm_branches;
  std::vector<AnfNodePtr> branch_output_nodes;
  auto old_calc_cnodes = FindCNodesAmongAlltoall(origin_nodes_topological, alltoall_pair);
  for (size_t i = 0; i < split_num; i++) {
    auto send_rank_ids = gpea_info->GetSendRankIds(i);
    auto recv_rank_ids = gpea_info->GetRecvRankIds(i);
    size_t split_branch_idx = LongToSize(send_group_ranks[i]);
    auto getitem = NewTupleGetItemNode(split, split_branch_idx);
    // create first neighbor exchange nodes
    std::vector<AnfNodePtr> front_neighbor_exchange_nodes;
    CreateNeighborExchangeNodes(getitem, front_split_dim, front_concat_dim, send_rank_ids, recv_rank_ids,
                                &front_neighbor_exchange_nodes);
    front_comm_branches.push_back(front_neighbor_exchange_nodes);
    // clone calculation nodes
    std::vector<AnfNodePtr> new_calc_nodes;
    CloneScaledGraph(old_calc_cnodes, front_neighbor_exchange_nodes.back(), split_num, gpea_info, &new_calc_nodes);
    MS_LOG(DEBUG) << "Create calculation done";
    // create second neighbor exchange nodes
    auto back_comm_split =
      NewSplitNode(new_calc_nodes.back(), back_split_dim, recv_rank_ids.size(), back_input_shape, back_input_dtype);
    std::vector<AnfNodePtr> back_neighbor_exchange_nodes;
    back_neighbor_exchange_nodes.push_back(back_comm_split);
    CreateNeighborExchangeNodes(back_comm_split, back_split_dim, back_concat_dim, recv_rank_ids, send_rank_ids,
                                &back_neighbor_exchange_nodes);
    branch_output_nodes.push_back(back_neighbor_exchange_nodes.back());
    back_comm_branches.push_back(back_neighbor_exchange_nodes);
  }
  MS_LOG(DEBUG) << "Create multi branch done";
  InsertDependOnBranches(manager, front_comm_branches, back_comm_branches);
  MS_LOG(DEBUG) << "InsertDependOnBranches done";

  // concat several branch into one branch
  auto maketuple = NewMakeTupleNode(branch_output_nodes);
  auto concat = NewConcatNode(maketuple, back_concat_dim, split_num);
  auto reorder_split = NewSplitNode(concat, back_concat_dim, split_num);
  std::vector<AnfNodePtr> reorder_getitem_nodes;
  std::vector<int64_t> sort_idx = gpea_info->GetSortedInputsIdx();
  MakeSortedSplitGetItemNodes(reorder_split, sort_idx, &reorder_getitem_nodes);
  auto reorder_maketuple = NewMakeTupleNode(reorder_getitem_nodes);
  auto reorder_concat = NewConcatNode(reorder_maketuple, back_concat_dim, split_num);
  MS_LOG(DEBUG) << "Create replace graph done";
  return reorder_concat;
}

void CreateAndReplaceAlltoall(const FuncGraphManagerPtr &manager, const std::vector<CNodePtr> &origin_nodes_topological,
                              const CNodePtrPair &alltoall_pair, GpeaInfo *gpea_info) {
  auto cnode = CreateReplaceGraph(manager, origin_nodes_topological, alltoall_pair, gpea_info);
  manager->Replace(alltoall_pair.second, cnode);
}

void CreateAndReplaceGraph(const FuncGraphManagerPtr &manager, const std::vector<CNodePtr> &origin_nodes_topological,
                           const std::vector<CNodePtrPair> &alltoall_pairs, GpeaInfo *gpea_info) {
  for (size_t i = 0; i < alltoall_pairs.size(); i++) {
    CreateAndReplaceAlltoall(manager, origin_nodes_topological, alltoall_pairs[i], gpea_info);
  }
}

void CheckReshapeScaleAxis(const std::vector<CNodePtr> &origin_nodes_topological, const CNodePtrPair &alltoall_pair,
                           GpeaInfo *gpea_info) {
  size_t front_alltoall_idx = FindAlltoallIndex(origin_nodes_topological, alltoall_pair.first);
  size_t back_alltoall_idx = FindAlltoallIndex(origin_nodes_topological, alltoall_pair.second);
  size_t reshape_cnode_num = 0;
  for (size_t i = front_alltoall_idx + 1; i < back_alltoall_idx; i++) {
    auto cnode = origin_nodes_topological[i];
    if (IsPrimitiveCNode(cnode, prim::kPrimReshape)) {
      reshape_cnode_num += 1;
    }
  }

  std::vector<uint32_t> reshape_scale_axis = gpea_info->GetReshapeScaleAxisVec();
  size_t axis_length = reshape_scale_axis.size();
  MS_LOG(DEBUG) << "The graph has " << reshape_cnode_num << " reshape nodes, which will be scaled on certain axis";
  if (axis_length == reshape_cnode_num) {
    MS_LOG(DEBUG) << "'reshape_scale_axis' " << reshape_scale_axis;
    return;
  } else {
    MS_LOG(DEBUG) << "'reshape_scale_axis' has " << axis_length
                  << " element, its length should be same as the number of scaled reshape node";
    MS_LOG(DEBUG) << "Show graph nodes start";
    for (size_t i = front_alltoall_idx; i < back_alltoall_idx + 1; i++) {
      auto cnode = origin_nodes_topological[i];
      auto prim_name = GetCNodePrimitive(cnode)->name();
      auto scope_name = cnode->scope()->name();
      std::string input_shape_string = "";
      std::string output_shape_string = "";
      std::string space = " ";
      size_t output_num = AnfUtils::GetOutputTensorNum(cnode);
      size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);
      for (size_t j = 0; j < input_num; j++) {
        auto shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, j);
        if (shape.size() > 0) {
          input_shape_string += space;
          input_shape_string += parallel::ShapeToString(shape);
        }
      }
      for (size_t j = 0; j < output_num; j++) {
        auto shape = common::AnfAlgo::GetOutputInferShape(cnode, j);
        if (shape.size() > 0) {
          output_shape_string += space;
          output_shape_string += parallel::ShapeToString(shape);
        }
      }

      MS_LOG(DEBUG) << "Node: " << prim_name;
      MS_LOG(DEBUG) << "Scope: " << scope_name;
      MS_LOG(DEBUG) << "Input shapes: " << input_shape_string;
      MS_LOG(DEBUG) << "Output shapes: " << output_shape_string;
    }
    MS_LOG(DEBUG) << "Show graph nodes end";
    MS_LOG(EXCEPTION) << "The size of 'reshape scale axis' is not equal to reshape nodes number in graph. There are "
                      << reshape_cnode_num
                      << " reshape nodes to be scaled. Please set correct scale axis for each reshape node";
  }
}

bool CheckUserSettings(const FuncGraphPtr &fg, GpeaInfo *gpea_info) {
  if (parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel) {
    MS_LOG(DEBUG) << "To activate the pass, set_auto_parallel_context 'parallel_mode' should be 'semi_auto_parallel'";
    return false;
  }

  if (!parallel::ParallelContext::GetInstance()->enable_all2all()) {
    MS_LOG(DEBUG) << "To activate the pass, set_auto_parallel_context 'enable_alltoall' should be true";
    return false;
  }

  if (fg->has_flag(kTraining)) {
    MS_LOG(DEBUG) << "To activate the pass, network 'set_train' should be false";
    return false;
  }

  gpea_info->DisplayInfo();

  int64_t gpea_num = gpea_info->GetGroupNum();
  if (gpea_num <= 1 || LongToSize(gpea_num) == GetDeviceNum()) {
    MS_LOG(DEBUG) << "To activate the pass, gpea_num " << gpea_num << " should between (1, " << GetDeviceNum() << ")";
    return false;
  }

  if (GetDeviceNum() % LongToSize(gpea_num) != 0) {
    MS_LOG(DEBUG) << "To activate the pass, device num " << GetDeviceNum() << " should be divisible by gpea_num "
                  << LongToSize(gpea_num);
    return false;
  }
  return true;
}
}  // namespace

size_t GetDeviceNum() { return parallel::g_device_manager->DeviceNum(); }

size_t GetGlobalRankID() { return LongToSize(parallel::g_device_manager->global_rank()); }

void SetGroupedPairwiseExchangeAllToAll(const pipeline::ResourcePtr &resource) {
  if (parallel::g_device_manager == nullptr) {
    MS_LOG(INFO) << "parallel::g_device_manager is not initialized.";
    return;
  }

  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr func_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);

  auto gpea_info = GpeaInfo();
  if (!CheckUserSettings(func_graph, &gpea_info)) {
    return;
  }

  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::list<CNodePtr> orders = func_graph->GetOrderedCnodes();
  std::vector<CNodePtr> origin_nodes_topological(orders.cbegin(), orders.cend());

  std::vector<CNodePtrPair> alltoall_pairs;
  FindAlltoallNodePairs(manager, origin_nodes_topological, &alltoall_pairs);
  MS_LOG(DEBUG) << "Find alltoall_pairs num: " << alltoall_pairs.size();
  if (alltoall_pairs.size() == 0) {
    MS_LOG(WARNING) << "Not find alltoall_pairs, skip the pass";
    return;
  }

  CheckReshapeScaleAxis(origin_nodes_topological, alltoall_pairs[0], &gpea_info);

  CreateAndReplaceGraph(manager, origin_nodes_topological, alltoall_pairs, &gpea_info);
  MS_LOG(DEBUG) << "CreateAndReplaceGraph done";

  // Renormalize, infer shape and set abstract for all nodes in graph
  abstract::AbstractBasePtrList args_abs;
  auto parameters = func_graph->parameters();
  (void)std::transform(parameters.begin(), parameters.end(), std::back_inserter(args_abs),
                       [](const AnfNodePtr &p) -> AbstractBasePtr { return p->abstract(); });
  FuncGraphPtr new_fg = pipeline::Renormalize(resource, func_graph, args_abs);
  resource->set_func_graph(new_fg);
  resource->set_args_abs(args_abs);
  MS_LOG(DEBUG) << "Renormalize done";
  return;
}
}  // namespace opt
}  // namespace mindspore
