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

#include "frontend/parallel/pass/offloading_packed_expert.h"
#include <memory>
#include <queue>
#include <utility>
#include <list>
#include <vector>
#include <string>
#include <algorithm>
#include <set>

#include "primitive/sequence_ops.h"
#include "primitive/other_ops.h"
#include "primitive/array_ops.h"
#include "primitive/framework_ops.h"
#include "utils/anf_utils.h"
#include "include/utils/utils.h"
#include "include/utils/anfalgo.h"
#include "include/utils/parallel_context.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/tensor_layout/tensor_info.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/jit/ps/action.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace parallel {
namespace {
using CNodePtrPair = std::pair<CNodePtr, CNodePtr>;
using OpeInfo = OffloadingPackedExpertInfo;

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
    if (IsPrimitiveCNode(cnode, prim::kPrimAlltoAll)) {
      alltoall_node = cnode;
      break;
    }
    if (cnode->HasAttr("expert_num") && cnode->HasAttr("pe_num")) {
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
    MS_EXCEPTION_IF_NULL(GetCNodePrimitive(marked_node));
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
    if (IsPrimitiveCNode(cnode, prim::kPrimAlltoAll)) {
      alltoall_node = cnode;
      break;
    }

    if (cnode->HasAttr("expert_num") && cnode->HasAttr("pe_num")) {
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
    MS_EXCEPTION_IF_NULL(GetCNodePrimitive(marked_node));
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
                           std::vector<CNodePtrPair> *alltoall_pairs, OpeInfo *ope_info) {
  std::vector<CNodePtr> visited_marked_nodes;
  size_t alltoall_pairs_cnt = 0;
  int64_t pe_num = 1;
  int64_t num_experts = 1;
  for (size_t i = 0; i < origin_nodes_topological.size(); i++) {
    auto cnode = origin_nodes_topological[i];
    if (!IsPrimitiveCNode(cnode)) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(GetCNodePrimitive(cnode));
    if (!GetCNodePrimitive(cnode)->HasAttr("expert_num") || !GetCNodePrimitive(cnode)->HasAttr("pe_num")) {
      continue;
    }
    pe_num = GetValue<int64_t>(GetCNodePrimitive(cnode)->GetAttr("pe_num"));
    num_experts = GetValue<int64_t>(GetCNodePrimitive(cnode)->GetAttr("expert_num"));

    if (std::find(visited_marked_nodes.begin(), visited_marked_nodes.end(), cnode) != visited_marked_nodes.end()) {
      continue;
    }

    visited_marked_nodes.push_back(cnode);
    auto alltoall_pair = FindAlltoallPair(manager, cnode, &visited_marked_nodes);
    if (alltoall_pair.first == nullptr || alltoall_pair.second == nullptr) {
      continue;
    }
    alltoall_pairs_cnt += 1;
    alltoall_pairs->push_back(alltoall_pair);
  }
  ope_info->SetExpertNumAndPeNum(num_experts, pe_num);
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
    MS_LOG(WARNING) << "split_num should not be zero.";
  }
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> split_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplit->name())),
                                          input_node, NewValueNode<int64_t>(split_dim),
                                          NewValueNode<int64_t>(split_num)};
  auto split = input_node->func_graph()->NewCNode(split_inputs);
  MS_EXCEPTION_IF_NULL(split);

  auto dtype = common::AnfAlgo::GetOutputInferDataType(input_node, 0);
  std::vector<TypeId> dtypes(split_num, dtype);
  auto shape = common::AnfAlgo::GetOutputInferShape(input_node, 0);
  // [16, 1, 64, 2048]->[4, 1, 64, 2048]
  shape[split_dim] /= SizeToLong(split_num);
  std::vector<ShapeVector> shapes(split_num, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, split.get());

  split->set_scope(input_node->scope());
  return split;
}

CNodePtr NewConcatNode(const AnfNodePtr &input_node, size_t concat_dim, size_t input_num) {
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> concat_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name())),
                                           input_node, NewValueNode(MakeValue(static_cast<int64_t>(concat_dim)))};
  auto concat = input_node->func_graph()->NewCNode(concat_inputs);
  MS_EXCEPTION_IF_NULL(concat);

  std::vector<TypeId> dtypes = {common::AnfAlgo::GetOutputInferDataType(input_node, 0)};
  auto shape = common::AnfAlgo::GetOutputInferShape(input_node, 0);
  // [2, 2, 64, 2048]
  shape[concat_dim] *= SizeToLong(input_num);
  std::vector<ShapeVector> shapes(1, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, concat.get());

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
    MS_LOG(INTERNAL_EXCEPTION) << "The number of MakeTuple inputs is not equal to sort index number";
  }

  for (size_t i = 0; i < sort_idx.size(); i++) {
    auto getitem = NewTupleGetItemNode(input_node, LongToSize(sort_idx[i]));
    auto graph_input_shape = common::AnfAlgo::GetOutputInferShape(getitem, 0);
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

// We implement SplitLoadNode to debug
void SplitLoadNode(const std::vector<CNodePtr> &calc_cnodes, size_t split_num,
                   std::vector<std::vector<AnfNodePtr>> *split_load_nodes) {
  for (size_t i = 0; i < calc_cnodes.size(); i++) {
    auto _cnode = calc_cnodes[i];
    if (IsPrimitiveCNode(_cnode, prim::kPrimLoad)) {
      size_t split_dim = 0;
      std::vector<AnfNodePtr> getitem_nodes;
      auto split = NewSplitNode(_cnode, split_dim, split_num);
      NewTupleGetItemNodes(split, split_num, &getitem_nodes);

      split_load_nodes->push_back(getitem_nodes);
    }
  }
}

size_t GetSplitIdx(const std::vector<int64_t> &shape, size_t packed_expert_num) {
  size_t idx = 0;
  bool shape_idx_found = false;

  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == SizeToLong(packed_expert_num)) {
      idx = i;
      shape_idx_found = true;
      break;
    }
  }
  if (!shape_idx_found && shape[idx] / SizeToLong(packed_expert_num) == 0) {
    idx = 1;
  }
  return idx;
}

CNodePtr CloneFrontAlltoAllNode(const AnfNodePtr &input_node, const AnfNodePtr &new_input_node,
                                size_t packed_expert_num) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(new_input_node);
  if (packed_expert_num == 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "packed_expert_num should not be zero.";
  }

  std::vector<AnfNodePtr> new_inputs;
  auto input_cnode = input_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(input_cnode);
  auto inputs = input_cnode->inputs();
  for (size_t i = 0; i < inputs.size(); i++) {
    auto input = inputs[i];
    if (input->isa<CNode>()) {
      new_inputs.push_back(new_input_node->cast<CNodePtr>());
    } else if (input->isa<ValueNode>()) {
      ValueNodePtr new_value_node = NewValueNode(GetValueNode(input));
      new_inputs.push_back(new_value_node);
    } else if (input->isa<Parameter>()) {
      new_inputs.push_back(input);
    }
  }

  auto alltoall = input_node->func_graph()->NewCNode(new_inputs);
  MS_EXCEPTION_IF_NULL(alltoall);

  std::vector<TypeId> dtypes = {common::AnfAlgo::GetOutputInferDataType(input_cnode, 0)};
  auto shape = common::AnfAlgo::GetOutputInferShape(input_cnode, 0);
  size_t split_idx = GetSplitIdx(shape, packed_expert_num);
  shape[split_idx] /= SizeToLong(packed_expert_num);
  std::vector<ShapeVector> shapes(1, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, alltoall.get());

  alltoall->set_scope(input_cnode->scope());
  return alltoall;
}

CNodePtr CloneBackAlltoAllNode(const AnfNodePtr &input_node, const AnfNodePtr &new_input_node,
                               size_t packed_expert_num) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(new_input_node);
  if (packed_expert_num == 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "packed_expert_num should not be zero.";
  }

  std::vector<AnfNodePtr> new_inputs;
  auto input_cnode = input_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(input_cnode);
  auto inputs = input_cnode->inputs();
  for (size_t i = 0; i < inputs.size(); i++) {
    auto input = inputs[i];
    if (input->isa<CNode>()) {
      new_inputs.push_back(new_input_node->cast<CNodePtr>());
    } else if (input->isa<ValueNode>()) {
      ValueNodePtr new_value_node = NewValueNode(GetValueNode(input));
      new_inputs.push_back(new_value_node);
    } else if (input->isa<Parameter>()) {
      new_inputs.push_back(input);
    }
  }

  auto alltoall = input_node->func_graph()->NewCNode(new_inputs);
  MS_EXCEPTION_IF_NULL(alltoall);

  std::vector<TypeId> dtypes = {common::AnfAlgo::GetOutputInferDataType(input_cnode, 0)};
  auto shape = common::AnfAlgo::GetOutputInferShape(input_cnode, 0);
  size_t split_idx = GetSplitIdx(shape, packed_expert_num);
  shape[split_idx] /= SizeToLong(packed_expert_num);
  std::vector<ShapeVector> shapes(1, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, alltoall.get());

  alltoall->set_scope(input_cnode->scope());
  return alltoall;
}

CNodePtr CloneReshapeNode(const AnfNodePtr &input_node, mindspore::HashMap<CNodePtr, CNodePtr> *cnode_map,
                          size_t scale_factor, size_t node_idx) {
  MS_EXCEPTION_IF_NULL(input_node);
  if (scale_factor == 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "scale_factor should not be zero.";
  }
  std::vector<AnfNodePtr> new_inputs;
  auto input_cnode = input_node->cast<CNodePtr>();
  if (!IsPrimitiveCNode(input_cnode, prim::kPrimReshape)) {
    MS_LOG(INTERNAL_EXCEPTION) << "input_node should be reshape cnode.";
  }
  auto inputs = input_cnode->inputs();
  for (size_t i = 0; i < inputs.size(); i++) {
    auto input = inputs[i];
    if (input->isa<CNode>()) {
      new_inputs.push_back((*cnode_map)[input->cast<CNodePtr>()]);
    } else if (input->isa<ValueNode>()) {
      ValueNodePtr new_value_node = NewValueNode(GetValueNode(input));
      new_inputs.push_back(new_value_node);
    } else if (input->isa<Parameter>()) {
      MS_LOG(INTERNAL_EXCEPTION) << "The input of reshape cnode should not be parameter, checking please.";
    }
  }

  auto shape_value_node = new_inputs[kIndex2]->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(shape_value_node);

  auto value_ptr = shape_value_node->value();
  MS_EXCEPTION_IF_NULL(value_ptr);

  auto value_tuple_ptr = value_ptr->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(value_tuple_ptr);

  std::vector<ValuePtr> value_ptr_vec = value_tuple_ptr->value();

  size_t scale_dim = 0;
  if (node_idx == kIndex0 || node_idx == kIndex3) {
    scale_dim = 1;
  }

  ShapeVector new_shape;
  for (size_t j = 0; j < value_ptr_vec.size(); j++) {
    auto shape_value = GetValue<int64_t>(value_ptr_vec[j]);
    if (j == scale_dim) {
      auto shape_change = shape_value / SizeToLong(scale_factor);
      if (shape_change > 0) {
        shape_value = shape_change;
      }
    }
    new_shape.push_back(shape_value);
  }
  new_inputs[kIndex2] = NewValueNode(MakeValue(new_shape));

  auto reshape_cnode = input_node->func_graph()->NewCNode(new_inputs);
  MS_EXCEPTION_IF_NULL(reshape_cnode);

  std::vector<TypeId> dtypes = {common::AnfAlgo::GetOutputInferDataType(input_cnode, 0)};
  auto shape = common::AnfAlgo::GetOutputInferShape(input_cnode, 0);
  auto shape_change = shape[scale_dim] / SizeToLong(scale_factor);
  if (shape_change > 0) {
    shape[scale_dim] = shape_change;
  }
  std::vector<ShapeVector> shapes(1, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, reshape_cnode.get());
  reshape_cnode->set_scope(input_cnode->scope());
  return reshape_cnode;
}

CNodePtr CloneCompNode(const AnfNodePtr &input_node, mindspore::HashMap<CNodePtr, CNodePtr> *cnode_map,
                       size_t scale_factor) {
  MS_EXCEPTION_IF_NULL(input_node);
  if (scale_factor == 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "scale_factor should not be zero.";
  }
  std::vector<AnfNodePtr> new_inputs;
  auto input_cnode = input_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(input_cnode);
  auto inputs = input_cnode->inputs();
  for (size_t j = 0; j < inputs.size(); j++) {
    auto input = inputs[j];
    if (input->isa<CNode>()) {
      new_inputs.push_back((*cnode_map)[input->cast<CNodePtr>()]);
    } else if (input->isa<ValueNode>()) {
      ValueNodePtr new_value_node = NewValueNode(GetValueNode(input));
      new_inputs.push_back(new_value_node);
    } else if (input->isa<Parameter>()) {
      new_inputs.push_back(input);
    }
  }

  auto comp_cnode = input_node->func_graph()->NewCNode(new_inputs);
  MS_EXCEPTION_IF_NULL(comp_cnode);

  std::vector<TypeId> dtypes = {common::AnfAlgo::GetOutputInferDataType(input_cnode, 0)};
  auto shape = common::AnfAlgo::GetOutputInferShape(input_cnode, 0);
  size_t split_idx = GetSplitIdx(shape, scale_factor);
  shape[split_idx] /= SizeToLong(scale_factor);
  std::vector<ShapeVector> shapes(1, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, comp_cnode.get());

  comp_cnode->set_scope(input_cnode->scope());
  return comp_cnode;
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
    MS_LOG(INTERNAL_EXCEPTION) << "Can not find alltoall node in origin_nodes_topological";
  }
  return LongToSize(idx);
}

void tranverse_all2all(std::set<size_t> *order_group, const std::vector<CNodePtr> &origin_nodes_topological,
                       const CNodePtr &start_node, size_t front_alltoall_idx, size_t back_alltoall_idx) {
  auto inputs2 = start_node->inputs();
  if (inputs2.empty()) {
    return;
  }

  auto idx_input = FindAlltoallIndex(origin_nodes_topological, start_node);
  if (IsPrimitiveCNode(start_node, prim::kPrimLoad)) {
    if (idx_input < back_alltoall_idx && idx_input > front_alltoall_idx) {
      order_group->insert(idx_input);
    }
    return;
  }

  if (idx_input < front_alltoall_idx) {
    return;
  }

  for (size_t j = 0; j < inputs2.size(); j++) {
    auto input = inputs2[j];
    if (input->isa<CNode>()) {
      auto node_to_find = input->cast<CNodePtr>();
      auto idx = FindAlltoallIndex(origin_nodes_topological, node_to_find);
      if (idx < back_alltoall_idx && idx > front_alltoall_idx) {
        order_group->insert(idx);
      }
      tranverse_all2all(order_group, origin_nodes_topological, node_to_find, front_alltoall_idx, back_alltoall_idx);
    }
  }
}

const std::vector<CNodePtr> FindCNodesAmongAlltoall(const std::vector<CNodePtr> &origin_nodes_topological,
                                                    const CNodePtrPair &alltoall_pair) {
  auto front_alltoall = alltoall_pair.first;
  auto back_alltoall = alltoall_pair.second;
  size_t front_alltoall_idx = FindAlltoallIndex(origin_nodes_topological, front_alltoall);
  size_t back_alltoall_idx = FindAlltoallIndex(origin_nodes_topological, back_alltoall);
  std::vector<CNodePtr> cnodes;
  std::set<size_t> order_group;
  tranverse_all2all(&order_group, origin_nodes_topological, back_alltoall, front_alltoall_idx, back_alltoall_idx);
  for (size_t i = front_alltoall_idx + 1; i < back_alltoall_idx; i++) {
    if (order_group.find(i) != order_group.end()) {
      cnodes.push_back(origin_nodes_topological[i]);
    }
  }
  return cnodes;
}

CNodePtr CreateDependNode(const AnfNodePtr &src_node, const AnfNodePtr &rely_node) {
  std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), src_node, rely_node};
  auto depend_node = src_node->func_graph()->NewCNode(depend_inputs);
  depend_node->set_abstract(src_node->abstract()->Clone());
  return depend_node;
}

void ClonePackedExpertScaledCompGraph(const std::vector<CNodePtr> &old_cnodes,
                                      const std::vector<std::vector<AnfNodePtr>> &split_load_nodes, size_t branch_idx,
                                      const AnfNodePtr &input_node, size_t scale_factor,
                                      std::vector<AnfNodePtr> *new_nodes, std::vector<AnfNodePtr> *allgather_nodes,
                                      std::vector<AnfNodePtr> *back_matmul_compute_branches,
                                      std::vector<CNodePtr> *first_split_comm_branches) {
  mindspore::HashMap<CNodePtr, CNodePtr> cnode_map;
  auto input_cnode = input_node->cast<CNodePtr>();
  auto old_input_node = old_cnodes[0]->input(1);
  auto old_input_cnode = old_input_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(old_input_cnode);
  cnode_map[old_input_cnode] = input_cnode;
  size_t reshape_cnt = 0;
  size_t load_cnode_cnt = 0;
  CNodePtr first_split = nullptr;
  AnfNodePtr last_batchmatmul_node = nullptr;

  for (size_t i = 0; i < old_cnodes.size(); i++) {
    CNodePtr new_cnode;
    auto cnode = old_cnodes[i];

    auto graph_input_shape = common::AnfAlgo::GetOutputInferShape(cnode, 0);

    if (IsPrimitiveCNode(cnode, prim::kPrimLoad)) {
      cnode_map[cnode] = split_load_nodes[load_cnode_cnt][branch_idx]->cast<CNodePtr>();
      new_nodes->push_back(split_load_nodes[load_cnode_cnt][branch_idx]);
      load_cnode_cnt += 1;
      continue;
    } else if (IsPrimitiveCNode(cnode, prim::kPrimReshape)) {
      new_cnode = CloneReshapeNode(cnode, &cnode_map, scale_factor, reshape_cnt);
      reshape_cnt += 1;
    } else {
      new_cnode = CloneCompNode(cnode, &cnode_map, scale_factor);
    }
    if (IsPrimitiveCNode(new_cnode, prim::kPrimAllGather)) {
      allgather_nodes->push_back(new_cnode);
    }

    if (!back_matmul_compute_branches->empty() && IsPrimitiveCNode(new_cnode, prim::kPrimSplit)) {
      if (first_split == nullptr) {
        first_split = new_cnode;
        new_cnode = CreateDependNode(new_cnode, back_matmul_compute_branches->back());
        first_split_comm_branches->push_back(first_split);
      }
    }

    if (IsPrimitiveCNode(new_cnode, prim::kPrimBatchMatMul)) {
      last_batchmatmul_node = new_cnode;
    }
    graph_input_shape = common::AnfAlgo::GetOutputInferShape(new_cnode, 0);
    cnode_map[cnode] = new_cnode;
    new_nodes->push_back(new_cnode);
  }
  back_matmul_compute_branches->push_back(last_batchmatmul_node);
}

CNodePtr CreateReplaceGraph(const FuncGraphManagerPtr &manager, const std::vector<CNodePtr> &origin_nodes_topological,
                            const CNodePtrPair &alltoall_pair, OpeInfo *ope_info) {
  auto front_alltoall = alltoall_pair.first;
  auto back_alltoall = alltoall_pair.second;
  auto graph_input = front_alltoall->input(1);

  size_t front_split_dim = GetSplitDimFromAlltoall(front_alltoall);
  auto graph_input_shape = common::AnfAlgo::GetOutputInferShape(graph_input, 0);
  size_t split_expert_num = LongToSize(graph_input_shape[front_split_dim]);
  auto front_split_for_each_expert = NewSplitNode(graph_input, front_split_dim, split_expert_num);

  std::vector<int64_t> front_reorder_experts_idx = ope_info->GetFrontReorderExpertsIdx();

  std::vector<int64_t> back_reorder_experts_idx = ope_info->GetBackReorderExpertsIdx();

  std::vector<AnfNodePtr> front_reorder_experts_getitem_nodes;
  MakeSortedSplitGetItemNodes(front_split_for_each_expert, front_reorder_experts_idx,
                              &front_reorder_experts_getitem_nodes);
  auto front_reorder_experts_maketuple = NewMakeTupleNode(front_reorder_experts_getitem_nodes);
  auto front_reorder_experts_concat = NewConcatNode(front_reorder_experts_maketuple, front_split_dim, split_expert_num);

  size_t split_packed_expert_num = LongToSize(ope_info->GetPackedExpertNum());
  auto split_for_packed_expert = NewSplitNode(front_reorder_experts_concat, front_split_dim, split_packed_expert_num);

  std::vector<AnfNodePtr> front_all2all_comm_branches;
  std::vector<AnfNodePtr> back_all2all_comm_branches;
  std::vector<AnfNodePtr> allgather_comm_branches;
  std::vector<CNodePtr> first_split_comm_branches;
  std::vector<AnfNodePtr> back_matmul_compute_branches;
  std::vector<AnfNodePtr> packed_expert_branch_output_nodes;
  auto old_comp_cnodes = FindCNodesAmongAlltoall(origin_nodes_topological, alltoall_pair);
  std::vector<std::vector<AnfNodePtr>> split_load_nodes;
  size_t split_load_num = split_packed_expert_num;
  SplitLoadNode(old_comp_cnodes, split_load_num, &split_load_nodes);

  size_t back_concat_dim = GetConcatDimFromAlltoall(back_alltoall);

  for (size_t i = 0; i < split_packed_expert_num; i++) {
    auto getitem_packed_expert_tokens = NewTupleGetItemNode(split_for_packed_expert, i);

    if (i > 0 && allgather_comm_branches.size() > i - 1) {
      getitem_packed_expert_tokens = CreateDependNode(getitem_packed_expert_tokens, allgather_comm_branches[i - 1]);
    }
    auto front_packed_expert_all2all =
      CloneFrontAlltoAllNode(front_alltoall, getitem_packed_expert_tokens, split_packed_expert_num);
    front_all2all_comm_branches.push_back(front_packed_expert_all2all);

    std::vector<AnfNodePtr> new_comp_nodes;
    ClonePackedExpertScaledCompGraph(old_comp_cnodes, split_load_nodes, LongToSize(i), front_packed_expert_all2all,
                                     split_packed_expert_num, &new_comp_nodes, &allgather_comm_branches,
                                     &back_matmul_compute_branches, &first_split_comm_branches);

    auto back_packed_expert_all2all =
      CloneBackAlltoAllNode(back_alltoall, new_comp_nodes.back(), split_packed_expert_num);
    back_all2all_comm_branches.push_back(back_packed_expert_all2all);
    packed_expert_branch_output_nodes.push_back(back_packed_expert_all2all);
  }

  auto maketuple_packed_expert_branch_output = NewMakeTupleNode(packed_expert_branch_output_nodes);

  auto concat_packed_expert_branch_output =
    NewConcatNode(maketuple_packed_expert_branch_output, back_concat_dim, split_packed_expert_num);

  auto back_split_for_each_expert = NewSplitNode(concat_packed_expert_branch_output, back_concat_dim, split_expert_num);

  std::vector<AnfNodePtr> back_reorder_experts_getitem_nodes;
  MakeSortedSplitGetItemNodes(back_split_for_each_expert, back_reorder_experts_idx,
                              &back_reorder_experts_getitem_nodes);
  auto back_reorder_experts_maketuple = NewMakeTupleNode(back_reorder_experts_getitem_nodes);

  auto back_reorder_experts_concat = NewConcatNode(back_reorder_experts_maketuple, back_concat_dim, split_expert_num);

  return back_reorder_experts_concat;
}

void CreateAndReplaceAlltoall(const FuncGraphManagerPtr &manager, const std::vector<CNodePtr> &origin_nodes_topological,
                              const CNodePtrPair &alltoall_pair, OpeInfo *ope_info) {
  auto cnode = CreateReplaceGraph(manager, origin_nodes_topological, alltoall_pair, ope_info);
  (void)manager->Replace(alltoall_pair.second, cnode);
}

void CreateAndReplaceGraph(const FuncGraphManagerPtr &manager, const std::vector<CNodePtr> &origin_nodes_topological,
                           const std::vector<CNodePtrPair> &alltoall_pairs, OpeInfo *ope_info) {
  for (size_t i = 0; i < alltoall_pairs.size(); i++) {
    CreateAndReplaceAlltoall(manager, origin_nodes_topological, alltoall_pairs[i], ope_info);
  }
}

bool CheckUserSettings(const FuncGraphPtr &fg, OpeInfo *ope_info) {
  if (parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel) {
    MS_LOG(INFO) << "To activate the pass, set_auto_parallel_context 'parallel_mode' should be 'semi_auto_parallel'";
    return false;
  }

  if (!parallel::ParallelContext::GetInstance()->enable_all2all()) {
    MS_LOG(INFO) << "To activate the pass, set_auto_parallel_context 'enable_alltoall' should be true";
    return false;
  }

  if (!MsContext::GetInstance()->IsKByKExecutorMode()) {
    MS_LOG(INFO) << "To activate the pass, KByKExecutorMode should be activated";
    return false;
  }

  if (!MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_OFFLOADING_PACKED_EXPERTS)) {
    MS_LOG(INFO) << "To activate the pass, enable_offloading_packed_experts should be activated";
    return false;
  }

  return true;
}
}  // namespace

bool SetOffloadingPackedExpertsForEachGraph(const FuncGraphPtr &func_graph, OpeInfo *ope_info) {
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::list<CNodePtr> orders = func_graph->GetOrderedCnodes();
  std::vector<CNodePtr> origin_nodes_topological(orders.cbegin(), orders.cend());

  std::vector<CNodePtrPair> alltoall_pairs;
  FindAlltoallNodePairs(manager, origin_nodes_topological, &alltoall_pairs, ope_info);
  if (alltoall_pairs.size() == 0) {
    return false;
  }

  CreateAndReplaceGraph(manager, origin_nodes_topological, alltoall_pairs, ope_info);
  return true;
}

bool SetOffloadingPackedExpert(const FuncGraphPtr &func_graph) {
  if (parallel::g_device_manager == nullptr) {
    MS_LOG(INFO) << "parallel::g_device_manager is not initialized.";
    return false;
  }
  MS_LOG(INFO) << " pass if (parallel::g_device_manager == nullptr)";
  MS_EXCEPTION_IF_NULL(func_graph);

  auto ope_info = OpeInfo();
  if (!CheckUserSettings(func_graph, &ope_info)) {
    MS_LOG(INFO) << " CheckUserSettings_not_pass";
    return false;
  }
  MS_LOG(INFO) << " pass CheckUserSettings(func_graph, &ope_info)";

  auto manager = func_graph->manager();
  auto graphs = manager->func_graphs();
  MS_LOG(DEBUG) << "subgraphs size: " << graphs.size();

  bool res = false;
  auto it = std::find_if(graphs.begin(), graphs.end(), [&](const auto &each_graph) {
    return SetOffloadingPackedExpertsForEachGraph(each_graph, &ope_info);
  });
  if (it != graphs.end()) {
    res = true;
  }
  return res;
}
}  // namespace parallel
}  // namespace mindspore
