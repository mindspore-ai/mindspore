/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "parallel/step_parallel.h"

#include <inttypes.h>
#include <sys/time.h>
#include <algorithm>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include "ir/tensor.h"
#include "ir/param_value_py.h"
#include "operator/ops.h"
#include "optimizer/optimizer.h"
#include "parallel/auto_parallel/graph_costmodel.h"
#include "parallel/context.h"
#include "parallel/device_manager.h"
#include "parallel/dynamic_creator.h"
#include "parallel/graph_util/generate_graph.h"
#include "parallel/graph_util/graph_info.h"
#include "parallel/graph_util/node_info.h"
#include "parallel/node_check.h"
#include "parallel/ops_info/matmul_info.h"
#include "parallel/strategy_checkpoint/parallel_strategy_checkpoint.h"
#include "utils/comm_manager.h"
#include "utils/symbolic.h"

using mindspore::tensor::Tensor;

namespace mindspore {
namespace parallel {
static const std::set<std::string> COMMUNICATION_OPS = {ALL_REDUCE, ALL_GATHER, ALL_TO_ALL, REDUCE_SCATTER};
static const std::set<std::string> INVALID_LOSS_OPS = {GET_NEXT, VIRTUALLOSS};
// g_RefMap, for CNode B input i is a RefKey[Parameter C],
// it will be one item in map with key: C, and value: (B, i)
static std::map<AnfNodePtr, std::pair<AnfNodePtr, int>> g_RefMap;

void SetCommunicationOpGroupLabel(std::vector<AnfNodePtr> new_node_input) {
  if (new_node_input.empty()) {
    return;
  }

  ValueNodePtr prim_anf_node = new_node_input[0]->cast<ValueNodePtr>();
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);
  MS_EXCEPTION_IF_NULL(prim);

  auto attrs = prim->attrs();
  auto iter = attrs.find(GROUP);
  if (iter != attrs.end()) {
    auto value = iter->second;
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<StringImm>()) {
      std::string hash_name = value->cast<StringImmPtr>()->value();
      MS_EXCEPTION_IF_NULL(g_device_manager);
      std::string rank_list_name = g_device_manager->FindRankListNameByHashName(hash_name);
      (void)prim->AddAttr(GROUP_RANKS, MakeValue(rank_list_name));
    }
  }
}

std::vector<AnfNodePtr> CreateInput(const Operator &op, const AnfNodePtr &node, const std::string &instance_name) {
  MS_EXCEPTION_IF_NULL(node);
  OperatorArgs arg_forward = op.second;
  ValuePtr pyop_instance = CreatOpInstance(arg_forward.first, op.first, instance_name);
  MS_EXCEPTION_IF_NULL(pyop_instance);
  OperatorParams params = arg_forward.second;

  std::vector<AnfNodePtr> new_node_input = {NewValueNode(pyop_instance), node};
  if (!params.empty()) {
    for (auto &param : params) {
      AnfNodePtr val = NewValueNode(param.first.second);
      MS_EXCEPTION_IF_NULL(val);
      int32_t position = param.second;
      (void)new_node_input.insert(new_node_input.begin() + position, val);
    }
  }

  // if the op have 'group' attr, set the rank list name for the op
  SetCommunicationOpGroupLabel(new_node_input);
  return new_node_input;
}

void InsertNode(const Operator &op, const CNodePtr &node, size_t index, const AnfNodePtr &pre_node,
                const FuncGraphPtr &func_graph, const std::string &instance_name) {
  // insert new node before the node
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  ScopePtr scope = node->scope();
  MS_EXCEPTION_IF_NULL(scope);
  std::vector<AnfNodePtr> node_input = CreateInput(op, pre_node, instance_name);
  CNodePtr new_node = func_graph->NewCNode(node_input);
  MS_EXCEPTION_IF_NULL(new_node);
  if (instance_name.find(SPLIT_SENS) == std::string::npos) {
    new_node->set_in_forward_flag(true);  // mark forward flag
  }
  auto new_node_value = node_input[0]->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(new_node_value);
  PrimitivePtr new_node_prim = new_node_value->value()->cast<PrimitivePtr>();
  new_node_prim->set_instance_name(instance_name);
  new_node_prim->set_attr("keep_value_node_input", MakeValue(true));
  new_node->set_scope(scope);
  node_input[0]->set_scope(scope);
  manager->SetEdge(node, SizeToInt(index), new_node);
}

std::string CreateInstanceName(const CNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsValueNode<Primitive>(node->input(0))) {
    MS_LOG(EXCEPTION) << "CreateInstanceName: " << node->ToString() << " doesn't have primitive";
  }
  std::string name_base = node->fullname_with_scope();
  std::string name = name_base + "_" + std::to_string(index);
  std::string instance_name = HashInstanceName(name);
  return instance_name;
}

void ForwardCommunication(OperatorVector forward_op, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  // step1:get graph manager distribute_operator
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto uses_set = manager->node_users()[node];
  CNodePtr node_to_insert = node;
  for (auto &uses_pair : uses_set) {
    auto uses_cnode = uses_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(uses_cnode);
    if (!IsValueNode<Primitive>(uses_cnode->input(0))) {
      break;
    }
    PrimitivePtr value_node_prim = GetValueNode<PrimitivePtr>(uses_cnode->input(0));
    MS_EXCEPTION_IF_NULL(value_node_prim);
    if (value_node_prim->name() == TUPLE_GETITEM) {
      if (uses_set.size() > 1) {
        MS_LOG(EXCEPTION) << "Now only support one output, but got " << uses_set.size();
      }
      node_to_insert = uses_cnode;
    }
  }
  MS_EXCEPTION_IF_NULL(node_to_insert);
  std::reverse(forward_op.begin(), forward_op.end());

  // step2:traverse op_list and insert node
  for (size_t index = 0; index < forward_op.size(); ++index) {
    std::string instance_name_base = FORWARD_OP;
    std::string instance_name = instance_name_base + "_" + CreateInstanceName(node, index);
    std::vector<AnfNodePtr> forward_input = CreateInput(forward_op[index], node_to_insert, instance_name);
    CNodePtr forward_node = func_graph->NewCNode(forward_input);  // using NewCNode to creat anfnode
    MS_EXCEPTION_IF_NULL(forward_node);
    ScopePtr scope = node->scope();
    MS_EXCEPTION_IF_NULL(scope);
    forward_node->set_scope(scope);
    forward_node->set_in_forward_flag(true);
    forward_input[0]->set_scope(scope);
    (void)manager->Replace(node_to_insert, forward_node);  // using Replace function to insert node
  }
}

CNodePtr InsertMakeTuple(const AnfNodePtr &prev, uint32_t num, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(prev);
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> make_tuple_inputs;
  make_tuple_inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
  for (uint32_t i = 0; i < num; i++) {
    std::vector<AnfNodePtr> tuple_get_item_inputs{NewValueNode(prim::kPrimTupleGetItem), prev,
                                                  CreatInt32Imm(UintToInt(i))};
    auto tuple_get_item = func_graph->NewCNode(tuple_get_item_inputs);
    MS_EXCEPTION_IF_NULL(tuple_get_item);
    make_tuple_inputs.push_back(tuple_get_item);
  }
  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(prev, make_tuple);
  return make_tuple;
}

void InsertRedistribution(const RedistributionOpListPtr &redistribution_oplist_ptr, const CNodePtr &node,
                          const FuncGraphPtr &func_graph, int pos, const CNodePtr &pre_node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(pre_node);
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if ((redistribution_oplist_ptr->first).size() != (redistribution_oplist_ptr->second).size()) {
    MS_LOG(EXCEPTION) << "size of OperatorVector and OutPutInfoVector must be the same!";
  }
  for (size_t index = 0; index < (redistribution_oplist_ptr->first).size(); ++index) {
    if (pos >= SizeToInt(node->inputs().size())) {
      MS_LOG(EXCEPTION) << "InsertRedistribution:pos can't be larger than node's inputs'size";
    }
    // Creat new node
    AnfNodePtr target_node = node->input(IntToSize(pos));
    MS_EXCEPTION_IF_NULL(target_node);
    // Creat instance_name
    auto op = (redistribution_oplist_ptr->first)[index];
    std::string op_name = (redistribution_oplist_ptr->first)[index].first;
    std::string instance_name_base = REDISTRIBUTION_OP;
    std::string instance_name = instance_name_base + "_" + CreateInstanceName(pre_node, index) + op_name;
    InsertNode(op, node, IntToSize(pos), target_node, func_graph, instance_name);
    if ((redistribution_oplist_ptr->second)[index].first) {
      target_node = node->input(IntToSize(pos));
      MS_EXCEPTION_IF_NULL(target_node);
      (void)InsertMakeTuple(target_node, (redistribution_oplist_ptr->second)[index].second, func_graph);
    }
  }
}

void InsertGetTensorSliceOp(const Operator &op, const CNodePtr &node, const FuncGraphPtr &func_graph, int pos,
                            const std::string &instance_name) {
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "InsertGetTensorSliceOp: the graph is null, the instance name is " << instance_name;
  }

  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (pos >= SizeToInt(node->inputs().size())) {
    MS_LOG(EXCEPTION) << "InsertGetTensorSliceOp: pos can't be larger than node's inputs'size, the instance name is "
                      << instance_name;
  }
  // Creat new node
  AnfNodePtr pre_node = node->input(IntToSize(pos));
  MS_EXCEPTION_IF_NULL(pre_node);
  InsertNode(op, node, IntToSize(pos), pre_node, func_graph, instance_name);
}

TensorLayout GetTensorInLayout(const CNodePtr &middle_node, const PrimitivePtr &middle_prim,
                               const OperatorInfoPtr &distribute_operator) {
  TensorInfo tensorinfo_in;
  if (middle_prim->name() == TUPLE_GETITEM) {
    auto value_node = middle_node->input(2)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    size_t index_s = IntToSize(GetValue<int>(value_node->value()));
    if (index_s >= distribute_operator->outputs_tensor_info().size()) {
      MS_LOG(EXCEPTION) << "The index out of range, index: " << index_s
                        << ", vector size: " << distribute_operator->outputs_tensor_info().size();
    }
    tensorinfo_in = distribute_operator->outputs_tensor_info()[index_s];
  } else {
    if (distribute_operator->outputs_tensor_info().empty()) {
      MS_LOG(EXCEPTION) << "The outputs tensor info is empty";
    }
    tensorinfo_in = distribute_operator->outputs_tensor_info()[0];
  }
  return tensorinfo_in.tensor_layout();
}

OperatorInfoPtr GetDistributeOperator(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsParallelCareNode(node)) {
    return nullptr;
  }
  OperatorInfoPtr distribute_operator = node->operator_info();
  if (distribute_operator == nullptr) {
    MS_LOG(EXCEPTION) << "GetDistributeOperator:distribute_operator is nullptr";
  }
  return distribute_operator;
}

void Redistribution(const std::pair<AnfNodePtr, int> &node_pair, const OperatorInfoPtr &distribute_operator,
                    const CNodePtr &middle_node, int index, TensorRedistribution tensor_redistribution,
                    const CNodePtr &pre_node) {
  FuncGraphPtr func_graph = middle_node->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Redistribution:get graph failed";
  }
  CNodePtr next_node = node_pair.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(next_node);
  auto middle_value = middle_node->input(0)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(middle_value);
  PrimitivePtr middle_prim = middle_value->value()->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(middle_prim);
  OperatorInfoPtr next_distribute_operator = GetDistributeOperator(next_node);
  if (next_distribute_operator == nullptr) {
    MS_LOG(EXCEPTION) << "Failure: " << next_node->ToString() << " GetDistributeOperator failed";
  }
  RankList dev_list = distribute_operator->global_device_list();
  std::string next_prim_name = GetValueNode<PrimitivePtr>(next_node->input(0))->name();
  MS_LOG(DEBUG) << "Redistribution: middle_prim " << middle_prim->name() << " next_prim " << next_prim_name;
  MS_LOG(DEBUG) << "Redistribution: middle_node " << middle_node->ToString() << " next_node " << next_node->ToString();
  // extract tensor layout in and out
  if (distribute_operator->outputs_tensor_info().empty()) {
    MS_LOG(EXCEPTION) << "Failure:pre_node's tensorinfo_in is empty";
  }

  if (IntToSize(index - 1) >= next_distribute_operator->inputs_tensor_info().size()) {
    MS_LOG(EXCEPTION) << "The index is out of range, the index is " << index - 1 << ", the vector size is "
                      << next_distribute_operator->inputs_tensor_info().size();
  }
  TensorInfo tensorinfo_out = next_distribute_operator->inputs_tensor_info()[IntToSize(index - 1)];
  TensorLayout tensorlayout_out = tensorinfo_out.tensor_layout();
  TensorLayout tensorlayout_in = GetTensorInLayout(middle_node, middle_prim, distribute_operator);
  if (tensor_redistribution.Init(tensorlayout_in, tensorlayout_out, dev_list) == FAILED) {
    MS_LOG(ERROR) << "Redistribution: middle_prim " << middle_prim->name() << " next_prim : " << next_prim_name;
    MS_LOG(ERROR) << "Redistribution: middle_node " << middle_node->ToString() << " next_node "
                  << next_node->ToString();
    DumpGraph(func_graph, "redistribution_error");
    MS_LOG(EXCEPTION) << "Failure:tensor_redistribution init failed";
  }
  RedistributionOpListPtr redistribution_oplist_ptr = tensor_redistribution.InferTensorRedistributionOperatorList();
  if (redistribution_oplist_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:InferTensorRedistribution failed";
  }
  MS_LOG(DEBUG) << "Redistribution size " << redistribution_oplist_ptr->first.size();
  if (!redistribution_oplist_ptr->first.empty()) {
    // insert node before next node
    InsertRedistribution(redistribution_oplist_ptr, next_node, func_graph, node_pair.second, pre_node);
  }
}

bool StrategyFound(std::unordered_map<std::string, ValuePtr> attrs) {
  auto iter = attrs.find(STRATEGY);
  return !((iter == attrs.end()) || (iter->second->type_name() == NONE));
}

bool HasStrategy(const FuncGraphPtr &root) {
  AnfNodePtr ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);

  for (auto &node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if ((cnode == nullptr) || !IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }

    ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);
    auto attrs = prim->attrs();
    if (StrategyFound(attrs)) {
      return true;
    }
  }

  return false;
}

bool IsCommunicationOp(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  return (COMMUNICATION_OPS.find(prim->name()) != COMMUNICATION_OPS.end());
}

bool FindCommunicationOp(const std::vector<AnfNodePtr> &all_nodes) {
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    ValueNodePtr prim_value_node = cnode->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prim_value_node);
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_value_node);
    MS_EXCEPTION_IF_NULL(prim);

    if (IsCommunicationOp(prim) && cnode->in_forward_flag()) {
      MS_EXCEPTION_IF_NULL(prim_value_node->scope());
      MS_LOG(INFO) << "The graph contain communication op: " << prim->name() << ", scope name is "
                   << prim_value_node->scope()->name();
      return true;
    }
  }
  return false;
}

bool IsParallelCareNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  ValueNodePtr prim_node = cnode->input(0)->cast<ValueNodePtr>();
  if (prim_node == nullptr) {
    return false;
  }
  PrimitivePtr prim = prim_node->value()->cast<PrimitivePtr>();
  if (prim == nullptr) {
    return false;
  }
  if (IsInBlackList(prim)) {
    MS_LOG(INFO) << "Parallel don't care node: " << prim->name();
    return false;
  }
  // get_next is not in the forward graph, we need mark the get_next as the forward node
  if (prim->name() == GET_NEXT) {
    return true;
  }
  if ((prim->name() == CAST) && (cnode->operator_info() == nullptr)) {
    return false;
  }

  return cnode->in_forward_flag();
}

void StepRedistribution(const CNodePtr &node, const OperatorInfoPtr &distribute_operator, const CNodePtr &insert_node,
                        const TensorRedistribution &tensor_redistribution, const CNodePtr &pre_node) {
  MS_EXCEPTION_IF_NULL(node->func_graph());
  FuncGraphManagerPtr manager = node->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodeIndexSet node_set = manager->node_users()[node];
  CNodePtr insert_node_new;
  if (IsValueNode<Primitive>(node->input(0))) {
    auto current_value = node->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(current_value);
    PrimitivePtr current_prim = current_value->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(current_prim);
    insert_node_new = ((current_prim->name() == TUPLE_GETITEM) ? node : insert_node);
  } else {
    insert_node_new = insert_node;
  }
  MS_EXCEPTION_IF_NULL(insert_node_new);
  for (auto &node_pair : node_set) {
    CNodePtr use_cnode = node_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(use_cnode);
    if (!IsValueNode<Primitive>(use_cnode->input(0))) {
      StepRedistribution(use_cnode, distribute_operator, insert_node_new, tensor_redistribution, pre_node);
    } else {
      ValueNodePtr prim_anf_node = use_cnode->input(0)->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(prim_anf_node);
      PrimitivePtr node_prim = prim_anf_node->value()->cast<PrimitivePtr>();
      MS_EXCEPTION_IF_NULL(node_prim);
      if (node_prim->name() == DEPEND && node_pair.second != 1) {
        continue;
      }
      if (IsParallelCareNode(use_cnode) && (use_cnode->operator_info() != nullptr)) {
        Redistribution(node_pair, distribute_operator, insert_node_new, node_pair.second, tensor_redistribution,
                       pre_node);
      } else {
        StepRedistribution(use_cnode, distribute_operator, insert_node_new, tensor_redistribution, pre_node);
      }
    }
  }
}

void SplitTensor(const AnfNodePtr &node, const CNodePtr &next_node, int index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(next_node);
  OperatorInfoPtr op_info = next_node->operator_info();
  MS_EXCEPTION_IF_NULL(op_info);

  // If the shape of tensor is [] or [1], no need to split it.
  Shapes shapes = GetNodeShape(node);
  if (shapes.size() != 1) {
    MS_LOG(EXCEPTION) << "Split tensor for " << op_info->name()
                      << ": GetNodeShape for tensor_node, output size is not 1";
  }
  Shape shape = shapes[0];
  std::string shape_str = ShapeToString(shape);
  if (shape.empty() || ((shape.size() == 1) && (shape[0] == 1))) {
    MS_LOG(INFO) << "Split tensor for " << op_info->name() << ": The shape is " << shape_str
                 << ", no need to split it.";
    return;
  }

  MS_LOG(INFO) << "Split tensor for " << op_info->name() << ": The shape of tensor is " << shape_str;

  // extract tensor layout
  if (IntToSize(index - 1) >= op_info->inputs_tensor_info().size()) {
    MS_LOG(EXCEPTION) << "The index is out of range, index is  " << index - 1 << ", vector size is  "
                      << op_info->inputs_tensor_info().size();
  }
  TensorInfo tensor_info = op_info->inputs_tensor_info()[IntToSize(index - 1)];
  TensorLayout tensor_layout = tensor_info.tensor_layout();

  // Use _GetTensorSlice operator to split the tensor
  FuncGraphPtr func_graph = next_node->func_graph();  // only cnode can get the graph
  MS_EXCEPTION_IF_NULL(func_graph);
  Operator op = CreateGetTensorSliceOp(tensor_layout);
  InsertGetTensorSliceOp(op, next_node, func_graph, index, SPLIT_TENSOR);
  if (!op_info->sub_ops().empty()) {
    auto sub_ops = op_info->sub_ops();
    for (size_t i = 0; i < sub_ops.size(); i++) {
      if (!sub_ops.at(i).empty()) {
        InsertGetTensorSliceOp(sub_ops.at(i).at(0), next_node, func_graph, index, SUB);
      }
    }
  }
}

void StepSplitTensor(const AnfNodePtr &node, const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodeIndexSet node_set = manager->node_users()[node];
  for (auto &node_pair : node_set) {
    CNodePtr use_cnode = node_pair.first->cast<CNodePtr>();
    if (use_cnode == nullptr || !IsValueNode<Primitive>(use_cnode->input(0))) {
      continue;
    }
    ValueNodePtr prim_anf_node = use_cnode->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prim_anf_node);
    PrimitivePtr use_cnode_prim = prim_anf_node->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(use_cnode_prim);
    if (use_cnode_prim->name() == DEPEND && node_pair.second != 1) {
      continue;
    }
    if (IsParallelCareNode(use_cnode)) {
      SplitTensor(node, use_cnode, node_pair.second);
    }
  }
}

std::vector<AnfNodePtr> ReplaceOpInput(const Operator &replace_op, const std::string &instance_name,
                                       const CNodePtr &node) {
  OperatorArgs arg_replace_op = replace_op.second;
  ValuePtr pyop_instance = CreatOpInstance(arg_replace_op.first, replace_op.first, instance_name);
  if (pyop_instance == nullptr) {
    MS_LOG(EXCEPTION) << "Failure: " << replace_op.first << " CreatOpInstance failed";
  }
  OperatorParams params = arg_replace_op.second;
  if (node->inputs().size() < 2) {
    // GetNext operator dose not has input
    if (node->inputs().size() == 1) {
      return {NewValueNode(pyop_instance)};
    }
    MS_LOG(EXCEPTION) << "Failure: " << node->ToString() << " size is smaller than 2";
  }
  std::vector<AnfNodePtr> replace_input = {NewValueNode(pyop_instance), node->input(1)};
  auto prim = GetValueNode<PrimitivePtr>(node->input(0));
  if (prim->name() == GATHERV2 || prim->name() == SPARSE_GATHERV2) {
    replace_input = {NewValueNode(pyop_instance), node->input(1), node->input(2)};
  }
  if (!params.empty()) {
    Param param_first = *(params.begin());
    int32_t first_position = param_first.second;
    if (first_position == 1) {
      replace_input.pop_back();
    }
    for (auto &param : params) {
      AnfNodePtr val = NewValueNode(param.first.second);
      if (val == nullptr) {
        MS_LOG(EXCEPTION) << "Failure:val is nullptr";
      }
      int32_t position = param.second;
      (void)replace_input.insert(replace_input.begin() + position, val);
    }
  }

  return replace_input;
}

void ReplaceOneOp(const Operator &replace_op, const CNodePtr &node) {
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:AddNode error since manager is nullptr";
  }
  std::string instance_name = CreateInstanceName(node, 0);
  std::vector<AnfNodePtr> replace_input;
  replace_input = ReplaceOpInput(replace_op, instance_name, node);
  CNodePtr replace_node = func_graph->NewCNode(replace_input);
  MS_EXCEPTION_IF_NULL(replace_node);
  ScopePtr scope = node->scope();
  MS_EXCEPTION_IF_NULL(scope);
  replace_node->set_scope(scope);
  replace_node->set_in_forward_flag(true);
  replace_input[0]->set_scope(scope);
  (void)manager->Replace(node, replace_node);
}

void StepReplaceOp(OperatorVector replace_op, const CNodePtr &node) {
  // step1:get graph manager distribute_operator
  OperatorInfoPtr distribute_operator = node->operator_info();
  if (distribute_operator == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:AddNode error since distribute_operator is nullptr";
  }
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:AddNode error since manager is nullptr";
  }
  // step2:traverse op_list and insert node
  std::reverse(replace_op.begin(), replace_op.end());
  auto replace_op_info = distribute_operator->replace_op_info();
  std::reverse(replace_op_info.begin(), replace_op_info.end());
  if (!replace_op_info.empty() && replace_op_info.size() != replace_op.size()) {
    MS_LOG(EXCEPTION) << "replace_op_info is not empty and size not equal to replace_op!";
  }
  bool replace_op_info_flag = !replace_op_info.empty();
  for (size_t index = 0; index < replace_op.size(); ++index) {
    std::string instance_name = CreateInstanceName(node, index);
    std::vector<AnfNodePtr> replace_input;
    if (index != replace_op.size() - 1) {
      replace_input = CreateInput(replace_op[index], node, instance_name);
    } else {
      replace_input = ReplaceOpInput(replace_op[index], instance_name, node);
    }
    CNodePtr replace_node = func_graph->NewCNode(replace_input);
    MS_EXCEPTION_IF_NULL(replace_node);
    ScopePtr scope = node->scope();
    MS_EXCEPTION_IF_NULL(scope);
    replace_node->set_scope(scope);
    if (index == replace_op.size() - 1) {
      (void)replace_node->set_operator_info(node->operator_info());
    }
    replace_node->set_in_forward_flag(true);
    replace_input[0]->set_scope(scope);
    if (replace_op_info_flag && replace_op_info[index].first) {
      auto new_cnode = InsertMakeTuple(replace_node, replace_op_info[index].second, func_graph);
      (void)manager->Replace(node, new_cnode);  // using Replace function to insert node
    } else {
      (void)manager->Replace(node, replace_node);  // using Replace function to insert node
    }
  }
  MS_LOG(INFO) << "Insert ReplaceOp success for " << distribute_operator->name();
}

bool IsSomePrimitive(const CNodePtr &cnode, const std::string &name) {
  ValueNodePtr anf_node = cnode->input(0)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(anf_node);
  PrimitivePtr prim = anf_node->value()->cast<PrimitivePtr>();
  return (prim->name() == name);
}

void StepReplaceGraph(const ReplaceGraphPtr &replace_graph, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(replace_graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(replace_graph->second);
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:AddNode error since manager is nullptr";
  }
  for (auto &replace_input : replace_graph->first) {
    auto pre_node = node->input(IntToSize(replace_input.second));
    manager->SetEdge(replace_input.first, 1, pre_node);
  }
  //  "(void)manager->Replace(replace_graph->first, pre_node);" can not be called
  auto replace_output = replace_graph->second;
  MS_EXCEPTION_IF_NULL(replace_output);
  (void)manager->Replace(node, replace_output);
}

int32_t GetTupleGetItemIndex(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().size() != 3) {
    MS_LOG(EXCEPTION) << cnode->ToString() << " size( " << cnode->inputs().size() << " ) is not 3";
  }

  if (!cnode->input(2)->isa<ValueNode>()) {
    MS_LOG(EXCEPTION) << "The index of tuple getitem is not a value node";
  }

  ValuePtr tuple_index_value = GetValueNode(cnode->input(2));
  MS_EXCEPTION_IF_NULL(tuple_index_value);
  if (!tuple_index_value->isa<Int32Imm>()) {
    MS_LOG(EXCEPTION) << "The index of tuple getitem is not int32";
  }
  return tuple_index_value->cast<Int32ImmPtr>()->value();
}

// Judge whether the node is a loss, and if there are multiple outputs,
// get which output is a grad according to the tuple getitem.
// Currently, it is not supported that the sens is a tuple.
LossNodeInfo GetLossNodeInfo(const AnfNodePtr &loss_node) {
  MS_EXCEPTION_IF_NULL(loss_node);
  FuncGraphPtr sub_graph = loss_node->func_graph();
  MS_EXCEPTION_IF_NULL(sub_graph);
  CNodePtr return_node = sub_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  if (return_node->inputs().size() < 2) {
    MS_LOG(EXCEPTION) << "Failure: " << return_node->ToString() << " size is smaller than 2";
  }
  AnfNodePtr pre_node = return_node->input(1);
  MS_EXCEPTION_IF_NULL(pre_node);

  LossNodeInfo node_info;

  // return -> cast
  auto pre_cnode = pre_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(pre_cnode);
  auto pre_prim = GetValueNode<PrimitivePtr>(pre_cnode->input(0));
  if (pre_prim->name() == CAST && pre_cnode->operator_info() == nullptr) {
    pre_node = pre_cnode->input(1);
  }

  // return -> loss
  if (pre_node == loss_node) {
    node_info.has_tuple_getitem = false;
    node_info.dout_index = 0;
    return node_info;
  }

  // return -> tuple_getitem -> loss
  auto cnode = pre_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto current_value = cnode->input(0)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(current_value);
  PrimitivePtr current_prim = current_value->value()->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(current_prim);
  // size of common cnode is larger than 1
  if (cnode->inputs().size() < 2) {
    MS_LOG(EXCEPTION) << cnode->ToString() << " size( " << cnode->inputs().size() << " ) is smaller than 2";
  }

  if ((current_prim->name() == TUPLE_GETITEM) && (cnode->input(1) == loss_node)) {
    // size of tuple_getitem cnode is 3
    auto tuple_index = GetTupleGetItemIndex(cnode);
    node_info.has_tuple_getitem = true;
    node_info.dout_index = tuple_index;
    return node_info;
  }

  MS_LOG(EXCEPTION) << "Invalid loss";
}

void InsertVirtualDivOp(const VirtualDivOp &virtual_div_op, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  size_t node_size = node->inputs().size();
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  for (size_t index = 1; index < node_size; ++index) {
    AnfNodePtr input = node->input(index);
    MS_EXCEPTION_IF_NULL(input);
    if (!input->isa<CNode>() && !input->isa<Parameter>()) {  // if it is not a tensor, continue
      MS_LOG(INFO) << "insert div op: the index  " << index << "  is not tensor, skip";
      continue;
    }

    for (size_t pos = 0; pos < virtual_div_op.size(); ++pos) {
      std::string instance_name = CreateInstanceName(node, pos);
      InsertNode(virtual_div_op[pos], node, index, node->input(index), func_graph, instance_name);
    }
    MS_LOG(INFO) << "insert div op for input index  " << index << "  of node";
  }
}

std::pair<AnfNodePtr, bool> FindParameter(const AnfNodePtr &node, const FuncGraphPtr &func_graph) {
  if (!node->isa<Parameter>() && !node->isa<CNode>() && !node->isa<ValueNode>()) {
    return std::make_pair(nullptr, false);
  } else if (node->isa<Parameter>()) {
    return std::make_pair(node, false);
  } else if (node->isa<ValueNode>()) {
    if (IsValueNode<RefKey>(node)) {
      std::vector<AnfNodePtr> param_v = FindParameterByRefKeyNode(node, func_graph);
      if (param_v.size() != 1) {
        MS_LOG(EXCEPTION) << "FindParameterByRefKeyNode failed, return vector size must be 1, real is  "
                          << param_v.size();
      }
      return std::make_pair(node, true);
    }
    return std::make_pair(nullptr, false);
  } else {
    CNodePtr cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!IsValueNode<Primitive>(cnode->input(0))) {
      for (size_t index = 0; index < cnode->inputs().size(); ++index) {
        if (!FindParameter(cnode->input(index), func_graph).first) {
          continue;
        }
        return FindParameter(cnode->input(index), func_graph);
      }
    } else {
      if (IsParallelCareNode(cnode)) {
        return std::make_pair(nullptr, false);
      } else {
        ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
        MS_EXCEPTION_IF_NULL(prim_anf_node);
        for (size_t index = 0; index < cnode->inputs().size(); ++index) {
          PrimitivePtr prim = prim_anf_node->value()->cast<PrimitivePtr>();
          MS_EXCEPTION_IF_NULL(prim);
          if (prim->name() == DEPEND && index != 1) {
            continue;
          }
          if (!FindParameter(cnode->input(index), func_graph).first) {
            continue;
          }
          return FindParameter(cnode->input(index), func_graph);
        }
      }
    }
  }
  return std::make_pair(nullptr, false);
}

std::pair<bool, CNodePtr> FindCNode(const AnfNodePtr &anode, const std::string &name, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(anode);
  MS_EXCEPTION_IF_NULL(anode->func_graph());
  FuncGraphManagerPtr manager = anode->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodeIndexSet node_set = manager->node_users()[anode];
  bool result = false;
  CNodePtr cnode_return = nullptr;
  for (auto &node_pair : node_set) {
    CNodePtr use_apply = node_pair.first->cast<CNodePtr>();
    if (use_apply == nullptr || !IsValueNode<Primitive>(use_apply->input(0))) {
      continue;
    }
    ValueNodePtr prim_anf_node = use_apply->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prim_anf_node);
    PrimitivePtr node_prim = prim_anf_node->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(node_prim);
    if (node_prim->name() == name && node_pair.second == 1) {
      if (use_apply->func_graph() == func_graph) {
        result = true;
        cnode_return = use_apply;
        MS_LOG(INFO) << "Find Primitive " << name << " in the same func_graph";
        continue;
      }
      MS_LOG(INFO) << "Find Primitive " << name << " in different func_graph";
    }
  }
  return std::make_pair(result, cnode_return);
}

bool IsCastBeforMirror(const CNodePtr &node, size_t index) {
  // only if cast_before_mirror is true, pre node is cast and type is not float32 return true
  if (!ParallelContext::GetInstance()->cast_before_mirror()) {
    return false;
  }
  auto pre_node = node->input(index);
  MS_EXCEPTION_IF_NULL(pre_node);
  auto cnode = pre_node->cast<CNodePtr>();
  if (cnode == nullptr || !IsValueNode<Primitive>(cnode->input(0))) {
    return false;
  }
  auto pre_value_node = cnode->input(0)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(pre_value_node);
  auto pre_prim = pre_value_node->value()->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(pre_prim);
  if (pre_prim->name() != CAST) {
    return false;
  }
  auto node_type = pre_node->Type();
  MS_EXCEPTION_IF_NULL(node_type);
  if (!node_type->isa<mindspore::TensorType>()) {
    MS_LOG(EXCEPTION) << "Unknown type.";
  }
  auto input_element_type = node_type->cast<mindspore::TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(input_element_type);
  auto type_id = input_element_type->type_id();

  return (type_id != kNumberTypeFloat32);
}

void InsertMirrorOps(const MirrorOps &mirror_ops, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  size_t node_size = node->inputs().size();
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (mirror_ops.size() != node_size - 1) {
    MS_LOG(EXCEPTION) << "Failure:Mirrorops's size is wrong! mirror_ops size is  " << mirror_ops.size()
                      << ", node_size is  " << node_size;
  }
  for (size_t index = 1; index < node_size; ++index) {
    OperatorVector backward_op = mirror_ops[index - 1];
    if (backward_op.empty()) {
      continue;
    }
    std::pair<AnfNodePtr, bool> param_node_pair = FindParameter(node->input(index), func_graph);
    if (!param_node_pair.first) {
      continue;
    }
    // not a RefKey
    if (!param_node_pair.second) {
      auto next_cnode = FindCNode(param_node_pair.first, MIRROR_OPERATOR, func_graph);
      // if there is already a MirrorOp in the same graph, use MirrorOp CNode as a input instead
      if (next_cnode.first) {
        MS_EXCEPTION_IF_NULL(next_cnode.second);
        manager->SetEdge(node, SizeToInt(index), next_cnode.second);
        continue;
      }
    }
    // if the parameter found is a RefKey, or no MirrorOp is found in the same graph, insert a new MirrorOp
    // only one MirrorOp in backward_op
    if (backward_op.size() != 1) {
      MS_LOG(EXCEPTION) << "backward_op size must be 1, real is  " << backward_op.size();
    }
    std::string instance_name = MIRROR_OP;
    if (IsCastBeforMirror(node, index)) {
      for (auto &op : backward_op) {
        // insert new node before the node
        CNodePtr cnode = node->input(index)->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(cnode);
        AnfNodePtr pre_node = cnode->input(1);
        InsertNode(op, cnode, size_t(1), pre_node, func_graph, instance_name);
      }
    } else {
      for (auto &op : backward_op) {
        AnfNodePtr pre_node = node->input(index);
        InsertNode(op, node, index, pre_node, func_graph, instance_name);
      }
    }
  }
}

void BackwardCommunication(const OperatorInfoPtr &distribute_operator, const CNodePtr &node,
                           const std::vector<std::pair<CNodePtr, CNodePtr>> &sens_loss_pairs) {
  MS_EXCEPTION_IF_NULL(distribute_operator);
  MS_EXCEPTION_IF_NULL(node);

  bool is_loss_cnode =
    std::any_of(sens_loss_pairs.begin(), sens_loss_pairs.end(),
                [node](const std::pair<CNodePtr, CNodePtr> &element) { return element.second == node; });

  MirrorOps mirror_ops = distribute_operator->mirror_ops();
  VirtualDivOp virtual_div_op = distribute_operator->virtual_div_op();
  // insert mirror op
  if (!mirror_ops.empty()) {
    MS_LOG(INFO) << "insert mirror op for " << distribute_operator->name();
    InsertMirrorOps(mirror_ops, node);
  }
  // insert virtual div op
  if (!virtual_div_op.empty() && is_loss_cnode) {
    MS_LOG(INFO) << "insert virtual div op for " << distribute_operator->name();
    InsertVirtualDivOp(virtual_div_op, node);
  }
}

std::string GetDisOpName(const std::string &prim_name) {
  std::string op_name = prim_name;
  if (!prim_name.empty() && (prim_name[0] == '_')) {
    op_name = prim_name.substr(1);
  }
  return op_name + "Info";
}

OperatorInfoPtr OperatorInstanceByName(const std::string &name, const PrimitiveAttrs &attrs,
                                       const std::vector<Shapes> &shape_list) {
  if (shape_list.size() != 2) {
    MS_LOG(ERROR) << "The size of shape list is not 2";
    return nullptr;
  }
  if (name.length() == 0) {
    MS_LOG(EXCEPTION) << "Length of name is zero!";
  }
  std::string distribute_opname = GetDisOpName(name);
  if (name == GATHERV2) {
    distribute_opname = name + "PInfo";
    auto data_parallel_iter = attrs.find(DATA_PARALLEL);
    if (data_parallel_iter != attrs.end()) {
      MS_EXCEPTION_IF_NULL(data_parallel_iter->second);
      if (!data_parallel_iter->second->isa<BoolImm>()) {
        MS_LOG(EXCEPTION) << ": data_parallel flag's type is not a bool.";
      }
      bool data_parallel = data_parallel_iter->second->cast<BoolImmPtr>()->value();
      if (data_parallel) {
        distribute_opname = name + "Info";
      }
    }
  }
  OperatorInfoPtr operator_ =
    (OperatorInfoPtr)DynCreator::Instance().Creat(distribute_opname, shape_list[0], shape_list[1], attrs, TOTAL_OPS);
  if (operator_ == nullptr) {
    MS_LOG(INFO) << "Creat " << name << " failed";
    return nullptr;
  }
  std::string origin_name = operator_->name();
  operator_->set_name(origin_name + std::to_string(TOTAL_OPS));
  MS_LOG(INFO) << "Successfully created operator " << origin_name;
  ++TOTAL_OPS;
  return operator_;
}

OperatorInfoPtr OperatorInstance(const PrimitivePtr &prim, const PrimitiveAttrs &attrs,
                                 const std::vector<Shapes> &shape_list) {
  MS_EXCEPTION_IF_NULL(prim);
  OperatorInfoPtr operator_ = OperatorInstanceByName(prim->name(), attrs, shape_list);
  if (operator_ == nullptr) {
    MS_LOG(INFO) << "Creat " << prim->name() << " failed, use batch parallel";
    operator_ = OperatorInstanceByName(BATCH_PARALLEL, attrs, shape_list);
    MS_EXCEPTION_IF_NULL(operator_);
  }
  return operator_;
}

OperatorInfoPtr NewOperatorInstance(const PrimitivePtr &prim, const PrimitiveAttrs &attrs,
                                    std::vector<Shapes> shape_list) {
  OperatorInfoPtr operator_ = OperatorInstance(prim, attrs, shape_list);
  for (size_t i = 0; i < shape_list[0].size(); ++i) {
    MS_LOG(INFO) << "No:  " << i << "  input's shape: " << ShapeToString(shape_list[0][i]);
  }
  return operator_;
}

StrategyPtr ExtractStrategy(std::unordered_map<std::string, ValuePtr> attrs) {
  ValueTuplePtr var = attrs[STRATEGY]->cast<ValueTuplePtr>();
  StrategyPtr strategyPtr;
  MS_LOG(INFO) << "Extract information: strategy " << attrs[STRATEGY]->ToString();
  if (var == nullptr) {
    MS_LOG(EXCEPTION) << "Strategy value is nullptr";
  }
  if (var->size() > 0) {
    std::vector<ValuePtr> elements = var->value();
    std::vector<Dimensions> strategy;
    for (uint32_t index = 0; index < elements.size(); ++index) {
      Dimensions dim;
      if (elements[index]->isa<ValueSequeue>()) {
        ValueTuplePtr value_tuple = elements[index]->cast<ValueTuplePtr>();
        std::vector<ValuePtr> value_vector = value_tuple->value();
        (void)std::transform(value_vector.begin(), value_vector.end(), std::back_inserter(dim),
                             [](const ValuePtr &value) { return static_cast<int32_t>(GetValue<int>(value)); });
        strategy.push_back(dim);
      } else {
        MS_LOG(EXCEPTION) << "Failure:Strategy's format is wrong! Need ValueSequeue";
      }
    }
    if (strategy.empty()) {
      MS_LOG(EXCEPTION) << "ExtractStrategy:failed to extract strategy";
    }
    strategyPtr = NewStrategy(0, strategy);
  }

  return strategyPtr;
}

Shapes GetNodeShape(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  Shapes shapes;
  BaseShapePtr base_shape_ptr = node->Shape();
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    if (IsValueNode<Primitive>(cnode->input(0))) {
      PrimitivePtr prim = GetValueNode<PrimitivePtr>(cnode->input(0));
      MS_EXCEPTION_IF_NULL(prim);
      if (prim->name() == MAKEREF) {
        AnfNodePtr ref_node = cnode->input(1);
        auto func_graph = cnode->func_graph();
        MS_EXCEPTION_IF_NULL(ref_node);
        MS_EXCEPTION_IF_NULL(func_graph);
        return GetRefKeyNodeShape(ref_node, func_graph);
      }
    }
    if (cnode->input(0)->isa<CNode>()) {
      if (cnode->inputs().size() < 2) {
        MS_LOG(EXCEPTION) << "GetNodeShape: " << node->ToString() << " size is samller than 2";
      }
      base_shape_ptr = cnode->input(1)->Shape();
    }
  }
  if (base_shape_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "GetNodeShape: " << node->ToString() << " shape_ptr is nullptr, full name is "
                      << node->fullname_with_scope();
  }
  auto tuple_shape_ptr = dyn_cast<abstract::TupleShape>(base_shape_ptr);
  if (tuple_shape_ptr != nullptr) {
    auto tuple_shape = tuple_shape_ptr->shape();
    for (auto &shape : tuple_shape) {
      auto each_shape = dyn_cast<abstract::Shape>(shape);
      MS_EXCEPTION_IF_NULL(each_shape);
      shapes.push_back(each_shape->shape());
    }
  } else {
    auto shape_ptr = dyn_cast<abstract::Shape>(base_shape_ptr);
    MS_EXCEPTION_IF_NULL(shape_ptr);
    shapes.push_back(shape_ptr->shape());
  }
  return shapes;
}

std::vector<AnfNodePtr> FindParameterByRefKeyNode(const AnfNodePtr &node, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> parameters;
  if (!IsValueNode<RefKey>(node)) {
    MS_LOG(ERROR) << "The node is not a ref key";
    return parameters;
  }

  auto ref_key = GetValueNode<RefKeyPtr>(node);
  MS_EXCEPTION_IF_NULL(ref_key);
  auto name = ref_key->tag();

  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto roots = manager->roots();
  if (roots.size() != 1) {
    MS_LOG(ERROR) << "The size of roots ( " << roots.size() << " ) is not 1";
    return parameters;
  }

  FuncGraphPtr root_g = roots.back();
  MS_EXCEPTION_IF_NULL(root_g);
  for (auto &param_node : root_g->parameters()) {
    auto param = param_node->cast<ParameterPtr>();
    if (param && (name == param->name())) {
      parameters.push_back(param_node);
      MS_LOG(INFO) << "The name of ref key is: " << name;
      return parameters;
    }
  }

  MS_LOG(ERROR) << "The name of ref key is: " << name << ", but have not found the parameter";
  return parameters;
}

Shapes GetRefKeyNodeShape(const AnfNodePtr &node, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);

  std::vector<AnfNodePtr> parameters = FindParameterByRefKeyNode(node, func_graph);
  if (parameters.size() != 1) {
    MS_LOG(EXCEPTION) << "Find parameter by ref key node failed";
  }

  Shapes input_shapes;
  input_shapes = GetNodeShape(parameters[0]);
  if (input_shapes.size() != 1) {
    MS_LOG(EXCEPTION) << "Get input shape failed";
  }

  MS_LOG(INFO) << "The parameter shape is " << ShapeToString(input_shapes[0]);
  return input_shapes;
}

std::vector<Shapes> ExtractShape(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  Shapes shape_inputs, shape_outputs;
  std::vector<Shapes> shape_all;
  std::vector<AnfNodePtr> all_inputs = node->inputs();
  std::vector<AnfNodePtr> node_inputs{all_inputs.begin() + 1, all_inputs.end()};

  size_t inputs_size = all_inputs.size();
  for (size_t i = 1; i < inputs_size; ++i) {
    Shapes input_shapes;
    AnfNodePtr input = all_inputs[i];
    if (IsValueNode<RefKey>(input)) {
      auto func_graph = node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      std::vector<AnfNodePtr> parameters = FindParameterByRefKeyNode(input, func_graph);
      if (parameters.size() != 1) {
        MS_LOG(EXCEPTION) << "Find parameter by ref key node failed";
      }
      std::pair<AnfNodePtr, int> node_pair = std::make_pair(node, SizeToInt(i));
      g_RefMap[parameters[0]] = node_pair;
      input_shapes = GetRefKeyNodeShape(input, func_graph);
    } else if (IsValueNode<Tensor>(input) || input->isa<CNode>() || input->isa<Parameter>()) {
      input_shapes = GetNodeShape(input);
    } else {
      continue;
    }
    if (input_shapes.size() != 1) {
      MS_LOG(EXCEPTION) << "ExtractShape:Get input shape failed";
    }
    shape_inputs.push_back(input_shapes[0]);
  }
  shape_all.push_back(shape_inputs);
  // extract out shape
  shape_outputs = GetNodeShape(node);
  shape_all.push_back(shape_outputs);
  return shape_all;
}

std::pair<AnfNodePtr, int> FindParallelCareNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodeIndexSet node_set = manager->node_users()[node];
  for (auto &node_pair : node_set) {
    CNodePtr cnode = node_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    ValueNodePtr prim_node_anf = cnode->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prim_node_anf);
    PrimitivePtr node_prim = prim_node_anf->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(node_prim);
    if (node_prim->name() == DEPEND && node_pair.second != 1) {
      continue;
    }
    if (IsParallelCareNode(cnode) && cnode->operator_info() != nullptr) {
      return node_pair;
    } else if (FindParallelCareNode(node_pair.first).first != nullptr) {
      return FindParallelCareNode(node_pair.first);
    }
  }
  return std::make_pair(nullptr, 0);
}

std::pair<AnfNodePtr, int> FindSubGraph(const FuncGraphPtr &graph, const AnfNodePtr &parameter) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(parameter);
  FuncGraphManagerPtr manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::pair<AnfNodePtr, int> prim_anf_node_pair = FindParallelCareNode(parameter);
  if (prim_anf_node_pair.first != nullptr) {
    return prim_anf_node_pair;
  } else {
    AnfNodeIndexSet param_sub_set = manager->node_users()[parameter];
    for (auto &param_pair : param_sub_set) {
      CNodePtr graph_cnode = param_pair.first->cast<CNodePtr>();
      if ((graph_cnode == nullptr) || !graph_cnode->input(0)->isa<CNode>()) {
        continue;
      }
      CNodePtr graph_cnode_inp0 = graph_cnode->input(0)->cast<CNodePtr>();
      if (!IsValueNode<FuncGraph>(graph_cnode_inp0->input(1))) {
        continue;
      }
      FuncGraphPtr graph_sub = GetValueNode<FuncGraphPtr>(graph_cnode_inp0->input(1));
      auto parameters = graph_sub->parameters();
      if (IntToSize(param_pair.second - 1) >= parameters.size()) {
        MS_LOG(EXCEPTION) << "The index is out of range, index is " << param_pair.second - 1 << ", vector size is "
                          << parameters.size();
      }
      std::pair<AnfNodePtr, int> res = FindSubGraph(graph_sub, parameters[IntToSize(param_pair.second - 1)]);
      if (res.first != nullptr) {
        return res;
      }
    }
  }
  return std::make_pair(nullptr, 0);
}

void SetParallelShape(const AnfNodePtr &parameter, const std::pair<AnfNodePtr, int> &res) {
  MS_EXCEPTION_IF_NULL(parameter);
  AbstractBasePtr abstract = parameter->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  MS_LOG(DEBUG) << "SetParallelShape " << parameter->ToString() << " shape " << parameter->Shape()->ToString();
  CNodePtr cnode = res.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  OperatorInfoPtr distribute_operator = cnode->operator_info();
  if (distribute_operator == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:node " << cnode->ToString() << " 's OperatorInfoPtr is nullptr";
  }

  if (IntToSize(res.second - 1) >= distribute_operator->inputs_tensor_info().size()) {
    MS_LOG(EXCEPTION) << "The index is out of range, index is  " << res.second - 1 << ", vector size is  "
                      << distribute_operator->inputs_tensor_info().size();
  }
  TensorInfo tensorinfo_in = distribute_operator->inputs_tensor_info()[IntToSize(res.second - 1)];
  Shape slice_shape = tensorinfo_in.slice_shape();
  MS_LOG(DEBUG) << "SetParallelShape slice_shape  " << parameter->ToString() << "  shape "
                << MakeValue(slice_shape)->ToString();
  std::shared_ptr<abstract::BaseShape> parallel_shape = std::make_shared<abstract::Shape>(slice_shape);
  MS_EXCEPTION_IF_NULL(parallel_shape);
  // Don't modify it in-place as the pointer of this AbstractValue may used as cache key in StaticAnalysis.
  auto cloned_abstract = abstract->Clone();
  MS_EXCEPTION_IF_NULL(cloned_abstract);
  cloned_abstract->set_shape(parallel_shape);
  parameter->set_abstract(cloned_abstract);
  TensorLayout tensor_layout = tensorinfo_in.tensor_layout();
  ParameterPtr parameter_ptr = parameter->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(parameter_ptr);
  parameter_ptr->set_tensor_layout(std::make_shared<TensorLayout>(tensor_layout));
}

void CoverSliceShape(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  auto parameters = root->parameters();
  for (auto &parameter : parameters) {
    MS_EXCEPTION_IF_NULL(parameter->Shape());
    auto iter = g_RefMap.find(parameter);
    if (iter != g_RefMap.end()) {
      SetParallelShape(parameter, g_RefMap[parameter]);
      continue;
    }
    std::pair<AnfNodePtr, int> res = FindSubGraph(root, parameter);
    if (res.first == nullptr) {
      MS_LOG(INFO) << "Parameter " << parameter->ToString() << " don't need to set parallel shape";
    } else {
      SetParallelShape(parameter, res);
      MS_LOG(DEBUG) << "Parameter " << parameter->ToString() << " shape " << parameter->Shape()->ToString();
    }
  }
  g_RefMap.clear();
}

bool ParameterIsCloned(const FuncGraphPtr &root, const AnfNodePtr &parameter_node) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(parameter_node);
  FuncGraphManagerPtr manager = root->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto cloned_parameter = parameter_node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(cloned_parameter);

  // find the clone parameter
  if (!cloned_parameter->has_default()) {
    return false;
  }

  auto param_value = std::dynamic_pointer_cast<ParamValuePy>(cloned_parameter->default_param());
  py::object clone_info = parse::python_adapter::GetPyObjAttr(param_value->value(), CLONE_INFO);
  bool cloned = py::cast<bool>(parse::python_adapter::GetPyObjAttr(clone_info, CLONED));
  if (!cloned) {
    return false;
  }

  MS_LOG(INFO) << "The parameter: " << cloned_parameter->name() << " is cloned";
  return true;
}

void SetClonedTensorShapeForOptimizer(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  for (auto &cloned_parameter_node : root->parameters()) {
    MS_EXCEPTION_IF_NULL(cloned_parameter_node);
    auto cloned_parameter = cloned_parameter_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(cloned_parameter);

    if (!ParameterIsCloned(root, cloned_parameter_node)) {
      continue;
    }

    // get the cloned index
    auto param_value = std::dynamic_pointer_cast<ParamValuePy>(cloned_parameter->default_param());
    py::object cloned_info = parse::python_adapter::GetPyObjAttr(param_value->value(), CLONE_INFO);
    int32_t cloned_index = py::cast<int32_t>(parse::python_adapter::GetPyObjAttr(cloned_info, CLONED_INDEX));

    // find the be cloned parameter
    bool found_be_cloned_parameter = false;
    ParameterPtr cloned_from_parameter = nullptr;
    AnfNodePtr cloned_from_node = nullptr;
    for (auto &be_cloned_parameter_node : root->parameters()) {
      MS_EXCEPTION_IF_NULL(be_cloned_parameter_node);
      auto be_cloned_parameter = be_cloned_parameter_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(be_cloned_parameter);
      if (!be_cloned_parameter->has_default()) {
        continue;
      }

      auto param_value_cloned = std::dynamic_pointer_cast<ParamValuePy>(be_cloned_parameter->default_param());
      py::object be_cloned_info = parse::python_adapter::GetPyObjAttr(param_value_cloned->value(), CLONE_INFO);
      if (!py::cast<bool>(parse::python_adapter::GetPyObjAttr(be_cloned_info, BE_CLONED))) {
        continue;
      }

      // get the be cloned index
      py::list be_cloned_index = parse::python_adapter::GetPyObjAttr(be_cloned_info, BE_CLONED_INDEX);
      for (auto &index : be_cloned_index) {
        if (cloned_index == py::cast<int32_t>(index)) {
          found_be_cloned_parameter = true;
          cloned_from_parameter = be_cloned_parameter;
          cloned_from_node = be_cloned_parameter_node;
          break;
        }
      }
    }

    if (found_be_cloned_parameter) {
      // set the shape and tensor layout for cloned parameter
      cloned_parameter->set_tensor_layout(cloned_from_parameter->tensor_layout());
      MS_EXCEPTION_IF_NULL(cloned_parameter_node->abstract());
      MS_EXCEPTION_IF_NULL(cloned_from_node->abstract());
      auto cloned_abstract = cloned_parameter_node->abstract()->Clone();
      MS_EXCEPTION_IF_NULL(cloned_abstract);
      cloned_abstract->set_shape(cloned_from_node->abstract()->GetShapeTrack());
      cloned_parameter_node->set_abstract(cloned_abstract);
      MS_LOG(INFO) << "The parameter: " << cloned_parameter->name()
                   << " is cloned, the be cloned parameter is: " << cloned_from_parameter->name()
                   << ", clone index is:  " << cloned_index;
    } else {
      MS_LOG(EXCEPTION) << "The parameter: " << cloned_parameter->name() << " is cloned, cloned index is  "
                        << cloned_index << ", but not found the be cloned parameter";
    }
  }
}

void SetVirtualDatasetStrategy(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  bool full_batch = ParallelContext::GetInstance()->full_batch();

  PrimitivePtr prim = GetValueNode<PrimitivePtr>(node->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->name() == VIRTUAL_DATA_SET) {
    CheckGlobalDeviceManager();
    int32_t dev_num;
    if (full_batch) {
      dev_num = 1;
    } else {
      dev_num = SizeToInt(g_device_manager->GetDeviceListByStageId(0).size());
    }
    auto attrs_temp = prim->attrs();
    std::vector<Shapes> shape_list = ExtractShape(node);
    if (shape_list.empty()) {
      MS_LOG(EXCEPTION) << "Failure:node " << node->ToString() << " failed to extract shape";
    }
    std::vector<ValuePtr> elements;
    for (size_t i = 0; i < shape_list[0].size(); i++) {
      if (shape_list[0][i].empty()) {
        MS_LOG(EXCEPTION) << "shape_list[ " << i << " ].size() is zero";
      }
      std::vector<int32_t> input_strategy = {dev_num};
      for (size_t j = 1; j < shape_list[0][i].size(); j++) {
        input_strategy.push_back(1);
      }
      elements.push_back(MakeValue(input_strategy));
    }
    ValueTuplePtr strategy = std::make_shared<ValueTuple>(elements);
    attrs_temp[STRATEGY] = strategy;
    (void)prim->SetAttrs(attrs_temp);
  }
}

void ExtractInformation(const std::vector<AnfNodePtr> &all_nodes) {
  // load strategy map from checkpoint
  StrategyMap stra_map;
  if (StrategyCheckpoint::GetInstance().LoadCheckPointOn()) {
    if (StrategyCheckpoint::GetInstance().Load(&stra_map) != SUCCESS) {
      MS_LOG(EXCEPTION) << "Load strategy checkpoint failed";
    }
  }
  for (auto &node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if ((cnode == nullptr) || !IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    SetVirtualDatasetStrategy(cnode);
    ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);
    auto attrs = prim->attrs();
    MS_LOG(INFO) << "extract information: node: " << node->ToString() << " prim " << prim->name();
    if (IsParallelCareNode(cnode)) {
      std::vector<Shapes> shape_list = ExtractShape(cnode);
      if (shape_list.empty()) {
        MS_LOG(EXCEPTION) << "Failure:node " << node->ToString() << " failed to extract shape";
      }
      OperatorInfoPtr operator_ = OperatorInstance(prim, attrs, shape_list);
      if (operator_ == nullptr) {
        MS_LOG(EXCEPTION) << "Failure:Primitive " << prim->name() << " OperatorInstance failed";
      }
      auto &inputs = cnode->inputs();
      std::vector<ValuePtr> input_value;
      for (size_t index = 1; index < inputs.size(); ++index) {
        if (inputs[index]->isa<ValueNode>()) {
          input_value.push_back(GetValueNode(inputs[index]));
        } else {
          input_value.emplace_back(nullptr);
        }
      }
      StrategyPtr strategyPtr = nullptr;
      (*operator_).set_input_value(input_value);
      (*operator_).set_outputs_dtype(cnode->Type());
      (*operator_).set_cnode(cnode);
      if (prim->name() == RESHAPE) {
        (void)cnode->set_operator_info(operator_);
        continue;
      }
      // load strategy checkpoint
      // key of strategy map
      std::string strategy_key_name = NodeParameterName(cnode);
      bool load_strategy_from_ckpt =
        StrategyCheckpoint::GetInstance().LoadCheckPointOn() && stra_map.find(strategy_key_name) != stra_map.end();
      if (!StrategyFound(attrs) && !load_strategy_from_ckpt) {
        MS_LOG(INFO) << "ExtractInformation: the strategy of node " << node->ToString() << " prim " << prim->name()
                     << " is empty, using batch parallel";
        std::shared_ptr<std::vector<Dimensions>> strategy_v_ptr = operator_->GenerateBatchStrategies();
        if (strategy_v_ptr == nullptr) {
          MS_LOG(EXCEPTION) << "Failure:Generate batch parallel strategy failed";
        }
        std::vector<ValuePtr> elements;
        for (size_t i = 0; i < strategy_v_ptr->size(); i++) {
          elements.push_back(MakeValue((*strategy_v_ptr)[i]));
        }
        ValueTuplePtr strategy = std::make_shared<ValueTuple>(elements);
        // display the strategy generated by batch parallel
        attrs[GEN_STRATEGY] = strategy;
        (void)prim->SetAttrs(attrs);
        MS_LOG(INFO) << "node " << node->ToString() << " prim " << prim->name() << " batch parallel strategy is "
                     << attrs[GEN_STRATEGY]->ToString();
        strategyPtr = NewStrategy(0, *strategy_v_ptr);
      } else if (load_strategy_from_ckpt) {
        strategyPtr = stra_map[strategy_key_name];
      } else {
        strategyPtr = ExtractStrategy(attrs);
      }
      if (strategyPtr != nullptr) {
        if (operator_->Init(strategyPtr) == FAILED) {
          MS_LOG(EXCEPTION) << "Failure:operator " << prim->name() << " init failed";
        }
        (void)cnode->set_operator_info(operator_);
      } else {
        MS_LOG(EXCEPTION) << "ERROR:strategy_ptr is nullptr";
      }
    }
  }
}

TensorLayout GetInputLayoutFromCNode(const std::pair<AnfNodePtr, int> &node_pair) {
  CNodePtr cnode = node_pair.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  OperatorInfoPtr distribute_operator = GetDistributeOperator(cnode);
  MS_EXCEPTION_IF_NULL(distribute_operator);
  int index = node_pair.second;
  if (index > SizeToInt(distribute_operator->inputs_tensor_info().size())) {
    MS_LOG(EXCEPTION) << "The index is out of range, the node_pair.second is  " << index - 1 << ", the vector size is  "
                      << distribute_operator->inputs_tensor_info().size();
  }
  TensorInfo tensorinfo_in = distribute_operator->inputs_tensor_info()[IntToSize(index - 1)];
  TensorLayout tensorlayout_in = tensorinfo_in.tensor_layout();
  return tensorlayout_in;
}

// if reshape's output connect to several primitive, return the first layout found
std::shared_ptr<TensorLayout> FindNextLayout(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(cnode->func_graph());
  FuncGraphManagerPtr manager = cnode->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodeIndexSet node_set = manager->node_users()[cnode];
  for (auto &node_pair : node_set) {
    CNodePtr use_apply = node_pair.first->cast<CNodePtr>();
    if (use_apply == nullptr || !IsValueNode<Primitive>(use_apply->input(0))) {
      continue;
    }
    ValueNodePtr prim_anf_node = use_apply->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prim_anf_node);
    PrimitivePtr node_prim = prim_anf_node->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(node_prim);
    MS_LOG(INFO) << "FindNextLayout prim " << node_prim->name();
    if (node_prim->name() == DEPEND && node_pair.second != 1) {
      continue;
    }
    if (IsParallelCareNode(use_apply) && (use_apply->operator_info() != nullptr)) {
      MS_LOG(INFO) << "FindNextLayout success prim " << node_prim->name();
      auto layout = GetInputLayoutFromCNode(node_pair);
      return std::make_shared<TensorLayout>(layout);
    }
    MS_LOG(DEBUG) << "FindNextLayout failed prim " << node_prim->name() << "  " << IsParallelCareNode(use_apply)
                  << "   " << (use_apply->operator_info() != nullptr);

    auto layout_ptr = FindNextLayout(use_apply);
    if (layout_ptr) {
      return layout_ptr;
    }
  }
  MS_LOG(WARNING) << "FindNextLayout return nullptr, if reshape is not the last primitive, there must be some error";
  return nullptr;
}

std::shared_ptr<TensorLayout> GetOutputLayoutFromCNode(const CNodePtr &cnode, size_t output_index) {
  MS_EXCEPTION_IF_NULL(cnode);
  OperatorInfoPtr distribute_operator = GetDistributeOperator(cnode);
  MS_EXCEPTION_IF_NULL(distribute_operator);
  if (distribute_operator->outputs_tensor_info().size() < output_index) {
    MS_LOG(EXCEPTION) << "outputs_tensor_info size is  " << distribute_operator->inputs_tensor_info().size()
                      << ", must be less than output_index  " << output_index;
  }
  TensorInfo tensorinfo_out = distribute_operator->outputs_tensor_info()[output_index];
  TensorLayout tensorlayout_out = tensorinfo_out.tensor_layout();
  return std::make_shared<TensorLayout>(tensorlayout_out);
}

std::shared_ptr<TensorLayout> FindPrevParallelCareNodeLayout(const AnfNodePtr &node, size_t output_index) {
  if (!node->isa<CNode>()) {
    return nullptr;
  }
  CNodePtr cnode = node->cast<CNodePtr>();
  if (!IsValueNode<Primitive>(cnode->input(0))) {
    return nullptr;
  }
  if (IsParallelCareNode(cnode) && (cnode->operator_info() != nullptr)) {
    auto layout_ptr = GetOutputLayoutFromCNode(cnode, output_index);
    if (!layout_ptr) {
      MS_LOG(EXCEPTION) << "Failure:GetLayoutFromCNode failed";
    }
    return layout_ptr;
  }
  return nullptr;
}

std::shared_ptr<TensorLayout> CreateParameterLayout(const AnfNodePtr &node) {
  // Create DataParallel tensor layout for parameter(support WideDeep).
  CheckGlobalDeviceManager();
  int32_t dev_num = SizeToInt(g_device_manager->GetDeviceListByStageId(0).size());
  TensorLayout input_tensor_layout;
  // create input_shape
  Shapes inputs_shape = GetNodeShape(node);
  Shape input_shape_array = inputs_shape[0];
  if (input_shape_array.empty()) {
    MS_LOG(EXCEPTION) << "Don't support reshape a scalar parameter.";
  }
  // create tensor_map
  size_t shape_size = input_shape_array.size();
  TensorMap input_tensor_map_array(SizeToInt(shape_size) - 1, -1);
  input_tensor_map_array.insert(input_tensor_map_array.begin(), 0);
  // create dev_matrix
  Shape dev_matrix_array = {dev_num};
  if (input_tensor_layout.InitFromVector(dev_matrix_array, input_tensor_map_array, input_shape_array) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Create tensor layout for parameter failed.";
  }
  return std::make_shared<TensorLayout>(input_tensor_layout);
}

std::shared_ptr<TensorLayout> FindPrevLayout(const AnfNodePtr &node) {
  if (node->isa<Parameter>()) {
    return CreateParameterLayout(node);
  }
  if (!node->isa<CNode>()) {
    return nullptr;
  }
  CNodePtr cnode = node->cast<CNodePtr>();
  if (!IsValueNode<Primitive>(cnode->input(0))) {
    return nullptr;
  }
  if (IsParallelCareNode(cnode) && (cnode->operator_info() != nullptr)) {
    auto layout_ptr = GetOutputLayoutFromCNode(cnode, 0);
    if (!layout_ptr) {
      MS_LOG(EXCEPTION) << "Failure:GetLayoutFromCNode failed";
    }
    return layout_ptr;
  }
  ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
  PrimitivePtr prim = prim_anf_node->value()->cast<PrimitivePtr>();
  if (prim->name() == TUPLE_GETITEM) {
    auto tuple_index = GetTupleGetItemIndex(cnode);
    auto layout_ptr = FindPrevParallelCareNodeLayout(cnode->input(1), IntToSize(tuple_index));
    if (!layout_ptr) {
      MS_LOG(EXCEPTION)
        << " Failure:FindPrevLayout failed, tuple_getitem before reshape, but there does not exit a parallel care node "
           "before tuple_getitem!";
    }
    return layout_ptr;
  }
  for (size_t index = 0; index < cnode->inputs().size(); ++index) {
    if (prim->name() == DEPEND && index != 1) {
      continue;
    }
    auto layout_ptr = FindPrevLayout(cnode->inputs()[index]);
    if (!layout_ptr) {
      continue;
    }
    return layout_ptr;
  }
  MS_LOG(WARNING) << "FindPrevLayout return nullptr, if reshape is not the first primitive, there must be some error";
  return nullptr;
}

void ReshapeInit(const std::vector<AnfNodePtr> &all_nodes) {
  for (auto &node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if ((cnode == nullptr) || !IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
    if (!IsParallelCareNode(cnode) || (cnode->operator_info() == nullptr)) {
      continue;
    }
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);
    MS_EXCEPTION_IF_NULL(prim);
    OperatorInfoPtr operator_info = cnode->operator_info();
    if (operator_info == nullptr) {
      MS_LOG(EXCEPTION) << "Failure:Primitive " << prim->ToString() << " OperatorInstance is nullptr";
    }
    if (prim->name() != RESHAPE) {
      continue;
    }
    auto attrs = prim->attrs();
    if (StrategyFound(attrs)) {
      MS_LOG(EXCEPTION) << "Setting strategy for Reshape goes for nothing!";
    }
    MS_ASSERT(cnode->inputs().size() == 3);
    auto prev_layout_ptr = FindPrevLayout(cnode->input(1));
    if (prev_layout_ptr) {
      auto reshape_info_ptr = std::dynamic_pointer_cast<ReshapeInfo>(operator_info);
      reshape_info_ptr->SetInputLayout(*prev_layout_ptr);
    }
    auto next_layout_ptr = FindNextLayout(cnode);
    if (next_layout_ptr) {
      auto reshape_info_ptr = std::dynamic_pointer_cast<ReshapeInfo>(operator_info);
      reshape_info_ptr->SetOutputLayout(*next_layout_ptr);
    }
    if (operator_info->Init(nullptr) == FAILED) {
      MS_LOG(EXCEPTION) << "Failure:operator " << prim->ToString() << " init failed";
    }
  }
}

CNodePtr FindLossCNode(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  CNodePtr return_node = func_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  if (return_node->size() < 2) {
    MS_LOG(EXCEPTION) << "Failure: " << return_node->ToString() << " size is smaller than 2";
  }
  AnfNodePtr pre_node = return_node->input(1);
  MS_EXCEPTION_IF_NULL(pre_node);

  auto pre_cnode = pre_node->cast<CNodePtr>();
  if (pre_cnode == nullptr) {
    return nullptr;
  }

  auto current_prim = GetValueNode<PrimitivePtr>(pre_cnode->input(0));
  // return -> cast
  if (current_prim->name() == CAST && pre_cnode->operator_info() == nullptr) {
    pre_cnode = pre_cnode->input(1)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(pre_cnode);
    current_prim = GetValueNode<PrimitivePtr>(pre_cnode->input(0));
  }

  // notice: the GetNext op has not input
  if (INVALID_LOSS_OPS.find(current_prim->name()) != INVALID_LOSS_OPS.end()) {
    MS_LOG(INFO) << "The loss is: " << current_prim->name();
    return pre_cnode;
  }

  // size of common cnode is larger than 1
  if (pre_cnode->size() < 2) {
    MS_LOG(EXCEPTION) << pre_cnode->ToString() << " size( " << pre_cnode->inputs().size() << " ) is smaller than 2";
  }

  // return -> tuple_getitem -> loss
  if (current_prim->name() == TUPLE_GETITEM) {
    AnfNodePtr pre_pre_node = pre_cnode->input(1);
    MS_EXCEPTION_IF_NULL(pre_pre_node);

    auto pre_pre_cnode = pre_pre_node->cast<CNodePtr>();
    auto value = pre_pre_cnode->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value);
    PrimitivePtr prim = value->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(prim);
    MS_LOG(DEBUG) << "The loss name is " << prim->name();
    return pre_pre_cnode;
  }

  // return -> make_tuple
  if (current_prim->name() == MAKE_TUPLE) {
    MS_LOG(EXCEPTION) << "The loss have make_tuple, it is not supported";
  }

  // return -> loss
  MS_LOG(DEBUG) << "The loss name is " << current_prim->name();
  return pre_cnode;
}

TensorLayouts GetLossNodeGradOutputLayout(const CNodePtr &loss_cnode) {
  TensorLayouts ret;
  MS_EXCEPTION_IF_NULL(loss_cnode);
  AnfNodePtr node = loss_cnode->cast<AnfNodePtr>();
  MS_EXCEPTION_IF_NULL(node);

  LossNodeInfo node_info = GetLossNodeInfo(node);
  ValueNodePtr prim_anf_node = loss_cnode->input(0)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(prim_anf_node);
  PrimitivePtr prim = prim_anf_node->value()->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(prim);
  if (INVALID_LOSS_OPS.find(prim->name()) != INVALID_LOSS_OPS.end()) {
    MS_LOG(WARNING) << "The loss name is: " << prim->name() << ", do nothing for split sens now";
    return ret;
  }

  OperatorInfoPtr operator_info = loss_cnode->operator_info();
  MS_EXCEPTION_IF_NULL(operator_info);
  TensorInfo loss_grad_tensor_info;
  size_t op_output_size = operator_info->outputs_tensor_info().size();
  MS_LOG(INFO) << "The loss name is " << operator_info->name() << ", the has tuple item is  "
               << node_info.has_tuple_getitem << ", the output size is  " << op_output_size << ", the dout_index is  "
               << node_info.dout_index;

  if ((op_output_size == 0) || (op_output_size <= IntToSize(node_info.dout_index))) {
    MS_LOG(EXCEPTION) << "The index is  " << node_info.dout_index << ", but the size of outputs is  " << op_output_size;
  }

  if (!node_info.has_tuple_getitem && (op_output_size > 1)) {
    MS_LOG(EXCEPTION) << "Currently, it is not supported that the sens is a tuple.";
  }

  loss_grad_tensor_info = operator_info->outputs_tensor_info()[IntToSize(node_info.dout_index)];
  ret.push_back(loss_grad_tensor_info.tensor_layout());
  return ret;
}

void SplitSens(const CNodePtr &grad_sens_node, const TensorLayout &loss_grad_layout) {
  MS_EXCEPTION_IF_NULL(grad_sens_node);
  if (grad_sens_node->size() <= 1) {
    MS_LOG(EXCEPTION) << "The size of grad sens node is smaller than 2";
  }
  AnfNodePtr sens_tensor_node = grad_sens_node->input(1);
  MS_EXCEPTION_IF_NULL(sens_tensor_node);
  Shapes sens_shapes = GetNodeShape(sens_tensor_node);
  if (sens_shapes.size() != 1) {
    MS_LOG(EXCEPTION) << "GetNodeShape for sens_tensor_node, output size is not 1";
  }
  // If the shape of sens tensor is [] or [1], no need to split it.
  Shape sens_shape = sens_shapes[0];
  if (sens_shape.empty() || ((sens_shape.size() == 1) && (sens_shape[0] == 1))) {
    if (sens_tensor_node->isa<Parameter>()) {
      auto sens_tensor_param = sens_tensor_node->cast<ParameterPtr>();
      MS_LOG(DEBUG) << "loss layout " << loss_grad_layout.ToString();
      sens_tensor_param->set_tensor_layout(std::make_shared<TensorLayout>(loss_grad_layout));
    }
    MS_LOG(INFO) << "The shape of sens is " << ShapeToString(sens_shape) << ", no need to split sens";
    return;
  }
  auto loss_shape = loss_grad_layout.tensor_shape().array();
  if (loss_shape != sens_shape) {
    MS_LOG(EXCEPTION) << "The shape of sens is not equal to loss output, it is unsupported now. Sens shape is "
                      << ShapeToString(sens_shape) << ", loss shape is " << ShapeToString(loss_shape);
  }
  MS_LOG(INFO) << "The shape of sens is " << ShapeToString(sens_shape) << ", split it.";

  if (!IsValueNode<Tensor>(sens_tensor_node)) {
    if (sens_tensor_node->isa<Parameter>()) {
      MS_LOG(DEBUG) << "loss layout " << loss_grad_layout.ToString();
      AbstractBasePtr abstract = sens_tensor_node->abstract();
      MS_EXCEPTION_IF_NULL(abstract);
      auto slice_shape = loss_grad_layout.slice_shape().array();
      std::shared_ptr<abstract::BaseShape> parallel_shape = std::make_shared<abstract::Shape>(slice_shape);
      MS_EXCEPTION_IF_NULL(parallel_shape);
      auto cloned_abstract = abstract->Clone();
      MS_EXCEPTION_IF_NULL(cloned_abstract);
      cloned_abstract->set_shape(parallel_shape);
      sens_tensor_node->set_abstract(cloned_abstract);
      auto sens_tensor_param = sens_tensor_node->cast<ParameterPtr>();
      sens_tensor_param->set_tensor_layout(std::make_shared<TensorLayout>(loss_grad_layout));
      return;
    }
    MS_LOG(EXCEPTION) << "The type of sens node is not Tensor or Parameter, it is unsupported now.";
  }

  // Use _GetTensorSlice operator to split the sens tensor
  FuncGraphPtr func_graph = grad_sens_node->func_graph();  // only cnode can get the graph
  MS_EXCEPTION_IF_NULL(func_graph);
  Operator op = CreateGetTensorSliceOp(loss_grad_layout);
  InsertGetTensorSliceOp(op, grad_sens_node, func_graph, 1, SPLIT_SENS);
}

void InsertForwardOps(const OperatorInfoPtr &distribute_operator, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(distribute_operator);
  MS_EXCEPTION_IF_NULL(cnode);
  OperatorVector forward_op = distribute_operator->forward_op();
  if (!forward_op.empty()) {
    MS_LOG(INFO) << "Insert forward op for " << distribute_operator->name();
    ForwardCommunication(forward_op, cnode);
  }
}

void StepReplace(const OperatorInfoPtr &distribute_operator, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(distribute_operator);
  MS_EXCEPTION_IF_NULL(cnode);
  // StepReplaceOp
  OperatorVector replace_op = distribute_operator->replace_op();
  if (!replace_op.empty()) {
    MS_LOG(INFO) << "StepReplaceOp " << cnode->ToString();
    StepReplaceOp(replace_op, cnode);
  }

  // StepReplaceGraph: after calling StepReplaceGraph, cnode can not be used anymore.
  ReplaceGraphPtr replace_graph = distribute_operator->replace_graph(cnode);
  if (!replace_op.empty() && replace_graph) {
    MS_LOG(EXCEPTION) << "Only one of replace_op or replace_op can be used";
  }
  if (replace_graph) {
    MS_LOG(INFO) << "StepReplaceGraph " << cnode->ToString();
    StepReplaceGraph(replace_graph, cnode);
  }
}

void HandleDropoutNode(const OperatorInfoPtr &distribute_operator, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(distribute_operator);
  MS_EXCEPTION_IF_NULL(cnode);

  std::string op_name = distribute_operator->name();
  if (op_name.find(DROPOUT_DO_MASK) == std::string::npos) {
    return;
  }

  DropoutDoMaskInfoPtr dropout_do_mask = std::dynamic_pointer_cast<DropoutDoMaskInfo>(distribute_operator);
  MS_EXCEPTION_IF_NULL(dropout_do_mask);
  std::vector<Operator> replace_op = dropout_do_mask->GetDropoutGenMaskReplaceOp(cnode);
  if (replace_op.empty()) {
    MS_LOG(DEBUG) << "No need to replace dropout_gen_mask";
    return;
  }
  if (cnode->inputs().size() != DROPOUT_DO_MASK_CNODE_INPUT_SIZE) {
    MS_LOG(EXCEPTION) << "The size of drop out do mask cnode's input is not " << DROPOUT_DO_MASK_CNODE_INPUT_SIZE;
  }
  ReplaceOneOp(replace_op[0], cnode->input(DROPOUT_GEN_MASK_INDEX)->cast<CNodePtr>());
}

void HandleSpecialNode(const OperatorInfoPtr &distribute_operator, const CNodePtr &cnode) {
  HandleDropoutNode(distribute_operator, cnode);
}

std::set<FuncGraphPtr> FindForwardGraphByRootNodes(const AnfNodeSet &root_all_nodes) {
  // J->CNode->Graph
  std::set<FuncGraphPtr> graph_set;
  for (auto &node : root_all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }

    auto cnode = node->cast<CNodePtr>();
    if ((cnode->size() < 2) || !IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    auto expect_j_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (expect_j_prim->name() != J) {
      continue;
    }
    if (IsValueNode<FuncGraph>(cnode->input(1))) {
      auto graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
      MS_LOG(DEBUG) << "Find the forward graph success";
      graph_set.insert(graph);
    }
  }
  return graph_set;
}

void StepSplitSens(const std::pair<CNodePtr, CNodePtr> &sens_loss_pair) {
  CNodePtr sens_node = sens_loss_pair.first;
  CNodePtr loss_node = sens_loss_pair.second;
  auto loss_grad_layout = GetLossNodeGradOutputLayout(loss_node);
  if (!loss_grad_layout.empty()) {
    SplitSens(sens_node, loss_grad_layout[0]);
  }
}

// Sens node satisfies the following conditions: cnode(sens)-->cnode(tuple_getitem)-->cnode-->cnode(J)
std::vector<std::pair<CNodePtr, CNodePtr>> GetSensLossPairs(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  std::vector<std::pair<CNodePtr, CNodePtr>> sens_loss_pairs;
  for (auto &node : root->nodes()) {
    if (!node->isa<CNode>()) {
      continue;
    }

    // cnode(sens)-->cnode(tuple_getitem)
    auto sens_cnode = node->cast<CNodePtr>();
    AnfNodePtr expect_tuple_getitem = sens_cnode->input(0);
    MS_EXCEPTION_IF_NULL(expect_tuple_getitem);
    if (!expect_tuple_getitem->isa<CNode>()) {
      continue;
    }

    auto expect_tuple_getitem_cnode = expect_tuple_getitem->cast<CNodePtr>();
    if (!IsSomePrimitive(expect_tuple_getitem_cnode, TUPLE_GETITEM)) {
      continue;
    }

    // cnode(sens)-->cnode(tuple_getitem)-->cnode
    AnfNodePtr expect_anonymous = expect_tuple_getitem_cnode->input(1);
    MS_EXCEPTION_IF_NULL(expect_anonymous);
    if (!expect_anonymous->isa<CNode>()) {
      continue;
    }

    // cnode(sens)-->cnode(tuple_getitem)-->cnode-->cnode(J)
    auto expect_anonymous_cnode = expect_anonymous->cast<CNodePtr>();
    AnfNodePtr expect_j = expect_anonymous_cnode->input(0);
    MS_EXCEPTION_IF_NULL(expect_j);
    if (!expect_j->isa<CNode>()) {
      continue;
    }
    auto expect_j_cnode = expect_j->cast<CNodePtr>();
    if (!IsSomePrimitive(expect_j_cnode, J)) {
      continue;
    }

    if (!IsValueNode<FuncGraph>(expect_j_cnode->input(1))) {
      MS_LOG(EXCEPTION) << "Sens can't find the corresponding graph.";
    }
    auto func_graph = GetValueNode<FuncGraphPtr>(expect_j_cnode->input(1));
    auto loss_cnode = FindLossCNode(func_graph);
    if (loss_cnode == nullptr) {
      MS_LOG(WARNING) << "Can not find the loss cnode";
      continue;
    }
    std::pair<CNodePtr, CNodePtr> sens_loss_pair = std::make_pair(sens_cnode, loss_cnode);
    sens_loss_pairs.push_back(sens_loss_pair);
  }
  return sens_loss_pairs;
}

void ParallelCommunication(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes,
                           const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(manager);
  TensorRedistribution tensor_redistribution;

  std::vector<std::pair<CNodePtr, CNodePtr>> sens_loss_pairs = GetSensLossPairs(root);
  bool has_backward = !sens_loss_pairs.empty();
  // split sens must before inserting the operators.
  for (auto &pair : sens_loss_pairs) {
    // If the shape of grad-sens tensor is not [] or [1], use get tensor slice to handel it.
    // If the type of sens node is not Tensor, it is unsupported now, do nothing default.
    StepSplitSens(pair);
  }

  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      if (!IsValueNode<Primitive>(cnode->input(0))) {
        continue;
      }
      OperatorInfoPtr distribute_operator = GetDistributeOperator(cnode);
      if (distribute_operator == nullptr) {
        continue;
      }

      // insert forward ops
      InsertForwardOps(distribute_operator, cnode);

      // insert redistribution ops
      StepRedistribution(cnode, distribute_operator, cnode, tensor_redistribution, cnode);

      // insert backward ops
      if (has_backward) {
        BackwardCommunication(distribute_operator, cnode, sens_loss_pairs);
      }

      HandleSpecialNode(distribute_operator, cnode);
    } else if (IsValueNode<Tensor>(node)) {
      StepSplitTensor(node, manager);
    }
  }

  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      if (!IsValueNode<Primitive>(cnode->input(0))) {
        continue;
      }
      OperatorInfoPtr distribute_operator = GetDistributeOperator(cnode);
      if (distribute_operator == nullptr) {
        continue;
      }
      // StepReplace
      StepReplace(distribute_operator, cnode);
    }
  }
}

namespace {
void RevertSymbolicKeyInstance(const FuncGraphPtr &root, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(node);
  auto symbolic_key = GetValueNode<SymbolicKeyInstancePtr>(node);
  MS_EXCEPTION_IF_NULL(symbolic_key);
  auto all_upstream_node = root->manager()->node_users()[node];
  for (auto &upstream_node : all_upstream_node) {
    FuncGraphPtr fg = upstream_node.first->func_graph();
    if (symbolic_key->node()->isa<Parameter>()) {
      for (auto &param : root->parameters()) {
        if (*param == *symbolic_key->node()) {
          AnfNodePtr reverted_node = root->NewCNode({NewValueNode(prim::kPrimEmbed), param});
          MS_EXCEPTION_IF_NULL(reverted_node);
          MS_LOG(DEBUG) << "before replace " << node->ToString() << " to node " << reverted_node->DebugString();
          (void)fg->manager()->Replace(node, reverted_node);
          MS_LOG(DEBUG) << "revert node " << node->ToString() << " to node " << reverted_node->DebugString();
        }
      }
    }
  }
}
}  // namespace

void HandleSymbolicKeyInstance(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes) {
  MS_EXCEPTION_IF_NULL(root);
  for (auto &node : all_nodes) {
    // revert back SymbolicKeyInstance to embed() primitive
    if (IsValueNode<SymbolicKeyInstance>(node)) {
      RevertSymbolicKeyInstance(root, node);
      continue;
    }
  }
}

std::string NodeParameterName(const CNodePtr &node) {
  std::vector<AnfNodePtr> node_inputs{node->inputs()};
  for (auto input : node_inputs) {
    if (input->isa<Parameter>()) {
      auto input_parameter = input->cast<ParameterPtr>();
      if (input_parameter->has_default()) {
        auto param_value = std::dynamic_pointer_cast<ParamValuePy>(input_parameter->default_param());
        if (py::cast<bool>(parse::python_adapter::GetPyObjAttr(param_value->value(), REQUIRES_GRAD))) {
          return py::cast<std::string>(parse::python_adapter::GetPyObjAttr(param_value->value(), PARAM_NAME));
        }
      }
    }
  }
  return "";
}

void CheckpointStrategy(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(DEBUG) << "Save strategy to checkpoint begin";
  StrategyMap stra_map;
  auto ret = func_graph->get_return();
  auto all_nodes = DeepScopedGraphSearch(ret);
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    auto cnode = node->cast<CNodePtr>();
    if ((cnode == nullptr) || !IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    std::string param_name = NodeParameterName(cnode);
    if (param_name.empty()) {
      continue;
    }
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_EXCEPTION_IF_NULL(prim);
    OperatorInfoPtr operator_info = cnode->operator_info();
    if (operator_info) {
      StrategyPtr strategyPtr = operator_info->strategy();
      MS_EXCEPTION_IF_NULL(node->scope());
      stra_map[param_name] = strategyPtr;
    }
  }
  if (StrategyCheckpoint::GetInstance().Save(stra_map) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Save strategy checkpoint failed";
  }
}

void SetForwardFlag(const std::vector<AnfNodePtr> &all_nodes) {
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }

    // CNode is globally unique.
    MS_LOG(DEBUG) << "Set forward flag " << cnode->DebugString() << ".";
    cnode->set_in_forward_flag(true);
  }
}

void SetForwardFlag(const AnfNodeSet &all_nodes) {
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }

    // CNode is globally unique.
    cnode->set_in_forward_flag(true);
  }
}

std::set<FuncGraphPtr> ForwardGraph(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  const auto &all_nodes = root->nodes();
  std::set<FuncGraphPtr> graph_set = FindForwardGraphByRootNodes(all_nodes);
  return graph_set;
}

std::vector<AnfNodePtr> FindRootForwardCNode(const FuncGraphPtr &graph, const AnfNodeSet &all_nodes) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> root_forward_nodes;
  auto loss_cnode = FindLossCNode(graph);
  if (loss_cnode == nullptr) {
    MS_LOG(WARNING) << "Can not find the loss cnode";
    return root_forward_nodes;
  }

  auto loss_cnode_id = loss_cnode->UniqueIdThroughCopy();
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto root_node_id = node->UniqueIdThroughCopy();
    if (loss_cnode_id == root_node_id) {
      root_forward_nodes = DeepLinkedGraphSearch(cnode);
      break;
    }
  }
  return root_forward_nodes;
}

void MarkForwardCNode(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  auto all_nodes = root->nodes();
  std::set<FuncGraphPtr> graph_set = FindForwardGraphByRootNodes(all_nodes);

  if (graph_set.empty()) {
    MS_LOG(INFO) << "Can not find the forward graph, so mark the ops in root graph";
    SetForwardFlag(all_nodes);
  } else {
    for (auto &func_graph : graph_set) {
      MS_LOG(INFO) << "The sub graph size of root is " << root->func_graphs_used().size();
      auto return_node = func_graph->get_return();
      MS_EXCEPTION_IF_NULL(return_node);
      auto all_dfs_nodes = DeepLinkedGraphSearch(return_node);
      SetForwardFlag(all_dfs_nodes);
      auto root_forward_nodes = FindRootForwardCNode(func_graph, all_nodes);
      if (root_forward_nodes.empty()) {
        continue;
      }
      // Mark forward flag for the nodes in root graph.
      SetForwardFlag(root_forward_nodes);
    }
  }
}

Status ParallelInit() {
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  int32_t device_num = ParallelContext::GetInstance()->device_num();
  int32_t global_rank = ParallelContext::GetInstance()->global_rank();
  std::string backend = ParallelContext::GetInstance()->communication_backend();
  std::string world_group;

  if (backend == HCCL_BACKEND) {
    world_group = HCCL_WORLD_GROUP;
  } else if (backend == NCCL_BACKEND) {
    world_group = NCCL_WORLD_GROUP;
  } else {
    MS_LOG(EXCEPTION) << "Invalid communication backend: " << backend;
  }

  uint32_t world_rank_size = 0;
  if (!ParallelContext::GetInstance()->device_num_is_set()) {
    if (!CommManager::GetInstance().GetRankSize(world_group, &world_rank_size)) {
      MS_LOG(EXCEPTION) << "Get rank size failed";
    }
    device_num = UintToInt(world_rank_size);
    MS_LOG(INFO) << "Get device num from communication model, the device num is  " << device_num;
  }

  uint32_t rank_id = 0;
  if (!ParallelContext::GetInstance()->global_rank_is_set()) {
    if (!CommManager::GetInstance().GetRankID(world_group, &rank_id)) {
      MS_LOG(EXCEPTION) << "Get rank id failed";
    }
    global_rank = UintToInt(rank_id);
    MS_LOG(INFO) << "Get global rank from communication model, the global rank is  " << global_rank;
  }

  if (!InitDevice(device_num, global_rank, backend)) {
    MS_LOG(ERROR) << "Init device failed";
    return FAILED;
  }

  MS_LOG(INFO) << "The parallel context: dev num: " << device_num << ", global rank: " << global_rank
               << ", backend: " << backend << ", mirror_mean: " << ParallelContext::GetInstance()->mirror_mean()
               << ", cast_before_mirror: " << ParallelContext::GetInstance()->cast_before_mirror();
  return SUCCESS;
}

bool StepParallel(const FuncGraphPtr &root, const opt::OptimizerPtr &optimizer) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(optimizer);
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  std::string parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  // assume no change to graph
  bool changes = false;
  // control whether use model_parallel mode
  if (!root->has_flag(AUTO_PARALLEL) || ((parallel_mode != AUTO_PARALLEL) && (parallel_mode != SEMI_AUTO_PARALLEL)) ||
      (root->has_flag(SEMI_AUTO_PARALLEL_RUN_ONCE_ONLY))) {
    if (!root->has_flag(CHECK_SET_STRATEGY_VALID_ONCE_ONLY)) {
      if (HasStrategy(root)) {
        MS_LOG(INFO) << "Strategies ignored in " << parallel_mode
                     << ", set_strategy() only valid in [semi_]auto_parallel.";
      }
      root->set_flag(CHECK_SET_STRATEGY_VALID_ONCE_ONLY, true);
    }

    return changes;
  }

  struct timeval start_time, end_time;
  (void)gettimeofday(&start_time, nullptr);

  MS_LOG(INFO) << "Now entering step parallel";
  DumpGraph(root, std::string(STEP_PARALLEL_BEGIN));

  pipeline::ResourceBasePtr res = optimizer->resource();
  MS_EXCEPTION_IF_NULL(res);

  FuncGraphManagerPtr manager = res->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodePtr ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  std::reverse(all_nodes.begin(), all_nodes.end());
  if (parallel_mode != AUTO_PARALLEL) {
    TOTAL_OPS = 0;
    if (ParallelInit() != SUCCESS) {
      MS_LOG(EXCEPTION) << "Parallel init failed";
    }

    // mark the forward cnodes, parallel only care these nodes
    MarkForwardCNode(root);

    if (FindCommunicationOp(all_nodes)) {
      MS_LOG(EXCEPTION) << "The graph contain communication op";
    }

    // extract shape and strategy, set operator_info
    ExtractInformation(all_nodes);
    ReshapeInit(all_nodes);
  }
  // save strategy as checkpoint for multi-train
  if (StrategyCheckpoint::GetInstance().SaveCheckPointOn()) {
    CheckpointStrategy(root);
  }

  HandleSymbolicKeyInstance(root, all_nodes);

  // cover Parallel shape
  CoverSliceShape(root);

  // set the shape for optimizer's clone tensor
  SetClonedTensorShapeForOptimizer(root);

  // ForwardCommunication BackwardCommunication TensorRedistribution
  ParallelCommunication(root, all_nodes, manager);

  DumpGraph(root, std::string(STEP_PARALLEL_END));

  // step parallel only run once
  root->set_flag(SEMI_AUTO_PARALLEL_RUN_ONCE_ONLY, true);
  res->results()[pipeline::kStepParallelGraph] = root;

  // in auto parallel mode, no need to check if stategies set
  root->set_flag(CHECK_SET_STRATEGY_VALID_ONCE_ONLY, true);

  (void)gettimeofday(&end_time, nullptr);
  uint64_t time = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  time += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "Now leaving step parallel, used time: " << time << " us";
  return changes;
}

// Needed by rec_parser
std::vector<std::string> ExtractInputsTensorName(const CNodePtr &node) {
  std::vector<std::string> name_inputs;
  std::vector<AnfNodePtr> all_inputs = node->inputs();
  std::vector<AnfNodePtr> node_inputs{all_inputs.begin() + 1, all_inputs.end()};

  std::string node_id = node->UniqueId();
  name_inputs.push_back(node_id);
  for (auto &input : node_inputs) {
    std::string name = input->UniqueId();
    name_inputs.push_back(name);
  }

  return name_inputs;
}
}  // namespace parallel
}  // namespace mindspore
