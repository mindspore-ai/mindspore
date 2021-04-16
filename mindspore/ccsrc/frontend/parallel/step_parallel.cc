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

#include "frontend/parallel/step_parallel.h"

#include <inttypes.h>
#include <sys/time.h>
#include <algorithm>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include "base/core_ops.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/context.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/node_check.h"
#include "frontend/parallel/ops_info/matmul_info.h"
#include "frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.h"
#include "ir/param_info.h"
#include "ir/tensor.h"
#include "utils/comm_manager.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"
#include "mindspore/core/utils/parallel_node_check.h"
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
#include "ps/util.h"
#include "ps/ps_context.h"
#endif

using mindspore::tensor::Tensor;

namespace mindspore {
namespace parallel {
static const std::set<std::string> COMMUNICATION_OPS = {ALL_REDUCE, ALL_GATHER, ALL_TO_ALL, REDUCE_SCATTER};
static const std::set<std::string> INVALID_LOSS_OPS = {GET_NEXT, VIRTUALLOSS, LOAD, UPDATESTATE};
// g_RefMap, for CNode B input i is a RefKey[Parameter C],
// it will be one item in map with key: C, and value: (B, i)
static std::map<AnfNodePtr, std::pair<AnfNodePtr, int64_t>> g_RefMap;

void SetCommunicationOpGroupLabel(std::vector<AnfNodePtr> new_node_input) {
  if (new_node_input.empty()) {
    return;
  }

  auto prim_anf_node = new_node_input[0]->cast<ValueNodePtr>();
  auto prim = GetValueNode<PrimitivePtr>(prim_anf_node);
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

void SetMiniStepOpDoMirrorLabel(std::vector<AnfNodePtr> new_node_input, bool accu_flag) {
  if (new_node_input.empty()) {
    return;
  }
  auto prim_anf_node = new_node_input[0]->cast<ValueNodePtr>();
  auto prim = GetValueNode<PrimitivePtr>(prim_anf_node);
  MS_EXCEPTION_IF_NULL(prim);

  auto attrs = prim->attrs();
  attrs[DO_MIRROR] = MakeValue<bool>(!accu_flag);
  prim->SetAttrs(attrs);
}

void SetAllReduceRecomputeFlag(const std::vector<AnfNodePtr> &new_node_input, const CNodePtr &node) {
  if (new_node_input.empty()) {
    return;
  }

  auto prim_anf_node = new_node_input[0]->cast<ValueNodePtr>();
  auto prim = GetValueNode<PrimitivePtr>(prim_anf_node);
  MS_EXCEPTION_IF_NULL(prim);
  auto attrs = prim->attrs();

  auto anf_node = node->input(0)->cast<ValueNodePtr>();
  auto prim_node = GetValueNode<PrimitivePtr>(anf_node);
  MS_EXCEPTION_IF_NULL(prim_node);
  auto node_attrs = prim_node->attrs();
  if (node_attrs.find(RECOMPUTE_COMM_OP) != node_attrs.end() && !GetValue<bool>(node_attrs[RECOMPUTE_COMM_OP])) {
    attrs[RECOMPUTE] = MakeValue<bool>(false);
    prim->SetAttrs(attrs);
    MS_LOG(INFO) << "Do not recompute the forward communication operator of " << prim_node->ToString();
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
      int64_t position = param.second;
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
  manager->SetEdge(node, SizeToLong(index), new_node);
  MS_LOG(INFO) << "Insert " << instance_name << " success";
}

bool ParameterIsCloned(const AnfNodePtr &parameter_node) {
  MS_EXCEPTION_IF_NULL(parameter_node);
  auto cloned_parameter = parameter_node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(cloned_parameter);

  // find the clone parameter
  if (!cloned_parameter->has_default()) {
    return false;
  }
  auto param_value = cloned_parameter->param_info();
  if (param_value == nullptr) {
    return false;
  }
  bool cloned = param_value->cloned();
  if (!cloned) {
    return false;
  }

  MS_LOG(INFO) << "The parameter: " << cloned_parameter->name() << " is cloned";
  return true;
}

std::vector<AnfNodePtr> CreateMirrorInput(const FuncGraphPtr &root, const Operator &op, const AnfNodePtr &node,
                                          const std::string &instance_name, const std::string &weight_name) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(root->manager());

  AnfNodePtr grad_accu = nullptr;
  std::string op_name = op.first;
  OperatorArgs arg_forward = op.second;

  int64_t grad_accumulation_step = ParallelContext::GetInstance()->grad_accumulation_step();

  if (grad_accumulation_step > 1) {
    auto parameters = root->parameters();
    bool find_grad_accu_node = false;
    for (auto &param : parameters) {
      if (!ParameterIsCloned(param)) {
        continue;
      }

      auto param_ptr = param->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(param_ptr);
      if (param_ptr->name().find(weight_name) != std::string::npos &&
          param_ptr->name().find(ACCU_GRADS) != std::string::npos) {
        find_grad_accu_node = true;
        grad_accu = param;
        MS_LOG(INFO) << "Find the accumulation grad node: " << param_ptr->name();
        break;
      }
    }

    if (!find_grad_accu_node) {
      if (op_name == MIRROR_MINI_STEP_OPERATOR) {
        op_name = MIRROR_OPERATOR;
        arg_forward.first.pop_back();
      } else if (op_name == MINI_STEP_ALL_GATHER) {
        MS_LOG(EXCEPTION) << "You should define `accu_grads` when enable gradient accumulation.";
      }
    }
  }

  ValuePtr pyop_instance = CreatOpInstance(arg_forward.first, op_name, instance_name);
  MS_EXCEPTION_IF_NULL(pyop_instance);
  OperatorParams params = arg_forward.second;

  std::vector<AnfNodePtr> new_node_input;
  if (op_name == MIRROR_MINI_STEP_OPERATOR || op_name == MINI_STEP_ALL_GATHER) {
    new_node_input = {NewValueNode(pyop_instance), node, grad_accu};
    MS_LOG(INFO) << "Insert the grad accumulation node as the mirror op's input";
  } else {
    new_node_input = {NewValueNode(pyop_instance), node};
  }

  if (!params.empty()) {
    for (auto &param : params) {
      AnfNodePtr val = NewValueNode(param.first.second);
      MS_EXCEPTION_IF_NULL(val);
      int64_t position = param.second;
      (void)new_node_input.insert(new_node_input.begin() + position, val);
    }
  }

  // if the op have 'group' attr, set the rank list name for the op
  SetCommunicationOpGroupLabel(new_node_input);
  // gradient accumulation
  if (grad_accumulation_step > 1) {
    SetMiniStepOpDoMirrorLabel(new_node_input, root->has_flag(ACCUMULATION));
  }
  return new_node_input;
}

void InsertMirrorNode(const FuncGraphPtr &root, const Operator &op, const CNodePtr &node, size_t index,
                      const AnfNodePtr &pre_node, const FuncGraphPtr &func_graph, const std::string &instance_name,
                      const std::string &param_name) {
  // insert new node before the node
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  ScopePtr scope = node->scope();
  MS_EXCEPTION_IF_NULL(scope);
  std::vector<AnfNodePtr> node_input = CreateMirrorInput(root, op, pre_node, instance_name, param_name);
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
  manager->SetEdge(node, SizeToLong(index), new_node);
  MS_LOG(INFO) << "Insert " << instance_name << " success";
}

// Replace pre_node with pre_node->op
static CNodePtr ReplaceNode(const Operator &op, const AnfNodePtr &pre_node, const FuncGraphPtr &func_graph,
                            const std::string &instance_name) {
  // insert new node before the node
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  ScopePtr scope = pre_node->scope();
  MS_EXCEPTION_IF_NULL(scope);
  std::vector<AnfNodePtr> node_input = CreateInput(op, pre_node, instance_name);
  CNodePtr new_node = func_graph->NewCNode(node_input);
  MS_EXCEPTION_IF_NULL(new_node);
  if (instance_name.find(SPLIT_SENS) == std::string::npos) {
    new_node->set_in_forward_flag(true);  // mark forward flag
  }
  auto new_node_prim = GetValueNode<PrimitivePtr>(node_input[0]);
  new_node_prim->set_instance_name(instance_name);
  new_node_prim->set_attr("keep_value_node_input", MakeValue(true));
  new_node->set_scope(scope);
  node_input[0]->set_scope(scope);
  manager->Replace(pre_node, new_node);
  MS_LOG(INFO) << "Insert " << instance_name << " success";
  return new_node;
}

// Replace pre_node with pre_node->op
static CNodePtr ReplaceMirrorNode(const FuncGraphPtr &root, const Operator &op, const AnfNodePtr &pre_node,
                                  const FuncGraphPtr &func_graph, const std::string &instance_name,
                                  const std::string &param_name) {
  // insert new node before the node
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  ScopePtr scope = pre_node->scope();
  MS_EXCEPTION_IF_NULL(scope);
  std::vector<AnfNodePtr> node_input = CreateMirrorInput(root, op, pre_node, instance_name, param_name);
  CNodePtr new_node = func_graph->NewCNode(node_input);
  MS_EXCEPTION_IF_NULL(new_node);
  if (instance_name.find(SPLIT_SENS) == std::string::npos) {
    new_node->set_in_forward_flag(true);  // mark forward flag
  }
  auto new_node_prim = GetValueNode<PrimitivePtr>(node_input[0]);
  new_node_prim->set_instance_name(instance_name);
  new_node_prim->set_attr("keep_value_node_input", MakeValue(true));
  new_node->set_scope(scope);
  node_input[0]->set_scope(scope);
  manager->Replace(pre_node, new_node);
  MS_LOG(INFO) << "Insert " << instance_name << " success";
  return new_node;
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
    if (value_node_prim->name() == prim::kTupleGetItem) {
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
    SetAllReduceRecomputeFlag(forward_input, node_to_insert);
    CNodePtr forward_node = func_graph->NewCNode(forward_input);  // using NewCNode to create anfnode
    MS_EXCEPTION_IF_NULL(forward_node);
    ScopePtr scope = node->scope();
    MS_EXCEPTION_IF_NULL(scope);
    forward_node->set_scope(scope);
    forward_node->set_in_forward_flag(true);
    forward_input[0]->set_scope(scope);
    (void)manager->Replace(node_to_insert, forward_node);  // using Replace function to insert node
  }
}

CNodePtr InsertMakeTuple(const AnfNodePtr &prev, uint64_t num, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(prev);
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> make_tuple_inputs;
  make_tuple_inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
  for (uint64_t i = 0; i < num; i++) {
    std::vector<AnfNodePtr> tuple_get_item_inputs{NewValueNode(prim::kPrimTupleGetItem), prev,
                                                  CreatInt64Imm(UlongToLong(i))};
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
                          const FuncGraphPtr &func_graph, int64_t pos, const CNodePtr &pre_node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(pre_node);
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if ((redistribution_oplist_ptr->first).size() != (redistribution_oplist_ptr->second).size()) {
    MS_LOG(EXCEPTION) << "size of OperatorVector and OutPutInfoVector must be the same!";
  }
  for (size_t index = 0; index < (redistribution_oplist_ptr->first).size(); ++index) {
    if (pos >= SizeToLong(node->inputs().size())) {
      MS_LOG(EXCEPTION) << "InsertRedistribution:pos can't be larger than node's inputs'size";
    }
    // Create new node
    AnfNodePtr target_node = node->input(LongToSize(pos));
    MS_EXCEPTION_IF_NULL(target_node);
    // Create instance_name
    auto op = (redistribution_oplist_ptr->first)[index];
    std::string op_name = (redistribution_oplist_ptr->first)[index].first;
    std::string instance_name_base = REDISTRIBUTION_OP;
    std::string instance_name = instance_name_base + "_" + CreateInstanceName(pre_node, index) + op_name;
    InsertNode(op, node, LongToSize(pos), target_node, func_graph, instance_name);
    if ((redistribution_oplist_ptr->second)[index].first) {
      target_node = node->input(LongToSize(pos));
      MS_EXCEPTION_IF_NULL(target_node);
      (void)InsertMakeTuple(target_node, (redistribution_oplist_ptr->second)[index].second, func_graph);
    }
  }
}

void InsertGetTensorSliceOp(const Operator &op, const CNodePtr &node, const FuncGraphPtr &func_graph, int64_t pos,
                            const std::string &instance_name) {
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "InsertGetTensorSliceOp: the graph is null, the instance name is " << instance_name;
  }

  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (pos >= SizeToLong(node->inputs().size())) {
    MS_LOG(EXCEPTION) << "InsertGetTensorSliceOp: pos can't be larger than node's inputs'size, the instance name is "
                      << instance_name;
  }
  // Create new node
  AnfNodePtr pre_node = node->input(LongToSize(pos));
  MS_EXCEPTION_IF_NULL(pre_node);
  InsertNode(op, node, LongToSize(pos), pre_node, func_graph, instance_name);
}

TensorLayout GetTensorInLayout(const CNodePtr &middle_node, const PrimitivePtr &middle_prim,
                               const OperatorInfoPtr &distribute_operator) {
  TensorInfo tensorinfo_in;
  if (middle_prim->name() == prim::kTupleGetItem) {
    auto value_node = middle_node->input(2)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    size_t index_s = LongToSize(GetValue<int64_t>(value_node->value()));
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

std::string GetPrimName(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsValueNode<Primitive>(node->input(0))) {
    MS_LOG(EXCEPTION) << "The node is not a primitive";
  }
  auto value_node = node->input(0)->cast<ValueNodePtr>();
  auto prim = GetValueNode<PrimitivePtr>(value_node);
  MS_EXCEPTION_IF_NULL(prim);
  return prim->name();
}

OperatorInfoPtr GetDistributeOperator(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsParallelCareNode(node)) {
    return nullptr;
  }
  OperatorInfoPtr distribute_operator = node->user_data<OperatorInfo>();
  if (distribute_operator == nullptr) {
    MS_LOG(EXCEPTION) << "Distribute operator is nullptr, the prim is " << GetPrimName(node);
  }
  return distribute_operator;
}

void Redistribution(const std::pair<AnfNodePtr, int64_t> &node_pair, const OperatorInfoPtr &distribute_operator,
                    const CNodePtr &middle_node, int64_t index, TensorRedistribution tensor_redistribution,
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
  RankList dev_list = distribute_operator->stage_device_list();
  std::string next_prim_name = GetValueNode<PrimitivePtr>(next_node->input(0))->name();
  MS_LOG(DEBUG) << "Redistribution: middle_prim " << middle_prim->name() << " next_prim " << next_prim_name;
  MS_LOG(DEBUG) << "Redistribution: middle_node " << middle_node->ToString() << " next_node " << next_node->ToString();
  // extract tensor layout in and out
  if (distribute_operator->outputs_tensor_info().empty()) {
    MS_LOG(WARNING) << "pre_node's tensorinfo_in is empty, operator name is " << distribute_operator->name();
    return;
  }

  if (LongToSize(index - 1) >= next_distribute_operator->inputs_tensor_info().size()) {
    MS_LOG(WARNING) << "The index is out of range, the index is " << index - 1 << ", the vector size is "
                    << next_distribute_operator->inputs_tensor_info().size() << "next operator name is "
                    << next_distribute_operator->name();
    return;
  }
  TensorInfo tensorinfo_out = next_distribute_operator->inputs_tensor_info()[LongToSize(index - 1)];
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
  if (IsInParallelBlackList(prim)) {
    MS_LOG(DEBUG) << "Parallel don't care node: " << prim->name();
    return false;
  }
  // get_next is not in the forward graph, we need mark the get_next as the forward node
  if (prim->name() == GET_NEXT) {
    return true;
  }
  if ((prim->name() == CAST) && !cnode->has_user_data<OperatorInfo>()) {
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

  if (AnfNodeIsPrimitive(node, MAKE_TUPLE) || AnfNodeIsPrimitive(node, MAKE_LIST)) {
    MS_LOG(INFO) << "No need to insert redistribution op between make_tuple node and the next node";
    return;
  }
  if (IsValueNode<Primitive>(node->input(0))) {
    auto current_value = node->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(current_value);
    PrimitivePtr current_prim = current_value->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(current_prim);
    insert_node_new = ((current_prim->name() == prim::kTupleGetItem) ? node : insert_node);
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
      if ((node_prim->name() == DEPEND && node_pair.second != 1) || node_prim->name() == UPDATESTATE) {
        continue;
      }
      if (IsParallelCareNode(use_cnode) && use_cnode->has_user_data<OperatorInfo>()) {
        Redistribution(node_pair, distribute_operator, insert_node_new, node_pair.second, tensor_redistribution,
                       pre_node);
      } else {
        StepRedistribution(use_cnode, distribute_operator, insert_node_new, tensor_redistribution, pre_node);
      }
    }
  }
}

void SplitTensor(const AnfNodePtr &node, const CNodePtr &next_node, int64_t index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(next_node);
  OperatorInfoPtr op_info = next_node->user_data<OperatorInfo>();
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
  if (LongToSize(index - 1) >= op_info->inputs_tensor_info().size()) {
    MS_LOG(EXCEPTION) << "The index is out of range, index is  " << index - 1 << ", vector size is  "
                      << op_info->inputs_tensor_info().size();
  }
  TensorInfo tensor_info = op_info->inputs_tensor_info()[LongToSize(index - 1)];
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

void SplitTensorList(const AnfNodePtr &node, const CNodePtr &next_node, int index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(next_node);
  if (next_node->inputs().size() != 2 || index != 1) {
    MS_LOG(INFO) << next_node->fullname_with_scope() << " Inputs must have only one input, get "
                 << next_node->inputs().size() - 1 << " index should be 1, get " << index;
    return;
  }
  OperatorInfoPtr op_info = next_node->user_data<OperatorInfo>();
  MS_EXCEPTION_IF_NULL(op_info);

  std::vector<ValuePtr> inputs_values;
  if (IsValueNode<ValueList>(node)) {
    inputs_values = node->cast<ValueNodePtr>()->value()->cast<ValueListPtr>()->value();
  } else {
    inputs_values = node->cast<ValueNodePtr>()->value()->cast<ValueTuplePtr>()->value();
  }
  if (inputs_values.size() != op_info->inputs_tensor_info().size()) {
    MS_LOG(EXCEPTION) << "The inputs size " << inputs_values.size() << ", is not equal to inputs shape size "
                      << op_info->inputs_tensor_info().size();
  }
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  FuncGraphPtr func_graph = next_node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  ScopePtr scope = next_node->scope();
  MS_EXCEPTION_IF_NULL(scope);
  for (size_t i = 0; i < inputs_values.size(); ++i) {
    auto value_ptr = inputs_values[i];
    auto tensor = value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    TensorInfo tensor_info = op_info->inputs_tensor_info()[i];
    TensorLayout tensor_layout = tensor_info.tensor_layout();
    auto value_node = NewValueNode(value_ptr)->cast<AnfNodePtr>();
    Operator op = CreateGetTensorSliceOp(tensor_layout);
    std::vector<AnfNodePtr> node_input = CreateInput(op, value_node, SPLIT_TENSOR);
    CNodePtr new_node = func_graph->NewCNode(node_input);
    new_node->set_in_forward_flag(true);
    auto new_node_value = node_input[0]->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(new_node_value);
    PrimitivePtr new_node_prim = new_node_value->value()->cast<PrimitivePtr>();
    new_node_prim->set_instance_name(SPLIT_TENSOR);
    new_node_prim->set_attr("keep_value_node_input", MakeValue(true));
    new_node->set_scope(scope);
    node_input[0]->set_scope(scope);
    make_tuple_inputs.push_back(new_node);
  }
  CNodePtr make_tuple = func_graph->NewCNode(make_tuple_inputs);
  manager->Replace(node, make_tuple);
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
      if (IsValueNode<ValueList>(node) || IsValueNode<ValueTuple>(node)) {
        SplitTensorList(node, use_cnode, node_pair.second);
      } else {
        SplitTensor(node, use_cnode, node_pair.second);
      }
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
  if (replace_op.first == EMBEDDING_LOOKUP) {
    replace_input = {NewValueNode(pyop_instance), node->input(1), node->input(2)};
  }
  if (!params.empty()) {
    Param param_first = *(params.begin());
    int64_t first_position = param_first.second;
    if (first_position == 1) {
      replace_input.pop_back();
    }
    for (auto &param : params) {
      AnfNodePtr val = NewValueNode(param.first.second);
      if (val == nullptr) {
        MS_LOG(EXCEPTION) << "Failure:val is nullptr";
      }
      int64_t position = param.second;
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
  if (node->inputs().size() == DROPOUT_DO_MASK_CNODE_INPUT_SIZE) {
    replace_input.push_back(node->input(3));
  }
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
  OperatorInfoPtr distribute_operator = node->user_data<OperatorInfo>();
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
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(replace_node->input(0));
    if (prim->name() == EMBEDDING_LOOKUP) {
      auto attrs = prim->attrs();
      attrs[TARGET] = MakeValue(CPU);
      (void)prim->SetAttrs(attrs);
    }
    if (index == replace_op.size() - 1) {
      replace_node->set_user_data<OperatorInfo>(node->user_data<OperatorInfo>());
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
  // Solve the input order
  // For example input_node:{segment_sum:1, segment_sum:2, gahter:2}
  // The Original code here will bind the all operations to the first inputs of these operatos
  // However, the segment_sum operation needs two inputs, To solve this
  // We maintain a dict to count the times of the same operations,
  // and bind the inputs according to the times of the op appears.
  static std::unordered_map<AnfNodePtr, int> input_map = {};
  static int appear_count = 0;
  for (auto &replace_input : replace_graph->first) {
    auto pre_node = node->input(LongToSize(replace_input.second));

    auto it = input_map.find(replace_input.first);
    if (it != input_map.end()) {
      appear_count = 1 + it->second;
    } else {
      appear_count = 1;
    }
    input_map[replace_input.first] = appear_count;
    manager->SetEdge(replace_input.first, appear_count, pre_node);
  }
  //  "(void)manager->Replace(replace_graph->first, pre_node);" can not be called
  auto replace_output = replace_graph->second;
  MS_EXCEPTION_IF_NULL(replace_output);
  (void)manager->Replace(node, replace_output);
}

int64_t GetTupleGetItemIndex(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().size() != 3) {
    MS_LOG(EXCEPTION) << cnode->ToString() << " size( " << cnode->inputs().size() << " ) is not 3";
  }

  if (!cnode->input(2)->isa<ValueNode>()) {
    MS_LOG(EXCEPTION) << "The index of tuple getitem is not a value node";
  }

  ValuePtr tuple_index_value = GetValueNode(cnode->input(2));
  MS_EXCEPTION_IF_NULL(tuple_index_value);
  if (!tuple_index_value->isa<Int64Imm>()) {
    MS_LOG(EXCEPTION) << "The index of tuple getitem is not int32";
  }
  return tuple_index_value->cast<Int64ImmPtr>()->value();
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
    // if it is not a tensor, continue
    if ((!input->isa<CNode>() && !input->isa<Parameter>()) || HasAbstractMonad(input)) {
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

static std::pair<AnfNodePtr, bool> FindParameterByValueNode(const AnfNodePtr &node, const FuncGraphPtr &func_graph) {
  if (IsValueNode<RefKey>(node)) {
    std::vector<AnfNodePtr> param_v = FindParameterByRefKeyNode(node, func_graph);
    if (param_v.size() != 1) {
      MS_LOG(EXCEPTION) << "FindParameterByRefKeyNode failed, return vector size must be 1, real is  "
                        << param_v.size();
    }
    auto param_ptr = param_v[0]->user_data<parallel::TensorLayout>();
    if (param_ptr != nullptr && !param_ptr->opt_shard_group().empty()) {
      return std::make_pair(nullptr, true);
    }
    return std::make_pair(node, true);
  }
  return std::make_pair(nullptr, false);
}

// Only used for InsertMirrorOps
std::pair<AnfNodePtr, bool> FindParameter(const AnfNodePtr &node, const FuncGraphPtr &func_graph) {
  if (!node->isa<Parameter>() && !node->isa<CNode>() && !node->isa<ValueNode>()) {
    return std::make_pair(nullptr, false);
  }

  if (node->isa<Parameter>()) {
    auto param_ptr = node->user_data<parallel::TensorLayout>();
    if (param_ptr != nullptr && !param_ptr->opt_shard_group().empty()) {
      return std::make_pair(nullptr, false);
    }
    return std::make_pair(node, false);
  }

  if (node->isa<ValueNode>()) {
    return FindParameterByValueNode(node, func_graph);
  }

  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!IsValueNode<Primitive>(cnode->input(0))) {
    for (size_t index = 0; index < cnode->inputs().size(); ++index) {
      if (!FindParameter(cnode->input(index), func_graph).first) {
        continue;
      }
      return FindParameter(cnode->input(index), func_graph);
    }
  }

  if (IsSomePrimitive(cnode, RECEIVE) && !cnode->has_user_data<OperatorInfo>()) {
    return std::make_pair(node, false);
  }

  if (IsParallelCareNode(cnode)) {
    return std::make_pair(nullptr, false);
  }

  ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(prim_anf_node);
  for (size_t index = 0; index < cnode->inputs().size(); ++index) {
    PrimitivePtr prim = prim_anf_node->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(prim);
    if ((prim->name() == DEPEND || prim->name() == LOAD) && index != 1) {
      continue;
    }
    if (!FindParameter(cnode->input(index), func_graph).first) {
      continue;
    }
    return FindParameter(cnode->input(index), func_graph);
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
  // only if gradient_fp32_sync is true, pre node is cast and type is not float32 return true
  if (!ParallelContext::GetInstance()->gradient_fp32_sync()) {
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

static bool CheckInsertMirrorOps(const MirrorOps &mirror_ops, const CNodePtr &node, size_t node_size) {
  if ((node->inputs().size() == 2) && (IsValueNode<ValueSequeue>(node->input(1)))) {
    MS_LOG(INFO) << "Input is ValueList, skip it.";
    return false;
  }

  if ((node->inputs().size() == 2) &&
      (AnfNodeIsPrimitive(node->input(1), MAKE_TUPLE) || AnfNodeIsPrimitive(node->input(1), MAKE_LIST))) {
    MS_LOG(INFO) << "The mirror for " << GetPrimName(node) << " has handle by make_tuple node";
    return false;
  }

  if (mirror_ops.size() != node_size - 1) {
    MS_LOG(EXCEPTION) << "Mirrorops's size is wrong! mirror_ops size is " << mirror_ops.size() << ", node_size is "
                      << node_size - 1;
  }
  return true;
}

void InsertMirrorOps(const FuncGraphPtr &root, const MirrorOps &mirror_ops, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  size_t node_size = node->inputs().size();
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto input : node->inputs()) {
    if (HasAbstractMonad(input)) {
      node_size--;
    }
  }

  if (!CheckInsertMirrorOps(mirror_ops, node, node_size)) {
    return;
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

    auto param_ptr = param_node_pair.first->cast<ParameterPtr>();
    std::string param_name;
    if (param_ptr != nullptr) {
      param_name = param_ptr->name();
    }

    // not a RefKey
    if (!param_node_pair.second) {
      int64_t grad_accumulation_step = ParallelContext::GetInstance()->grad_accumulation_step();
      std::string mirror_op_name;
      if (grad_accumulation_step > 1) {
        mirror_op_name = MIRROR_MINI_STEP_OPERATOR;
      } else {
        mirror_op_name = MIRROR_OPERATOR;
      }
      auto next_cnode = FindCNode(param_node_pair.first, mirror_op_name, func_graph);
      // if there is already a MirrorOp in the same graph, use MirrorOp CNode as a input instead
      if (next_cnode.first) {
        MS_EXCEPTION_IF_NULL(next_cnode.second);
        // param->cast->op, insert mirror before cast
        if (node->input(index)->isa<CNode>()) {
          auto pre_cnode = node->input(index)->cast<CNodePtr>();
          auto pre_prim = GetValueNode<PrimitivePtr>(pre_cnode->input(0));
          if ((pre_prim->name() == CAST) || (pre_prim->name() == LOAD)) {
            manager->SetEdge(pre_cnode, 1, next_cnode.second);
            continue;
          }
        }
        manager->SetEdge(node, SizeToLong(index), next_cnode.second);
        continue;
      }
    }
    // if the parameter found is a RefKey, or no MirrorOp is found in the same graph, insert a new MirrorOp
    // only one MirrorOp in backward_op
    if (backward_op.size() != 1) {
      MS_LOG(EXCEPTION) << "backward_op size must be 1, real is  " << backward_op.size();
    }
    std::string instance_name = MIRROR_OP;
    CNodePtr cnode = node->input(index)->cast<CNodePtr>();
    if (IsCastBeforMirror(node, index) || (cnode != nullptr && IsSomePrimitive(cnode, LOAD))) {
      for (auto &op : backward_op) {
        // insert new node before the node
        MS_EXCEPTION_IF_NULL(cnode);
        AnfNodePtr pre_node = cnode->input(1);
        InsertMirrorNode(root, op, cnode, size_t(1), pre_node, func_graph, instance_name, param_name);
        auto comm_op = cnode->input(size_t(1))->cast<CNodePtr>();
        // add fusion flag
        AddCommOpFusionType(comm_op, param_node_pair.first);
      }
      continue;
    }
    for (auto &op : backward_op) {
      AnfNodePtr pre_node = node->input(index);
      InsertMirrorNode(root, op, node, index, pre_node, func_graph, instance_name, param_name);
      auto comm_op = node->input(index)->cast<CNodePtr>();
      // add fusion flag
      // pipeline mirror would not be set, which should be supported later
      AddCommOpFusionType(comm_op, param_node_pair.first);
    }
  }
}

void BackwardCommunication(const FuncGraphPtr &root, const OperatorInfoPtr &distribute_operator, const CNodePtr &node,
                           const std::vector<std::pair<CNodePtr, LossNodeInfo>> &sens_loss_pairs) {
  MS_EXCEPTION_IF_NULL(distribute_operator);
  MS_EXCEPTION_IF_NULL(node);

  bool is_loss_cnode =
    std::any_of(sens_loss_pairs.begin(), sens_loss_pairs.end(),
                [node](const std::pair<CNodePtr, LossNodeInfo> &element) { return element.second.loss_node == node; });

  MirrorOps mirror_ops = distribute_operator->mirror_ops();
  VirtualDivOp virtual_div_op = distribute_operator->virtual_div_op();
  // insert mirror op
  if (!mirror_ops.empty()) {
    MS_LOG(INFO) << "insert mirror op for " << distribute_operator->name();
    InsertMirrorOps(root, mirror_ops, node);
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
    (OperatorInfoPtr)DynCreator::Instance().Create(distribute_opname, shape_list[0], shape_list[1], attrs, TOTAL_OPS);
  if (operator_ == nullptr) {
    MS_LOG(INFO) << "Create " << name << " failed";
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
    if (IsInBatchParallelBlackList(prim)) {
      MS_LOG(EXCEPTION) << "Operator " << prim->name() << " is not supported yet in auto parallel mode.";
    }
    MS_LOG(INFO) << "Create " << prim->name() << " failed, use batch parallel";
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
  int64_t stage_id = g_device_manager->stage_id();

  MS_LOG(INFO) << "Extract information: strategy " << attrs[STRATEGY]->ToString();
  if (var == nullptr) {
    MS_LOG(EXCEPTION) << "Strategy value is nullptr";
  }
  if (var->size() > 0) {
    std::vector<ValuePtr> elements = var->value();
    Strategys strategy;
    for (uint64_t index = 0; index < elements.size(); ++index) {
      Dimensions dim;
      if (elements[index]->isa<ValueSequeue>()) {
        ValueTuplePtr value_tuple = elements[index]->cast<ValueTuplePtr>();
        std::vector<ValuePtr> value_vector = value_tuple->value();
        (void)std::transform(value_vector.begin(), value_vector.end(), std::back_inserter(dim),
                             [](const ValuePtr &value) { return static_cast<int64_t>(GetValue<int64_t>(value)); });
        strategy.push_back(dim);
      } else {
        MS_LOG(EXCEPTION) << "Failure: Strategy's format is wrong! Need ValueSequence";
      }
    }
    if (strategy.empty()) {
      MS_LOG(EXCEPTION) << "ExtractStrategy: failed to extract strategy";
    }
    strategyPtr = NewStrategy(stage_id, strategy);
  }

  return strategyPtr;
}

Shapes GetValueListShape(const AnfNodePtr &node) {
  Shapes shapes;
  std::vector<ValuePtr> inputs_seq;
  if (IsValueNode<ValueList>(node)) {
    inputs_seq = node->cast<ValueNodePtr>()->value()->cast<ValueListPtr>()->value();
  } else if (IsValueNode<ValueTuple>(node)) {
    inputs_seq = node->cast<ValueNodePtr>()->value()->cast<ValueTuplePtr>()->value();
  } else {
    MS_LOG(EXCEPTION) << "node is eigther ValueList or ValueTuple";
  }
  for (auto &ele : inputs_seq) {
    auto tensor = ele->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    auto one_shape = tensor->shape();
    shapes.push_back(one_shape);
  }
  return shapes;
}

Shapes GetNodeShape(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  Shapes shapes;
  if (IsValueNode<ValueList>(node) || IsValueNode<ValueTuple>(node)) {
    return GetValueListShape(node);
  }
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
        MS_LOG(EXCEPTION) << "GetNodeShape: " << node->ToString() << " size is smaller than 2";
      }
      base_shape_ptr = cnode->input(1)->Shape();
    }
  }
  if (base_shape_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "GetNodeShape: " << node->ToString() << " shape_ptr is nullptr, full name is "
                      << node->fullname_with_scope();
  }
  auto tuple_shape_ptr = dyn_cast<abstract::SequeueShape>(base_shape_ptr);
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
    if (HasAbstractMonad(input)) {
      continue;
    }
    if (IsValueNode<RefKey>(input)) {
      auto func_graph = node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      std::vector<AnfNodePtr> parameters = FindParameterByRefKeyNode(input, func_graph);
      if (parameters.size() != 1) {
        MS_LOG(EXCEPTION) << "Find parameter by ref key node failed";
      }
      std::pair<AnfNodePtr, int64_t> node_pair = std::make_pair(node, SizeToLong(i));
      g_RefMap[parameters[0]] = node_pair;
      input_shapes = GetRefKeyNodeShape(input, func_graph);
    } else if (input->isa<CNode>() || IsValueNode<Tensor>(input) || input->isa<Parameter>() ||
               ((IsValueNode<ValueList>(input) || IsValueNode<ValueTuple>(input)) && (inputs_size == 2))) {
      input_shapes = GetNodeShape(input);
    } else {
      continue;
    }
    if (input_shapes.size() != 1) {
      if (inputs_size == 2) {  // like concat
        shape_inputs = input_shapes;
        break;
      } else {
        MS_LOG(EXCEPTION) << "ExtractShape: Get input shape failed";
      }
    }
    shape_inputs.push_back(input_shapes[0]);
  }
  shape_all.push_back(shape_inputs);
  // extract out shape
  shape_outputs = GetNodeShape(node);
  shape_all.push_back(shape_outputs);
  return shape_all;
}

std::pair<AnfNodePtr, int64_t> FindParallelCareNode(const AnfNodePtr &node, int32_t recursion_num) {
  if (recursion_num >= RECURSION_LIMIT) {
    return std::make_pair(nullptr, 0);
  }

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
    if ((node_prim->name() == DEPEND && node_pair.second != 1) || IsPrimitiveCNode(cnode, prim::kPrimReceive)) {
      continue;
    }
    if (IsParallelCareNode(cnode) && cnode->has_user_data<OperatorInfo>()) {
      return node_pair;
    } else {
      auto tmp_pair = FindParallelCareNode(node_pair.first, recursion_num + 1);
      if (tmp_pair.first != nullptr) {
        return tmp_pair;
      }
    }
  }
  return std::make_pair(nullptr, 0);
}

std::pair<AnfNodePtr, int64_t> FindSubGraph(const FuncGraphPtr &graph, const AnfNodePtr &parameter) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(parameter);
  FuncGraphManagerPtr manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::pair<AnfNodePtr, int64_t> prim_anf_node_pair = FindParallelCareNode(parameter, 0);
  if (prim_anf_node_pair.first != nullptr) {
    return prim_anf_node_pair;
  } else {
    AnfNodeIndexSet param_sub_set = manager->node_users()[parameter];
    for (auto &param_pair : param_sub_set) {
      CNodePtr param_cnode = param_pair.first->cast<CNodePtr>();
      AnfNodePtr graph_value_node;
      if (param_cnode->input(0)->isa<CNode>()) {
        graph_value_node = param_cnode->input(0)->cast<CNodePtr>()->input(1);
      } else {
        graph_value_node = param_cnode->input(0);
      }
      if (!IsValueNode<FuncGraph>(graph_value_node)) {
        continue;
      }
      FuncGraphPtr graph_sub = GetValueNode<FuncGraphPtr>(graph_value_node);
      auto parameters = graph_sub->parameters();
      if (LongToSize(param_pair.second - 1) >= parameters.size()) {
        MS_LOG(EXCEPTION) << "The index is out of range, index is " << param_pair.second - 1 << ", vector size is "
                          << parameters.size();
      }
      std::pair<AnfNodePtr, int64_t> res = FindSubGraph(graph_sub, parameters[LongToSize(param_pair.second - 1)]);
      if (res.first != nullptr) {
        return res;
      }
    }
  }
  return std::make_pair(nullptr, 0);
}

static void InsertAllGatherOp(const FuncGraphPtr &root, const std::string &group, const std::pair<AnfNodePtr, int> &res,
                              const AnfNodePtr &node, const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(res.first);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = res.first->cast<CNodePtr>();
  auto graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  auto cnode_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_EXCEPTION_IF_NULL(cnode_prim);
  Operator op;
  CNodePtr allgather;
  if (op_name == MINI_STEP_ALL_GATHER) {
    op = CreateMiniStepAllGatherOp(group);
    auto param_name = node->cast<ParameterPtr>()->name();
    if (cnode_prim->name() == CAST) {
      allgather = ReplaceMirrorNode(root, op, cnode, graph, PARALLEL_OPTIMIZER_ALLGATHER, param_name);
    } else {
      InsertMirrorNode(root, op, cnode, res.second, node, graph, PARALLEL_OPTIMIZER_ALLGATHER, param_name);
      allgather = cnode->input(res.second)->cast<CNodePtr>();
    }
  } else {
    op = CreateAllGatherOp(group);
    if (cnode_prim->name() == CAST) {
      allgather = ReplaceNode(op, cnode, graph, PARALLEL_OPTIMIZER_ALLGATHER);
    } else {
      InsertNode(op, cnode, res.second, node, graph, PARALLEL_OPTIMIZER_ALLGATHER);
      allgather = cnode->input(res.second)->cast<CNodePtr>();
    }
  }
  // add fusion flag
  AddCommOpFusionType(allgather, node);
  // add gradients mean
  AddCommOpMeanFlag(allgather);
}

static void ApplyParallelOptOnParam(const FuncGraphPtr &root, const AnfNodePtr &parameter,
                                    const std::string &opt_shard_group) {
  if (opt_shard_group.empty()) {
    return;
  }
  FuncGraphManagerPtr manager = root->manager();
  MS_EXCEPTION_IF_NULL(manager);
  int64_t grad_accumulation_step = ParallelContext::GetInstance()->grad_accumulation_step();
  std::string op_name;
  if (grad_accumulation_step > 1) {
    op_name = MINI_STEP_ALL_GATHER;
  } else {
    op_name = ALL_GATHER;
  }
  auto param_sub_set = manager->node_users()[parameter];
  bool insert_flag = false;
  for (auto &param_pair : param_sub_set) {
    auto cnode = param_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->in_forward_flag()) {
      OperatorInfoPtr distribute_operator = cnode->user_data<OperatorInfo>();
      if (distribute_operator == nullptr) {
        MS_LOG(WARNING) << "Parallel optimizer: " << GetPrimName(cnode) << " 's OperatorInfoPtr is nullptr";
      } else if (IntToSize(param_pair.second - 1) >= distribute_operator->inputs_tensor_info().size()) {
        MS_LOG(EXCEPTION) << "The index is out of range, index is  " << param_pair.second - 1 << ", vector size is  "
                          << distribute_operator->inputs_tensor_info().size();
      }
      if (insert_flag) {
        auto next_cnode = FindCNode(parameter, op_name, cnode->func_graph());
        if (next_cnode.first) {
          manager->SetEdge(cnode, SizeToLong(param_pair.second), next_cnode.second);
          MS_LOG(INFO) << "Parallel optimizer is applied between " << parameter->ToString() << " and "
                       << GetPrimName(cnode);
          continue;
        }
      } else {
        // insert allgather operator between shard parameter and cnode
        InsertAllGatherOp(root, opt_shard_group, param_pair, parameter, op_name);
        MS_LOG(INFO) << "Parallel optimizer is applied between " << parameter->ToString() << " and "
                     << GetPrimName(cnode);
        insert_flag = true;
      }
    }
  }
}

// When this function returns non-empty string, that means parallel optimizer is applied on this parameter.
std::string SetParallelShape(const AnfNodePtr &parameter, const std::pair<AnfNodePtr, int64_t> &res) {
  MS_EXCEPTION_IF_NULL(parameter);
  AbstractBasePtr abstract = parameter->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  MS_LOG(DEBUG) << "SetParallelShape " << parameter->ToString() << " shape " << parameter->Shape()->ToString();
  CNodePtr cnode = res.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  OperatorInfoPtr distribute_operator = cnode->user_data<OperatorInfo>();
  if (distribute_operator == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:node " << cnode->ToString() << " 's OperatorInfoPtr is nullptr";
  }
  if (LongToSize(res.second - 1) >= distribute_operator->inputs_tensor_info().size()) {
    MS_LOG(EXCEPTION) << "The index is out of range, index is  " << res.second - 1 << ", vector size is  "
                      << distribute_operator->inputs_tensor_info().size();
  }
  TensorInfo tensorinfo_in = distribute_operator->inputs_tensor_info()[LongToSize(res.second - 1)];
  TensorLayout tensor_layout = tensorinfo_in.tensor_layout();
  Shape slice_shape = tensor_layout.slice_shape().array();
  std::string opt_shard_group;
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  bool enable_parallel_optimizer = ParallelContext::GetInstance()->enable_parallel_optimizer();
  if (enable_parallel_optimizer) {
    if (!ParameterRequireGrad(parameter)) {
      // only trainable parameters need parallel optimizer
      MS_LOG(INFO) << "Parallel optimizer: " << parameter->ToString() << " is not trainable parameter.";
    } else if (parameter->cast<ParameterPtr>()->param_info() &&
               !parameter->cast<ParameterPtr>()->param_info()->parallel_optimizer()) {
      MS_LOG(INFO) << "Parallel optimizer: " << parameter->ToString() << " does not need weight shard.";
    } else if (tensor_layout.GenerateOptShardSliceShape() == Status::SUCCESS) {
      // get a totally shard tensor slice shape if the weight is repeated on devices
      // and the shape of the first dimension could be divided
      // apply parallel optimizer on parameters
      // create communication group for allgather operator
      slice_shape = tensor_layout.opt_shard_slice_shape();
      std::vector<Group> dev_group;
      if (distribute_operator->CreateGroupByTensorMap(tensor_layout.origin_tensor_map().array(), &dev_group) ==
            Status::SUCCESS &&
          !dev_group.empty()) {
        opt_shard_group = dev_group[0].name();
        // set communication group in tensor layout for checkpoint saving
        tensor_layout.set_opt_shard_group(opt_shard_group);
        MS_LOG(INFO) << "Parallel optimizer: create group " << opt_shard_group << " for " << parameter->ToString()
                     << " success.";
      } else {
        MS_LOG(WARNING) << "Parallel optimizer: create group for " << parameter->ToString() << " failed.";
      }
    } else {
      MS_LOG(INFO) << "Parallel optimizer: " << parameter->ToString() << "'s shape does not satisfy the conditions.";
    }
  }
  MS_LOG(INFO) << "SetParallelShape slice_shape  " << parameter->ToString() << "  shape "
               << MakeValue(slice_shape)->ToString() << ", op name is " << distribute_operator->name();
  std::shared_ptr<abstract::BaseShape> parallel_shape = std::make_shared<abstract::Shape>(slice_shape);
  MS_EXCEPTION_IF_NULL(parallel_shape);
  // Don't modify it in-place as the pointer of this AbstractValue may used as cache key in StaticAnalysis.
  auto cloned_abstract = abstract->Clone();
  MS_EXCEPTION_IF_NULL(cloned_abstract);
  cloned_abstract->set_shape(parallel_shape);
  parameter->set_abstract(cloned_abstract);
  ParameterPtr parameter_ptr = parameter->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(parameter_ptr);
  parameter_ptr->set_user_data<TensorLayout>(std::make_shared<TensorLayout>(tensor_layout));
  return opt_shard_group;
}

void CoverSliceShape(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  auto parameters = root->parameters();
  for (auto &parameter : parameters) {
    MS_EXCEPTION_IF_NULL(parameter->Shape());
    auto iter = g_RefMap.find(parameter);
    if (iter != g_RefMap.end()) {
      std::string group = SetParallelShape(parameter, g_RefMap[parameter]);
      // find all forward nodes that use parameter in graphs and insert allgather if group is not empty
      ApplyParallelOptOnParam(root, parameter, group);
      continue;
    }
    std::pair<AnfNodePtr, int64_t> res = FindSubGraph(root, parameter);
    if (res.first == nullptr) {
      MS_LOG(INFO) << "Parameter " << parameter->ToString() << " don't need to set parallel shape";
    } else {
      std::string group = SetParallelShape(parameter, res);
      // find all forward nodes that use parameter in graphs and insert allgather if group is not empty
      ApplyParallelOptOnParam(root, parameter, group);
      MS_LOG(DEBUG) << "Parameter " << parameter->ToString() << " shape " << parameter->Shape()->ToString();
    }
  }
  g_RefMap.clear();
}

void SetClonedTensorShapeForOptimizer(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  for (auto &cloned_parameter_node : root->parameters()) {
    MS_EXCEPTION_IF_NULL(cloned_parameter_node);
    auto cloned_parameter = cloned_parameter_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(cloned_parameter);

    if (!ParameterIsCloned(cloned_parameter_node)) {
      continue;
    }
    auto param_value = cloned_parameter->param_info();
    if (param_value == nullptr) {
      continue;
    }
    // get the cloned index
    int64_t cloned_index = param_value->cloned_index();

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

      auto param_value_in = be_cloned_parameter->param_info();
      if (param_value_in == nullptr) {
        continue;
      }
      if (!param_value_in->be_cloned()) {
        continue;
      }

      // get the be cloned index
      auto &be_cloned_index = param_value_in->be_cloned_index();
      if (std::find(be_cloned_index.begin(), be_cloned_index.end(), cloned_index) != be_cloned_index.end()) {
        found_be_cloned_parameter = true;
        cloned_from_parameter = be_cloned_parameter;
        cloned_from_node = be_cloned_parameter_node;
      }
    }

    if (found_be_cloned_parameter) {
      // set the shape and tensor layout for cloned parameter
      std::string param_name = cloned_parameter_node->cast<ParameterPtr>()->name();
      if (cloned_from_parameter->user_data<TensorLayout>() == nullptr) {
        MS_LOG(WARNING) << "The parameter " << param_name << " has not tensor layout, skip it";
        continue;
      }
      cloned_parameter->set_user_data<TensorLayout>(cloned_from_parameter->user_data<TensorLayout>());
      MS_EXCEPTION_IF_NULL(cloned_parameter_node->abstract());
      MS_EXCEPTION_IF_NULL(cloned_from_node->abstract());
      auto cloned_abstract = cloned_parameter_node->abstract()->Clone();
      MS_EXCEPTION_IF_NULL(cloned_abstract);
      if (param_name.find(ACCU_GRADS) != std::string::npos) {
        auto slice_shape = cloned_from_parameter->user_data<TensorLayout>()->slice_shape().array();
        std::shared_ptr<abstract::BaseShape> parallel_shape = std::make_shared<abstract::Shape>(slice_shape);
        MS_EXCEPTION_IF_NULL(parallel_shape);
        cloned_abstract->set_shape(parallel_shape);
      } else {
        cloned_abstract->set_shape(cloned_from_node->abstract()->GetShapeTrack());
      }
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
    int64_t dev_num;
    if (full_batch) {
      dev_num = 1;
    } else {
      dev_num = SizeToLong(g_device_manager->stage_device_num());
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
      Dimensions input_strategy = {dev_num};
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

// find previous parallel care node.
bool FindPreNodes(const AnfNodePtr &node, vector<std::string> *unique_ids, size_t accum = 0) {
  MS_EXCEPTION_IF_NULL(unique_ids);
  // if previous node is a parameter, handle it in the outsize.
  accum += 1;
  if (accum > MAX_RECURSIVE_DEPTH) {
    return false;
  }
  if (node->isa<Parameter>()) {
    return false;
  }
  if (!node->isa<CNode>()) {
    return false;
  }
  CNodePtr cnode = node->cast<CNodePtr>();
  if (!IsValueNode<Primitive>(cnode->input(0))) {
    return false;
  }
  ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
  PrimitivePtr prim = prim_anf_node->value()->cast<PrimitivePtr>();
  if (IsParallelCareNode(cnode) && prim->name() != MAKE_TUPLE && prim->name() != MAKE_LIST) {
    unique_ids->push_back(cnode->UniqueId());
    return true;
  }
  bool find = false;
  for (size_t index = 0; index < cnode->inputs().size(); ++index) {
    if (prim->name() == DEPEND && index != 1) {
      continue;
    }
    if (FindPreNodes(cnode->inputs()[index], unique_ids, accum)) {
      find = true;
      continue;
    }
  }
  return find;
}

void FindLastNodesUniqueId(const std::vector<AnfNodePtr> &all_nodes, std::vector<std::string> *unique_ids) {
  MS_EXCEPTION_IF_NULL(unique_ids);
  for (auto &node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if ((cnode == nullptr) || !IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);
    if (prim->name() == RETURN) {
      if (!FindPreNodes(cnode, unique_ids)) {
        MS_LOG(WARNING) << "cannot find the last parallel care node in eval graph";
      }
    }
  }
}

StrategyPtr GenerateBatchParallelStrategy(const OperatorInfoPtr operator_, const PrimitivePtr prim) {
  MS_EXCEPTION_IF_NULL(operator_);
  MS_EXCEPTION_IF_NULL(prim);
  StrategyPtr strategyPtr;
  std::shared_ptr<Strategys> strategy_v_ptr = operator_->GenerateBatchStrategies();
  MS_EXCEPTION_IF_NULL(strategy_v_ptr);
  strategyPtr = NewStrategy(0, *strategy_v_ptr);
  std::vector<ValuePtr> elements;
  for (size_t i = 0; i < strategy_v_ptr->size(); i++) {
    elements.push_back(MakeValue((*strategy_v_ptr)[i]));
  }
  ValueTuplePtr strategy = std::make_shared<ValueTuple>(elements);
  // display the strategy generated by batch parallel
  auto attrs = prim->attrs();
  attrs[GEN_STRATEGY] = strategy;
  (void)prim->SetAttrs(attrs);
  MS_LOG(INFO) << "prim " << prim->name() << " batch parallel strategy is " << attrs[GEN_STRATEGY]->ToString();
  return strategyPtr;
}

void SetLastNodeStrategy(const StrategyPtr strategyPtr) {
  auto strategys = strategyPtr->GetInputDim();
  for (size_t i = 0; i < strategys.size(); ++i) {
    for (size_t j = 0; j < strategys[i].size(); ++j) {
      strategys[i][j] = 1;
    }
  }
  strategyPtr->ResetInputs(strategys);
}

static bool CheckExtractInfomation(const CNodePtr &cnode) {
  if ((cnode == nullptr) || !IsValueNode<Primitive>(cnode->input(0))) {
    return false;
  }

  ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);
  if ((prim->name() == MAKE_TUPLE) || (prim->name() == MAKE_LIST) || (prim->name() == RECEIVE)) {
    return false;
  }

  if (!IsParallelCareNode(cnode)) {
    return false;
  }
  return true;
}

void ExtractInformation(const std::vector<AnfNodePtr> &all_nodes, bool is_training) {
  // load strategy map from checkpoint
  StrategyMap stra_map;
  if (StrategyCheckpoint::GetInstance().LoadCheckPointOn() &&
      (StrategyCheckpoint::GetInstance().Load(&stra_map) != SUCCESS)) {
    MS_LOG(EXCEPTION) << "Load strategy checkpoint failed";
  }
  vector<std::string> last_forward_node_ids;
  if (!is_training) {
    FindLastNodesUniqueId(all_nodes, &last_forward_node_ids);
    MS_LOG(INFO) << "there are " << last_forward_node_ids.size() << " output nodes in eval/predict";
  }

  for (auto &node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (!CheckExtractInfomation(cnode)) {
      continue;
    }

    SetVirtualDatasetStrategy(cnode);
    ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);

    auto attrs = prim->attrs();
    MS_LOG(INFO) << "extract information: node: " << node->ToString() << " prim " << prim->name();

    std::vector<Shapes> shape_list = ExtractShape(cnode);
    if (shape_list.empty()) {
      MS_LOG(EXCEPTION) << "Failure:node " << node->ToString() << " failed to extract shape";
    }
    OperatorInfoPtr operator_ = OperatorInstance(prim, attrs, shape_list);
    MS_EXCEPTION_IF_NULL(operator_);

    auto &inputs = cnode->inputs();
    std::vector<ValuePtr> input_value;
    for (size_t index = 1; index < inputs.size(); ++index) {
      if (inputs[index]->isa<ValueNode>()) {
        input_value.push_back(GetValueNode(inputs[index]));
        continue;
      }
      input_value.emplace_back(nullptr);
    }
    StrategyPtr strategyPtr = nullptr;
    (*operator_).set_input_value(input_value);
    (*operator_).set_outputs_dtype(cnode->Type());
    (*operator_).set_cnode(cnode);
    if (prim->name() == RESHAPE) {
      cnode->set_user_data<OperatorInfo>(operator_);
      continue;
    }
    // load strategy checkpoint
    // key of strategy map
    std::string strategy_key_name = "";
    auto param_names = NodeParameterName(cnode);
    if (!param_names.empty()) {
      strategy_key_name = prim->name() + "_" + param_names[0].first;
    }
    bool load_strategy_from_ckpt =
      StrategyCheckpoint::GetInstance().LoadCheckPointOn() && stra_map.find(strategy_key_name) != stra_map.end();
    bool is_last_nodes = std::find(last_forward_node_ids.begin(), last_forward_node_ids.end(), cnode->UniqueId()) !=
                         last_forward_node_ids.end();
    bool full_batch = ParallelContext::GetInstance()->full_batch();
    if ((is_last_nodes && !full_batch) || (!StrategyFound(attrs) && !load_strategy_from_ckpt)) {
      MS_LOG(INFO) << "ExtractInformation: the strategy of node " << node->ToString() << " prim " << prim->name()
                   << " is empty, using batch parallel";
      strategyPtr = GenerateBatchParallelStrategy(operator_, prim);
    } else if (StrategyFound(attrs)) {
      strategyPtr = ExtractStrategy(attrs);
    } else {
      strategyPtr = stra_map[strategy_key_name];
    }

    MS_EXCEPTION_IF_NULL(strategyPtr);
    if (is_last_nodes && full_batch) {
      SetLastNodeStrategy(strategyPtr);
    }
    if (operator_->Init(strategyPtr) == FAILED) {
      MS_LOG(EXCEPTION) << "Failure:operator " << prim->name() << " init failed";
    }
    cnode->set_user_data<OperatorInfo>(operator_);
  }
}

TensorLayout GetInputLayoutFromCNode(const std::pair<AnfNodePtr, int64_t> &node_pair) {
  CNodePtr cnode = node_pair.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  OperatorInfoPtr distribute_operator = GetDistributeOperator(cnode);
  MS_EXCEPTION_IF_NULL(distribute_operator);
  int64_t index = node_pair.second;
  if (index > SizeToLong(distribute_operator->inputs_tensor_info().size())) {
    MS_LOG(EXCEPTION) << "The index is out of range, the node_pair.second is  " << index - 1 << ", the vector size is  "
                      << distribute_operator->inputs_tensor_info().size();
  }
  TensorInfo tensorinfo_in = distribute_operator->inputs_tensor_info()[LongToSize(index - 1)];
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
    if (IsParallelCareNode(use_apply) && use_apply->has_user_data<OperatorInfo>()) {
      MS_LOG(INFO) << "FindNextLayout success prim " << node_prim->name();
      auto layout = GetInputLayoutFromCNode(node_pair);
      return std::make_shared<TensorLayout>(layout);
    }
    MS_LOG(DEBUG) << "FindNextLayout failed prim " << node_prim->name() << "  " << IsParallelCareNode(use_apply)
                  << "   " << use_apply->has_user_data<OperatorInfo>();

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
  if (distribute_operator->outputs_tensor_info().size() <= output_index) {
    MS_LOG(EXCEPTION) << "outputs_tensor_info size is  " << distribute_operator->inputs_tensor_info().size()
                      << ", must be greater than output_index  " << output_index;
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
  if (IsParallelCareNode(cnode) && cnode->has_user_data<OperatorInfo>()) {
    auto layout_ptr = GetOutputLayoutFromCNode(cnode, output_index);
    if (!layout_ptr) {
      MS_LOG(EXCEPTION) << "Failure:GetLayoutFromCNode failed";
    }
    return layout_ptr;
  }
  return nullptr;
}

std::shared_ptr<TensorLayout> FindParameterNextLayout(const AnfNodePtr &node, size_t accum = 0) {
  FuncGraphManagerPtr manager = node->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  accum += 1;
  if (accum > MAX_RECURSIVE_DEPTH) {
    return nullptr;
  }
  AnfNodeIndexSet node_set = manager->node_users()[node];
  for (auto &node_pair : node_set) {
    if (IsPrimitiveCNode(node_pair.first, prim::kPrimLoad)) {
      auto layout_param = FindParameterNextLayout(node_pair.first, accum);
      if (!layout_param) {
        continue;
      }
      return layout_param;
    }
    CNodePtr use_apply = node_pair.first->cast<CNodePtr>();
    if (use_apply == nullptr || !IsValueNode<Primitive>(use_apply->input(0))) {
      continue;
    }
    ValueNodePtr prim_anf_node = use_apply->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prim_anf_node);
    PrimitivePtr node_prim = prim_anf_node->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(node_prim);
    if ((node_prim->name() == DEPEND && node_pair.second != 1) || node_prim->name() == RESHAPE) {
      continue;
    }
    if (IsParallelCareNode(use_apply) && use_apply->has_user_data<OperatorInfo>()) {
      auto layout = GetInputLayoutFromCNode(node_pair);
      return std::make_shared<TensorLayout>(layout);
    }
  }
  return nullptr;
}

std::shared_ptr<TensorLayout> CreateParameterLayout(const AnfNodePtr &node) {
  // Create DataParallel tensor layout for parameter(support WideDeep).
  auto next_layout = FindParameterNextLayout(node);
  if (next_layout != nullptr) {
    return next_layout;
  }
  CheckGlobalDeviceManager();
  int64_t dev_num = g_device_manager->stage_device_num();
  TensorLayout input_tensor_layout;
  // create input_shape
  Shapes inputs_shape = GetNodeShape(node);
  Shape input_shape_array = inputs_shape[0];
  if (input_shape_array.empty()) {
    MS_LOG(EXCEPTION) << "Don't support reshape a scalar parameter.";
  }
  // create tensor_map
  size_t shape_size = input_shape_array.size();
  TensorMap input_tensor_map_array(SizeToLong(shape_size) - 1, -1);
  input_tensor_map_array.insert(input_tensor_map_array.begin(), 0);
  // create dev_matrix
  Shape dev_matrix_array = {dev_num};
  if (input_tensor_layout.InitFromVector(dev_matrix_array, input_tensor_map_array, input_shape_array) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Create tensor layout for parameter failed.";
  }
  return std::make_shared<TensorLayout>(input_tensor_layout);
}

RedistributionOpListPtr InferSensRedistribution(const AnfNodePtr &node, const TensorLayout &loss_layout) {
  MS_EXCEPTION_IF_NULL(node);
  TensorRedistribution tensor_redistribution;
  // create stand alone layout:TensorMap:[all -1],dev_matrix:[dev_num].
  CheckGlobalDeviceManager();
  int64_t dev_num = g_device_manager->stage_device_num();
  TensorLayout stand_alone_layout;
  Shapes inputs_shape = GetNodeShape(node);
  if (inputs_shape.empty()) {
    MS_LOG(EXCEPTION) << "InferSensRedistribution failed cause inputs shape is empty.";
  }
  Shape input_shape_array = inputs_shape[0];
  if (input_shape_array.empty()) {
    MS_LOG(INFO) << "No need to redistribution for sens.";
    return nullptr;
  }
  // TensorMap
  TensorMap stand_alone_tensor_map_array(SizeToLong(input_shape_array.size()), -1);
  // Dev_matrix
  Shape dev_matrix_array = {dev_num};
  if (stand_alone_layout.InitFromVector(dev_matrix_array, stand_alone_tensor_map_array, input_shape_array) == FAILED) {
    MS_LOG(EXCEPTION) << "Create tensor layout for Sens failed.";
  }

  // Infer Redistribution op list for stand alone and loss layout.
  RankList dev_list = g_device_manager->GetDeviceListInThisStage();
  if (tensor_redistribution.Init(stand_alone_layout, loss_layout, dev_list) == FAILED) {
    MS_LOG(EXCEPTION) << "Redistribution for Sens init failed.";
  }
  RedistributionOpListPtr sens_redistribution_list = tensor_redistribution.InferTensorRedistributionOperatorList();
  MS_EXCEPTION_IF_NULL(sens_redistribution_list);

  return sens_redistribution_list;
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
  if (IsParallelCareNode(cnode) && cnode->has_user_data<OperatorInfo>() &&
      !IsPrimitiveCNode(node, prim::kPrimReshape)) {
    auto layout_ptr = GetOutputLayoutFromCNode(cnode, 0);
    if (!layout_ptr) {
      MS_LOG(EXCEPTION) << "Failure:GetLayoutFromCNode failed";
    }
    return layout_ptr;
  }
  ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
  PrimitivePtr prim = prim_anf_node->value()->cast<PrimitivePtr>();
  if (prim->name() == prim::kTupleGetItem) {
    auto tuple_index = GetTupleGetItemIndex(cnode);
    auto layout_ptr = FindPrevParallelCareNodeLayout(cnode->input(1), LongToSize(tuple_index));
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
    if (!IsParallelCareNode(cnode) || !cnode->has_user_data<OperatorInfo>()) {
      continue;
    }
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);
    MS_EXCEPTION_IF_NULL(prim);
    OperatorInfoPtr operator_info = cnode->user_data<OperatorInfo>();
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

CNodePtr HandleDependLoss(const CNodePtr &cnode, size_t accum = 0) {
  // Handle return->depend->loss
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  accum += 1;
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->name() == DEPEND && accum < MAX_RECURSIVE_DEPTH) {
    auto depend_before = cnode->input(1)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(depend_before);
    return HandleDependLoss(depend_before, accum);
  }
  return cnode;
}

LossNodeInfo FindLossCNode(const FuncGraphPtr &func_graph) {
  LossNodeInfo loss_node_info;
  MS_EXCEPTION_IF_NULL(func_graph);
  CNodePtr return_node = func_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  if (return_node->size() < 2) {
    MS_LOG(EXCEPTION) << "Failure: " << return_node->ToString() << " size is smaller than 2";
  }
  AnfNodePtr pre_node = return_node->input(1);
  MS_EXCEPTION_IF_NULL(pre_node);
  if (IsPrimitiveCNode(pre_node, prim::kPrimDepend)) {
    pre_node = pre_node->cast<CNodePtr>()->input(1);
    MS_EXCEPTION_IF_NULL(pre_node);
  }

  auto pre_cnode = pre_node->cast<CNodePtr>();
  if (pre_cnode == nullptr || !IsValueNode<Primitive>(pre_cnode->input(0))) {
    return loss_node_info;
  }
  if (!IsValueNode<Primitive>(pre_cnode->input(0))) {
    MS_LOG(DEBUG) << "pre_cnode:" << pre_cnode->ToString();
    return loss_node_info;
  }
  auto prim = GetValueNode<PrimitivePtr>(pre_cnode->input(0));
  // return -> cast
  if (prim->name() == CAST && !pre_cnode->has_user_data<OperatorInfo>()) {
    pre_cnode = pre_cnode->input(1)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(pre_cnode);
  }
  pre_cnode = HandleDependLoss(pre_cnode);
  auto current_prim = GetValueNode<PrimitivePtr>(pre_cnode->input(0));

  // notice: the GetNext op has not input
  if (INVALID_LOSS_OPS.find(current_prim->name()) != INVALID_LOSS_OPS.end()) {
    MS_LOG(INFO) << "The loss is: " << current_prim->name();
    loss_node_info.loss_node = pre_cnode;
    return loss_node_info;
  }

  // size of common cnode is larger than 1
  if (pre_cnode->size() < 2) {
    MS_LOG(EXCEPTION) << pre_cnode->ToString() << " size( " << pre_cnode->inputs().size() << " ) is smaller than 2";
  }

  // return -> tuple_getitem -> loss
  if (current_prim->name() == prim::kTupleGetItem) {
    auto tuple_index = GetTupleGetItemIndex(pre_cnode);
    AnfNodePtr pre_pre_node = pre_cnode->input(1);
    MS_EXCEPTION_IF_NULL(pre_pre_node);

    auto pre_pre_cnode = pre_pre_node->cast<CNodePtr>();
    loss_node_info.has_tuple_getitem = true;
    loss_node_info.dout_index = tuple_index;
    loss_node_info.loss_node = pre_pre_cnode;
    return loss_node_info;
  }

  // return -> make_tuple
  if (current_prim->name() == MAKE_TUPLE) {
    MS_LOG(WARNING) << "The loss have make_tuple, it is not supported";
    return loss_node_info;
  }

  // return -> loss
  loss_node_info.loss_node = pre_cnode;
  MS_LOG(DEBUG) << "The loss name is " << current_prim->name();
  return loss_node_info;
}

TensorLayouts GetLossNodeGradOutputLayout(const LossNodeInfo &node_info) {
  TensorLayouts ret;
  auto loss_cnode = node_info.loss_node;
  MS_EXCEPTION_IF_NULL(loss_cnode);

  ValueNodePtr prim_anf_node = loss_cnode->input(0)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(prim_anf_node);
  PrimitivePtr prim = prim_anf_node->value()->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(prim);
  if (INVALID_LOSS_OPS.find(prim->name()) != INVALID_LOSS_OPS.end()) {
    MS_LOG(WARNING) << "The loss name is: " << prim->name() << ", do nothing for split sens now";
    return ret;
  }

  OperatorInfoPtr operator_info = loss_cnode->user_data<OperatorInfo>();
  MS_EXCEPTION_IF_NULL(operator_info);
  TensorInfo loss_grad_tensor_info;
  size_t op_output_size = operator_info->outputs_tensor_info().size();
  MS_LOG(INFO) << "The loss name is " << operator_info->name() << ", the has tuple item is  "
               << node_info.has_tuple_getitem << ", the output size is  " << op_output_size << ", the dout_index is  "
               << node_info.dout_index;

  if ((op_output_size == 0) || (op_output_size <= LongToSize(node_info.dout_index))) {
    MS_LOG(EXCEPTION) << "The index is  " << node_info.dout_index << ", but the size of outputs is  " << op_output_size;
  }

  if (!node_info.has_tuple_getitem && (op_output_size > 1)) {
    MS_LOG(EXCEPTION) << "Currently, it is not supported that the sens is a tuple.";
  }

  loss_grad_tensor_info = operator_info->outputs_tensor_info()[LongToSize(node_info.dout_index)];
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
      sens_tensor_param->set_user_data<TensorLayout>(std::make_shared<TensorLayout>(loss_grad_layout));
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
      sens_tensor_param->set_user_data<TensorLayout>(std::make_shared<TensorLayout>(loss_grad_layout));
      return;
    }
    if (sens_tensor_node->isa<CNode>()) {
      auto op_list_ptr = InferSensRedistribution(sens_tensor_node, loss_grad_layout);
      if (op_list_ptr == nullptr) {
        return;
      }
      auto sens_tensor_cnode = sens_tensor_node->cast<CNodePtr>();
      auto func_graph = grad_sens_node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      InsertRedistribution(op_list_ptr, grad_sens_node, func_graph, 1, sens_tensor_cnode);
      return;
    }
    MS_LOG(EXCEPTION) << "The type of sens node is not Tensor or Parameter or CNode, it is unsupported now.";
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

void HandleTileNode(const OperatorInfoPtr &distribute_operator, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->size() < 3 || !IsValueNode<Primitive>(cnode->input(0))) {
    return;
  }
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (prim->name() != TILE) {
    return;
  }

  TileInfoPtr tile = std::dynamic_pointer_cast<TileInfo>(distribute_operator);
  MS_EXCEPTION_IF_NULL(tile);
  tile->UpdateMultiples(cnode);
}

void HandleSpecialNode(const OperatorInfoPtr &distribute_operator, const CNodePtr &cnode) {
  HandleDropoutNode(distribute_operator, cnode);
  HandleTileNode(distribute_operator, cnode);
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
      auto manager = graph->manager();
      MS_EXCEPTION_IF_NULL(manager);
      auto graph_used = manager->func_graphs_used_total(graph);
      for (auto &sub_graph : graph_used) {
        graph_set.insert(sub_graph);
      }
    }
  }
  return graph_set;
}

void StepSplitSens(const std::pair<CNodePtr, LossNodeInfo> &sens_loss_pair) {
  CNodePtr sens_node = sens_loss_pair.first;
  auto loss_node = sens_loss_pair.second;
  auto loss_grad_layout = GetLossNodeGradOutputLayout(loss_node);
  if (!loss_grad_layout.empty()) {
    SplitSens(sens_node, loss_grad_layout[0]);
  }
}

// Sens node satisfies the following conditions: cnode(sens)-->cnode(tuple_getitem)-->cnode-->cnode(J)
std::vector<std::pair<CNodePtr, LossNodeInfo>> GetSensLossPairs(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  std::vector<std::pair<CNodePtr, LossNodeInfo>> sens_loss_pairs;
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
    if (!IsSomePrimitive(expect_tuple_getitem_cnode, prim::kTupleGetItem)) {
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
    auto loss_node_info = FindLossCNode(func_graph);
    if (loss_node_info.loss_node == nullptr) {
      MS_LOG(WARNING) << "Can not find the loss cnode";
      continue;
    }
    std::pair<CNodePtr, LossNodeInfo> sens_loss_pair = std::make_pair(sens_cnode, loss_node_info);
    sens_loss_pairs.push_back(sens_loss_pair);
  }
  return sens_loss_pairs;
}

bool IsLastStage() {
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto stage_num = g_device_manager->stage_num();
  auto stage_id = g_device_manager->stage_id();
  return ((stage_num - 1) == stage_id);
}

void ParallelCommunication(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes,
                           const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(manager);
  TensorRedistribution tensor_redistribution;

  std::vector<std::pair<CNodePtr, LossNodeInfo>> sens_loss_pairs = GetSensLossPairs(root);
  bool has_backward = !sens_loss_pairs.empty();
  // split sens must before inserting the operators.
  for (auto &pair : sens_loss_pairs) {
    // If the shape of grad-sens tensor is not [] or [1], use get tensor slice to handle it.
    // If the type of sens node is not Tensor, it is unsupported now, do nothing default.
    if (IsLastStage()) {
      StepSplitSens(pair);
    }
  }

  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      // the make_tuple is parallel care node, but it may have not operator info
      if (!IsParallelCareNode(cnode) || !cnode->has_user_data<OperatorInfo>()) {
        continue;
      }

      OperatorInfoPtr distribute_operator = GetDistributeOperator(cnode);
      MS_EXCEPTION_IF_NULL(distribute_operator);

      // insert forward ops
      if (!IsSomePrimitive(cnode, RECEIVE)) {
        InsertForwardOps(distribute_operator, cnode);
      }

      // insert redistribution ops
      StepRedistribution(cnode, distribute_operator, cnode, tensor_redistribution, cnode);

      // insert backward ops
      if (has_backward && !IsSomePrimitive(cnode, RECEIVE)) {
        BackwardCommunication(root, distribute_operator, cnode, sens_loss_pairs);
      }

      HandleSpecialNode(distribute_operator, cnode);
    } else if (IsValueNode<Tensor>(node) || IsValueNode<ValueList>(node) || IsValueNode<ValueTuple>(node)) {
      StepSplitTensor(node, manager);
    }
  }

  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      if (!IsParallelCareNode(cnode) || !cnode->has_user_data<OperatorInfo>() || IsSomePrimitive(cnode, RECEIVE)) {
        continue;
      }

      OperatorInfoPtr distribute_operator = GetDistributeOperator(cnode);
      MS_EXCEPTION_IF_NULL(distribute_operator);
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

bool IsCohesiveNode(const CNodePtr &cnode) {
  return IsPrimitiveCNode(cnode, prim::kPrimCast) || IsPrimitiveCNode(cnode, prim::kPrimLoad) ||
         IsPrimitiveCNode(cnode, prim::kPrimAllGather) || IsPrimitiveCNode(cnode, prim::kPrimMiniStepAllGather);
}

std::vector<std::pair<std::string, int64_t>> NodeParameterName(const CNodePtr &node, int64_t index) {
  std::vector<AnfNodePtr> node_inputs{node->inputs()};
  std::vector<std::pair<std::string, int64_t>> param_names;
  for (int64_t i = 0; i < UlongToLong(node_inputs.size()); ++i) {
    int64_t idx = index > i ? index : i;
    auto input = node_inputs[i];
    if (input->isa<Parameter>()) {
      auto input_parameter = input->cast<ParameterPtr>();
      if (input_parameter->has_default() && ParameterRequireGrad(input_parameter)) {
        param_names.push_back({input_parameter->name(), idx});
      }
    } else if (input->isa<CNode>()) {
      CNodePtr cnode = input->cast<CNodePtr>();
      if (!IsValueNode<Primitive>(cnode->input(0))) {
        continue;
      }
      if (IsCohesiveNode(cnode) && cnode->inputs().size() >= 1) {
        auto input_param_names = NodeParameterName(cnode, idx);
        param_names.insert(param_names.end(), input_param_names.begin(), input_param_names.end());
      }
    }
  }
  return param_names;
}

void CheckpointStrategy(const std::vector<AnfNodePtr> &all_nodes) {
  StrategyMap stra_map;
  TensorInfoMap tensor_info_map;
  ManualShapeMap manual_shape_map;
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    auto cnode = node->cast<CNodePtr>();
    if ((cnode == nullptr) || !IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    auto param_names = NodeParameterName(cnode);
    if (param_names.empty()) {
      continue;
    }
    string param_name = param_names[0].first;
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_EXCEPTION_IF_NULL(prim);
    OperatorInfoPtr operator_info = cnode->user_data<OperatorInfo>();
    if (operator_info) {
      if (operator_info->name().find(RESHAPEINFO) != std::string::npos) {
        continue;
      }
      std::vector<TensorInfo> input_tensor_info = operator_info->inputs_tensor_info();
      std::string stratey_key_name = prim->name() + "_" + param_name;
      stra_map[stratey_key_name] = operator_info->strategy();
      for (auto param_name_pair : param_names) {
        if (param_name_pair.second - 1 >= UlongToLong(input_tensor_info.size())) {
          continue;
        }
        tensor_info_map[param_name_pair.first] = input_tensor_info[param_name_pair.second - 1];
      }
      if (operator_info->name().find(EMBEDDING_LOOKUP) != std::string::npos ||
          operator_info->name().find(GATHERV2) != std::string::npos) {
        auto gatherv2_info = std::dynamic_pointer_cast<GatherPInfo>(operator_info);
        auto param_split_shapes = gatherv2_info->param_split_shapes();
        auto index_offsets = gatherv2_info->index_offsets();
        if (param_split_shapes.size() != index_offsets.size()) {
          MS_LOG(EXCEPTION) << "In manual split, the param_split_shapes and index_offsets length should be same.";
        }
        std::vector<std::pair<int64_t, int64_t>> manual_shape;
        for (int64_t i = 0; i < UlongToLong(param_split_shapes.size()); ++i) {
          manual_shape.push_back({param_split_shapes[i], index_offsets[i]});
        }
        manual_shape_map[param_name] = manual_shape;
      }
    }
  }

  if (StrategyCheckpoint::GetInstance().Save(stra_map, tensor_info_map, &manual_shape_map) != SUCCESS) {
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
  auto loss_cnode = FindLossCNode(graph).loss_node;
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

void InsertShapeOp(const CNodePtr &node, const AnfNodePtr &pre_node, const FuncGraphPtr &root) {
  // shape op doesn't have params and attrs.
  OperatorParams params;
  OperatorAttrs attrs;
  auto shape_value = GetValueNode(node->input(2))->cast<ValueSequeuePtr>();
  MS_EXCEPTION_IF_NULL(shape_value);
  auto shape = shape_value->value();
  if (shape.empty()) {
    return;
  }
  OperatorArgs args = std::make_pair(attrs, params);
  Operator op = std::make_pair(SHAPE_OP, args);
  InsertNode(op, node, 2, pre_node, root, "shape");
}

static AnfNodePtr FindGrad(const CNodePtr &cnode, size_t accum = 0) {
  accum += 1;
  if (accum > MAX_RECURSIVE_DEPTH) {
    return nullptr;
  }
  for (auto &node : cnode->inputs()) {
    if (!node->isa<CNode>()) {
      continue;
    }
    if (!IsPrimitiveCNode(node, prim::kPrimEnvGetItem)) {
      return FindGrad(node->cast<CNodePtr>(), accum);
    } else {
      return node;
    }
  }
  return nullptr;
}

void HandleRootReshapeAndSaveStrategy(const std::vector<AnfNodePtr> &all_nodes) {
  // If root graph has reshape op. Find the corresponding parameter.
  // Reshape's shape is the shape of the parameter.
  auto executor = pipeline::ExecutorPy::GetInstance();
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!IsValueNode<Primitive>(cnode->input(0)) || cnode == nullptr) {
      continue;
    }
    if (cnode->in_forward_flag()) {
      // Save strategy in executor
      OperatorInfoPtr op_info = cnode->user_data<OperatorInfo>();
      if (op_info) {
        auto stra_ptr = op_info->strategy();
        if (stra_ptr) {
          auto strategy = stra_ptr->GetInputDim();
          // fullname with scope should be found in step parallel end ir
          executor->SetCNodeStrategy(cnode->fullname_with_scope(), strategy);
        }
      }
      continue;
    }

    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim->name() != RESHAPE) {
      continue;
    }
    auto root = node->func_graph();
    auto grad_node = FindGrad(cnode);
    if (grad_node) {
      InsertShapeOp(cnode, grad_node, root);
    }
  }
}

void MarkForwardCNode(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  auto all_nodes = root->nodes();
  auto graph_set = FindForwardGraphByRootNodes(all_nodes);

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
  int64_t device_num = ParallelContext::GetInstance()->device_num();
  int64_t global_rank = ParallelContext::GetInstance()->global_rank();
  int32_t split_stage_num = ParallelContext::GetInstance()->pipeline_stage_split_num();
  std::string parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string backend = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  std::string world_group;
  std::string communication_backend;
  if (backend == kAscendDevice || backend == kDavinciDevice) {
    world_group = HCCL_WORLD_GROUP;
    communication_backend = HCCL_BACKEND;
  } else if (backend == kGPUDevice) {
    world_group = NCCL_WORLD_GROUP;
    communication_backend = NCCL_BACKEND;
  } else {
    MS_LOG(ERROR) << "Invalid communication backend: " << backend;
    return FAILED;
  }

  if (split_stage_num <= 0) {
    MS_LOG(ERROR) << "Invalid stage num " << split_stage_num << ", expected a positive stage number";
    return FAILED;
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

  if ((device_num <= 0) || (device_num > MAX_DEVICE_NUM)) {
    MS_LOG(ERROR) << "Invalid device num " << device_num;
    return FAILED;
  }

  // the device_num maybe get from communication interface
  if (device_num % split_stage_num != 0) {
    MS_LOG(ERROR) << "Device num " << device_num << "  can't be divided by stage num " << split_stage_num;
    return FAILED;
  }

  if ((global_rank < 0) || (global_rank >= device_num)) {
    MS_LOG(ERROR) << "Global rank " << global_rank << " is out of range, the device num is " << device_num;
    return FAILED;
  }

  std::vector<int64_t> stages;
  for (int i = 0; i < split_stage_num; i++) {
    stages.push_back(device_num / split_stage_num);
  }

  if ((split_stage_num > 1) && (parallel_mode != SEMI_AUTO_PARALLEL)) {
    MS_LOG(ERROR) << "To enable the pipeline parallel, please set the parallel mode to " << SEMI_AUTO_PARALLEL;
    return FAILED;
  }

  if (!InitDevice(device_num, global_rank, communication_backend, stages)) {
    MS_LOG(ERROR) << "Init device failed";
    return FAILED;
  }

  MS_LOG(INFO) << "The parallel context: dev num: " << device_num << ", global rank: " << global_rank
               << ", backend: " << backend << ", gradients_mean: " << ParallelContext::GetInstance()->gradients_mean()
               << ", gradient_fp32_sync: " << ParallelContext::GetInstance()->gradient_fp32_sync();

  return SUCCESS;
}

void HandleForwardMakeTupleAndMakeList(const std::vector<AnfNodePtr> &all_nodes) {
  for (auto &node : all_nodes) {
    if (!AnfNodeIsPrimitive(node, MAKE_TUPLE) && !AnfNodeIsPrimitive(node, MAKE_LIST)) {
      continue;
    }

    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!cnode->in_forward_flag()) {
      continue;
    }

    FuncGraphManagerPtr manager = cnode->func_graph()->manager();
    MS_EXCEPTION_IF_NULL(manager);
    std::string op_type = AnfNodeIsPrimitive(node, MAKE_TUPLE) ? MAKE_TUPLE : MAKE_LIST;

    auto make_tuple_list_user = manager->node_users()[cnode];
    if (make_tuple_list_user.size() != 1) {
      MS_LOG(EXCEPTION) << "Now the " << op_type << "'s user must be 1, but got " << make_tuple_list_user.size();
    }
    CNodePtr make_tuple_list_next_cnode = make_tuple_list_user.pop().first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple_list_next_cnode);

    std::string make_tuple__list_user_prim_name = GetPrimName(make_tuple_list_next_cnode);
    if (!IsParallelCareNode(make_tuple_list_next_cnode)) {
      MS_LOG(INFO) << "The " << op_type << "'s user is " << make_tuple__list_user_prim_name
                   << ", no need to set operator info";
      continue;
    }
    if (make_tuple_list_next_cnode->inputs().size() != 2) {
      MS_LOG(EXCEPTION) << "Now the " << op_type << "'s user only support 1 input, but got "
                        << make_tuple_list_next_cnode->inputs().size() - 1;
    }

    MS_LOG(INFO) << "Set the " << op_type << "'s operator info, and the op name is " << make_tuple__list_user_prim_name;
    OperatorInfoPtr op_info = GetDistributeOperator(make_tuple_list_next_cnode);
    MS_EXCEPTION_IF_NULL(op_info);
    cnode->set_user_data<OperatorInfo>(op_info);
  }
}

RefKeyPair CNodeWithRefKeys(const AnfNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<AnfNodePtr> refkeys;
  if (cnode->isa<CNode>()) {
    auto cnode_ptr = cnode->cast<CNodePtr>();
    auto inputs = cnode_ptr->inputs();
    for (auto &one_input : inputs) {
      if (IsValueNode<RefKey>(one_input)) {
        refkeys.push_back(one_input);
      }
    }
    if (refkeys.size() >= 1) {
      return std::make_pair(cnode, refkeys);
    }
  }
  return {nullptr, refkeys};
}

ParameterUsersInfo FindParameterNodeUsers(const AnfNodePtr &node, bool (*IsCareNode)(const CNodePtr &)) {
  // In this case, node is a Parameter
  ParameterUsersInfo parameter_user_info;
  MS_EXCEPTION_IF_NULL(node->func_graph());
  MS_EXCEPTION_IF_NULL(node->func_graph()->manager());
  auto candidate_set = node->func_graph()->manager()->node_users()[node];
  for (auto &candidate : candidate_set) {
    auto candidate_node = candidate.first;
    if (IsPrimitiveCNode(candidate_node, prim::kPrimLoad)) {
      if (candidate.second != 1) {
        continue;
      }
      auto load_node_users = node->func_graph()->manager()->node_users()[candidate_node];
      for (auto &node_user : load_node_users) {
        auto cnode = node_user.first->cast<CNodePtr>();
        if (cnode == nullptr || !cnode->has_user_data<OperatorInfo>() || IsSomePrimitive(cnode, RECEIVE)) {
          continue;
        }
        (void)parameter_user_info.second.second.insert(node_user);
      }
    } else {
      auto c = candidate_node->cast<CNodePtr>();
      if (c == nullptr || !c->has_user_data<OperatorInfo>() || IsSomePrimitive(c, RECEIVE)) {
        continue;
      }
      (void)parameter_user_info.second.second.insert(candidate);
    }
  }
  parameter_user_info.first = node->cast<ParameterPtr>()->name();
  parameter_user_info.second.first = node;
  return parameter_user_info;
}

ParameterUsersInfo FindRefKeyNodeUsers(const RefKeyPair &ref_key_pair, bool (*IsCareNode)(const CNodePtr &)) {
  // Dealing with the RefKey case
  ParameterUsersInfo parameter_user_info;
  auto refkeys = ref_key_pair.second;
  auto cnode = ref_key_pair.first;

  auto cnode_ptr = cnode->cast<CNodePtr>();
  if ((cnode_ptr == nullptr) || !IsValueNode<Primitive>(cnode_ptr->input(0)) || !IsCareNode(cnode_ptr)) {
    return parameter_user_info;
  }

  if (refkeys.size() > 1) {
    MS_LOG(EXCEPTION) << "CNode: " << cnode->fullname_with_scope() << "'s inputs have more than 1 RefKeys";
  }
  MS_EXCEPTION_IF_NULL(cnode->func_graph());
  auto cnode_func_graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(cnode->func_graph()->manager());

  // Find the RefKey being used
  auto candidate_set_by_refkey = cnode_func_graph->manager()->node_users()[refkeys[0]];
  for (auto &candidate : candidate_set_by_refkey) {
    auto candidate_node = candidate.first;
    auto c = candidate_node->cast<CNodePtr>();
    if ((c == nullptr) || !IsValueNode<Primitive>(c->input(0)) || !IsCareNode(c)) {
      continue;
    }
    parameter_user_info.second.second.add(candidate);
  }

  // Find the corresponding Parameter being used
  std::vector<AnfNodePtr> parameters = FindParameterByRefKeyNode(refkeys[0], cnode_func_graph);
  if (parameters.size() != 1) {
    MS_LOG(EXCEPTION) << "Find parameter by ref key node failed";
  }
  parameter_user_info.first = parameters[0]->cast<ParameterPtr>()->name();
  parameter_user_info.second.first = parameters[0];
  auto candidate_set_by_para = cnode_func_graph->manager()->node_users()[parameters[0]];
  for (auto &candidate : candidate_set_by_para) {
    auto candidate_node = candidate.first;
    auto c = candidate_node->cast<CNodePtr>();
    if ((c == nullptr) || !IsValueNode<Primitive>(c->input(0)) || !IsCareNode(c)) {
      continue;
    }
    (void)parameter_user_info.second.second.insert(candidate);
  }
  return parameter_user_info;
}

ParameterUsersInfo FindParameterUsers(const AnfNodePtr &node, bool (*IsCareNode)(const CNodePtr &)) {
  ParameterUsersInfo parameter_users_info;

  auto cnode_with_refkeys = CNodeWithRefKeys(node);
  if (cnode_with_refkeys.first != nullptr) {
    // the node is a ref key node
    return FindRefKeyNodeUsers(cnode_with_refkeys, IsCareNode);
  } else if (node->isa<Parameter>()) {
    // the node is a parameter node
    return FindParameterNodeUsers(node, IsCareNode);
  }

  return parameter_users_info;
}

Shape ParameterSliceShape(const std::pair<AnfNodePtr, int64_t> &param_info) {
  auto user_cnode = param_info.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(user_cnode);
  auto user_input_index = param_info.second;
  OperatorInfoPtr op_info = user_cnode->user_data<OperatorInfo>();
  MS_EXCEPTION_IF_NULL(op_info);

  size_t input_tensor_info_size = op_info->inputs_tensor_info().size();
  if (SizeToLong(input_tensor_info_size) <= user_input_index - 1) {
    MS_LOG(EXCEPTION) << op_info->name() << ": the size of inputs tensor info is " << input_tensor_info_size
                      << ", but the index is " << user_input_index - 1;
  }
  TensorInfo tensor_info = op_info->inputs_tensor_info()[user_input_index - 1];
  MS_LOG(DEBUG) << "The op name is " << op_info->name() << ", the parameter index is " << user_input_index - 1
                << ", the slice shape is " << ShapeToString(tensor_info.slice_shape()) << ", the origin shape is "
                << ShapeToString(tensor_info.shape());
  return tensor_info.slice_shape();
}

void CheckParameterSplit(const std::vector<AnfNodePtr> &all_nodes) {
  for (auto &node : all_nodes) {
    ParameterUsersInfo parameter_users_info = FindParameterUsers(node, IsParallelCareNode);
    auto users_set = parameter_users_info.second.second;
    if (users_set.size() <= 1) {
      continue;
    }

    auto parameter_name = parameter_users_info.first;
    MS_LOG(INFO) << "The parameter: " << parameter_name << " has " << users_set.size() << " users";
    auto first_user = users_set.pop();
    Shape first_user_slice_shape = ParameterSliceShape(first_user);

    for (auto &user : users_set) {
      Shape user_slice_shape = ParameterSliceShape(user);
      if (first_user_slice_shape != user_slice_shape) {
        MS_LOG(EXCEPTION) << "The parameter: " << parameter_name
                          << " has multiple users, but the split strategies are different";
      }
    }
  }
}

bool CreateGroupsByCkptFile(const std::string &file) {
  GroupInfoMap group_info_map;
  if (StrategyCheckpoint::GetInstance().LoadGroupInfo(file, &group_info_map) != SUCCESS) {
    return false;
  }

  if (CreateGroups(group_info_map) != SUCCESS) {
    return false;
  }
  MS_LOG(INFO) << "Create groups by checkpoint file success";
  return true;
}

bool IsUsedParameter(const FuncGraphPtr &graph, const AnfNodePtr &parameter, size_t accum = 0) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(parameter);
  accum += 1;
  if (accum > MAX_RECURSIVE_DEPTH) {
    return false;
  }
  auto manager = graph->manager();
  auto node_users = manager->node_users()[parameter];
  if (node_users.empty()) {
    return false;
  }
  for (auto node_user : node_users) {
    auto use_node = node_user.first->cast<CNodePtr>();
    if (IsValueNode<FuncGraph>(use_node->input(0))) {
      auto graph_sub = GetValueNode<FuncGraphPtr>(use_node->input(0));
      auto parameters = graph_sub->parameters();
      auto parameter_sub = parameters[node_user.second - 1];
      return IsUsedParameter(graph_sub, parameter_sub);
    }
    if (use_node->input(0)->isa<CNode>()) {
      auto cnode = use_node->input(0)->cast<CNodePtr>();
      if (!IsSomePrimitive(cnode, J) || !IsValueNode<FuncGraph>(cnode->input(1))) {
        return true;
      }
      auto graph_sub = GetValueNode<FuncGraphPtr>(cnode->input(1));
      auto parameters = graph_sub->parameters();
      auto parameter_sub = parameters[node_user.second - 1];
      return IsUsedParameter(graph_sub, parameter_sub);
    }
    return true;
  }
  return true;
}

static void HandleNoUsedParameter(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  bool full_batch = ParallelContext::GetInstance()->full_batch();
  if (full_batch) {
    return;
  }

  // in grad accumulation mode, if use dynamic lr, it has some parameters in optimizer which no used for first graph,
  // but used for second graph(such as global_step), so can not change their shapes
  int64_t grad_accumulation_step = ParallelContext::GetInstance()->grad_accumulation_step();
  if (grad_accumulation_step > 1) {
    MS_LOG(INFO) << "In grad accumulation mode, do not handle no used parameters";
    return;
  }

  auto dev_num = g_device_manager->stage_device_num();
  auto parameters = root->parameters();
  for (auto &parameter : parameters) {
    if (IsUsedParameter(root, parameter)) {
      continue;
    }
    auto parameter_shape = GetNodeShape(parameter);
    if (parameter_shape.empty()) {
      continue;
    }
    Shape slice_shape = parameter_shape[0];
    if (slice_shape.empty()) {
      continue;
    }
    slice_shape[0] = slice_shape[0] / dev_num;
    auto slice_shape_ptr = std::make_shared<abstract::Shape>(slice_shape);
    auto abstract = parameter->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    auto abstract_cloned = abstract->Clone();
    MS_EXCEPTION_IF_NULL(abstract_cloned);
    abstract_cloned->set_shape(slice_shape_ptr);
    parameter->set_abstract(abstract_cloned);
  }
}

static bool IsFullySplitParameter(const ParameterPtr &param_ptr) {
  auto tensor_layout = param_ptr->user_data<parallel::TensorLayout>();
  if (tensor_layout == nullptr) {
    return false;
  }

  auto dev_mat_shape = tensor_layout->device_arrangement().array();
  auto tensor_map = tensor_layout->tensor_map().array();
  int64_t rank = g_device_manager->global_rank();
  RankList rank_list = g_device_manager->GetDeviceListInThisStage();
  DeviceMatrix dev_matrix(rank, rank_list, dev_mat_shape);
  RankList group_devices;
  if (dev_matrix.GetDevicesByTensorMap(tensor_map, &group_devices) != SUCCESS) {
    MS_LOG(WARNING) << "Get devices by tensor map failed, invalid tensor layout";
    return false;
  }

  if (group_devices.size() == 1) {
    MS_LOG(INFO) << "The parameter: " << param_ptr->name() << " is fully split";
    return true;
  }
  return false;
}

static AnfNodePtr FindGradAccuParameter(const std::vector<AnfNodePtr> &parameters, const std::string &name) {
  for (auto &parameter : parameters) {
    auto param_ptr = parameter->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    if (param_ptr->name() == name) {
      continue;
    }
    if (param_ptr->name().find(name) != std::string::npos && param_ptr->name().find("accu_grad") != std::string::npos) {
      return parameter;
    }
  }
  return nullptr;
}

static void InsertFullySplitParamGradAccu(const std::pair<AnfNodePtr, int> &node_user,
                                          const FuncGraphManagerPtr &manager, const AnfNodePtr &accu_parameter) {
  auto cnode = node_user.first->cast<CNodePtr>();
  auto prim = GetCNodePrimitive(cnode);
  if (prim == nullptr) {
    MS_LOG(WARNING) << cnode->DebugString() << " can not insert fully split param grad accumulation node";
    return;
  }
  OperatorAttrs attrs;
  auto py_instance = CreatOpInstance(attrs, "_VirtualAdd", "grad_accu");
  auto value_node = NewValueNode(py_instance);
  std::vector<AnfNodePtr> virtual_node_input = {value_node, cnode->input(node_user.second), accu_parameter};
  auto graph = cnode->func_graph();
  auto virtual_node = graph->NewCNode(virtual_node_input);
  manager->SetEdge(cnode, node_user.second, virtual_node);
}

static void HandleFullySplitParameters(const FuncGraphPtr &root) {
  int64_t grad_accumulation_step = ParallelContext::GetInstance()->grad_accumulation_step();
  if ((grad_accumulation_step <= 1) || root->has_flag(ACCUMULATION)) {
    return;
  }

  auto parameters = root->parameters();
  auto node_users_map = root->manager()->node_users();
  for (auto &parameter : parameters) {
    auto param_ptr = parameter->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);

    if (!IsFullySplitParameter(param_ptr)) {
      continue;
    }

    auto accu_parameter = FindGradAccuParameter(parameters, param_ptr->name());
    if (!accu_parameter) {
      continue;  // some parameters no need to handle, such as itself or lr
    }

    auto node_users = node_users_map[parameter];
    for (auto &user : node_users) {
      auto node = user.first;
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (!cnode->in_forward_flag()) {
        continue;
      }
      InsertFullySplitParamGradAccu(user, root->manager(), accu_parameter);
      MS_LOG(INFO) << "Insert full split assign add node for " << param_ptr->name();
      break;  // only need to insert once, if the parameter has many users
    }
  }
}

bool StepParallel(const FuncGraphPtr &root, const opt::OptimizerPtr &optimizer) {
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
  if (ps::PSContext::instance()->is_server() || ps::PSContext::instance()->is_scheduler()) {
    return false;
  }
#endif
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
    auto pipeline_stages = ParallelContext::GetInstance()->pipeline_stage_split_num();
    if (pipeline_stages <= 1 && ParallelInit() != SUCCESS) {
      MS_LOG(EXCEPTION) << "Parallel init failed";
    }

    // mark the forward cnodes, parallel only care these nodes
    MarkForwardCNode(root);

    if (FindCommunicationOp(all_nodes)) {
      MS_LOG(EXCEPTION) << "The graph contain communication op";
    }

    // extract shape and strategy, set operator_info
    ExtractInformation(all_nodes, root->has_flag(TRAINING));
    ReshapeInit(all_nodes);
  }

  HandleRootReshapeAndSaveStrategy(all_nodes);

  HandleForwardMakeTupleAndMakeList(all_nodes);

  // if the input or parameter has multiple users, check whether its split strategies are consistent.
  CheckParameterSplit(all_nodes);

  HandleSymbolicKeyInstance(root, all_nodes);

  // cover Parallel shape
  CoverSliceShape(root);

  // handle input is not used
  HandleNoUsedParameter(root);

  // set the shape for optimizer's clone tensor
  SetClonedTensorShapeForOptimizer(root);

  // save strategy as checkpoint for multi-train
  if (StrategyCheckpoint::GetInstance().SaveCheckPointOn()) {
    CheckpointStrategy(all_nodes);
  }

  // ForwardCommunication BackwardCommunication TensorRedistribution
  ParallelCommunication(root, all_nodes, manager);

  auto group_info = g_device_manager->group_info();
  if (StrategyCheckpoint::GetInstance().group_info_save_on() &&
      StrategyCheckpoint::GetInstance().SaveGroupInfo(group_info) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Save group info failed";
  }

  // handle full split parammeters in grad accumulation, do not contain optimizer-sharding's parameter
  HandleFullySplitParameters(root);

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
