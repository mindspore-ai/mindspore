/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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

#include <cinttypes>
#include <algorithm>
#include <chrono>
#include <map>
#include <unordered_map>
#include <memory>
#include <set>
#include <string>
#include <queue>
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/structure_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "utils/hash_map.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/ops_info/gather_info.h"
#include "frontend/parallel/ops_info/reshape_info.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "frontend/parallel/tensor_layout/prime_generator.h"
#include "frontend/parallel/graph_util/pipeline_split_utils.h"
#include "frontend/parallel/graph_util/fold_pipeline_split_utils.h"
#include "frontend/parallel/pipeline_transformer/pipeline_interleave.h"
#include "frontend/parallel/graph_util/grad_accumulation_utils.h"
#include "frontend/parallel/node_check.h"
#include "frontend/parallel/silent_check/silent_check.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/ops_info/matmul_info.h"
#include "frontend/parallel/dynamic_shape/dynamic_shape.h"
#include "frontend/parallel/tensor_layout/tensor_transform.h"
#include "ir/param_info.h"
#include "ir/tensor.h"
#include "utils/trace_base.h"
#include "include/common/utils/comm_manager.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"
#include "mindspore/core/utils/parallel_node_check.h"
#include "frontend/parallel/parallel_optimizer/opt_param_mgr.h"
#include "mindspore/core/ops/conv_pool_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/ops_func_impl/flash_attention_score.h"

#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/backend/distributed/ps/util.h"
#include "include/backend/distributed/ps/ps_context.h"
#endif

using mindspore::tensor::Tensor;

namespace mindspore {
namespace parallel {
static const std::set<std::string> INVALID_LOSS_OPS = {GET_NEXT, VIRTUALLOSS, LOAD, UPDATESTATE};
static const std::set<std::string> NO_INPUT_TENSOR_OPS = {UNIFORM_REAL, STANDARD_NORMAL};
const uint32_t MAX_BFS_DEPTH = 7;
const char kSilentCheckEnvEnable[] = "1";

static void SetAllReduceRecomputeFlag(const std::vector<AnfNodePtr> &new_node_input, const CNodePtr &node) {
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
    (void)prim->SetAttrs(attrs);
    MS_LOG(INFO) << "Do not recompute the forward communication operator of " << prim_node->ToString();
  }
}

// Replace pre_node with pre_node->op
static CNodePtr ReplaceNode(const Operator &op, const AnfNodePtr &pre_node, const FuncGraphPtr &func_graph,
                            const std::string &instance_name, const std::string &param_name = "",
                            const FuncGraphPtr &root = nullptr) {
  // insert new node before the node
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  ScopePtr scope = pre_node->scope();
  MS_EXCEPTION_IF_NULL(scope);
  std::vector<AnfNodePtr> node_input;
  if (root && !param_name.empty()) {
    node_input = CreateMirrorInput(root, op, pre_node, instance_name, param_name);
  } else {
    node_input = CreateInput(op, pre_node, instance_name);
  }
  CNodePtr new_node = func_graph->NewCNode(node_input);
  MS_EXCEPTION_IF_NULL(new_node);
  if (instance_name.find(SPLIT_SENS) == std::string::npos) {
    new_node->set_in_forward_flag(true);  // mark forward flag
  }
  auto new_node_prim = GetValueNode<PrimitivePtr>(node_input[0]);
  new_node_prim->set_instance_name(instance_name);
  new_node_prim->set_attr("keep_value_node_input", MakeValue(true));
  if (instance_name.find(NOT_RECOMPUTE) != std::string::npos) {
    new_node_prim->set_attr("recompute", MakeValue(false));
  } else if (instance_name.find(RECOMPUTE) != std::string::npos) {
    new_node_prim->set_attr("recompute", MakeValue(true));
  }
  new_node->set_scope(scope);
  node_input[0]->set_scope(scope);
  (void)manager->Replace(pre_node, new_node);
  MS_LOG(INFO) << "Insert " << instance_name << " success";
  return new_node;
}

void ForwardCommunicationForMultiOut(OperatorVector forward_op, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  // step1:get graph manager distribute_operator
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto uses_set = manager->node_users()[node];
  // For GMM, its out always be tuplegetitem, so we need to find the real user of GMM
  std::vector<CNodePtr> node_to_insert = {};
  for (auto &uses_pair : uses_set) {
    auto uses_cnode = uses_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(uses_cnode);
    if (!IsValueNode<Primitive>(uses_cnode->input(0))) {
      break;
    }
    PrimitivePtr value_node_prim = GetValueNode<PrimitivePtr>(uses_cnode->input(0));
    MS_EXCEPTION_IF_NULL(value_node_prim);
    if (value_node_prim->name() == prim::kPrimTupleGetItem->name()) {
      node_to_insert.push_back(uses_cnode);
    }
  }
  if (node_to_insert.empty()) {
    MS_LOG(ERROR) << "The output of " << node->DebugString()
                  << "does not have a tuplegetitem node. Forward communication can not be inserted, the correctness of "
                     "current op can not be ensured.";
    return;
  }
  std::reverse(forward_op.begin(), forward_op.end());

  // step2:traverse op_list and insert node
  for (size_t index = 0; index < forward_op.size(); ++index) {
    std::string instance_name_base = FORWARD_OP;
    std::string instance_name = instance_name_base + "_" + CreateInstanceName(node, index);
    std::vector<AnfNodePtr> forward_input = CreateInput(forward_op[index], node_to_insert[index], instance_name);
    SetAllReduceRecomputeFlag(forward_input, node_to_insert[index]);
    CNodePtr forward_node = func_graph->NewCNode(forward_input);  // using NewCNode to create anfnode
    MS_EXCEPTION_IF_NULL(forward_node);
    ScopePtr scope = node->scope();
    MS_EXCEPTION_IF_NULL(scope);
    forward_node->set_scope(scope);
    forward_node->set_in_forward_flag(true);
    forward_node->AddPrimalAttr(kPrimalAttrForwardCommNodeUniqueId, MakeValue<std::string>(forward_node->UniqueId()));
    if (node_to_insert[index]->HasPrimalAttr(MICRO)) {
      forward_node->AddPrimalAttr(MICRO, node_to_insert[index]->GetPrimalAttr(MICRO));
    }
    forward_input[0]->set_scope(scope);
    (void)manager->Replace(node_to_insert[index], forward_node);  // using Replace function to insert node
  }
}

void ForwardCommunication(OperatorVector forward_op, const CNodePtr &node) {
  if (dyn_cast<abstract::SequenceShape>(node->Shape()) != nullptr) {
    // For Ops like GMM has multiple output
    MS_LOG(INFO) << "The input node " << node->DebugString()
                 << " has multiple output, enter ForwardCommunicationForMultiOut";
    ForwardCommunicationForMultiOut(forward_op, node);
    return;
  }
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
    if (value_node_prim->name() == prim::kPrimTupleGetItem->name()) {
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
    forward_node->AddPrimalAttr(kPrimalAttrForwardCommNodeUniqueId, MakeValue<std::string>(forward_node->UniqueId()));
    if (node_to_insert->HasPrimalAttr(MICRO)) {
      forward_node->AddPrimalAttr(MICRO, node_to_insert->GetPrimalAttr(MICRO));
    }
    forward_input[0]->set_scope(scope);
    (void)manager->Replace(node_to_insert, forward_node);  // using Replace function to insert node
  }
}

static CNodePtr InsertMakeTuple(const AnfNodePtr &prev, uint64_t num, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(prev);
  MS_EXCEPTION_IF_NULL(func_graph);
  ScopeGuard scope_guard(prev->scope());
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

static void InsertRedistribution(const RedistributionOpListPtr &redistribution_oplist_ptr, const CNodePtr &node,
                                 const FuncGraphPtr &func_graph, int64_t pos, const CNodePtr &pre_node,
                                 const TensorRedistributionPtr &tensor_redistribution) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(pre_node);
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if ((redistribution_oplist_ptr->first).size() != (redistribution_oplist_ptr->second).size()) {
    MS_LOG(EXCEPTION) << "size of OperatorVector and OutPutInfoVector must be the same!";
  }

  for (size_t index = 0; index < (redistribution_oplist_ptr->first).size(); ++index) {
    if (pos >= SizeToLong(node->size())) {
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
    auto prim_out = GetCNodePrimitive(node);
    auto prim_in = GetCNodePrimitive(pre_node);
    if (prim_out != nullptr && prim_in != nullptr) {
      auto prim_out_attr = prim_out->attrs();
      auto prim_in_attr = prim_in->attrs();
      std::string recompute_str = "";
      if (prim_out_attr.find(RECOMPUTE_COMM_OP) != prim_out_attr.end()) {
        recompute_str = GetValue<bool>(prim_out_attr[RECOMPUTE_COMM_OP]) ? RECOMPUTE : NOT_RECOMPUTE;
      }
      if (recompute_str.empty() && prim_in_attr.find(RECOMPUTE_COMM_OP) != prim_in_attr.end()) {
        recompute_str = GetValue<bool>(prim_in_attr[RECOMPUTE_COMM_OP]) ? RECOMPUTE : NOT_RECOMPUTE;
      }
      instance_name = instance_name + "_" + recompute_str;
    }
    InsertNode(op, node, LongToSize(pos), target_node, func_graph, instance_name, "", nullptr, tensor_redistribution);
    if ((redistribution_oplist_ptr->second)[index].first) {
      target_node = node->input(LongToSize(pos));
      MS_EXCEPTION_IF_NULL(target_node);
      (void)InsertMakeTuple(target_node, (redistribution_oplist_ptr->second)[index].second, func_graph);
    }
  }
}

static void InsertGetTensorSliceOp(const Operator &op, const CNodePtr &node, const FuncGraphPtr &func_graph,
                                   int64_t pos, const std::string &instance_name) {
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "InsertGetTensorSliceOp: the graph is null, the instance name is " << instance_name;
  }

  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (pos >= SizeToLong(node->size())) {
    MS_LOG(EXCEPTION) << "InsertGetTensorSliceOp: pos can't be larger than node's inputs'size, the instance name is "
                      << instance_name;
  }
  // Create new node
  AnfNodePtr pre_node = node->input(LongToSize(pos));
  MS_EXCEPTION_IF_NULL(pre_node);
  InsertNode(op, node, LongToSize(pos), pre_node, func_graph, instance_name);
}

TensorLayout GetTensorInLayoutForNewShape(const AnfNodePtr &pre_node, std::vector<int> get_item_index) {
  TensorLayout tensorinfo_in_layout;
  auto pre_cnode = pre_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(pre_cnode);
  auto distribute_operator = GetDistributeOperator(pre_cnode);
  MS_EXCEPTION_IF_NULL(distribute_operator);
  TensorInfoBasePtr tensorinfo_in;
  auto tensor_info_pos = get_item_index.front();
  get_item_index.erase(get_item_index.begin());
  if (tensor_info_pos != -1) {
    if (tensor_info_pos >= SizeToInt(distribute_operator->outputs_tensor_info_new().size())) {
      MS_LOG(EXCEPTION) << "The index out of range. Node: " << pre_node->DebugString() << " index: " << tensor_info_pos
                        << " outputs_tensor_info's size: " << distribute_operator->outputs_tensor_info().size();
    }
    tensorinfo_in = distribute_operator->outputs_tensor_info_new()[IntToSize(tensor_info_pos)];
  } else {
    tensorinfo_in = distribute_operator->outputs_tensor_info_new()[0];
  }
  for (const auto &index : get_item_index) {
    tensorinfo_in = tensorinfo_in->GetElement(IntToLong(index));
  }
  tensorinfo_in_layout = tensorinfo_in->GetValue().tensor_layout();
  return tensorinfo_in_layout;
}

TensorLayout GetTensorInLayout(const AnfNodePtr &pre_node, std::vector<int> get_item_index) {
  TensorLayout tensorinfo_in_layout;
  auto pre_cnode = pre_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(pre_cnode);
  auto distribute_operator = GetDistributeOperator(pre_cnode);
  if (!distribute_operator->outputs_tensor_info_new().empty()) {
    return GetTensorInLayoutForNewShape(pre_node, get_item_index);
  }
  MS_EXCEPTION_IF_NULL(distribute_operator);
  if (get_item_index.size() != 1) {
    // If does not have outputes_tensor_info_new, the outputs only have one tensor info
    // thus the get item index must only have one value
    MS_LOG(EXCEPTION) << "The get_item_index size is not 1, the size is " << get_item_index.size();
  }
  if (get_item_index[get_item_index.size() - 1] != -1) {
    if (get_item_index[get_item_index.size() - 1] >= SizeToInt(distribute_operator->outputs_tensor_info().size())) {
      MS_LOG(EXCEPTION) << "The index out of range. Node: " << pre_node->DebugString() << " index: " << get_item_index
                        << " outputs_tensor_info's size: " << distribute_operator->outputs_tensor_info().size();
    }
    auto tensorinfo_in =
      distribute_operator->outputs_tensor_info()[IntToSize(get_item_index[get_item_index.size() - 1])];
    tensorinfo_in_layout = tensorinfo_in.tensor_layout();
  } else {
    if (distribute_operator->outputs_tensor_info().empty()) {
      MS_LOG(EXCEPTION) << "The outputs tensor info is empty. Node:" << pre_node->DebugString();
    }
    auto tensorinfo_in = distribute_operator->outputs_tensor_info()[0];
    tensorinfo_in_layout = tensorinfo_in.tensor_layout();
  }
  return tensorinfo_in_layout;
}

Status ObtainOutputTensorLayout(const OperatorInfoPtr &next_distribute_operator,
                                const std::pair<AnfNodePtr, std::vector<int>> &node_pair, const CNodePtr &next_cnode,
                                const bool &using_func_param_op_info, TensorLayout *tensorlayout_out) {
  bool next_dist_op_has_tuple = !next_distribute_operator->inputs_tensor_info_new().empty();
  if (next_dist_op_has_tuple) {
    auto next_inputs_tensor_info = using_func_param_op_info ? next_distribute_operator->outputs_tensor_info_new()
                                                            : next_distribute_operator->inputs_tensor_info_new();
    auto it = std::find_if(node_pair.second.begin(), node_pair.second.end(), [&](const auto &input_idx) {
      return LongToSize(input_idx - 1) >= next_inputs_tensor_info.size();
    });
    if (it != node_pair.second.end()) {
      MS_LOG(INFO) << "The index is out of range, the index is " << (*it - 1) << ", the vector size is "
                   << next_inputs_tensor_info.size() << ", next node is " << next_cnode->DebugString();
      return FAILED;
    }
    auto tensorinfo_out_ptr = next_inputs_tensor_info[LongToSize(node_pair.second[0] - 1)];
    if (tensorinfo_out_ptr->is_list()) {
      for (size_t i = 1; i < node_pair.second.size(); ++i) {
        tensorinfo_out_ptr = tensorinfo_out_ptr->GetElement(LongToSize(node_pair.second[i] - 1));
      }
    }
    TensorInfo tensorinfo_out = tensorinfo_out_ptr->GetValue();
    *tensorlayout_out = tensorinfo_out.tensor_layout();
    return SUCCESS;
  }
  auto next_inputs_tensor_info = using_func_param_op_info ? next_distribute_operator->outputs_tensor_info()
                                                          : next_distribute_operator->inputs_tensor_info();
  size_t out_layout_index = LongToSize(node_pair.second[node_pair.second.size() - 1] - 1);
  if (out_layout_index >= next_inputs_tensor_info.size()) {
    MS_LOG(INFO) << "The index is out of range, the index is " << out_layout_index << ", the vector size is "
                 << next_inputs_tensor_info.size() << ", next node is " << next_cnode->DebugString();
    return FAILED;
  }
  TensorInfo tensorinfo_out = next_inputs_tensor_info[out_layout_index];
  *tensorlayout_out = tensorinfo_out.tensor_layout();
  return SUCCESS;
}

void InsertRedistributionForMicroInterleaved(const TensorRedistributionPtr &tensor_redistribution,
                                             const std::pair<AnfNodePtr, int64_t> &node_pair,
                                             const FuncGraphPtr &func_graph, const CNodePtr &attr_cnode,
                                             const CNodePtr &real_pre_node) {
  auto redistribution_oplist_ptr_vector = tensor_redistribution->InferTensorRedistributionOperatorVirtualGraphs();
  auto next_cnode = node_pair.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(next_cnode);
  auto next_cnode_index = node_pair.second;
  // create VirtualConverterBeginNode
  MS_EXCEPTION_IF_NULL(real_pre_node);
  auto virtual_converter_begin =
    CreateVirtualConverterBeginNode(real_pre_node, redistribution_oplist_ptr_vector.size());
  std::vector<CNodePtr> tuple_get_item_vector;
  for (size_t i = 0; i < redistribution_oplist_ptr_vector.size(); ++i) {
    if (redistribution_oplist_ptr_vector[i]->first.empty()) {
      return;
    }
    // create tuple_get_item
    std::vector<AnfNodePtr> tuple_get_item_inputs{NewValueNode(prim::kPrimTupleGetItem), virtual_converter_begin,
                                                  CreatInt64Imm(UlongToLong(i))};
    auto tuple_get_item_cnode = func_graph->NewCNode(tuple_get_item_inputs);
    tuple_get_item_vector.push_back(tuple_get_item_cnode);
  }
  // create VirtualConverterEndNode
  auto virtual_converter_end = CreateVirtualConverterEndNode(func_graph, tuple_get_item_vector);
  auto manager = func_graph->manager();
  (void)manager->SetEdge(next_cnode, next_cnode_index, virtual_converter_end);
  // add recompute_comm_op attrs
  auto prim_out = GetCNodePrimitive(next_cnode);
  if (prim_out != nullptr && prim_out->HasAttr(RECOMPUTE_COMM_OP)) {
    auto out_recompute_comm_op_attr = prim_out->GetAttr(RECOMPUTE_COMM_OP);
    auto virtual_converter_end_prim = GetCNodePrimitive(virtual_converter_end);
    virtual_converter_end_prim->AddAttr(RECOMPUTE_COMM_OP, out_recompute_comm_op_attr);
  }
  std::vector<std::vector<std::vector<int64_t>>> ag_group_ranks_vectors;

  for (size_t i = 0; i < redistribution_oplist_ptr_vector.size(); ++i) {
    auto redistribution_oplist_ptr = redistribution_oplist_ptr_vector[i];
    if (!tensor_redistribution->IsAssembledStaticShape()) {
      redistribution_oplist_ptr = TensorTransform::GetInstance()->OptimizeTensorRedistributionOperatorList(
        redistribution_oplist_ptr, tensor_redistribution->input_shape());
    }
    // Get allgather group_ranks attr in redistribution_oplist_ptr
    std::vector<std::vector<int64_t>> ag_group_ranks_vector;
    for (size_t findex = 0; findex < (redistribution_oplist_ptr->first).size(); ++findex) {
      // Create instance_name
      auto index = (redistribution_oplist_ptr->first).size() - 1 - findex;
      auto op = (redistribution_oplist_ptr->first)[index];
      std::string op_name = (redistribution_oplist_ptr->first)[index].first;
      if (op_name == ALL_GATHER) {
        auto group_ranks_attr = (redistribution_oplist_ptr->first)[index].second.first[1].second;
        auto group_ranks = GetValue<std::vector<int64_t>>(group_ranks_attr);
        ag_group_ranks_vector.push_back(group_ranks);
      }
    }
    ag_group_ranks_vectors.push_back(ag_group_ranks_vector);
    InsertRedistribution(redistribution_oplist_ptr, virtual_converter_end, func_graph, i + 1, attr_cnode,
                         tensor_redistribution);
  }
  ConvertInterleaveAllGatherToConcat(func_graph, virtual_converter_end, ag_group_ranks_vectors);
}

static void Redistribution(const std::pair<AnfNodePtr, std::vector<int>> &node_pair, const AnfNodePtr &pre_node,
                           const std::vector<int> &get_item_index) {
  MS_LOG(DEBUG) << "Do Redistribution for " << node_pair.first->fullname_with_scope();
  auto next_cnode = node_pair.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(next_cnode);
  auto func_graph = next_cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto pre_cnode = pre_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(pre_cnode);
  auto distribute_operator = GetDistributeOperator(pre_cnode);
  MS_EXCEPTION_IF_NULL(distribute_operator);
  auto dev_list = distribute_operator->stage_device_list();
  OperatorInfoPtr next_distribute_operator;
  bool using_func_param_op_info = false;
  if (IsValueNode<FuncGraph>(next_cnode->input(0))) {
    auto fg = GetValueNode<FuncGraphPtr>(next_cnode->input(0));
    auto fg_parameters = fg->parameters();
    auto param = fg_parameters[IntToSize(node_pair.second[node_pair.second.size() - 1] - 1)];
    if (param->has_user_data<OperatorInfo>()) {
      MS_LOG(INFO) << "Func call node:" << next_cnode->DebugString() << " has operator info.";
      next_distribute_operator = param->user_data<OperatorInfo>();
      using_func_param_op_info = true;
    } else {
      next_distribute_operator = GetDistributeOperator(next_cnode);
    }
  } else {
    next_distribute_operator = GetDistributeOperator(next_cnode);
  }
  MS_LOG(DEBUG) << "Redistribution for pre_node: " << pre_cnode->DebugString()
                << " next_node: " << next_cnode->DebugString();
  MS_EXCEPTION_IF_NULL(next_distribute_operator);

  auto tensor_redistribution = next_distribute_operator->CreateTensorRedistribution();
  tensor_redistribution->SetPreAndNextCNode(pre_cnode, next_cnode);
  MS_LOG(DEBUG) << "Redistribution for pre_node: " << pre_cnode->DebugString()
                << "next_node: " << next_cnode->DebugString();

  // extract tensor layout in and out
  if (distribute_operator->outputs_tensor_info().empty() && distribute_operator->outputs_tensor_info_new().empty()) {
    MS_LOG(WARNING) << "pre_node's tensorinfo_in is empty, operator name is " << distribute_operator->name();
    return;
  }
  TensorLayout tensorlayout_out;
  auto status = ObtainOutputTensorLayout(next_distribute_operator, node_pair, next_cnode, using_func_param_op_info,
                                         &tensorlayout_out);
  if (status != SUCCESS) {
    return;
  }
  TensorLayout tensorlayout_in = GetTensorInLayout(pre_node, get_item_index);
  if (IsPrimitiveCNode(pre_node, prim::kPrimReceive)) {
    tensorlayout_in = *(pre_node->user_data<TensorLayout>());
  }

  if (tensor_redistribution->Init(tensorlayout_in, tensorlayout_out, dev_list) == FAILED) {
    MS_LOG(ERROR) << "Redistribution: pre_node " << pre_cnode->DebugString() << " next_node "
                  << next_cnode->DebugString();
    DumpGraph(func_graph, "redistribution_error");
    MS_LOG(EXCEPTION) << "Failure:tensor_redistribution init failed";
  }
  if (tensorlayout_in.GetVirtualRank().size() > 1 || tensorlayout_out.GetVirtualRank().size() > 1) {
    auto real_pre_node = next_cnode->input(node_pair.second[node_pair.second.size() - 1])->cast<CNodePtr>();
    InsertRedistributionForMicroInterleaved(tensor_redistribution,
                                            {node_pair.first, node_pair.second[node_pair.second.size() - 1]},
                                            func_graph, pre_cnode, real_pre_node);
    return;
  }
  RedistributionOpListPtr redistribution_oplist_ptr = tensor_redistribution->InferTensorRedistributionOperatorList();
  if (redistribution_oplist_ptr == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Infer tensor redistribution failed.";
  }
  if (!tensor_redistribution->IsAssembledStaticShape()) {
    redistribution_oplist_ptr = TensorTransform::GetInstance()->OptimizeTensorRedistributionOperatorList(
      redistribution_oplist_ptr, tensor_redistribution->input_shape());
  }

  if (redistribution_oplist_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:InferTensorRedistribution failed";
  }
  MS_LOG(DEBUG) << "Redistribution size " << redistribution_oplist_ptr->first.size();
  if (!redistribution_oplist_ptr->first.empty()) {
    // the last one is the pos of node in maketuple
    tensor_redistribution->CreateAssembledDynamicMapping(next_cnode, pre_cnode, func_graph,
                                                         node_pair.second[node_pair.second.size() - 1]);
    // insert node before next node
    InsertRedistribution(redistribution_oplist_ptr, next_cnode, func_graph,
                         node_pair.second[node_pair.second.size() - 1], pre_cnode, tensor_redistribution);
  }
  // Rollback to dynamic shape.
  if (tensor_redistribution->IsAssembledStaticShape() &&
      tensor_redistribution->ResetLayoutTransfer() != Status::SUCCESS) {
    MS_LOG(WARNING) << "Failed to reset layout transfer.";
  }
}

static void StepRedistribution(const CNodePtr &cnode, const NodeUsersMap &node_users_map) {
  MS_LOG(DEBUG) << "Do StepRedistribution for " << cnode->fullname_with_scope();
  MS_EXCEPTION_IF_NULL(cnode->func_graph());
  FuncGraphManagerPtr manager = cnode->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  // In pipeline parallel mode, redistribution is inserted after receive, not send.
  if (IsPrimitiveCNode(cnode, prim::kPrimSend) || IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) ||
      IsPrimitiveCNode(cnode, prim::kPrimMakeList)) {
    return;
  }
  // Find Redistribution next_nodes
  // next_node.first.second = (pos in next node input(don't need to -1), pos in tuple(need to -1))
  std::vector<std::pair<std::pair<AnfNodePtr, std::vector<int>>, std::vector<int>>> next_nodes;
  RedistributionNextNode(cnode, manager, node_users_map, {-1}, -1, &next_nodes);
  if (next_nodes.empty()) {
    return;
  }

  // Find Redistribution pre_nodes
  std::vector<AnfNodePtr> pre_nodes;
  RedistributionPreNode(cnode, manager, &pre_nodes);
  if (pre_nodes.size() > 1) {
    MS_LOG(EXCEPTION) << " Don't support Redistribution has multiple pre_node.";
  }

  // Insert Redistribution nodes between pre_nodes and next_nodes
  for (auto &pre_node : pre_nodes) {
    for (auto &next_node : next_nodes) {
      MS_LOG(INFO) << "===========Do Redistribution start============" << std::endl
                   << pre_node->fullname_with_scope() << "->" << next_node.first.first->fullname_with_scope() << "("
                   << next_node.first.second << ")";
      Redistribution(next_node.first, pre_node, next_node.second);
      MS_LOG(INFO) << "===========Do Redistribution end  ============";
    }
    for (const auto &next_node : next_nodes) {
      if (!next_node.first.first->has_user_data(FUNC_PARAM)) {
        continue;
      }
      if (pre_node->func_graph() == next_node.first.first->func_graph()) {
        continue;
      }
      auto param = next_node.first.first->user_data<AnfNode>(FUNC_PARAM);
      auto distribute_operator = GetDistributeOperator(pre_node->cast<CNodePtr>());
      param->set_user_data<OperatorInfo>(distribute_operator);
      break;
    }
  }
}

static void SplitTensor(const AnfNodePtr &node, const CNodePtr &next_node, int64_t index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(next_node);
  OperatorInfoPtr op_info = next_node->user_data<OperatorInfo>();
  if (!op_info) {
    return;
  }

  if (op_info->name().find(FILLV2) != std::string::npos) {
    MS_LOG(INFO) << "FillV2 operator info no need to split tensor";
    return;
  }

  if (op_info->name().find(STAND_ALONE) != std::string::npos) {
    MS_LOG(INFO) << "Stand alone operator info no need to split tensor";
    return;
  }

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
  TensorLayout tensor_layout;
  auto inputs_info_size = op_info->inputs_tensor_info_new().empty() ? op_info->inputs_tensor_info().size()
                                                                    : op_info->inputs_tensor_info_new().size();
  if (LongToSize(index - 1) >= inputs_info_size) {
    if (IsIgnoreSplitTensor(next_node, index - 1)) {
      MS_LOG(INFO) << op_info->name() << ": no need to split tensor for index " << (index - 1);
      return;
    }
    MS_LOG(EXCEPTION) << op_info->name() << ": The index is out of range, index is  " << (index - 1)
                      << ", vector size is  " << inputs_info_size;
  }
  if (op_info->inputs_tensor_info_new().empty()) {
    TensorInfo tensor_info = op_info->inputs_tensor_info()[LongToSize(index - 1)];
    tensor_layout = tensor_info.tensor_layout();
  } else {
    auto tensor_info = op_info->inputs_tensor_info_new()[LongToSize(index - 1)];
    tensor_layout = tensor_info->GetValue().tensor_layout();
  }

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

static void SplitTensorList(const AnfNodePtr &node, const CNodePtr &next_node, int index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(next_node);
  if (((next_node->size() != kSizeTwo) && !IsSomePrimitiveList(next_node, SUPPORT_NEW_SHAPEBASE_OPS)) || index != 1) {
    MS_LOG(INFO) << next_node->fullname_with_scope() << " Inputs must have only one input, get "
                 << (next_node->size() - 1) << " index should be 1, get " << index;
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
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  FuncGraphPtr func_graph = next_node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (op_info->inputs_tensor_info_new().empty()) {
    if (inputs_values.size() != op_info->inputs_tensor_info().size()) {
      MS_LOG(EXCEPTION) << "The inputs size " << inputs_values.size() << ", is not equal to inputs shape size "
                        << op_info->inputs_tensor_info().size();
    }
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
  } else {
    if (inputs_values.size() != op_info->inputs_tensor_info_new()[index - 1]->size()) {
      MS_LOG(EXCEPTION) << "The inputs size " << inputs_values.size() << ", is not equal to inputs shape size "
                        << op_info->inputs_tensor_info_new()[index - 1]->size();
    }
    auto corresponding_tensor_info = op_info->inputs_tensor_info_new()[index - 1];
    ScopePtr scope = next_node->scope();
    MS_EXCEPTION_IF_NULL(scope);
    for (size_t i = 0; i < inputs_values.size(); ++i) {
      auto value_ptr = inputs_values[i];
      auto tensor = value_ptr->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      TensorInfo tensor_info = corresponding_tensor_info->GetElement(SizeToLong(i))->GetValue();
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
  }
  CNodePtr make_tuple = func_graph->NewCNode(make_tuple_inputs);
  (void)manager->Replace(node, make_tuple);
}

static void StepSplitTensor(const AnfNodePtr &node, const FuncGraphManagerPtr &manager) {
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
    if ((use_cnode_prim->name() == DEPEND && node_pair.second != 1) ||
        NO_INPUT_TENSOR_OPS.find(use_cnode_prim->name()) != NO_INPUT_TENSOR_OPS.end()) {
      continue;
    }
    if (IsParallelCareNode(use_cnode)) {
      if (IsPrimitiveCNode(use_cnode, prim::kPrimReceive)) {
        continue;
      }
      if (IsValueNode<ValueList>(node) || IsValueNode<ValueTuple>(node)) {
        SplitTensorList(node, use_cnode, node_pair.second);
      } else {
        SplitTensor(node, use_cnode, node_pair.second);
      }
    }
  }
}

static void StepReplaceOp(OperatorVector replace_op, const CNodePtr &node) {
  MS_LOG(INFO) << "Start StepReplaceOp for " << node->fullname_with_scope();
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

  // When reshape(bool), insert cast in the begin and end of op_list to avoid AllGather(bool).
  auto reshape_type_str = node->abstract()->BuildType()->ToString();
  auto replace_op_info = distribute_operator->replace_op_info();
  if (IsPrimitiveCNode(node, prim::kPrimReshape) && reshape_type_str.find(BOOL) != std::string::npos) {
    auto cast_int = CreateCastOp(kInt32);
    auto cast_bool = CreateCastOp(kBool);
    (void)replace_op.insert(replace_op.cbegin(), cast_int);
    (void)replace_op.insert(replace_op.cend(), cast_bool);
    (void)replace_op_info.insert(replace_op_info.cbegin(), {false, 1});
    (void)replace_op_info.insert(replace_op_info.cend(), {false, 1});
  }

  // step2:traverse op_list and insert node
  std::reverse(replace_op.begin(), replace_op.end());
  std::reverse(replace_op_info.begin(), replace_op_info.end());
  if (!replace_op_info.empty() && replace_op_info.size() != replace_op.size()) {
    MS_LOG(EXCEPTION) << "replace_op_info is not empty and size not equal to replace_op!";
  }
  bool replace_op_info_flag = !replace_op_info.empty();
  for (size_t index = 0; index < replace_op.size(); ++index) {
    std::string instance_name = CreateInstanceName(node, index);
    std::string full_inst_name = std::string(REDISTRIBUTION_OP) + "_" + instance_name;
    std::vector<AnfNodePtr> replace_input;
    if (index != replace_op.size() - 1) {
      replace_input = CreateInput(replace_op[index], node, full_inst_name, node);
    } else {
      replace_input = ReplaceOpInput(replace_op[index], full_inst_name, node);
    }
    CNodePtr replace_node = func_graph->NewCNode(replace_input);
    MS_EXCEPTION_IF_NULL(replace_node);
    ScopePtr scope = node->scope();
    MS_EXCEPTION_IF_NULL(scope);
    replace_node->set_scope(scope);
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(replace_node->input(0));
    PrimitivePtr origin_prim = GetValueNode<PrimitivePtr>(node->input(0));
    SetUserAttrs(origin_prim->attrs(), prim);
    auto origin_prim_attrs = origin_prim->attrs();
    if (origin_prim_attrs.find(RECOMPUTE_COMM_OP) != origin_prim_attrs.end()) {
      auto do_recompute = GetValue<bool>(origin_prim_attrs[RECOMPUTE_COMM_OP]);
      MS_LOG(INFO) << "The redistribution node in reshape would not be recomputed.";
      prim->set_attr(RECOMPUTE, MakeValue(do_recompute));
    }
    if (index == replace_op.size() - 1) {
      replace_node->set_user_data<OperatorInfo>(node->user_data<OperatorInfo>());
      replace_node->set_primal_attrs(node->primal_attrs());
    }
    replace_node->AddPrimalAttr(kPrimalAttrForwardCommNodeUniqueId, MakeValue<std::string>(replace_node->UniqueId()));
    if (node->HasPrimalAttr(MICRO)) {
      replace_node->AddPrimalAttr(MICRO, node->GetPrimalAttr(MICRO));
    }
    replace_node->set_in_forward_flag(true);
    replace_input[0]->set_scope(scope);
    if (replace_op_info_flag && replace_op_info[index].first) {
      auto new_cnode = InsertMakeTuple(replace_node, replace_op_info[index].second, func_graph);
      new_cnode->set_primal_attrs(node->primal_attrs());
      (void)manager->Replace(node, new_cnode);  // using Replace function to insert node
    } else {
      (void)manager->Replace(node, replace_node);  // using Replace function to insert node
    }
  }
  MS_LOG(INFO) << "Insert ReplaceOp success for " << distribute_operator->name();
}

static void StepReplaceGraph(const ReplaceGraphPtr &replace_graph, const CNodePtr &node,
                             const OperatorInfoPtr &op_info) {
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
  // For example input_node:{segment_sum:1, segment_sum:2, gather:2}
  // The Original code here will bind the all operations to the first inputs of these operators
  // However, the segment_sum operation needs two inputs, To solve this
  // We maintain a dict to count the times of the same operations,
  // and bind the inputs according to the times of the op appears.
  mindspore::HashMap<AnfNodePtr, int> input_map = {};
  static int appear_count = 0;
  for (auto &replace_input : replace_graph->first) {
    auto pre_node = node->input(LongToSize(replace_input.second));

    auto it = input_map.find(replace_input.first);
    if (it != input_map.end()) {
      appear_count = 1 + it->second;
    } else {
      appear_count = 1;
    }
    auto replace_input_cnode = replace_input.first->cast<CNodePtr>();
    replace_input_cnode->set_user_data<OperatorInfo>(op_info);
    size_t inputs_size = replace_input_cnode->size();
    while (IntToSize(appear_count) < inputs_size && replace_input_cnode->input(appear_count)->func_graph() != nullptr) {
      ++appear_count;
    }
    if (IntToSize(appear_count) >= inputs_size) {
      MS_LOG(EXCEPTION) << "No replaceable virtual_input_node";
    }
    input_map[replace_input.first] = appear_count;
    replace_input_cnode->set_in_forward_flag(true);
    manager->SetEdge(replace_input.first, appear_count, pre_node);
  }
  //  "(void)manager->Replace(replace_graph->first, pre_node);" can not be called
  auto replace_output = replace_graph->second->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(replace_output);
  replace_output->set_in_forward_flag(true);
  replace_output->set_primal_attrs(node->primal_attrs());
  (void)manager->Replace(node, replace_output);
}

static void InsertVirtualDivOp(const VirtualDivOp &virtual_div_op, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  size_t node_size = node->size();
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  if (IsSomePrimitive(node, DROPOUT_DO_MASK)) {
    MS_LOG(INFO) << "Handle dropout do mask, only insert the virtual div to input[0]";
    node_size = 2;
  }

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

static void InsertRealDivOpToNodeInput(const CNodePtr &node, int64_t scale, const string &instance_name) {
  MS_EXCEPTION_IF_NULL(node);
  if (scale == 0) {
    MS_LOG(EXCEPTION) << "Find the scale value is 0, you should check the mirror operators's group size.";
  }
  size_t node_size = node->size();
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  // instance the real div operator
  Operator div_op = CreateDivOp(LongToFloat(scale));

  // Insert it as the input of the node
  for (size_t index = 1; index < node_size; ++index) {
    AnfNodePtr input = node->input(index);
    MS_EXCEPTION_IF_NULL(input);
    // if it is not a tensor, continue
    if ((!input->isa<CNode>() && !input->isa<Parameter>()) || HasAbstractMonad(input)) {
      continue;
    }
    InsertNode(div_op, node, index, node->input(index), func_graph, instance_name);
  }
}

static void InsertAllReduceToNodeInput(const CNodePtr &node, const std::string &group,
                                       const std::string &instance_name) {
  MS_EXCEPTION_IF_NULL(node);
  size_t node_size = node->size();
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  // instance the real div operator
  CheckGlobalDeviceManager();
  Operator allreduce_op = CreateAllReduceOp(REDUCE_OP_SUM, group);

  // Insert it as the input of the node
  for (size_t index = 1; index < node_size; ++index) {
    AnfNodePtr input = node->input(index);
    MS_EXCEPTION_IF_NULL(input);
    // if it is not a tensor, continue
    if ((!input->isa<CNode>() && !input->isa<Parameter>()) || HasAbstractMonad(input)) {
      continue;
    }

    InsertNode(allreduce_op, node, index, node->input(index), func_graph, instance_name);
  }
}

static FuncGraphPtr PynativeParallelGraph(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes) {
  FuncGraphPtr real_graph = root;
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    auto expect_shard_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (expect_shard_prim->name() != SHARD) {
      continue;
    }
    real_graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
  }
  return real_graph;
}

// find previous parallel care node's next node.
static bool FindPreNodes(const AnfNodePtr &node, std::vector<std::string> *unique_ids, std::vector<size_t> *indexes,
                         size_t curr_depth) {
  if (curr_depth > MAX_RECURSIVE_DEPTH) {
    MS_LOG(WARNING) << "When find the previous node, exceeded the maximum recursion depth: " << MAX_RECURSIVE_DEPTH;
    return false;
  }
  MS_EXCEPTION_IF_NULL(unique_ids);
  MS_EXCEPTION_IF_NULL(indexes);
  if (!node->isa<CNode>()) {
    return false;
  }
  CNodePtr pre_cnode = node->cast<CNodePtr>();
  if (!IsValueNode<Primitive>(pre_cnode->input(0))) {
    return false;
  }
  bool find = false;
  for (size_t index = 1; index < pre_cnode->size(); ++index) {
    if (IsPrimitiveCNode(pre_cnode, prim::kPrimDepend) && index > 1) {
      // For Depend, only the first input will be output.
      break;
    }
    auto next_node = pre_cnode->inputs()[index];
    if (!next_node->isa<CNode>() || next_node->isa<Parameter>()) {
      return false;
    }
    CNodePtr cnode = next_node->cast<CNodePtr>();
    if (!IsValueNode<Primitive>(cnode->input(0))) {
      return false;
    }
    if (IsParallelCareNode(cnode) && !IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) &&
        !IsPrimitiveCNode(cnode, prim::kPrimMakeList)) {
      unique_ids->push_back(pre_cnode->UniqueId());
      indexes->push_back(index);
      find = true;
      continue;
    }
    if (FindPreNodes(cnode, unique_ids, indexes, ++curr_depth)) {
      find = true;
    }
  }
  return find;
}

void InsertVirtualOutput(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes) {
  auto real_graph = PynativeParallelGraph(root, all_nodes);
  auto out_pair = GetRealKernelNode(real_graph->output(), -1, nullptr, false);
  auto out_node = out_pair.first;
  MS_EXCEPTION_IF_NULL(out_node);
  OperatorParams params;
  OperatorAttrs attrs;
  OperatorArgs args = std::make_pair(attrs, params);
  Operator op = std::make_pair(VIRTUAL_OUTPUT, args);
  if (IsPrimitiveCNode(out_node, prim::kPrimMakeTuple)) {
    auto tuple = out_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(tuple);
    for (size_t i = 1; i < tuple->size(); ++i) {
      auto cur_input = tuple->input(i);
      Shapes shape_outputs = GetNodeShape(cur_input);
      if (shape_outputs[0].empty()) {
        continue;
      }
      InsertNode(op, tuple, i, cur_input, tuple->func_graph(), VIRTUAL_OUTPUT);
      auto virtual_output_abstract = cur_input->abstract()->Clone();
      std::shared_ptr<abstract::BaseShape> virtual_output_shape = std::make_shared<abstract::Shape>(shape_outputs[0]);
      virtual_output_abstract->set_shape(virtual_output_shape);
      auto virtual_output_node = tuple->input(i);
      virtual_output_node->set_abstract(virtual_output_abstract);
    }
  } else {
    Shapes shape_outputs = GetNodeShape(out_node);
    if (shape_outputs[0].empty() || out_node->isa<Parameter>()) {
      return;
    }
    auto node_input = CreateInput(op, out_node, VIRTUAL_OUTPUT);
    auto cur_graph = out_node->cast<CNodePtr>()->func_graph();
    MS_EXCEPTION_IF_NULL(cur_graph);
    auto new_node = cur_graph->NewCNode(node_input);
    auto manager = cur_graph->manager();
    (void)manager->Replace(out_node, new_node);
    auto virtual_output_abstract = out_node->abstract()->Clone();
    std::shared_ptr<abstract::BaseShape> virtual_output_shape = std::make_shared<abstract::Shape>(shape_outputs[0]);
    virtual_output_abstract->set_shape(virtual_output_shape);
    new_node->set_abstract(virtual_output_abstract);
  }
}

bool InsertMirrorBeforeCast(const CNodePtr &node, size_t index) {
  // only if gradient_fp32_sync is true, pre node is cast and type is not float32 return true
  bool is_gradient_fp32_sync = ParallelContext::GetInstance()->gradient_fp32_sync();
  auto pre_node = node->input(index);
  MS_EXCEPTION_IF_NULL(pre_node);
  auto cnode = pre_node->cast<CNodePtr>();
  if (cnode == nullptr || !IsValueNode<Primitive>(cnode->input(0))) {
    return false;
  }
  if (ParallelContext::GetInstance()->enable_parallel_optimizer() && IsInAllGatherNodeList(cnode)) {
    pre_node = cnode->input(1);
  }
  if (!IsPrimitiveCNode(pre_node, prim::kPrimCast)) {
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
  if (!is_gradient_fp32_sync && type_id != kNumberTypeFloat32) {
    return false;
  }

  return true;
}

static bool CheckInsertMirrorOps(const MirrorOps &mirror_ops, const CNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimSend)) {
    return true;
  }
  constexpr size_t kSingleArgCNodeSize = 2;
  if ((node->size() == kSingleArgCNodeSize || IsSomePrimitiveList(node, INPUT_IS_TUPLE_OR_LIST_OPS)) &&
      (IsValueNode<ValueSequence>(node->input(1)))) {
    MS_LOG(INFO) << "Input is ValueList, skip it.";
    return false;
  }

  if ((node->size() == kSingleArgCNodeSize || IsSomePrimitiveList(node, INPUT_IS_TUPLE_OR_LIST_OPS)) &&
      (AnfNodeIsPrimitive(node->input(1), MAKE_TUPLE) || AnfNodeIsPrimitive(node->input(1), MAKE_LIST))) {
    MS_LOG(INFO) << "The mirror for " << GetPrimName(node) << " has handle by make_tuple node";
    return false;
  }
  return true;
}

// only used for InsertMirrorOps
static CNodePtr SkipTrivialNodesMoveUp(CNodePtr node) {
  MS_EXCEPTION_IF_NULL(node);
  while (True) {
    if (IsPrimitiveCNode(node, prim::kPrimLoad) || IsInTrivialNodeList(node) || IsInAllGatherNodeList(node)) {
      if (IsPrimitiveCNode(node->input(1), prim::kPrimMicroStepAllGather)) {
        return node;
      }
      if (node->input(1)->isa<Parameter>()) {
        return node;
      }
      node = node->input(1)->cast<CNodePtr>();
    } else {
      MS_LOG(EXCEPTION) << "The node " << node->fullname_with_scope()
                        << " is a abnormal node in inserting mirror node.";
    }
  }
}

static void CreateMirrorForParam(const ParameterPtr param_ptr, OperatorVector *backward_op, bool *is_shared_param) {
  std::string opt_shard_mirror_group;
  if (param_ptr->user_data<TensorLayout>()) {
    opt_shard_mirror_group = param_ptr->user_data<TensorLayout>()->opt_shard_mirror_group();
    *is_shared_param = param_ptr->user_data<TensorLayout>()->is_shared_param();
  }
  if (!opt_shard_mirror_group.empty()) {
    // mirror ops is covered in not fully use opt shard case
    uint32_t group_rank_size = 0;
    if (!CommManager::GetInstance().GetRankSize(opt_shard_mirror_group, &group_rank_size)) {
      MS_LOG(EXCEPTION) << "Got the group size from the group " << opt_shard_mirror_group << " failed";
    }
    *backward_op = CreateMirrorOps(opt_shard_mirror_group, static_cast<size_t>(group_rank_size));
  }
}

static void DoInsertMirrorOps(const FuncGraphPtr &root, const MirrorOps &mirror_ops, const CNodePtr &node) {
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto mirror_size = mirror_ops.size();
  if (IsPrimitiveCNode(node, prim::kPrimSend)) {
    mirror_size = 1;
  }

  for (size_t index = 1; index <= mirror_size; ++index) {
    OperatorVector backward_op = mirror_ops[index - 1];
    if (IsPrimitiveCNode(node, prim::kPrimSend)) {
      auto param_index = GetValue<int>(node->GetPrimalAttr(PARAM_INDEX));
      backward_op = mirror_ops[IntToSize(param_index)];
    }
    if (backward_op.empty()) {
      continue;
    }
    std::pair<AnfNodePtr, bool> param_node_pair = FindParameter(node->input(index), func_graph);
    if (!param_node_pair.first) {
      continue;
    }

    auto param_ptr = param_node_pair.first->cast<ParameterPtr>();
    std::string param_name;
    bool is_shared_param = false;
    if (param_ptr) {
      param_name = param_ptr->name();
      if (!param_ptr->param_info() || !param_ptr->param_info()->requires_grad()) {
        MS_LOG(INFO) << param_name << " do not need gradient. Skip inserting mirror.";
        continue;
      }
      CreateMirrorForParam(param_ptr, &backward_op, &is_shared_param);
    }
    // not a RefKey
    std::string mirror_op_name = MirrorOpName();
    AnfNodePtr pre_node = node->input(index);
    if (!param_node_pair.second) {
      auto next_cnode = FindCNode(param_node_pair.first, mirror_op_name, func_graph, 0);
      // if there is already a MirrorOp in the same graph, use MirrorOp CNode as a input instead
      if (next_cnode.first) {
        MS_EXCEPTION_IF_NULL(next_cnode.second);
        // assume Load is inserted next to parameter
        // skip Load moving up and insert mirror next to the parameter
        if (pre_node->cast<CNodePtr>()) {
          CNodePtr load_node = SkipTrivialNodesMoveUp(node->input(index)->cast<CNodePtr>());
          manager->SetEdge(load_node, 1, next_cnode.second);
        } else {
          manager->SetEdge(node, static_cast<int>(index), next_cnode.second);
        }
        MS_LOG(INFO) << "Find parameter " << param_name << " for node " << GetPrimName(node->cast<CNodePtr>())
                     << " and share the mirror.";
        AddNodeMirrorInfo(node->cast<CNodePtr>(), param_name);
        continue;
      }
    }
    // if the parameter found is a RefKey, or no MirrorOp is found in the same graph, insert a new MirrorOp
    // only one MirrorOp in backward_op
    if (backward_op.size() != 1) {
      MS_LOG(EXCEPTION) << "backward_op size must be 1, real is " << backward_op.size();
    }
    auto op = backward_op[0];
    if (pre_node->cast<CNodePtr>() && (InsertMirrorBeforeCast(node, index) || is_shared_param ||
                                       IsPrimitiveCNode(pre_node, prim::kPrimMirrorSilentCheck))) {
      // assume Load is inserted next to parameter
      // skip Load moving up and insert mirror next to the parameter
      CNodePtr load_node = SkipTrivialNodesMoveUp(pre_node->cast<CNodePtr>());
      InsertNode(op, load_node, 1, load_node->input(1), func_graph, mirror_op_name, param_name, root);
      auto comm_op = load_node->input(1)->cast<CNodePtr>();
      // add fusion flag
      auto fusion_id = AddCommOpFusionType(comm_op, param_node_pair.first);
      MS_LOG(INFO) << "Find parameter " << param_name << " for node " << GetPrimName(node->cast<CNodePtr>())
                   << " and insert mirror before Load";
      AddCommOpParamFlag(comm_op);
      AddNodeFusionInfo(node, comm_op, "all_reduce", param_name, fusion_id);
      continue;
    }
    InsertNode(op, node, index, pre_node, func_graph, mirror_op_name, param_name, root);
    MS_LOG(INFO) << "Find parameter " << param_name << " for node " << GetPrimName(node->cast<CNodePtr>())
                 << " and insert mirror before the node";
    auto comm_op = node->input(index)->cast<CNodePtr>();
    // add fusion flag
    // pipeline mirror would not be set, which should be supported later
    auto fusion_id = AddCommOpFusionType(comm_op, param_node_pair.first);
    AddCommOpParamFlag(comm_op);
    AddNodeFusionInfo(node, comm_op, "all_reduce", param_name, fusion_id);
  }
}

static void InsertMirrorOps(const FuncGraphPtr &root, const MirrorOps &mirror_ops, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!CheckInsertMirrorOps(mirror_ops, node)) {
    return;
  }

  DoInsertMirrorOps(root, mirror_ops, node);
}

static void BackwardCommunication(const FuncGraphPtr &root, const OperatorInfoPtr &distribute_operator,
                                  const CNodePtr &node,
                                  const std::vector<std::pair<CNodePtr, LossNodeInfo>> &sens_loss_pairs) {
  MS_EXCEPTION_IF_NULL(distribute_operator);
  MS_EXCEPTION_IF_NULL(node);

  if (IsPrimitiveCNode(node, prim::kPrimReceive)) {
    return;
  }
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
  if (!virtual_div_op.empty() && is_loss_cnode && IsLastStage()) {
    MS_LOG(INFO) << "insert virtual div op for " << distribute_operator->name();
    InsertVirtualDivOp(virtual_div_op, node);
  }
}

static std::pair<AnfNodePtr, int64_t> FindParallelCareNode(const AnfNodePtr &node, int32_t recursion_num) {
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
    if (IsPrimitiveCNode(cnode, prim::kPrimMirrorSilentCheck) && node_pair.second != 1) {
      continue;
    }
    ValueNodePtr prim_node_anf = cnode->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prim_node_anf);
    PrimitivePtr node_prim = prim_node_anf->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(node_prim);
    if ((node_prim->name() == DEPEND && node_pair.second != 1) || IsPrimitiveCNode(cnode, prim::kPrimReceive) ||
        IsPrimitiveCNode(cnode, prim::kPrimSend)) {
      continue;
    }
    if (node_prim->name() == UPDATESTATE && node_pair.second > 0) {
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

static std::pair<AnfNodePtr, int64_t> FindSubGraph(const FuncGraphPtr &graph, const AnfNodePtr &parameter) {
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
        MS_LOG(EXCEPTION) << "The index is out of range, index is: " << (param_pair.second - 1) << ", vector size is "
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

static CNodePtr InsertAllGatherAfterCast(const std::pair<AnfNodePtr, int> &node_pair) {
  if (ParallelContext::GetInstance()->pipeline_stage_split_num() <= 1) {
    return nullptr;
  }
  auto cnode = node_pair.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  // skip Load moving down and assume it only has one node user
  CNodePtr res = cnode;
  if (IsSomePrimitive(res, LOAD)) {
    res = manager->node_users()[cnode].begin()->first->cast<CNodePtr>();
  }
  // return true only if cnode is Cast from fp32 to fp16
  if (!IsSomePrimitive(res, CAST)) {
    return nullptr;
  }
  auto node_type = res->Type();
  MS_EXCEPTION_IF_NULL(node_type);
  if (!node_type->isa<mindspore::TensorType>()) {
    MS_LOG(EXCEPTION) << "Unknown type.";
  }
  auto input_element_type = node_type->cast<mindspore::TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(input_element_type);
  auto type_id = input_element_type->type_id();

  if (type_id != kNumberTypeFloat32) {
    return res;
  } else {
    return nullptr;
  }
}

void AddAllGatherAttrs(const CNodePtr &allgather, const CNodePtr &cnode, const AnfNodePtr &node,
                       const std::string &op_name, bool add_accu, bool is_with_mirror, bool grad_accumulation_shard) {
  // add fusion flag
  auto fusion_id = AddCommOpFusionType(allgather, node);
  auto param_ptr = node->cast<ParameterPtr>();
  auto param_name = param_ptr->name();
  AddNodeFusionInfo(cnode, allgather, "reduce_scatter", param_name, fusion_id);
  // add gradients mean
  AddCommOpMeanFlag(allgather);
  AddCNodePrimAttr(allgather, "with_mirror_operator", MakeValue<bool>(is_with_mirror));
  if (op_name == MICRO_STEP_ALL_GATHER) {
    // When grad_accumulation_shard is enabled, the ReduceScatter is inserted at each micro step
    // so no need to do backward for the micro_step_allgather
    AddCNodePrimAttr(allgather, DO_MIRROR, MakeValue<bool>(!grad_accumulation_shard));
  } else if (op_name == MINI_STEP_ALL_GATHER) {
    // We need to manually set the add_accu to be false if it's father node is MirrorMiniStep
    AddCNodePrimAttr(allgather, ADD_ACCU, MakeValue<bool>(!add_accu && !is_with_mirror));
    AddCNodePrimAttr(allgather, DO_MIRROR, MakeValue<bool>(!grad_accumulation_shard || !add_accu));
  }
}

static void InsertAllGatherOp(const FuncGraphPtr &root, const std::string &group, const std::pair<AnfNodePtr, int> &res,
                              const AnfNodePtr &node, const std::string &op_name, bool is_shared_param) {
  MS_EXCEPTION_IF_NULL(res.first);
  MS_EXCEPTION_IF_NULL(node);
  bool grad_accumulation_shard = ParallelContext::GetInstance()->grad_accumulation_shard();
  auto cnode = res.first->cast<CNodePtr>();
  auto graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  Operator op;
  CNodePtr allgather;
  auto param_name = node->cast<ParameterPtr>()->name();
  if (op_name == MICRO_STEP_ALL_GATHER) {
    op = CreateMicroStepAllGatherOp(group);
  } else {
    op = CreateAllGatherOp(group);
  }
  CNodePtr cast_node = InsertAllGatherAfterCast(res);
  auto param_ptr = node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(param_ptr);
  bool is_with_mirror = false;
  if (param_ptr->user_data<TensorLayout>()) {
    auto opt_shard_mirror_group = param_ptr->user_data<TensorLayout>()->opt_shard_mirror_group();
    is_with_mirror = !opt_shard_mirror_group.empty();
    if (!param_ptr->param_info()->parallel_optimizer()) {
      auto mirror_group = mirror_group_list(param_ptr->user_data<TensorLayout>());
      is_with_mirror = mirror_group.size() > 1;
    }
  }
  if (!is_shared_param && cast_node) {
    allgather = ReplaceNode(op, cast_node, graph, PARALLEL_OPTIMIZER_ALLGATHER_NOT_COMPUTE, param_name, root);
    MS_LOG(INFO) << "Parallel optimizer is applied before Cast for " << param_name;
  } else {
    auto pre_node = node;
    AnfNodePtr pre_node_ = node;
    auto &node_user_map = manager->node_users();
    TypePtr next_node_dtype = FindChildCastWithFP32ToFP16(res, node_user_map);
    if (next_node_dtype) {
      MS_LOG(INFO) << "Inserting Cast from float32 to float16 for node " << node->fullname_with_scope() << " for saving"
                   << " communication.";
      pre_node_ = CreateFP16Cast(cnode, pre_node, next_node_dtype);
    }
    InsertNode(op, cnode, IntToSize(res.second), pre_node_, graph, PARALLEL_OPTIMIZER_ALLGATHER_NOT_COMPUTE, param_name,
               root);
    allgather = cnode->input(IntToSize(res.second))->cast<CNodePtr>();
    MS_LOG(INFO) << "Parallel optimizer is applied before " << cnode->DebugString() << " for " << param_name;
  }
  bool add_accu = root->has_flag(kAccumulation);
  AddAllGatherAttrs(allgather, cnode, node, op_name, add_accu, is_with_mirror, grad_accumulation_shard);
}

bool IsForwardCNode(const CNodePtr &cnode) {
  if (cnode->in_forward_flag()) {
    return true;
  }
  if (cnode->input(0) && IsValueNode<FuncGraph>(cnode->input(0))) {
    auto func_graph = GetValueNode<FuncGraphPtr>(cnode->input(0));
    auto orders = func_graph->GetOrderedCnodes();
    return std::any_of(orders.begin(), orders.end(), [](const auto &c_node) { return c_node->in_forward_flag(); });
  }
  return false;
}

void InsertParallelOpt(const FuncGraphPtr &root, const AnfNodePtr &parameter, const std::string &opt_shard_group,
                       const std::string &op_name) {
  // insert all gather
  FuncGraphManagerPtr manager = root->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto param_sub_set = manager->node_users()[parameter];
  bool insert_flag = false;
  for (auto &param_pair : param_sub_set) {
    auto cnode = param_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (IsForwardCNode(cnode) && !IsPrimitiveCNode(cnode, prim::kPrimReceive) &&
        !(IsPrimitiveCNode(cnode, prim::kPrimDepend) && param_pair.second == INDEX_TWO)) {
      if (insert_flag) {
        // if there are multiple node users, they share one same allgather
        auto next_cnode = FindCNode(parameter, op_name, cnode->func_graph(), 0);
        if (next_cnode.first) {
          manager->SetEdge(cnode, param_pair.second, next_cnode.second);
          auto param_ptr = parameter->cast<ParameterPtr>();
          MS_EXCEPTION_IF_NULL(param_ptr);
          AddNodeMirrorInfo(cnode, param_ptr->name());
          MS_LOG(INFO) << "Parallel optimizer is shared between " << parameter->ToString() << " and "
                       << GetPrimName(cnode);
        } else {
          MS_LOG(ERROR) << "Can not find the shared AllGather with multiple node users.";
        }
      } else {
        // insert allgather operator between shard parameter and cnode
        auto param_ptr = parameter->cast<ParameterPtr>();
        MS_EXCEPTION_IF_NULL(param_ptr);
        bool is_shared_param = param_ptr->user_data<TensorLayout>()->is_shared_param();
        InsertAllGatherOp(root, opt_shard_group, param_pair, parameter, op_name, is_shared_param);
        insert_flag = true;
      }
    }
  }
}

static void ApplyParallelOptOnParam(const FuncGraphPtr &root, const AnfNodePtr &parameter,
                                    const std::string &opt_shard_group) {
  auto enable_opt_shard = ParallelContext::GetInstance()->enable_parallel_optimizer();
  if (!enable_opt_shard) {
    return;
  }
  MS_EXCEPTION_IF_NULL(parameter);
  if (ParameterIsCloned(parameter)) {
    return;
  }

  int32_t split_stage_num = ParallelContext::GetInstance()->pipeline_stage_split_num();
  if (opt_shard_group.empty() &&
      (split_stage_num <= 1 || !ParameterRequireGrad(parameter) || !root->has_flag(kTraining))) {
    return;
  }

  // set all gather type
  int64_t grad_accumulation_step = ParallelContext::GetInstance()->grad_accumulation_step();
  std::string op_name = ALL_GATHER;
  if (root->has_flag(kTraining)) {
    if ((grad_accumulation_step > 1 || split_stage_num > 1) && ParameterRequireGrad(parameter)) {
      op_name = MICRO_STEP_ALL_GATHER;
    }
  }

  // insert all gather
  InsertParallelOpt(root, parameter, opt_shard_group, op_name);
}

// When this function returns non-empty string, that means parallel optimizer is applied on this parameter.
static std::string SetParallelShape(const AnfNodePtr &parameter, const std::pair<AnfNodePtr, int64_t> &res,
                                    const FuncGraphPtr &root, const int &idx) {
  // check null for param and cnode
  MS_EXCEPTION_IF_NULL(parameter);
  auto param_shape = parameter->Shape();

  MS_EXCEPTION_IF_NULL(param_shape);

  CNodePtr cnode = res.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  // get slice_shape
  OperatorInfoPtr distribute_operator = cnode->user_data<OperatorInfo>();
  if (distribute_operator == nullptr) {
    MS_LOG(EXCEPTION) << "node " << cnode->ToString() << " 's distribute_operator is nullptr";
  }
  TensorLayout tensor_layout;
  if (distribute_operator->inputs_tensor_info_new().empty()) {
    if (LongToSize(res.second - 1) >= distribute_operator->inputs_tensor_info().size()) {
      MS_LOG(EXCEPTION) << "The parameter index is not in inputs_tensor_info. index = " << (res.second - 1)
                        << ", inputs_tensor_info size = " << distribute_operator->inputs_tensor_info().size();
    }
    TensorInfo tensorinfo_in = distribute_operator->inputs_tensor_info()[LongToSize(res.second - 1)];
    tensor_layout = tensorinfo_in.tensor_layout();
  } else {
    TensorInfoBasePtr tensorinfo_in;
    if (idx == -1) {
      tensorinfo_in = distribute_operator->inputs_tensor_info_new()[LongToSize(res.second - 1)];
    } else {
      // idx != -1, input is maketuple
      tensorinfo_in = distribute_operator->inputs_tensor_info_new()[LongToSize(idx)];
    }
    if (tensorinfo_in->is_list()) {
      if (idx == -1) {
        MS_LOG(EXCEPTION) << "The input of " << distribute_operator->name() << " is a list, but idx is -1.";
      }
      tensor_layout = tensorinfo_in->GetElement(res.second - 1)->GetValue().tensor_layout();
    } else {
      tensor_layout = tensorinfo_in->GetValue().tensor_layout();
    }
  }
  Shape slice_shape = tensor_layout.base_slice_shape().array();

  // generate shard group
  std::string opt_shard_group;
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  bool enable_parallel_optimizer = ParallelContext::GetInstance()->enable_parallel_optimizer();
  if (enable_parallel_optimizer) {
    std::unique_ptr<OptParamMgr> apOptParamMgr = createOptParamMgr(root);
    opt_shard_group = apOptParamMgr->ShardOptGroup(parameter, &tensor_layout, distribute_operator);
    // set the shape of parameter to sliced shape
    if (!opt_shard_group.empty()) {
      slice_shape = tensor_layout.opt_shard_slice_shape();
    }
    MS_LOG(INFO) << "the shape of " << parameter->ToString() << "(original: " << param_shape->ToString() << ")"
                 << " will be sliced into " << MakeValue(slice_shape)->ToString() << " in op "
                 << distribute_operator->name();
  }

  AbstractBasePtr abstract = parameter->abstract();
  if (abstract == nullptr) {
    MS_LOG(EXCEPTION) << "parameter " << parameter->ToString() << ": abstract is nullptr";
  }

  AbstractBasePtr cloned_abstract = abstract->Clone();
  if (cloned_abstract == nullptr) {
    MS_LOG(EXCEPTION) << "parameter " << parameter->ToString() << ": abstract clone failed";
  }

  cloned_abstract->set_shape(std::make_shared<abstract::Shape>(slice_shape));
  parameter->set_abstract(cloned_abstract);
  ParameterPtr parameter_ptr = parameter->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(parameter_ptr);
  if (tensor_layout.IsInterleavedParallel()) {
    MS_LOG(EXCEPTION) << "parameter " << parameter->ToString() << " can not set to interleaved parallel";
  }
  parameter_ptr->set_user_data<TensorLayout>(std::make_shared<TensorLayout>(tensor_layout));
  if (ParallelContext::GetInstance()->direct_split() && parameter_ptr->has_default()) {
    auto layout = parameter_ptr->user_data<TensorLayout>();
    MS_LOG(INFO) << "parameter: " << parameter->ToString() << parameter->Shape()->ToString()
                 << "parameter_ptr->default_param()" << parameter_ptr->default_param() << "LAYOUT"
                 << layout->ToString();
    SliceTensorObj(parameter_ptr, layout);
  }
  return opt_shard_group;
}

int ObtainActualInputIdxForSupportedOps(const AnfNodeIndexSet &node_set) {
  int idx = 0;
  for (const auto &node_pair : node_set) {
    auto use_cnode = node_pair.first->cast<CNodePtr>();
    if (IsSomePrimitiveList(use_cnode, SUPPORT_NEW_SHAPEBASE_OPS)) {
      idx = node_pair.second;
    }
  }
  return idx;
}

static void CoverSliceShape(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  auto parameters = root->parameters();
  FuncGraphManagerPtr manager = root->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &node_users_map = manager->node_users();
  for (auto &parameter : parameters) {
    MS_EXCEPTION_IF_NULL(parameter->Shape());
    auto iter = g_RefMap.find(parameter);
    if (iter != g_RefMap.cend()) {
      auto node_set = node_users_map.at(g_RefMap[parameter].first);
      auto idx = ObtainActualInputIdxForSupportedOps(node_set);
      std::string group = SetParallelShape(parameter, g_RefMap[parameter], root, idx - 1);
      // find all forward nodes that use parameter in graphs and insert allgather if group is not empty
      SetSharedParameterFlag(root, parameter);
      ApplyParallelOptOnParam(root, parameter, group);
      continue;
    }

    std::pair<AnfNodePtr, int64_t> res = FindSubGraph(root, parameter);
    if (res.first == nullptr) {
      MS_LOG(INFO) << "Parameter " << parameter->ToString() << " is not in graph, thus no need to set parallel shape";
      if (parameter->has_user_data<TensorLayout>()) {
        auto param_abstract = parameter->abstract()->Clone();
        auto tensor_layout = parameter->user_data<TensorLayout>();
        Shape slice_shape = tensor_layout->base_slice_shape().array();
        param_abstract->set_shape(std::make_shared<abstract::Shape>(slice_shape));
        parameter->set_abstract(param_abstract);
      }
    } else {
      auto node_set = node_users_map.at(res.first);
      auto idx = ObtainActualInputIdxForSupportedOps(node_set);
      std::string group = SetParallelShape(parameter, res, root, idx - 1);
      // find all forward nodes that use parameter in graphs and insert allgather if group is not empty
      SetSharedParameterFlag(root, parameter);
      ApplyParallelOptOnParam(root, parameter, group);
      MS_LOG(DEBUG) << "Parameter " << parameter->ToString() << " shape " << parameter->Shape()->ToString();
    }
  }
  g_RefMap.clear();
}

static void PreProcessActualSeqLenInputForFlashAttentionScore(const FuncGraphPtr &root,
                                                              const std::vector<AnfNodePtr> &all_nodes) {
  auto manager = root->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimFlashAttentionScore)) {
      auto fa_cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(fa_cnode);
      auto fa_inputs = fa_cnode->inputs();
      for (size_t index = ops::kFlashAttentionScoreInputActualSeqQlenIndex;
           index <= ops::kFlashAttentionScoreInputActualSeqKVlenIndex; ++index) {
        auto input = fa_inputs.at(index + 1);
        if (IsValueNode<None>(input)) {
          continue;
        }
        // Transfer Tuple to Tensor
        if (IsPrimitiveCNode(input, prim::kPrimTensorToTuple)) {
          // Eliminate TensorToTuple
          manager->SetEdge(fa_cnode, index + 1, input->cast<CNodePtr>()->input(kIndex1));
          MS_LOG(DEBUG) << "Eliminate TensorToTuple for " << fa_cnode->fullname_with_scope() << ", index is "
                        << index + 1;
        } else {
          auto dtype = NewValueNode(MakeValue<int64_t>(kInt64->type_id()));
          dtype->set_abstract(abstract::FromValue((int64_t)(kInt64->type_id())));
          auto tuple_to_tensor_cnode =
            fa_cnode->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleToTensor), input, dtype});
          auto abs = GenerateAbsByOpInfer(GetCNodePrimitive(tuple_to_tensor_cnode), {input, dtype});
          tuple_to_tensor_cnode->set_abstract(abs);
          manager->SetEdge(fa_cnode, index + 1, tuple_to_tensor_cnode);
          MS_LOG(DEBUG) << "Insert TupleToTensor for " << fa_cnode->fullname_with_scope() << ", index is " << index + 1;
        }
      }
    }
  }
}

static void PostProcessActualSeqLenInputForFlashAttentionScore(const FuncGraphPtr &root,
                                                               const std::vector<AnfNodePtr> &all_nodes) {
  auto manager = root->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimFlashAttentionScore)) {
      auto fa_cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(fa_cnode);
      auto fa_inputs = fa_cnode->inputs();
      for (size_t index = ops::kFlashAttentionScoreInputActualSeqQlenIndex;
           index <= ops::kFlashAttentionScoreInputActualSeqKVlenIndex; ++index) {
        auto input = fa_inputs.at(index + 1);
        auto input_abs = input->abstract();
        if (IsValueNode<None>(input)) {
          continue;
        }

        if (IsPrimitiveCNode(input, prim::kPrimTupleToTensor)) {
          // Eliminate TupleToTensor
          manager->SetEdge(fa_cnode, index + 1, input->cast<CNodePtr>()->input(kIndex1));
          MS_LOG(DEBUG) << "Eliminate TensorToTuple for " << fa_cnode->fullname_with_scope() << ", index is "
                        << index + 1;
        } else {
          // Transfer Tensor to Tuple
          auto tensor_to_tuple_cnode =
            fa_cnode->func_graph()->NewCNode({NewValueNode(prim::kPrimTensorToTuple), input});
          manager->SetEdge(fa_cnode, index + 1, tensor_to_tuple_cnode);
          MS_LOG(DEBUG) << "Insert TensorToTuple for " << fa_cnode->fullname_with_scope() << ", index is " << index + 1;
        }
      }
    }
  }
}

ValuePtr ObtainStrategyForNewShapes(const ShapeBasePtr &shape, const int64_t &dev_num) {
  ValuePtr stra_value_ptr;
  if (shape->is_list()) {
    std::vector<ValuePtr> elements;
    for (size_t i = 0; i < shape->size(); ++i) {
      auto value_stra = ObtainStrategyForNewShapes(shape->GetElement(SizeToLong(i)), dev_num);
      elements.emplace_back(value_stra);
    }
    stra_value_ptr = std::make_shared<ValueTuple>(elements);
  } else {
    Dimensions stra;
    stra.push_back(dev_num);
    for (size_t j = 1; j < shape->size(); ++j) {
      stra.push_back(1);
    }
    stra_value_ptr = MakeValue(stra);
  }
  return stra_value_ptr;
}

void ObtainElementsForStrategyNewShape(const std::vector<NewShapes> &new_shape_list, const int64_t &dev_num,
                                       std::vector<ValuePtr> *elements) {
  for (size_t i = 0; i < new_shape_list[0].size(); i++) {
    if (new_shape_list[0][i]->empty()) {
      (void)elements->emplace_back(MakeValue(Dimensions()));
      continue;
    }
    auto input_strategy = ObtainStrategyForNewShapes(new_shape_list[0][i], dev_num);
    (void)elements->emplace_back(MakeValue(input_strategy));
  }
}

void ObtainElementsForStrategy(const std::vector<Shapes> &shape_list, const int64_t &dev_num,
                               std::vector<ValuePtr> *elements) {
  for (size_t i = 0; i < shape_list[0].size(); i++) {
    if (shape_list[0][i].empty()) {
      (void)elements->emplace_back(MakeValue(Dimensions()));
      continue;
    }
    Dimensions input_strategy;
    input_strategy.push_back(dev_num);
    if (shape_list[0][i][0] > 0 && shape_list[0][i][0] % dev_num != 0) {
      MS_LOG(EXCEPTION) << "The shapes of dataset is " << shape_list[0]
                        << ", the batch dim can not be evenly div by dev_num " << dev_num;
    }
    for (size_t j = 1; j < shape_list[0][i].size(); j++) {
      input_strategy.push_back(1);
    }
    (void)elements->emplace_back(MakeValue(input_strategy));
  }
}

std::pair<std::vector<Shapes>, std::vector<NewShapes>> ObtainShape(const CNodePtr &node) {
  std::vector<Shapes> shape_list;
  std::vector<NewShapes> new_shape_list;
  if (HasSupportedValueSequence(node)) {
    new_shape_list = ExtractNewShape(node);
  } else {
    shape_list = ExtractShape(node);
  }
  return std::make_pair(shape_list, new_shape_list);
}

void SetVirtualDatasetStrategy(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  bool full_batch = ParallelContext::GetInstance()->full_batch();

  PrimitivePtr prim = GetValueNode<PrimitivePtr>(node->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->name() == VIRTUAL_DATA_SET || prim->name() == VIRTUAL_OUTPUT) {
    CheckGlobalDeviceManager();
    auto attrs_temp = prim->attrs();
    if (!ParallelContext::GetInstance()->dataset_strategy().empty() && prim->name() == VIRTUAL_DATA_SET) {
      std::vector<ValuePtr> elements;
      auto dataset_strategy = ParallelContext::GetInstance()->dataset_strategy();
      (void)std::transform(dataset_strategy.begin(), dataset_strategy.end(), std::back_inserter(elements),
                           [](auto input_stra) { return MakeValue(input_stra); });
      ValueTuplePtr strategy = std::make_shared<ValueTuple>(elements);
      attrs_temp[IN_STRATEGY] = strategy;
      (void)prim->SetAttrs(attrs_temp);
      if (prim->HasAttr(REPEAT_DIM_DIRECT) && GetValue<std::string>(prim->GetAttr(REPEAT_DIM_DIRECT)) == RIGHT) {
        ParallelContext::GetInstance()->set_dataset_repeat_dim_right(true);
        MS_LOG(INFO) << "dataset repeat dim is right";
      }
      return;
    }
    int64_t dev_num;
    if (full_batch) {
      dev_num = 1;
    } else {
      dev_num = g_device_manager->stage_device_num();
    }
    if (dev_num == 0) {
      MS_LOG(EXCEPTION) << "Device Num must be larger than 0, but got 0.";
    }
    std::vector<Shapes> shape_list;
    std::vector<NewShapes> new_shape_list;
    if (InDynamicGraph(node)) {
      shape_list = ExtractRealDivisor(node);
      MS_LOG(INFO) << "The node is in dynamic shape graph, the real divisor is " << ShapesToString(shape_list[0]);
    } else {
      std::tie(shape_list, new_shape_list) = ObtainShape(node);
    }
    if (shape_list.empty() && new_shape_list.empty()) {
      MS_LOG(EXCEPTION) << "Failure:node " << node->ToString() << " failed to extract shape";
    }
    std::vector<ValuePtr> elements;
    if (new_shape_list.empty()) {
      ObtainElementsForStrategy(shape_list, dev_num, &elements);
    } else {
      ObtainElementsForStrategyNewShape(new_shape_list, dev_num, &elements);
    }
    ValueTuplePtr strategy = std::make_shared<ValueTuple>(elements);
    attrs_temp[IN_STRATEGY] = strategy;
    (void)prim->SetAttrs(attrs_temp);
  }
}

static bool CheckExtractInformation(const CNodePtr &cnode) {
  if ((cnode == nullptr) || !IsValueNode<Primitive>(cnode->input(0))) {
    return false;
  }

  ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);
  if ((prim->name() == MAKE_TUPLE) || (prim->name() == MAKE_LIST) || (prim->name() == RECEIVE)) {
    return false;
  }

  return IsParallelCareNode(cnode);
}

StrategyPtr GenerateStandAloneStra(const OperatorInfoPtr &op_info) {
  StrategyPtr in_strategy;
  if (op_info->inputs_shape_new().empty()) {
    in_strategy = GenerateStandAloneStrategy(op_info->inputs_shape());
  } else {
    in_strategy = GenerateStandAloneStrategyForNewShapes(op_info->inputs_shape_new());
  }
  return in_strategy;
}

void CheckStrategyAndShape(const StrategyPtr &in_strategy, const OperatorInfoPtr &op_info) {
  MS_EXCEPTION_IF_NULL(in_strategy);
  auto has_tuple_stra = in_strategy->HasTupleInTupleStrategy();
  auto has_new_shape = !op_info->inputs_shape_new().empty();
  if (has_tuple_stra != has_new_shape) {
    MS_LOG(EXCEPTION)
      << "One of the strategy or input shape have tuple in tuple input, but the other does not; in_strategy is "
      << has_tuple_stra << ", input shape is " << has_new_shape;
  }
}

static void ExtractStrategyAndInit(const CNodePtr &cnode, const PrimitivePtr &prim, const OperatorInfoPtr &op_info) {
  StrategyPtr in_strategy = nullptr, out_strategy = nullptr;
  auto attrs = prim->attrs();

  // load strategy map from checkpoint
  StrategyMap stra_map;
  if (StrategyCheckpoint::GetInstance().LoadCheckPointOn() &&
      (StrategyCheckpoint::GetInstance().Load(&stra_map) != SUCCESS)) {
    MS_LOG(EXCEPTION) << "Load strategy checkpoint failed";
  }

  std::string strategy_key_name = "";
  auto param_names = NodeParameterName(cnode, -1, 0);
  if (!param_names.empty()) {
    strategy_key_name = prim->name() + "_" + param_names[0].first;
  }
  std::vector<std::shared_ptr<TensorLayout>> in_tensor_layouts;
  std::vector<std::shared_ptr<TensorLayout>> out_tensor_layouts;
  if (ExtractUserConfigLayout(attrs, op_info->inputs_shape(), op_info->outputs_shape(), &in_tensor_layouts,
                              &out_tensor_layouts) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Failure:operator " << prim->name() << " extract configured layout failed"
                      << trace::DumpSourceLines(cnode);
  }
  if (in_tensor_layouts.empty() && out_tensor_layouts.empty()) {
    bool load_strategy_from_ckpt =
      StrategyCheckpoint::GetInstance().LoadCheckPointOn() && stra_map.find(strategy_key_name) != stra_map.end();
    if (!prim->HasAttr(STAND_ALONE)) {
      if (((!StrategyFound(attrs) && !load_strategy_from_ckpt) && !cnode->HasPrimalAttr(IN_STRATEGY)) ||
          prim->HasAttr(BATCH_PARALLEL)) {
        MS_LOG(INFO) << "ExtractInformation: the strategy of node " << cnode->ToString() << " prim " << prim->name()
                     << " is empty, using batch parallel";
        in_strategy = GenerateBatchParallelStrategy(op_info, prim);
      } else if (cnode->HasPrimalAttr(IN_STRATEGY)) {
        in_strategy = ExtractStrategy(cnode->GetPrimalAttr(IN_STRATEGY));
        out_strategy = ExtractStrategy(cnode->GetPrimalAttr(OUT_STRATEGY));
      } else if (StrategyFound(attrs)) {
        in_strategy = ExtractStrategy(attrs[IN_STRATEGY]);
        out_strategy = ExtractStrategy(attrs[OUT_STRATEGY]);
      } else {
        in_strategy = stra_map[strategy_key_name];
      }
    } else {
      in_strategy = GenerateStandAloneStra(op_info);
    }
    CheckStrategyAndShape(in_strategy, op_info);
  }
  if (op_info->Init(in_strategy, out_strategy, in_tensor_layouts, out_tensor_layouts) == FAILED) {
    MS_LOG(EXCEPTION) << "Failure:operator " << prim->name() << " init failed" << trace::DumpSourceLines(cnode);
  }
}

void ExtractInformation(const std::vector<AnfNodePtr> &all_nodes) {
  SetStridedSliceSplitStrategy(all_nodes);
  for (auto &node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (!CheckExtractInformation(cnode) || IsPrimitiveCNode(node, prim::kPrimSend)) {
      continue;
    }

    SetVirtualDatasetStrategy(cnode);
    ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);

    OperatorInfoPtr operator_ = CreateOperatorInfo(cnode);
    MS_EXCEPTION_IF_NULL(operator_);

    if (prim->name() == RESHAPE) {
      cnode->set_user_data<OperatorInfo>(operator_);
      continue;
    }

    ExtractStrategyAndInit(cnode, prim, operator_);
    cnode->set_user_data<OperatorInfo>(operator_);
  }
}

// if reshape's output connect to several primitive, return the first layout found
static std::shared_ptr<TensorLayout> FindNextLayout(const AnfNodePtr &cnode, bool *next_is_reshape,
                                                    mindspore::HashSet<AnfNodePtr> *visit, int make_tuple_index,
                                                    int tuple_get_index,
                                                    const std::shared_ptr<TensorLayout> &pre_layout) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(next_is_reshape);
  MS_EXCEPTION_IF_NULL(visit);
  MS_EXCEPTION_IF_NULL(cnode->func_graph());
  FuncGraphManagerPtr manager = cnode->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodeIndexSet node_set = manager->node_users()[cnode];
  for (auto &node_pair : node_set) {
    auto use_apply = node_pair.first->cast<CNodePtr>();
    if (visit->find(use_apply) != visit->end()) {
      continue;
    }
    (void)(visit->insert(use_apply));

    if (IsPrimitiveCNode(use_apply, prim::kPrimPrint) || IsPrimitiveCNode(use_apply, prim::kPrimTensorDump)) {
      return pre_layout;
    }

    if (IsValueNode<FuncGraph>(use_apply->input(0))) {
      auto fg = GetValueNode<FuncGraphPtr>(use_apply->input(0));
      MS_EXCEPTION_IF_NULL(fg);
      auto fg_parameters = fg->parameters();
      auto param = fg_parameters[IntToSize(node_pair.second - 1)];
      auto next_layout = FindNextLayout(param, next_is_reshape, visit, make_tuple_index, tuple_get_index, pre_layout);
      if (next_layout != nullptr) {
        return next_layout;
      }
    }

    if (IsPrimitiveCNode(use_apply, prim::kPrimReturn)) {
      auto fg = use_apply->func_graph();
      auto fg_map = fg->func_graph_cnodes_index();
      for (auto &fg_use : fg_map) {
        auto fg_node = fg_use.first->first->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(fg_node);
        auto next_layout =
          FindNextLayout(fg_node, next_is_reshape, visit, make_tuple_index, tuple_get_index, pre_layout);
        if (next_layout != nullptr) {
          return next_layout;
        }
      }
    }

    if (IsPrimitiveCNode(use_apply, prim::kPrimTupleGetItem)) {
      auto temp = LongToInt(GetTupleGetItemIndex(use_apply));
      if (temp != make_tuple_index - 1 && make_tuple_index > 0) {
        continue;
      }
      temp = make_tuple_index > 0 ? -1 : temp;
      auto next_layout = FindNextLayout(use_apply, next_is_reshape, visit, temp, -1, pre_layout);
      if (next_layout != nullptr) {
        return next_layout;
      }
    }

    if (use_apply == nullptr || !IsValueNode<Primitive>(use_apply->input(0))) {
      continue;
    }
    if (IsPrimitiveCNode(use_apply, prim::kPrimReshape)) {
      *next_is_reshape = true;
      continue;
    }
    if (IsOneOfPrimitiveCNode(use_apply, {prim::kPrimDepend, prim::kPrimUpdateState}) && node_pair.second != 1) {
      continue;
    }
    if (IsPrimitiveCNode(use_apply, prim::kPrimMakeTuple)) {
      make_tuple_index = node_pair.second;
      auto next_layout =
        FindNextLayout(use_apply, next_is_reshape, visit, make_tuple_index, tuple_get_index, pre_layout);
      if (next_layout != nullptr) {
        return next_layout;
      }
    }
    if (IsParallelCareNode(use_apply) && use_apply->has_user_data<OperatorInfo>() &&
        IsSomePrimitiveList(use_apply, SUPPORT_NEW_SHAPEBASE_OPS)) {
      MS_LOG(INFO) << "FindNextLayout success node " << use_apply->DebugString() << ", in support new shapebase ops";
      *next_is_reshape = false;
      auto layout = GetInputLayoutFromCNode(node_pair, make_tuple_index);
      return std::make_shared<TensorLayout>(layout);
    }
    if (IsParallelCareNode(use_apply) && use_apply->has_user_data<OperatorInfo>()) {
      if (make_tuple_index > 0) {
        node_pair.second = make_tuple_index;
      }
      MS_LOG(INFO) << "FindNextLayout success node " << use_apply->DebugString();
      *next_is_reshape = false;
      auto layout = GetInputLayoutFromCNode(node_pair, -1);
      return std::make_shared<TensorLayout>(layout);
    }
    MS_LOG(DEBUG) << "FindNextLayout failed node " << use_apply->DebugString() << "  " << IsParallelCareNode(use_apply)
                  << "   " << use_apply->has_user_data<OperatorInfo>();

    auto layout_ptr = FindNextLayout(use_apply, next_is_reshape, visit, make_tuple_index, tuple_get_index, pre_layout);
    if (layout_ptr) {
      return layout_ptr;
    }
  }
  return nullptr;
}

static std::shared_ptr<TensorLayout> GetOutputLayoutFromCNode(const CNodePtr &cnode, size_t output_index) {
  MS_EXCEPTION_IF_NULL(cnode);
  OperatorInfoPtr distribute_operator = GetDistributeOperator(cnode);
  MS_EXCEPTION_IF_NULL(distribute_operator);
  TensorLayout tensorlayout_out;
  if (distribute_operator->outputs_tensor_info_new().empty()) {
    if (distribute_operator->outputs_tensor_info().size() <= output_index) {
      MS_LOG(EXCEPTION) << "outputs_tensor_info size is  " << distribute_operator->outputs_tensor_info().size()
                        << ", must be greater than output_index  " << output_index;
    }
    TensorInfo tensorinfo_out = distribute_operator->outputs_tensor_info()[output_index];
    tensorlayout_out = tensorinfo_out.tensor_layout();
  } else {
    if (distribute_operator->outputs_tensor_info_new().size() <= output_index) {
      MS_LOG(EXCEPTION) << "outputs_tensor_info size is  " << distribute_operator->outputs_tensor_info_new().size()
                        << ", must be greater than output_index  " << output_index;
    }
    auto tensorinfo_out = distribute_operator->outputs_tensor_info_new()[output_index];
    if (tensorinfo_out->is_list()) {
      MS_LOG(EXCEPTION) << "For " << cnode->DebugString() << ": the " << output_index
                        << " out tensorinfo is a list, which does not support yet";
    }
    tensorlayout_out = tensorinfo_out->GetValue().tensor_layout();
  }
  return std::make_shared<TensorLayout>(tensorlayout_out);
}

static std::shared_ptr<TensorLayout> FindPrevParallelCareNodeLayout(const AnfNodePtr &node, size_t output_index) {
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

static RedistributionOpListPtr InferSensRedistribution(const AnfNodePtr &node, const TensorLayout &loss_layout) {
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

// reshape1 ---> depend ---> call @sub_graph(x, y, z)
// sub_graph(x, y, z): reshape2(y)
// find the reshape1 through y
static AnfNodePtr RefParameterToActualNode(const AnfNodePtr &node) {
  if (!node->isa<Parameter>()) {
    return nullptr;
  }
  auto node_param_ptr = node->cast<ParameterPtr>();
  if (node_param_ptr->has_default()) {
    return node;
  }
  auto sub_func_graph = node_param_ptr->func_graph();
  auto call_cnodes_map = sub_func_graph->func_graph_cnodes_index();
  auto sub_graph_parameters = sub_func_graph->parameters();
  auto curr_param_iter = std::find(sub_graph_parameters.begin(), sub_graph_parameters.end(), node);
  if (curr_param_iter == sub_graph_parameters.end()) {
    MS_LOG(EXCEPTION) << "Cannot find param " << node_param_ptr->DebugString() << " in current sub_graph";
  }
  size_t curr_param_index = static_cast<size_t>(curr_param_iter - sub_graph_parameters.begin());
  for (const auto &node_pair : call_cnodes_map) {
    if (!node_pair.first->first->isa<CNode>() || node_pair.first->second > 0) {
      continue;
    }
    auto cnode = node_pair.first->first->cast<CNodePtr>();
    auto cnode_input = cnode->input(curr_param_index + 1);
    auto pre_cnode = GetInputNodeWithFilter(cnode_input, [&](const CNodePtr &cnode) {
      bool filter = IsPrimitiveCNode(cnode, prim::kPrimCast) || IsPrimitiveCNode(cnode, prim::kPrimLoad) ||
                    IsPrimitiveCNode(cnode, prim::kPrimDepend);
      return std::make_pair(filter, 1);
    });
    if (pre_cnode) {
      return pre_cnode;
    }
  }
  return nullptr;
}

static bool IsCommonOp(const AnfNodePtr &node) {
  CNodePtr cnode = node->cast<CNodePtr>();
  bool is_comm_op =
    IsParallelCareNode(cnode) && cnode->has_user_data<OperatorInfo>() && !IsPrimitiveCNode(node, prim::kPrimReshape);
  return is_comm_op;
}

static std::shared_ptr<TensorLayout> FindPrevLayout(const AnfNodePtr &node, bool *is_input_param) {
  if (node->isa<Parameter>()) {
    auto node_param_ptr = node->cast<ParameterPtr>();
    if (node_param_ptr->has_default()) {
      // Only when the real input of Reshape is a parameter that the strategy of Reshape will be assigned to this
      // parameter.
      *is_input_param = true;
      return CreateParameterLayout(node);
    }

    // the node is parameter of sub-graph
    auto actual_node = RefParameterToActualNode(node);
    if (actual_node) {
      return FindPrevLayout(actual_node, is_input_param);
    }
    return nullptr;
  }
  if (!node->isa<CNode>()) {
    return nullptr;
  }
  CNodePtr cnode = node->cast<CNodePtr>();
  if (IsValueNode<FuncGraph>(cnode->input(0))) {
    auto fg = GetValueNode<FuncGraphPtr>(cnode->input(0));
    auto pre_node = GetRealKernelNode(fg->output(), -1, nullptr).first;
    if (!pre_node) {
      return nullptr;
    }
    return FindPrevLayout(pre_node, is_input_param);
  }
  if (!IsValueNode<Primitive>(cnode->input(0))) {
    return nullptr;
  }
  if (IsPrimitiveCNode(node, prim::kPrimReceive)) {
    return cnode->user_data<TensorLayout>();
  }
  if (IsCommonOp(node)) {
    auto layout_ptr = GetOutputLayoutFromCNode(cnode, 0);
    if (!layout_ptr) {
      MS_LOG(EXCEPTION) << "Failure:GetLayoutFromCNode failed";
    }
    return layout_ptr;
  }
  ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
  PrimitivePtr prim = prim_anf_node->value()->cast<PrimitivePtr>();
  if (prim->name() == prim::kPrimTupleGetItem->name()) {
    auto tuple_index = GetTupleGetItemIndex(cnode);
    auto tuple_getitem_input = cnode->input(1)->cast<CNodePtr>();
    if (IsValueNode<FuncGraph>(tuple_getitem_input->input(0))) {
      auto fg = GetValueNode<FuncGraphPtr>(tuple_getitem_input->input(0));
      auto pre_node = GetRealKernelNode(fg->output(), tuple_index, nullptr).first;
      if (!pre_node) {
        return nullptr;
      }
      return FindPrevLayout(pre_node, is_input_param);
    }
    auto layout_ptr = FindPrevParallelCareNodeLayout(cnode->input(1), LongToSize(tuple_index));
    if (!layout_ptr) {
      MS_LOG(EXCEPTION) << " Failure:FindPrevLayout failed, tuple_getitem before reshape, but there does not exit a "
                           "parallel care node "
                           "before tuple_getitem!";
    }
    return layout_ptr;
  }
  for (size_t index = 0; index < cnode->size(); ++index) {
    if (prim->name() == DEPEND && index != 1) {
      continue;
    }
    auto layout_ptr = FindPrevLayout(cnode->inputs()[index], is_input_param);
    if (!layout_ptr) {
      continue;
    }
    return layout_ptr;
  }
  return nullptr;
}

static void ReshapeInit(const std::vector<AnfNodePtr> &all_nodes) {
  MS_LOG(DEBUG) << "=============Do ReshapeInit start=============";
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

    bool is_input_param = false;
    auto prev_layout_ptr = FindPrevLayout(cnode->input(1), &is_input_param);
    if (prev_layout_ptr) {
      auto reshape_info_ptr = std::dynamic_pointer_cast<ReshapeInfo>(operator_info);
      reshape_info_ptr->SetInputLayout(*prev_layout_ptr);
    } else {
      MS_LOG(WARNING)
        << "FindPrevLayout return nullptr, if reshape is not the first primitive, there must be some error";
    }
    auto attrs = prim->attrs();
    if (StrategyFound(attrs) && !is_input_param) {
      MS_LOG(EXCEPTION) << "Setting strategy for Reshape goes for nothing!";
    }
    MS_ASSERT(cnode->size() == RESHAPE_INPUT_SIZE);

    bool is_next_reshape = false;
    mindspore::HashSet<AnfNodePtr> visit;
    auto next_layout_ptr = FindNextLayout(cnode, &is_next_reshape, &visit, -1, -1, prev_layout_ptr);
    if (next_layout_ptr == nullptr) {
      std::string is_reshape = is_next_reshape ? "true" : "false";
      MS_LOG(WARNING) << "FindNextLayout for " << cnode->fullname_with_scope()
                      << " return nullptr, and is_next_reshape is " << is_next_reshape
                      << ". If reshape is not the last primitive, there must be some error.";
    }
    if (next_layout_ptr) {
      auto reshape_info_ptr = std::dynamic_pointer_cast<ReshapeInfo>(operator_info);
      reshape_info_ptr->SetOutputLayout(*next_layout_ptr);
    } else if (is_next_reshape && prev_layout_ptr != nullptr) {
      auto reshape_info_ptr = std::dynamic_pointer_cast<ReshapeInfo>(operator_info);
      reshape_info_ptr->SetOutputLayout(*prev_layout_ptr);
    }
    if (operator_info->Init(nullptr, nullptr) == FAILED) {
      MS_LOG(EXCEPTION) << "Failure:operator " << prim->ToString() << " init failed";
    }
  }
  MS_LOG(DEBUG) << "=============Do ReshapeInit end=============";
}

static CNodePtr HandleDependLoss(const CNodePtr &cnode, size_t curr_depth) {
  if (curr_depth > MAX_RECURSIVE_DEPTH) {
    MS_LOG(WARNING) << "When handling the loss node of Depend, exceeded the max recursive depth: "
                    << MAX_RECURSIVE_DEPTH;
    return nullptr;
  }
  // Handle return->depend->loss
  if (IsPrimitiveCNode(cnode, prim::kPrimDepend) ||
      (IsPrimitiveCNode(cnode, prim::kPrimCast) && !cnode->has_user_data<OperatorInfo>())) {
    auto depend_before = cnode->input(1)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(depend_before);
    return HandleDependLoss(depend_before, ++curr_depth);
  }
  return cnode;
}

static LossNodeInfo FindLossCNode(const FuncGraphPtr &func_graph) {
  LossNodeInfo loss_node_info;
  MS_EXCEPTION_IF_NULL(func_graph);
  CNodePtr return_node = func_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  if (return_node->size() < 2) {
    MS_LOG(EXCEPTION) << "Failure: " << return_node->DebugString() << " size is smaller than 2";
  }
  auto pre_node_pair = GetRealKernelNode(return_node->input(1), -1, nullptr);
  auto pre_node = pre_node_pair.first;
  MS_EXCEPTION_IF_NULL(pre_node);
  auto pre_cnode = pre_node->cast<CNodePtr>();

  if (pre_cnode == nullptr || !IsValueNode<Primitive>(pre_cnode->input(0))) {
    return loss_node_info;
  }
  if (!IsValueNode<Primitive>(pre_cnode->input(0))) {
    MS_LOG(DEBUG) << "pre_cnode:" << pre_cnode->ToString();
    return loss_node_info;
  }
  auto current_prim = GetValueNode<PrimitivePtr>(pre_cnode->input(0));
  // notice: the GetNext op has not input
  if (INVALID_LOSS_OPS.find(current_prim->name()) != INVALID_LOSS_OPS.end()) {
    MS_LOG(INFO) << "The loss is: " << current_prim->name();
    loss_node_info.loss_node = pre_cnode;
    return loss_node_info;
  }

  // return -> tuple_getitem -> loss
  if (pre_node_pair.second != -1) {
    loss_node_info.has_tuple_getitem = true;
    loss_node_info.dout_index = pre_node_pair.second;
    loss_node_info.loss_node = pre_cnode;
    return loss_node_info;
  }

  // return -> make_tuple
  if (current_prim->name() == MAKE_TUPLE) {
    return loss_node_info;
  }

  // return -> loss
  loss_node_info.loss_node = pre_cnode;
  MS_LOG(DEBUG) << "The loss name is " << current_prim->name();
  return loss_node_info;
}

static TensorLayouts GetLossNodeGradOutputLayout(const LossNodeInfo &node_info) {
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
  if (!operator_info) {
    return ret;
  }
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

static void SplitSens(const CNodePtr &grad_sens_node, const TensorLayout &loss_grad_layout) {
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
  auto loss_tensor_map = loss_grad_layout.tensor_map_before();
  bool multi_split = std::any_of(loss_tensor_map.begin(), loss_tensor_map.end(),
                                 [](const auto &tensor_map) { return tensor_map.size() != 1; });
  if ((loss_shape != sens_shape) && !multi_split) {
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
    bool is_dynamic = InDynamicGraph(sens_tensor_node->cast<CNodePtr>());
    if (sens_tensor_node->isa<CNode>() && !is_dynamic) {
      auto op_list_ptr = InferSensRedistribution(sens_tensor_node, loss_grad_layout);
      if (op_list_ptr == nullptr) {
        return;
      }
      auto sens_tensor_cnode = sens_tensor_node->cast<CNodePtr>();
      auto func_graph = grad_sens_node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      TensorRedistributionPtr tensor_redistribution = std::make_shared<TensorRedistribution>();
      InsertRedistribution(op_list_ptr, grad_sens_node, func_graph, 1, sens_tensor_cnode, tensor_redistribution);
      return;
    }
    if (is_dynamic) {
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

static void InsertForwardOps(const OperatorInfoPtr &distribute_operator, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(distribute_operator);
  MS_EXCEPTION_IF_NULL(cnode);
  if (IsPrimitiveCNode(cnode, prim::kPrimReceive)) {
    return;
  }
  OperatorVector forward_op = distribute_operator->forward_op();
  // for gmm, its make tuple will inherit its op info,
  // which will lead to insert allreduce for maketuple.
  if (!forward_op.empty() && !IsPrimitiveCNode(cnode, prim::kPrimMakeTuple)) {
    MS_LOG(INFO) << "Insert forward op for " << distribute_operator->name();
    ForwardCommunication(forward_op, cnode);
  }
}

static void StepReplace(const std::vector<AnfNodePtr> &all_nodes) {
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      if (!IsParallelCareNode(cnode) || !cnode->has_user_data<OperatorInfo>() || IsSomePrimitive(cnode, RECEIVE) ||
          IsSomePrimitive(cnode, SEND)) {
        continue;
      }

      OperatorInfoPtr distribute_operator = GetDistributeOperator(cnode);
      // StepReplace
      MS_EXCEPTION_IF_NULL(distribute_operator);
      auto replace_op = distribute_operator->replace_op();
      if (!replace_op.empty()) {
        MS_LOG(INFO) << "StepReplaceOp " << cnode->ToString();
        StepReplaceOp(replace_op, cnode);
      }

      // StepReplaceGraph: after calling StepReplaceGraph, cnode can not be used anymore.
      auto replace_graph = distribute_operator->replace_graph(cnode);
      if (!replace_op.empty() && replace_graph) {
        MS_LOG(EXCEPTION) << "Only one of replace_op or replace_op can be used";
      }
      if (replace_graph) {
        MS_LOG(INFO) << "StepReplaceGraph " << cnode->ToString();
        StepReplaceGraph(replace_graph, cnode, distribute_operator);
      }
      if (distribute_operator->name().find(RESHAPEINFO) != std::string::npos) {
        auto reshape_info = std::dynamic_pointer_cast<ReshapeInfo>(distribute_operator);
        if (!reshape_info->InterleavedParallel()) {
          continue;
        }
        auto reshape_redis = reshape_info->ReshapeRedistribution();
        InsertRedistributionForMicroInterleaved(reshape_redis, {cnode, 1}, cnode->func_graph(), cnode,
                                                cnode->input(kIndex1)->cast<CNodePtr>());
        if (!IsPrimitiveCNode(cnode->input(kIndex1), prim::kPrimVirtualConverterEnd)) {
          continue;
        }
        auto virtual_converter_end = cnode->input(kIndex1)->cast<CNodePtr>();
        auto func_graph = cnode->func_graph();
        MS_EXCEPTION_IF_NULL(func_graph);
        auto manager = func_graph->manager();
        MS_EXCEPTION_IF_NULL(manager);
        manager->Replace(cnode, virtual_converter_end);
      }
    }
  }
}

static void StepSplitSens(const std::pair<CNodePtr, LossNodeInfo> &sens_loss_pair) {
  CNodePtr sens_node = sens_loss_pair.first;
  auto loss_node = sens_loss_pair.second;
  auto loss_grad_layout = GetLossNodeGradOutputLayout(loss_node);
  if (!loss_grad_layout.empty()) {
    SplitSens(sens_node, loss_grad_layout[0]);
  }
}

// Sens node satisfies the following conditions: cnode(sens)-->cnode(tuple_getitem)-->cnode-->cnode(J)
static std::vector<std::pair<CNodePtr, LossNodeInfo>> GetSensLossPairs(const FuncGraphPtr &root) {
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
    if (!IsSomePrimitive(expect_tuple_getitem_cnode, prim::kPrimTupleGetItem->name())) {
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

static void HandleSens(const std::vector<std::pair<CNodePtr, LossNodeInfo>> &sens_loss_pairs) {
  // split sens must before inserting the operators.
  for (auto &pair : sens_loss_pairs) {
    // If the shape of grad-sens tensor is not [] or [1], use get tensor slice to handle it.
    // If the type of sens node is not Tensor, it is unsupported now, do nothing default.
    if (IsLastStage()) {
      StepSplitSens(pair);
    }
  }
  return;
}

static void ParallelCommunication(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes,
                                  const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(manager);

  std::vector<std::pair<CNodePtr, LossNodeInfo>> sens_loss_pairs = GetSensLossPairs(root);
  auto has_backward = HasBackward(root);
  // split sens must before inserting the operators.
  HandleSens(sens_loss_pairs);

  const auto &node_users_map = manager->node_users();
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      if (IsValueNode<FuncGraph>(cnode->input(0))) {
        StepRedistribution(cnode, node_users_map);
        continue;
      }
      // the make_tuple is parallel care node, but it may have not operator info
      if ((!IsParallelCareNode(cnode) || !cnode->has_user_data<OperatorInfo>()) && !IsControlFlowNode(cnode)) {
        continue;
      }
      OperatorInfoPtr distribute_operator = nullptr;
      if (!IsControlFlowNode(cnode)) {
        distribute_operator = GetDistributeOperator(cnode);
        MS_EXCEPTION_IF_NULL(distribute_operator);
      }

      // skip Send Receive
      auto parallel_context = parallel::ParallelContext::GetInstance();
      MS_EXCEPTION_IF_NULL(parallel_context);
      auto is_pp_interleave = parallel_context->pipeline_interleave();
      if (!cnode->HasPrimalAttr(PIPELINE_PARAM) || is_pp_interleave) {
        // insert forward ops
        if (!IsControlFlowNode(cnode)) {
          InsertForwardOps(distribute_operator, cnode);
        }

        // insert redistribution ops
        StepRedistribution(cnode, node_users_map);
      }
      // insert backward ops
      if (!IsControlFlowNode(cnode) && (has_backward || IsPynativeParallel())) {
        BackwardCommunication(root, distribute_operator, cnode, sens_loss_pairs);
      }
      if (!IsControlFlowNode(cnode)) {
        distribute_operator->ReplaceNodeInputOrAttrs();
      }
    } else if (IsValueNode<Tensor>(node) || IsValueNode<ValueList>(node) || IsValueNode<ValueTuple>(node)) {
      StepSplitTensor(node, manager);
    }
  }
  // StepReplace
  StepReplace(all_nodes);
}

static bool IsGatherInfo(const std::string &name) {
  std::vector<std::string> gather_info_names = {"GatherInfo", "SparseGatherV2Info", "EmbeddingLookupInfo"};
  for (std::string info_name : gather_info_names) {
    if (name.find(info_name) != std::string::npos) {
      return true;
    }
  }
  return false;
}

void AssignStrategyMap(const StrategyPtr &stra, const std::string &strategy_key_name, StrategyMap *stra_map) {
  if (stra) {
    (*stra_map)[strategy_key_name] = stra;
  } else {
    Strategies new_stra_v;
    StrategyPtr new_stra = std::make_shared<Strategy>(g_device_manager->stage_id(), new_stra_v);
    (*stra_map)[strategy_key_name] = new_stra;
  }
}

void AssignManualShapeMapForGather(const OperatorInfoPtr &operator_info, const std::string &param_name,
                                   ManualShapeMap *manual_shape_map) {
  if (IsGatherInfo(operator_info->name())) {
    auto gather_info = std::dynamic_pointer_cast<GatherInfo>(operator_info);
    auto param_split_shapes = gather_info->param_split_shapes();
    auto index_offsets = gather_info->index_offsets();
    if (param_split_shapes.size() != index_offsets.size()) {
      MS_LOG(EXCEPTION) << "In manual split, the param_split_shapes and index_offsets length should be same.";
    }
    std::vector<std::pair<int64_t, int64_t>> manual_shape;
    for (int64_t i = 0; i < UlongToLong(param_split_shapes.size()); ++i) {
      (void)manual_shape.emplace_back(std::make_pair(param_split_shapes[LongToSize(i)], index_offsets[LongToSize(i)]));
    }
    (*manual_shape_map)[param_name] = manual_shape;
  }
}

static void CheckpointStrategy(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root) {
  if (!StrategyCheckpoint::GetInstance().SaveCheckPointOn()) {
    return;
  }

  StrategyMap stra_map;
  TensorInfoMap tensor_info_map;
  ManualShapeMap manual_shape_map;
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    auto cnode = node->cast<CNodePtr>();
    if ((cnode == nullptr) || !IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    auto param_names = NodeParameterName(cnode, -1, 0);
    if (param_names.empty()) {
      continue;
    }
    string param_name = param_names[0].first;
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_EXCEPTION_IF_NULL(prim);
    OperatorInfoPtr operator_info = cnode->user_data<OperatorInfo>();
    if (operator_info) {
      std::string strategy_key_name = prim->name() + "_" + param_name;
      StrategyPtr stra;
      if (operator_info->name().find(RESHAPEINFO) != std::string::npos) {
        auto reshape_info = std::dynamic_pointer_cast<ReshapeInfo>(operator_info);
        stra = reshape_info->get_input_shard_strategy();
        if (stra == nullptr) {
          MS_LOG(INFO) << "Reshape has not input strategy, Skipped";
          continue;
        }
      } else {
        stra = operator_info->strategy();
      }
      AssignStrategyMap(stra, strategy_key_name, &stra_map);

      for (auto param_name_pair : param_names) {
        tensor_info_map[param_name_pair.first] = param_name_pair.second->user_data<TensorLayout>();
      }
      AssignManualShapeMapForGather(operator_info, param_name, &manual_shape_map);
    }
  }
  for (auto &cloned_parameter_node : root->parameters()) {
    MS_EXCEPTION_IF_NULL(cloned_parameter_node);
    auto cloned_parameter = cloned_parameter_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(cloned_parameter);

    if (!ParameterIsCloned(cloned_parameter_node) && !IsStrategySaved(cloned_parameter_node)) {
      continue;
    }
    std::string cloned_param_name = cloned_parameter_node->cast<ParameterPtr>()->name();
    auto cloned_param_layout = cloned_parameter_node->user_data<TensorLayout>();
    if (cloned_param_layout == nullptr) {
      continue;
    }
    tensor_info_map[cloned_param_name] = cloned_param_layout;
  }
  if (StrategyCheckpoint::GetInstance().Save(stra_map, tensor_info_map, manual_shape_map) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Save strategy checkpoint failed";
  }
}

static void SetForwardFlag(const std::vector<AnfNodePtr> &all_nodes) {
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

static void SetForwardFlag(const AnfNodeSet &all_nodes) {
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
  auto ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  auto all_nodes = TopoSort(ret, SuccDeeperSimple);
  std::set<FuncGraphPtr> graph_set = FindForwardGraphByRootNodes(all_nodes);
  return graph_set;
}

static std::vector<AnfNodePtr> FindRootForwardCNode(const FuncGraphPtr &graph,
                                                    const std::vector<AnfNodePtr> &all_nodes) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> root_forward_nodes;
  auto loss_cnode = FindLossCNode(graph).loss_node;
  if (loss_cnode == nullptr) {
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

static void InsertShapeOp(const CNodePtr &node, const AnfNodePtr &pre_node, const FuncGraphPtr &root) {
  // shape op doesn't have params and attrs.
  OperatorParams params;
  OperatorAttrs attrs;
  auto shape_value = GetValueNode(node->input(2))->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(shape_value);
  auto shape = shape_value->value();
  if (shape.empty()) {
    return;
  }
  OperatorArgs args = std::make_pair(attrs, params);
  Operator op = std::make_pair(SHAPE_OP, args);
  InsertNode(op, node, 2, pre_node, root, "shape");
}

static AnfNodePtr FindGrad(const CNodePtr &cnode, size_t curr_depth) {
  if (curr_depth > MAX_RECURSIVE_DEPTH) {
    MS_LOG(WARNING) << "When finding Grad nodes, exceeded the maximum recursion depth: " << MAX_RECURSIVE_DEPTH;
    return nullptr;
  }
  for (auto &node : cnode->inputs()) {
    if (!node->isa<CNode>()) {
      continue;
    }
    if (!IsPrimitiveCNode(node, prim::kPrimEnvironGet)) {
      return FindGrad(node->cast<CNodePtr>(), ++curr_depth);
    } else {
      return node;
    }
  }
  return nullptr;
}

static void HandleRootReshapeAndSaveStrategy(const std::vector<AnfNodePtr> &all_nodes) {
  // If root graph has reshape op. Find the corresponding parameter.
  // Reshape's shape is the shape of the parameter.
  auto executor = pipeline::GraphExecutorPy::GetInstance();
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

    Shape origin_dst_shape = GetValue<std::vector<int64_t>>(cnode->input(2)->cast<ValueNodePtr>()->value());
    if (origin_dst_shape.size() == 1 && origin_dst_shape[0] == -1) {
      continue;
    }
    auto root = node->func_graph();
    auto grad_node = FindGrad(cnode, 0);
    if (grad_node) {
      InsertShapeOp(cnode, grad_node, root);
    }
  }
}

void MarkForwardCNode(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  auto ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  auto all_nodes = TopoSort(ret, SuccDeeperSimple);
  auto graph_set = FindForwardGraphByRootNodes(all_nodes);

  if (graph_set.empty()) {
    MS_LOG(INFO) << "Can not find the forward graph, so mark the ops in root graph";
    auto fgs = root->manager()->func_graphs();
    for (auto fg = fgs.cbegin(); fg != fgs.cend(); ++fg) {
      SetForwardFlag((*fg)->nodes());
    }
  } else {
    for (auto func_graph = graph_set.cbegin(); func_graph != graph_set.cend(); ++func_graph) {
      MS_LOG(INFO) << "The sub graph size of root is " << root->func_graphs_used().size();
      auto return_node = (*func_graph)->get_return();
      MS_EXCEPTION_IF_NULL(return_node);
      auto all_dfs_nodes = DeepLinkedGraphSearch(return_node);
      SetForwardFlag(all_dfs_nodes);
      auto root_forward_nodes = FindRootForwardCNode(*func_graph, all_nodes);
      if (root_forward_nodes.empty()) {
        continue;
      }
      // Mark forward flag for the nodes in root graph.
      SetForwardFlag(root_forward_nodes);
    }
  }
}

OperatorInfoPtr set_make_list_for_ifa(CNodePtr make_list, const CNodePtr &next_node) {
  ValueNodePtr anf_node = next_node->input(0)->cast<ValueNodePtr>();
  if (!anf_node) {
    return nullptr;
  }
  PrimitivePtr prim = anf_node->value()->cast<PrimitivePtr>();
  if (!prim) {
    return nullptr;
  }
  if (prim->name() != INCRE_FLASH_ATTENTION) {
    return nullptr;
  }

  int kv_index = 1;
  OperatorInfoPtr operator_make_list = CreateOperatorInfo(make_list);
  auto make_list_prim = GetValueNode<PrimitivePtr>(make_list->input(0));
  if (make_list_prim->HasAttr(STAND_ALONE)) {
    (void)make_list_prim->DelAttr(STAND_ALONE);
  }
  OperatorInfoPtr next_operator = next_node->user_data<OperatorInfo>();
  StrategyPtr next_node_strategy = next_operator->strategy();
  Strategies key_value_strategies;
  Dimensions key_value_dim = next_node_strategy->GetInputDim().at(kv_index);
  key_value_strategies.push_back(key_value_dim);
  auto make_list_stage = next_node_strategy->GetInputStage();
  auto make_list_new_in_stra = NewStrategy(make_list_stage, key_value_strategies);
  operator_make_list->set_strategy(make_list_new_in_stra);

  std::vector<TensorInfo> kv_in_tensor_info(1, next_operator->inputs_tensor_info()[kv_index]);
  operator_make_list->set_inputs_tensor_info(kv_in_tensor_info);
  return operator_make_list;
}

static void HandleForwardMakeTupleAndMakeList(const std::vector<AnfNodePtr> &all_nodes) {
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

    // MakeTuple has multiple users, each user's TensorInfo must be same.
    auto make_tuple_list_next_node = CheckMakeTupleSplit(node, manager);
    if (make_tuple_list_next_node == nullptr) {
      continue;
    }
    auto make_tuple_list_next_cnode = make_tuple_list_next_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple_list_next_cnode);
    if (!IsSomePrimitiveList(make_tuple_list_next_cnode, INPUT_IS_TUPLE_OR_LIST_OPS)) {
      continue;
    }

    OperatorInfoPtr op_info = set_make_list_for_ifa(cnode, make_tuple_list_next_cnode);
    if (op_info == nullptr) {
      op_info = GetDistributeOperator(make_tuple_list_next_cnode);
    }
    MS_EXCEPTION_IF_NULL(op_info);
    cnode->set_user_data<OperatorInfo>(op_info);
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

static void ReorderForPipelineSplit(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager,
                                    int64_t pipeline_stages) {
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto is_pp_interleave = parallel_context->pipeline_interleave();
  if (is_pp_interleave) {
    return;
  }
  if (!root->has_flag(kSkipAutoParallelCompile) && !root->has_flag(BACKWARD) && pipeline_stages > 1) {
    root->set_flag(BACKWARD, true);
    if (IsTraining(manager)) {
      if (parallel_context->enable_fold_pipeline()) {
        MS_LOG(INFO) << "Begin Fold Pipeline Reorder. ";
        FoldPipelineReorder(root);
      } else {
        Reorder(root);
      }
    } else {
      ReorderForPredict(root, manager);
    }
  }
}

static void ReorderForGradAccumulation(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager) {
  if (!root->has_flag(kSkipAutoParallelCompile) && !root->has_flag(BACKWARD) &&
      ParallelContext::GetInstance()->grad_accumulation_step() > 1) {
    root->set_flag(BACKWARD, true);
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    const auto cell_reuse = context->CellReuseLevel() != CellReuseLevel::kNoCellReuse;
    DumpGraph(root, "before_reorder");
    if (IsTraining(manager)) {
      if (cell_reuse) {
        TagMicroBatchBpEndInCellShare(root, manager);
      }
      std::unordered_map<int64_t, std::vector<CNodePtr>> forward_start;
      std::unordered_map<int64_t, std::vector<CNodePtr>> backward_end;
      ExtractMicroBatchBorderNodes(root, &forward_start, &backward_end);
      ReorderGradAccumulation(root, forward_start, backward_end);
      DumpGraph(root, "after_reorder");
    } else {
      MS_LOG(EXCEPTION) << "Current not support predict with grad_accu";
    }
  }
}

static void HandleDataParallel() {
  std::string parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode == kDataParallel) {
    auto group_info_save_path = common::GetEnv("GROUP_INFO_FILE");
    if (!group_info_save_path.empty()) {
      std::vector<std::pair<std::string, std::vector<uint32_t>>> group_info;
      int64_t device_num = GetCommInfo().device_num;
      RankList comm_group;
      for (size_t i = 0; i < size_t(device_num); ++i) {
        comm_group.push_back(i);
      }
      ParallelContext::GetInstance()->set_group_ckpt_save_file(group_info_save_path);
      if (StrategyCheckpoint::GetInstance().SaveGroupInfo(group_info, comm_group) != SUCCESS) {
        MS_LOG(EXCEPTION) << "Save group info failed";
      }
    }
  }
}

static void MicroBatchPreProcess(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager,
                                 const std::vector<AnfNodePtr> &all_nodes) {
  auto pipeline_stages = ParallelContext::GetInstance()->pipeline_stage_split_num();
  if (pipeline_stages > 1) {
    HandleMicroBatch(all_nodes, manager);
    ParameterStartNode(all_nodes, manager);
    LastStageEndNode(all_nodes, manager, root);
    return;
  }
  TagMicroBatchStart(manager, all_nodes);
  TagMicroBatchEnd(manager, all_nodes);
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const auto no_cell_reuse = context->CellReuseLevel() == CellReuseLevel::kNoCellReuse;
  bool enable_grad_accu = ParallelContext::GetInstance()->grad_accumulation_step() > 1;
  if (no_cell_reuse && enable_grad_accu) {
    TagMicroBatchBpEndPrim(root);
    TagMicroBatchBpEnd(root);
  }
}

static void MicroBatchPostProcess(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes) {
  auto pipeline_stages = ParallelContext::GetInstance()->pipeline_stage_split_num();
  if (pipeline_stages > 1) {
    AddVirtualAssignAdd(root);
    HandleReceiveParam(root);
    LabelGenMaskMicro(root);
    return;
  }
  if (ParallelContext::GetInstance()->grad_accumulation_step() > 1) {
    AddVirtualAssignAdd(root);
    LabelGenMaskMicro(root);
  }
}

static void InsertAllReduceForNormValue(const AnfNodePtr &res_node) {
  auto cnode = res_node->cast<CNodePtr>();
  auto graphs = res_node->func_graph();
  MS_EXCEPTION_IF_NULL(graphs);
  auto manager = graphs->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_user_map = manager->node_users();
  if (!IsSomePrimitive(cnode, EXPAND_DIMS)) {
    MS_LOG(ERROR) << "Expected the operator expand_dims, but found the " << GetPrimName(cnode)
                  << "This may cause the calculation of the global norm incorrect";
    return;
  }
  auto pipeline_stages = ParallelContext::GetInstance()->pipeline_stage_split_num();
  auto find_node = res_node;
  uint32_t limits = 0;
  while (!IsSomePrimitive(find_node->cast<CNodePtr>(), SQRT) && limits < MAX_BFS_DEPTH) {
    auto users = node_user_map.at(find_node);
    if (users.empty()) {
      return;
    }
    find_node = users.front().first;
    ++limits;
  }
  if (!find_node || !IsSomePrimitive(find_node->cast<CNodePtr>(), SQRT)) {
    return;
  }
  auto anf_node = find_node->cast<CNodePtr>();
  if (anf_node->size() > 1 && IsSomePrimitive(anf_node->input(1)->cast<CNodePtr>(), ALL_REDUCE)) {
    return;
  }
  auto sqrt_node = find_node;
  auto cur_stage_rank_list = g_device_manager->GetDeviceListInThisStage();
  Group cur_stage_device_list;
  if (g_device_manager->CreateGroup(cur_stage_rank_list, &cur_stage_device_list) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Create the communication group for allreduce in calculating global norm failed, "
                         "the rank_list is: "
                      << cur_stage_rank_list;
  }
  InsertAllReduceToNodeInput(sqrt_node->cast<CNodePtr>(), cur_stage_device_list.name(), PARALLEL_GLOBALNORM);
  MS_LOG(INFO) << "Insert the AllReduce for global norm value in stages succeed.";
  if (pipeline_stages > 1) {
    MS_LOG(INFO) << "Insert the AllReduce for global norm value between stages succeed.";
    auto ranks_between_stages = g_device_manager->GetDeviceListBetweenStage();
    Group group_between_stages;
    if (g_device_manager->CreateGroup(ranks_between_stages, &group_between_stages) != SUCCESS) {
      MS_LOG(EXCEPTION) << "Create the communication group for allreduce in calculating global norm "
                           "with pipeline parallel failed, the rank_list is: "
                        << cur_stage_rank_list;
    }
    InsertAllReduceToNodeInput(sqrt_node->cast<CNodePtr>(), group_between_stages.name(), PARALLEL_GLOBALNORM_BETWEEN);
  }
}

static AnfNodePtr FindExpandDimsWIthGradScale(const AnfNodePtr &node_ptr, const NodeUsersMap &node_users_map,
                                              uint32_t limits) {
  std::queue<AnfNodePtr> visited;
  AnfNodePtr queue_node = nullptr;
  CNodePtr cnode = nullptr;
  AnfNodePtr last_node = nullptr;
  uint32_t depth = 0;
  if (!node_ptr) {
    return nullptr;
  }
  visited.push(node_ptr);
  while (!visited.empty()) {
    queue_node = visited.front();
    visited.pop();
    cnode = queue_node->cast<CNodePtr>();
    // MAKE_TUPLE will not appear after the load in the forward graph
    if (IsSomePrimitive(cnode, EXPAND_DIMS)) {
      auto value = GetAttrsFromAnfNode(queue_node, GRAD_SCALE);
      if (!value || !GetValue<bool>(value)) {
        continue;
      }
      return queue_node;
    }
    if (!IsSomePrimitiveList(cnode,
                             {ENVIRONGET, MUL, SQUARE, REDUCE_SUM, EXPAND_DIMS, DEPEND, CAST, REF_TO_EMBED, EMBED})) {
      continue;
    }
    auto node_set = node_users_map.at(queue_node);
    for (auto &node_user : node_set) {
      visited.push(node_user.first);
    }
    if (!last_node || last_node == queue_node) {
      if (++depth == limits) {
        break;
      }
      last_node = visited.back();
    }
  }
  return nullptr;
}

static void InsertDivAndAllReduceForNorm(const NodeUsersMap &node_user_map, const AnfNodePtr &parameter,
                                         uint32_t dev_num) {
  auto params_user_set = node_user_map.at(parameter);
  for (auto &param_pair : params_user_set) {
    auto cnode = param_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->in_forward_flag()) {
      continue;
    }
    constexpr size_t bfs_depth = 10;
    auto expand_dims_node = FindExpandDimsWIthGradScale(cnode, node_user_map, bfs_depth);
    if (!expand_dims_node) {
      continue;
    }
    auto value = GetAttrsFromAnfNode(expand_dims_node, GRAD_SCALE);
    if (!value || !GetValue<bool>(value)) {
      continue;
    }
    if (dev_num > 0) {
      InsertRealDivOpToNodeInput(expand_dims_node->cast<CNodePtr>(), dev_num, PARALLEL_GLOBALNORM_DIV);
      MS_LOG(INFO) << "Insert the realdiv with " << dev_num << " for the parameter " << parameter->fullname_with_scope()
                   << " succeed!";
    }
    // If already inserted allreduce, the pattern will not be matched and thus no allreduce will be inserted.
    InsertAllReduceForNormValue(expand_dims_node);
  }
}

static AnfNodePtr GetMirrorOp(const NodeUsersMap &node_user_map, const AnfNodePtr &parameter) {
  auto params_user_set = node_user_map.at(parameter);
  for (auto &param_pair : params_user_set) {
    auto cnode = param_pair.first->cast<CNodePtr>();
    std::vector<AnfNodePtr> candidate = {cnode};
    if (!cnode->in_forward_flag()) {
      continue;
    }
    while (IsInTrivialNodeList(cnode) || IsSomePrimitive(cnode, LOAD) ||
           IsPrimitiveCNode(cnode, prim::kPrimMicroStepAllGather) || IsPrimitiveCNode(cnode, prim::kPrimAllGather)) {
      auto load_users = node_user_map.at(cnode);
      cnode = node_user_map.at(cnode).front().first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      (void)std::transform(load_users.begin(), load_users.end(), std::back_inserter(candidate),
                           [](const auto &v) { return v.first; });
    }
    for (auto &node : candidate) {
      auto local_cnode = node->cast<CNodePtr>();
      if (!IsPrimitiveCNode(local_cnode, prim::kPrimMirror) &&
          !IsPrimitiveCNode(local_cnode, prim::kPrimMirrorMicroStep) &&
          !IsPrimitiveCNode(local_cnode, prim::kPrimMirrorMiniStep)) {
        continue;
      }
      return node;
    }
  }
  return nullptr;
}

static void HandleGlobalNormScale(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager) {
  auto parameters = root->parameters();
  const auto &node_user_map = manager->node_users();
  MS_LOG(INFO) << "Start to process the global norm";

  for (auto &parameter : parameters) {
    int64_t dev_num = 0;
    if (!ParameterRequireGrad(parameter)) {
      continue;
    }
    auto mirror_node = GetMirrorOp(node_user_map, parameter);
    auto device_num_ptr = GetAttrsFromAnfNode(mirror_node, DEV_NUM);
    if (device_num_ptr && device_num_ptr->isa<Int64Imm>()) {
      dev_num = GetValue<int64_t>(device_num_ptr);
    }
    InsertDivAndAllReduceForNorm(node_user_map, parameter, LongToUint(dev_num));
  }
}

static void MoveMicroMirrorOutCallFunc(const FuncGraphPtr &root) {
  AnfNodePtr ret_after = root->get_return();
  MS_EXCEPTION_IF_NULL(ret_after);
  auto all_nodes = TopoSort(ret_after, SuccDeeperSimple);
  auto manager = root->manager();
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimMirrorMicroStep)) {
      continue;
    }
    auto micro_mirror = node->cast<CNodePtr>();
    auto param_anf_node = GetInputNodeWithFilter(micro_mirror, [&](const CNodePtr &cnode) {
      bool filter = IsPrimitiveCNode(cnode, prim::kPrimMirrorMicroStep) || IsPrimitiveCNode(cnode, prim::kPrimLoad) ||
                    IsPrimitiveCNode(cnode, prim::kPrimDepend);
      return std::make_pair(filter, 1);
    });
    if (!param_anf_node->isa<Parameter>()) {
      continue;
    }
    auto param = param_anf_node->cast<ParameterPtr>();
    if (param->has_default()) {
      continue;
    }
    auto sub_func_graph = param_anf_node->func_graph();
    auto call_cnodes_map = sub_func_graph->func_graph_cnodes_index();
    auto sub_graph_parameters = sub_func_graph->parameters();
    auto curr_param_iter = std::find(sub_graph_parameters.begin(), sub_graph_parameters.end(), param_anf_node);
    if (curr_param_iter == sub_graph_parameters.end()) {
      MS_LOG(EXCEPTION) << "Cannot find param " << param_anf_node->DebugString() << " in current sub_graph";
    }
    size_t curr_param_index = static_cast<size_t>(curr_param_iter - sub_graph_parameters.begin());
    AnfNodePtr call_nodes_common_param_input = nullptr;
    FuncGraphPtr call_nodes_func_graph = nullptr;
    for (const auto &node_pair : call_cnodes_map) {
      if (!node_pair.first->first->isa<CNode>() || node_pair.first->second > 0) {
        continue;
      }
      auto cnode = node_pair.first->first->cast<CNodePtr>();
      call_nodes_func_graph = cnode->func_graph();
      auto cnode_input = cnode->input(curr_param_index + 1);
      if (!call_nodes_common_param_input) {
        call_nodes_common_param_input = cnode_input;
      }
      if (call_nodes_common_param_input != cnode_input) {
        call_nodes_common_param_input = nullptr;
        break;
      }
    }
    if (!call_nodes_common_param_input || !call_nodes_func_graph) {
      continue;
    }
    // Insert new MicroMirror in root func
    if (!IsPrimitiveCNode(call_nodes_common_param_input, prim::kPrimMirrorMicroStep)) {
      auto new_mirror_node =
        NewMicroMirrorPrimByMicroMirror(call_nodes_func_graph, micro_mirror, call_nodes_common_param_input);
      for (const auto &node_pair : call_cnodes_map) {
        if (!node_pair.first->first->isa<CNode>() || node_pair.first->second > 0) {
          continue;
        }
        manager->SetEdge(node_pair.first->first, curr_param_index + 1, new_mirror_node);
      }
    }

    // Remove MicroMirror in call_func
    (void)manager->Replace(micro_mirror, micro_mirror->input(kIndex1));
  }
}

static void MergeMicroMirrorForSharedParameter(const FuncGraphPtr &root) {
  AnfNodePtr ret_after = root->get_return();
  MS_EXCEPTION_IF_NULL(ret_after);
  auto all_nodes = TopoSort(ret_after, SuccDeeperSimple);
  auto manager = root->manager();
  std::unordered_map<ParameterPtr, std::vector<CNodePtr>> param_mirror_map;
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimMirrorMicroStep)) {
      continue;
    }
    auto micro_mirror = node->cast<CNodePtr>();
    auto param_anf_node = GetInputNodeWithFilter(micro_mirror, [&](const CNodePtr &cnode) {
      bool filter = IsPrimitiveCNode(cnode, prim::kPrimMirrorMicroStep) || IsPrimitiveCNode(cnode, prim::kPrimLoad) ||
                    IsPrimitiveCNode(cnode, prim::kPrimDepend) ||
                    IsPrimitiveCNode(cnode, prim::kPrimMicroStepAllGather);
      return std::make_pair(filter, 1);
    });
    if (!param_anf_node->isa<Parameter>()) {
      continue;
    }
    auto param = param_anf_node->cast<ParameterPtr>();
    param_mirror_map[param].push_back(micro_mirror);
  }
  for (const auto &parm_pair : param_mirror_map) {
    if (parm_pair.second.size() <= 1) {
      continue;
    }
    MS_LOG(INFO) << "Parameter " << parm_pair.first->name() << " still has multi mirror user, merge those mirror.";
    auto mirror0 = parm_pair.second.front();
    for (size_t i = 1; i < parm_pair.second.size(); ++i) {
      (void)manager->Replace(parm_pair.second[i], mirror0);
    }
  }
}

static void BroadcastMultiOutputs(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager, const Group &group) {
  auto output = root->get_return()->input(1)->cast<CNodePtr>();
  auto output_abstract = output->abstract();
  MS_EXCEPTION_IF_NULL(output_abstract);
  auto abstract_tuple = output_abstract->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(abstract_tuple);
  auto abstract_list = abstract_tuple->elements();

  AnfNodePtrList make_tuple_input = {NewValueNode(prim::kPrimMakeTuple)};
  for (size_t i = 0; i < abstract_list.size(); i++) {
    auto abstract = abstract_list[i];
    MS_EXCEPTION_IF_NULL(abstract);

    // TupleGetItem
    auto idx = NewValueNode(SizeToLong(i));
    CNodePtr tuple_getitem = root->NewCNode({NewValueNode(prim::kPrimTupleGetItem), output, idx});
    MS_EXCEPTION_IF_NULL(tuple_getitem);
    tuple_getitem->set_abstract(abstract);

    // Depend: prevent disorder and CSE
    if (i > 0) {
      tuple_getitem = root->NewCNode({NewValueNode(prim::kPrimDepend), tuple_getitem, make_tuple_input[i]});
      MS_EXCEPTION_IF_NULL(tuple_getitem);
      tuple_getitem->set_abstract(abstract);
    }

    // Allreduce
    CNodePtr allreduce = root->NewCNode({NewValueNode(prim::kPrimAllReduce), tuple_getitem});
    MS_EXCEPTION_IF_NULL(allreduce);
    allreduce->set_abstract(abstract);
    common::AnfAlgo::SetNodeAttr(OP, MakeValue(REDUCE_OP_SUM), allreduce);
    common::AnfAlgo::SetNodeAttr(GROUP, MakeValue(group.name()), allreduce);
    // Disable GE allreduce fusion.
    common::AnfAlgo::SetNodeAttr(FUSION, MakeValue(static_cast<int64_t>(0)), allreduce);

    make_tuple_input.push_back(allreduce);
  }

  CNodePtr make_tuple_node = root->NewCNode(make_tuple_input);
  MS_EXCEPTION_IF_NULL(make_tuple_node);
  make_tuple_node->set_abstract(abstract_tuple);
  (void)manager->Replace(output, make_tuple_node);
}

static void BroadcastLastResult(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager) {
  auto stage_num = parallel::ParallelContext::GetInstance()->pipeline_stage_split_num();
  auto pipeline_result_broadcast = parallel::ParallelContext::GetInstance()->pipeline_result_broadcast();
  if (IsTraining(manager) || stage_num <= 1 || pipeline_result_broadcast == false) {
    return;
  }

  std::vector<int64_t> rank_list = g_device_manager->GetDeviceListBetweenStage();
  Group group;
  if (g_device_manager->CreateGroup(rank_list, &group) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Create communication group between all pipeline stages failed, the rank_list is: "
                      << rank_list;
  }

  auto return_node = root->get_return();
  const auto &abstract = return_node->abstract();
  if (abstract->isa<abstract::AbstractTuple>()) {
    return BroadcastMultiOutputs(root, manager, group);
  }

  InsertAllReduceToNodeInput(return_node, group.name(), PARALLEL_RESULT_BROADCAST);
  return_node->input(1)->set_abstract(abstract);
}

static void RecordFlopsOriginShape(const FuncGraphManagerPtr &mng) {
  for (const auto &each_graph : mng->func_graphs()) {
    std::list<CNodePtr> graph_orders = each_graph->GetOrderedCnodes();
    std::vector<CNodePtr> origin_nodes_topological(graph_orders.cbegin(), graph_orders.cend());
    for (const auto &node : origin_nodes_topological) {
      if (IsPrimitiveCNode(node, prim::kPrimConv2D) || IsPrimitiveCNode(node, prim::kPrimBatchMatMul) ||
          IsPrimitiveCNode(node, prim::kPrimMatMul)) {
        node->AddPrimalAttr(kAttrOriginOutputShape, MakeValue(node->abstract()->GetShapeTrack()->GetShapeVector()));
        node->AddPrimalAttr(
          kAttrOriginInputShapes,
          MakeValue<std::vector<ShapeVector>>({node->input(kIndex1)->abstract()->GetShapeTrack()->GetShapeVector(),
                                               node->input(kIndex2)->abstract()->GetShapeTrack()->GetShapeVector()}));
      } else if (IsPrimitiveCNode(node, prim::kPrimFlashAttentionScore)) {
        node->AddPrimalAttr(
          kAttrOriginInputShapes,
          MakeValue<std::vector<ShapeVector>>({node->input(kIndex1)->abstract()->GetShapeTrack()->GetShapeVector(),
                                               node->input(kIndex2)->abstract()->GetShapeTrack()->GetShapeVector()}));
      }
    }
  }
}

bool IsVirtualDatasetDynamicShape(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto all_nodes = TopoSort(func_graph->get_return());
  for (const auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim == nullptr) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(prim);
    if (prim->name() == VIRTUAL_DATA_SET) {
      MS_LOG(INFO) << "VIRTUAL_DATA_SET: " << cnode->DebugString();
      for (size_t i = 1; i < cnode->inputs().size(); ++i) {
        auto input_node = cnode->input(i);
        auto base_shape = input_node->Shape();
        MS_EXCEPTION_IF_NULL(base_shape);
        std::vector<int64_t> shape_vec = base_shape->GetShapeVector();
        MS_LOG(INFO) << "VIRTUAL_DATA_SET: " << node->fullname_with_scope() << ", shape:" << shape_vec;
        if (std::find(shape_vec.begin(), shape_vec.end(), -1) != shape_vec.end()) {
          return true;
        }
      }
    }
  }
  return false;
}

static void HandleSilentCheck(const FuncGraphPtr &root, const FuncGraphManagerPtr &mng) {
  auto env = common::GetEnv(NPU_ASD_ENABLE);
  if (env != kSilentCheckEnvEnable) {
    return;
  }
  auto sdc = std::make_shared<SilentCheck>(root, mng);
  if (sdc == nullptr) {
    MS_LOG(EXCEPTION) << "The silent check env got nullptr;";
  }
  sdc->GetLossScale();
  sdc->ModifySilentCheckOps();
}

static void ParallelPartProcess(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root,
                                const FuncGraphManagerPtr &manager) {
  ReshapeInit(all_nodes);

  SetCastForParamNotRecompute(all_nodes);

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

  HandleCameAndAdaFactorOpt(root, all_nodes, manager);

  InsertUniformRealForTaggedNodes(manager, all_nodes);

  auto adasum_param_tensor_layout_map = AdaSumParamTensorLayout(root);
  bool is_apply_adasum = HandleAdaSum(root, all_nodes, &adasum_param_tensor_layout_map);

  if (MergeEntireShapeForDynamic(root) != Status::SUCCESS) {
    MS_LOG(EXCEPTION) << "Merge entire shape for dynamic shape failed.";
  }

  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto is_pp_interleave = parallel_context->pipeline_interleave();
  std::shared_ptr<PipelinePostProcess> pipeline_processor;
  auto pipeline_stages = ParallelContext::GetInstance()->pipeline_stage_split_num();
  if (pipeline_stages > 1 && is_pp_interleave) {
    pipeline_processor =
      std::make_shared<PipelinePostProcess>(manager, g_device_manager->stage_id(), pipeline_stages, root);
    pipeline_processor->Init(all_nodes);
    pipeline_processor->ModifySendRecvAttr(all_nodes);
  }
  // ForwardCommunication BackwardCommunication TensorRedistribution
  ParallelCommunication(root, all_nodes, manager);
  SplitNotParallelCareOpsInterleaved(root);
  EraseVirtualConverter(root);
  if (is_apply_adasum) {
    HandleMirrorInAdaSum(root, &adasum_param_tensor_layout_map);
  }

  if (pipeline_stages > 1 && is_pp_interleave) {
    MS_EXCEPTION_IF_NULL(pipeline_processor);
    pipeline_processor->GraphPartition(all_nodes);
    pipeline_processor->ElimGraphStage();
    pipeline_processor->ModifyParameterList();
  }

  // save strategy as checkpoint for multi-train
  auto all_nodes_after_pp = TopoSort(root->get_return(), SuccDeeperSimple);
  if (StrategyCheckpoint::GetInstance().SaveCheckPointOn()) {
    CheckpointStrategy(all_nodes_after_pp, root);
  }
  auto comm_group = FindCommonMirrorGroup(root);
  StrategyCheckpoint::GetInstance().set_common_mirror_group(comm_group);
  MoveMicroMirrorOutCallFunc(root);
  HandleGlobalNormScale(root, manager);
  if (pipeline_stages > 1 && is_pp_interleave) {
    pipeline_processor->HandleSendParam();
    MarkForwardCNode(root);
  }
  MergeMicroMirrorForSharedParameter(root);
  // Insert TensorToTuple for FlashAttentionScore if input actual_seq_len is tensor
  PostProcessActualSeqLenInputForFlashAttentionScore(root, all_nodes);
  return;
}

bool StepParallel(const FuncGraphPtr &root, const opt::OptimizerPtr &optimizer) {
#if defined(__linux__) && defined(WITH_BACKEND)
  if (ps::PSContext::instance()->is_server() || ps::PSContext::instance()->is_scheduler()) {
    return false;
  }
#endif
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  std::string parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  HandleDataParallel();
  FuncGraphManagerPtr manager;
  pipeline::ResourceBasePtr res;
  if (optimizer == nullptr) {
    manager = root->manager();
    res = std::make_shared<pipeline::Resource>();
    res->set_manager(manager);
  } else {
    res = optimizer->resource();
    MS_EXCEPTION_IF_NULL(res);
    manager = res->manager();
  }

  MS_EXCEPTION_IF_NULL(manager);
  auto pipeline_stages = ParallelContext::GetInstance()->pipeline_stage_split_num();
  if (IsTraining(manager)) {
    root->set_flag(kTraining, true);
  }
  // assume no change to graph
  bool changes = false;
  // control whether use model_parallel mode
  if (!IsAutoParallelCareGraph(root) || (root->has_flag(SEMI_AUTO_PARALLEL_RUN_ONCE_ONLY)) || HasNestedMetaFg(root)) {
    if (!root->has_flag(CHECK_SET_STRATEGY_VALID_ONCE_ONLY)) {
      MS_LOG(INFO) << "Strategies would be ignored in " << parallel_mode
                   << ", shard() only valid in [semi_]auto_parallel.";
      root->set_flag(CHECK_SET_STRATEGY_VALID_ONCE_ONLY, true);
    }
    ReorderForPipelineSplit(root, manager, pipeline_stages);
    ReorderForGradAccumulation(root, manager);
    return changes;
  }

  MSLogTime msTime;
  msTime.Start();
  DumpGraph(root, std::string(STEP_PARALLEL_BEGIN));
  RecordFlopsOriginShape(manager);
  AnfNodePtr ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  std::reverse(all_nodes.begin(), all_nodes.end());
  bool merged = MergeConcatSlice(all_nodes, manager);
  if (merged) {
    all_nodes = TopoSort(ret, SuccDeeperSimple);
  }
  if (pipeline_stages <= 1 && parallel_mode != kAutoParallel && ParallelInit() != SUCCESS) {
    MS_LOG(EXCEPTION) << "Parallel init failed";
  }

  // Insert TupleToTensor for FA if actual_seq_len input is tuple type.
  PreProcessActualSeqLenInputForFlashAttentionScore(root, all_nodes);

  MicroBatchPreProcess(root, manager, all_nodes);
  // mark the forward cnodes, parallel only care these nodes
  MarkForwardCNode(root);
  HandleSilentCheck(root, manager);
  // tag dynamic shape graph
  TagDynamicShapeFuncGraph(root);
  UpdateMicroBatchInterleavedStatus(all_nodes);
  if (parallel_mode != kAutoParallel) {
    TOTAL_OPS = 0;
    ExceptionIfHasCommunicationOp(all_nodes);

    if (IsInsertVirtualOutput(root)) {
      InsertVirtualOutput(root, all_nodes);
      AnfNodePtr ret_after = root->get_return();
      MS_EXCEPTION_IF_NULL(ret_after);
      all_nodes = TopoSort(ret_after, SuccDeeperSimple);
    }

    // extract shape and strategy, set operator_info
    ExtractInformation(all_nodes);
  }

  ParallelPartProcess(all_nodes, root, manager);
  BroadcastLastResult(root, manager);
  MicroBatchPostProcess(root, all_nodes);
  DumpGraph(root, std::string(STEP_PARALLEL_END));

  // step parallel only run once
  root->set_flag(SEMI_AUTO_PARALLEL_RUN_ONCE_ONLY, true);
  // Keep all func graph for parallel before save result.
  SetReserved(root);
  res->SetResult(pipeline::kStepParallelGraph, root);

  // in auto parallel mode, no need to check if strategies set
  root->set_flag(CHECK_SET_STRATEGY_VALID_ONCE_ONLY, true);

  msTime.End();
  uint64_t time = msTime.GetRunTimeUS();
  MS_LOG(INFO) << "Now leaving step parallel, used time: " << time << " us";
  return changes;
}
}  // namespace parallel
}  // namespace mindspore
