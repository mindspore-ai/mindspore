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

#include "frontend/parallel/parallel_processor.h"

#include <cinttypes>
#include <algorithm>
#include <chrono>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <queue>

#include "primitive/sequence_ops.h"
#include "primitive/other_ops.h"
#include "primitive/array_ops.h"
#include "primitive/structure_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/arithmetic_ops.h"
#include "frontend/operator/ops.h"
#include "include/frontend/optimizer/optimizer.h"
#include "include/frontend/parallel/graph_costmodel.h"
#include "frontend/parallel/parallel_processor_context.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/ops_info/gather_info.h"
#include "frontend/parallel/ops_info/reshape_info.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "frontend/parallel/graph_util/parallel_tensordump.h"
#include "frontend/parallel/graph_util/pipeline_split_utils.h"
#include "frontend/parallel/node_check.h"
#include "frontend/parallel/parallel_node_check.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/ops_info/matmul_info.h"
#include "frontend/parallel/dynamic_shape/dynamic_shape.h"
#include "include/frontend/parallel/tensor_layout/tensor_transform.h"
#include "frontend/parallel/strategy_utils.h"
#include "frontend/parallel/strategy_loader.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "include/utils/parallel_context.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_v.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/pass_manager/helper.h"

using mindspore::tensor::Tensor;

namespace mindspore {
namespace parallel {
namespace {
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
    MS_EXCEPTION_IF_NULL(sens_cnode);
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
    MS_EXCEPTION_IF_NULL(expect_anonymous_cnode);
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
      MS_LOG_WITH_NODE(EXCEPTION, sens_cnode) << "Sens can't find the corresponding graph.";
    }
    auto func_graph = GetValueNode<FuncGraphPtr>(expect_j_cnode->input(1));
    auto loss_node_info = FindLossCNode(func_graph);
    if (loss_node_info.loss_node == nullptr) {
      MS_LOG(WARNING) << "Can not find the loss cnode";
      continue;
    }

    if (loss_node_info.has_make_tuple) {
      auto sens_cnode_input = sens_cnode->input(kIndex1);
      MS_EXCEPTION_IF_NULL(sens_cnode_input);
      if (IsPrimitiveCNode(sens_cnode_input, prim::kPrimMakeTuple)) {
        sens_cnode = sens_cnode_input->cast<CNodePtr>();
        MS_LOG(INFO) << "Change sens cnode to its input, which is primitive MakeTuple";
      } else {
        MS_LOG_WITH_NODE(EXCEPTION, sens_cnode_input)
          << "Can not find the loss cnode for multi output, find node is " << sens_cnode_input->DebugString();
      }
      auto loss_cnode = loss_node_info.loss_node;
      MS_EXCEPTION_IF_NULL(loss_cnode);
      MS_EXCEPTION_IF_NULL(sens_cnode);
      if (sens_cnode->size() != loss_cnode->size()) {
        MS_LOG_WITH_NODE(EXCEPTION, sens_cnode) << "for multi output, sens cnode size is not equal to loss cnode size";
      }
      for (size_t i = 1; i < sens_cnode->size(); ++i) {
        auto loss_input_node = GetInputNodeBySkipDependAndUpdateState(loss_cnode, i);
        auto loss_input_cnode = loss_input_node->cast<CNodePtr>();
        // when return multi losses, one or multi of them may not be involved in the calculation and will be the initial
        // value 0, so we need to ignore them.
        if (loss_input_cnode == nullptr) {
          continue;
        }
        LossNodeInfo real_loss_node_info;
        real_loss_node_info.loss_node = loss_input_cnode;
        real_loss_node_info.dout_index = 0;
        (void)sens_loss_pairs.emplace_back(std::make_pair(sens_cnode, real_loss_node_info));
      }
    } else {
      (void)sens_loss_pairs.emplace_back(std::make_pair(sens_cnode, loss_node_info));
    }
  }
  return sens_loss_pairs;
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
    MS_LOG_WITH_NODE(EXCEPTION, node) << "size of OperatorVector and OutPutInfoVector must be the same!";
  }

  auto pos_u = LongToSize(pos);
  if (pos_u >= node->size()) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "InsertRedistribution:pos can't be larger than node's inputs'size";
  }
  for (size_t index = 0; index < (redistribution_oplist_ptr->first).size(); ++index) {
    // Create new node
    AnfNodePtr target_node = node->input(pos_u);
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
    InsertNode(op, node, pos_u, target_node, func_graph, instance_name, "", nullptr, tensor_redistribution);
    if ((redistribution_oplist_ptr->second)[index].first) {
      target_node = node->input(pos_u);
      MS_EXCEPTION_IF_NULL(target_node);
      (void)InsertMakeTuple(target_node, (redistribution_oplist_ptr->second)[index].second, func_graph);
    }
  }
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
    MS_LOG_WITH_NODE(EXCEPTION, node) << "InferSensRedistribution failed cause inputs shape is empty.";
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
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Create tensor layout for Sens failed.";
  }

  // Infer Redistribution op list for stand alone and loss layout.
  RankList dev_list = g_device_manager->GetDeviceListInThisStage();
  if (tensor_redistribution.Init(stand_alone_layout, loss_layout, dev_list) == FAILED) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Redistribution for Sens init failed.";
  }
  RedistributionOpListPtr sens_redistribution_list = tensor_redistribution.InferTensorRedistributionOperatorList();
  MS_EXCEPTION_IF_NULL(sens_redistribution_list);

  return sens_redistribution_list;
}

static void InsertGetTensorSliceOp(const Operator &op, const CNodePtr &node, const FuncGraphPtr &func_graph,
                                   int64_t pos, const std::string &instance_name) {
  if (func_graph == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "InsertGetTensorSliceOp: the graph is null, the instance name is "
                                      << instance_name;
  }

  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (pos >= SizeToLong(node->size())) {
    MS_LOG_WITH_NODE(EXCEPTION, node)
      << "InsertGetTensorSliceOp: pos can't be larger than node's inputs'size, the instance name is " << instance_name;
  }
  // Create new node
  AnfNodePtr pre_node = node->input(LongToSize(pos));
  MS_EXCEPTION_IF_NULL(pre_node);
  InsertNode(op, node, LongToSize(pos), pre_node, func_graph, instance_name);
}

static void SplitSens(const CNodePtr &grad_sens_node, const TensorLayout &loss_grad_layout) {
  MS_EXCEPTION_IF_NULL(grad_sens_node);
  if (grad_sens_node->size() <= 1) {
    MS_LOG_WITH_NODE(EXCEPTION, grad_sens_node) << "The size of grad sens node is smaller than 2";
  }
  AnfNodePtr sens_tensor_node = grad_sens_node->input(1);
  MS_EXCEPTION_IF_NULL(sens_tensor_node);
  Shapes sens_shapes = GetNodeShape(sens_tensor_node);
  if (sens_shapes.size() != 1) {
    MS_LOG_WITH_NODE(EXCEPTION, grad_sens_node) << "GetNodeShape for sens_tensor_node, output size is not 1";
  }
  // If the shape of sens tensor is [] or [1], no need to split it.
  Shape sens_shape = sens_shapes[0];
  if (sens_shape.empty() || ((sens_shape.size() == 1) && (sens_shape[0] == 1))) {
    if (sens_tensor_node->isa<Parameter>()) {
      auto sens_tensor_param = sens_tensor_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(sens_tensor_param);
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
    MS_LOG_WITH_NODE(EXCEPTION, grad_sens_node)
      << "The shape of sens is not equal to loss output, it is unsupported now. Sens shape is "
      << ShapeToString(sens_shape) << ", loss shape is " << ShapeToString(loss_shape);
  }
  MS_LOG(INFO) << "The shape of sens is " << ShapeToString(sens_shape) << ", split it.";

  if (!IsValueNode<Tensor>(sens_tensor_node)) {
    if (sens_tensor_node->isa<Parameter>()) {
      MS_LOG(DEBUG) << "loss layout " << loss_grad_layout.ToString();
      AbstractBasePtr abstract = sens_tensor_node->abstract();
      MS_EXCEPTION_IF_NULL(abstract);
      auto slice_shape = loss_grad_layout.base_slice_shape().array();
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
    bool is_dynamic = IsForwardDynamicShape();
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
    MS_LOG_WITH_NODE(EXCEPTION, grad_sens_node)
      << "The type of sens node is not Tensor or Parameter or CNode, it is unsupported now.";
  }

  // Use _GetTensorSlice operator to split the sens tensor
  FuncGraphPtr func_graph = grad_sens_node->func_graph();  // only cnode can get the graph
  MS_EXCEPTION_IF_NULL(func_graph);
  Operator op = CreateGetTensorSliceOp(loss_grad_layout);
  InsertGetTensorSliceOp(op, grad_sens_node, func_graph, 1, SPLIT_SENS);
}

static void StepSplitSens(const std::pair<CNodePtr, LossNodeInfo> &sens_loss_pair) {
  CNodePtr sens_node = sens_loss_pair.first;
  auto loss_node = sens_loss_pair.second;
  auto loss_grad_layout = GetLossNodeGradOutputLayout(loss_node);
  if (!loss_grad_layout.empty()) {
    SplitSens(sens_node, loss_grad_layout[0]);
  }
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

void InsertRedistributionForMicroInterleaved(const TensorRedistributionPtr &tensor_redistribution,
                                             const std::pair<AnfNodePtr, int64_t> &node_pair,
                                             const FuncGraphPtr &func_graph, const CNodePtr &attr_cnode,
                                             const AnfNodePtr &real_pre_node) {
  auto redistribution_oplist_ptr_vector = tensor_redistribution->InferTensorRedistributionOperatorVirtualGraphs();
  auto next_cnode = node_pair.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(next_cnode);
  auto next_cnode_index = node_pair.second;
  // create VirtualConverterBeginNode
  MS_EXCEPTION_IF_NULL(real_pre_node);
  auto virtual_converter_begin =
    CreateVirtualConverterBeginNode(real_pre_node, redistribution_oplist_ptr_vector.size());
  std::vector<CNodePtr> tuple_get_item_vector;
  int64_t fine_grain_block_index = -1;
  if (IsPrimitiveCNode(real_pre_node) &&
      ((GetCNodePrimitive(real_pre_node) != nullptr) &&
       (GetCNodePrimitive(real_pre_node)->HasAttr(kAttrFineGrainedInterleavedBlockIndex)))) {
    fine_grain_block_index =
      GetValue<int64_t>(GetCNodePrimitive(real_pre_node)->GetAttr(kAttrFineGrainedInterleavedBlockIndex));
  }
  if (IsPrimitiveCNode(next_cnode) &&
      ((GetCNodePrimitive(next_cnode) != nullptr) &&
       (GetCNodePrimitive(next_cnode)->HasAttr(kAttrFineGrainedInterleavedBlockIndex)))) {
    fine_grain_block_index =
      GetValue<int64_t>(GetCNodePrimitive(next_cnode)->GetAttr(kAttrFineGrainedInterleavedBlockIndex));
  }
  auto interleaved_num = ParallelContext::GetInstance()->fine_grained_micro_interleaved_size();
  auto stage_begin_rank = g_device_manager->stage_device_num() * g_device_manager->stage_id() * interleaved_num;
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
    MS_EXCEPTION_IF_NULL(virtual_converter_end_prim);
    virtual_converter_end_prim->AddAttr(RECOMPUTE_COMM_OP, out_recompute_comm_op_attr);
  }
  if (fine_grain_block_index >= 0) {
    auto virtual_converter_end_prim = GetCNodePrimitive(virtual_converter_end);
    MS_EXCEPTION_IF_NULL(virtual_converter_end_prim);
    virtual_converter_end_prim->AddAttr(kAttrFineGrainedInterleavedBlockIndex,
                                        MakeValue<int64_t>(fine_grain_block_index));
  }
  std::vector<std::vector<std::vector<int64_t>>> ag_group_ranks_vectors;

  for (size_t i = 0; i < redistribution_oplist_ptr_vector.size(); ++i) {
    auto redistribution_oplist_ptr = redistribution_oplist_ptr_vector[i];
    auto virtual_rank = tensor_redistribution->GetVirtualRankList().at(i);
    if (!tensor_redistribution->IsAssembledStaticShape()) {
      redistribution_oplist_ptr = TensorTransform::GetInstance()->OptimizeTensorRedistributionOperatorList(
        redistribution_oplist_ptr, tensor_redistribution->input_shape(), virtual_rank);
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
        std::set<int64_t> new_group_ranks_set;
        std::transform(group_ranks.begin(), group_ranks.end(),
                       std::inserter(new_group_ranks_set, new_group_ranks_set.begin()), [&](int64_t g_rank) {
                         return int64_t((g_rank - stage_begin_rank) / interleaved_num) +
                                stage_begin_rank / interleaved_num;
                       });
        MS_EXCEPTION_IF_NULL(GetCNodePrimitive(virtual_converter_end));
        if (new_group_ranks_set.size() <= group_ranks.size() &&
            GetCNodePrimitive(virtual_converter_end)->HasAttr(RECOMPUTE_COMM_OP)) {
          GetCNodePrimitive(virtual_converter_end)->EraseAttr(RECOMPUTE_COMM_OP);
        }
        ag_group_ranks_vector.push_back(group_ranks);
      }
    }
    ag_group_ranks_vectors.push_back(ag_group_ranks_vector);
    InsertRedistribution(redistribution_oplist_ptr, virtual_converter_end, func_graph, i + 1, attr_cnode,
                         tensor_redistribution);
  }
  ConvertInterleaveAllGatherToConcat(func_graph, virtual_converter_end, ag_group_ranks_vectors);
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
      MS_LOG_WITH_NODE(EXCEPTION, pre_cnode)
        << "The index out of range. Node: " << pre_node->DebugString() << " index: " << tensor_info_pos
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

Status ObtainOutputTensorLayout(const OperatorInfoPtr &next_distribute_operator,
                                const std::pair<AnfNodePtr, int> &node_pair, const CNodePtr &next_cnode,
                                const bool &using_func_param_op_info, TensorLayout *tensorlayout_out) {
  bool next_dist_op_has_tuple = !next_distribute_operator->inputs_tensor_info_new().empty();
  if (next_dist_op_has_tuple) {
    auto next_inputs_tensor_info = using_func_param_op_info ? next_distribute_operator->outputs_tensor_info_new()
                                                            : next_distribute_operator->inputs_tensor_info_new();
    if (LongToSize(node_pair.second - 1) >= next_inputs_tensor_info.size()) {
      MS_LOG(INFO) << "The index is out of range, the index is " << (node_pair.second - 1) << ", the vector size is "
                   << next_inputs_tensor_info.size() << ", next node is " << next_cnode->DebugString();
      return FAILED;
    }
    auto tensorinfo_out_ptr = next_inputs_tensor_info[LongToSize(node_pair.second - 1)];
    TensorInfo tensorinfo_out = tensorinfo_out_ptr->GetValue();
    *tensorlayout_out = tensorinfo_out.tensor_layout();
    return SUCCESS;
  }
  auto next_inputs_tensor_info = using_func_param_op_info ? next_distribute_operator->outputs_tensor_info()
                                                          : next_distribute_operator->inputs_tensor_info();
  size_t out_layout_index = LongToSize(node_pair.second - 1);
  if (out_layout_index >= next_inputs_tensor_info.size()) {
    MS_LOG(INFO) << "The index is out of range, the index is " << out_layout_index << ", the vector size is "
                 << next_inputs_tensor_info.size() << ", next node is " << next_cnode->DebugString();
    return FAILED;
  }
  TensorInfo tensorinfo_out = next_inputs_tensor_info[out_layout_index];
  *tensorlayout_out = tensorinfo_out.tensor_layout();
  return SUCCESS;
}

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

static void InserForwardCommunicationForSingleOutput(const CNodePtr &node, const OperatorVector &forward_op,
                                                     const ForwardOpList &forward_op_list) {
  MS_EXCEPTION_IF_NULL(node);

  // step1:get graph manager distribute_operator
  const FuncGraphPtr &func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &uses_set = manager->node_users()[node];

  // step2: get node to insert forward communication
  CNodePtrList node_to_insert = {};
  bool has_tuple_getitem = false;
  for (auto &uses_pair : uses_set) {
    auto uses_cnode = uses_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(uses_cnode);
    if (!IsValueNode<Primitive>(uses_cnode->input(0))) {
      break;
    }

    // tuple_get_item node
    PrimitivePtr value_node_prim = GetValueNode<PrimitivePtr>(uses_cnode->input(0));
    MS_EXCEPTION_IF_NULL(value_node_prim);
    if (value_node_prim->name() == prim::kPrimTupleGetItem->name()) {
      has_tuple_getitem = true;
      node_to_insert.push_back(uses_cnode);
    }
  }

  if (!has_tuple_getitem) {
    node_to_insert.push_back(node);
  }

  // get forward_op_list
  ForwardOpList op_comm_list = forward_op_list.empty() ? ForwardOpList{forward_op} : forward_op_list;

  // step2: traverse op_list and insert node
  CNodePtr forward_node;
  for (size_t out_idx = 0; out_idx < op_comm_list.size(); ++out_idx) {
    ForwardOp forward_op_idx = op_comm_list[out_idx];  // op_list of output_idx, {op0, op1, op2}
    std::reverse(forward_op_idx.begin(), forward_op_idx.end());
    FwdCommDumpHandlerPtr fwd_dump_handler =
      std::make_shared<FwdCommunicationParallelTensorDumpHandler>(node_to_insert[out_idx]);
    fwd_dump_handler->CollectDumpNodes(node_to_insert[out_idx], !forward_op_list.empty());
    for (size_t index = 0; index < forward_op_idx.size(); ++index) {
      std::string instance_name_base = FORWARD_OP;
      std::string instance_name = instance_name_base + "_" + CreateInstanceName(node, index);
      std::vector<AnfNodePtr> forward_input =
        CreateInput(forward_op_idx[index], node_to_insert[out_idx], instance_name);
      SetAllReduceRecomputeFlag(forward_input, node_to_insert[out_idx]);
      forward_node = func_graph->NewCNode(forward_input);  // using NewCNode to create anfnode
      MS_EXCEPTION_IF_NULL(forward_node);
      ScopePtr scope = node->scope();
      MS_EXCEPTION_IF_NULL(scope);
      forward_node->set_scope(scope);
      forward_node->set_in_forward_flag(true);
      forward_node->AddPrimalAttr(kPrimalAttrForwardCommNodeUniqueId, MakeValue<std::string>(forward_node->UniqueId()));
      if (node_to_insert[out_idx]->HasPrimalAttr(MICRO)) {
        forward_node->AddPrimalAttr(MICRO, node_to_insert[out_idx]->GetPrimalAttr(MICRO));
      }
      forward_input[0]->set_scope(scope);
      (void)manager->Replace(node_to_insert[out_idx], forward_node);  // using Replace function to insert node
    }
    (void)fwd_dump_handler->MakeInModeBwdHookBeforeFwdComm();
    (void)fwd_dump_handler->MakeOutModeDumpBeforeFwdComm();
  }
}

static void ForwardCommunicationForMultiOut(const CNodePtr &node, const OperatorVector &forward_op,
                                            const ForwardOpList &forward_op_list) {
  MS_EXCEPTION_IF_NULL(node);

  // step1: get graph manager distribute_operator
  const FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  const FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  // step2: get_tuple_item from outputs node of distributed operator
  // insert tuple_get_item cnode to get output_idx
  const size_t outputs_num = forward_op_list.size();
  const std::shared_ptr<AnfNode> node_ = node->cast<AnfNodePtr>();
  MS_EXCEPTION_IF_NULL(node_);
  std::vector<AnfNodePtr> tuple_outputs;
  for (size_t output_idx = 0; output_idx < outputs_num; ++output_idx) {
    CNodePtr node_output_idx = mindspore::opt::CreatTupleGetItemNode(func_graph, node_, output_idx);
    const std::shared_ptr<AnfNode> node_output_idx_ = node_output_idx->cast<AnfNodePtr>();
    MS_EXCEPTION_IF_NULL(node_output_idx_);
    tuple_outputs.push_back(node_output_idx_);
  }

  // step3: merge all tuple_get_item nodes into tuple_node
  CNodePtr forward_node;
  if (!tuple_outputs.empty()) {
    forward_node = mindspore::opt::CreateMakeTupleNode(func_graph, tuple_outputs);
    MS_EXCEPTION_IF_NULL(forward_node);
    MS_LOG_WITH_NODE(INFO, forward_node) << "Create MakeTuple Node: " << common::AnfAlgo::GetCNodeName(forward_node);
  }

  // step4: insert forward communication
  (void)manager->Replace(node, forward_node);  // using replace function to insert node
  ForwardOpList forward_op_list_reverse = forward_op_list;
  std::reverse(forward_op_list_reverse.begin(), forward_op_list_reverse.end());
  InserForwardCommunicationForSingleOutput(node, forward_op, forward_op_list_reverse);
}

// only used for InsertMirrorOps
static CNodePtr SkipTrivialNodesMoveUp(CNodePtr node) {
  MS_EXCEPTION_IF_NULL(node);
  while (True) {
    if (IsPrimitiveCNode(node, prim::kPrimLoad) || IsInTrivialNodeList(node) || IsInAllGatherNodeList(node)) {
      size_t index = IsPrimitiveCNode(node, prim::kPrimDumpGradient) ? kDumpGradientSkipIndex : 1;
      if (IsPrimitiveCNode(node->input(index), prim::kPrimMicroStepAllGather) &&
          !ParallelContext::GetInstance()->zero3()) {
        return node;
      }
      if (node->input(index)->isa<Parameter>()) {
        return node;
      }
      node = node->input(index)->cast<CNodePtr>();
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, node) << "The node " << node->fullname_with_scope()
                                        << " is a abnormal node in inserting mirror node.";
    }
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
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Unknown type.";
  }
  auto input_element_type = node_type->cast<mindspore::TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(input_element_type);
  auto type_id = input_element_type->type_id();
  if (!is_gradient_fp32_sync && type_id != kNumberTypeFloat32) {
    return false;
  }

  return true;
}

static bool NeedGradient(const ParameterPtr &param_ptr) {
  if (param_ptr->param_info() && param_ptr->param_info()->requires_grad()) {
    return true;
  }
  return false;
}

OperatorVector MirrorOpInOptShard(const ParameterPtr &param_ptr) {
  OperatorVector backward_op;
  std::string opt_shard_mirror_group;
  if (param_ptr->user_data<TensorLayout>()) {
    opt_shard_mirror_group = param_ptr->user_data<TensorLayout>()->opt_shard_mirror_group();
  }
  if (!opt_shard_mirror_group.empty()) {
    // mirror ops is covered in not fully use opt shard case
    uint32_t group_rank_size = 0;
    if (!CommManager::GetInstance().GetRankSize(opt_shard_mirror_group, &group_rank_size)) {
      MS_LOG(EXCEPTION) << "Got the group size from the group " << opt_shard_mirror_group << " failed";
    }
    backward_op = CreateMirrorOps(opt_shard_mirror_group, static_cast<size_t>(group_rank_size));
  } else if (ParallelContext::GetInstance()->zero3() &&
             ((param_ptr->user_data<TensorLayout>() != nullptr) &&
              (!param_ptr->user_data<TensorLayout>()->opt_shard_group().empty()))) {
    Group local_rank_group;
    (void)g_device_manager->CreateGroup({g_device_manager->global_rank()}, &local_rank_group);
    backward_op = CreateMirrorOps(local_rank_group.name(), 1);
  }
  return backward_op;
}

static void DoInsertMirrorOps(const FuncGraphPtr &root, const MirrorOps &mirror_ops, const CNodePtr &node) {
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto mirror_size = IsPrimitiveCNode(node, prim::kPrimSend) ? 1 : mirror_ops.size();

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
      if (!NeedGradient(param_ptr)) {
        MS_LOG(INFO) << param_name << " do not need gradient. Skip inserting mirror.";
        continue;
      }
      if (param_ptr->user_data<TensorLayout>()) {
        is_shared_param = param_ptr->user_data<TensorLayout>()->is_shared_param();
      }
      auto opt_shard_mirror = MirrorOpInOptShard(param_ptr);
      if (!opt_shard_mirror.empty()) {
        backward_op = opt_shard_mirror;
      }
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
      MS_LOG_WITH_NODE(EXCEPTION, node) << "backward_op size must be 1, real is " << backward_op.size();
    }
    auto op = backward_op[0];
    if (pre_node->cast<CNodePtr>() && (InsertMirrorBeforeCast(node, index) || is_shared_param)) {
      // assume Load is inserted next to parameter
      // skip Load moving up and insert mirror next to the parameter
      CNodePtr load_node = SkipTrivialNodesMoveUp(pre_node->cast<CNodePtr>());

      CNodePtr comm_op = nullptr;
      if (IsPrimitiveCNode(load_node, prim::kPrimMicroStepAllGather)) {
        InsertNode(op, node, index, pre_node, func_graph, mirror_op_name, param_name, root);
        comm_op = node->input(index)->cast<CNodePtr>();
      } else {
        InsertNode(op, load_node, 1, load_node->input(1), func_graph, mirror_op_name, param_name, root);
        comm_op = load_node->input(1)->cast<CNodePtr>();
      }
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

static bool CheckInsertMirrorOpsForNewShape(const MirrorOps &mirror_ops, const CNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimSend)) {
    return true;
  }
  constexpr size_t kSingleArgCNodeSize = 2;
  auto all_inputs = node->inputs();
  auto has_value_seq = std::all_of(all_inputs.begin() + 1, all_inputs.end(),
                                   [&node](const AnfNodePtr &input) { return IsValueSequence(input); });
  if ((node->size() == kSingleArgCNodeSize || IsSomePrimitiveList(node, INPUT_IS_TUPLE_OR_LIST_OPS)) && has_value_seq) {
    MS_LOG(INFO) << "Input is ValueList, skip it.";
    return false;
  }
  auto has_make_tuple_list = std::all_of(all_inputs.begin() + 1, all_inputs.end(), [&node](const AnfNodePtr &input) {
    return (AnfNodeIsPrimitive(input, MAKE_TUPLE) || AnfNodeIsPrimitive(input, MAKE_LIST));
  });
  if ((node->size() == kSingleArgCNodeSize || IsSomePrimitiveList(node, INPUT_IS_TUPLE_OR_LIST_OPS)) &&
      (has_make_tuple_list)) {
    MS_LOG(INFO) << "The mirror for " << GetPrimName(node) << " has handle by make_tuple node";
    return false;
  }
  return true;
}

static void InsertMirrorOps(const FuncGraphPtr &root, const MirrorOps &mirror_ops, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsSupportNewShapeBaseNode(node) && !CheckInsertMirrorOps(mirror_ops, node)) {
    return;
  }

  if (IsSupportNewShapeBaseNode(node) && !CheckInsertMirrorOpsForNewShape(mirror_ops, node)) {
    return;
  }

  DoInsertMirrorOps(root, mirror_ops, node);
}

static void InsertVirtualDivOp(const VirtualDivOp &virtual_div_op, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  size_t node_size = node->size();
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  static auto const kDropoutDoMaskInputNum = 2;
  if (IsSomePrimitive(node, DROPOUT_DO_MASK)) {
    MS_LOG(INFO) << "Handle dropout do mask, only insert the virtual div to input[0]";
    node_size = kDropoutDoMaskInputNum;
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

static CNodePtr CreateSliceForTensorList(const AnfNodePtr &node, const CNodePtr &next_node, int index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(next_node);
  if ((next_node->size() != kSizeTwo || index != 1) && !IsSupportNewShapeBaseNode(next_node)) {
    MS_LOG(INFO) << next_node->fullname_with_scope() << " Inputs must have only one input, get "
                 << (next_node->size() - 1) << " index should be 1, get " << index;
    return nullptr;
  }
  OperatorInfoPtr op_info = next_node->user_data<OperatorInfo>();
  MS_EXCEPTION_IF_NULL(op_info);

  std::vector<ValuePtr> inputs_values;
  MS_EXCEPTION_IF_NULL(node->cast<ValueNodePtr>());
  if (IsValueNode<ValueList>(node)) {
    MS_EXCEPTION_IF_NULL(node->cast<ValueNodePtr>()->value()->cast<ValueListPtr>());
    inputs_values = node->cast<ValueNodePtr>()->value()->cast<ValueListPtr>()->value();
  } else {
    MS_EXCEPTION_IF_NULL(node->cast<ValueNodePtr>()->value()->cast<ValueTuplePtr>());
    inputs_values = node->cast<ValueNodePtr>()->value()->cast<ValueTuplePtr>()->value();
  }
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  FuncGraphPtr func_graph = next_node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  TensorInfoBasePtr new_tensor_infos = nullptr;
  if (op_info->inputs_tensor_info_new().empty()) {
    if (inputs_values.size() != op_info->inputs_tensor_info().size()) {
      MS_LOG_WITH_NODE(EXCEPTION, next_node)
        << "The inputs size " << inputs_values.size() << ", is not equal to inputs shape size "
        << op_info->inputs_tensor_info().size();
    }
  } else {
    if (inputs_values.size() != op_info->inputs_tensor_info_new()[index - 1]->size()) {
      MS_LOG_WITH_NODE(EXCEPTION, next_node)
        << "The inputs size " << inputs_values.size() << ", is not equal to inputs shape size "
        << op_info->inputs_tensor_info_new()[index - 1]->size() << ", index is " << index - 1;
    }
    new_tensor_infos = op_info->inputs_tensor_info_new()[index - 1];
  }

  ScopePtr scope = next_node->scope();
  MS_EXCEPTION_IF_NULL(scope);
  for (size_t i = 0; i < inputs_values.size(); ++i) {
    TensorInfo tensor_info;
    if (new_tensor_infos != nullptr) {
      auto elem = new_tensor_infos->GetElement(SizeToLong(i));
      MS_EXCEPTION_IF_NULL(elem);
      tensor_info = elem->GetValue();
    } else {
      tensor_info = op_info->inputs_tensor_info()[i];
    }
    if (tensor_info == TensorInfo()) {
      MS_LOG(INFO) << "Tensor info for " << i << "th input of node " << node->DebugString()
                   << " is TensorInfo. It is not need to split";
      return nullptr;
    }
    auto value_ptr = inputs_values[i];
    auto tensor = value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    TensorLayout tensor_layout = tensor_info.tensor_layout();
    auto value_node = NewValueNode(value_ptr)->cast<AnfNodePtr>();
    Operator op = CreateGetTensorSliceOp(tensor_layout);
    std::vector<AnfNodePtr> node_input = CreateInput(op, value_node, SPLIT_TENSOR);
    CNodePtr new_node = func_graph->NewCNode(node_input);
    new_node->set_in_forward_flag(true);
    auto new_node_value = node_input[0]->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(new_node_value);
    PrimitivePtr new_node_prim = new_node_value->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(new_node_prim);
    new_node_prim->set_instance_name(SPLIT_TENSOR);
    new_node_prim->set_attr("keep_value_node_input", MakeValue(true));
    new_node->set_scope(scope);
    node_input[0]->set_scope(scope);
    make_tuple_inputs.push_back(new_node);
  }

  CNodePtr make_tuple = func_graph->NewCNode(make_tuple_inputs);
  auto prim = GetValueNode<PrimitivePtr>(next_node->input(0));
  if (prim == nullptr) {
    return nullptr;
  }
  (void)prim->AddAttr(KEEP_ALIVE, MakeValue(true));
  return make_tuple;
}

static CNodePtr CreateSliceForTensor(const AnfNodePtr &node, const CNodePtr &next_node, int64_t index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(next_node);
  OperatorInfoPtr op_info = next_node->user_data<OperatorInfo>();
  if (!op_info) {
    return nullptr;
  }

  if (op_info->name().find(FILLV2) != std::string::npos) {
    MS_LOG(INFO) << "FillV2 operator info no need to split tensor";
    return nullptr;
  }

  if (op_info->name().find(STAND_ALONE) != std::string::npos) {
    MS_LOG(INFO) << "Stand alone operator info no need to split tensor";
    return nullptr;
  }

  // If the shape of tensor is [] or [1], no need to split it.
  Shapes shapes = GetNodeShape(node);
  if (shapes.size() != 1) {
    MS_LOG_WITH_NODE(EXCEPTION, next_node)
      << "Split tensor for " << op_info->name() << ": GetNodeShape for tensor_node, output size is not 1";
  }
  Shape shape = shapes[0];
  std::string shape_str = ShapeToString(shape);
  if (shape.empty() || ((shape.size() == 1) && (shape[0] == 1))) {
    MS_LOG(INFO) << "Split tensor for " << op_info->name() << ": The shape is " << shape_str
                 << ", no need to split it.";
    return nullptr;
  }

  MS_LOG(INFO) << "Split tensor for " << op_info->name() << ": The shape of tensor is " << shape_str;

  // extract tensor layout
  TensorLayout tensor_layout;
  auto inputs_info_size = op_info->inputs_tensor_info_new().empty() ? op_info->inputs_tensor_info().size()
                                                                    : op_info->inputs_tensor_info_new().size();
  if (LongToSize(index - 1) >= inputs_info_size) {
    if (IsIgnoreSplitTensor(next_node, index - 1)) {
      MS_LOG(INFO) << op_info->name() << ": no need to split tensor for index " << (index - 1);
      return nullptr;
    }
    MS_LOG_WITH_NODE(EXCEPTION, next_node) << op_info->name() << ": The index is out of range, index is  "
                                           << (index - 1) << ", vector size is  " << inputs_info_size;
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
  const AnfNodePtrList get_tensor_slice_input = CreateInput(op, node, SPLIT_TENSOR);
  CNodePtr slice_node = func_graph->NewCNode(get_tensor_slice_input);
  if (!op_info->sub_ops().empty()) {
    auto sub_ops = op_info->sub_ops();
    for (size_t i = 0; i < sub_ops.size(); i++) {
      if (!sub_ops.at(i).empty()) {
        const AnfNodePtrList sub_input = CreateInput(sub_ops.at(i).at(0), slice_node, SUB);
        const CNodePtr new_sub_node = func_graph->NewCNode(sub_input);
        slice_node = new_sub_node;
      }
    }
  }
  auto prim = GetValueNode<PrimitivePtr>(next_node->input(0));
  if (prim == nullptr) {
    return nullptr;
  }
  (void)prim->AddAttr(KEEP_ALIVE, MakeValue(true));
  return slice_node;
}

static void CollectTensorRealUser(const AnfNodePtr &node, const FuncGraphManagerPtr &manager,
                                  mindspore::CompactSet<std::pair<AnfNodePtr, int>> *collect_set) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(collect_set);
  static const std::set<std::string> NO_INPUT_TENSOR_OPS = {UNIFORM_REAL, STANDARD_NORMAL};
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
    if ((use_cnode_prim->name() == DEPEND && node_pair.second != 1) || use_cnode_prim->name() == TENSORDUMP ||
        NO_INPUT_TENSOR_OPS.find(use_cnode_prim->name()) != NO_INPUT_TENSOR_OPS.end()) {
      continue;
    }
    if ((use_cnode_prim->name() == DEPEND && node_pair.second == 1) || IsSomePrimitive(use_cnode, DUMPGRADIENT)) {
      CollectTensorRealUser(use_cnode, manager, collect_set);
    }
    if (IsParallelCareNode(use_cnode)) {
      if (IsPrimitiveCNode(use_cnode, prim::kPrimReceive)) {
        continue;
      }
      // Find parallel care node, collect it.
      collect_set->insert(node_pair);
    }
  }
}

static void StepSplitTensor(const AnfNodePtr &node, const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(manager);
  mindspore::CompactSet<std::pair<AnfNodePtr, int>> real_kernel_collector;
  (void)CollectTensorRealUser(node, manager, &real_kernel_collector);
  if (real_kernel_collector.size() == SIZE_ZERO) {
    return;
  }
  // Todo: Need check whether the input tensor split strategies are consistent.
  std::pair<AnfNodePtr, int> node_with_parallel_info = *(real_kernel_collector.cbegin());
  CNodePtr slice_node = nullptr;
  if (IsValueNode<ValueList>(node) || IsValueNode<ValueTuple>(node)) {
    // Create _GetTensorSlice tuple to split tensor list.
    slice_node =
      CreateSliceForTensorList(node, node_with_parallel_info.first->cast<CNodePtr>(), node_with_parallel_info.second);
  } else {
    slice_node =
      CreateSliceForTensor(node, node_with_parallel_info.first->cast<CNodePtr>(), node_with_parallel_info.second);
  }
  if (slice_node) {
    (void)manager->Replace(node, slice_node);
  }
  return;
}

static void StepReplaceOp(OperatorVector replace_op, const CNodePtr &node) {
  MS_LOG(INFO) << "Start StepReplaceOp for " << node->fullname_with_scope();
  // step1:get graph manager distribute_operator
  OperatorInfoPtr distribute_operator = node->user_data<OperatorInfo>();
  if (distribute_operator == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Failure:AddNode error since distribute_operator is nullptr";
  }
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Failure:AddNode error since manager is nullptr";
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
    MS_LOG_WITH_NODE(EXCEPTION, node) << "replace_op_info is not empty and size not equal to replace_op!";
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
    if (GetPrimitiveFlag(origin_prim, kAttrOffload)) {
      prim->set_attr(kAttrOffload, MakeValue(true));
      const auto &backward_prefetch_iter = origin_prim_attrs.find(kAttrBackwardPrefetch);
      if (backward_prefetch_iter != origin_prim_attrs.end()) {
        prim->set_attr(kAttrBackwardPrefetch, backward_prefetch_iter->second);
      }
    }
    if (prim->name() == GET_NEXT && origin_prim_attrs.find(SYMBOLS) != origin_prim_attrs.end()) {
      prim->set_attr(SYMBOLS, origin_prim_attrs[SYMBOLS]);
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
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Failure:AddNode error since manager is nullptr";
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
    MS_EXCEPTION_IF_NULL(replace_input_cnode);
    replace_input_cnode->set_user_data<OperatorInfo>(op_info);
    size_t inputs_size = replace_input_cnode->size();
    while (IntToSize(appear_count) < inputs_size && replace_input_cnode->input(appear_count)->func_graph() != nullptr) {
      ++appear_count;
    }
    if (IntToSize(appear_count) >= inputs_size) {
      MS_LOG_WITH_NODE(EXCEPTION, node) << "No replaceable virtual_input_node";
    }
    input_map[replace_input.first] = appear_count;
    replace_input_cnode->set_in_forward_flag(true);
    manager->SetEdge(replace_input.first, appear_count, pre_node);
  }

  auto replace_output = replace_graph->second->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(replace_output);
  replace_output->set_in_forward_flag(true);
  replace_output->set_primal_attrs(node->primal_attrs());
  (void)manager->Replace(node, replace_output);
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
        MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Only one of replace_op or replace_op can be used";
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
        MS_EXCEPTION_IF_NULL(GetCNodePrimitive(cnode));
        if (GetCNodePrimitive(cnode)->HasAttr(RECOMPUTE)) {
          MS_EXCEPTION_IF_NULL(GetCNodePrimitive(cnode));
          GetCNodePrimitive(cnode)->AddAttr(RECOMPUTE_COMM_OP, GetCNodePrimitive(cnode)->GetAttr(RECOMPUTE));
        }
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

OperatorInfoPtr GetNextDistributeOperator(size_t pos_in_param, CNodePtr *next_cnode_ptr, int *input_pos_ptr,
                                          bool *using_func_param_op_info_ptr) {
  MS_EXCEPTION_IF_NULL(next_cnode_ptr);
  MS_EXCEPTION_IF_NULL(input_pos_ptr);
  MS_EXCEPTION_IF_NULL(using_func_param_op_info_ptr);
  auto next_cnode = *next_cnode_ptr;
  int input_pos = *input_pos_ptr;
  bool using_func_param_op_info = *using_func_param_op_info_ptr;
  OperatorInfoPtr next_distribute_operator = nullptr;
  if (IsValueNode<FuncGraph>(next_cnode->input(0))) {
    auto fg = GetValueNode<FuncGraphPtr>(next_cnode->input(0));
    MS_EXCEPTION_IF_NULL(fg);
    auto fg_parameters = fg->parameters();
    auto param = fg_parameters[IntToSize(input_pos - 1)];
    MS_EXCEPTION_IF_NULL(param);
    if (param->has_user_data(INDEX_OPERATOR_INFO)) {
      auto index_operator_info = param->user_data<IndexOperatorMap>(INDEX_OPERATOR_INFO);
      MS_EXCEPTION_IF_NULL(index_operator_info);
      using_func_param_op_info = index_operator_info->count(pos_in_param) != 0 ? true : false;
    }
    if (using_func_param_op_info) {
      MS_LOG(INFO) << "Func call node:" << next_cnode->DebugString() << " has operator info.";
      auto info_ptr = param->template user_data<IndexOperatorMap>(INDEX_OPERATOR_INFO);
      MS_EXCEPTION_IF_NULL(info_ptr);
      next_distribute_operator = info_ptr->at(pos_in_param);
      const auto &abstract = param->abstract();
      MS_EXCEPTION_IF_NULL(abstract);
      bool is_tuple_param = false;
      size_t param_input_size = 1;
      if (abstract->template isa<abstract::AbstractSequence>()) {
        const auto &sequence_abs = abstract->template cast_ptr<abstract::AbstractSequence>();
        MS_EXCEPTION_IF_NULL(sequence_abs);
        const auto &ele = sequence_abs->elements();
        param_input_size = ele.size();
        is_tuple_param = param_input_size > 1 ? true : false;
      }
      auto input = next_cnode->input(input_pos);
      auto get_make_tuple = [&input](const FuncGraphPtr &func_graph, size_t param_input_size) {
        std::vector<AnfNodePtr> make_tuple_inputs;
        make_tuple_inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
        for (size_t i = 0; i < param_input_size; i++) {
          std::vector<AnfNodePtr> tuple_get_item_inputs{NewValueNode(prim::kPrimTupleGetItem), input,
                                                        CreatInt64Imm(UlongToLong(i))};
          auto tuple_get_item = func_graph->NewCNode(tuple_get_item_inputs);
          MS_EXCEPTION_IF_NULL(tuple_get_item);
          make_tuple_inputs.push_back(tuple_get_item);
        }
        auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
        MS_EXCEPTION_IF_NULL(make_tuple);
        return make_tuple;
      };
      if (is_tuple_param && !IsPrimitiveCNode(input, prim::kPrimMakeTuple)) {
        const auto &func_graph = next_cnode->func_graph();
        auto make_tuple = get_make_tuple(func_graph, param_input_size);
        MS_EXCEPTION_IF_NULL(make_tuple);
        FuncGraphManagerPtr manager = func_graph->manager();
        MS_EXCEPTION_IF_NULL(manager);
        (void)manager->SetEdge(next_cnode, input_pos, make_tuple);
        input = make_tuple;
      }
      if (is_tuple_param) {
        next_cnode = input->cast<CNodePtr>();
        input_pos = static_cast<int>(pos_in_param) + 1;
      }
      using_func_param_op_info = true;
    } else {
      next_distribute_operator = GetDistributeOperator(next_cnode);
    }
  } else {
    next_distribute_operator = GetDistributeOperator(next_cnode);
  }
  *next_cnode_ptr = next_cnode;
  *input_pos_ptr = input_pos;
  *using_func_param_op_info_ptr = using_func_param_op_info;
  return next_distribute_operator;
}

}  // namespace

void ParallelProcessor::Redistribution(const std::pair<AnfNodePtr, PosPair> &node_pair, const AnfNodePtr &pre_node,
                                       const std::vector<int> &get_item_index,
                                       std::pair<AnfNodePtr, int> *new_next_node) {
  MS_LOG(DEBUG) << "Do Redistribution for " << node_pair.first->fullname_with_scope();
  MS_EXCEPTION_IF_NULL(new_next_node);
  auto next_cnode = node_pair.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(next_cnode);
  auto func_graph = next_cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto pre_cnode = pre_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(pre_cnode);
  auto distribute_operator = GetDistributeOperator(pre_cnode);
  MS_EXCEPTION_IF_NULL(distribute_operator);
  auto dev_list = distribute_operator->stage_device_list();
  bool using_func_param_op_info = false;
  auto pos_pair = node_pair.second;
  auto input_pos = pos_pair.first;
  auto pos_in_param = pos_pair.second;
  OperatorInfoPtr next_distribute_operator =
    GetNextDistributeOperator(pos_in_param, &next_cnode, &input_pos, &using_func_param_op_info);
  new_next_node->first = next_cnode;
  new_next_node->second = input_pos;
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
  auto status = ObtainOutputTensorLayout(next_distribute_operator, {next_cnode, input_pos}, next_cnode,
                                         using_func_param_op_info, &tensorlayout_out);
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
    MS_LOG_WITH_NODE(EXCEPTION, pre_cnode) << "Failure:tensor_redistribution init failed";
  }
  if (tensorlayout_in.GetVirtualRank().size() > 1 || tensorlayout_out.GetVirtualRank().size() > 1) {
    auto real_pre_node = next_cnode->input(input_pos);
    InsertRedistributionForMicroInterleaved(tensor_redistribution, {next_cnode, input_pos}, func_graph, pre_cnode,
                                            real_pre_node);
    return;
  }
  RedistributionOpListPtr redistribution_oplist_ptr = tensor_redistribution->InferTensorRedistributionOperatorList();
  if (redistribution_oplist_ptr == nullptr) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, pre_cnode) << "Infer tensor redistribution failed.";
  }
  if (!tensor_redistribution->IsAssembledStaticShape()) {
    redistribution_oplist_ptr = TensorTransform::GetInstance()->OptimizeTensorRedistributionOperatorList(
      redistribution_oplist_ptr, tensor_redistribution->input_shape());
  }

  if (redistribution_oplist_ptr == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, pre_cnode) << "Failure:InferTensorRedistribution failed";
  }
  MS_LOG(DEBUG) << "Redistribution size " << redistribution_oplist_ptr->first.size();
  if (!redistribution_oplist_ptr->first.empty()) {
    // the last one is the pos of node in maketuple
    tensor_redistribution->CreateAssembledDynamicMapping(next_cnode, pre_cnode, func_graph, input_pos);
    // insert node before next node
    InsertRedistribution(redistribution_oplist_ptr, next_cnode, func_graph, input_pos, pre_cnode,
                         tensor_redistribution);
  }
  // Rollback to dynamic shape.
  if (tensor_redistribution->IsAssembledStaticShape() &&
      tensor_redistribution->ResetLayoutTransfer() != Status::SUCCESS) {
    MS_LOG(WARNING) << "Failed to reset layout transfer.";
  }
}

void ParallelProcessor::StepRedistribution(const CNodePtr &cnode, const NodeUsersMap &node_users_map) {
  MS_LOG(DEBUG) << "Do StepRedistribution for " << cnode->fullname_with_scope();
  MS_EXCEPTION_IF_NULL(cnode->func_graph());
  FuncGraphManagerPtr manager = cnode->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  // In pipeline parallel mode, redistribution is inserted after receive, not send.
  if (IsOneOfPrimitiveCNode(cnode, {prim::kPrimSend, prim::kPrimMakeTuple, prim::kPrimMakeList,
                                    prim::kPrimTensorToScalar, prim::kPrimDType, prim::kPrimDtypeToEnum})) {
    return;
  }
  // Find Redistribution next_nodes
  // next_node.first.second = (pos in next node input(don't need to -1), pos in tuple(need to -1))
  std::vector<std::pair<std::pair<AnfNodePtr, PosPair>, std::vector<int>>> next_nodes;
  RedistributionNextNode(cnode, manager, node_users_map, {-1}, -1, &next_nodes);
  if (next_nodes.empty()) {
    return;
  }
  // Find Redistribution pre_nodes
  std::vector<AnfNodePtr> pre_nodes;
  RedistributionPreNode(cnode, manager, &pre_nodes);
  if (pre_nodes.size() > 1) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << " Don't support Redistribution has multiple pre_node.";
  }
  // Insert Redistribution nodes between pre_nodes and next_nodes
  for (auto &pre_node : pre_nodes) {
    std::vector<std::pair<std::pair<AnfNodePtr, int>, std::vector<int>>> next_nodes_tensor_dump;
    for (auto &next_node : next_nodes) {
      MS_LOG(INFO) << "===========Do Redistribution start============" << std::endl
                   << pre_node->fullname_with_scope() << "->" << next_node.first.first->fullname_with_scope() << "("
                   << next_node.first.second << ")";
      std::pair<AnfNodePtr, int> new_next_node;
      Redistribution(next_node.first, pre_node, next_node.second, &new_next_node);
      MS_LOG(INFO) << "===========Do Redistribution end  ============";
      next_nodes_tensor_dump.push_back(std::make_pair(new_next_node, next_node.second));
    }
    RedistributionDumpHandlerPtr redistribution_dump_handler =
      std::make_shared<RedistributionParallelTensorDumpHandler>(pre_nodes, next_nodes_tensor_dump, manager);
    (void)redistribution_dump_handler->HandleDumpAfterRedistributionNode();
    for (const auto &next_node : next_nodes) {
      if (!next_node.first.first->has_user_data(PARAM_OUT_INDEX)) {
        continue;
      }
      if (pre_node->func_graph() == next_node.first.first->func_graph()) {
        continue;
      }
      auto param_out = next_node.first.first->user_data<ParamOutIndex>(PARAM_OUT_INDEX);
      MS_EXCEPTION_IF_NULL(param_out);
      const auto &param = param_out->first;
      MS_EXCEPTION_IF_NULL(param);
      auto distribute_operator = GetDistributeOperator(pre_node->cast<CNodePtr>());
      auto index_operator_info = std::make_shared<IndexOperatorMap>();
      if (param->has_user_data(INDEX_OPERATOR_INFO)) {
        auto info_ptr = param->user_data<IndexOperatorMap>(INDEX_OPERATOR_INFO);
        MS_EXCEPTION_IF_NULL(info_ptr);
        *index_operator_info = *info_ptr;
      }
      index_operator_info->emplace(param_out->second, distribute_operator);
      param->set_user_data<IndexOperatorMap>(INDEX_OPERATOR_INFO, index_operator_info);
      break;
    }
  }
}

TensorLayout ParallelProcessor::GetTensorInLayout(const AnfNodePtr &pre_node, std::vector<int> get_item_index) {
  TensorLayout tensorinfo_in_layout;
  auto pre_cnode = pre_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(pre_cnode);
  auto distribute_operator = GetDistributeOperator(pre_cnode);
  MS_EXCEPTION_IF_NULL(distribute_operator);
  if (!distribute_operator->outputs_tensor_info_new().empty()) {
    return GetTensorInLayoutForNewShape(pre_node, get_item_index);
  }
  if (get_item_index.size() != 1) {
    // If does not have outputes_tensor_info_new, the outputs only have one tensor info
    // thus the get item index must only have one value
    auto all_minus_one = std::any_of(get_item_index.begin(), get_item_index.end(), [](int i) { return i == -1; });
    MS_LOG(INFO) << "The get_item_index size is not 1, the size is " << get_item_index.size() << ", the last item is "
                 << get_item_index[get_item_index.size() - 1] << ", all_minus_one is " << all_minus_one;
  }
  if (get_item_index[get_item_index.size() - 1] != -1) {
    if (get_item_index[get_item_index.size() - 1] >= SizeToInt(distribute_operator->outputs_tensor_info().size())) {
      MS_LOG_WITH_NODE(EXCEPTION, pre_cnode)
        << "The index out of range. Node: " << pre_node->DebugString() << " index: " << get_item_index
        << " outputs_tensor_info's size: " << distribute_operator->outputs_tensor_info().size();
    }
    auto tensorinfo_in =
      distribute_operator->outputs_tensor_info()[IntToSize(get_item_index[get_item_index.size() - 1])];
    tensorinfo_in_layout = tensorinfo_in.tensor_layout();
  } else {
    if (distribute_operator->outputs_tensor_info().empty()) {
      MS_LOG_WITH_NODE(EXCEPTION, pre_cnode) << "The outputs tensor info is empty. Node:" << pre_node->DebugString();
    }
    auto tensorinfo_in = distribute_operator->outputs_tensor_info()[0];
    tensorinfo_in_layout = tensorinfo_in.tensor_layout();
  }
  return tensorinfo_in_layout;
}

void ParallelProcessor::ForwardCommunication(const OperatorVector &forward_op, const ForwardOpList &forward_op_list,
                                             const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG_WITH_NODE(INFO, node) << "Insert ForwardCommunication to node: " << common::AnfAlgo::GetCNodeName(node);

  if (dyn_cast<abstract::SequenceShape>(node->Shape()) != nullptr && !forward_op_list.empty()) {
    // For Ops like grouped_matmul has multiple output
    MS_LOG(INFO) << "The input node " << node->DebugString()
                 << " has multiple output, enter ForwardCommunicationForMultiOut";
    ForwardCommunicationForMultiOut(node, forward_op, forward_op_list);
    return;
  }

  InserForwardCommunicationForSingleOutput(node, forward_op, forward_op_list);
}

void ParallelProcessor::InsertForwardOps(const OperatorInfoPtr &distribute_operator, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(distribute_operator);
  MS_EXCEPTION_IF_NULL(cnode);
  if (IsPrimitiveCNode(cnode, prim::kPrimReceive)) {
    return;
  }
  const auto &forward_op = distribute_operator->forward_op();
  const auto &forward_op_list = distribute_operator->forward_op_list();
  // for gmm, its make tuple will inherit its op info,
  // which will lead to insert allreduce for maketuple.
  bool has_forward_op = !forward_op.empty() || !forward_op_list.empty();
  if (has_forward_op && !IsPrimitiveCNode(cnode, prim::kPrimMakeTuple)) {
    MS_LOG(INFO) << "Insert forward op for " << distribute_operator->name();
    ForwardCommunication(forward_op, forward_op_list, cnode);
  }
}

void ParallelProcessor::Process() {
  auto root = processor_context_->root;
  auto manager = processor_context_->manager;
  auto &all_nodes = processor_context_->all_nodes;
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
      MS_EXCEPTION_IF_NULL(cnode);
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
      if (!cnode->HasPrimalAttr(PIPELINE_PARAM) || processor_context_->is_pp_interleave) {
        // insert forward ops
        if (!IsControlFlowNode(cnode)) {
          InsertForwardOps(distribute_operator, cnode);
        }

        // insert redistribution ops
        StepRedistribution(cnode, node_users_map);
      }
      // insert backward ops
      if (!IsControlFlowNode(cnode) && has_backward) {
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
}  // namespace parallel
}  // namespace mindspore
