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

#include "frontend/parallel/parallel_preprocessor.h"

#include <cinttypes>
#include <algorithm>
#include <chrono>
#include <list>
#include <string>
#include <memory>
#include <utility>
#include <vector>

#include "primitive/sequence_ops.h"
#include "primitive/other_ops.h"
#include "primitive/array_ops.h"
#include "primitive/structure_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/conv_pool_ops.h"
#include "primitive/nn_ops.h"
#include "include/utils/parallel_context.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
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
#include "frontend/parallel/tensor_layout/prime_generator.h"
#include "frontend/parallel/graph_util/pipeline_split_utils.h"
#include "frontend/parallel/graph_util/fold_pipeline_split_utils.h"
#include "frontend/parallel/pipeline_transformer/pipeline_interleave.h"
#include "frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.h"
#include "frontend/parallel/graph_util/grad_accumulation_utils.h"
#include "frontend/parallel/interleaved_parallel/interleaved_parallel.h"
#include "frontend/parallel/node_check.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/ops_info/matmul_info.h"
#include "frontend/parallel/dynamic_shape/dynamic_shape.h"
#include "include/frontend/parallel/tensor_layout/tensor_transform.h"
#include "frontend/parallel/strategy_utils.h"
#include "frontend/parallel/strategy_loader.h"
#include "frontend/parallel/parallel_node_check.h"
#include "frontend/parallel/parallel_optimizer/opt_param_mgr.h"
#include "ir/param_info.h"
#include "ir/tensor.h"
#include "ir/graph_utils.h"
#include "utils/trace_base.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"
#include "mindspore/ops/infer/ops_func_impl/flash_attention_score.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "include/frontend/jit/ps/executor/graph_executor_py.h"
#include "mindspore/ccsrc/frontend/parallel/ops_info/operator_info.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_e.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr auto kDumpIrParallelDetail = "1";

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

void ClearCnodesForOperator(const std::vector<AnfNodePtr> &all_nodes) {
  for (auto &node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (!StrategyUtils::CheckExtractInformation(cnode) || IsPrimitiveCNode(node, prim::kPrimSend)) {
      continue;
    }

    auto find_iter = cnode->attrs().find(OP_INFO_CREATED);
    if (find_iter != cnode->attrs().end()) {
      auto op = GetDistributeOperator(cnode);
      if (op != nullptr) {
        op->clear_cnodes();
      }
      continue;
    }
  }
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
          auto cnode = input->cast<CNodePtr>();
          MS_EXCEPTION_IF_NULL(cnode);
          manager->SetEdge(fa_cnode, index + 1, cnode->input(kIndex1));
          MS_LOG(DEBUG) << "Eliminate TensorToTuple for " << fa_cnode->fullname_with_scope() << ", index is "
                        << index + 1;
        } else {
          auto dtype = NewValueNode(MakeValue<int64_t>(kInt64->type_id()));
          dtype->set_abstract(abstract::FromValue(static_cast<int64_t>(kInt64->type_id())));
          auto tuple_to_tensor_cnode =
            fa_cnode->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleToTensor), input, dtype});
          MS_EXCEPTION_IF_NULL(GetCNodePrimitive(tuple_to_tensor_cnode));
          auto abs = GenerateAbsByOpInfer(GetCNodePrimitive(tuple_to_tensor_cnode), {input, dtype});
          tuple_to_tensor_cnode->set_abstract(abs);
          manager->SetEdge(fa_cnode, index + 1, tuple_to_tensor_cnode);
          MS_LOG(DEBUG) << "Insert TupleToTensor for " << fa_cnode->fullname_with_scope() << ", index is " << index + 1;
        }
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
    BroadCastSeqChunk(root);
    LastStageEndNode(all_nodes, manager, root);
    return;
  }
  TagMicroBatchStart(manager, all_nodes);
  TagMicroBatchEnd(manager, all_nodes);
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const auto no_cell_reuse = context->CellReuseLevel() == CellReuseLevel::kNoCellReuse;
  bool enable_grad_accu = ParallelContext::GetInstance()->grad_accumulation_step() > 1;
  bool is_optimizer_level_2 = ParallelContext::GetInstance()->grad_accumulation_shard();
  bool is_optimizer_level_3 = ParallelContext::GetInstance()->zero3();
  if (enable_grad_accu && no_cell_reuse && (is_optimizer_level_2 || is_optimizer_level_3)) {
    MS_LOG(EXCEPTION) << "For optimizer level 2/3, only support with lazy_inline mode.";
  }
  if (no_cell_reuse && enable_grad_accu) {
    TagMicroBatchBpEndPrim(root);
    TagMicroBatchBpEnd(root);
  }
}

static bool IsCommonOp(const AnfNodePtr &node) {
  CNodePtr cnode = node->cast<CNodePtr>();
  bool is_comm_op =
    IsParallelCareNode(cnode) && cnode->has_user_data<OperatorInfo>() && !IsPrimitiveCNode(node, prim::kPrimReshape);
  return is_comm_op;
}

static std::shared_ptr<TensorLayout> GetOutputLayoutFromCNode(const CNodePtr &cnode, size_t output_index) {
  MS_EXCEPTION_IF_NULL(cnode);
  OperatorInfoPtr distribute_operator = GetDistributeOperator(cnode);
  MS_EXCEPTION_IF_NULL(distribute_operator);
  TensorLayout tensorlayout_out;
  if (distribute_operator->outputs_tensor_info_new().empty()) {
    if (distribute_operator->outputs_tensor_info().size() <= output_index) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "outputs_tensor_info size is  "
                                         << distribute_operator->outputs_tensor_info().size()
                                         << ", must be greater than output_index  " << output_index;
    }
    TensorInfo tensorinfo_out = distribute_operator->outputs_tensor_info()[output_index];
    tensorlayout_out = tensorinfo_out.tensor_layout();
  } else {
    if (distribute_operator->outputs_tensor_info_new().size() <= output_index) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "outputs_tensor_info size is  "
                                         << distribute_operator->outputs_tensor_info_new().size()
                                         << ", must be greater than output_index  " << output_index;
    }
    auto tensorinfo_out = distribute_operator->outputs_tensor_info_new()[output_index];
    if (tensorinfo_out->is_list()) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "For " << cnode->DebugString() << ": the " << output_index
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
  MS_EXCEPTION_IF_NULL(cnode);
  if (!IsValueNode<Primitive>(cnode->input(0))) {
    return nullptr;
  }
  if (IsParallelCareNode(cnode) && cnode->has_user_data<OperatorInfo>()) {
    auto layout_ptr = GetOutputLayoutFromCNode(cnode, output_index);
    if (!layout_ptr) {
      MS_LOG_WITH_NODE(EXCEPTION, node) << "Failure:GetLayoutFromCNode failed";
    }
    if (IsPrimitiveCNode(cnode)) {
      auto prim = GetCNodePrimitive(cnode);
      MS_EXCEPTION_IF_NULL(prim);
      if (prim->HasAttr(kAttrFineGrainedInterleavedBlockIndex)) {
        layout_ptr->set_fine_grain_block_index(GetValue<int64_t>(prim->GetAttr(kAttrFineGrainedInterleavedBlockIndex)));
      }
    }
    return layout_ptr;
  }
  return nullptr;
}

static CNodePtr GetArgumentByParameter(const AnfNodePtr &node) {
  auto fg = node->func_graph();
  auto parameters = fg->parameters();
  auto iter = std::find(parameters.begin(), parameters.end(), node);
  if (iter == parameters.end()) {
    return nullptr;
  }
  auto pos = std::distance(parameters.begin(), iter);
  auto fg_used_map = fg->func_graph_cnodes_index();
  for (auto &cur_fg_use : fg_used_map) {
    if (cur_fg_use.first->second != 0) {
      continue;
    }
    auto cur_fg = cur_fg_use.first->first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cur_fg);
    auto argument = cur_fg->input(pos + 1);
    if (argument->isa<Parameter>()) {
      return GetArgumentByParameter(argument);
    } else if (argument->isa<CNode>()) {
      return argument->cast<CNodePtr>();
    }
  }
  return nullptr;
}

std::shared_ptr<TensorLayout> FindPrevLayout(const AnfNodePtr &node, bool *is_input_param);
std::shared_ptr<TensorLayout> FindPrevLayoutByParameter(const AnfNodePtr &node, bool *is_input_param) {
  auto node_param_ptr = node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(node_param_ptr);
  if (node_param_ptr->has_default()) {
    // Only when the real input of Reshape is a parameter that the strategy of Reshape will be assigned to this
    // parameter.
    *is_input_param = true;
    return CreateParameterLayout(node);
  }

  // the node is parameter of sub-graph
  auto actual_node = RefParameterToActualNode(node, [&](const CNodePtr &cnode) {
    bool filter = IsPrimitiveCNode(cnode, prim::kPrimCast) || IsPrimitiveCNode(cnode, prim::kPrimLoad) ||
                  IsPrimitiveCNode(cnode, prim::kPrimDepend);
    return std::make_pair(filter, 1);
  });
  if (actual_node) {
    return FindPrevLayout(actual_node, is_input_param);
  }
  return nullptr;
}

std::shared_ptr<TensorLayout> FindPrevLayoutByCallNode(const CNodePtr &cnode, int64_t tuple_index,
                                                       bool *is_input_param) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto fg = GetValueNode<FuncGraphPtr>(cnode->input(0));
  MS_EXCEPTION_IF_NULL(fg);
  auto pre_node = GetRealKernelNode(fg->output(), tuple_index, nullptr).first;
  if (!pre_node) {
    return nullptr;
  }
  return FindPrevLayout(pre_node, is_input_param);
}

std::shared_ptr<TensorLayout> FindPreLayoutForTupleGetitem(const CNodePtr &cnode, bool *is_input_param) {
  auto tuple_index = GetTupleGetItemIndex(cnode);
  CNodePtr tuple_getitem_real_input = nullptr;
  auto tuple_getitem_input = cnode->input(1);
  if (tuple_getitem_input->isa<CNode>()) {
    tuple_getitem_real_input = tuple_getitem_input->cast<CNodePtr>();
  } else if (tuple_getitem_input->isa<Parameter>()) {
    tuple_getitem_real_input = GetArgumentByParameter(tuple_getitem_input);
  }
  if (IsPrimitiveCNode(tuple_getitem_real_input, prim::kPrimMakeTuple)) {
    return FindPrevLayout(tuple_getitem_real_input->input(tuple_index + 1), is_input_param);
  }
  MS_EXCEPTION_IF_NULL(tuple_getitem_real_input);
  if (IsValueNode<FuncGraph>(tuple_getitem_real_input->input(0))) {
    return FindPrevLayoutByCallNode(tuple_getitem_real_input, tuple_index, is_input_param);
  }
  auto layout_ptr = FindPrevParallelCareNodeLayout(tuple_getitem_real_input, LongToSize(tuple_index));
  if (!layout_ptr) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode)
      << " Failure:FindPrevLayout failed, tuple_getitem before reshape, but there does not exit a "
         "parallel care node before tuple_getitem!";
  }
  return layout_ptr;
}

std::shared_ptr<TensorLayout> FindPrevLayout(const AnfNodePtr &node, bool *is_input_param) {
  if (node->isa<Parameter>()) {
    return FindPrevLayoutByParameter(node, is_input_param);
  }
  if (!node->isa<CNode>()) {
    return nullptr;
  }
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (IsValueNode<FuncGraph>(cnode->input(0))) {
    return FindPrevLayoutByCallNode(cnode, -1, is_input_param);
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
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Failure:GetLayoutFromCNode failed";
    }
    if (IsPrimitiveCNode(cnode)) {
      auto prim = GetCNodePrimitive(cnode);
      MS_EXCEPTION_IF_NULL(prim);
      if (prim->HasAttr(kAttrFineGrainedInterleavedBlockIndex)) {
        layout_ptr->set_fine_grain_block_index(GetValue<int64_t>(prim->GetAttr(kAttrFineGrainedInterleavedBlockIndex)));
      }
    }
    return layout_ptr;
  }
  ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(prim_anf_node);
  PrimitivePtr prim = prim_anf_node->value()->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->name() == prim::kPrimTupleGetItem->name()) {
    return FindPreLayoutForTupleGetitem(cnode, is_input_param);
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

static void InsertShapeOp(const CNodePtr &node, const AnfNodePtr &pre_node, const FuncGraphPtr &root) {
  // shape op doesn't have params and attrs.
  OperatorParams params;
  OperatorAttrs attrs;
  auto shape_value_temp = GetValueNode(node->input(2));
  MS_EXCEPTION_IF_NULL(shape_value_temp);
  auto shape_value = shape_value_temp->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(shape_value);
  auto shape = shape_value->value();
  if (shape.empty()) {
    return;
  }
  OperatorArgs args = std::make_pair(attrs, params);
  Operator op = std::make_pair(SHAPE_OP, args);
  static auto const kShapeIndex = 2;
  InsertNode(op, node, kShapeIndex, pre_node, root, "shape");
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

OperatorInfoPtr CreateOperatorInfoForMakeTuple(const CNodePtr make_tuple_node, const CNodePtr &next_node,
                                               const int input_pos) {
  // For tuple input ops, copy the corresponding op info from ops to make tuple
  OperatorInfoPtr operator_make_tuple;
  auto is_new_shapebase_node = IsSupportNewShapeBaseNode(next_node);
  if (!is_new_shapebase_node) {
    // For ops like concat/stack, copy the whole op info to make tuple
    MS_LOG(DEBUG) << "make tuple node " << make_tuple_node->DebugString() << " has the same op info as next node "
                  << next_node->DebugString();
    operator_make_tuple = GetDistributeOperator(next_node);
  } else {
    // For ops like gmm/fia, create a new op info which contains the corresponding info
    operator_make_tuple = CreateOperatorInfo(make_tuple_node);
    MS_LOG(DEBUG) << "make tuple node " << make_tuple_node->DebugString() << " take the input " << input_pos << " of "
                  << next_node->DebugString() << " as the op info";
    auto make_tuple_prim = GetValueNode<PrimitivePtr>(make_tuple_node->input(0));
    MS_EXCEPTION_IF_NULL(make_tuple_prim);
    if (make_tuple_prim->HasAttr(STAND_ALONE)) {
      (void)make_tuple_prim->DelAttr(STAND_ALONE);
    }
    OperatorInfoPtr next_operator = next_node->user_data<OperatorInfo>();
    MS_EXCEPTION_IF_NULL(next_operator);
    if (!next_operator->mirror_ops_new().empty()) {
      if (next_operator->mirror_ops_new().size() <= LongToSize(input_pos)) {
        MS_LOG(EXCEPTION) << "The size of mirror ops is not enough, which is " << next_operator->mirror_ops_new().size()
                          << ", but the input pos is " << input_pos;
      }
      auto corresponding_mirror_ops = next_operator->mirror_ops_new().at(input_pos)->GetAllElements();
      operator_make_tuple->set_mirror_ops(corresponding_mirror_ops);
    }

    // Copy tensor info
    if (next_operator->inputs_tensor_info_new().empty()) {
      if (next_operator->inputs_tensor_info().size() <= LongToSize(input_pos)) {
        MS_LOG(EXCEPTION) << "The size of inputs tensor info is not enough, which is "
                          << next_operator->inputs_tensor_info().size() << ", but the input pos is " << input_pos;
      }
      std::vector<TensorInfo> corresponding_tensor_info(1, next_operator->inputs_tensor_info()[input_pos]);
      operator_make_tuple->set_inputs_tensor_info(corresponding_tensor_info);
    } else {
      if (next_operator->inputs_tensor_info_new().size() <= LongToSize(input_pos)) {
        MS_LOG(EXCEPTION) << "The size of inputs tensor info is not enough, which is "
                          << next_operator->inputs_tensor_info_new().size() << ", but the input pos is " << input_pos;
      }
      auto input_tensor_info = next_operator->inputs_tensor_info_new()[input_pos];
      std::vector<TensorInfoBasePtr> corresponding_tensor_info;
      for (size_t i = 0; i < input_tensor_info->size(); ++i) {
        corresponding_tensor_info.push_back(input_tensor_info->GetElement(SizeToLong(i)));
      }
      operator_make_tuple->set_inputs_tensor_info_new(corresponding_tensor_info);
    }

    // Copy strategy if have
    StrategyPtr next_node_strategy = next_operator->strategy();
    if (next_node_strategy != nullptr) {
      StrategyPtr make_tuple_new_in_stra;
      if (next_node_strategy->GetInputNewDim().empty()) {
        Strategies corresponding_strategies;
        auto make_tuple_stage = next_node_strategy->GetInputStage();
        if (next_node_strategy->GetInputDim().size() <= LongToSize(input_pos)) {
          MS_LOG(EXCEPTION) << "The size of input dim is not enough, which is "
                            << next_node_strategy->GetInputDim().size() << ", but the input pos is " << input_pos;
        }
        Dimensions corresponding_dim = next_node_strategy->GetInputDim().at(input_pos);
        corresponding_strategies.push_back(corresponding_dim);
        make_tuple_new_in_stra = NewStrategy(make_tuple_stage, corresponding_strategies);
      } else {
        // For ops which supports NewShapeBase
        NewStrategies corresponding_strategies;
        auto make_tuple_stage = next_node_strategy->GetInputStage();
        if (next_node_strategy->GetInputNewDim().size() <= LongToSize(input_pos)) {
          MS_LOG(EXCEPTION) << "The size of input new dim is not enough, which is "
                            << next_node_strategy->GetInputNewDim().size() << ", but the input pos is " << input_pos;
        }
        NewDimensions corresponding_dim = next_node_strategy->GetInputNewDim().at(input_pos);

        for (size_t i = 0; i < corresponding_dim->size(); ++i) {
          corresponding_strategies.push_back(corresponding_dim->GetElement(SizeToLong(i)));
        }
        make_tuple_new_in_stra = NewStrategy(make_tuple_stage, corresponding_strategies);
      }
      MS_LOG(DEBUG) << "The strategy set to " << make_tuple_node->DebugString() << " is "
                    << make_tuple_new_in_stra->ToString();
      operator_make_tuple->set_strategy(make_tuple_new_in_stra);
    } else {
      MS_LOG(INFO) << "For next cnode " << next_node->DebugString() << ", do not have strategy";
    }
  }
  return operator_make_tuple;
}

static OperatorInfoPtr SetMakeListForIFA(CNodePtr make_list, const CNodePtr &next_node) {
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
  if (make_list_prim == nullptr) {
    return nullptr;
  }
  if (make_list_prim->HasAttr(STAND_ALONE)) {
    (void)make_list_prim->DelAttr(STAND_ALONE);
  }
  OperatorInfoPtr next_operator = next_node->user_data<OperatorInfo>();
  if (next_operator == nullptr) {
    return nullptr;
  }
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

void AddAllGatherAttrs(const CNodePtr &allgather, const CNodePtr &cnode, const AnfNodePtr &node,
                       const std::string &op_name, bool add_accu, bool is_with_mirror, bool grad_accumulation_shard) {
  // add fusion flag
  auto fusion_id = AddCommOpFusionType(allgather, node);
  auto param_ptr = node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(param_ptr);
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
    MS_LOG_WITH_NODE(EXCEPTION, res) << "Unknown type.";
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
  MS_EXCEPTION_IF_NULL(new_node_prim);
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

static void InsertAllGatherOp(const FuncGraphPtr &root, const std::string &group, const std::pair<AnfNodePtr, int> &res,
                              const AnfNodePtr &node, const std::string &op_name, bool is_shared_param) {
  MS_EXCEPTION_IF_NULL(res.first);
  MS_EXCEPTION_IF_NULL(node);
  bool grad_accumulation_shard = ParallelContext::GetInstance()->grad_accumulation_shard();
  auto cnode = res.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  Operator op;
  CNodePtr allgather;
  auto param_ptr = node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(param_ptr);
  auto param_name = param_ptr->name();
  auto real_param = RefParameterToActualNode(node, [&](const CNodePtr &cnode) {
    bool filter = IsPrimitiveCNode(cnode, prim::kPrimCast) || IsPrimitiveCNode(cnode, prim::kPrimLoad) ||
                  IsPrimitiveCNode(cnode, prim::kPrimDepend);
    return std::make_pair(filter, 1);
  });
  if (real_param) {
    param_ptr = real_param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    param_name = param_ptr->name();
  }

  if (op_name == MICRO_STEP_ALL_GATHER) {
    op = CreateMicroStepAllGatherOp(group);
  } else {
    op = CreateAllGatherOp(group);
  }
  CNodePtr cast_node = InsertAllGatherAfterCast(res);
  param_ptr = node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(param_ptr);
  bool is_with_mirror = false;
  if (param_ptr->user_data<TensorLayout>()) {
    auto opt_shard_mirror_group = param_ptr->user_data<TensorLayout>()->opt_shard_mirror_group();
    is_with_mirror = !opt_shard_mirror_group.empty();
    MS_EXCEPTION_IF_NULL(param_ptr->param_info());
    if (!param_ptr->param_info()->parallel_optimizer() ||
        param_ptr->user_data<TensorLayout>()->opt_shard_slice_shape().empty()) {
      auto mirror_group = mirror_group_list(param_ptr->user_data<TensorLayout>());
      is_with_mirror = mirror_group.size() > 1;
    }
  }
  if (ParallelContext::GetInstance()->zero3()) {
    is_with_mirror = true;
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
  return false;
}

void InsertParallelOpt(const FuncGraphManagerPtr &manager, const AnfNodeIndexSet &param_sub_set,
                       const FuncGraphPtr &root, const AnfNodePtr &parameter, const std::string &opt_shard_group,
                       const std::string &op_name) {
  bool insert_flag = false;
  auto param_ptr = parameter->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(param_ptr);
  bool is_shared_param =
    param_ptr->has_user_data<TensorLayout>() && param_ptr->user_data<TensorLayout>()->is_shared_param();
  for (auto &param_pair : param_sub_set) {
    auto cnode = param_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (IsForwardCNode(cnode) && !IsPrimitiveCNode(cnode, prim::kPrimReceive) &&
        !IsPrimitiveCNode(cnode, prim::kPrimSend) &&
        !(IsPrimitiveCNode(cnode, prim::kPrimDepend) && param_pair.second == INDEX_TWO)) {
      if (insert_flag) {
        // if there are multiple node users, they share one same allgather
        auto next_cnode = FindCNode(parameter, op_name, cnode->func_graph(), 0);
        if (next_cnode.first) {
          manager->SetEdge(cnode, param_pair.second, next_cnode.second);
          AddNodeMirrorInfo(cnode, param_ptr->name());
          MS_LOG(INFO) << "Parallel optimizer is shared between " << parameter->ToString() << " and "
                       << GetPrimName(cnode);
        } else {
          MS_LOG(WARNING) << "Can not find the shared AllGather with multiple node users.";
        }
      } else {
        InsertAllGatherOp(root, opt_shard_group, param_pair, parameter, op_name, is_shared_param);
        insert_flag = true;
      }
    }
  }
}

bool CheckApplyZero(const FuncGraphPtr &root, const AnfNodePtr &parameter, const std::string &opt_shard_group) {
  auto enable_opt_shard = ParallelContext::GetInstance()->enable_parallel_optimizer();
  if (!enable_opt_shard) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(parameter);
  if (ParameterIsCloned(parameter)) {
    return false;
  }

  int32_t split_stage_num = ParallelContext::GetInstance()->pipeline_stage_split_num();
  if (opt_shard_group.empty() &&
      (split_stage_num <= 1 || !ParameterRequireGrad(parameter) || !root->has_flag(kTraining))) {
    return false;
  }
  if (!parameter->isa<Parameter>()) {
    return false;
  }
  return true;
}

static void ApplyParallelOptOnParam(const FuncGraphManagerPtr &manager, const AnfNodeIndexSet &param_sub_set,
                                    const FuncGraphPtr &root, const AnfNodePtr &parameter,
                                    const std::string &opt_shard_group) {
  if (!CheckApplyZero(root, parameter, opt_shard_group)) {
    return;
  }
  int32_t split_stage_num = ParallelContext::GetInstance()->pipeline_stage_split_num();
  // set all gather type
  int64_t grad_accumulation_step = ParallelContext::GetInstance()->grad_accumulation_step();
  std::string op_name = ALL_GATHER;
  if (root->has_flag(kTraining)) {
    if ((grad_accumulation_step > 1 || split_stage_num > 1) && ParameterRequireGrad(parameter)) {
      op_name = MICRO_STEP_ALL_GATHER;
    }
  }
  auto param_ptr = parameter->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(param_ptr);
  auto param_info = param_ptr->param_info();
  auto cell_reuse = MsContext::GetInstance()->CellReuseLevel() != CellReuseLevel::kNoCellReuse;
  if (cell_reuse && ParallelContext::GetInstance()->zero3()) {
    auto param_users = GetOutputNodesWithFilter(parameter, [&](const AnfNodePtr &anode) {
      return IsPrimitiveCNode(anode, prim::kPrimCast) || IsPrimitiveCNode(anode, prim::kPrimDepend) ||
             IsPrimitiveCNode(anode, prim::kPrimLoad);
    });
    std::vector<FuncGraphPtr> fg_list;
    for (const auto &param_user : param_users) {
      auto cnode = param_user.first->cast<CNodePtr>();
      if (!cnode || !IsValueNode<FuncGraph>(cnode->input(0))) {
        continue;
      }
      auto fg = GetValueNode<FuncGraphPtr>(cnode->input(0));
      if (fg == nullptr || std::find(fg_list.begin(), fg_list.end(), fg) != fg_list.end()) {
        continue;
      }
      fg_list.push_back(fg);
      auto fg_param = fg->parameters()[param_user.second - 1];
      auto new_param_sub_set = manager->node_users()[fg_param];
      InsertParallelOpt(manager, new_param_sub_set, root, fg_param, opt_shard_group, op_name);
    }
    return;
  }
  // insert all gather
  InsertParallelOpt(manager, param_sub_set, root, parameter, opt_shard_group, op_name);
}

// When this function returns non-empty string, that means parallel optimizer is applied on this parameter.
static std::string SetParallelShape(const AnfNodePtr &parameter, const std::pair<AnfNodePtr, int64_t> &res,
                                    const FuncGraphPtr &root) {
  // check null for param and cnode
  MS_EXCEPTION_IF_NULL(parameter);
  auto param_shape = parameter->Shape();

  MS_EXCEPTION_IF_NULL(param_shape);

  CNodePtr cnode = res.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  // get slice_shape
  OperatorInfoPtr distribute_operator = cnode->user_data<OperatorInfo>();
  if (distribute_operator == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "node " << cnode->ToString() << " 's distribute_operator is nullptr";
  }
  TensorLayout tensor_layout;
  if (distribute_operator->inputs_tensor_info_new().empty()) {
    if (LongToSize(res.second - 1) >= distribute_operator->inputs_tensor_info().size()) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, cnode)
        << "The parameter index is not in inputs_tensor_info. index = " << (res.second - 1)
        << ", inputs_tensor_info size = " << distribute_operator->inputs_tensor_info().size();
    }
    TensorInfo tensorinfo_in = distribute_operator->inputs_tensor_info()[LongToSize(res.second - 1)];
    tensor_layout = tensorinfo_in.tensor_layout();
  } else {
    if (LongToSize(res.second - 1) >= distribute_operator->inputs_tensor_info_new().size()) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, cnode)
        << "The parameter index is not in inputs_tensor_info. index = " << (res.second - 1)
        << ", inputs_tensor_info size = " << distribute_operator->inputs_tensor_info_new().size();
    }
    auto tensorinfo_in = distribute_operator->inputs_tensor_info_new()[LongToSize(res.second - 1)];
    tensor_layout = tensorinfo_in->GetValue().tensor_layout();
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
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "parameter " << parameter->ToString() << ": abstract is nullptr";
  }

  AbstractBasePtr cloned_abstract = abstract->Clone();
  if (cloned_abstract == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "parameter " << parameter->ToString() << ": abstract clone failed";
  }

  cloned_abstract->set_shape(std::make_shared<abstract::Shape>(slice_shape));
  parameter->set_abstract(cloned_abstract);
  ParameterPtr parameter_ptr = parameter->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(parameter_ptr);
  if (tensor_layout.IsInterleavedParallel()) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, cnode)
      << "parameter " << parameter->ToString() << " can not set to interleaved parallel";
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

static std::pair<AnfNodePtr, int64_t> FindParallelCareNode(const AnfNodePtr &node, size_t recursion_num) {
  if (recursion_num >= MAX_RECURSIVE_DEPTH) {
    return std::make_pair(nullptr, 0);
  }

  MS_EXCEPTION_IF_NULL(node);
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodeIndexSet &node_set = manager->node_users()[node];
  AnfNodeIndexSet node_sub_set;
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
    auto node_prim_name = node_prim->name();
    if (node_prim_name != GET_NEXT && node_prim_name != VIRTUAL_OUTPUT && !cnode->in_forward_flag()) {
      continue;
    }
    if ((node_prim_name == DEPEND && node_pair.second != 1) || node_prim_name == RECEIVE || node_prim_name == SEND) {
      continue;
    }
    if (node_prim_name == UPDATESTATE && node_pair.second > 0) {
      continue;
    }
    // Generator is in PARALLEL_BLACK_LIST_, return to skip find the posterior node
    if (node_prim_name == GENERATOR) {
      MS_LOG(DEBUG) << "FindParallelCareNode meets Generator, parameter may be 'seed' or 'offset', CNode info: "
                    << cnode->DebugString();
      return std::make_pair(nullptr, 0);
    }
    if (IsParallelCareNode(cnode) && cnode->has_user_data<OperatorInfo>()) {
      return node_pair;
    }
    node_sub_set.insert(node_pair);
  }
  for (auto &node_pair : node_sub_set) {
    auto tmp_pair = FindParallelCareNode(node_pair.first, recursion_num + 1);
    if (tmp_pair.first != nullptr) {
      return tmp_pair;
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
  }
  AnfNodeIndexSet &param_sub_set = manager->node_users()[parameter];
  for (auto &param_pair : param_sub_set) {
    CNodePtr param_cnode = param_pair.first->cast<CNodePtr>();
    AnfNodePtr graph_value_node;
    MS_EXCEPTION_IF_NULL(param_cnode);
    auto cnode = param_cnode->input(0)->cast<CNodePtr>();
    if (cnode) {
      graph_value_node = cnode->input(1);
    } else {
      graph_value_node = param_cnode->input(0);
    }
    if (!IsValueNode<FuncGraph>(graph_value_node)) {
      continue;
    }
    FuncGraphPtr graph_sub = GetValueNode<FuncGraphPtr>(graph_value_node);
    auto parameters = graph_sub->parameters();
    if (LongToSize(param_pair.second - 1) >= parameters.size()) {
      MS_LOG_WITH_NODE(EXCEPTION, param_cnode) << "The index is out of range, index is: " << (param_pair.second - 1)
                                               << ", vector size is " << parameters.size();
    }
    std::pair<AnfNodePtr, int64_t> res = FindSubGraph(graph_sub, parameters[LongToSize(param_pair.second - 1)]);
    if (res.first != nullptr) {
      return res;
    }
  }
  return std::make_pair(nullptr, 0);
}

static void SetParameterSliceShape(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  auto parameters = root->parameters();
  FuncGraphManagerPtr manager = root->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &param_sub_map = manager->node_users();
  for (auto &parameter : parameters) {
    if (ParallelContext::GetInstance()->get_redundancy_node().find(parameter) !=
        ParallelContext::GetInstance()->get_redundancy_node().end()) {
      continue;
    }
    auto param_sub_set = param_sub_map.at(parameter);
    MS_EXCEPTION_IF_NULL(parameter->Shape());
    parallel::StrategyLayout::GetInstance()->SetParamGlobalShape(parameter);
    parallel::StrategyLayout::GetInstance()->SetParamType(parameter);

    auto iter = g_RefMap.find(parameter);
    if (iter != g_RefMap.cend()) {
      std::string group = SetParallelShape(parameter, g_RefMap[parameter], root);
      // find all forward nodes that use parameter in graphs and insert allgather if group is not empty
      SetSharedParameterFlag(root, parameter);
      ApplyParallelOptOnParam(manager, param_sub_set, root, parameter, group);
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
      std::string group = SetParallelShape(parameter, res, root);
      // find all forward nodes that use parameter in graphs and insert allgather if group is not empty
      SetSharedParameterFlag(root, parameter);
      ApplyParallelOptOnParam(manager, param_sub_set, root, parameter, group);
      MS_LOG(DEBUG) << "Parameter " << parameter->ToString() << " shape " << parameter->Shape()->ToString();
    }
  }
  g_RefMap.clear();
}

// if reshape's output connect to several primitive, return the first layout found
std::shared_ptr<TensorLayout> FindNextLayout(const AnfNodePtr &cnode, bool *next_is_reshape,
                                             mindspore::HashSet<AnfNodePtr> *visit, int make_tuple_index,
                                             int tuple_get_index, const std::shared_ptr<TensorLayout> &pre_layout);

std::shared_ptr<TensorLayout> FindNextLayoutForSpecialNode(const std::pair<AnfNodePtr, int64_t> &node_pair,
                                                           bool *next_is_reshape, mindspore::HashSet<AnfNodePtr> *visit,
                                                           bool *skip, int make_tuple_index, int tuple_get_index,
                                                           const std::shared_ptr<TensorLayout> &pre_layout) {
  auto use_apply = node_pair.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(use_apply);
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
      auto next_layout = FindNextLayout(fg_node, next_is_reshape, visit, make_tuple_index, tuple_get_index, pre_layout);
      if (next_layout != nullptr) {
        return next_layout;
      }
    }
  }

  if (IsPrimitiveCNode(use_apply, prim::kPrimTupleGetItem)) {
    auto temp = LongToInt(GetTupleGetItemIndex(use_apply));
    if (temp != make_tuple_index - 1 && make_tuple_index > 0) {
      *skip = true;
      return nullptr;
    }
    temp = make_tuple_index > 0 ? -1 : temp;
    auto next_layout = FindNextLayout(use_apply, next_is_reshape, visit, temp, -1, pre_layout);
    if (next_layout != nullptr) {
      return next_layout;
    }
  }

  if (IsPrimitiveCNode(use_apply, prim::kPrimMakeTuple)) {
    auto next_layout = FindNextLayout(use_apply, next_is_reshape, visit, node_pair.second, tuple_get_index, pre_layout);
    if (next_layout != nullptr) {
      return next_layout;
    }
  }
  return nullptr;
}

std::shared_ptr<TensorLayout> FindNextLayoutForParallelCareNode(const std::pair<AnfNodePtr, int64_t> &node_pair,
                                                                bool *next_is_reshape,
                                                                mindspore::HashSet<AnfNodePtr> *visit,
                                                                int make_tuple_index, int tuple_get_index,
                                                                const std::shared_ptr<TensorLayout> &pre_layout) {
  auto use_apply = node_pair.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(use_apply);
  if (!IsParallelCareNode(use_apply) || !use_apply->has_user_data<OperatorInfo>()) {
    return nullptr;
  }

  if (IsSupportNewShapeBaseNode(use_apply)) {
    MS_LOG(INFO) << "FindNextLayout success node " << use_apply->DebugString() << ", in support new shapebase ops";
    *next_is_reshape = false;
    auto layout = GetInputLayoutFromCNode(node_pair, make_tuple_index);
    if (IsPrimitiveCNode(node_pair.first)) {
      auto prim = GetCNodePrimitive(node_pair.first);
      MS_EXCEPTION_IF_NULL(prim);
      if (prim->HasAttr(kAttrFineGrainedInterleavedBlockIndex)) {
        layout.set_fine_grain_block_index(GetValue<int64_t>(prim->GetAttr(kAttrFineGrainedInterleavedBlockIndex)));
      }
    }
    return std::make_shared<TensorLayout>(layout);
  } else {
    auto node_pair_new = node_pair;
    if (make_tuple_index > 0) {
      node_pair_new.second = make_tuple_index;
    }
    MS_LOG(INFO) << "FindNextLayout success node " << use_apply->DebugString();
    *next_is_reshape = false;
    auto layout = GetInputLayoutFromCNode(node_pair_new, -1);
    if (IsPrimitiveCNode(node_pair.first)) {
      auto prim = GetCNodePrimitive(node_pair.first);
      MS_EXCEPTION_IF_NULL(prim);
      if (prim->HasAttr(kAttrFineGrainedInterleavedBlockIndex)) {
        layout.set_fine_grain_block_index(GetValue<int64_t>(prim->GetAttr(kAttrFineGrainedInterleavedBlockIndex)));
      }
    }
    return std::make_shared<TensorLayout>(layout);
  }
}

std::shared_ptr<TensorLayout> FindNextLayout(const AnfNodePtr &cnode, bool *next_is_reshape,
                                             mindspore::HashSet<AnfNodePtr> *visit, int make_tuple_index,
                                             int tuple_get_index, const std::shared_ptr<TensorLayout> &pre_layout) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(next_is_reshape);
  MS_EXCEPTION_IF_NULL(visit);
  MS_EXCEPTION_IF_NULL(cnode->func_graph());
  FuncGraphManagerPtr manager = cnode->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto node_set = NextNodeUsers(cnode);
  for (auto &node_pair : node_set) {
    auto use_apply = node_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(use_apply);
    if (visit->find(use_apply) != visit->end()) {
      continue;
    }
    (void)(visit->insert(use_apply));

    if (IsPrimitiveCNode(use_apply, prim::kPrimPrint) || IsPrimitiveCNode(use_apply, prim::kPrimTensorDump)) {
      return pre_layout;
    }

    bool skip = false;
    auto next_layout = FindNextLayoutForSpecialNode(node_pair, next_is_reshape, visit, &skip, make_tuple_index,
                                                    tuple_get_index, pre_layout);
    if (next_layout != nullptr) {
      return next_layout;
    }
    if (skip) {
      continue;
    }

    if (!IsValueNode<Primitive>(use_apply->input(0))) {
      continue;
    }

    if (IsPrimitiveCNode(use_apply, prim::kPrimReshape)) {
      *next_is_reshape = true;
      continue;
    }
    if (IsOneOfPrimitiveCNode(use_apply, {prim::kPrimDepend, prim::kPrimUpdateState}) && node_pair.second != 1) {
      continue;
    }

    next_layout = FindNextLayoutForParallelCareNode(node_pair, next_is_reshape, visit, make_tuple_index,
                                                    tuple_get_index, pre_layout);
    if (next_layout != nullptr) {
      return next_layout;
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

void InitReshapeOpInfo(const std::vector<AnfNodePtr> &all_nodes) {
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
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Failure:Primitive " << prim->ToString() << " OperatorInstance is nullptr";
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
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Setting strategy for Reshape goes for nothing!";
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
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Failure:operator " << prim->ToString() << " init failed";
    }
  }
  MS_LOG(DEBUG) << "=============Do ReshapeInit end=============";
}
}  // namespace

void ParallelPreprocessor::HandleRootReshapeAndSaveStrategy(const std::vector<AnfNodePtr> &all_nodes) {
  // If root graph has reshape op. Find the corresponding parameter.
  // Reshape's shape is the shape of the parameter.
  auto executor = pipeline::GraphExecutorPy::GetInstance();
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    if (IsValueNode<FuncGraph>(cnode->input(0))) {
      cnode->set_in_forward_flag(true);
      continue;
    }
    if (!IsValueNode<Primitive>(cnode->input(0))) {
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
    if (prim == nullptr || prim->name() != RESHAPE) {
      continue;
    }

    auto origin_dst_shape_val = cnode->input(2)->cast<ValueNodePtr>();
    if (!origin_dst_shape_val) {
      continue;
    }

    Shape origin_dst_shape = GetValue<std::vector<int64_t>>(origin_dst_shape_val->value());
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

void ParallelPreprocessor::HandleForwardMakeTupleAndMakeList(const std::vector<AnfNodePtr> &all_nodes) {
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

    AnfNodePtr make_tuple_list_next_node;
    int input_pos;
    // MakeTuple has multiple users, each user's TensorInfo must be same.
    std::tie(make_tuple_list_next_node, input_pos) = CheckMakeTupleSplit(node, manager);
    if (make_tuple_list_next_node == nullptr) {
      continue;
    }
    auto make_tuple_list_next_cnode = make_tuple_list_next_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple_list_next_cnode);
    if (!IsSomePrimitiveList(make_tuple_list_next_cnode, INPUT_IS_TUPLE_OR_LIST_OPS)) {
      continue;
    }

    OperatorInfoPtr op_info = SetMakeListForIFA(cnode, make_tuple_list_next_cnode);
    if (op_info == nullptr) {
      op_info = CreateOperatorInfoForMakeTuple(cnode, make_tuple_list_next_cnode, input_pos - 1);
    }
    MS_EXCEPTION_IF_NULL(op_info);
    cnode->set_user_data<OperatorInfo>(op_info);
  }
}

void ParallelPreprocessor::ExtractInformation(const std::vector<AnfNodePtr> &all_nodes) {
  SetStridedSliceSplitStrategy(all_nodes);
  for (auto &node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (!StrategyUtils::CheckExtractInformation(cnode) || IsPrimitiveCNode(node, prim::kPrimSend)) {
      continue;
    }

    if (CheckShardingPropagation()) {
      auto find_iter = cnode->attrs().find(OP_INFO_CREATED);
      if (find_iter != cnode->attrs().end()) {
        auto op = GetDistributeOperator(cnode);
        if (op != nullptr) {
          op->set_cnode(cnode);
        }
        continue;
      }
    }

    StrategyUtils::SetGetNextLayout(cnode);
    StrategyUtils::SetVirtualDatasetStrategy(cnode);
    ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);

    OperatorInfoPtr operator_ = CreateOperatorInfo(cnode);
    MS_EXCEPTION_IF_NULL(operator_);
    MS_EXCEPTION_IF_NULL(prim);

    if (prim->name() == RESHAPE) {
      cnode->set_user_data<OperatorInfo>(operator_);
      continue;
    }

    StrategyUtils::ExtractStrategyAndInit(cnode, prim, operator_);
    cnode->set_user_data<OperatorInfo>(operator_);
  }
}

void ParallelPreprocessor::PipelinePreProcess() {
  if (processor_context_->pipeline_stages > 1 && processor_context_->is_pp_interleave) {
    auto pipeline_processor =
      std::make_shared<PipelinePostProcess>(processor_context_->manager, g_device_manager->stage_id(),
                                            processor_context_->pipeline_stages, processor_context_->root);
    MS_EXCEPTION_IF_NULL(pipeline_processor);
    pipeline_processor->Init(processor_context_->all_nodes);
    pipeline_processor->ModifySendRecvAttr(processor_context_->all_nodes);
    processor_context_->pipeline_processor = pipeline_processor;
  }
}

void ParallelPreprocessor::MarkAndModifyGraph() {
  auto root = processor_context_->root;
  auto manager = processor_context_->manager;
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(manager);

  RecordFlopsOriginShape(manager);

  AnfNodePtr ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  std::reverse(all_nodes.begin(), all_nodes.end());

  bool grad_accumulation_shard = GradAccumulationStep(all_nodes) > 1;
  if ((processor_context_->pipeline_stages > 1 && processor_context_->is_pp_interleave) || grad_accumulation_shard) {
    bool merged = MergeConcatSlice(all_nodes, manager);
    if (merged) {
      all_nodes = TopoSort(ret, SuccDeeperSimple);
    }
  }

  // Different pass may have different cnodes handle, need clear the old first and then reset cnodes
  if (CheckShardingPropagation()) {
    ClearCnodesForOperator(all_nodes);
  }
  // Insert TupleToTensor for FA if actual_seq_len input is tuple type.
  PreProcessActualSeqLenInputForFlashAttentionScore(root, all_nodes);

  MicroBatchPreProcess(root, manager, all_nodes);
  // mark the forward cnodes, parallel only care these nodes
  MarkForwardCNode(root);
  UpdateMicroBatchInterleavedStatus(all_nodes);
  if (processor_context_->parallel_mode != kAutoParallel) {
    TOTAL_OPS = 0;
    ExceptionIfHasCommunicationOp(all_nodes);

    if (IsInsertVirtualOutput(root)) {
      InsertVirtualOutput(root, all_nodes);
      AnfNodePtr ret_after = root->get_return();
      MS_EXCEPTION_IF_NULL(ret_after);
      all_nodes = TopoSort(ret_after, SuccDeeperSimple);
    }
  }

  SetCastForParamNotRecompute(all_nodes);
  processor_context_->all_nodes = all_nodes;
}

void ParallelPreprocessor::SetOperatorInfo() {
  auto &all_nodes = processor_context_->all_nodes;
  if (processor_context_->parallel_mode != kAutoParallel || CheckShardingPropagation()) {
    // semi: extract shape and strategy, set operator_info
    // auto: create opInfo for step parallel generated op and reset cnode for existing ones
    ExtractInformation(all_nodes);

    // dump IR detail in semi_auto_parallel and recursive_programming mode
    std::string env_var = common::GetEnv("MS_DEV_DUMP_IR_PARALLEL_DETAIL");
    auto parallel_mode = ParallelContext::GetInstance()->parallel_mode();
    auto strategy_search_mode = ParallelContext::GetInstance()->strategy_search_mode();
    if (!env_var.empty() && env_var == kDumpIrParallelDetail &&
        (parallel_mode == "semi_auto_parallel" || strategy_search_mode == "recursive_programming")) {
      for (const auto &node : all_nodes) {
        if (node->has_user_data<OperatorInfo>()) {
          auto operator_info = node->user_data<OperatorInfo>();
          MS_EXCEPTION_IF_NULL(operator_info);

          MS_EXCEPTION_IF_NULL(operator_info);
          TensorMaps inputs_tensor_map = operator_info->inputs_tensor_map();
          TensorMaps outputs_tensor_map = operator_info->outputs_tensor_map();
          Shape device_matrix = operator_info->dev_matrix_shape();

          auto prim = GetCNodePrimitive(node);
          MS_EXCEPTION_IF_NULL(prim);

          prim->set_attr(INPUTS_TENSOR_MAP, MakeValue(inputs_tensor_map));
          prim->set_attr(OUTPUTS_TENSOR_MAP, MakeValue(outputs_tensor_map));
          prim->set_attr(DEVICE_MATRIX, MakeValue(device_matrix));
        }
      }
    }
  }

  InitReshapeOpInfo(all_nodes);
}

void ParallelPreprocessor::SetParameterInfo() {
  auto root = processor_context_->root;
  auto manager = processor_context_->manager;
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(manager);
  auto &all_nodes = processor_context_->all_nodes;

  // if the input or parameter has multiple users, check whether its split strategies are consistent.
  CheckParameterSplit(all_nodes);

  HandleSymbolicKeyInstance(root, all_nodes);

  // set parallel slice shape for parameters
  SetParameterSliceShape(root);

  // handle input is not used
  HandleNoUsedParameter(root);

  // set the shape for optimizer's clone tensor
  SetClonedTensorShapeForOptimizer(root);

  HandleCameAndAdaFactorOpt(root, all_nodes, manager);

  processor_context_->adasum_param_tensor_layout_map = AdaSumParamTensorLayout(root);
  bool is_apply_adasum = HandleAdaSum(root, all_nodes, &(processor_context_->adasum_param_tensor_layout_map));
  processor_context_->is_apply_adasum = is_apply_adasum;
}

void ParallelPreprocessor::Process() {
  auto root = processor_context_->root;
  MS_EXCEPTION_IF_NULL(root);
  auto &all_nodes = processor_context_->all_nodes;

  MarkAndModifyGraph();

  SetOperatorInfo();

  HandleRootReshapeAndSaveStrategy(all_nodes);

  HandleForwardMakeTupleAndMakeList(all_nodes);

  SetParameterInfo();

  if (MergeEntireShapeForDynamic(root) != Status::SUCCESS) {
    MS_LOG(EXCEPTION) << "Merge entire shape for dynamic shape failed.";
  }

  PipelinePreProcess();
}
}  // namespace parallel
}  // namespace mindspore
