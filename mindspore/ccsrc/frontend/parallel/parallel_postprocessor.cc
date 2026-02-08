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

#include "frontend/parallel/parallel_postprocessor.h"

#include <cinttypes>
#include <algorithm>
#include <chrono>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <queue>
#include <vector>
#include <unordered_map>

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
#include "frontend/parallel/graph_util/parallel_tensordump.h"
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
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace parallel {
namespace {
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
    if (micro_mirror->HasAttr(kPipelineSendSharedParam)) {
      continue;
    }
    size_t max_loop = 0;
    while (micro_mirror != nullptr && micro_mirror->func_graph() != root && max_loop < SIZE_FOUR) {
      auto anf_micro_mirror = MoveSingeMirrorOutCallFunc(micro_mirror);
      if (anf_micro_mirror != nullptr) {
        micro_mirror = anf_micro_mirror->cast<CNodePtr>();
      } else {
        micro_mirror = nullptr;
      }
      max_loop++;
    }
    if (micro_mirror != nullptr && micro_mirror->func_graph() != root) {
      DumpGraph(root, "move_mirror_failed");
      MS_LOG(EXCEPTION) << "Move mirror operator to root graph failed, micro_mirror:" << micro_mirror->DebugString();
    }
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

static void InsertAllReduceForNormValue(const AnfNodePtr &res_node) {
  auto cnode = res_node->cast<CNodePtr>();
  auto graphs = res_node->func_graph();
  MS_EXCEPTION_IF_NULL(graphs);
  auto manager = graphs->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (!IsSomePrimitive(cnode, EXPAND_DIMS)) {
    MS_LOG(ERROR) << "Expected the operator expand_dims, but found the " << GetPrimName(cnode)
                  << "This may cause the calculation of the global norm incorrect";
    return;
  }
  auto pipeline_stages = ParallelContext::GetInstance()->pipeline_stage_split_num();
  auto find_node = res_node;
  uint32_t limits = 0;
  const uint32_t MAX_BFS_DEPTH = 15;
  bool find_sqrt = false;
  std::queue<AnfNodePtr> anf_queue;
  anf_queue.push(res_node);
  while (!anf_queue.empty() && limits < MAX_BFS_DEPTH) {
    auto queue_end = anf_queue.front();
    anf_queue.pop();
    const auto &user_set = manager->node_users()[queue_end];
    if (user_set.empty()) {
      continue;
    }
    for (const auto &pair : user_set) {
      anf_queue.push(pair.first);
      if (IsSomePrimitive(pair.first->cast<CNodePtr>(), SQRT)) {
        find_node = pair.first;
        find_sqrt = true;
        break;
      }
    }
    if (find_sqrt) {
      break;
    }
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
    MS_LOG_WITH_NODE(EXCEPTION, sqrt_node)
      << "Create the communication group for allreduce in calculating global norm failed, "
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
      MS_LOG_WITH_NODE(EXCEPTION, sqrt_node)
        << "Create the communication group for allreduce in calculating global norm "
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
    if (!IsSomePrimitiveList(cnode, {ENVIRONGET, MUL, MULS, SQUARE, REDUCE_SUM, EXPAND_DIMS, DEPEND, CAST, REF_TO_EMBED,
                                     EMBED, LOAD, SUM_EXT})) {
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

static void InsertRealDivOpToNodeInput(const CNodePtr &node, int64_t scale, const string &instance_name) {
  MS_EXCEPTION_IF_NULL(node);
  if (scale == 0) {
    MS_LOG_WITH_NODE(EXCEPTION, node)
      << "Find the scale value is 0, you should check the mirror operators's group size.";
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

static void InsertDivAndAllReduceForNorm(const NodeUsersMap &node_user_map, const AnfNodePtr &parameter,
                                         uint32_t dev_num) {
  auto params_user_set = node_user_map.at(parameter);
  for (auto &param_pair : params_user_set) {
    auto cnode = param_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->in_forward_flag()) {
      continue;
    }
    constexpr size_t bfs_depth = 100;
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
    MS_EXCEPTION_IF_NULL(cnode);
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
    MS_EXCEPTION_IF_NULL(param_anf_node);
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
        MS_EXCEPTION_IF_NULL(input);
        auto input_abs = input->abstract();
        if (IsValueNode<None>(input)) {
          continue;
        }

        if (IsPrimitiveCNode(input, prim::kPrimTupleToTensor)) {
          // Eliminate TupleToTensor
          auto input_ptr = input->cast<CNodePtr>();
          MS_EXCEPTION_IF_NULL(input_ptr);
          manager->SetEdge(fa_cnode, index + 1, input_ptr->input(kIndex1));
          MS_LOG(DEBUG) << "Eliminate TensorToTuple for " << fa_cnode->fullname_with_scope() << ", index is "
                        << index + 1;
        } else {
          // Transfer Tensor to Tuple
          auto tensor_to_tuple_cnode =
            fa_cnode->func_graph()->NewCNode({NewValueNode(prim::kPrimTensorToTuple), input});
          manager->SetEdge(fa_cnode, index + 1, tensor_to_tuple_cnode);
          auto prim = GetCNodePrimitive(tensor_to_tuple_cnode);
          MS_EXCEPTION_IF_NULL(prim);
          prim->AddAttr(kAttrRecompute, MakeValue(true));
          MS_LOG(DEBUG) << "Insert TensorToTuple for " << fa_cnode->fullname_with_scope() << ", index is " << index + 1;
        }
      }
    }
  }
}

static void BroadcastMultiOutputs(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager, const Group &group) {
  MS_EXCEPTION_IF_NULL(root);
  auto output = root->get_return()->input(1)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(output);
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

void AssignStrategyMap(const StrategyPtr &stra, const std::string &strategy_key_name, StrategyMap *stra_map) {
  if (stra) {
    (*stra_map)[strategy_key_name] = stra;
  } else {
    Strategies new_stra_v;
    StrategyPtr new_stra = std::make_shared<Strategy>(g_device_manager->stage_id(), new_stra_v);
    (*stra_map)[strategy_key_name] = new_stra;
  }
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

static void MicroBatchPostProcess(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes) {
  const auto &pipeline_stages = ParallelContext::GetInstance()->pipeline_stage_split_num();
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
}  // namespace

void CheckpointOnline(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root) {
  StrategyMap stra_map;
  TensorInfoMap tensor_info_map;
  ManualShapeMap manual_shape_map;
  CheckpointStrategy(all_nodes, root, &stra_map, &tensor_info_map, &manual_shape_map);
  if (StrategyCheckpoint::GetInstance().SaveOnline(stra_map, tensor_info_map, manual_shape_map) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Save strategy checkpoint online failed.";
  }
}

void CheckpointOffline(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root) {
  if (!StrategyCheckpoint::GetInstance().SaveCheckPointOn()) {
    MS_LOG(INFO) << "The param of save_checkpoint_on_ is False";
    return;
  }

  StrategyMap stra_map;
  TensorInfoMap tensor_info_map;
  ManualShapeMap manual_shape_map;
  CheckpointStrategy(all_nodes, root, &stra_map, &tensor_info_map, &manual_shape_map);
  if (StrategyCheckpoint::GetInstance().Save(stra_map, tensor_info_map, manual_shape_map) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Save strategy checkpoint failed. Please check the file permission or the disk usage.";
  }
}

void CheckpointStrategy(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root, StrategyMap *stra_map,
                        TensorInfoMap *tensor_info_map, ManualShapeMap *manual_shape_map) {
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
    param_names.erase(std::remove_if(param_names.begin(), param_names.end(),
                                     [](const auto &param_pair) {
                                       return param_pair.second->param_info()->is_pipeline_shared_param();
                                     }),
                      param_names.end());
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
      AssignStrategyMap(stra, strategy_key_name, stra_map);

      for (auto param_name_pair : param_names) {
        (*tensor_info_map)[param_name_pair.first] = param_name_pair.second->user_data<TensorLayout>();
      }
      AssignManualShapeMapForGather(operator_info, param_name, manual_shape_map);
    }
  }
  for (auto &cloned_parameter_node : root->parameters()) {
    MS_EXCEPTION_IF_NULL(cloned_parameter_node);
    auto cloned_parameter = cloned_parameter_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(cloned_parameter);

    auto cloned_param_info = cloned_parameter->param_info();
    if (cloned_param_info != nullptr) {
      if (cloned_param_info->is_pipeline_shared_param()) {
        continue;
      }
    }

    if (!ParameterIsCloned(cloned_parameter_node) && !IsStrategySaved(cloned_parameter_node)) {
      continue;
    }
    std::string cloned_param_name = cloned_parameter_node->cast<ParameterPtr>()->name();
    auto cloned_param_layout = cloned_parameter_node->user_data<TensorLayout>();
    if (cloned_param_layout == nullptr) {
      continue;
    }
    (*tensor_info_map)[cloned_param_name] = cloned_param_layout;
  }
}

static void SetSharedAttrForOptimizerParameter(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  auto root_params = root->parameters();
  AnfNodePtrList sub_root_params;
  GetSubRootParams(root_params, &sub_root_params);
  for (auto &cloned_parameter_node : root_params) {
    if (ParallelContext::GetInstance()->get_redundancy_node().count(cloned_parameter_node)) {
      continue;
    }
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
    for (auto &be_cloned_parameter_node : sub_root_params) {
      MS_EXCEPTION_IF_NULL(be_cloned_parameter_node);
      auto be_cloned_parameter = be_cloned_parameter_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(be_cloned_parameter);
      auto param_value_in = be_cloned_parameter->param_info();
      // get the be cloned index
      MS_EXCEPTION_IF_NULL(param_value_in);
      auto &be_cloned_index = param_value_in->be_cloned_index();
      if (std::find(be_cloned_index.begin(), be_cloned_index.end(), cloned_index) != be_cloned_index.end()) {
        // add shared attr for cloned parameters
        if (param_value_in->is_pipeline_shared_param()) {
          param_value->set_is_pipeline_shared_param(true);
        }
      }
    }
  }
}

void ParallelPostprocessor::PipelinePostProcessStep1() {
  auto pipeline_processor = processor_context_->pipeline_processor;
  if (pipeline_processor == nullptr) {
    const auto &root = processor_context_->root;
    MS_EXCEPTION_IF_NULL(root);
    const auto &all_nodes_pp = processor_context_->all_nodes;
    CheckpointOnline(all_nodes_pp, root);
  }
  if (pipeline_processor != nullptr) {
    MS_EXCEPTION_IF_NULL(pipeline_processor);
    if (parallel::ParallelContext::GetInstance()->fine_grained_micro_interleaved_size() > 0) {
      auto root = processor_context_->root;
      MS_EXCEPTION_IF_NULL(root);
      auto all_nodes_pp = TopoSort(root->get_return(), SuccDeeperSimple);
      pipeline_processor->Init(all_nodes_pp);
      CheckpointOnline(all_nodes_pp, root);
      pipeline_processor->GraphPartition(all_nodes_pp);
    } else {
      const auto &root = processor_context_->root;
      MS_EXCEPTION_IF_NULL(root);
      const auto &all_nodes_pp = processor_context_->all_nodes;
      CheckpointOnline(all_nodes_pp, root);
      pipeline_processor->GraphPartition(processor_context_->all_nodes);
    }
    AddVirtualAssignKvCache(processor_context_->root);
    pipeline_processor->ElimGraphStage();
    pipeline_processor->ModifyParameterList();
  }
  parallel::ParallelContext::GetInstance()->set_fine_grained_micro_interleaved_size(-1);
}

void ParallelPostprocessor::PipelinePostProcessStep2() {
  auto pipeline_processor = processor_context_->pipeline_processor;
  if (pipeline_processor != nullptr) {
    pipeline_processor->HandleSendParam();
    MarkForwardCNode(processor_context_->root);
  }
}

static void DecorateDumpPathIfDumpOps(const AnfNodePtrList &all_nodes, const FuncGraphManagerPtr &manager) {
  // For TensorDump and DumpGradient, If dump mode is 'in', suffix 'in' for path.
  // For example: dumppath.npy -> dumppath_in.npy.
  for (const AnfNodePtr &node : all_nodes) {
    if (IsSomePrimitive(node->cast<CNodePtr>(), DUMPGRADIENT)) {
      CNodePtr dump_gradient = node->cast<CNodePtr>();
      const std::string hook_mode = GetDumpHookInputOutputAttr(dump_gradient);
      if (hook_mode != IN_MODE) {
        continue;
      }
      const std::string ori_path = GetValue<std::string>(GetValueNode(dump_gradient->input(kIndex1)));
      (void)manager->SetEdge(dump_gradient, kIndex1, NewValueNode(MakeValue(GetInModeSuffixedDumpPath(ori_path))));
    } else if (IsSomePrimitive(node->cast<CNodePtr>(), TENSORDUMP)) {
      CNodePtr tensordump = node->cast<CNodePtr>();
      std::string dump_mode = GetDumpInputOutputAttr(tensordump);
      if (dump_mode != IN_MODE) {
        continue;
      }
      const std::string ori_path = GetValue<std::string>(GetValueNode(tensordump->input(kIndex1)));
      (void)manager->SetEdge(tensordump, kIndex1, NewValueNode(MakeValue(GetInModeSuffixedDumpPath(ori_path))));
    }
  }
}

void ParallelPostprocessor::Process() {
  auto root = processor_context_->root;
  auto manager = processor_context_->manager;
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(manager);

  SplitNotParallelCareOpsInterleaved(root);
  EraseVirtualConverter(root);
  if (processor_context_->is_apply_adasum) {
    HandleMirrorInAdaSum(root, &(processor_context_->adasum_param_tensor_layout_map));
  }

  PipelinePostProcessStep1();
  SetSharedAttrForOptimizerParameter(root);

  // save strategy as checkpoint for multi-train
  auto all_nodes_after_pp = TopoSort(root->get_return(), SuccDeeperSimple);

  if (StrategyCheckpoint::GetInstance().SaveCheckPointOn()) {
    CheckpointOffline(all_nodes_after_pp, root);
  }
  DecorateDumpPathIfDumpOps(all_nodes_after_pp, manager);
  auto comm_group = FindCommonMirrorGroup(root);
  StrategyCheckpoint::GetInstance().set_common_mirror_group(comm_group);

  MoveMicroMirrorOutCallFunc(root);
  HandleGlobalNormScale(root, manager);

  PipelinePostProcessStep2();
  MergeMicroMirrorForSharedParameter(root);

  // Insert TensorToTuple for FlashAttentionScore if input actual_seq_len is tensor
  PostProcessActualSeqLenInputForFlashAttentionScore(root, all_nodes_after_pp);
  BroadcastLastResult(root, manager);
  MicroBatchPostProcess(root, all_nodes_after_pp);
  UpdateParamSymbolicShape(root);
}
}  // namespace parallel
}  // namespace mindspore
