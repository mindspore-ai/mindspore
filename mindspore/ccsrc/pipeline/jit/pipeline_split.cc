/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include <set>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "pipeline/jit/pipeline_split.h"
#include "utils/ms_context.h"
#include "include/common/utils/comm_manager.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/pipeline_transformer/pipeline_transformer.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/parameter_manager.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/backend/distributed/ps/util.h"
#include "include/backend/distributed/ps/ps_context.h"
#endif

namespace mindspore {
namespace pipeline {
std::string GetWorldGroup() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string world_group;
  std::string backend = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (backend == kAscendDevice) {
    world_group = parallel::HCCL_WORLD_GROUP;
  } else if (backend == kGPUDevice) {
    world_group = parallel::NCCL_WORLD_GROUP;
  } else {
    MS_LOG(EXCEPTION) << "Invalid backend: " << backend;
  }
  return world_group;
}

static int64_t GetRank() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto world_group = GetWorldGroup();
  int64_t global_rank = parallel::ParallelContext::GetInstance()->global_rank();
  uint32_t rank_id = 0;
  if (!parallel::ParallelContext::GetInstance()->global_rank_is_set()) {
    if (!CommManager::GetInstance().GetRankID(world_group, &rank_id)) {
      MS_LOG(EXCEPTION) << "Get rank id failed.";
    }
    global_rank = UintToInt(rank_id);
  }
  return global_rank;
}

static int64_t InferStage(int64_t rank_id, int64_t stage_num, int64_t device_num) {
  if (stage_num == 0) {
    MS_LOG(EXCEPTION) << "stage_num is zero";
  }
  if (device_num % stage_num != 0) {
    MS_LOG(EXCEPTION) << "Device_num must be divisible by the stage_num, got device_num: " << device_num
                      << "stage_num: " << stage_num;
  }
  auto per_stage_rank_num = device_num / stage_num;
  return rank_id / per_stage_rank_num;
}

static bool HasVirtualDataset(const std::vector<AnfNodePtr> &all_nodes) {
  for (auto &node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimVirtualDataset)) {
      return true;
    }
  }
  return false;
}

static CNodePtr CreateTupleGetItem(const AnfNodePtr &node, size_t index, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto idx = NewValueNode(SizeToLong(index));
  MS_EXCEPTION_IF_NULL(idx);
  auto imm = std::make_shared<Int64Imm>(SizeToLong(index));
  auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(imm);
  idx->set_abstract(abstract_scalar);
  CNodePtr tuple_get_item = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), node, idx});
  MS_EXCEPTION_IF_NULL(tuple_get_item);
  tuple_get_item->set_scope(node->scope());
  auto input_abstract_tuple = node->abstract()->cast_ptr<abstract::AbstractTuple>();
  MS_EXCEPTION_IF_NULL(input_abstract_tuple);
  auto tuple_get_item_abstract = input_abstract_tuple->elements()[index];
  MS_EXCEPTION_IF_NULL(tuple_get_item_abstract);
  tuple_get_item->set_abstract(tuple_get_item_abstract);
  return tuple_get_item;
}

static CNodePtr CreateVirtualDataset(const FuncGraphPtr &func_graph) {
  mindspore::parallel::OperatorAttrs attrs;
  ValuePtr pyop_instance = mindspore::parallel::CreateOpInstance(attrs, mindspore::parallel::VIRTUAL_DATA_SET,
                                                                 mindspore::parallel::VIRTUAL_DATA_SET);
  auto value_node = NewValueNode(pyop_instance);
  std::vector<AbstractBasePtr> abstract_list;
  std::vector<AnfNodePtr> virtual_dataset_node_inputs = {value_node};
  for (size_t index = 0; index < func_graph->get_inputs().size(); index++) {
    if (!HasAbstractMonad(func_graph->get_inputs()[index])) {
      auto graph_input_index = func_graph->get_inputs()[index];
      auto virtual_dataset_abstract = graph_input_index->abstract()->Clone();
      MS_EXCEPTION_IF_NULL(virtual_dataset_abstract);
      (void)abstract_list.emplace_back(virtual_dataset_abstract);
      virtual_dataset_node_inputs.push_back(func_graph->get_inputs()[index]);
    }
  }
  CNodePtr virtual_dataset_node = func_graph->NewCNode(virtual_dataset_node_inputs);
  MS_EXCEPTION_IF_NULL(virtual_dataset_node);
  virtual_dataset_node->set_in_forward_flag(true);
  virtual_dataset_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  return virtual_dataset_node;
}

static std::set<FuncGraphPtr> FindForwardGraph(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes) {
  std::set<FuncGraphPtr> graph_sets;
  auto manager = root->manager();
  if (!parallel::IsAutoParallelCareGraph(root)) {
    return graph_sets;
  }
  std::set<AnfNodePtr> input_parameters;
  for (auto &anf_param : root->parameters()) {
    auto param = anf_param->cast_ptr<Parameter>();
    if (!param->has_default()) {
      (void)input_parameters.insert(anf_param);
    }
  }
  for (const auto &input_parameter : input_parameters) {
    auto node_users_map = root->manager()->node_users();
    auto node_users = node_users_map[input_parameter];
    for (auto node_user : node_users) {
      auto cnode = node_user.first->cast_ptr<CNode>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (IsValueNode<Primitive>(cnode->inputs()[0]) ||
          (IsValueNode<FuncGraph>(cnode->inputs()[0]) && !parallel::IsTraining(manager))) {
        (void)graph_sets.insert(cnode->func_graph());
      }
    }
  }
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast_ptr<CNode>();
    if ((cnode->size() < NODE_INPUT_NUM) || !IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    auto expect_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    FuncGraphPtr fun_graph = nullptr;
    if (expect_prim->name() == mindspore::parallel::J || expect_prim->name() == mindspore::parallel::SHARD) {
      if (IsValueNode<FuncGraph>(cnode->inputs()[1])) {
        fun_graph = GetValueNode<FuncGraphPtr>(cnode->inputs()[1]);
      } else {
        fun_graph = node->func_graph();
      }
      (void)graph_sets.insert(fun_graph);
    }
  }
  return graph_sets;
}

static void InsertVirtualDataset(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes) {
  MS_EXCEPTION_IF_NULL(root);
  std::set<FuncGraphPtr> forward_graph_set = FindForwardGraph(root, all_nodes);
  for (const auto &forward_graph : forward_graph_set) {
    FuncGraphManagerPtr manager = forward_graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    std::vector<AnfNodePtr> graph_inputs = forward_graph->get_inputs();
    // SetEdge will be called later, so deep copy is required.
    auto node_user_map = manager->node_users();
    auto virtual_dataset_node = CreateVirtualDataset(forward_graph);
    std::map<size_t, CNodePtr> parameter_index_map;
    for (size_t index = 0; index < graph_inputs.size(); index++) {
      if (HasAbstractMonad(graph_inputs[index])) {
        continue;
      }
      auto node_users = node_user_map[graph_inputs[index]];
      for (const auto &node_user : node_users) {
        auto cnode = node_user.first->cast<CNodePtr>();
        for (size_t input_index = 1; input_index < cnode->inputs().size(); input_index++) {
          if (!IsValueNode<Primitive>(cnode->inputs()[0]) && !IsValueNode<FuncGraph>(cnode->inputs()[0]) &&
              !IsPrimitiveCNode(cnode->input(0), prim::kPrimVmap)) {
            continue;
          }
          bool is_node_input_flag = !(IsValueNode<mindspore::tensor::Tensor>(cnode->inputs()[input_index]) ||
                                      IsValueNode<ValueList>(cnode->inputs()[input_index]) ||
                                      IsValueNode<ValueTuple>(cnode->inputs()[input_index]));
          auto node_input_iter = find(graph_inputs.begin(), graph_inputs.end(), cnode->inputs()[input_index]);
          bool is_match = node_input_iter != graph_inputs.end() && is_node_input_flag &&
                          !HasAbstractMonad(cnode->inputs()[input_index]);
          if (!is_match) {
            continue;
          }
          size_t node_input_index = LongToSize(node_input_iter - graph_inputs.begin());
          if (parameter_index_map.empty() || parameter_index_map.count(node_input_index) == 0) {
            parameter_index_map[node_input_index] =
              CreateTupleGetItem(virtual_dataset_node, node_input_index, forward_graph);
          }
          manager->SetEdge(cnode, SizeToInt(input_index), parameter_index_map[node_input_index]);
          manager->SetEdge(parameter_index_map[node_input_index], 1, virtual_dataset_node);
        }
      }
    }
  }
}

// If graph has shard node, set flag 'kPynativeShard' for root graph
void SetPynativeShardFlagIfHasShardNode(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes) {
  for (auto &node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimShard)) {
      root->set_flag(parallel::kPynativeShard, true);
      break;
    }
  }
}

// Only auto_parallel and semi_auto_parallel support PipelineSplit
bool PipelineSplit(const ResourcePtr &res) {
#if defined(__linux__) && defined(WITH_BACKEND)
  if (ps::PSContext::instance()->is_server() || ps::PSContext::instance()->is_scheduler()) {
    return true;
  }
#endif
  MS_EXCEPTION_IF_NULL(res);
  auto parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode != parallel::kSemiAutoParallel && parallel_mode != parallel::kAutoParallel) {
    MS_LOG(INFO) << "Only auto_parallel and semi_auto_parallel support pipeline split.";
    return true;
  }

  auto manager = res->manager();
  auto root = res->func_graph();
  AnfNodePtr ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);

  SetPynativeShardFlagIfHasShardNode(root, all_nodes);
  if (!HasVirtualDataset(all_nodes)) {
    InsertVirtualDataset(root, all_nodes);
  }
  auto stage_num = parallel::ParallelContext::GetInstance()->pipeline_stage_split_num();
  if (stage_num <= 1) {
    MS_LOG(INFO) << "The parameter 'stage_num' is: " << stage_num << ". No need Pipeline split.";
    return true;
  }

  auto global_rank = GetRank();
  auto world_group = GetWorldGroup();
  uint32_t world_rank_size = 0;
  int64_t device_num = 0;
  if (!parallel::ParallelContext::GetInstance()->device_num_is_set()) {
    if (!CommManager::GetInstance().GetRankSize(world_group, &world_rank_size)) {
      MS_LOG(EXCEPTION) << "Get rank size failed";
    }
    device_num = UintToInt(world_rank_size);
    MS_LOG(INFO) << "Get device num from communication model, the device num is  " << device_num;
  } else {
    device_num = parallel::ParallelContext::GetInstance()->device_num();
  }

  if (device_num < 1) {
    MS_LOG(ERROR) << "For 'PipelineSplit', the argument 'device_num' must be positive, "
                     "but got the value of device_num: "
                  << device_num;
  }
  if (global_rank < 0) {
    MS_LOG(ERROR) << "For 'PipelineSplit', the argument 'global_rank' must be nonnegative, "
                     "but got the value of global_rank: "
                  << global_rank;
  }
  auto stage = InferStage(global_rank, stage_num, device_num);
  static const auto gen_mask_not_fusion = (common::GetEnv("GENMASK_NOT_FUSION") == "1");
  auto per_stage_rank_num = device_num / stage_num;
  if (parallel::ParallelInit() != parallel::SUCCESS) {
    MS_LOG(EXCEPTION) << "parallel init failed.";
  }
  auto transformer =
    std::make_shared<parallel::PipelineTransformer>(manager, stage, root, global_rank, per_stage_rank_num);
  // step1: Do color graph
  transformer->Coloring();
  transformer->MainGraph();
  // step2: Do color broadcast
  transformer->BroadCastColoring();
  if (!gen_mask_not_fusion) {
    transformer->LabelGenMaskFusion();
  }
  transformer->LabelMicroBatch();
  // step3: Handle shared parameters
  transformer->ParameterColoring();
  // step4: Cut Graph
  transformer->CutGraph();
  // step5: Handle Sens
  if (parallel::IsTraining(manager)) {
    transformer->CoverSensShape();
  }
  // step6: Elim Graph stages and no used parameter
  transformer->ModifyParameterList();
  transformer->ElimGraphStage();
  return true;
}
}  // namespace pipeline
}  // namespace mindspore
