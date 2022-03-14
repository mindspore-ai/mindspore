/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#if ((defined ENABLE_CPU) && (!defined _WIN32) && !defined(__APPLE__))
#include "ps/util.h"
#include "ps/ps_context.h"
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
    auto cnode = node->cast<CNodePtr>();
    if (IsPrimitiveCNode(cnode, prim::kPrimVirtualDataset)) {
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
  auto input_abstract_tuple = node->abstract()->cast<abstract::AbstractTuplePtr>();
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
      abstract_list.emplace_back(virtual_dataset_abstract);
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
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
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
      graph_sets.insert(fun_graph);
    }
  }
  graph_sets.insert(root);
  return graph_sets;
}

static void InsertVirtualDataset(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes) {
  MS_EXCEPTION_IF_NULL(root);
  std::set<FuncGraphPtr> forward_graph_set = FindForwardGraph(root, all_nodes);
  for (auto forward_graph : forward_graph_set) {
    FuncGraphManagerPtr manager = forward_graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    std::vector<AnfNodePtr> graph_inputs = forward_graph->get_inputs();
    auto node_user_map = manager->node_users();
    auto virtual_dataset_node = CreateVirtualDataset(forward_graph);
    std::map<size_t, CNodePtr> parameter_index_map;
    for (size_t index = 0; index < graph_inputs.size(); index++) {
      if (HasAbstractMonad(graph_inputs[index])) {
        continue;
      }
      auto node_users = node_user_map[graph_inputs[index]];
      for (auto node_user : node_users) {
        auto cnode = node_user.first->cast<CNodePtr>();
        for (size_t input_index = 1; input_index < cnode->inputs().size(); input_index++) {
          if (!IsValueNode<Primitive>(cnode->inputs()[0])) {
            continue;
          }
          bool is_node_input_flag = !(IsValueNode<mindspore::tensor::Tensor>(cnode->inputs()[input_index]) ||
                                      IsValueNode<ValueList>(cnode->inputs()[input_index]) ||
                                      IsValueNode<ValueTuple>(cnode->inputs()[input_index]));
          if (find(graph_inputs.begin(), graph_inputs.end(), cnode->inputs()[input_index]) != graph_inputs.end() &&
              is_node_input_flag && !HasAbstractMonad(cnode->inputs()[input_index])) {
            auto node_input_iter = find(graph_inputs.begin(), graph_inputs.end(), cnode->inputs()[input_index]);
            size_t node_input_index = node_input_iter - graph_inputs.begin();
            if (parameter_index_map.empty() || parameter_index_map.count(node_input_index) == 0) {
              parameter_index_map[node_input_index] =
                CreateTupleGetItem(virtual_dataset_node, node_input_index, forward_graph);
            }
            manager->SetEdge(cnode, input_index, parameter_index_map[node_input_index]);
            manager->SetEdge(parameter_index_map[node_input_index], 1, virtual_dataset_node);
          }
        }
      }
    }
  }
}

void GenerateDefaultStrategy(const ValueNodePtr &axes, const std::vector<AnfNodePtr> &nodes, const int64_t device_num,
                             std::vector<std::vector<int64_t>> *default_strategy) {
  auto strategies = axes->value()->cast<ValueTuplePtr>()->value();
  size_t i = 0;
  for (auto &strategy : strategies) {
    auto node = nodes[i];
    if (strategy->isa<None>()) {
      auto node_size = common::AnfAlgo::GetOutputInferShape(node, 0).size();
      std::vector<int64_t> current_d_strategy(node_size, 1);
      if (node_size >= 1) {
        current_d_strategy[0] = device_num;
      }
      default_strategy->push_back(current_d_strategy);
    } else {
      auto current_strategy = GetValue<std::vector<int64_t>>(strategy);
      default_strategy->push_back(current_strategy);
    }
    i += 1;
  }
}

bool CheckLayout(const ValueNodePtr &axes, bool *need_default_strategy, size_t *axes_size) {
  auto strategies = axes->value()->cast<ValueTuplePtr>()->value();
  for (auto &strategy : strategies) {
    *axes_size += 1;
    if (strategy->isa<None>()) {
      *need_default_strategy = true;
      continue;
    }
    if (!strategy->isa<ValueTuple>()) {
      return false;
    }
    auto elements = strategy->cast<ValueTuplePtr>()->value();

    for (auto &element : elements) {
      if (!element->isa<Int64Imm>()) {
        return false;
      }
    }
  }
  return true;
}

void HandleStrategyForOneHot(std::vector<ValuePtr> *strategy) {
  // onehot needs to set layout for output, modify the strategy with an additional dimension
  auto input_strategy = GetValue<std::vector<int64_t>>(strategy->at(0));
  input_strategy.push_back(1);
  strategy->at(0) = MakeValue(input_strategy);
}

void HandleStrategyForMatMul(std::vector<ValuePtr> *strategy, const CNodePtr &cnode) {
  // handle strategy for matmul to deal with corresponding dimension
  auto left_matrix_strategy = GetValue<std::vector<int64_t>>(strategy->at(0));
  auto right_matrix_strategy = GetValue<std::vector<int64_t>>(strategy->at(1));
  auto index_a = left_matrix_strategy.size() - 1;
  auto index_b = index_a - 1;
  auto attrs = GetCNodePrimitive(cnode)->attrs();
  bool transpose_a = attrs[parallel::TRANSPOSE_A]->cast<BoolImmPtr>()->value();
  bool transpose_b = attrs[parallel::TRANSPOSE_B]->cast<BoolImmPtr>()->value();
  if (transpose_a) {
    index_a -= 1;
  }
  if (transpose_b) {
    index_b += 1;
  }
  if (left_matrix_strategy[index_a] != right_matrix_strategy[index_b]) {
    if (left_matrix_strategy[index_a] == 1) {
      left_matrix_strategy[index_a] = right_matrix_strategy[index_b];
    } else {
      right_matrix_strategy[index_b] = left_matrix_strategy[index_a];
    }
    strategy->at(0) = MakeValue(left_matrix_strategy);
    strategy->at(1) = MakeValue(right_matrix_strategy);
  }
}

void GetInputNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *input_nodes) {
  auto parameters = func_graph->parameters();
  for (auto &parameter : parameters) {
    if (parameter->cast<ParameterPtr>()->name() == "u" || parameter->cast<ParameterPtr>()->name() == "io") {
      continue;
    }
    input_nodes->push_back(parameter);
  }
}

void GetOutputNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *output_nodes) {
  auto return_node = func_graph->get_return();
  auto real_return_node = return_node->cast<CNodePtr>()->input(1);
  while (IsPrimitiveCNode(real_return_node, prim::kPrimDepend)) {
    real_return_node = real_return_node->cast<CNodePtr>()->input(1);
  }
  if (!IsPrimitiveCNode(real_return_node, prim::kPrimMakeTuple)) {
    output_nodes->push_back(real_return_node);
  } else {
    auto cnode = real_return_node->cast<CNodePtr>();
    for (size_t i = 1; i < cnode->inputs().size(); ++i) {
      output_nodes->push_back(cnode->input(i));
    }
  }
}

bool CheckDeviceNum(const std::vector<std::vector<int64_t>> &strategies, const int64_t &device_num) {
  for (size_t i = 0; i < strategies.size(); ++i) {
    auto strategy = strategies[i];
    int64_t required_num = 1;
    std::for_each(strategy.begin(), strategy.end(), [&](int64_t const &data) { required_num *= data; });
    if (required_num > device_num) {
      MS_LOG(ERROR) << "required device number: " << required_num
                    << " is larger than available device number: " << device_num << " at index: " << i;
      return false;
    }
    if (device_num % required_num != 0) {
      MS_LOG(ERROR) << "required device number: " << required_num
                    << " is not divisible by device number: " << device_num << " at index: " << i;
      return false;
    }
  }
  return true;
}

void SetOutputLayout(const FuncGraphPtr &func_graph, const AnfNodePtr &out_axes, const int64_t &device_num) {
  auto out_axes_tuple = out_axes->cast<ValueNodePtr>();
  bool need_default_strategy = false;
  size_t out_axes_size = 0;
  if (!IsValueNode<ValueTuple>(out_axes_tuple) ||
      !CheckLayout(out_axes_tuple, &need_default_strategy, &out_axes_size)) {
    MS_LOG(EXCEPTION) << "out_axes should be a two-dimension tuple";
  }
  std::vector<AnfNodePtr> output_nodes;
  GetOutputNodes(func_graph, &output_nodes);
  if (output_nodes.size() != out_axes_size) {
    MS_LOG(EXCEPTION) << "Output number: " << output_nodes.size()
                      << " is not equal to out_axes number: " << out_axes_size;
  }

  std::vector<std::vector<int64_t>> output_strategy;
  if (need_default_strategy) {
    GenerateDefaultStrategy(out_axes_tuple, output_nodes, device_num, &output_strategy);
  } else {
    output_strategy = GetValue<std::vector<std::vector<int64_t>>>(out_axes_tuple->value());
  }
  MS_LOG(WARNING) << "The output strategy will be overwritten as data-parallel";

  for (size_t i = 0; i < output_nodes.size(); ++i) {
    auto node = output_nodes[i];
    auto output_shape = common::AnfAlgo::GetOutputInferShape(node, 0);
    if (output_shape.size() != output_strategy[i].size()) {
      MS_LOG(EXCEPTION) << "Output dimension: " << output_shape.size()
                        << " is not equal to out_axes dimension: " << output_strategy[i].size() << " at index " << i;
    }
    std::vector<ValuePtr> elements;
    elements.push_back(MakeValue(output_strategy[i]));
    auto prim = GetCNodePrimitive(node);
    auto attrs_temp = prim->attrs();
    ValueTuplePtr strategy = std::make_shared<ValueTuple>(elements);
    attrs_temp[parallel::OUT_STRATEGY] = strategy;
    (void)prim->SetAttrs(attrs_temp);
  }
}

void SetInputLayout(const FuncGraphPtr &func_graph, const AnfNodePtr &in_axes, const int64_t &device_num) {
  auto in_axes_tuple = in_axes->cast<ValueNodePtr>();
  bool need_default_strategy = false;
  size_t in_axes_size = 0;
  if (!IsValueNode<ValueTuple>(in_axes_tuple) || !CheckLayout(in_axes_tuple, &need_default_strategy, &in_axes_size)) {
    MS_LOG(EXCEPTION) << "in_axes should be a two-dimension tuple";
  }
  std::vector<AnfNodePtr> input_nodes;
  GetInputNodes(func_graph, &input_nodes);
  if (input_nodes.size() != in_axes_size) {
    MS_LOG(EXCEPTION) << "Input numbers: " << input_nodes.size()
                      << " is not equal to in_axes numbers: " << in_axes_size;
  }
  std::vector<std::vector<int64_t>> input_strategy;
  if (need_default_strategy) {
    GenerateDefaultStrategy(in_axes_tuple, input_nodes, device_num, &input_strategy);
  } else {
    input_strategy = GetValue<std::vector<std::vector<int64_t>>>(in_axes_tuple->value());
  }
  if (!CheckDeviceNum(input_strategy, device_num)) {
    MS_LOG(EXCEPTION) << "check device number failed";
  }
  std::set<CNodePtr> concerned_nodes;
  FuncGraphManagerPtr manager = func_graph->manager();
  auto parameters = func_graph->parameters();
  for (size_t i = 0; i < parameters.size(); ++i) {
    auto parameter = parameters[i];
    if (parameter->cast<ParameterPtr>()->name() == "u" || parameter->cast<ParameterPtr>()->name() == "io") {
      continue;
    }
    auto output_shape = common::AnfAlgo::GetOutputInferShape(parameter, 0);
    if (output_shape.size() != input_strategy[i].size()) {
      MS_LOG(EXCEPTION) << "Input dimension: " << output_shape.size()
                        << " is not equal to in_axes dimension: " << input_strategy[i].size() << " at index " << i;
    }
    AnfNodeIndexSet param_sub_set = manager->node_users()[parameter];
    for (auto &param_pair : param_sub_set) {
      CNodePtr param_cnode = param_pair.first->cast<CNodePtr>();
      concerned_nodes.insert(param_cnode);
    }
  }
  for (auto &cnode : concerned_nodes) {
    auto current_inputs = cnode->inputs();
    std::vector<ValuePtr> elements;
    for (size_t i = 1; i < current_inputs.size(); ++i) {
      auto current_input = current_inputs[i];
      if (current_input->isa<ValueNode>()) {
        auto current_value = current_input->cast<ValueNodePtr>()->value();
        if (!current_value->isa<mindspore::tensor::Tensor>()) {
          continue;
        }
      }
      auto iter = std::find(parameters.begin(), parameters.end(), current_input);
      if (iter != parameters.end()) {
        elements.push_back(MakeValue(input_strategy[iter - parameters.begin()]));
      } else {
        auto shape = current_input->Shape()->cast<abstract::ShapePtr>();
        auto dimension = shape->shape().size();
        std::vector<int64_t> default_strategy(dimension, 1);
        elements.push_back(MakeValue(default_strategy));
      }
    }
    if (IsPrimitiveCNode(cnode, prim::kPrimMatMul) || IsPrimitiveCNode(cnode, prim::kPrimBatchMatMul)) {
      HandleStrategyForMatMul(&elements, cnode);
    }
    if (IsPrimitiveCNode(cnode, prim::kPrimOneHot)) {
      HandleStrategyForOneHot(&elements);
    }
    ValueTuplePtr strategy = std::make_shared<ValueTuple>(elements);
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    auto attrs_temp = prim->attrs();
    attrs_temp[parallel::IN_STRATEGY] = strategy;
    (void)prim->SetAttrs(attrs_temp);
  }
}

void SetStrategyForShard(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes,
                         const int64_t &device_num) {
  root->set_flag("training", true);
  for (auto &node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimShard)) {
      root->set_flag("auto_parallel", true);
      auto cnode = node->cast<CNodePtr>();
      auto vnode = cnode->input(1)->cast<ValueNodePtr>();
      auto in_axes = cnode->input(2);
      auto out_axes = cnode->input(3);
      ScopeGuard scope_guard(vnode->scope());
      auto func_graph = GetValueNode<FuncGraphPtr>(vnode);
      MS_EXCEPTION_IF_NULL(func_graph);
      SetInputLayout(func_graph, in_axes, device_num);
      SetOutputLayout(func_graph, out_axes, device_num);
    }
  }
}

// Only auto_parallel and semi_auto_parallel support PipelineSplit
bool PipelineSplit(const ResourcePtr &res) {
#if ((defined ENABLE_CPU) && (!defined _WIN32) && !defined(__APPLE__))
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
  auto execution_mode = MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE);
  if ((execution_mode == kPynativeMode) &&
      (parallel_mode == parallel::kSemiAutoParallel || parallel_mode == parallel::kAutoParallel)) {
    if (!parallel::ParallelContext::GetInstance()->device_num_is_set()) {
      MS_LOG(EXCEPTION) << "device_num must be set when use shard function";
    }
    auto device_num_shard = parallel::ParallelContext::GetInstance()->device_num();
    SetStrategyForShard(root, all_nodes, device_num_shard);
  }

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
  transformer->LabelMicroBatch();
  // step3: Handle shared parameters
  transformer->ParameterColoring();
  // step4: Cut Graph
  transformer->CutGraph();
  // step5: Handle Sens
  if (root->has_flag(parallel::kTraining)) {
    transformer->CoverSensShape();
  }
  // step6: Elim Graph stages and no used parameter
  transformer->ModifyParameterList();
  transformer->ElimGraphStage();
  return true;
}
}  // namespace pipeline
}  // namespace mindspore
