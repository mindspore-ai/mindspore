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
#include "runtime/graph_scheduler/mem_swap_scheduler.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <set>
#include <utility>

#include "runtime/device/gsm/swap_strategy_builder.h"
#include "runtime/graph_scheduler/scheduler_helper.h"
#include "runtime/device/memory_offload_strategy.h"
#include "runtime/graph_scheduler/control_node_parser.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace runtime {
constexpr size_t kFirstVirtualNode = 0;
constexpr size_t kSecondVirtualNodeOffset = 1;
namespace {
AbstractActor *GetCtrlActor(const ControlNodeParserPtr &parser, const KernelGraph *graph, const string &actor_suffix) {
  MS_EXCEPTION_IF_NULL(parser);
  MS_EXCEPTION_IF_NULL(graph);
  const auto func_graph = parser->FetchFuncGraphByKernelGraph(graph);
  if (func_graph == nullptr) {
    return nullptr;
  }
  const std::string &actor_name = func_graph->ToString() + actor_suffix;
  return dynamic_cast<EntranceActor *>(FetchActor(actor_name));
}

std::map<size_t, size_t> GetActionTensors(const std::shared_ptr<device::SwapAction> &swap_action,
                                          const std::shared_ptr<device::SwapStrategy> &swap_strategy,
                                          const HashMap<AnfNodePtr, size_t> &real_parameters,
                                          std::vector<DeviceTensor *> *fixed_device_address,
                                          std::vector<size_t> *real_parameter_index) {
  MS_EXCEPTION_IF_NULL(swap_action);
  MS_EXCEPTION_IF_NULL(swap_strategy);
  MS_EXCEPTION_IF_NULL(fixed_device_address);
  MS_EXCEPTION_IF_NULL(real_parameter_index);
  std::map<size_t, size_t> tensor_indexes;
  std::set<size_t> is_real_parameter;
  for (const auto &tensor_action : swap_action->actions_) {
    MS_EXCEPTION_IF_NULL(tensor_action);
    if (tensor_action->tensor_id_ >= swap_strategy->tensor_infos_.size()) {
      MS_LOG(EXCEPTION) << "Invalid tensor id " << tensor_action->tensor_id_;
    }
    const auto &tensor_info = swap_strategy->tensor_infos_[tensor_action->tensor_id_];
    MS_EXCEPTION_IF_NULL(tensor_info);
    std::vector<size_t> real_tensor_ids;
    if (tensor_info->fused_tensor_ids_.empty()) {
      (void)real_tensor_ids.emplace_back(tensor_info->tensor_id_);
    } else {
      real_tensor_ids = tensor_info->fused_tensor_ids_;
    }
    for (const auto real_tensor_id : real_tensor_ids) {
      if (tensor_indexes.find(real_tensor_id) != tensor_indexes.end()) {
        continue;
      }
      if (real_tensor_id >= swap_strategy->tensor_infos_.size()) {
        MS_LOG(EXCEPTION) << "Invalid tensor id " << real_tensor_id;
      }
      const auto &real_tensor_info = swap_strategy->tensor_infos_[real_tensor_id];
      MS_EXCEPTION_IF_NULL(real_tensor_info);
      const auto &node = real_tensor_info->node_;
      const auto &real_parameter_iter = real_parameters.find(node);
      if (real_parameter_iter == real_parameters.end()) {
        const auto &output_addr = AnfAlgo::GetMutableOutputAddr(node, real_tensor_info->index_, false);
        tensor_indexes[real_tensor_id] = {fixed_device_address->size()};
        (void)fixed_device_address->emplace_back(output_addr.get());
      } else {
        tensor_indexes[real_tensor_id] = {real_parameter_index->size()};
        (void)real_parameter_index->emplace_back(real_parameter_iter->second);
        (void)is_real_parameter.insert(real_tensor_id);
      }
    }
  }
  for (auto &tensor_index : tensor_indexes) {
    if (is_real_parameter.count(tensor_index.first) != 0) {
      tensor_index.second += fixed_device_address->size();
    }
  }
  return tensor_indexes;
}

void GenActionIndexList(const std::map<size_t, size_t> &tensors_id_index_map,
                        const std::shared_ptr<device::SwapAction> &swap_action,
                        const std::shared_ptr<device::SwapStrategy> &swap_strategy,
                        std::vector<std::pair<device::SwapActionType, vector<size_t>>> *actor_actions) {
  MS_EXCEPTION_IF_NULL(swap_action);
  MS_EXCEPTION_IF_NULL(swap_strategy);
  MS_EXCEPTION_IF_NULL(actor_actions);
  std::vector<vector<size_t>> alloc_action_map;
  std::map<device::SwapActionType, vector<size_t>> move_action_map;
  const auto &actions = swap_action->actions_;
  for (const auto &tensor_action : actions) {
    MS_EXCEPTION_IF_NULL(tensor_action);
    if (tensor_action->tensor_id_ >= swap_strategy->tensor_infos_.size()) {
      MS_LOG(EXCEPTION) << "Invalid tensor id " << tensor_action->tensor_id_;
    }
    const auto &tensor_info = swap_strategy->tensor_infos_[tensor_action->tensor_id_];
    MS_EXCEPTION_IF_NULL(tensor_info);
    if (tensor_action->action_ == device::SwapActionType::kAllocHBM) {
      std::vector<size_t> indexes;
      (void)std::copy(tensor_info->fused_tensor_ids_.begin(), tensor_info->fused_tensor_ids_.end(),
                      std::back_inserter(indexes));
      (void)alloc_action_map.emplace_back(indexes);
    } else if (tensor_action->action_ != device::SwapActionType::kUnDefined) {
      const auto tensor_id = tensor_info->tensor_id_;
      (void)move_action_map[tensor_action->action_].emplace_back(tensors_id_index_map.at(tensor_id));
    } else {
      MS_LOG(EXCEPTION) << "Undefined swap action type.";
    }
  }
  (void)std::transform(alloc_action_map.begin(), alloc_action_map.end(), std::back_inserter(*actor_actions),
                       [](const std::vector<size_t> &tensor_idxes) {
                         return std::make_pair(device::SwapActionType::kAllocHBM, tensor_idxes);
                       });
  (void)std::transform(move_action_map.begin(), move_action_map.end(), std::back_inserter(*actor_actions),
                       [](const std::pair<device::SwapActionType, vector<size_t>> &action) {
                         return std::make_pair(action.first, action.second);
                       });
}
}  // namespace

void MemSwapScheduler::GetRealParameters(const KernelGraphPtr &graph, const ControlNodeParserPtr &parser,
                                         HashMap<AnfNodePtr, size_t> *real_parameters) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(parser);
  MS_EXCEPTION_IF_NULL(real_parameters);
  const auto &entrance_actor =
    dynamic_cast<EntranceActor *>(GetCtrlActor(parser, graph.get(), kEntranceActorNameSuffix));
  if (entrance_actor == nullptr) {
    return;
  }
  for (const auto &input : graph->input_nodes()) {
    if (parser->IsControlFlowDataArrow(graph, input)) {
      const auto &front_node = GetFrontNodeByKernelGraph(input, graph.get());
      const size_t index = entrance_actor->FetchNodePosition(front_node);
      (void)real_parameters->insert({input, index});
    }
  }
}

void MemSwapScheduler::BuildSwapActorForGraph(const KernelGraphPtr &graph, const ControlNodeParserPtr &parser,
                                              const DeviceContext *device_context,
                                              std::vector<MemSwapActorPtr> *actors) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(parser);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(actors);
  if (graph->is_dynamic_shape() || device_context->GetDeviceType() == device::DeviceType::kCPU) {
    return;
  }
  device::SwapStrategyBuilder builder;
  const auto &swap_context = std::make_shared<device::SwapContext>();
  auto swap_strategy = builder.Build(graph, swap_context);
  MS_EXCEPTION_IF_NULL(swap_strategy);
  graph_strategy_map_[graph->graph_id()] = swap_strategy;

  if (swap_strategy->actions_.empty()) {
    return;
  }
  // Real parameter - output index of EntranceActor
  HashMap<AnfNodePtr, size_t> real_parameters;
  GetRealParameters(graph, parser, &real_parameters);

  static size_t swap_actor_num = 0;
  for (const auto &iter : swap_strategy->actions_) {
    // Fixed DeviceAddress in MemorySwapActor.
    std::vector<DeviceTensor *> fixed_device_address;
    // Output index of EntranceActor for real parameter whose DeviceAddress is changeable.
    std::vector<size_t> real_parameter_index;
    auto tensors_id_index_map =
      GetActionTensors(iter.second, swap_strategy, real_parameters, &fixed_device_address, &real_parameter_index);

    // SwapActionType - index of target DeviceAddress(fixed or changeable) in MemorySwapActor.
    std::vector<std::pair<device::SwapActionType, vector<size_t>>> actor_actions;
    GenActionIndexList(tensors_id_index_map, iter.second, swap_strategy, &actor_actions);

    const string swap_actor_name = kMemSwapActorNamePrefix + std::to_string(swap_actor_num++);
    auto swap_actor = std::make_shared<MemorySwapActor>(swap_actor_name, recorder_aid_, kDefaultStreamIndex,
                                                        fixed_device_address, device_context, actor_actions);
    (void)actors->emplace_back(swap_actor);
    // Link data arrow from EntranceActor to MemorySwapActor later in Link
    data_dependency_[swap_actor].swap(real_parameter_index);
    action_actor_map_[iter.first] = swap_actor;
  }
}

std::vector<std::vector<MemSwapActorPtr>> MemSwapScheduler::Build(const GraphCompilerInfo &graph_compiler_info,
                                                                  const AID *recorder_aid) {
  recorder_aid_ = recorder_aid;
  std::vector<std::vector<MemSwapActorPtr>> swap_actors;
  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto device_context = graph_compiler_info.device_contexts_[i];
    const auto &graph = graph_compiler_info.graphs_[i];
    std::vector<MemSwapActorPtr> actors;
    if (device_context == nullptr || graph == nullptr || graph->is_dynamic_shape()) {
      (void)swap_actors.emplace_back(actors);
      continue;
    }
    BuildSwapActorForGraph(graph, graph_compiler_info.control_node_parser_, device_context, &actors);
    (void)swap_actors.emplace_back(actors);
  }
  return swap_actors;
}

AbstractActor *MemSwapScheduler::GetActorForLink(size_t id, const std::shared_ptr<device::SwapStrategy> &strategy,
                                                 const KernelGraphPtr &graph, const ControlNodeParserPtr &parser,
                                                 ActorSet *actor_set) const {
  MS_EXCEPTION_IF_NULL(strategy);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(parser);
  MS_EXCEPTION_IF_NULL(actor_set);
  AbstractActor *ret = nullptr;
  if (id == kFirstVirtualNode) {
    ret = dynamic_cast<EntranceActor *>(GetCtrlActor(parser, graph.get(), kEntranceActorNameSuffix));
    if (ret == nullptr) {
      ret = actor_set->data_prepare_actor_.get();
    }
  } else if (id == graph->execution_order().size() + kSecondVirtualNodeOffset) {
    ret = dynamic_cast<ExitActor *>(GetCtrlActor(parser, graph.get(), kExitActorNameSuffix));
    if (ret == nullptr) {
      ret = actor_set->loop_count_actor_.get();
    }
  }
  if (ret != nullptr) {
    return ret;
  }
  const auto &node_iter = strategy->nodes_.find(id);
  if (node_iter != strategy->nodes_.end()) {
    const auto &node = node_iter->second;
    const auto kernel_type = FetchKernelTransformType(node, graph, {}, GraphExecutionStrategy::kPipeline);
    return FetchActor(kernel_type, actor_set->name_, node, graph);
  }
  const auto &action_iter = action_actor_map_.find(id);
  if (action_iter == action_actor_map_.end()) {
    MS_LOG(EXCEPTION) << "Can not find Actor for action id " << id;
  }
  return action_iter->second.get();
}

void MemSwapScheduler::Link(const GraphCompilerInfo &graph_compiler_info, ActorSet *actor_set) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  for (const auto &graph : graph_compiler_info.graphs_) {
    const auto &strategy_iter = graph_strategy_map_.find(graph->graph_id());
    if (strategy_iter == graph_strategy_map_.end()) {
      continue;
    }
    const auto &strategy = strategy_iter->second;
    MS_EXCEPTION_IF_NULL(strategy);
    for (const auto &link : strategy->links_) {
      MS_EXCEPTION_IF_NULL(link);
      const auto from_actor =
        GetActorForLink(link->from_, strategy, graph, graph_compiler_info.control_node_parser_, actor_set);
      MS_EXCEPTION_IF_NULL(from_actor);
      const auto to_actor =
        GetActorForLink(link->to_, strategy, graph, graph_compiler_info.control_node_parser_, actor_set);
      MS_EXCEPTION_IF_NULL(to_actor);
      SchedulerHelper::AddControlArrow(from_actor, to_actor);
    }
    const auto &entrance_actor = dynamic_cast<EntranceActor *>(
      GetCtrlActor(graph_compiler_info.control_node_parser_, graph.get(), kEntranceActorNameSuffix));
    if (entrance_actor == nullptr) {
      return;
    }
    for (const auto &action_iter : strategy->actions_) {
      const auto &actor_iter = action_actor_map_.find(action_iter.first);
      if (actor_iter == action_actor_map_.end()) {
        continue;
      }
      const auto &data_dependency_iter = data_dependency_.find(actor_iter->second);
      if (data_dependency_iter == data_dependency_.end()) {
        continue;
      }
      for (size_t i = 0; i < data_dependency_iter->second.size(); ++i) {
        const auto &output_index = data_dependency_iter->second[i];
        SchedulerHelper::AddDataArrow(entrance_actor, actor_iter->second.get(), output_index, i);
      }
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
