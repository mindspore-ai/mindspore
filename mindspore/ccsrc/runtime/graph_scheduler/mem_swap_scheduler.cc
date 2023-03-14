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

#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/offload_context.h"
#include "runtime/device/gsm/swap_strategy_builder.h"
#include "runtime/device/memory_offload_strategy.h"
#include "runtime/graph_scheduler/scheduler_helper.h"
#include "runtime/graph_scheduler/control_node_parser.h"

namespace mindspore {
namespace runtime {
constexpr size_t kFirstVirtualNode = 0;
constexpr size_t kSecondVirtualNodeOffset = 1;
constexpr char kOffloadTargetDDR[] = "cpu";
constexpr char kOffloadTargetDisk[] = "disk";
namespace {
AbstractActor *GetCtrlActor(const ControlNodeParserPtr &parser, const KernelGraph *graph, const string &actor_suffix) {
  MS_EXCEPTION_IF_NULL(parser);
  MS_EXCEPTION_IF_NULL(graph);
  const auto func_graph = parser->FetchFuncGraphByKernelGraph(graph);
  if (func_graph == nullptr) {
    return nullptr;
  }
  const std::string &actor_name = func_graph->ToString() + actor_suffix;
  return FetchActor(actor_name);
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
      real_tensor_ids.emplace_back(tensor_info->tensor_id_);
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
        fixed_device_address->emplace_back(output_addr.get());
      } else {
        tensor_indexes[real_tensor_id] = {real_parameter_index->size()};
        real_parameter_index->emplace_back(real_parameter_iter->second);
        is_real_parameter.insert(real_tensor_id);
      }
    }
  }
  for (auto &tensor_index : tensor_indexes) {
    if (is_real_parameter.count(tensor_index.first)) {
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
      std::transform(tensor_info->fused_tensor_ids_.begin(), tensor_info->fused_tensor_ids_.end(),
                     std::back_inserter(indexes), [&](size_t tensor_id) { return tensors_id_index_map.at(tensor_id); });
      alloc_action_map.emplace_back(indexes);
    } else if (tensor_action->action_ != device::SwapActionType::kUnDefined) {
      const auto tensor_id = tensor_info->tensor_id_;
      move_action_map[tensor_action->action_].emplace_back(tensors_id_index_map.at(tensor_id));
    } else {
      MS_LOG(EXCEPTION) << "Undefined swap action type.";
    }
  }
  std::transform(alloc_action_map.begin(), alloc_action_map.end(), std::back_inserter(*actor_actions),
                 [](const std::vector<size_t> &tensor_idxes) {
                   return std::make_pair(device::SwapActionType::kAllocHBM, tensor_idxes);
                 });
  std::transform(move_action_map.begin(), move_action_map.end(), std::back_inserter(*actor_actions),
                 [](const std::pair<device::SwapActionType, vector<size_t>> &action) {
                   return std::make_pair(action.first, action.second);
                 });
}

std::shared_ptr<device::SwapContext> GetSwapContext() {
  const auto &context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const auto &offload_context = OffloadContext::GetInstance();
  MS_EXCEPTION_IF_NULL(offload_context);
  const auto &swap_context = std::make_shared<device::SwapContext>();
  const auto max_hbm_size = context->get_param<float>(MS_CTX_MAX_DEVICE_MEMORY);
  swap_context->hbm_mem_size_ = FloatToSize(max_hbm_size * kGBToByte);
  swap_context->ddr_mem_size_ = offload_context->offload_ddr_size();
  swap_context->disk_mem_size_ = offload_context->offload_disk_size();
  const auto &offload_param = offload_context->offload_param();
  swap_context->offload_param_to_ddr_ = (offload_param == kOffloadTargetDDR);
  swap_context->offload_param_to_disk_ = (offload_param == kOffloadTargetDisk);
  const auto &offload_checkpoint = offload_context->offload_checkpoint();
  swap_context->offload_checkpoint_to_ddr_ = (offload_checkpoint == kOffloadTargetDDR);
  swap_context->offload_checkpoint_to_disk_ = (offload_checkpoint == kOffloadTargetDisk);
  return swap_context;
}
}  // namespace

void MemSwapScheduler::GetRealParameters(const KernelGraphPtr &graph, const ControlNodeParserPtr &parser) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(parser);
  const auto &entrance_actor =
    dynamic_cast<EntranceActor *>(GetCtrlActor(parser, graph.get(), kEntranceActorNameSuffix));
  if (entrance_actor == nullptr) {
    return;
  }
  HashMap<AnfNodePtr, size_t> real_parameters;
  for (const auto &input : graph->input_nodes()) {
    if (parser->IsControlFlowDataArrow(graph, input)) {
      const auto &front_node = GetFrontNodeByKernelGraph(input, graph.get());
      const size_t index = entrance_actor->FetchNodePosition(front_node);
      real_parameters.insert({input, index});
    }
  }
  real_parameters_[graph->graph_id()].swap(real_parameters);
}

void MemSwapScheduler::AddSwappableRootParameter(const mindspore::runtime::GraphCompilerInfo &graph_compiler_info) {
  DeviceContext *device_context = nullptr;
  std::shared_ptr<device::SwapManager> swap_manager = nullptr;
  for (const auto &context : graph_compiler_info.device_contexts_) {
    if (context == nullptr) {
      continue;
    }
    if (context->device_res_manager_ == nullptr) {
      continue;
    }
    swap_manager = context->device_res_manager_->swap_manager();
    if (swap_manager != nullptr) {
      device_context = context;
      break;
    }
  }
  if (swap_manager == nullptr) {
    return;
  }
  MS_EXCEPTION_IF_NULL(device_context);
  for (const auto &parameter : graph_compiler_info.origin_parameters_order_) {
    const auto &device_tensors = DeviceTensorStore::GetInstance().Fetch(parameter.get());
    if (device_tensors.empty()) {
      MS_LOG(INFO) << "Device tensor store is empty for parameter " << parameter->DebugString();
      continue;
    }
    for (const auto &device_tensor : device_tensors) {
      if (device_tensor != nullptr && device_tensor->GetDeviceType() == device_context->GetDeviceType()) {
        swap_manager->AddSwappableTensor(device_tensor);
      }
    }
  }
}

void MemSwapScheduler::AddSwappableTensors(const mindspore::device::DeviceContext *device_context,
                                           const std::shared_ptr<device::SwapStrategy> &strategy,
                                           const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(device_context);
  const auto &swap_manager = device_context->device_res_manager_->swap_manager();
  MS_EXCEPTION_IF_NULL(swap_manager);
  const auto &tensor_infos = strategy->tensor_infos_;
  const auto &ref_map = graph->GetRefMap();
  for (const auto &tensor_info : tensor_infos) {
    if (tensor_info->is_workspace_ || tensor_info->node_ == nullptr) {
      continue;
    }
    // Continuous memory are not swappable.
    if (tensor_info->real_tensor_id_ != tensor_info->tensor_id_ || !tensor_info->fused_tensor_ids_.empty()) {
      continue;
    }
    const auto &anf_with_out_index = std::make_pair(tensor_info->node_, tensor_info->index_);
    if (ref_map.find(anf_with_out_index) != ref_map.end()) {
      continue;
    }
    const auto &device_address = AnfAlgo::GetMutableOutputAddr(tensor_info->node_, tensor_info->index_, false);
    MS_EXCEPTION_IF_NULL(device_address);
    swap_manager->AddSwappableTensor(device_address);
    device_address->set_swappable(true);
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
  const auto &swap_context = GetSwapContext();
  auto swap_strategy = builder.Build(graph, swap_context);
  MS_EXCEPTION_IF_NULL(swap_strategy);
  graph_strategy_map_[graph->graph_id()] = swap_strategy;
  AddSwappableTensors(device_context, swap_strategy, graph);

  if (swap_strategy->actions_.empty()) {
    return;
  }
  GetRealParameters(graph, parser);

  static size_t swap_actor_num = 0;
  const auto graph_id = graph->graph_id();
  for (const auto &iter : swap_strategy->actions_) {
    // Fixed DeviceAddress in MemorySwapActor.
    std::vector<DeviceTensor *> fixed_device_address;
    // Output index of EntranceActor for real parameter whose DeviceAddress is changeable.
    std::vector<size_t> real_parameter_index;
    auto tensors_id_index_map = GetActionTensors(iter.second, swap_strategy, real_parameters_[graph->graph_id()],
                                                 &fixed_device_address, &real_parameter_index);

    // SwapActionType - index of target DeviceAddress(fixed or changeable) in MemorySwapActor.
    std::vector<std::pair<device::SwapActionType, vector<size_t>>> actor_actions;
    GenActionIndexList(tensors_id_index_map, iter.second, swap_strategy, &actor_actions);

    const string swap_actor_name = kMemSwapActorNamePrefix + std::to_string(swap_actor_num++);
    auto swap_actor = std::make_shared<MemorySwapActor>(swap_actor_name, recorder_aid_, kDefaultStreamIndex,
                                                        fixed_device_address, device_context, actor_actions);
    actors->emplace_back(swap_actor);
    // Link data arrow from EntranceActor to MemorySwapActor later in Link
    data_dependency_[graph_id][swap_actor].swap(real_parameter_index);
    action_actor_map_[graph_id][iter.first] = swap_actor;
  }
}

std::vector<std::vector<MemSwapActorPtr>> MemSwapScheduler::Build(const GraphCompilerInfo &graph_compiler_info,
                                                                  const AID *recorder_aid) {
  recorder_aid_ = recorder_aid;
  std::vector<std::vector<MemSwapActorPtr>> swap_actors;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->get_param<bool>(MS_CTX_ENABLE_MEM_OFFLOAD)) {
    return swap_actors;
  }
  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto device_context = graph_compiler_info.device_contexts_[i];
    const auto &graph = graph_compiler_info.graphs_[i];
    std::vector<MemSwapActorPtr> actors;
    if (device_context == nullptr || graph == nullptr || graph->is_dynamic_shape()) {
      swap_actors.emplace_back(actors);
      continue;
    }
    BuildSwapActorForGraph(graph, graph_compiler_info.control_node_parser_, device_context, &actors);
    swap_actors.emplace_back(actors);
  }
  AddSwappableRootParameter(graph_compiler_info);
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
  const auto &action_actor_map = action_actor_map_.find(graph->graph_id());
  if (action_actor_map == action_actor_map_.end()) {
    MS_LOG(EXCEPTION) << "Can not find Actor for action id " << id << " in graph " << graph->graph_id();
  }
  const auto &action_iter = (action_actor_map->second).find(id);
  if (action_iter == (action_actor_map->second).end()) {
    MS_LOG(EXCEPTION) << "Can not find Actor for action id " << id << " in graph " << graph->graph_id();
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
      continue;
    }
    const auto &action_actor_map = action_actor_map_.find(graph->graph_id());
    if (action_actor_map == action_actor_map_.end()) {
      continue;
    }
    const auto &data_dependency = data_dependency_.find(graph->graph_id());
    if (data_dependency == data_dependency_.end()) {
      continue;
    }
    for (const auto &action_iter : strategy->actions_) {
      const auto &actor_iter = (action_actor_map->second).find(action_iter.first);
      if (actor_iter == (action_actor_map->second).end()) {
        continue;
      }
      const auto &data_dependency_iter = (data_dependency->second).find(actor_iter->second);
      if (data_dependency_iter == (data_dependency->second).end()) {
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
