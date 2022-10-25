/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "runtime/graph_scheduler/memory_swap_node_scheduler.h"

#include <algorithm>
#include <map>
#include <utility>
#include <string>
#include <memory>

#include "runtime/device/memory_offload_strategy.h"
#include "runtime/graph_scheduler/scheduler_helper.h"

namespace mindspore {
namespace runtime {
constexpr size_t kKindOfSwapActor = 2;
constexpr size_t kMaxMemReuseFactor = 9;
constexpr size_t kMinMemReuseFactor = 5;
constexpr size_t kRetryFactor = 1;
constexpr size_t kReuseFactorDenominator = 10;
namespace {
AbstractActor *GetEntranceActorByKernelGraph(const ControlNodeParserPtr &parser, const KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(parser);
  const auto func_graph = parser->FetchFuncGraphByKernelGraph(graph);
  if (func_graph == nullptr) {
    return nullptr;
  }
  const std::string &entrance_actor_name = func_graph->ToString() + kEntranceActorNameSuffix;
  return FetchActor(entrance_actor_name);
}

AbstractActor *GetExitActorByKernelGraph(const ControlNodeParserPtr &parser, const KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(parser);
  const auto func_graph = parser->FetchFuncGraphByKernelGraph(graph);
  if (func_graph == nullptr) {
    return nullptr;
  }
  const std::string exit_actor_name = func_graph->ToString() + kExitActorNameSuffix;
  return FetchActor(exit_actor_name);
}

AbstractActor *GetLatestKernelActor(const KernelGraph *graph, size_t index, const ControlNodeParserPtr &parser) {
  MS_EXCEPTION_IF_NULL(graph);
  if (index == 0) {
    return GetEntranceActorByKernelGraph(parser, graph);
  }
  AbstractActor *latest_actor = nullptr;
  size_t latest_index = index - 1;
  while (latest_actor == nullptr && latest_index > 0) {
    const auto &latest_kernel = graph->execution_order()[latest_index];
    if (IsSkippedKernelActor(latest_kernel)) {
      latest_index -= 1;
      continue;
    }
    latest_actor = FetchActor(latest_kernel->fullname_with_scope());
    latest_index -= 1;
  }
  return latest_actor == nullptr ? GetEntranceActorByKernelGraph(parser, graph) : latest_actor;
}

AbstractActor *GetNextKernelActor(const KernelGraph *graph, size_t index, const ControlNodeParserPtr &parser) {
  MS_EXCEPTION_IF_NULL(graph);
  AbstractActor *next_actor = nullptr;
  size_t next_index = index + 1;
  while (next_actor == nullptr && next_index < graph->execution_order().size()) {
    const auto &next_kernel = graph->execution_order()[next_index];
    if (IsSkippedKernelActor(next_kernel)) {
      next_index += 1;
      continue;
    }
    next_actor = FetchActor(next_kernel->fullname_with_scope());
    next_index += 1;
  }
  return next_actor == nullptr ? GetExitActorByKernelGraph(parser, graph) : next_actor;
}
}  // namespace
std::vector<std::vector<MemSwapActorPtr>> MemorySwapNodeScheduler::Build(const GraphCompilerInfo &graph_compiler_info,
                                                                         const AID *recorder_aid) {
  recorder_aid_ = recorder_aid;
  std::vector<std::vector<MemSwapActorPtr>> swap_actors;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->get_param<bool>(MS_CTX_ENABLE_MEM_OFFLOAD)) {
    return swap_actors;
  }
  struct MemAllocated {
    explicit MemAllocated(DeviceContext *device_context) : device_context_(device_context) {}
    ~MemAllocated() {
      for (const auto &mem : mem_allocated_) {
        device_context_->device_res_manager_->FreeMemory(mem.second);
      }
    }
    DeviceContext *device_context_;
    std::map<DeviceTensor *, void *> mem_allocated_;
  };
  std::shared_ptr<MemAllocated> mem_allocated = nullptr;
  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto device_context = graph_compiler_info.device_contexts_[i];
    std::vector<MemSwapActorPtr> sub_graph_swap_actors;
    const auto &graph = graph_compiler_info.graphs_[i];
    // Sub graph executed on CPU do not need memory offload.
    // Graph with dynamic shape kernel dost not support memory offload.
    if (device_context == nullptr || device_context->GetDeviceType() == device::DeviceType::kCPU ||
        graph->is_dynamic_shape()) {
      (void)swap_actors.emplace_back(std::move(sub_graph_swap_actors));
      continue;
    }
    if (mem_allocated == nullptr) {
      mem_allocated = std::make_shared<MemAllocated>(device_context);
    }
    auto offload_strategy =
      GenMemOffloadStrategy(graph, graph_compiler_info.device_contexts_[i], graph_compiler_info.control_node_parser_,
                            &(mem_allocated->mem_allocated_));
    if (offload_strategy == nullptr) {
      (void)swap_actors.emplace_back(std::move(sub_graph_swap_actors));
      continue;
    }
    (void)swap_actors.emplace_back(std::move(
      GenSwapActorsForGraph(graph, device_context, offload_strategy, graph_compiler_info.control_node_parser_)));
  }
  return swap_actors;
}

MemOffloadStrategyPtr MemorySwapNodeScheduler::GenMemOffloadStrategy(const KernelGraphPtr &graph,
                                                                     const DeviceContext *device_context,
                                                                     const ControlNodeParserPtr &parser,
                                                                     std::map<DeviceTensor *, void *> *mem_allocated) {
  const auto &mem_statistic = CollectMemStatistic(graph, device_context, parser);
  const auto total_available_mem_size = device_context->device_res_manager_->GetAvailableMemSize();
  for (size_t factor = kMaxMemReuseFactor; factor > kMinMemReuseFactor; factor -= kRetryFactor) {
    auto mem_offload_strategy = std::make_shared<device::MemOffloadStrategy<DeviceTensor *>>(mem_statistic);
    mem_offload_strategy->set_mem_size(total_available_mem_size * factor / kReuseFactorDenominator);
    mem_offload_strategy->Execute();
    std::map<DeviceTensor *, void *> mem_allocated_temp(*mem_allocated);
    if (MockStrategy(mem_offload_strategy, device_context, graph->execution_order().size(), &mem_allocated_temp)) {
      *mem_allocated = mem_allocated_temp;
      return mem_offload_strategy;
    }
  }
  return nullptr;
}

DeviceTensor *MemorySwapNodeScheduler::GetNodeOutputDeviceTensor(
  const mindspore::AnfNodePtr &node, size_t output_idx, const mindspore::KernelGraphPtr &graph,
  const mindspore::device::DeviceContext *device_context) const {
  auto device_address = AnfAlgo::GetMutableOutputAddr(node, output_idx, true).get();
  if (node->isa<CNode>()) {
    return device_address;
  }
  auto front_node = AnfAlgo::FetchFrontNodeByBackendNode(node, *graph);
  if (front_node == nullptr || front_node->isa<CNode>()) {
    return device_address;
  }
  auto real_device_address = DeviceTensorStore::GetInstance().Fetch(front_node.get(), device_context->GetDeviceType());
  if (real_device_address != nullptr) {
    return real_device_address;
  }
  return device_address;
}

void MemorySwapNodeScheduler::CollectGraphInputMemStatistic(const mindspore::KernelGraphPtr &graph,
                                                            const mindspore::device::DeviceContext *device_context,
                                                            const mindspore::runtime::ControlNodeParserPtr &parser,
                                                            device::GraphMemStatistic<DeviceTensor *> *statistic) {
  constexpr size_t first_step = 0;
  for (const auto &input : graph->input_nodes()) {
    auto device_address = GetNodeOutputDeviceTensor(input, 0, graph, device_context);
    const auto &parameter = input->cast<ParameterPtr>();
    if (common::AnfAlgo::IsParameterWeight(parameter)) {
      statistic->Record(device_address, device::kInit, device_address->GetSize(), device::kMemPriorityHigh, first_step);
    }
    if (parser->IsControlFlowDataArrow(graph, input)) {
      formal_parameters_[device_address] = input;
    }
  }
}

void MemorySwapNodeScheduler::CollectKernelInputMemStatistic(size_t kernel_index,
                                                             const mindspore::KernelGraphPtr &graph,
                                                             const mindspore::device::DeviceContext *device_context,
                                                             device::GraphMemStatistic<DeviceTensor *> *statistic,
                                                             HashSet<const void *> *offload_conflict) const {
  const auto &node = graph->execution_order()[kernel_index];
  auto kernel_mod = AnfAlgo::GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  const bool is_communication_node = common::AnfAlgo::IsCommunicationOp(node);
  auto input_num = std::min(kernel_mod->GetInputSizeList().size(), node->inputs().size() - 1);
  const bool continuous_input_mem = is_communication_node && input_num > 1;
  size_t total_size = 0;
  std::vector<size_t> size_list;
  std::vector<DeviceTensor *> device_tensors;
  for (size_t index = 0; index < input_num; ++index) {
    const auto &prev_node_output = common::AnfAlgo::GetPrevNodeOutput(node, index, true);
    const auto &input_address =
      GetNodeOutputDeviceTensor(prev_node_output.first, prev_node_output.second, graph, device_context);
    statistic->Record(input_address, device::kGet, input_address->GetSize(), device::kMemPriorityLow, kernel_index);
    if (continuous_input_mem) {
      total_size += input_address->GetSize();
      (void)size_list.emplace_back(input_address->GetSize());
      (void)device_tensors.emplace_back(input_address);
    }
    (void)offload_conflict->insert(input_address);
    device::MemoryOffloadConflict::GetInstance().AddOffloadBacklog(input_address);
  }
  if (continuous_input_mem) {
    statistic->continuous_mem_info_helper_->AddContinuousMemInfo(true, kernel_index, total_size, size_list,
                                                                 device_tensors);
  }
}

void MemorySwapNodeScheduler::CollectKernelOutputMemStatistic(size_t kernel_index,
                                                              const mindspore::KernelGraphPtr &graph,
                                                              device::GraphMemStatistic<DeviceTensor *> *statistic,
                                                              HashSet<const void *> *offload_conflict) const {
  const auto &node = graph->execution_order()[kernel_index];
  auto kernel_mod = AnfAlgo::GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  const bool is_communication_node = common::AnfAlgo::IsCommunicationOp(node);
  const auto output_num = kernel_mod->GetOutputSizeList().size();
  const bool continuous_output_mem = is_communication_node && output_num > 1;
  size_t total_size = 0;
  std::vector<size_t> size_list;
  std::vector<DeviceTensor *> device_tensors;
  for (size_t index = 0; index < output_num; ++index) {
    const auto &output_address = AnfAlgo::GetMutableOutputAddr(node, index).get();
    statistic->Record(output_address, device::kGet, output_address->GetSize(), device::kMemPriorityLow, kernel_index);
    if (continuous_output_mem) {
      total_size += output_address->GetSize();
      (void)size_list.emplace_back(output_address->GetSize());
      (void)device_tensors.emplace_back(output_address);
    }
    (void)offload_conflict->insert(output_address);
    device::MemoryOffloadConflict::GetInstance().AddOffloadBacklog(output_address);
  }
  if (continuous_output_mem) {
    statistic->continuous_mem_info_helper_->AddContinuousMemInfo(false, kernel_index, total_size, size_list,
                                                                 device_tensors);
  }
}

void MemorySwapNodeScheduler::CollectKernelWorkspaceMemStatistic(size_t kernel_index,
                                                                 const mindspore::KernelGraphPtr &graph,
                                                                 device::GraphMemStatistic<DeviceTensor *> *statistic,
                                                                 HashSet<const void *> *offload_conflict) const {
  const auto &node = graph->execution_order()[kernel_index];
  auto kernel_mod = AnfAlgo::GetKernelMod(node);
  for (size_t index = 0; index < kernel_mod->GetWorkspaceSizeList().size(); ++index) {
    const auto &workspace_address = AnfAlgo::GetMutableWorkspaceAddr(node, index).get();
    statistic->Record(workspace_address, device::kGet, workspace_address->GetSize(), device::kMemPriorityLow,
                      kernel_index);
    (void)offload_conflict->insert(workspace_address);
    device::MemoryOffloadConflict::GetInstance().AddOffloadBacklog(workspace_address);
  }
}

device::GraphMemStatistic<DeviceTensor *> MemorySwapNodeScheduler::CollectMemStatistic(
  const KernelGraphPtr &graph, const DeviceContext *device_context, const ControlNodeParserPtr &parser) {
  device::GraphMemStatistic<DeviceTensor *> mem_statistic;
  mem_statistic.total_compute_index_ = graph->execution_order().size();
  CollectGraphInputMemStatistic(graph, device_context, parser, &mem_statistic);
  for (size_t i = 0; i < graph->execution_order().size(); ++i) {
    HashSet<const void *> offload_conflict;
    CollectKernelInputMemStatistic(i, graph, device_context, &mem_statistic, &offload_conflict);
    CollectKernelOutputMemStatistic(i, graph, &mem_statistic, &offload_conflict);
    CollectKernelWorkspaceMemStatistic(i, graph, &mem_statistic, &offload_conflict);
    device::MemoryOffloadConflict::GetInstance().AddMemoryOffloadConflict(offload_conflict);
  }
  return mem_statistic;
}

bool MemorySwapNodeScheduler::MockStrategy(const MemOffloadStrategyPtr &strategy, const DeviceContext *device_context,
                                           size_t execution_size,
                                           std::map<DeviceTensor *, void *> *mem_allocated) const {
  for (size_t step = 0; step < execution_size; ++step) {
    const auto continuous_mem_info = strategy->GetContinuousMemAllocInfo(step);
    for (const auto &continuous_mem : continuous_mem_info) {
      const auto &ret = device_context->device_res_manager_->AllocateContinuousMemory(continuous_mem->align_size_list_);
      if (ret.size() != continuous_mem->key_index_map_.size()) {
        return false;
      }
      for (const auto &key_index : continuous_mem->key_index_map_) {
        (*mem_allocated)[key_index.first] = ret[key_index.second];
      }
    }
    std::vector<DeviceTensor *> tensor_to_use;
    for (const auto &prev_mem_event : strategy->GetPreComputeEvents(step)) {
      if (prev_mem_event->type == device::kGet) {
        (void)tensor_to_use.emplace_back(prev_mem_event->key);
      } else if (mem_allocated->count(prev_mem_event->key) == 0) {
        void *ptr = nullptr;
        try {
          ptr = device_context->device_res_manager_->AllocateMemory(prev_mem_event->mem_size);
        } catch (std::runtime_error &e) {
          return false;
        }
        if (ptr == nullptr) {
          return false;
        }
        (*mem_allocated)[prev_mem_event->key] = ptr;
      }
    }
    for (const auto &tensor : tensor_to_use) {
      if (mem_allocated->count(tensor) == 0) {
        return false;
      }
    }
    for (const auto &post_mem_events : strategy->GetPostComputeEvents(step)) {
      const auto &iter = mem_allocated->find(post_mem_events->key);
      if (iter == mem_allocated->end()) {
        continue;
      }
      device_context->device_res_manager_->FreeMemory(iter->second);
      (void)mem_allocated->erase(iter);
    }
  }
  return true;
}

std::vector<MemSwapActorPtr> MemorySwapNodeScheduler::GenSwapActorsForGraph(
  const mindspore::KernelGraphPtr &graph, mindspore::device::DeviceContext *device_context,
  const mindspore::runtime::MemOffloadStrategyPtr &strategy, const ControlNodeParserPtr &parser) {
  const auto &entrance_actor = dynamic_cast<EntranceActor *>(GetEntranceActorByKernelGraph(parser, graph.get()));
  std::vector<MemSwapActorPtr> swap_actors;
  for (size_t j = 0; j < graph->execution_order().size(); ++j) {
    // Collect swap in events scheduled and generator MemorySwapActor
    const auto pre_events = strategy->GetPreComputeEvents(j);
    device::MemEventPtrList<DeviceTensor *> swap_in_events;
    std::vector<size_t> real_parameter_index_from_entrance;
    // Do not set DeviceTensor* to MemorySwapActor when DeviceTensor is fetched from a formal parameter.
    // DeviceTensor on formal parameter will not be used when actors running, link data arrows from EntranceActor to
    // MemorySwapActor.
    FilterRealParamInPreEvent(pre_events, graph, entrance_actor, &swap_in_events, &real_parameter_index_from_entrance);
    const auto &continuous_mem_info = strategy->GetContinuousMemAllocInfo(j);
    const auto &swap_in_actor = GenSwapInActor(graph->execution_order()[j], device_context, swap_in_events,
                                               continuous_mem_info, real_parameter_index_from_entrance.size());
    if (swap_in_actor != nullptr && !real_parameter_index_from_entrance.empty()) {
      real_parameter_map_[swap_in_actor] =
        std::make_pair(entrance_actor, std::move(real_parameter_index_from_entrance));
    }
    (void)swap_actors.emplace_back(swap_in_actor);
    // Collect swap out events scheduled and generator MemorySwapActor
    device::MemEventPtrList<DeviceTensor *> swap_out_events;
    const auto post_events = strategy->GetPostComputeEvents(j);
    std::vector<size_t> real_parameter_index_from_entrance_out;
    std::vector<bool> swap_out_real_parameter;
    for (const auto &post_event : post_events) {
      const auto &iter = formal_parameters_.find(post_event->key);
      if (iter != formal_parameters_.end()) {
        const auto &front_node = GetFrontNodeByKernelGraph(iter->second, graph.get());
        MS_EXCEPTION_IF_NULL(entrance_actor);
        const size_t index = entrance_actor->FetchNodePosition(front_node);
        (void)real_parameter_index_from_entrance_out.emplace_back(index);
        (void)swap_out_real_parameter.emplace_back(post_event->type == device::kSwapOut);
      } else {
        (void)swap_out_events.emplace_back(post_event);
      }
    }
    const auto &swap_out_actor = GenSwapOutActor(graph->execution_order()[j], swap_out_events, swap_out_real_parameter);
    if (swap_out_actor != nullptr && !real_parameter_index_from_entrance_out.empty()) {
      real_parameter_map_[swap_out_actor] =
        std::make_pair(entrance_actor, std::move(real_parameter_index_from_entrance_out));
    }
    (void)swap_actors.emplace_back(swap_out_actor);
  }
  return swap_actors;
}

void MemorySwapNodeScheduler::FilterRealParamInPreEvent(const device::MemEventPtrList<DeviceTensor *> &pre_events,
                                                        const mindspore::KernelGraphPtr &graph,
                                                        const EntranceActor *entrance_actor,
                                                        device::MemEventPtrList<DeviceTensor *> *swap_in_events,
                                                        std::vector<size_t> *entrance_index) const {
  for (const auto &pre_event : pre_events) {
    if (pre_event->type == device::kSwapIn || pre_event->type == device::kInit) {
      const auto &iter = formal_parameters_.find(pre_event->key);
      if (iter != formal_parameters_.end()) {
        const auto &front_node = GetFrontNodeByKernelGraph(iter->second, graph.get());
        MS_EXCEPTION_IF_NULL(entrance_actor);
        const size_t index = entrance_actor->FetchNodePosition(front_node);
        (void)entrance_index->emplace_back(index);
      } else {
        (void)swap_in_events->emplace_back(pre_event);
      }
    }
  }
}

MemSwapActorPtr MemorySwapNodeScheduler::GenSwapInActor(
  const CNodePtr &kernel, DeviceContext *device_context, const device::MemEventPtrList<DeviceTensor *> &swap_events,
  const std::vector<device::ContinuousMemInfoPtr<DeviceTensor *>> &continuous_mem_info,
  size_t real_parameter_size) const {
  if (swap_events.empty() && continuous_mem_info.empty() && real_parameter_size == 0) {
    return nullptr;
  }
  vector<bool> is_init_swap;
  std::vector<DeviceTensor *> device_tensor;
  (void)std::transform(swap_events.cbegin(), swap_events.cend(), std::back_inserter(device_tensor),
                       [](const device::MemEventPtr<DeviceTensor *> &event) { return event->key; });
  std::vector<std::vector<DeviceTensor *>> continuous_device_tensors;
  std::vector<std::vector<size_t>> continuous_device_tensor_sizes;
  for (const auto &continuous_info : continuous_mem_info) {
    std::vector<DeviceTensor *> device_tensors(continuous_info->align_size_list_.size(), nullptr);
    std::vector<size_t> tensor_sizes(continuous_info->align_size_list_.size(), 0);
    for (const auto &key_index : continuous_info->key_index_map_) {
      device_tensors[key_index.second] = key_index.first;
      tensor_sizes[key_index.second] = continuous_info->align_size_list_[key_index.second];
    }
    (void)continuous_device_tensors.emplace_back(device_tensors);
    (void)continuous_device_tensor_sizes.emplace_back(tensor_sizes);
  }
  const std::string name = kernel->fullname_with_scope() + kMemSwapInActorNameSuffix;
  const auto stream_id = AnfAlgo::GetStreamId(kernel);
  auto swap_actor =
    std::make_shared<MemorySwapInActor>(name, recorder_aid_, stream_id, device_context, device_tensor,
                                        continuous_device_tensors, continuous_device_tensor_sizes, real_parameter_size);
  return swap_actor;
}

MemSwapActorPtr MemorySwapNodeScheduler::GenSwapOutActor(const CNodePtr &kernel,
                                                         const device::MemEventPtrList<DeviceTensor *> &swap_events,
                                                         const std::vector<bool> &swap_out_real_parameter) const {
  if (swap_events.empty()) {
    return nullptr;
  }
  std::vector<DeviceTensor *> device_tensor_to_swap;
  std::vector<DeviceTensor *> device_tensor_to_free;
  for (const auto &event : swap_events) {
    if (event->type == device::kSwapOut) {
      (void)device_tensor_to_swap.emplace_back(event->key);
    } else if (event->type == device::kFree) {
      // Offload DeviceTensor with max original_ref_count_ which will not be used anymore in current kernel graph.
      // Just free DeviceTensor with max original_ref_count_. original_ref_count_ may have not been set appropriately,
      // judge it in MemorySwapOutActor.
      (void)device_tensor_to_free.emplace_back(event->key);
    }
  }
  const std::string name = kernel->fullname_with_scope() + kMemSwapOutActorNameSuffix;
  const auto stream_id = AnfAlgo::GetStreamId(kernel);
  auto swap_actor = std::make_shared<MemorySwapOutActor>(name, recorder_aid_, stream_id, device_tensor_to_swap,
                                                         device_tensor_to_free, swap_out_real_parameter);
  return swap_actor;
}

void MemorySwapNodeScheduler::Link(const GraphCompilerInfo &graph_compiler_info, ActorSet *actor_set) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->get_param<bool>(MS_CTX_ENABLE_MEM_OFFLOAD)) {
    return;
  }
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);
  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    if (!actor_set->swap_actors_[i].empty() && actor_set->swap_actors_[i][0] != nullptr) {
      const auto &entrance_actor = GetEntranceActorByKernelGraph(parser, graph_compiler_info.graphs_[i].get());
      if (entrance_actor != nullptr) {
        SchedulerHelper::AddControlArrow(entrance_actor, actor_set->swap_actors_[i][0].get());
      } else {
        SchedulerHelper::AddControlArrow(actor_set->data_prepare_actor_.get(), actor_set->swap_actors_[i][0].get());
      }
    }
    LinkControlArrowBySwapActor(graph_compiler_info.graphs_[i], graph_compiler_info.control_node_parser_,
                                actor_set->swap_actors_[i]);
  }
  LinkDataArrowForRealParameter();
}

void MemorySwapNodeScheduler::LinkControlArrowBySwapActor(const KernelGraphPtr &graph,
                                                          const ControlNodeParserPtr &parser,
                                                          const std::vector<MemSwapActorPtr> &swap_actors) const {
  if (swap_actors.empty()) {
    return;
  }
  if (graph->execution_order().size() * kKindOfSwapActor != swap_actors.size()) {
    MS_LOG(EXCEPTION) << "The size of swap swap_actors[" << swap_actors.size() << "] should be the twice of kernel["
                      << graph->execution_order().size() << "]";
  }
  for (size_t i = 0; i < graph->execution_order().size(); ++i) {
    const auto &kernel = graph->execution_order()[i];
    if (IsSkippedKernelActor(kernel)) {
      continue;
    }
    const std::shared_ptr<MemorySwapActor> &swap_in_actor = swap_actors.at(i * kKindOfSwapActor);
    AbstractActor *kernel_actor = nullptr;
    if (swap_in_actor != nullptr) {
      if (i != 0) {
        const auto &pre_kernel_actor = GetLatestKernelActor(graph.get(), i, parser);
        MS_EXCEPTION_IF_NULL(pre_kernel_actor);
        SchedulerHelper::AddControlArrow(pre_kernel_actor, swap_in_actor.get());
      }
      kernel_actor = FetchActor(kernel->fullname_with_scope());
      MS_EXCEPTION_IF_NULL(kernel_actor);
      SchedulerHelper::AddControlArrow(swap_in_actor.get(), kernel_actor);
    }
    const std::shared_ptr<MemorySwapActor> &swap_out_actor = swap_actors.at(i * kKindOfSwapActor + 1);
    if (swap_out_actor != nullptr) {
      if (i < graph->execution_order().size()) {
        const auto &next_kernel_actor = GetNextKernelActor(graph.get(), i, parser);
        if (next_kernel_actor != nullptr) {
          SchedulerHelper::AddControlArrow(swap_out_actor.get(), next_kernel_actor);
        }
      }
      if (kernel_actor == nullptr) {
        kernel_actor = FetchActor(kernel->fullname_with_scope());
        MS_EXCEPTION_IF_NULL(kernel_actor);
      }
      SchedulerHelper::AddControlArrow(kernel_actor, swap_out_actor.get());
    }
  }
}

void MemorySwapNodeScheduler::LinkDataArrowForRealParameter() const {
  for (const auto &iter : real_parameter_map_) {
    const auto &entrance_output_index = iter.second.second;
    for (size_t i = 0; i < entrance_output_index.size(); ++i) {
      SchedulerHelper::AddDataArrow(iter.second.first, iter.first.get(), entrance_output_index[i], i);
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
