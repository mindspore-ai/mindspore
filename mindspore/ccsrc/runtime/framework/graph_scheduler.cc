/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "runtime/framework/graph_scheduler.h"
#include "mindrt/src/actor/actormgr.h"
#include "mindrt/include/async/async.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/optimizer/common/helper.h"
#include "utils/config_manager.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
namespace {
bool IsDeviceQueueDSActor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<CNode>() && (AnfAlgo::GetCNodeName(node) == kGetNextOpName)) {
    return true;
  }
  return false;
}

bool IsHostQueueDSActor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<Parameter>() && (!AnfAlgo::IsParameterWeight(node->cast<ParameterPtr>()))) {
    return true;
  }
  return false;
}

bool IsKernelActor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<CNode>() && (AnfAlgo::GetCNodeName(node) != kGetNextOpName)) {
    return true;
  }
  return false;
}

// Judge whether the device tensor of the node is persistent or not.
bool IsPersistentDeviceTensor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<ValueNode>()) {
    return true;
  }
  if (node->isa<Parameter>() && AnfAlgo::IsParameterWeight(node->cast<ParameterPtr>())) {
    return true;
  }
  return false;
}

KernelActor *FindKernelActor(const std::unordered_map<std::string, KernelActorPtr> &kernel_actors_map,
                             const std::string &name) {
  auto iter = kernel_actors_map.find(name);
  if (iter != kernel_actors_map.end()) {
    return iter->second.get();
  }
  return nullptr;
}

DeviceQueueDataSourceActor *FindDeviceQueueDSActor(const std::vector<DataSourceActorPtr> &data_source_actors) {
  for (auto &actor : data_source_actors) {
    MS_EXCEPTION_IF_NULL(actor);
    if (actor->GetAID().Name().find("_DeviceQueueDataSourceActor") != string::npos) {
      auto device_queue_ds_actor = dynamic_cast<DeviceQueueDataSourceActor *>(actor.get());
      return device_queue_ds_actor;
    }
  }
  return nullptr;
}

HostQueueDataSourceActor *FindHostQueueDSActor(const std::vector<DataSourceActorPtr> &data_source_actors) {
  for (auto &actor : data_source_actors) {
    MS_EXCEPTION_IF_NULL(actor);
    if (actor->GetAID().Name().find("_HostQueueDataSourceActor") != string::npos) {
      auto device_queue_ds_actor = dynamic_cast<HostQueueDataSourceActor *>(actor.get());
      return device_queue_ds_actor;
    }
  }
  return nullptr;
}

// Update the reference count of device tensor by the output index of node.
void UpdateRefCount(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  auto device_tensor = AnfAlgo::GetMutableOutputAddr(node, output_idx);
  MS_EXCEPTION_IF_NULL(device_tensor);
  device_tensor->IncreaseRefCount();
  device_tensor->ResetRefCountUsed();
}
}  // namespace

ActorSet *GraphScheduler::Transform(const KernelGraphPtr &graph, const DeviceContext *device_context,
                                    const std::vector<tensor::TensorPtr> *input_tensors,
                                    GraphExecutionStrategy strategy) {
  PersistDeviceTensor(graph);
  auto actor_set = Build(graph, device_context);
  graph_to_actors_.emplace(graph, actor_set);
  Link(actor_set.get(), graph, strategy);
  return actor_set.get();
}

void GraphScheduler::Schedule(const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  auto actorMgr = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actorMgr);

  // Schedule dats source actors.
  for (auto &data_source_actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(data_source_actor);
    auto base_actor = static_cast<ActorReference>(data_source_actor);
    (void)actorMgr->Spawn(base_actor);
  }

  // Schedule kernel actors.
  for (auto &kernel_actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    auto base_actor = static_cast<ActorReference>(kernel_actor);
    (void)actorMgr->Spawn(base_actor);
  }

  // Schedule loop count actor.
  if (actor_set->loop_count_actor_ != nullptr) {
    auto base_actor = static_cast<ActorReference>(actor_set->loop_count_actor_);
    (void)actorMgr->Spawn(base_actor);
  }
}

bool GraphScheduler::Run(const ActorSet *actor_set, GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(actor_set);
  // Construct OpContext.
  OpContext<DeviceTensor> op_context;
  auto sequential_num = uuids::RandomBasedGenerator::GenerateRandomUuid();
  op_context.sequential_num_ = &sequential_num;
  Promise<int> result;
  op_context.results_->push_back(result);

  // Trigger no input kernel actor running.
  for (auto &no_input_kernel_actor : actor_set->no_input_kernel_actors_) {
    MS_EXCEPTION_IF_NULL(no_input_kernel_actor);
    Async(no_input_kernel_actor->GetAID(), &KernelActor::RunOpControl, nullptr, &op_context);
  }

  // Trigger data source actor running.
  for (auto &data_source_actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(data_source_actor);
    Async(data_source_actor->GetAID(), &DataSourceActor::FetchData, &op_context);
  }

  // Trigger kernel actor running in the step execution strategy.
  if (strategy == GraphExecutionStrategy::kStep) {
    for (auto &kernel_actor : actor_set->kernel_actors_) {
      MS_EXCEPTION_IF_NULL(kernel_actor);
      Async(kernel_actor->GetAID(), &KernelActor::RunOpControl, nullptr, &op_context);
    }
  }

  // Get the run result.
  auto result_future = result.GetFuture();
  result_future.Wait();
  if (!result_future.IsOK()) {
    return false;
  }
  return true;
}

ActorSet *GraphScheduler::Fetch(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto iter = graph_to_actors_.find(graph);
  if (iter != graph_to_actors_.end()) {
    return iter->second.get();
  } else {
    MS_LOG(ERROR) << "Can't find the actors map of graph: " << graph->ToString();
    return nullptr;
  }
}

ActorSetPtr GraphScheduler::Build(const KernelGraphPtr &graph, const DeviceContext *device_context) {
  auto actor_set = std::make_shared<ActorSet>();
  MS_EXCEPTION_IF_NULL(actor_set);

  auto data_source_actors = BuildDataSourceActor(graph);
  actor_set->data_source_actors_.swap(data_source_actors);

  auto kernel_actors = BuildKernelActor(graph, device_context);
  actor_set->kernel_actors_.swap(kernel_actors);

  auto loop_count_actor = BuildLoopCountActor(graph);
  actor_set->loop_count_actor_ = loop_count_actor;

  return actor_set;
}

void GraphScheduler::Link(ActorSet *actor_set, const KernelGraphPtr &graph, GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(graph);
  std::unordered_map<std::string, KernelActorPtr> kernel_actors_temp_map;
  for (auto &actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(actor);
    kernel_actors_temp_map.emplace(actor->GetAID().Name(), actor);
  }

  // Foreach the execution order to link the actors.
  auto execution_order = graph->execution_order();
  for (auto &kernel : execution_order) {
    if (!IsKernelActor(kernel)) {
      continue;
    }
    auto kernel_actor = FindKernelActor(kernel_actors_temp_map, kernel->fullname_with_scope());
    // Link the control arrows of kernel actor.
    LinkControlArrowForKernelActor(kernel_actor, actor_set->loop_count_actor_.get(), graph, strategy);

    for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(kernel); ++i) {
      KernelWithIndex from_kernel_with_output_idx = AnfAlgo::GetPrevNodeOutput(kernel, i);
      KernelWithIndex to_kernel_with_input_idx = std::make_pair(kernel, i);
      auto from_kernel = from_kernel_with_output_idx.first;

      if (IsDeviceQueueDSActor(from_kernel)) {
        // Link the data arrows of device queue data source actor.
        auto from_actor = FindDeviceQueueDSActor(actor_set->data_source_actors_);
        LinkDataArrowForDeviceDSActor(from_actor, kernel_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
      } else if (IsHostQueueDSActor(from_kernel)) {
        // Link the data arrows of host queue data source actor.
        auto from_actor = FindHostQueueDSActor(actor_set->data_source_actors_);
        LinkDataArrowForHostDSActor(from_actor, kernel_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
      } else {
        // Link the data arrows of kernel actor.
        auto from_actor = FindKernelActor(kernel_actors_temp_map, from_kernel->fullname_with_scope());
        LinkDataArrowForKernelActor(from_actor, kernel_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
      }
    }
  }

  // BuildNoInputKernelActor depends on whether kernel actors have input, so must be behind the link of kernel actors.
  auto no_input_kernel_actors = BuildNoInputKernelActor(graph);
  actor_set->no_input_kernel_actors_.swap(no_input_kernel_actors);

  // Link the control arrows of loop count actor, which depends on the no input kernel actors.
  LinkControlArrowForLoopCountActor(actor_set->loop_count_actor_.get(), graph);
}

std::vector<DataSourceActorPtr> GraphScheduler::BuildDataSourceActor(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<DataSourceActorPtr> data_source_actors;

  // Build host queue data source actor.
  HostQueueDSActorPtr host_queue_ds_actor = nullptr;
  for (auto &input_node : graph->input_nodes()) {
    MS_EXCEPTION_IF_NULL(input_node);
    if (IsHostQueueDSActor(input_node)) {
      if (host_queue_ds_actor == nullptr) {
        auto actor_name = graph->ToString() + "_" + "HostQueueDataSourceActor";
        MS_LOG(INFO) << "Create host queue data source actor: " << actor_name;
        auto host_queue = std::make_shared<HostTensorQueue>();
        graph_to_host_queue_.emplace(graph, host_queue);
        host_queue_ds_actor = std::make_shared<HostQueueDataSourceActor>(actor_name, 1, host_queue);
        data_source_actors.emplace_back(host_queue_ds_actor);
      }
      host_queue_ds_actor->data_nodes_.emplace_back(input_node);
    }
  }

  // Build device queue data source actor.
  auto execution_order = graph->execution_order();
  auto iter = std::find_if(execution_order.begin(), execution_order.end(),
                           [](const CNodePtr &node) { return IsDeviceQueueDSActor(node); });
  if (iter != execution_order.end()) {
    auto actor_name = graph->ToString() + "_" + "DeviceQueueDataSourceActor";
    MS_LOG(INFO) << "Create queue data source actor: " << actor_name;
    auto device_queue_ds_actor = std::make_shared<DeviceQueueDataSourceActor>(actor_name, 1);
    MS_EXCEPTION_IF_NULL(device_queue_ds_actor);
    data_source_actors.emplace_back(device_queue_ds_actor);
    device_queue_ds_actor->data_kernel_ = *iter;
  }
  return data_source_actors;
}

std::vector<KernelActorPtr> GraphScheduler::BuildKernelActor(const KernelGraphPtr &graph,
                                                             const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<KernelActorPtr> kernel_actors;

  auto execution_order = graph->execution_order();
  for (auto &kernel : execution_order) {
    if (IsKernelActor(kernel)) {
      auto kernel_actor = std::make_shared<KernelActor>(kernel->fullname_with_scope(), kernel, device_context);
      MS_EXCEPTION_IF_NULL(kernel_actor);
      kernel_actors.emplace_back(kernel_actor);
    }
  }
  return kernel_actors;
}

std::vector<KernelActorPtr> GraphScheduler::BuildNoInputKernelActor(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<KernelActorPtr> no_input_kernel_actors;

  auto actor_set = Fetch(graph);
  MS_EXCEPTION_IF_NULL(actor_set);
  for (auto &kernel_actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    if ((kernel_actor->input_datas_num_ == 0) && (kernel_actor->input_controls_num_ == 0)) {
      no_input_kernel_actors.emplace_back(kernel_actor);
    }
  }
  return no_input_kernel_actors;
}

LoopCountActorPtr GraphScheduler::BuildLoopCountActor(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto loop_count = ConfigManager::GetInstance().iter_num();
  auto actor_name = graph->ToString() + "_" + "LoopCountActor";
  auto loop_count_actor = std::make_shared<LoopCountActor>(actor_name, loop_count);
  MS_EXCEPTION_IF_NULL(loop_count_actor);
  return loop_count_actor;
}

void GraphScheduler::LinkDataArrowForDeviceDSActor(DeviceQueueDataSourceActor *from_actor, KernelActor *to_actor,
                                                   KernelWithIndex from_kernel_with_output_idx,
                                                   KernelWithIndex to_kernel_with_input_idx) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);

  auto from_kernel = from_kernel_with_output_idx.first;
  MS_EXCEPTION_IF_NULL(from_kernel);
  auto from_output_index = from_kernel_with_output_idx.second;
  auto to_input_index = to_kernel_with_input_idx.second;

  auto to_aid = to_actor->GetAID();
  auto op_arrow = std::make_shared<OpArrow>(from_output_index, to_aid, to_input_index);
  from_actor->output_op_arrows_.emplace_back(op_arrow);
  to_actor->input_datas_num_++;

  // Update the reference count of device tensor.
  UpdateRefCount(from_kernel, from_output_index);
}

void GraphScheduler::LinkDataArrowForHostDSActor(HostQueueDataSourceActor *from_actor, KernelActor *to_actor,
                                                 KernelWithIndex from_kernel_with_output_idx,
                                                 KernelWithIndex to_kernel_with_input_idx) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);

  auto from_kernel = from_kernel_with_output_idx.first;
  MS_EXCEPTION_IF_NULL(from_kernel);
  auto from_output_index = from_kernel_with_output_idx.second;
  auto to_input_index = to_kernel_with_input_idx.second;

  auto data_nodes = from_actor->data_nodes_;
  auto iter = find(data_nodes.begin(), data_nodes.end(), from_kernel);
  if (iter == data_nodes.end()) {
    MS_LOG(EXCEPTION) << "Parameter node: " << from_kernel->fullname_with_scope() << " is not exist.";
  }
  auto position = IntToSize(std::distance(data_nodes.begin(), iter));
  auto to_aid = to_actor->GetAID();
  auto op_arrow = std::make_shared<OpArrow>(position, to_aid, to_input_index);
  from_actor->output_op_arrows_.emplace_back(op_arrow);
  to_actor->input_datas_num_++;

  // Update the reference count of device tensor.
  UpdateRefCount(from_kernel, from_output_index);
}

void GraphScheduler::LinkDataArrowForKernelActor(KernelActor *from_actor, KernelActor *to_actor,
                                                 KernelWithIndex from_kernel_with_output_idx,
                                                 KernelWithIndex to_kernel_with_input_idx) {
  MS_EXCEPTION_IF_NULL(to_actor);
  auto from_kernel = from_kernel_with_output_idx.first;
  MS_EXCEPTION_IF_NULL(from_kernel);
  auto from_output_index = from_kernel_with_output_idx.second;
  auto to_input_index = to_kernel_with_input_idx.second;

  if (IsPersistentDeviceTensor(from_kernel)) {
    to_actor->device_tensor_store_keys_.emplace_back(to_input_index, static_cast<void *>(from_kernel.get()));
  } else if (IsKernelActor(from_kernel)) {
    MS_EXCEPTION_IF_NULL(from_actor);
    auto to_aid = to_actor->GetAID();
    auto op_arrow = std::make_shared<OpArrow>(from_output_index, to_aid, to_input_index);
    from_actor->output_op_arrows_.emplace_back(op_arrow);
    to_actor->input_datas_num_++;

    // Update the reference count of device tensor.
    UpdateRefCount(from_kernel, from_output_index);
  }
}

void GraphScheduler::LinkControlArrowForKernelActor(KernelActor *from_actor, LoopCountActor *to_actor,
                                                    const KernelGraphPtr &graph, GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(graph);

  if (strategy == GraphExecutionStrategy::kStep) {
    from_actor->input_controls_num_++;
  }

  if (opt::IsNotRealUsedByOthers(graph, from_actor->kernel_)) {
    auto to_aid = to_actor->GetAID();
    from_actor->output_op_controls_.emplace_back(to_aid);
    to_actor->input_controls_num_++;
  }
}

void GraphScheduler::LinkControlArrowForLoopCountActor(LoopCountActor *loop_count_actor, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(loop_count_actor);

  auto actor_set = Fetch(graph);
  MS_EXCEPTION_IF_NULL(actor_set);

  // Set the source data actor.
  for (auto &data_source_actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(data_source_actor);
    loop_count_actor->data_source_aids_.emplace_back(data_source_actor->GetAID());
  }

  // Set the no input kernel actor.
  for (auto &no_input_kernel_actor : actor_set->no_input_kernel_actors_) {
    MS_EXCEPTION_IF_NULL(no_input_kernel_actor);
    loop_count_actor->no_input_kernel_aids_.emplace_back(no_input_kernel_actor->GetAID());
  }
}

void GraphScheduler::PersistDeviceTensor(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);

  for (auto &value_node : graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    auto device_tensor = AnfAlgo::GetMutableOutputAddr(value_node, 0);
    DeviceTensorStore::GetInstance().Insert(value_node.get(), device_tensor);
    device_tensor->set_ref_count(SIZE_MAX);
    device_tensor->ResetRefCountUsed();
  }

  for (auto &input_node : graph->input_nodes()) {
    MS_EXCEPTION_IF_NULL(input_node);
    if (IsPersistentDeviceTensor(input_node)) {
      auto device_tensor = AnfAlgo::GetMutableOutputAddr(input_node, 0);
      DeviceTensorStore::GetInstance().Insert(input_node.get(), device_tensor);
      device_tensor->set_ref_count(SIZE_MAX);
      device_tensor->ResetRefCountUsed();
    }
  }
}

}  // namespace runtime
}  // namespace mindspore
