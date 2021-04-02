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
#include "runtime/framework/actor/memory_manager_actor.h"
#include "mindrt/src/actor/actormgr.h"
#include "mindrt/include/async/async.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/optimizer/common/helper.h"
#include "utils/config_manager.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils.h"
#include "common/trans.h"

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

//  The branch processing of PrepareDataForValueNode that value type is tensor.
void PrepareDataForValueNodeTensor(const ValueNodePtr &node, const ValuePtr &node_value,
                                   const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node_value);
  MS_EXCEPTION_IF_NULL(device_context);

  std::vector<TensorPtr> tensors;
  TensorValueToTensor(node_value, &tensors);

  for (size_t i = 0; i < tensors.size(); i++) {
    const auto &tensor = tensors[i];
    if (tensor == nullptr) {
      MS_LOG(WARNING) << "Tensor is null";
      return;
    }

    const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(node, i);
    MS_EXCEPTION_IF_NULL(device_tensor);
    // If the ptr of device tensor is not nullptr, it indicates that the device data has been prepared.
    if (device_tensor->GetPtr() != nullptr) {
      return;
    }
    MS_LOG(INFO) << "Prepare device data for value node: " << node->fullname_with_scope() << ", output index: " << i;
    tensor->set_device_address(device_tensor);

    // Allocate device memory.
    if (!device_context->AllocateMemory(device_tensor.get(), device_tensor->GetSize())) {
      MS_LOG(EXCEPTION) << "Device memory isn't enough and alloc failed, node name: " << node->fullname_with_scope()
                        << ", alloc size: " << device_tensor->GetSize();
    }

    // Copy data from host tensor to device.
    if (!device_tensor->SyncHostToDevice(trans::GetRuntimePaddingShape(node, 0), LongToSize(tensor->data().nbytes()),
                                         tensor->data_type(), tensor->data_c())) {
      MS_LOG(EXCEPTION) << "SyncHostToDevice failed, node name: " << node->fullname_with_scope();
    }
  }
}

// Prepare the device data for persistent device tensor of value node.
void PrepareDataForValueNode(const ValueNodePtr &node, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);
  auto &node_value = node->value();
  MS_EXCEPTION_IF_NULL(node_value);

  if (node_value->isa<tensor::Tensor>() || node_value->isa<ValueTuple>()) {
    //  The branch processing that value type is tensor.
    PrepareDataForValueNodeTensor(node, node_value, device_context);
  } else if (node_value->isa<StringImm>()) {
    const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(node, 0);
    MS_EXCEPTION_IF_NULL(device_tensor);
    // If the ptr of device tensor is not nullptr, it indicates that the device data has been prepared.
    if (device_tensor->GetPtr() != nullptr) {
      return;
    }
    MS_LOG(INFO) << "Prepare device data for value node: " << node->fullname_with_scope();

    // Allocate device memory.
    if (!device_context->AllocateMemory(device_tensor.get(), device_tensor->GetSize())) {
      MS_LOG(EXCEPTION) << "Device memory isn't enough and alloc failed, node name: " << node->fullname_with_scope()
                        << ", alloc size: " << device_tensor->GetSize();
    }

    // Copy data from value to device.
    auto value = GetValue<std::string>(node_value);
    size_t tensor_size = value.size();
    ShapeVector shape = {1, SizeToLong(tensor_size)};
    if (!device_tensor->SyncHostToDevice(shape, tensor_size, kNumberTypeUInt8, value.data())) {
      MS_LOG(EXCEPTION) << "SyncHostToDevice failed, node name: " << node->fullname_with_scope();
    }
  }
}

// Prepare the device data for persistent device tensor of weight node from host tensor.
void PrepareDataForWeightNode(const AnfNodePtr &node, const TensorPtr &tensor, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(tensor);
  const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(node, 0);
  MS_EXCEPTION_IF_NULL(device_tensor);
  // If the ptr of device tensor is not nullptr, it indicates that the device data has been prepared.
  if (device_tensor->GetPtr() != nullptr) {
    return;
  }
  MS_LOG(INFO) << "Prepare device data for weight node: " << node->fullname_with_scope();
  tensor->set_device_address(device_tensor);

  // Allocate device memory.
  if (!device_context->AllocateMemory(device_tensor.get(), device_tensor->GetSize())) {
    MS_LOG(EXCEPTION) << "Device memory isn't enough and alloc failed, node name: " << node->fullname_with_scope()
                      << ", alloc size: " << device_tensor->GetSize();
  }

  // Copy data from host tensor to device.
  if (!device_tensor->SyncHostToDevice(trans::GetRuntimePaddingShape(node, 0), LongToSize(tensor->data().nbytes()),
                                       tensor->data_type(), tensor->data_c())) {
    MS_LOG(EXCEPTION) << "SyncHostToDevice failed, node name: " << node->fullname_with_scope();
  }
}

BaseRef CreateOutputTensor(const session::KernelWithIndex &node_output_pair, const KernelGraphPtr &graph,
                           const std::vector<tensor::TensorPtr> &input_tensors) {
  auto &node = node_output_pair.first;
  auto output_index = node_output_pair.second;
  MS_EXCEPTION_IF_NULL(node);

  if (node->isa<ValueNode>()) {
    // If node is a value node, return the value.
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    return value_node->value();
  } else if (node->isa<Parameter>()) {
    // If node is a parameter node, return tensor from input_tensors.
    MS_EXCEPTION_IF_NULL(graph);
    const auto &input_nodes = graph->inputs();
    auto iter = find(input_nodes.begin(), input_nodes.end(), node);
    if (iter == input_nodes.end()) {
      MS_LOG(EXCEPTION) << "Parameter node: " << node->fullname_with_scope() << " is not exist.";
    }
    auto position = IntToSize(std::distance(input_nodes.begin(), iter));
    return input_tensors[position];
  } else {
    // Create tensor.
    TypeId type_id = AnfAlgo::GetOutputDeviceDataType(node, output_index);
    if (type_id == kTypeUnknown) {
      type_id = AnfAlgo::GetOutputInferDataType(node, output_index);
    }
    std::vector<int64_t> temp_shape;
    auto shape = AnfAlgo::GetOutputInferShape(node, output_index);
    (void)std::copy(shape.begin(), shape.end(), std::back_inserter(temp_shape));
    auto tensor = std::make_shared<tensor::Tensor>(type_id, temp_shape);
    MS_EXCEPTION_IF_NULL(tensor);
    tensor->set_padding_type(AnfAlgo::GetOutputReshapeType(node, output_index));

    // Set device address to tensor.
    const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(node, output_index);
    MS_EXCEPTION_IF_NULL(device_tensor);
    tensor->set_device_address(device_tensor);
    device_tensor->set_ref_count(SIZE_MAX);
    device_tensor->ResetRefCountUsed();
    return tensor;
  }
}

BaseRef CreateOutputTensors(const AnfNodePtr &output_node, const KernelGraphPtr &graph,
                            const std::vector<tensor::TensorPtr> &input_tensors) {
  MS_EXCEPTION_IF_NULL(output_node);
  auto item_with_index = AnfAlgo::VisitKernelWithReturnType(output_node, 0);
  MS_EXCEPTION_IF_NULL(item_with_index.first);

  // Special handle for make tuple.
  if (AnfAlgo::CheckPrimitiveType(item_with_index.first, prim::kPrimMakeTuple)) {
    auto cnode = item_with_index.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    VectorRef ret;
    for (size_t i = 1; i < cnode->inputs().size(); ++i) {
      auto out = CreateOutputTensors(cnode->input(i), graph, input_tensors);
      ret.push_back(out);
    }
    return ret;
  }

  // If the node return nothing, return an empty vectorRef.
  if (AnfAlgo::GetOutputTensorNum(item_with_index.first) == 0) {
    return VectorRef();
  }

  return CreateOutputTensor(item_with_index, graph, input_tensors);
}
}  // namespace

void GraphScheduler::Initialize() {
  if (init_) {
    return;
  }
  init_ = true;

  auto actorMgr = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actorMgr);

  // Create the thread pool of actor runtime.
  auto max_thread_num = GetMaxThreadNum();
  MS_LOG(INFO) << "Max available thread number: " << max_thread_num;
  actorMgr->Initialize(max_thread_num);

  // Create memory manager actor.
  auto memory_manager_actor = std::make_shared<MemoryManagerActor>();
  MS_EXCEPTION_IF_NULL(memory_manager_actor);
  memory_manager_aid_ = memory_manager_actor->GetAID();
  // Schedule memory manager actor, bind single thread to response to memory alloc and free quickly.
  auto base_actor = static_cast<ActorReference>(memory_manager_actor);
  (void)actorMgr->Spawn(base_actor, false);
}

ActorSet *GraphScheduler::Transform(const KernelGraphPtr &graph, const DeviceContext *device_context,
                                    const std::vector<tensor::TensorPtr> *input_tensors,
                                    GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Graph(" << graph->ToString() << ") transforms actor begin.";

  Initialize();
  PersistDeviceTensor(graph);
  auto actor_set = Build(graph, device_context);
  graph_to_actors_.emplace(graph, actor_set);
  Link(actor_set.get(), graph, strategy);

  if (!CheckActorValid(actor_set.get())) {
    MS_LOG(EXCEPTION) << "The actor set of " << graph->ToString() << " is invalid.";
  }

  MS_LOG(INFO) << "Graph(" << graph->ToString() << ") transforms actor end.";
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

void GraphScheduler::PrepareRun(const KernelGraphPtr &graph, const std::vector<TensorPtr> *input_tensors,
                                VectorRef *const &outputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input_tensors);
  MS_EXCEPTION_IF_NULL(outputs);
  // Get the device context for the first kernel actor.
  const auto &actor_set = Fetch(graph);
  MS_EXCEPTION_IF_NULL(actor_set);
  const auto &first_kernel_actor = actor_set->kernel_actors_[0];
  MS_EXCEPTION_IF_NULL(first_kernel_actor);
  const auto &device_context = first_kernel_actor->device_context_;

  // 1.Prepare the data of device tensor store(value nodes of graph).
  for (const auto &value_node : graph->graph_value_nodes()) {
    if (AnfAlgo::OutputAddrExist(value_node, 0)) {
      PrepareDataForValueNode(value_node, device_context);
    }
  }

  // 1.Prepare the data of device tensor store(weights of graph), and fill the host tensors for non weighted parameters.
  std::vector<TensorPtr> host_tensors;
  const auto &input_nodes = graph->input_nodes();
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    const auto &input_node = input_nodes[i];
    const auto &input_tensor = (*input_tensors)[i];
    MS_EXCEPTION_IF_NULL(input_node);
    if (IsPersistentDeviceTensor(input_node)) {
      // Prepare the device data for weights.
      PrepareDataForWeightNode(input_node, input_tensor, device_context);
    } else {
      // Fill the host tensors for non weighted parameters.
      host_tensors.emplace_back(input_tensor);
    }
  }

  // 2.Prepare the data of host tensor queue(non weighted parameters of graph).
  const auto &host_tensor_queue = FetchHostQueue(graph);
  MS_EXCEPTION_IF_NULL(host_tensor_queue);
  host_tensor_queue->PushData(host_tensors);

  // 3.Prepare the output tensor of graph.
  for (const auto &output_node : graph->outputs()) {
    MS_EXCEPTION_IF_NULL(output_node);
    MS_LOG(INFO) << "Create node output: " << output_node->fullname_with_scope();
    outputs->emplace_back(CreateOutputTensors(output_node, graph, *input_tensors));
  }
}

bool GraphScheduler::Run(const ActorSet *actor_set, GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(actor_set);
  // Construct OpContext.
  OpContext<DeviceTensor> op_context;
  uuids::uuid sequential_num;
  std::vector<Promise<int>> result(1);
  op_context.sequential_num_ = &sequential_num;
  op_context.results_ = &result;

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
  auto result_future = result[0].GetFuture();
  result_future.Wait();
  if (!result_future.IsOK()) {
    return false;
  }

  // Sync device stream.
  const auto &first_kernel_actor = actor_set->kernel_actors_[0];
  MS_EXCEPTION_IF_NULL(first_kernel_actor);
  const auto &device_context = first_kernel_actor->device_context_;
  MS_EXCEPTION_IF_NULL(device_context);
  if (!device_context->SyncStream()) {
    MS_LOG(ERROR) << "Sync stream failed.";
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

  auto data_source_actors = BuildDataSourceActor(graph, device_context);
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
      KernelWithIndex from_kernel_with_output_idx = AnfAlgo::GetPrevNodeOutput(kernel, i, true);
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

std::vector<DataSourceActorPtr> GraphScheduler::BuildDataSourceActor(const KernelGraphPtr &graph,
                                                                     const DeviceContext *device_context) {
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
        host_queue_ds_actor =
          std::make_shared<HostQueueDataSourceActor>(actor_name, 1, device_context, memory_manager_aid_, host_queue);
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
    auto device_queue_ds_actor =
      std::make_shared<DeviceQueueDataSourceActor>(actor_name, 1, device_context, memory_manager_aid_);
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
      auto kernel_actor =
        std::make_shared<KernelActor>(kernel->fullname_with_scope(), kernel, device_context, memory_manager_aid_);
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
  MS_LOG(INFO) << "Create loop count actor: " << actor_name;
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

  // The manager of graph member is weak ptr, so need created and used in the function IsNotRealUsedByOthers.
  const auto &manager = Manage(graph, true);
  MS_EXCEPTION_IF_NULL(manager);
  if (opt::IsNotRealUsedByOthers(graph, from_actor->kernel_)) {
    MS_EXCEPTION_IF_NULL(from_actor->kernel_);
    MS_LOG(INFO) << from_actor->kernel_->fullname_with_scope() << " is not real used by other nodes.";
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

bool GraphScheduler::CheckActorValid(const ActorSet *actor_set) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  // Check the data source actors.
  for (const auto &data_source_actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(data_source_actor);
    if (data_source_actor->output_op_arrows_.size() == 0) {
      MS_LOG(ERROR) << data_source_actor->GetAID().Name() << " has no user.";
      return false;
    }
  }

  // Check the kernel actors.
  for (const auto &kernel_actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    if (kernel_actor->output_op_arrows_.size() + kernel_actor->output_op_controls_.size() == 0) {
      MS_LOG(ERROR) << kernel_actor->GetAID().Name() << " has no user.";
      return false;
    }

    auto input_num = AnfAlgo::GetInputTensorNum(kernel_actor->kernel_);
    auto input_data_num = kernel_actor->input_datas_num_;
    auto device_tensor_store_num = kernel_actor->device_tensor_store_keys_.size();
    if (input_data_num + device_tensor_store_num != input_num) {
      MS_LOG(ERROR) << "The input building of " << kernel_actor->GetAID().Name()
                    << " is wrong, input data num: " << input_data_num
                    << ", device tensor store num: " << device_tensor_store_num << ", total input num: " << input_num;
      return false;
    }
  }

  // Check the loop count actor.
  const auto &loop_count_actor = actor_set->loop_count_actor_;
  if (loop_count_actor != nullptr) {
    if (loop_count_actor->input_controls_num_ == 0) {
      MS_LOG(ERROR) << loop_count_actor->GetAID().Name() << " has no source.";
      return false;
    }
  }

  return true;
}

void GraphScheduler::PersistDeviceTensor(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);

  for (auto &value_node : graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    if (!AnfAlgo::OutputAddrExist(value_node, 0)) {
      MS_LOG(INFO) << "The device address is not exist: " << value_node->ToString();
      continue;
    }
    auto device_tensor = AnfAlgo::GetMutableOutputAddr(value_node, 0);
    DeviceTensorStore::GetInstance().Insert(value_node.get(), device_tensor);
    device_tensor->set_ref_count(SIZE_MAX);
    device_tensor->ResetRefCountUsed();
  }

  for (auto &input_node : graph->input_nodes()) {
    MS_EXCEPTION_IF_NULL(input_node);
    if (IsPersistentDeviceTensor(input_node)) {
      auto device_tensor = AnfAlgo::GetMutableOutputAddr(input_node, 0);
      MS_EXCEPTION_IF_NULL(device_tensor);
      DeviceTensorStore::GetInstance().Insert(input_node.get(), device_tensor);
      device_tensor->set_ref_count(SIZE_MAX);
      device_tensor->ResetRefCountUsed();
    }
  }
}

HostTensorQueue *GraphScheduler::FetchHostQueue(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &iter = graph_to_host_queue_.find(graph);
  if (iter != graph_to_host_queue_.end()) {
    return iter->second.get();
  } else {
    MS_LOG(ERROR) << "Can't find the host tensor queue map of graph: " << graph->ToString();
    return nullptr;
  }
}

void GraphScheduler::DumpActor(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &actor_set = Fetch(graph);
  MS_EXCEPTION_IF_NULL(actor_set);
  std::string filename = "./actor_set_" + graph->ToString() + ".ir";
  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file [" << filename << "] failed!";
    return;
  }

  ofs << "[Data source actors]\n";
  for (const auto &data_source_actor : actor_set->data_source_actors_) {
    DumpDSActor(data_source_actor.get(), ofs);
    ofs << "\n";
  }

  ofs << "\n[Kernel actors]\n";
  for (const auto &kernel_actor : actor_set->kernel_actors_) {
    DumpKernelActor(kernel_actor.get(), ofs);
    ofs << "\n";
  }

  ofs << "\n[No input kernel actors]\n";
  for (const auto &no_input_kernel_actor : actor_set->no_input_kernel_actors_) {
    DumpKernelActor(no_input_kernel_actor.get(), ofs);
    ofs << "\n";
  }

  ofs << "\n[Loop count actor]\n";
  const auto &loop_count_actor = actor_set->loop_count_actor_;
  if (loop_count_actor != nullptr) {
    DumpLoopCountActor(loop_count_actor.get(), ofs);
    ofs << "\n";
  }
}

void GraphScheduler::DumpDSActor(const DataSourceActor *actor, std::ofstream &ofs) const {
  MS_EXCEPTION_IF_NULL(actor);
  const auto &actor_name = actor->GetAID().Name();

  MS_EXCEPTION_IF_NULL(actor->device_context_);
  ofs << "\tactor_name:" << actor_name << "\tdevice_context:" << actor->device_context_->device_context_key().ToString()
      << "\n";

  if (actor_name.find("_DeviceQueueDataSourceActor") != string::npos) {
    // Dump the member info of device queue data source actor.
    const auto &device_queue_ds_actor = dynamic_cast<const DeviceQueueDataSourceActor *>(actor);
    const auto &data_kernel = device_queue_ds_actor->data_kernel_;
    MS_EXCEPTION_IF_NULL(data_kernel);
    ofs << "\t\tdata_kernel_name:" << data_kernel->fullname_with_scope()
        << "\tinput_number:" << AnfAlgo::GetInputTensorNum(data_kernel)
        << "\toutput_number:" << AnfAlgo::GetOutputTensorNum(data_kernel) << "\n";
    for (size_t i = 0; i < AnfAlgo::GetOutputTensorNum(data_kernel); ++i) {
      const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(data_kernel, i, false);
      MS_EXCEPTION_IF_NULL(device_tensor);
      ofs << "\t\t\toutput_index:" << i << "\tptr:" << device_tensor->GetPtr() << "\tsize:" << device_tensor->GetSize()
          << "\tref_count:" << device_tensor->ref_count_dynamic_used() << "\n ";
    }
  } else if (actor_name.find("_HostQueueDataSourceActor") != string::npos) {
    // Dump the member info of host queue data source actor.
    const auto &host_queue_ds_actor = dynamic_cast<const HostQueueDataSourceActor *>(actor);
    ofs << "\t\tdata_nodes:" << host_queue_ds_actor->data_nodes_.size() << "\n";
    for (size_t i = 0; i < host_queue_ds_actor->data_nodes_.size(); ++i) {
      const auto &data_node = host_queue_ds_actor->data_nodes_[i];
      MS_EXCEPTION_IF_NULL(data_node);
      const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(data_node, 0, false);
      MS_EXCEPTION_IF_NULL(device_tensor);
      ofs << "\t\t\tnode_order_number:" << i << "\tnode_name:" << data_node->fullname_with_scope()
          << "\tptr:" << device_tensor->GetPtr() << "\tsize:" << device_tensor->GetSize()
          << "\tref_count:" << device_tensor->ref_count_dynamic_used() << "\n ";
    }
  }

  ofs << "\t\toutput_data_arrows:" << actor->output_op_arrows_.size() << "\n ";
  for (const auto &data_arrow : actor->output_op_arrows_) {
    MS_EXCEPTION_IF_NULL(data_arrow);
    ofs << "\t\t\tfrom_output_index:" << data_arrow->from_output_index_
        << "\tto_actor_name:" << data_arrow->to_op_id_.Name() << "\tto_input_index:" << data_arrow->to_input_index_
        << "\n";
  }
}

void GraphScheduler::DumpLoopCountActor(const LoopCountActor *actor, std::ofstream &ofs) const {
  MS_EXCEPTION_IF_NULL(actor);
  ofs << "\tactor_name:" << actor->GetAID().Name() << "\tloop_count:" << actor->loop_count_
      << "\tinput_controls_num:" << actor->input_controls_num_ << "\n";

  ofs << "\t\toutput_control_arrows:" << (actor->data_source_aids_.size() + actor->no_input_kernel_aids_.size())
      << "\n ";
  for (const auto &aid : actor->data_source_aids_) {
    ofs << "\t\t\tto_actor_name:" << aid.Name() << "\n";
  }
  for (const auto &aid : actor->no_input_kernel_aids_) {
    ofs << "\t\t\tto_actor_name:" << aid.Name() << "\n";
  }
}

void GraphScheduler::DumpKernelActor(const KernelActor *actor, std::ofstream &ofs) const {
  MS_EXCEPTION_IF_NULL(actor);
  MS_EXCEPTION_IF_NULL(actor->device_context_);
  ofs << "\tactor_name:" << actor->GetAID().Name()
      << "\tdevice_context:" << actor->device_context_->device_context_key().ToString()
      << "\tinput_data_num:" << actor->input_datas_num_ << "\tinput_controls_num:" << actor->input_controls_num_
      << "\n";

  const auto &kernel = actor->kernel_;
  MS_EXCEPTION_IF_NULL(kernel);
  ofs << "\t\tkernel_name:" << kernel->fullname_with_scope() << "\tinput_number:" << AnfAlgo::GetInputTensorNum(kernel)
      << "\toutput_number:" << AnfAlgo::GetOutputTensorNum(kernel) << "\n";
  for (size_t i = 0; i < AnfAlgo::GetOutputTensorNum(kernel); ++i) {
    const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(kernel, i, false);
    MS_EXCEPTION_IF_NULL(device_tensor);
    ofs << "\t\t\toutput_index:" << i << "\tptr:" << device_tensor->GetPtr() << "\tsize:" << device_tensor->GetSize()
        << "\tref_count:" << device_tensor->ref_count_dynamic_used() << "\n ";
  }

  ofs << "\t\tdevice_tensor_stores:" << actor->device_tensor_store_keys_.size() << "\n ";
  for (const auto &device_tensor_store_key : actor->device_tensor_store_keys_) {
    const auto &node = reinterpret_cast<AnfNode *>(device_tensor_store_key.second);
    MS_EXCEPTION_IF_NULL(node);
    ofs << "\t\t\tto_input_index:" << device_tensor_store_key.first
        << "\tfrom_node_name:" << node->fullname_with_scope() << "\n";
  }

  ofs << "\t\toutput_data_arrows:" << actor->output_op_arrows_.size() << "\n ";
  for (const auto &data_arrow : actor->output_op_arrows_) {
    MS_EXCEPTION_IF_NULL(data_arrow);
    ofs << "\t\t\tfrom_output_index:" << data_arrow->from_output_index_
        << "\tto_actor_name:" << data_arrow->to_op_id_.Name() << "\tto_input_index:" << data_arrow->to_input_index_
        << "\n";
  }

  ofs << "\t\toutput_control_arrows:" << actor->output_op_controls_.size() << "\n ";
  for (const auto &aid : actor->output_op_controls_) {
    ofs << "\t\t\tto_actor_name:" << aid.Name() << "\n";
  }
}

}  // namespace runtime
}  // namespace mindspore
