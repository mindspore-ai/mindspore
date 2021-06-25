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
#include "runtime/framework/actor/debug_actor.h"
#include "runtime/framework/actor/recorder_actor.h"
#include "runtime/hardware/device_context_manager.h"
#include "mindrt/src/actor/actormgr.h"
#include "mindrt/include/async/async.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/optimizer/common/helper.h"
#include "utils/config_manager.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils.h"
#include "utils/ms_context.h"
#include "common/trans.h"
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/recorder_manager.h"
#endif
#ifdef ENABLE_DEBUGGER
#include "debug/debugger/debugger.h"
#endif
namespace mindspore {
namespace runtime {
namespace {
bool IsNeedInsertCopyActor(const DeviceContext *from_devcie_context, const DeviceContext *to_devcie_context) {
  MS_EXCEPTION_IF_NULL(from_devcie_context);
  MS_EXCEPTION_IF_NULL(to_devcie_context);

  if (from_devcie_context->GetDeviceAddressType() == to_devcie_context->GetDeviceAddressType()) {
    return false;
  } else {
    return true;
  }
}

void UpdateRefCount(DeviceTensor *device_tensor, bool is_max_ref_count = false) {
  MS_EXCEPTION_IF_NULL(device_tensor);
  if (is_max_ref_count) {
    device_tensor->set_original_ref_count(SIZE_MAX);
  } else {
    device_tensor->IncreaseOriginalRefCount();
  }
  device_tensor->ResetRefCount();
}

// Update the reference count of device tensor by the output index of node.
void UpdateRefCount(const AnfNodePtr &node, size_t output_idx, bool is_max_ref_count = false) {
  MS_EXCEPTION_IF_NULL(node);
  auto device_tensor = AnfAlgo::GetMutableOutputAddr(node, output_idx, false);
  UpdateRefCount(device_tensor.get(), is_max_ref_count);
}

AnfNodePtr FetchFrontNodeByBackendNode(const AnfNodePtr &backend_node, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(backend_node);
  MS_EXCEPTION_IF_NULL(graph);
  auto front_node = graph->GetFrontAnfByBackendAnf(backend_node);
  // PyNative forward graph does not has front node, using backend node instead.
  if (front_node == nullptr) {
    front_node = backend_node;
  }
  return front_node;
}

KernelWithIndex FetchFrontNodeWithIndexByGraphOutput(const KernelWithIndex &output_with_index,
                                                     const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto front_node_with_index = graph->GetFrontNodeWithIndexByGraphOutput(output_with_index);
  // PyNative forward graph does not has front node, using backend node instead.
  if (front_node_with_index.first == nullptr) {
    front_node_with_index = output_with_index;
  }
  return front_node_with_index;
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

    const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(node, i, false);
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
                                         tensor->data_type(), tensor->data_c(), tensor->device_info().host_format_)) {
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
    const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(node, 0, false);
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
void PrepareDataForWeightNode(const AnfNodePtr &backend_node, const AnfNodePtr &front_node, const TensorPtr &tensor,
                              const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(backend_node);
  MS_EXCEPTION_IF_NULL(front_node);
  MS_EXCEPTION_IF_NULL(device_context);

  if (tensor == nullptr) {
    return;
  }

  auto device_tensor = AnfAlgo::GetMutableOutputAddr(backend_node, 0, false);
  auto host_tensor_address = std::dynamic_pointer_cast<DeviceTensor>(tensor->device_address());
  // Use the device address of host tensor to set device tensor.
  if (host_tensor_address != device_tensor) {
    if (host_tensor_address == nullptr) {
      MS_EXCEPTION_IF_NULL(device_tensor);
      host_tensor_address = device_context->CreateDeviceAddress(nullptr, device_tensor->GetSize(),
                                                                device_tensor->format(), device_tensor->type_id());
      tensor->set_device_address(host_tensor_address);
      UpdateRefCount(host_tensor_address.get(), true);
    }
    MS_EXCEPTION_IF_NULL(host_tensor_address);
    AnfAlgo::SetOutputAddr(host_tensor_address, 0, backend_node.get());
    DeviceTensorStore::GetInstance().Insert(front_node.get(), host_tensor_address);
  }

  // If the ptr of device tensor is not nullptr, it indicates that the device data has been prepared.
  if (host_tensor_address->GetPtr() != nullptr) {
    return;
  }
  MS_LOG(INFO) << "Prepare device data for weight node: " << backend_node->fullname_with_scope();

  // Allocate device memory and copy data from host tensor to device.
  if (!device_context->AllocateMemory(host_tensor_address.get(), host_tensor_address->GetSize())) {
    MS_LOG(EXCEPTION) << "Device memory isn't enough and alloc failed, node name: "
                      << backend_node->fullname_with_scope();
  }
  if (!host_tensor_address->SyncHostToDevice(trans::GetRuntimePaddingShape(backend_node, 0),
                                             LongToSize(tensor->data().nbytes()), tensor->data_type(), tensor->data_c(),
                                             tensor->device_info().host_format_)) {
    MS_LOG(EXCEPTION) << "SyncHostToDevice failed, node name: " << backend_node->fullname_with_scope();
  }

  // Allocate another device memory and copy data from host tensor to another device(if exist).
  const auto &device_tensors = DeviceTensorStore::GetInstance().Fetch(front_node.get());
  if (device_tensors.size() > 1) {
    auto another_device_tensor = (device_tensors[0] == host_tensor_address) ? device_tensors[1] : device_tensors[0];
    MS_EXCEPTION_IF_NULL(another_device_tensor);
    auto another_device_type = another_device_tensor->DeviceType();
    const auto &another_device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {device::kDeviceTypeToName.at(another_device_type), device_context->device_context_key().device_id_});
    MS_EXCEPTION_IF_NULL(another_device_context);
    if (!another_device_context->AllocateMemory(another_device_tensor.get(), another_device_tensor->GetSize())) {
      MS_LOG(EXCEPTION) << "Device memory isn't enough and alloc failed, node name: "
                        << backend_node->fullname_with_scope();
    }
    if (!another_device_tensor->SyncHostToDevice(trans::GetRuntimePaddingShape(backend_node, 0),
                                                 LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                                 tensor->data_c())) {
      MS_LOG(EXCEPTION) << "SyncHostToDevice failed, node name: " << backend_node->fullname_with_scope();
    }
  }
}

// In control flow, all weight nodes associated with the host weight parameter need to use the same device tensor.
void PrepareDataForControlWeightNode(
  const AnfNodePtr &node, const AnfNodePtr &front_node, const TensorPtr &tensor, const DeviceContext *device_context,
  const std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> &host_parameter_to_weights = {}) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(front_node);
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(device_context);

  auto device_tensors = DeviceTensorStore::GetInstance().Fetch(front_node.get());
  if (device_tensors.empty()) {
    PrepareDataForWeightNode(node, front_node, tensor, device_context);
  }

  const auto iter = host_parameter_to_weights.find(front_node);
  if (iter == host_parameter_to_weights.end()) {
    return;
  }

  // Fetch all the device tensors of host weight node and insert as the weight of other nodes.
  const auto &sub_front_nodes = host_parameter_to_weights.at(front_node);
  device_tensors = DeviceTensorStore::GetInstance().Fetch(front_node.get());
  for (const auto &sub_front_node : sub_front_nodes) {
    for (const auto &device_tensor : device_tensors) {
      if (sub_front_node == nullptr) {
        MS_LOG(EXCEPTION) << "Front node is empty!";
      }
      DeviceTensorStore::GetInstance().Insert(sub_front_node.get(), device_tensor);
    }
  }
}

void EraseValueNodeTensor(const std::vector<int64_t> *tensors_mask, const std::vector<TensorPtr> *input_tensors,
                          std::vector<TensorPtr> *input_tensors_without_value_node) {
  MS_EXCEPTION_IF_NULL(input_tensors);
  if (input_tensors->size() != tensors_mask->size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_tensors->size() << " should be equal to tensors mask size "
                      << tensors_mask->size();
  }
  for (size_t index = 0; index < tensors_mask->size(); ++index) {
    if (tensors_mask->at(index) != kValueNodeTensorMask) {
      input_tensors_without_value_node->emplace_back(input_tensors->at(index));
    }
  }
}

TensorPtr FetchInputTensor(const GraphCompilerInfo &graph_compiler_info, size_t graph_index, size_t input_index) {
  if (graph_index < graph_compiler_info.input_tensors_.size()) {
    const std::vector<TensorPtr> *input_tensors = graph_compiler_info.input_tensors_[graph_index];
    if (input_index < input_tensors->size()) {
      return input_tensors->at(input_index);
    }
  }
  return nullptr;
}

void PrepareDataForHostDataSourceActor(const std::unordered_map<AnfNodePtr, size_t> &data_node_position_map,
                                       const AnfNodePtr &node, const TensorPtr &tensor,
                                       std::vector<TensorPtr> *host_tensors,
                                       const DeviceContext *device_context = nullptr,
                                       GraphExecutionStrategy strategy = GraphExecutionStrategy::kPipeline) {
  MS_EXCEPTION_IF_NULL(tensor);

  // Fill the host tensors for non weighted parameters.
  const auto &iter = data_node_position_map.find(node);
  if (iter == data_node_position_map.end()) {
    return;
  }

  (*host_tensors)[iter->second] = tensor;
  auto device_address = std::dynamic_pointer_cast<DeviceTensor>(tensor->device_address());
  if (device_address != nullptr) {
    AnfAlgo::SetOutputAddr(device_address, 0, node.get());
    return;
  }

  if (strategy == GraphExecutionStrategy::kStep) {
    auto node_device_address = AnfAlgo::GetMutableOutputAddr(node, 0, false);
    MS_EXCEPTION_IF_NULL(node_device_address);
    tensor->set_device_address(node_device_address);
    UpdateRefCount(node_device_address.get(), true);

    MS_EXCEPTION_IF_NULL(device_context);
    if (!device_context->AllocateMemory(node_device_address.get(), node_device_address->GetSize())) {
      MS_LOG(EXCEPTION) << "Device memory isn't enough and alloc failed, node name: " << node->fullname_with_scope();
    }

    if (!node_device_address->SyncHostToDevice(trans::GetRuntimePaddingShape(node, 0),
                                               LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                               tensor->data_c(), tensor->device_info().host_format_)) {
      MS_LOG(EXCEPTION) << "SyncHostToDevice failed.";
    }
  }
}

inline bool IsSingleOpActorSet(const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  return actor_set->kernel_actors_.size() == 1;
}
}  // namespace

void GraphScheduler::Clear() {
  // Terminate all actors.
  auto actorMgr = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actorMgr);
  actorMgr->Finalize();

  // Clear the member of DeviceTensorStore.
  DeviceTensorStore::GetInstance().Clear();

  // Clear global maps.
  actors_.clear();
  actor_name_to_actor_.clear();
  actor_to_host_queue_.clear();
  device_tensor_to_actor_.clear();

  // Clear local maps and vectors.
  graph_output_to_actor_.clear();
  front_node_to_actor_.clear();
  copy_actors_.clear();

  // Delete the thread pool.
  delete thread_pool_;
  thread_pool_ = nullptr;
}

void GraphScheduler::Initialize() {
  // Local maps and vectors clear.
  graph_output_to_actor_.clear();
  front_node_to_actor_.clear();
  copy_actors_.clear();

  if (init_) {
    return;
  }
  init_ = true;

  auto actorMgr = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actorMgr);
  actorMgr->Initialize();

  // Create the thread pool of actor runtime and Set the OMP_NUM_THREADS env.
  size_t actor_thread_num = 0;
  size_t OMP_thread_num = 0;
  ComputeThreadNums(&actor_thread_num, &OMP_thread_num);
  thread_pool_ = ActorThreadPool::CreateThreadPool(actor_thread_num, kThreadWait);
  MS_EXCEPTION_IF_NULL(thread_pool_);
  std::string OMP_env = std::to_string(OMP_thread_num);
  common::SetEnv("OMP_NUM_THREADS", OMP_env.c_str(), 0);
  auto OMP_thread_num_used = common::GetEnv("OMP_NUM_THREADS");
  MS_LOG(INFO) << "The actor thread number: " << actor_thread_num
               << ", the computed OMP thread number : " << OMP_thread_num
               << ", the used OMP thread number : " << stoi(OMP_thread_num_used);

  // Create and schedule memory manager actor.
  auto memory_manager_actor = std::make_shared<MemoryManagerActor>();
  MS_EXCEPTION_IF_NULL(memory_manager_actor);
  memory_manager_aid_ = memory_manager_actor->GetAID();
  auto base_actor = static_cast<ActorReference>(memory_manager_actor);
  base_actor->set_thread_pool(thread_pool_);
  // Bind single thread to response to memory alloc and free quickly.
  (void)actorMgr->Spawn(base_actor, false);

// Create and schedule recorder actor.
#ifdef ENABLE_DUMP_IR
  if (mindspore::RecorderManager::Instance().RdrEnable()) {
    auto recorder_actor = std::make_shared<RecorderActor>();
    MS_EXCEPTION_IF_NULL(recorder_actor);
    recorder_aid_ = &(recorder_actor->GetAID());
    auto base_recorder_actor = static_cast<ActorReference>(recorder_actor);
    base_recorder_actor->set_thread_pool(thread_pool_);
    (void)actorMgr->Spawn(base_recorder_actor, true);
  }
#endif
// Create and schedule debug actor.
#ifdef ENABLE_DEBUGGER
  auto debugger = mindspore::Debugger::GetInstance();
  if (debugger->DebuggerBackendEnabled()) {
    auto debug_actor = std::make_shared<DebugActor>();
    MS_EXCEPTION_IF_NULL(debug_actor);
    debug_aid_ = &(debug_actor->GetAID());
    auto base_debug_actor = static_cast<ActorReference>(debug_actor);
    base_debug_actor->set_thread_pool(thread_pool_);
    (void)actorMgr->Spawn(base_debug_actor, true);
  }
#endif
}

ActorSet *GraphScheduler::Transform(const GraphCompilerInfo &graph_compiler_info, GraphExecutionStrategy strategy) {
  MS_LOG(INFO) << "Graph(" << graph_compiler_info.name_ << ") transforms actor begin.";
  if (graph_compiler_info.graphs_.size() == 0) {
    MS_LOG(EXCEPTION) << "The number of graphs is zero.";
  }
  if (graph_compiler_info.graphs_.size() != graph_compiler_info.device_contexts_.size()) {
    MS_LOG(EXCEPTION) << "The number of graphs is not equal to the number of device contexts.";
  }

  PersistDeviceTensor(graph_compiler_info);
  const auto &actor_set = Build(graph_compiler_info, strategy);
  CacheGraphOutputToActor(graph_compiler_info);
  Link(actor_set.get(), graph_compiler_info, strategy);
  // The copy actors are built in the link, so need push into the actor set after link.
  actor_set->copy_actors_ = copy_actors_;

  actors_.emplace(actor_set->name_, actor_set);

  DumpActor(actor_set.get(), graph_compiler_info);
  if (!CheckActorValid(actor_set.get(), strategy)) {
    MS_LOG(EXCEPTION) << "The actor set of " << graph_compiler_info.name_ << " is invalid.";
  }
  MS_LOG(INFO) << "Graph(" << graph_compiler_info.name_ << ") transforms actor end.";
  return actor_set.get();
}

void GraphScheduler::Schedule(const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  std::vector<ActorReference> actors;

  // Collect actors.
  for (auto &data_source_actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(data_source_actor);
    actors.emplace_back(static_cast<ActorReference>(data_source_actor));
  }
  for (auto &kernel_actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    actors.emplace_back(static_cast<ActorReference>(kernel_actor));
  }
  for (auto &switch_actor : actor_set->switch_actors_) {
    MS_EXCEPTION_IF_NULL(switch_actor);
    actors.emplace_back(static_cast<ActorReference>(switch_actor));
  }
  for (auto &gather_actor : actor_set->gather_actors_) {
    MS_EXCEPTION_IF_NULL(gather_actor);
    actors.emplace_back(static_cast<ActorReference>(gather_actor));
  }
  for (auto &copy_actor : actor_set->copy_actors_) {
    MS_EXCEPTION_IF_NULL(copy_actor);
    actors.emplace_back(static_cast<ActorReference>(copy_actor));
  }
  if (actor_set->loop_count_actor_ != nullptr) {
    actors.emplace_back(static_cast<ActorReference>(actor_set->loop_count_actor_));
  }
  if (actor_set->output_actor_ != nullptr) {
    actors.emplace_back(static_cast<ActorReference>(actor_set->output_actor_));
  }

  // Schedule actors.
  auto actorMgr = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actorMgr);
  for (auto actor : actors) {
    actor->set_thread_pool(thread_pool_);
    (void)actorMgr->Spawn(actor);
  }
}

void GraphScheduler::PrepareRun(const ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info,
                                const std::vector<std::vector<TensorPtr>> &input_tensors,
                                GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(actor_set);
  std::vector<TensorPtr> host_tensors;
  std::string actor_name = actor_set->name_ + "_HostDSActor";
  const auto &host_data_source_actor = dynamic_cast<HostQueueDataSourceActor *>(FetchActor(actor_name));
  if (host_data_source_actor != nullptr) {
    host_tensors.resize(host_data_source_actor->data_nodes_.size());
  }

  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    const auto &device_context = graph_compiler_info.device_contexts_[i];
    MS_EXCEPTION_IF_NULL(graph);

    // 1.Prepare the data of device tensor store(value nodes of graph).
    for (const auto &value_node : graph->graph_value_nodes()) {
      if (AnfAlgo::OutputAddrExist(value_node, 0)) {
        PrepareDataForValueNode(value_node, device_context);
      }
    }

    // 1.Prepare the data of device tensor store(weights of graph), and fill host tensors for non weighted parameters.
    const auto &input_nodes = graph->input_nodes();
    const auto &tensors = input_tensors[i];
    for (size_t j = 0; j < input_nodes.size(); ++j) {
      const auto &input_node = input_nodes[j];
      const auto &input_tensor = tensors[j];
      MS_EXCEPTION_IF_NULL(input_node);
      if (IsPersistentDeviceTensor(input_node)) {
        // Prepare the device data for weights.
        const auto front_node = FetchFrontNodeByBackendNode(input_node, graph);
        PrepareDataForWeightNode(input_node, front_node, input_tensor, device_context);
      } else if (IsHostQueueDSActor(input_node, graph, input_tensor, graph_compiler_info.origin_parameters_order_,
                                    strategy)) {
        MS_EXCEPTION_IF_NULL(host_data_source_actor);
        PrepareDataForHostDataSourceActor(host_data_source_actor->data_node_position_map_, input_node, input_tensor,
                                          &host_tensors, device_context, strategy);
      }
    }
  }

  // 2.Prepare the continuous memory for communication kernel.
  if (actor_set->loop_count_actor_ != nullptr) {
    auto alloc_list_list = actor_set->loop_count_actor_->continuous_memory_alloc_list_list_;
    auto size_list_list = actor_set->loop_count_actor_->size_list_list_;
    auto total_size_list = actor_set->loop_count_actor_->total_size_list_;
    auto device_contexts = actor_set->loop_count_actor_->device_contexts_;
    if ((alloc_list_list.size() != size_list_list.size()) || (size_list_list.size() != total_size_list.size()) ||
        (total_size_list.size() != device_contexts.size())) {
      MS_LOG(EXCEPTION)
        << "The size of alloc_list_list, size_list_list, total_size_list and device_contexts are not equal.";
    }
    for (size_t i = 0; i < alloc_list_list.size(); ++i) {
      auto &alloc_list = alloc_list_list[i];
      auto &size_list = size_list_list[i];
      auto &total_size = total_size_list[i];
      auto &device_context = device_contexts[i];
      if (!device_context->AllocateContinuousMemory(alloc_list, total_size, size_list)) {
        MS_LOG(EXCEPTION) << "Device memory isn't enough and alloc failed, alloc size: " << total_size;
      }
    }
  }

  // 3.Prepare the data which belongs to control node.
  PrepareDataForControlNode(graph_compiler_info.control_node_parser_, graph_compiler_info.origin_parameters_order_,
                            input_tensors.back(), host_data_source_actor->data_node_position_map_, &host_tensors);

  // 4.Prepare the data of host tensor queue(non weighted parameters of graph).
  if (host_data_source_actor != nullptr) {
    const auto &host_tensor_queue = FetchHostQueue(actor_set->name_);
    MS_EXCEPTION_IF_NULL(host_tensor_queue);
    host_tensor_queue->Push(host_tensors);
  }
}

void GraphScheduler::PrepareDataForControlNode(const ControlNodeParserPtr &control_node_parser,
                                               const std::vector<AnfNodePtr> &origin_parameters,
                                               const std::vector<TensorPtr> &tensors,
                                               const std::unordered_map<AnfNodePtr, size_t> &data_node_position_map,
                                               std::vector<TensorPtr> *host_tensors) {
  const auto &control_node_parameters = control_node_parser->GetControlNodeParameter();

  for (size_t j = 0; j < control_node_parameters.size(); ++j) {
    const auto &input_node = control_node_parameters[j];
    const auto &input_tensor = tensors[j];
    MS_EXCEPTION_IF_NULL(input_node);
    if (IsPersistentDeviceTensor(input_node)) {
      const auto &front_to_backend_parameters = control_node_parser->front_to_backend_parameters_;
      const auto &iter = front_to_backend_parameters.find(input_node);
      if (iter == front_to_backend_parameters.end()) {
        MS_LOG(EXCEPTION) << "Cannot find backend node for weight parameter:"
                          << AnfAlgo::GetNodeDebugString(input_node);
      }
      const auto &node_with_context = iter->second;
      PrepareDataForControlWeightNode(node_with_context.first, input_node, input_tensor, node_with_context.second,
                                      control_node_parser->host_parameter_to_weights_);
    } else if (find(origin_parameters.begin(), origin_parameters.end(), input_node) != origin_parameters.end()) {
      PrepareDataForHostDataSourceActor(data_node_position_map, input_node, input_tensor, host_tensors);
    }
  }

  for (const auto &value_node_with_context : control_node_parser->front_value_nodes_) {
    if (AnfAlgo::OutputAddrExist(value_node_with_context.first, 0)) {
      PrepareDataForValueNode(value_node_with_context.first->cast<ValueNodePtr>(), value_node_with_context.second);
    }
  }
}

bool GraphScheduler::Run(const ActorSet *actor_set, GraphExecutionStrategy strategy,
                         const std::vector<TensorPtr> *input_tensors) {
  MS_EXCEPTION_IF_NULL(actor_set);
  // Construct OpContext.
  OpContext<DeviceTensor> op_context;
  uuids::uuid sequential_num;
  std::vector<Promise<int>> result(1);
  // Step mode does not need sequential number.
  op_context.sequential_num_ = (strategy == GraphExecutionStrategy::kPipeline) ? &sequential_num : nullptr;
  op_context.results_ = &result;

  // Trigger data source actor running.
  for (auto &data_source_actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(data_source_actor);
    Async(data_source_actor->GetAID(), &DataSourceActor::FetchData, &op_context);
  }

  // Trigger no input kernel actor running.
  for (auto &no_input_kernel_actor : actor_set->no_input_kernel_actors_) {
    MS_EXCEPTION_IF_NULL(no_input_kernel_actor);
    Async(no_input_kernel_actor->GetAID(), &KernelActor::RunOpControl, nullptr, &op_context);
  }

  // Trigger kernel actor running in the step execution strategy.
  if (strategy == GraphExecutionStrategy::kStep && IsSingleOpActorSet(actor_set)) {
    MS_EXCEPTION_IF_NULL(input_tensors);
    for (auto &kernel_actor : actor_set->kernel_actors_) {
      MS_EXCEPTION_IF_NULL(kernel_actor);
      Async(kernel_actor->GetAID(), &KernelActor::RunOpControlWithInputTensor, nullptr, &op_context, input_tensors);
    }
  }

  // Trigger output actor running when there are no data source actor and kernel actor.
  if ((actor_set->data_source_actors_.size() == 0) && (actor_set->kernel_actors_.size() == 0)) {
    MS_EXCEPTION_IF_NULL(actor_set->output_actor_);
    Async(actor_set->output_actor_->GetAID(), &OutputActor::CollectLoopCount, actor_set->output_actor_->loop_count_,
          &op_context);
  }

  // Get the run result.
  auto result_future = result[0].GetFuture();
  result_future.Wait();
  MsException::Instance().CheckException();
  return result_future.IsOK();
}

ActorSet *GraphScheduler::Fetch(const ActorInfo &actor_info) const {
  auto iter = actors_.find(actor_info);
  if (iter != actors_.end()) {
    return iter->second.get();
  } else {
    MS_LOG(ERROR) << "Can't find the actors map of " << actor_info;
    return nullptr;
  }
}

ActorSetPtr GraphScheduler::Build(const GraphCompilerInfo &graph_compiler_info, GraphExecutionStrategy strategy) {
  auto actor_set = std::make_shared<ActorSet>(graph_compiler_info.name_);
  MS_EXCEPTION_IF_NULL(actor_set);

  auto host_queue = std::make_shared<HostTensorQueue>();
  actor_to_host_queue_.emplace(actor_set->name_, host_queue);
  actor_set->data_source_actors_ = BuildDataSourceActor(graph_compiler_info, host_queue);
  actor_set->kernel_actors_ = BuildKernelActor(graph_compiler_info);
  actor_set->loop_count_actor_ = BuildLoopCountActor(graph_compiler_info, strategy);
  actor_set->output_actor_ = BuildOutputActor(graph_compiler_info, strategy);
  actor_set->switch_actors_ = BuildSwitchActor(graph_compiler_info);
  actor_set->gather_actors_ = BuildGatherActor(graph_compiler_info);

  return actor_set;
}

void GraphScheduler::CacheGraphOutputToActor(const GraphCompilerInfo &graph_compiler_info) {
  for (const auto &graph : graph_compiler_info.graphs_) {
    MS_EXCEPTION_IF_NULL(graph);
    auto outputs = AnfAlgo::GetAllOutputWithIndex(graph->output());
    for (const auto &output_with_index : outputs) {
      auto output_kernel = output_with_index.first;
      MS_EXCEPTION_IF_NULL(output_kernel);
      auto origin_output_with_index = graph->GetFrontNodeWithIndexByGraphOutput(output_with_index);
      if (origin_output_with_index.first == nullptr) {
        continue;
      }

      auto actor_output_index = output_with_index.second;
      OpActor<DeviceTensor> *actor = nullptr;
      if (IsKernelActor(output_kernel)) {
        actor = FetchActor(output_kernel->fullname_with_scope());
      } else if (IsDeviceQueueDSActor(output_kernel)) {
        std::string actor_name = graph_compiler_info.name_ + "_DeviceDSActor" + "_" + std::to_string(graph->graph_id());
        actor = FetchActor(actor_name);
      } else if (IsHostQueueDSActor(output_kernel, graph, nullptr, graph_compiler_info.origin_parameters_order_)) {
        actor = FetchActor(graph_compiler_info.name_ + "_HostDSActor");
        const auto &host_ds_actor = dynamic_cast<HostQueueDataSourceActor *>(actor);
        MS_EXCEPTION_IF_NULL(host_ds_actor);
        // Get the position of output kernel in the data source actor.
        actor_output_index = host_ds_actor->FetchDataNodePosition(output_kernel);
      } else if (IsPersistentDeviceTensor(output_kernel)) {
        MS_LOG(INFO) << "The graph " << graph->graph_id() << " output node:" << output_kernel->fullname_with_scope()
                     << " is device tensor store.";
        continue;
      } else {
        MS_LOG(WARNING) << "Invalid graph output node:" << output_kernel->fullname_with_scope();
        continue;
      }

      MS_EXCEPTION_IF_NULL(actor);
      MS_LOG(INFO) << "Cache the graph " << graph->graph_id() << " output node:" << output_kernel->fullname_with_scope()
                   << " with index: " << output_with_index.second << " to actor:" << actor->GetAID().Name()
                   << " with index:" << actor_output_index;
      graph_output_to_actor_.emplace(origin_output_with_index, GraphOutputPair(actor, actor_output_index));
    }
  }
}

void GraphScheduler::Link(ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info,
                          GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(actor_set);
  std::vector<KernelActor *> auto_monad_actors;

  // Foreach the execution order to link the actors.
  for (size_t index = 0; index < graph_compiler_info.graphs_.size(); ++index) {
    const auto &graph = graph_compiler_info.graphs_[index];
    MS_EXCEPTION_IF_NULL(graph);
    auto execution_order = graph->execution_order();
    for (auto &kernel : execution_order) {
      if (IsSkippedKernelActor(kernel) || (!IsKernelActor(kernel))) {
        continue;
      }
      const auto &kernel_actor = dynamic_cast<KernelActor *>(FetchActor(kernel->fullname_with_scope()));
      MS_EXCEPTION_IF_NULL(kernel_actor);

      for (size_t i = 0; i < AnfAlgo::GetInputNum(kernel); ++i) {
        auto input_node = AnfAlgo::GetInputNode(kernel, i);
        // Link the control arrows of kernel actor by the auto monad, the inputs include monad node.
        LinkControlArrowByAutoMonad(kernel_actor, input_node);
        if (HasAbstractMonad(input_node)) {
          auto_monad_actors.emplace_back(kernel_actor);
          continue;  // No data arrow for monad input.
        }

        KernelWithIndex from_kernel_with_output_idx = AnfAlgo::VisitKernelWithReturnType(input_node, 0, false);
        KernelWithIndex to_kernel_with_input_idx = std::make_pair(kernel, i);
        TensorPtr tensor = IsSingleOpActorSet(actor_set) ? FetchInputTensor(graph_compiler_info, index, i) : nullptr;
        // The gather of linking data arrows of kernel by the different from kernel type.
        LinkDataArrow(kernel_actor, graph_compiler_info, graph, from_kernel_with_output_idx, to_kernel_with_input_idx,
                      tensor);
      }
    }
    // Link the control arrows for allreduce kernel by the send/recv nodes in the kernel graph.
    LinkControlArrowBySendRecvNodes(graph);
    // Link the control arrows by the communication nodes to ensure communication nodes running order.
    LinkControlArrowByCommunicationNode(graph);
  }

  // Link the arrow by control node.
  LinkArrowByControlNode(graph_compiler_info, actor_set);

  // Auto monad actor may modify the device tensor store.
  LinkDeviceTensorStoreForAutoMonadActor(auto_monad_actors);

  // BuildNoInputKernelActor depends on whether kernel actors have input, so must be behind the link of kernel actors.
  actor_set->no_input_kernel_actors_ = BuildNoInputKernelActor(actor_set, strategy);

  // Link the control arrows of loop count actor, which depends on the no input kernel actors.
  LinkControlArrowForLoopCountActor(actor_set->loop_count_actor_.get(), actor_set);

  // Link the output result arrows for output actors.
  LinkOutputResultArrowForOutputActor(actor_set->output_actor_.get(), graph_compiler_info);
}

std::vector<DataSourceActorPtr> GraphScheduler::BuildDataSourceActor(const GraphCompilerInfo &graph_compiler_info,
                                                                     const HostTensorQueuePtr &host_queue) {
  std::vector<DataSourceActorPtr> data_source_actors;
  HostQueueDSActorPtr host_queue_ds_actor = nullptr;
  size_t data_node_position = 0;
  std::unordered_map<AnfNodePtr, size_t> front_node_position_temp_map;

  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    const auto &device_context = graph_compiler_info.device_contexts_[i];
    MS_EXCEPTION_IF_NULL(graph);
    // Build host queue data source actor.
    const std::vector<AnfNodePtr> &input_nodes = graph->input_nodes();
    const std::vector<TensorPtr> *input_tensors = nullptr;
    TensorPtr tensor = nullptr;
    std::vector<TensorPtr> tensors_without_value_node;

    if (graph_compiler_info.input_tensors_.size() != 0) {
      // Erase value node tensor.
      EraseValueNodeTensor(graph_compiler_info.tensors_mask_[i], graph_compiler_info.input_tensors_[i],
                           &tensors_without_value_node);
      if (tensors_without_value_node.size() != input_nodes.size()) {
        MS_LOG(EXCEPTION) << "Tensor input:" << tensors_without_value_node.size()
                          << " is not equal graph inputs:" << input_nodes.size();
      }
      input_tensors = &tensors_without_value_node;
    }

    for (size_t j = 0; j < input_nodes.size(); j++) {
      const auto &input_node = input_nodes[j];
      MS_EXCEPTION_IF_NULL(input_node);
      if (input_tensors != nullptr && j < input_tensors->size()) {
        tensor = input_tensors->at(j);
      }
      if (IsHostQueueDSActor(input_node, graph, tensor, graph_compiler_info.origin_parameters_order_)) {
        if (host_queue_ds_actor == nullptr) {
          auto actor_name = graph_compiler_info.name_ + "_HostDSActor";
          MS_LOG(INFO) << "Create host queue data source actor: " << actor_name;
          host_queue_ds_actor = std::make_shared<HostQueueDataSourceActor>(actor_name, 1, memory_manager_aid_, nullptr,
                                                                           nullptr, host_queue);
          InsertActor(host_queue_ds_actor.get());
          data_source_actors.emplace_back(host_queue_ds_actor);
        }

        const auto &front_node = FetchFrontNodeByBackendNode(input_node, graph);
        // In the scenario where multiple backend nodes correspond to the same front node, only the first backend node
        // is saved in the host queue data source actor.
        if (front_node_position_temp_map.count(front_node) > 0) {
          host_queue_ds_actor->data_node_position_map_.emplace(input_node, front_node_position_temp_map[front_node]);
          continue;
        }
        host_queue_ds_actor->data_nodes_.emplace_back(input_node);
        host_queue_ds_actor->device_contexts_.emplace_back(device_context);
        host_queue_ds_actor->data_node_position_map_.emplace(input_node, data_node_position);
        front_node_position_temp_map.emplace(front_node, data_node_position);
        data_node_position++;
      }
    }

    // Build device queue data source actor.
    const auto &execution_order = graph->execution_order();
    const auto &iter = std::find_if(execution_order.begin(), execution_order.end(),
                                    [](const CNodePtr &node) { return IsDeviceQueueDSActor(node); });
    if (iter != execution_order.end()) {
      auto actor_name = graph_compiler_info.name_ + "_DeviceDSActor" + "_" + std::to_string(graph->graph_id());
      MS_LOG(INFO) << "Create queue data source actor: " << actor_name;
      auto device_queue_ds_actor = std::make_shared<DeviceQueueDataSourceActor>(
        actor_name, 1, device_context, memory_manager_aid_, debug_aid_, recorder_aid_);
      MS_EXCEPTION_IF_NULL(device_queue_ds_actor);
      InsertActor(device_queue_ds_actor.get());
      data_source_actors.emplace_back(device_queue_ds_actor);
      device_queue_ds_actor->data_kernel_ = *iter;
      device_queue_ds_actor->kernel_info_ = static_cast<device::KernelInfo *>((*iter)->kernel_info());
    }
  }

  const auto &front_to_backend_parameter = graph_compiler_info.control_node_parser_->front_to_backend_parameters_;

  // Initialize the parameter in the control node, first get all the front parameters in the control node, then find
  // the corresponding backend parameter from the map, and insert it into the host data source actor
  std::vector<AnfNodePtr> control_node_parameters = graph_compiler_info.control_node_parser_->GetControlNodeParameter();
  for (const auto parameter : control_node_parameters) {
    if (IsPersistentDeviceTensor(parameter)) {
      continue;
    }
    auto backend_iter = front_to_backend_parameter.find(parameter);
    if (backend_iter == front_to_backend_parameter.end()) {
      MS_LOG(EXCEPTION) << "Cannot find backend node for front node:" << AnfAlgo::GetNodeDebugString(parameter);
    }

    if (host_queue_ds_actor == nullptr) {
      auto actor_name = graph_compiler_info.name_ + "_HostDSActor";
      MS_LOG(INFO) << "Create host queue data source actor: " << actor_name;
      host_queue_ds_actor =
        std::make_shared<HostQueueDataSourceActor>(actor_name, 1, memory_manager_aid_, nullptr, nullptr, host_queue);
      InsertActor(host_queue_ds_actor.get());
      data_source_actors.emplace_back(host_queue_ds_actor);
    }

    const auto &backend_node = backend_iter->second.first;
    auto iter = find(host_queue_ds_actor->data_nodes_.begin(), host_queue_ds_actor->data_nodes_.end(), backend_node);

    if (iter != host_queue_ds_actor->data_nodes_.end()) {
      host_queue_ds_actor->data_node_position_map_.emplace(parameter, iter - host_queue_ds_actor->data_nodes_.begin());
    } else {
      host_queue_ds_actor->data_node_position_map_.emplace(parameter, host_queue_ds_actor->data_nodes_.size());
      host_queue_ds_actor->data_nodes_.emplace_back(backend_iter->second.first);
      host_queue_ds_actor->device_contexts_.emplace_back(backend_iter->second.second);
    }
  }
  return data_source_actors;
}

std::vector<KernelActorPtr> GraphScheduler::BuildKernelActor(const GraphCompilerInfo &graph_compiler_info) {
  std::vector<KernelActorPtr> kernel_actors;

  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    const auto &device_context = graph_compiler_info.device_contexts_[i];
    MS_EXCEPTION_IF_NULL(graph);
    auto execution_order = graph->execution_order();
    for (auto &kernel : execution_order) {
      if (IsKernelActor(kernel) && (!IsSkippedKernelActor(kernel))) {
        auto kernel_actor = std::make_shared<KernelActor>(kernel->fullname_with_scope(), kernel, device_context,
                                                          memory_manager_aid_, debug_aid_, recorder_aid_);
        MS_EXCEPTION_IF_NULL(kernel_actor);
        InsertActor(kernel_actor.get());
        kernel_actors.emplace_back(kernel_actor);
        auto front_node = graph->GetFrontAnfByBackendAnf(kernel);
        if (front_node != nullptr) {
          front_node_to_actor_[front_node] = kernel_actor;
        }
      }
    }
  }
  return kernel_actors;
}

LoopCountActorPtr GraphScheduler::BuildLoopCountActor(const GraphCompilerInfo &graph_compiler_info,
                                                      GraphExecutionStrategy strategy) {
  if (strategy == GraphExecutionStrategy::kStep) {
    return nullptr;
  }

  auto loop_count = ConfigManager::GetInstance().iter_num();
  auto actor_name = graph_compiler_info.name_ + "_LoopCountActor";
  auto loop_count_actor =
    std::make_shared<LoopCountActor>(actor_name, loop_count, memory_manager_aid_, debug_aid_, recorder_aid_);
  MS_LOG(INFO) << "Create loop count actor: " << actor_name;
  MS_EXCEPTION_IF_NULL(loop_count_actor);

  // Cache the nodes which need continuous memory.
  for (size_t index = 0; index < graph_compiler_info.graphs_.size(); ++index) {
    const auto &graph = graph_compiler_info.graphs_[index];
    MS_EXCEPTION_IF_NULL(graph);
    auto &execution_order = graph->execution_order();
    for (auto &kernel : execution_order) {
      if (!AnfAlgo::IsCommunicationOp(kernel)) {
        continue;
      }

      auto key = std::make_pair(kernel, graph_compiler_info.device_contexts_[index]);
      auto value = std::make_pair(false, false);
      if (AnfAlgo::GetInputTensorNum(kernel) > 1) {
        value.first = true;
      }
      if (AnfAlgo::GetOutputTensorNum(kernel) > 1) {
        value.second = true;
      }
      if ((value.first == true) || (value.second == true)) {
        loop_count_actor->continuous_memory_nodes_[key] = value;
      }
    }
  }

  InsertActor(loop_count_actor.get());
  return loop_count_actor;
}

OutputActorPtr GraphScheduler::BuildOutputActor(const GraphCompilerInfo &graph_compiler_info,
                                                GraphExecutionStrategy strategy) {
  auto loop_count = ConfigManager::GetInstance().iter_num();
  auto actor_name = graph_compiler_info.name_ + "_" + "OutputActor";
  bool need_loop_count = (strategy == GraphExecutionStrategy::kPipeline) ? true : false;

  auto output_actor =
    std::make_shared<OutputActor>(actor_name, loop_count, graph_compiler_info.outputs_num_, need_loop_count);
  MS_LOG(INFO) << "Create output actor: " << actor_name;
  MS_EXCEPTION_IF_NULL(output_actor);
  InsertActor(output_actor.get());
  return output_actor;
}

std::vector<KernelActorPtr> GraphScheduler::BuildNoInputKernelActor(const ActorSet *actor_set,
                                                                    GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(actor_set);
  std::vector<KernelActorPtr> no_input_kernel_actors;

  for (auto &kernel_actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    // Framework will trigger kernel actor running in the step execution strategy.
    if (strategy == GraphExecutionStrategy::kStep && IsSingleOpActorSet(actor_set)) {
      kernel_actor->input_controls_num_++;
      continue;
    }

    if ((kernel_actor->input_datas_num_ == 0) && (kernel_actor->input_controls_num_ == 0)) {
      // Check whether the kernel actor belongs to the root graph.
      // In general, all no input nodes belong to the root funcgraph, and the corresponding gather actor should be
      // empty. In control flow, the control arrow of the no input node in the sub funcgraph should be sent by the
      // gather actor and should not be placed in the no input list.
      const auto &graph = kernel_actor->kernel_->func_graph();
      if (graph != nullptr) {
        const auto &kernel_graph = dynamic_cast<KernelGraph *>(graph.get());
        const auto func_graph = kernel_graph->GetFuncGraph();
        if (func_graph != nullptr && FetchActor(func_graph->ToString()) != nullptr) {
          continue;
        }
      }

      no_input_kernel_actors.emplace_back(kernel_actor);
    }
  }
  return no_input_kernel_actors;
}

std::vector<SwitchActorPtr> GraphScheduler::BuildSwitchActor(const GraphCompilerInfo &graph_compiler_info) {
  std::vector<SwitchActorPtr> switch_actors;
  std::unordered_map<AnfNodePtr, AnfNodePtr> front_to_backend_kernel;
  for (const auto &pair : front_node_to_actor_) {
    front_to_backend_kernel[pair.first] = pair.second->kernel_;
  }

  for (const auto &control_node : graph_compiler_info.control_nodes_) {
    if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitch) ||
        AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitchLayer)) {
      auto actor_name = control_node->fullname_with_scope();
      auto switch_actor = std::make_shared<SwitchActor>(actor_name, graph_compiler_info.device_contexts_[0],
                                                        control_node->cast<CNodePtr>());
      switch_actor->Initialize();

      // Fetch all the input nodes of switch actor.
      switch_actor->FetchInputNode(graph_compiler_info.origin_parameters_order_,
                                   graph_compiler_info.control_node_parser_->front_to_backend_parameters_,
                                   front_to_backend_kernel);
      InsertActor(switch_actor.get());
      switch_actors.emplace_back(switch_actor);
    }
  }
  return switch_actors;
}

std::vector<GatherActorPtr> GraphScheduler::BuildGatherActor(const GraphCompilerInfo &graph_compiler_info) {
  std::vector<GatherActorPtr> gather_actors;

  bool is_main_return = true;
  // Each funcgraph has a return node, get the funcgraph from the return node, and create a gather actor.
  std::unordered_map<AnfNodePtr, AnfNodePtr> front_to_backend_kernel;
  for (const auto &pair : front_node_to_actor_) {
    front_to_backend_kernel[pair.first] = pair.second->kernel_;
  }

  for (const auto &control_node : graph_compiler_info.control_nodes_) {
    // Root funcgraph does not need to create a gather actor.
    if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) {
      if (is_main_return) {
        is_main_return = false;
        continue;
      }

      const auto &cnode = control_node->cast<CNodePtr>();
      const auto inputs = cnode->inputs();
      // If the output of funcgraph is a value node, no need to create gather actor.
      if (inputs[kReturnInputPos]->isa<ValueNode>()) {
        continue;
      }

      auto func_graph = control_node->func_graph();
      auto actor_name = func_graph->ToString();
      std::vector<AnfNodePtr> parameters;
      for (const auto &parameter : func_graph->get_inputs()) {
        if (!HasAbstractMonad(parameter)) {
          parameters.emplace_back(parameter);
        }
      }

      const auto &loop_count_actor_name = graph_compiler_info.name_ + "_LoopCountActor";
      const auto &loop_count_actor = FetchActor(loop_count_actor_name);
      MS_EXCEPTION_IF_NULL(loop_count_actor);
      const auto &output_actor_name = graph_compiler_info.name_ + "_" + "OutputActor";
      const auto &output_actor = FetchActor(output_actor_name);
      MS_EXCEPTION_IF_NULL(output_actor);

      auto gather_actor =
        std::make_shared<GatherActor>(actor_name, parameters, loop_count_actor->GetAID(), output_actor->GetAID());
      gather_actor->FetchBackendInputNode(func_graph, graph_compiler_info.origin_parameters_order_,
                                          graph_compiler_info.control_node_parser_->front_to_backend_parameters_,
                                          graph_compiler_info.control_node_parser_->func_graph_to_parameters_,
                                          front_to_backend_kernel);
      InsertActor(gather_actor.get());
      gather_actors.emplace_back(gather_actor);
    }
  }

  return gather_actors;
}

void GraphScheduler::LinkDataArrow(KernelActor *to_actor, const GraphCompilerInfo &graph_compiler_info,
                                   const KernelGraphPtr &graph, KernelWithIndex from_kernel_with_output_idx,
                                   KernelWithIndex to_kernel_with_input_idx, const TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(graph);

  auto from_kernel = from_kernel_with_output_idx.first;
  auto front_node = GetFrontNodeByBackendNode(from_kernel);
  if (IsDeviceQueueDSActor(from_kernel)) {
    // Link the data arrows of device queue data source actor.
    std::string actor_name = graph_compiler_info.name_ + "_DeviceDSActor" + "_" + std::to_string(graph->graph_id());
    const auto &from_actor = dynamic_cast<DeviceQueueDataSourceActor *>(FetchActor(actor_name));
    LinkDataArrowForDeviceDSActor(from_actor, to_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
  } else if (front_node != nullptr && IsGatherActor(front_node, actor_name_to_actor_)) {
    // Link the data arrows of gather actor.
    auto func_graph = GetFuncgraphByBackendNode(from_kernel);
    if (func_graph == nullptr) {
      MS_LOG(EXCEPTION) << "Cannot find funcgraph of node:" << AnfAlgo::GetNodeDebugString(from_kernel);
    }
    auto actor_name = func_graph->ToString();
    const auto &from_actor = dynamic_cast<GatherActor *>(FetchActor(actor_name));
    LinkDataArrowForGatherActor(from_actor, to_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
  } else if (IsHostQueueDSActor(from_kernel, graph, tensor, graph_compiler_info.origin_parameters_order_)) {
    // Link the data arrows of host queue data source actor.
    std::string actor_name = graph_compiler_info.name_ + "_HostDSActor";
    const auto &from_actor = dynamic_cast<HostQueueDataSourceActor *>(FetchActor(actor_name));
    LinkDataArrowForHostDSActor(from_actor, to_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
  } else if (IsKernelActor(from_kernel)) {
    // Link the data arrows of kernel actor.
    const auto &from_actor = dynamic_cast<KernelActor *>(FetchActor(from_kernel->fullname_with_scope()));
    LinkDataArrowForKernelActor(from_actor, to_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
  } else if (IsInternalParameter(from_kernel, graph)) {
    // Link data arrow for internal parameter, convert internal parameter to actor by internal parameter cache to link.
    LinkDataArrowForInternalParameter(from_kernel, graph_compiler_info.origin_parameters_order_, graph, to_actor,
                                      to_kernel_with_input_idx);
  } else if (IsPersistentDeviceTensor(from_kernel)) {
    const auto devcie_tensor_store_key = FetchFrontNodeByBackendNode(from_kernel, graph);
    to_actor->device_tensor_store_keys_.emplace_back(to_kernel_with_input_idx.second, devcie_tensor_store_key.get());
  } else {
    // May exist the from kernel that no need link in the pynative mode.
    MS_LOG(DEBUG) << "Invalid from kernel: " << from_kernel->fullname_with_scope();
  }
}

void GraphScheduler::LinkDataArrowForInternalParameter(const AnfNodePtr &internal_parameter,
                                                       const std::vector<AnfNodePtr> &host_parameters,
                                                       const KernelGraphPtr &graph, KernelActor *to_actor,
                                                       KernelWithIndex to_kernel_with_input_idx) {
  MS_EXCEPTION_IF_NULL(internal_parameter);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(to_actor);

  // Parameter ---> front node ---> actor.
  auto front_node_with_index = graph->GetFrontNodeByInternalParameter(internal_parameter);
  MS_EXCEPTION_IF_NULL(front_node_with_index.first);
  const auto &front_output_with_index =
    AnfAlgo::VisitKernelWithReturnType(front_node_with_index.first, front_node_with_index.second, false);
  auto front_output_node = front_output_with_index.first;
  MS_EXCEPTION_IF_NULL(front_output_node);
  MS_LOG(INFO) << "Link data arrow for internal parameter:" << internal_parameter->fullname_with_scope()
               << ", corresponding front node:" << front_output_node->fullname_with_scope()
               << " with output index:" << front_output_with_index.second;
  if (IsPersistentDeviceTensor(front_output_node)) {
    to_actor->device_tensor_store_keys_.emplace_back(to_kernel_with_input_idx.second, front_output_node.get());
    return;
  }
  if (graph_output_to_actor_.count(front_output_with_index) == 0 && (!IsSwitchActor(front_output_node))) {
    MS_LOG(EXCEPTION) << "Can't find actor by front node:" << front_output_node->fullname_with_scope()
                      << ", internal parameter:" << internal_parameter->fullname_with_scope();
  }
  auto actor_pair = graph_output_to_actor_[front_output_with_index];

  if (IsDeviceQueueDSActor(front_output_node)) {
    auto from_actor = dynamic_cast<DeviceQueueDataSourceActor *>(actor_pair.first);
    auto from_kernel_with_output_idx = KernelWithIndex(from_actor->data_kernel_, actor_pair.second);
    LinkDataArrowForDeviceDSActor(from_actor, to_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
  } else if (IsSwitchActor(front_output_node)) {
    const auto &actor_name = front_output_node->fullname_with_scope();
    const auto &actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    auto switch_actor = dynamic_cast<SwitchActor *>(actor);
    LinkDataArrowForSwitchActor(switch_actor, to_actor, to_kernel_with_input_idx.second);
  } else if (IsKernelActor(front_output_node)) {
    auto from_actor = dynamic_cast<KernelActor *>(actor_pair.first);
    auto from_kernel_with_output_idx = KernelWithIndex(from_actor->kernel_, actor_pair.second);
    LinkDataArrowForKernelActor(from_actor, to_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
  } else if (IsHostQueueDSActor(front_output_node, graph, nullptr, host_parameters)) {
    auto from_actor = dynamic_cast<HostQueueDataSourceActor *>(actor_pair.first);
    auto from_kernel_with_output_idx = KernelWithIndex(from_actor->data_nodes_[actor_pair.second], 0);
    LinkDataArrowForHostDSActor(from_actor, to_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
  } else {
    MS_LOG(EXCEPTION) << "Invalid internal parameter: " << internal_parameter->fullname_with_scope();
  }
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

  if (IsNeedInsertCopyActor(from_actor->device_context_, to_actor->device_context_)) {
    LinkDataArrowForCopyActor(from_actor, to_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
  } else {
    auto to_aid = to_actor->GetAID();
    auto op_arrow = std::make_shared<DataArrow>(from_output_index, to_aid, to_input_index);
    from_actor->output_data_arrows_.emplace_back(op_arrow);
    to_actor->input_datas_num_++;
    to_actor->input_data_arrow_aids_.emplace_back(from_actor->GetAID());

    // Update the reference count of device tensor.
    UpdateRefCount(from_kernel, from_output_index);
  }
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

  // Get the position of from kernel in the data source actor.
  auto position = from_actor->FetchDataNodePosition(from_kernel);

  if (IsNeedInsertCopyActor(from_actor->device_contexts_[position], to_actor->device_context_)) {
    LinkDataArrowForCopyActor(from_actor, to_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
  } else {
    auto to_aid = to_actor->GetAID();
    auto op_arrow = std::make_shared<DataArrow>(position, to_aid, to_input_index);
    from_actor->output_data_arrows_.emplace_back(op_arrow);
    to_actor->input_datas_num_++;
    to_actor->input_data_arrow_aids_.emplace_back(from_actor->GetAID());

    // Update the reference count of device tensor.
    UpdateRefCount(from_actor->data_nodes_[position], from_output_index);
  }
}

void GraphScheduler::LinkDataArrowForKernelActor(KernelActor *from_actor, KernelActor *to_actor,
                                                 KernelWithIndex from_kernel_with_output_idx,
                                                 KernelWithIndex to_kernel_with_input_idx) {
  MS_EXCEPTION_IF_NULL(to_actor);
  if (IsSkippedKernelActor(from_kernel_with_output_idx.first)) {
    auto real_kernel_with_index = AnfAlgo::GetPrevNodeOutput(from_kernel_with_output_idx.first, 0);
    MS_EXCEPTION_IF_NULL(real_kernel_with_index.first);
    LinkControlArrowBySkippedNode(to_actor, from_kernel_with_output_idx.first);

    // Update the from kernel info by the real node info.
    MS_LOG(INFO) << "Link data arrow for inplace node, aggregate node: "
                 << to_kernel_with_input_idx.first->fullname_with_scope()
                 << ", aggregate input index: " << to_kernel_with_input_idx.second
                 << ", skip node: " << from_kernel_with_output_idx.first->fullname_with_scope()
                 << ", real node: " << real_kernel_with_index.first->fullname_with_scope();
    from_kernel_with_output_idx.first = real_kernel_with_index.first;
    from_kernel_with_output_idx.second = real_kernel_with_index.second;
    from_actor = dynamic_cast<KernelActor *>(FetchActor(from_kernel_with_output_idx.first->fullname_with_scope()));
  }

  MS_EXCEPTION_IF_NULL(from_actor);
  auto from_kernel = from_kernel_with_output_idx.first;
  MS_EXCEPTION_IF_NULL(from_kernel);
  auto from_output_index = from_kernel_with_output_idx.second;
  auto to_input_index = to_kernel_with_input_idx.second;

  if (IsNeedInsertCopyActor(from_actor->device_context_, to_actor->device_context_)) {
    LinkDataArrowForCopyActor(from_actor, to_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
  } else {
    auto to_aid = to_actor->GetAID();
    auto op_arrow = std::make_shared<DataArrow>(from_output_index, to_aid, to_input_index);
    from_actor->output_data_arrows_.emplace_back(op_arrow);
    to_actor->input_datas_num_++;
    to_actor->input_data_arrow_aids_.emplace_back(from_actor->GetAID());

    // Update the reference count of device tensor.
    UpdateRefCount(from_kernel, from_output_index);
  }
}

void GraphScheduler::LinkDataArrowForCopyActor(OpActor<DeviceTensor> *from_actor, KernelActor *to_actor,
                                               KernelWithIndex from_kernel_with_output_idx,
                                               KernelWithIndex to_kernel_with_input_idx) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  auto from_kernel = from_kernel_with_output_idx.first;
  MS_EXCEPTION_IF_NULL(from_kernel);
  auto to_devcie_context = to_actor->device_context_;
  MS_EXCEPTION_IF_NULL(to_devcie_context);
  auto from_output_index = from_kernel_with_output_idx.second;
  auto to_input_index = to_kernel_with_input_idx.second;

  std::string name = "copy_from:" + from_actor->GetAID().Name() + "_node:" + from_kernel->fullname_with_scope() +
                     "_output_index:" + std::to_string(from_output_index);
  CopyActor *copy_actor = dynamic_cast<CopyActor *>(FetchActor(name));
  // Link between from actor and copy actor.
  if (copy_actor == nullptr) {
    // Create the copy actor.
    auto copy_actor_shared_ptr = std::make_shared<CopyActor>(name, memory_manager_aid_);
    copy_actors_.emplace_back(copy_actor_shared_ptr);
    copy_actor = copy_actor_shared_ptr.get();
    MS_EXCEPTION_IF_NULL(copy_actor);
    InsertActor(copy_actor);

    // Link.
    const DeviceContext *from_devcie_context = nullptr;
    auto from_device_tensor = AnfAlgo::GetMutableOutputAddr(from_kernel, from_output_index, false);
    auto op_arrow_to_copy = std::make_shared<DataArrow>(from_output_index, copy_actor->GetAID(), 0);
    if (IsDeviceQueueDSActor(from_kernel)) {
      auto real_from_actor = dynamic_cast<DeviceQueueDataSourceActor *>(from_actor);
      from_devcie_context = real_from_actor->device_context_;
      real_from_actor->output_data_arrows_.emplace_back(op_arrow_to_copy);
    } else if (IsKernelActor(from_kernel)) {
      auto real_from_actor = dynamic_cast<KernelActor *>(from_actor);
      from_devcie_context = real_from_actor->device_context_;
      real_from_actor->output_data_arrows_.emplace_back(op_arrow_to_copy);
    } else if (IsHostQueueDSActor(from_kernel)) {
      auto real_from_actor = dynamic_cast<HostQueueDataSourceActor *>(from_actor);
      auto position = real_from_actor->FetchDataNodePosition(from_kernel);
      from_devcie_context = real_from_actor->device_contexts_[position];
      op_arrow_to_copy->from_output_index_ = position;
      real_from_actor->output_data_arrows_.emplace_back(op_arrow_to_copy);
      from_device_tensor =
        AnfAlgo::GetMutableOutputAddr(real_from_actor->data_nodes_[position], from_output_index, false);
    }
    copy_actor->input_datas_num_++;

    // Set the member of the copy actor.
    MS_EXCEPTION_IF_NULL(from_device_tensor);
    copy_actor->output_ = to_devcie_context->CreateDeviceAddress(
      nullptr, from_device_tensor->GetSize(), from_device_tensor->format(), from_device_tensor->type_id());
    MS_EXCEPTION_IF_NULL(from_devcie_context);
    copy_actor->input_device_context_ = from_devcie_context;
    copy_actor->output_device_context_ = to_devcie_context;

    // Update the reference count of device tensor.
    UpdateRefCount(from_device_tensor.get());
  }

  // If the copy actor already exists, only need link between copy actor and to actor.
  auto op_arrow_from_copy = std::make_shared<DataArrow>(0, to_actor->GetAID(), to_input_index);
  copy_actor->output_data_arrows_.emplace_back(op_arrow_from_copy);
  to_actor->input_datas_num_++;
  UpdateRefCount(copy_actor->output_.get());
}

void GraphScheduler::LinkControlArrowByAutoMonad(KernelActor *to_actor, const AnfNodePtr &from_node) {
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(from_node);
  if (!from_node->isa<CNode>()) {
    return;
  }
  // Find the real input node, include the monad node and make tuple node.
  const std::vector<PrimitivePtr> return_types = {prim::kPrimDepend, prim::kPrimUpdateState, prim::kPrimLoad,
                                                  prim::kPrimMakeTuple};
  const auto &input_kernel_with_output_idx = AnfAlgo::VisitKernelWithReturnType(from_node, 0, false, return_types);
  MS_EXCEPTION_IF_NULL(input_kernel_with_output_idx.first);
  if (!input_kernel_with_output_idx.first->isa<CNode>()) {
    return;
  }
  const auto &input_cnode = input_kernel_with_output_idx.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(input_cnode);
  // Make tuple node needs to be expanded.
  if (AnfAlgo::CheckPrimitiveType(input_cnode, prim::kPrimMakeTuple)) {
    for (size_t i = 1; i < input_cnode->inputs().size(); ++i) {
      LinkControlArrowByAutoMonad(to_actor, input_cnode->input(i));
    }
    return;
  }

  // Get the real depend input by monad node which needs to link the control arrow.
  std::vector<AnfNodePtr> real_depend_inputs;
  if (AnfAlgo::CheckPrimitiveType(input_cnode, prim::kPrimDepend)) {
    real_depend_inputs.push_back(input_cnode->input(kDependAttachNodeIndex));
  } else if (AnfAlgo::CheckPrimitiveType(input_cnode, prim::kPrimUpdateState)) {
    for (size_t i = kUpdateStateRealInput; i < input_cnode->inputs().size(); ++i) {
      real_depend_inputs.push_back(input_cnode->input(i));
    }
  } else if (AnfAlgo::CheckPrimitiveType(input_cnode, prim::kPrimLoad)) {
    real_depend_inputs.push_back(input_cnode->input(kLoadStateInput));
  }

  const std::unordered_set<PrimitivePtr, PrimitiveHasher, PrimitiveEqual> recursion_prims = {
    prim::kPrimDepend, prim::kPrimUpdateState, prim::kPrimLoad, prim::kPrimMakeTuple};
  for (const auto &real_depend_input : real_depend_inputs) {
    auto real_depend_input_with_idx = AnfAlgo::VisitKernelWithReturnType(real_depend_input, 0, false, return_types);
    auto real_depend_kernel = real_depend_input_with_idx.first;
    // The monad node and make tuple node need recursion.
    if (AnfAlgo::IsOneOfPrimitiveCNode(real_depend_kernel, recursion_prims)) {
      LinkControlArrowByAutoMonad(to_actor, real_depend_kernel);
      continue;
    }

    if (!IsKernelActor(real_depend_kernel)) {
      continue;
    }
    // Link the control arrow between the kernel actors.
    const auto &from_actor = dynamic_cast<KernelActor *>(FetchActor(real_depend_kernel->fullname_with_scope()));
    MS_LOG(INFO) << "Link control arrow by auto monad, from actor:  " << real_depend_kernel->fullname_with_scope()
                 << ", to actor: " << to_actor->GetAID().Name();
    MS_EXCEPTION_IF_NULL(from_actor);
    from_actor->output_control_arrows_.emplace_back(to_actor->GetAID());
    to_actor->input_controls_num_++;
  }
}

void GraphScheduler::LinkControlArrowBySkippedNode(KernelActor *to_actor, const AnfNodePtr &skipped_node) {
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(skipped_node);
  auto to_aid = to_actor->GetAID();

  // Link the control arrow from all the inputs of skipped node to the user of skipped node.
  auto input_num = AnfAlgo::GetInputTensorNum(skipped_node);
  for (size_t i = 0; i < input_num; ++i) {
    auto kernel_with_index = AnfAlgo::GetPrevNodeOutput(skipped_node, i, false);
    MS_EXCEPTION_IF_NULL(kernel_with_index.first);
    auto from_actor = dynamic_cast<KernelActor *>(FetchActor(kernel_with_index.first->fullname_with_scope()));
    MS_EXCEPTION_IF_NULL(from_actor);
    MS_LOG(INFO) << "Link control arrow by skipped node: " << skipped_node->fullname_with_scope()
                 << ", from actor: " << from_actor->GetAID().Name() << ", to actor: " << to_actor->GetAID().Name();
    from_actor->output_control_arrows_.emplace_back(to_aid);
    to_actor->input_controls_num_++;
  }
}

void GraphScheduler::LinkControlArrowBySendRecvNodes(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  for (auto &from_iter : graph->allreduce_from_send_recv_pairs()) {
    auto to_allreduce_node = from_iter.first;
    MS_LOG(INFO) << "Link control arrow for to_allreduce_node: " << to_allreduce_node->fullname_with_scope();
    auto from_send_node = from_iter.second.first;
    auto from_recv_node = from_iter.second.second;
    auto to_allreduce_actor = dynamic_cast<KernelActor *>(FetchActor(to_allreduce_node->fullname_with_scope()));
    auto from_send_actor = dynamic_cast<KernelActor *>(FetchActor(from_send_node->fullname_with_scope()));
    auto from_recv_actor = dynamic_cast<KernelActor *>(FetchActor(from_recv_node->fullname_with_scope()));

    // inputs of to_allreduce_actor  --> from_send_actor
    for (auto &input_aid : to_allreduce_actor->input_data_arrow_aids_) {
      auto input_actor = dynamic_cast<KernelActor *>(FetchActor(input_aid.Name()));
      input_actor->output_control_arrows_.emplace_back(from_send_actor->GetAID());
      from_send_actor->input_controls_num_++;
    }

    // from_send_actor --> from_recv_actor
    from_send_actor->output_control_arrows_.emplace_back(from_recv_actor->GetAID());
    from_recv_actor->input_controls_num_++;

    // from_recv_actor --> to_allreduce_actor
    from_recv_actor->output_control_arrows_.emplace_back(to_allreduce_actor->GetAID());
    to_allreduce_actor->input_controls_num_++;
  }

  for (auto &to_iter : graph->allreduce_to_send_recv_pairs()) {
    auto from_allreduce_node = to_iter.first;
    MS_LOG(INFO) << "Link control arrow for from_allreduce_node: " << from_allreduce_node->fullname_with_scope();
    auto to_send_node = to_iter.second.first;
    auto to_recv_node = to_iter.second.second;
    auto from_allreduce_actor = dynamic_cast<KernelActor *>(FetchActor(from_allreduce_node->fullname_with_scope()));
    auto to_send_actor = dynamic_cast<KernelActor *>(FetchActor(to_send_node->fullname_with_scope()));
    auto to_recv_actor = dynamic_cast<KernelActor *>(FetchActor(to_recv_node->fullname_with_scope()));

    // from_allreduce_actor  --> to_send_actor
    from_allreduce_actor->output_control_arrows_.emplace_back(to_send_actor->GetAID());
    to_send_actor->input_controls_num_++;

    // to_send_actor --> to_recv_actor
    to_send_actor->output_control_arrows_.emplace_back(to_recv_actor->GetAID());
    to_recv_actor->input_controls_num_++;

    // to_recv_actor --> outputs of from_allreduce_actor
    for (auto &output_data_arrow : from_allreduce_actor->output_data_arrows_) {
      auto output_actor = dynamic_cast<KernelActor *>(FetchActor(output_data_arrow->to_op_id_.Name()));
      to_recv_actor->output_control_arrows_.emplace_back(output_actor->GetAID());
      output_actor->input_controls_num_++;
    }

    // In the scene of allreduce op and computing op parallel multi stream, the input memory of allreduce can be reused
    // only when the recv node runs finished, which is expressed by the reference count increased.
    for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(from_allreduce_node); ++i) {
      auto device_tensor = AnfAlgo::GetPrevNodeMutableOutputAddr(from_allreduce_node, i, false);
      MS_EXCEPTION_IF_NULL(device_tensor);
      UpdateRefCount(device_tensor.get());
      to_recv_actor->external_reference_tensors_.emplace_back(device_tensor.get());
    }
  }
}

void GraphScheduler::LinkControlArrowByCommunicationNode(const KernelGraphPtr &graph) {
  std::vector<CNodePtr> communication_nodes;
  auto execution_order = graph->execution_order();
  for (auto &kernel : execution_order) {
    if (AnfAlgo::IsCommunicationOp(kernel)) {
      communication_nodes.emplace_back(kernel);
    }
  }

  for (size_t i = 1; i < communication_nodes.size(); ++i) {
    auto from_actor = dynamic_cast<KernelActor *>(FetchActor(communication_nodes[i - 1]->fullname_with_scope()));
    auto to_actor = dynamic_cast<KernelActor *>(FetchActor(communication_nodes[i]->fullname_with_scope()));
    MS_EXCEPTION_IF_NULL(from_actor);
    MS_EXCEPTION_IF_NULL(to_actor);
    from_actor->output_control_arrows_.emplace_back(to_actor->GetAID());
    to_actor->input_controls_num_++;
  }
}

void GraphScheduler::LinkControlArrowForLoopCountActor(LoopCountActor *loop_count_actor, const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  // There is no loop count actor in step mode.
  if (loop_count_actor == nullptr) {
    return;
  }

  // Collect the actors which have no output.
  std::vector<MemoryAwareActor *> no_output_actors;
  for (auto &kernel_actor : actor_set->kernel_actors_) {
    if ((kernel_actor->output_data_arrows_.size() == 0) && (kernel_actor->output_control_arrows_.size() == 0)) {
      MS_EXCEPTION_IF_NULL(kernel_actor->kernel_);
      MS_LOG(INFO) << kernel_actor->kernel_->fullname_with_scope() << " is not real used by other nodes.";
      no_output_actors.emplace_back(kernel_actor.get());
    }
  }
  for (auto &data_actor : actor_set->data_source_actors_) {
    if ((data_actor->output_data_arrows_.size() == 0) && (data_actor->output_control_arrows_.size() == 0)) {
      no_output_actors.emplace_back(data_actor.get());
    }
  }
  for (auto &copy_actor : copy_actors_) {
    if ((copy_actor->output_data_arrows_.size() == 0) && (copy_actor->output_control_arrows_.size() == 0)) {
      no_output_actors.emplace_back(copy_actor.get());
    }
  }
  // No output actor --> loop count actor.
  for (auto &no_output_actor : no_output_actors) {
    no_output_actor->output_control_arrows_.emplace_back(loop_count_actor->GetAID());
    loop_count_actor->branch_id_to_input_controls_num_[kMainBranchID]++;
  }

  // Loop count actor --> data source actor.
  for (auto &data_source_actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(data_source_actor);
    loop_count_actor->data_source_aids_.emplace_back(data_source_actor->GetAID());
  }

  // Loop count actor --> no input kernel actor.
  for (auto &no_input_kernel_actor : actor_set->no_input_kernel_actors_) {
    MS_EXCEPTION_IF_NULL(no_input_kernel_actor);
    loop_count_actor->no_input_kernel_aids_.emplace_back(no_input_kernel_actor->GetAID());
    no_input_kernel_actor->input_controls_num_++;
  }

  // Loop count actor --> output actor.
  MS_EXCEPTION_IF_NULL(actor_set->output_actor_);
  loop_count_actor->output_aid_ = actor_set->output_actor_->GetAID();
}

void GraphScheduler::LinkOutputResultArrowForOutputActor(OutputActor *to_actor,
                                                         const GraphCompilerInfo &graph_compiler_info) {
  MS_EXCEPTION_IF_NULL(to_actor);

  size_t number = 0;
  for (const auto &graph : graph_compiler_info.graphs_) {
    MS_EXCEPTION_IF_NULL(graph);
    ++number;
    auto outputs = AnfAlgo::GetAllOutputWithIndex(graph->output());
    std::set<std::pair<int, std::vector<size_t>>> unique_output_positions;
    std::set<KernelWithIndex> unique_outputs;
    for (const auto &output : outputs) {
      unique_outputs.insert(output);
    }
    for (const auto &output_with_index : unique_outputs) {
      MS_EXCEPTION_IF_NULL(output_with_index.first);
      auto origin_output_with_index = FetchFrontNodeWithIndexByGraphOutput(output_with_index, graph);
      const auto &iter = graph_compiler_info.origin_outputs_order_.find(origin_output_with_index);
      if (iter == graph_compiler_info.origin_outputs_order_.end()) {
        continue;
      }

      // Skip duplicate position.
      if (unique_output_positions.count(iter->second) > 0) {
        continue;
      }
      unique_output_positions.insert(iter->second);
      for (auto &output_position : iter->second.second) {
        to_actor->device_contexts_[output_position] = graph_compiler_info.device_contexts_[number - 1];
        // The device tensor of graph out need be taken over by host tensor, so set the max reference count.
        UpdateRefCount(output_with_index.first, output_with_index.second, true);

        // The graph output is from device tensor store.
        if (IsPersistentDeviceTensor(output_with_index.first)) {
          to_actor->device_tensor_store_keys_[iter->second.first].emplace_back(output_position,
                                                                               output_with_index.first);
          continue;
        }

        // The graph output is from kernel actor.
        if (IsKernelActor(output_with_index.first)) {
          const auto &from_actor =
            dynamic_cast<KernelActor *>(FetchActor(output_with_index.first->fullname_with_scope()));
          MS_EXCEPTION_IF_NULL(from_actor);
          auto op_arrow = std::make_shared<DataArrow>(output_with_index.second, to_actor->GetAID(), output_position);
          from_actor->output_result_arrows_.emplace_back(op_arrow);
          continue;
        }

        // The graph output is from data source actor.
        std::string actor_name;
        DataSourceActor *from_actor = nullptr;
        size_t from_actor_output_index = 0;
        if (IsHostQueueDSActor(output_with_index.first, graph, nullptr, graph_compiler_info.origin_parameters_order_)) {
          actor_name = graph_compiler_info.name_ + "_HostDSActor";
          const auto &host_queue_ds_actor = dynamic_cast<HostQueueDataSourceActor *>(FetchActor(actor_name));
          from_actor_output_index = host_queue_ds_actor->FetchDataNodePosition(output_with_index.first);
          UpdateRefCount(host_queue_ds_actor->data_nodes_[from_actor_output_index], output_with_index.second, true);
          from_actor = static_cast<DataSourceActor *>(host_queue_ds_actor);
        } else if (IsDeviceQueueDSActor(output_with_index.first)) {
          actor_name = graph_compiler_info.name_ + "_DeviceDSActor" + "_" + std::to_string(graph->graph_id());
          from_actor = dynamic_cast<DataSourceActor *>(FetchActor(actor_name));
          from_actor_output_index = output_with_index.second;
        }

        // When the input is a parameter node, it should be connected by gather actor.
        if (from_actor == nullptr) {
          if (output_with_index.first->isa<CNode>()) {
            MS_LOG(EXCEPTION) << "Cannot find kernel actor for kernel:"
                              << output_with_index.first->fullname_with_scope();
          } else {
            continue;
          }
        }
        MS_EXCEPTION_IF_NULL(from_actor);
        auto op_arrow = std::make_shared<DataArrow>(from_actor_output_index, to_actor->GetAID(), output_position);
        from_actor->output_result_arrows_.emplace_back(op_arrow);
      }
    }
  }
}

void GraphScheduler::LinkOutputResultArrowForSwitchActor(const GraphCompilerInfo &graph_compiler_info,
                                                         const ActorSet *actor_set) {
  const auto &to_actor = actor_set->output_actor_;
  const auto &loop_count_actor = actor_set->loop_count_actor_;
  const auto &switch_actors = actor_set->switch_actors_;
  if (to_actor == nullptr || loop_count_actor == nullptr) {
    return;
  }

  for (const auto &from_actor : switch_actors) {
    MS_EXCEPTION_IF_NULL(from_actor);
    auto origin_output_with_index = KernelWithIndex(from_actor->node_, 0);
    const auto &iter = graph_compiler_info.origin_outputs_order_.find(origin_output_with_index);
    if (iter == graph_compiler_info.origin_outputs_order_.end()) {
      continue;
    }

    // If the switch actor is in the output list, the output of switch actor should be sent to the output actor.
    // And need to link a control arrow to the loop count actor.
    for (const auto pos : iter->second.second) {
      to_actor->device_contexts_[pos] = from_actor->device_context_;
    }

    for (size_t i = 0; i < from_actor->branch_inputs_pos_.size(); ++i) {
      const auto &input_pos = from_actor->branch_inputs_pos_[i];
      if (input_pos.empty()) {
        MS_LOG(EXCEPTION) << "Invalid input num in switch actor:" << from_actor->GetAID();
      }

      for (const auto pos : iter->second.second) {
        auto op_arrow = std::make_shared<DataArrow>(input_pos[0], to_actor->GetAID(), pos);
        from_actor->output_branch_result_arrows_[i].emplace_back(op_arrow);
      }

      from_actor->output_branch_control_arrows_[i].emplace_back(loop_count_actor->GetAID());
    }
    loop_count_actor->branch_id_to_input_controls_num_[kMainBranchID]++;
  }
}

void GraphScheduler::LinkDeviceTensorStoreForAutoMonadActor(const std::vector<KernelActor *> &auto_monad_actors) {
  const size_t kNeedUpdateDeviceTensorStoreNum = 2;
  for (auto &kernel_actor : auto_monad_actors) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    for (auto &device_tensor_store_key : kernel_actor->device_tensor_store_keys_) {
      auto device_tensors = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second);
      if (device_tensors.size() < kNeedUpdateDeviceTensorStoreNum) {
        continue;
      }

      // Create the copy actor.
      std::string name = "copy_from:" + kernel_actor->GetAID().Name() +
                         "_device_tensor_store:" + device_tensor_store_key.second->fullname_with_scope();
      auto copy_actor = std::make_shared<CopyActor>(name, memory_manager_aid_);
      MS_EXCEPTION_IF_NULL(copy_actor);
      copy_actors_.emplace_back(copy_actor);
      InsertActor(copy_actor.get());

      // Set the member of the copy actor.
      copy_actor->device_tensor_store_key_ = std::pair<size_t, AnfNode *>(0, device_tensor_store_key.second);
      auto input_device_context = kernel_actor->device_context_;
      copy_actor->input_device_context_ = input_device_context;
      auto another_device_tensor = (device_tensors[0]->DeviceType() == input_device_context->GetDeviceAddressType())
                                     ? device_tensors[1]
                                     : device_tensors[0];
      MS_EXCEPTION_IF_NULL(another_device_tensor);
      auto another_device_type = another_device_tensor->DeviceType();
      const auto &another_device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
        {device::kDeviceTypeToName.at(another_device_type), input_device_context->device_context_key().device_id_});
      MS_EXCEPTION_IF_NULL(another_device_context);
      copy_actor->output_device_context_ = another_device_context;

      // LInk from copy actor to kernel actor users.
      if (kernel_actor->output_control_arrows_.size() == 0) {
        MS_LOG(INFO) << "The kernel actor has no control arrow:" << kernel_actor->GetAID().Name();
      }
      for (auto &output_contorl : kernel_actor->output_control_arrows_) {
        copy_actor->output_control_arrows_.emplace_back(output_contorl);
        auto to_actor = FetchActor(output_contorl.Name());
        MS_EXCEPTION_IF_NULL(to_actor);
        if (output_contorl.Name().find("_LoopCountActor") != string::npos) {
          auto real_to_actor = dynamic_cast<LoopCountActor *>(to_actor);
          real_to_actor->branch_id_to_input_controls_num_[kMainBranchID]++;
        } else {
          auto real_to_actor = dynamic_cast<KernelActor *>(to_actor);
          real_to_actor->input_controls_num_++;
        }
      }
      // Link from kernel actor to copy actor.
      kernel_actor->output_control_arrows_.emplace_back(copy_actor->GetAID());
      copy_actor->input_controls_num_++;
    }
  }
}

void GraphScheduler::LinkArrowByControlNode(const GraphCompilerInfo &graph_compiler_info, ActorSet *actor_set) {
  for (const auto &node : graph_compiler_info.control_nodes_) {
    CNodePtr cnode = node->cast<CNodePtr>();
    auto inputs = cnode->inputs();
    // Link data arrow for switch node.
    if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimSwitch) ||
        AnfAlgo::CheckPrimitiveType(node, prim::kPrimSwitchLayer)) {
      auto actor = actor_name_to_actor_[node->fullname_with_scope()];
      LinkDataArrowForSwitchActor(graph_compiler_info, dynamic_cast<SwitchActor *>(actor));
    } else if (inputs[0]->isa<ValueNode>() && IsValueNode<FuncGraph>(inputs[0])) {
      // Link the data arrow for the input of the call node.
      auto func_graph = GetValueNode<FuncGraphPtr>(inputs[0]);
      auto actor = actor_name_to_actor_[func_graph->ToString()];
      for (size_t i = kCallInputStartPos; i < inputs.size(); ++i) {
        LinkDataArrowByControlNode(graph_compiler_info, inputs[i], actor, i - kCallInputStartPos);
      }
    } else if (inputs[0]->isa<CNode>()) {
      // Link switch inputs which is on the call node.
      if ((!AnfAlgo::CheckPrimitiveType(inputs[0], prim::kPrimSwitch)) &&
          (!AnfAlgo::CheckPrimitiveType(inputs[0], prim::kPrimSwitchLayer))) {
        MS_LOG(EXCEPTION) << "First input node of call node is not switch, node:"
                          << AnfAlgo::GetNodeDebugString(inputs[0]);
      }

      auto switch_op_actor = FetchActor(inputs[0]->fullname_with_scope());
      if (switch_op_actor == nullptr) {
        MS_LOG(EXCEPTION) << "Cannot find actor of switch node:" << AnfAlgo::GetNodeDebugString(inputs[0]);
      }
      auto switch_actor = dynamic_cast<SwitchActor *>(switch_op_actor);
      for (size_t i = kCallInputStartPos; i < inputs.size(); ++i) {
        switch_actor->AddCommonInput(inputs[i]);
        auto pos = switch_actor->FetchDataNodePosition(inputs[i]);
        LinkDataArrowByControlNode(graph_compiler_info, inputs[i], switch_actor, pos - kCallInputStartPos);
      }
    }
  }

  LinkBranchArrowForGatherActor(graph_compiler_info, actor_set);

  LinkControlArrowForGatherActor(&(actor_set->gather_actors_), actor_set->loop_count_actor_.get(),
                                 graph_compiler_info.graphs_);

  LinkOutputResultArrowForGatherActor(graph_compiler_info, actor_set);

  LinkOutputResultArrowForSwitchActor(graph_compiler_info, actor_set);
}

void GraphScheduler::LinkDataArrowForGatherActor(GatherActor *from_actor, KernelActor *to_actor,
                                                 KernelWithIndex from_kernel_with_output_idx,
                                                 KernelWithIndex to_kernel_with_input_idx) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);

  auto from_kernel = from_kernel_with_output_idx.first;
  MS_EXCEPTION_IF_NULL(from_kernel);
  auto to_input_index = to_kernel_with_input_idx.second;

  auto front_node = GetFrontNodeByBackendNode(from_kernel);
  if (front_node == nullptr) {
    MS_LOG(EXCEPTION) << "Cannot find front node of node:" << AnfAlgo::GetNodeDebugString(from_kernel);
  }

  auto position = from_actor->FetchDataNodePosition(front_node);

  auto to_aid = to_actor->GetAID();
  auto op_arrow = std::make_shared<DataArrow>(position, to_aid, to_input_index);
  from_actor->output_data_arrows_.emplace_back(op_arrow);
  to_actor->input_datas_num_++;
}

void GraphScheduler::LinkDataArrowByCallInput(const GraphCompilerInfo &graph_compiler_info, const AnfNodePtr &call_node,
                                              OpActor<DeviceTensor> *to_actor, const size_t to_index) {
  // Fetch all the funcgraph that call node would call.
  const auto cnode = call_node->cast<CNodePtr>();
  std::vector<FuncGraphPtr> func_graphs = FetchFuncGraphbyCallNode(cnode);

  // Collect the output of each funcgraph.
  for (const auto &func_graph : func_graphs) {
    // The output of funcgraph can only have one.
    auto outputs = AnfAlgo::GetAllOutputWithIndex(func_graph->output());
    if (outputs.size() != 1) {
      MS_LOG(EXCEPTION) << "Output of func graph is more than one, func graph:" << func_graph->ToString();
    }

    auto output_with_index = outputs[0];
    if (IsKernelActor(output_with_index.first)) {
      // Input is a kernel actor.
      const auto &iter = front_node_to_actor_.find(output_with_index.first);
      if (iter == front_node_to_actor_.end()) {
        MS_LOG(EXCEPTION) << "Cannot find kernel actor of front node:"
                          << AnfAlgo::GetNodeDebugString(output_with_index.first);
      }
      auto from_actor = iter->second;
      auto op_arrow = std::make_shared<DataArrow>(output_with_index.second, to_actor->GetAID(), to_index);
      from_actor->output_data_arrows_.emplace_back(op_arrow);
      auto device_tensor = AnfAlgo::GetMutableOutputAddr(from_actor->kernel_, output_with_index.second, false);
      UpdateRefCount(device_tensor.get(), true);
    } else if (output_with_index.first->isa<Parameter>()) {
      // Input is a parameter from gather actor.
      const auto &actor_name = func_graph->ToString();
      auto actor = FetchActor(actor_name);
      MS_EXCEPTION_IF_NULL(actor);
      auto gather_actor = dynamic_cast<GatherActor *>(actor);
      MS_EXCEPTION_IF_NULL(gather_actor);

      const auto &iter =
        find(gather_actor->data_nodes_.begin(), gather_actor->data_nodes_.end(), output_with_index.first);
      if (iter == gather_actor->data_nodes_.end()) {
        MS_LOG(EXCEPTION) << "Cannot find parameter:" << AnfAlgo::GetNodeDebugString(output_with_index.first)
                          << " in funcgraph";
      }
      const size_t pos = iter - gather_actor->data_nodes_.begin();
      auto op_arrow = std::make_shared<DataArrow>(pos, to_actor->GetAID(), to_index);
      gather_actor->output_data_arrows_.emplace_back(op_arrow);
    } else if (output_with_index.first->isa<ValueNode>()) {
      // If the output is a value node, then the value node needs to be sent by the switch actor.
      const auto &call_inputs = cnode->inputs();
      if (AnfAlgo::CheckPrimitiveType(call_inputs[0], prim::kPrimSwitch)) {
        const auto &actor_name = call_inputs[0]->fullname_with_scope();
        const auto &actor = FetchActor(actor_name);
        MS_EXCEPTION_IF_NULL(actor);
        auto switch_actor = dynamic_cast<SwitchActor *>(actor);
        MS_EXCEPTION_IF_NULL(switch_actor);

        // Add output for each branch of switch.
        for (size_t i = 0; i < switch_actor->branch_inputs_pos_.size(); ++i) {
          if (switch_actor->branch_inputs_pos_[i].empty()) {
            MS_LOG(EXCEPTION) << "No input for switch actor:" << actor_name << " branch:" << i;
          }

          const auto from_index = switch_actor->branch_inputs_pos_[i][0];
          auto op_arrow = std::make_shared<DataArrow>(from_index, to_actor->GetAID(), to_index);
          switch_actor->output_branch_arrows_[i].emplace_back(op_arrow);
        }
      } else {
        MS_LOG(EXCEPTION) << "Invalid input for call node:" << AnfAlgo::GetNodeDebugString(call_node);
      }
    } else {
      MS_LOG(EXCEPTION) << "Output of func graph is not a parameter or kernel, func graph:" << func_graph->ToString()
                        << " output node:" << AnfAlgo::GetNodeDebugString(output_with_index.first);
    }
  }
}

void GraphScheduler::LinkDataArrowForSwitchActor(SwitchActor *from_actor, KernelActor *to_actor,
                                                 const size_t to_index) {
  MS_EXCEPTION_IF_NULL(from_actor);

  for (size_t i = 0; i < from_actor->output_branch_arrows_.size(); ++i) {
    if (from_actor->branch_inputs_pos_[i].empty()) {
      MS_LOG(EXCEPTION) << "No input for switch actor:" << from_actor->GetAID() << " branch:" << i;
    }
    const auto from_index = from_actor->branch_inputs_pos_[i][0];
    auto op_arrow = std::make_shared<DataArrow>(from_index, to_actor->GetAID(), to_index);
    from_actor->output_branch_arrows_[i].emplace_back(op_arrow);
  }
  to_actor->input_datas_num_++;
}

void GraphScheduler::LinkDataArrowByControlNode(const GraphCompilerInfo &graph_compiler_info,
                                                const AnfNodePtr &input_node, OpActor<DeviceTensor> *to_actor,
                                                const size_t to_index) {
  const auto &parameters = graph_compiler_info.origin_parameters_order_;
  const auto &front_to_backend_parameter = graph_compiler_info.control_node_parser_->front_to_backend_parameters_;

  if (IsCallNode(input_node)) {
    // The actor input is a call node.
    LinkDataArrowByCallInput(graph_compiler_info, input_node, to_actor, to_index);
  } else if (IsGatherActor(input_node, actor_name_to_actor_)) {
    // The actor input is a parameter in gather actor.
    auto from_actor = dynamic_cast<GatherActor *>(actor_name_to_actor_[input_node->func_graph()->ToString()]);
    auto position = from_actor->FetchDataNodePosition(input_node);
    auto op_arrow = std::make_shared<DataArrow>(position, to_actor->GetAID(), to_index);
    from_actor->output_data_arrows_.emplace_back(op_arrow);
  } else if (IsKernelActor(input_node)) {
    // The actor input is a cnode.
    auto input_witch_index = AnfAlgo::VisitKernelWithReturnType(input_node, 0);
    if (front_node_to_actor_.find(input_witch_index.first) == front_node_to_actor_.end()) {
      MS_LOG(EXCEPTION) << "Cannot find switch actor input_node:" << AnfAlgo::GetNodeDebugString(input_node);
    }

    auto op_arrow = std::make_shared<DataArrow>(input_witch_index.second, to_actor->GetAID(), to_index);
    auto from_actor = front_node_to_actor_[input_witch_index.first];
    from_actor->output_data_arrows_.emplace_back(op_arrow);
    auto device_tensor = AnfAlgo::GetMutableOutputAddr(from_actor->kernel_, input_witch_index.second, false);
    UpdateRefCount(device_tensor.get(), true);
  } else if (find(parameters.begin(), parameters.end(), input_node) != parameters.end()) {
    // The actor input is a parameter in host data source actor.
    std::string actor_name = graph_compiler_info.name_ + "_HostDSActor";

    auto actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    auto from_actor = dynamic_cast<HostQueueDataSourceActor *>(actor);
    MS_EXCEPTION_IF_NULL(from_actor);

    auto backend_iter = front_to_backend_parameter.find(input_node);
    if (backend_iter == front_to_backend_parameter.end()) {
      MS_LOG(EXCEPTION) << "Cannot find backend node for front node:" << AnfAlgo::GetNodeDebugString(input_node);
    }

    const auto &backend_node = backend_iter->second.first;
    auto iter = from_actor->data_node_position_map_.find(input_node);
    if (iter == from_actor->data_node_position_map_.end()) {
      MS_LOG(EXCEPTION) << "Cannot find data node in data source actor, node:"
                        << AnfAlgo::GetNodeDebugString(backend_node);
    }

    auto op_arrow = std::make_shared<DataArrow>(iter->second, to_actor->GetAID(), to_index);
    from_actor->output_data_arrows_.emplace_back(op_arrow);
    auto device_tensor = AnfAlgo::GetMutableOutputAddr(from_actor->data_nodes_[iter->second], 0, false);
    UpdateRefCount(device_tensor.get(), true);
  } else {
    MS_LOG(EXCEPTION) << "Cannot find actor of switch input_node:" << AnfAlgo::GetNodeDebugString(input_node);
  }
}

void GraphScheduler::LinkDataArrowForSwitchActor(const GraphCompilerInfo &graph_compiler_info, SwitchActor *actor) {
  // Link switch input.
  const auto &inputs = actor->input_nodes_;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto input = inputs[i];
    if (input->isa<ValueNode>()) {
      continue;
    }
    LinkDataArrowByControlNode(graph_compiler_info, input, actor, i);
  }

  // Link switch output.
  for (size_t i = 0; i < actor->branch_func_graph_.size(); ++i) {
    auto func_graph = actor->branch_func_graph_[i];
    if (func_graph == nullptr || func_graph->output()->isa<ValueNode>()) {
      continue;
    }

    auto gather_name = func_graph->ToString();
    if (actor_name_to_actor_.find(gather_name) == actor_name_to_actor_.end()) {
      MS_LOG(EXCEPTION) << "Cannot find gather actor for funcgraph:" << gather_name
                        << ",switch input size:" << actor->input_nodes_.size();
    }
    auto to_actor = dynamic_cast<GatherActor *>(actor_name_to_actor_[gather_name]);
    for (size_t j = 0; j < actor->branch_inputs_pos_[i].size(); ++j) {
      auto pos = actor->branch_inputs_pos_[i][j];
      auto op_arrow = std::make_shared<DataArrow>(pos, to_actor->GetAID(), j);
      actor->output_branch_arrows_[i].emplace_back(op_arrow);
    }
  }
}

void GraphScheduler::LinkControlArrowForGatherActor(std::vector<GatherActorPtr> *from_actors, LoopCountActor *to_actor,
                                                    const std::vector<KernelGraphPtr> &graphs) {
  if (from_actors == nullptr || to_actor == nullptr) {
    return;
  }

  // Link control arrow to kernel actor.
  for (size_t i = 0; i < graphs.size(); ++i) {
    const auto &kernel_graph = graphs[i];
    MS_EXCEPTION_IF_NULL(kernel_graph);
    const auto &func_graph = kernel_graph->GetFuncGraph();
    if (func_graph == nullptr) {
      continue;
    }
    const auto &actor = FetchActor(func_graph->ToString());
    if (actor == nullptr) {
      continue;
    }
    const auto &gather_actor = dynamic_cast<GatherActor *>(actor);
    MS_EXCEPTION_IF_NULL(gather_actor);

    // When gather actor is not empty, it means the control arrow of no input kernel actor needs to be sent by gather.
    for (const auto &kernel : kernel_graph->execution_order()) {
      if (IsKernelActor(kernel) && (!IsSkippedKernelActor(kernel))) {
        const auto &kernel_actor = dynamic_cast<KernelActor *>(FetchActor(kernel->fullname_with_scope()));
        MS_EXCEPTION_IF_NULL(kernel_actor);

        if ((kernel_actor->input_datas_num_ == 0) && (kernel_actor->input_controls_num_ == 0)) {
          gather_actor->output_control_arrows_.emplace_back(kernel_actor->GetAID());
          kernel_actor->input_controls_num_ = 1;
        }
      }
    }
  }

  // link control arrow to loop count actor.
  for (auto &from_actor : *from_actors) {
    MS_EXCEPTION_IF_NULL(from_actor);

    // If the gather actor has no output, then adds the output control to loop count actor.
    if (from_actor->output_data_arrows_.size() == 0 && from_actor->output_control_arrows_.size() == 0) {
      auto to_aid = to_actor->GetAID();
      from_actor->output_control_arrows_.emplace_back(to_aid);
      to_actor->branch_id_to_input_controls_num_[kMainBranchID]++;
    }
  }
}

void GraphScheduler::LinkBranchArrowForGatherActor(const GraphCompilerInfo &graph_compiler_info,
                                                   const ActorSet *actor_set) {
  if (graph_compiler_info.control_nodes_.empty()) {
    return;
  }

  const auto func_graph = graph_compiler_info.control_nodes_[0]->func_graph();
  const auto &loop_count_actor = actor_set->loop_count_actor_.get();
  const auto &output_actor = actor_set->output_actor_.get();

  // If there is only one branch output, set the branch id of the loop count to 0, no need to send the branch id.
  auto outputs = graph_compiler_info.control_node_parser_->front_output_nodes_;
  if (outputs.size() == 1) {
    return;
  }

  loop_count_actor->branch_id_ = kInvalidBranchID;
  output_actor->branch_id_ = kInvalidBranchID;

  std::vector<FuncGraphPtr> output_func_graphs;
  for_each(outputs.begin(), outputs.end(),
           [&output_func_graphs](const AnfNodePtr &output) { output_func_graphs.push_back(output->func_graph()); });
  int func_graph_num = SizeToInt(output_func_graphs.size());
  std::unordered_map<FuncGraphPtr, size_t> graph_to_control_num;

  // Count the control arrow num of gather actor.
  for (int i = 0; i < func_graph_num; ++i) {
    auto output_func_graph = output_func_graphs[i];
    auto actor_name = output_func_graph->ToString();
    auto actor = FetchActor(actor_name);
    if (actor == nullptr) {
      continue;
    }
    const auto &from_actor = dynamic_cast<GatherActor *>(actor);
    MS_EXCEPTION_IF_NULL(from_actor);

    from_actor->branch_id_ = i;
    graph_to_control_num[output_func_graph] = 0;
    if ((from_actor->output_data_arrows_.size() == 0) && (from_actor->output_control_arrows_.size() == 0)) {
      graph_to_control_num[output_func_graph]++;
    }
  }

  // Count the control arrow num of kernel actor.
  for (const auto &kernel_actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    if ((kernel_actor->output_data_arrows_.size() == 0) && (kernel_actor->output_control_arrows_.size() == 0)) {
      MS_EXCEPTION_IF_NULL(kernel_actor->kernel_);
      const auto &sub_func_graph = FetchFuncGraphByNode(kernel_actor->kernel_);
      if (sub_func_graph == nullptr) {
        MS_LOG(EXCEPTION) << "Cannot get funcgraph from kernel:" << kernel_actor->kernel_->fullname_with_scope();
      }

      if (graph_to_control_num.find(sub_func_graph) != graph_to_control_num.end()) {
        graph_to_control_num[sub_func_graph]++;
      } else {
        for (auto &pair : graph_to_control_num) {
          pair.second++;
        }
      }
    }
  }

  for (size_t i = 0; i < graph_to_control_num.size(); ++i) {
    // Branch id starts from 1.
    auto branch_id = SizeToInt(i) + kSubBranchStartID;
    auto sub_func_graph = output_func_graphs[i];
    auto gather_actor_name = sub_func_graph->ToString();
    auto actor = FetchActor(gather_actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    auto gather_actor = dynamic_cast<GatherActor *>(actor);
    MS_EXCEPTION_IF_NULL(gather_actor);

    gather_actor->branch_id_ = branch_id;
    loop_count_actor->branch_id_to_input_controls_num_[branch_id] = graph_to_control_num[sub_func_graph];
  }

  // If the switch actor is linked to the output actor, it will link a control arrow to the loop count actor,
  // and this should be recorded.
  for (const auto &from_actor : actor_set->switch_actors_) {
    MS_EXCEPTION_IF_NULL(from_actor);
    auto origin_output_with_index = KernelWithIndex(from_actor->node_, 0);
    const auto &iter = graph_compiler_info.origin_outputs_order_.find(origin_output_with_index);
    if (iter == graph_compiler_info.origin_outputs_order_.end()) {
      continue;
    }
    loop_count_actor->branch_id_to_input_controls_num_[iter->second.first]++;
  }
}

void GraphScheduler::LinkOutputResultArrowForGatherActor(const GraphCompilerInfo &graph_compiler_info,
                                                         const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  OutputActor *to_actor = actor_set->output_actor_.get();
  MS_EXCEPTION_IF_NULL(to_actor);

  for (const auto gather_actor : actor_set->gather_actors_) {
    MS_EXCEPTION_IF_NULL(gather_actor);

    for (size_t i = 0; i < gather_actor->data_nodes_.size(); ++i) {
      const auto front_node = gather_actor->data_nodes_[i];
      auto origin_output_with_index = KernelWithIndex(front_node, 0);
      const auto &iter = graph_compiler_info.origin_outputs_order_.find(origin_output_with_index);
      if (iter == graph_compiler_info.origin_outputs_order_.end()) {
        continue;
      }

      for (auto &output_position : iter->second.second) {
        MS_LOG(INFO) << "Link output node:" << AnfAlgo::GetNodeDebugString(origin_output_with_index.first)
                     << " branch id:" << iter->second.first << " index:" << output_position
                     << " for gather actor:" << gather_actor->GetAID();

        auto op_arrow = std::make_shared<DataArrow>(i, to_actor->GetAID(), output_position);
        gather_actor->output_result_arrows_.emplace_back(op_arrow);
        const auto &backend_nodes = gather_actor->front_to_backend_parameter_[front_node];
        if (backend_nodes.empty()) {
          MS_LOG(EXCEPTION) << "No backend node for data node:" << AnfAlgo::GetNodeDebugString(front_node);
        }

        const auto &backend_node = backend_nodes[0].first;
        if (backend_node->isa<Parameter>()) {
          std::string actor_name = graph_compiler_info.name_ + "_HostDSActor";
          auto actor = FetchActor(actor_name);
          MS_EXCEPTION_IF_NULL(actor);
          auto host_ds_actor = dynamic_cast<HostQueueDataSourceActor *>(actor);
          MS_EXCEPTION_IF_NULL(host_ds_actor);

          const auto &data_nodes = host_ds_actor->data_nodes_;
          const auto &node_iter = find(data_nodes.begin(), data_nodes.end(), backend_node);
          if (node_iter == data_nodes.end()) {
            MS_LOG(EXCEPTION) << "Cannot find backend node in host data source actor, node:"
                              << AnfAlgo::GetNodeDebugString(backend_node);
          }
          to_actor->device_contexts_[output_position] = host_ds_actor->device_contexts_[node_iter - data_nodes.begin()];
        } else {
          auto actor_base = FetchActor(backend_node->fullname_with_scope());
          MS_EXCEPTION_IF_NULL(actor_base);
          auto kernel_actor = dynamic_cast<KernelActor *>(actor_base);
          MS_EXCEPTION_IF_NULL(kernel_actor);
          to_actor->device_contexts_[output_position] = kernel_actor->device_context_;
        }
      }
    }
  }
}

bool GraphScheduler::CheckActorValid(const ActorSet *actor_set, GraphExecutionStrategy strategy) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  // Check the data source actors.
  for (const auto &data_source_actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(data_source_actor);
    if (data_source_actor->output_data_arrows_.size() + data_source_actor->output_result_arrows_.size() == 0) {
      MS_LOG(ERROR) << data_source_actor->GetAID().Name() << " has no user.";
      return false;
    }
  }

  if (strategy == GraphExecutionStrategy::kStep) {
    return true;
  }

  // Check the kernel actors.
  for (const auto &kernel_actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    if (kernel_actor->output_data_arrows_.size() + kernel_actor->output_control_arrows_.size() == 0) {
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

  // Check the copy actors.
  for (const auto &copy_actor : actor_set->copy_actors_) {
    MS_EXCEPTION_IF_NULL(copy_actor);
    if (copy_actor->output_data_arrows_.size() + copy_actor->output_control_arrows_.size() == 0) {
      MS_LOG(ERROR) << copy_actor->GetAID().Name() << " has no user.";
      return false;
    }

    const size_t kCopyActorInputDataNum = 1;
    auto input_data_num = copy_actor->input_datas_num_;
    auto device_tensor_store_num = (copy_actor->device_tensor_store_key_.second == nullptr) ? 0 : 1;
    if (input_data_num + device_tensor_store_num != kCopyActorInputDataNum) {
      MS_LOG(ERROR) << "The input building of " << copy_actor->GetAID().Name()
                    << " is wrong, input data num: " << input_data_num
                    << ", device tensor store num: " << device_tensor_store_num
                    << ", total input num: " << kCopyActorInputDataNum;
      return false;
    }
  }

  // Check the loop count actor.
  const auto &loop_count_actor = actor_set->loop_count_actor_;
  if ((loop_count_actor != nullptr) &&
      (actor_set->data_source_actors_.size() + actor_set->kernel_actors_.size() + actor_set->copy_actors_.size() > 0)) {
    if (loop_count_actor->branch_id_to_input_controls_num_[kMainBranchID] == 0) {
      MS_LOG(ERROR) << loop_count_actor->GetAID().Name() << " has no source.";
      return false;
    }
  }

  return true;
}

void GraphScheduler::PersistDeviceTensor(const GraphCompilerInfo &graph_compiler_info) {
  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    const auto &device_context = graph_compiler_info.device_contexts_[i];
    MS_EXCEPTION_IF_NULL(graph);
    MS_EXCEPTION_IF_NULL(device_context);

    for (auto &value_node : graph->graph_value_nodes()) {
      MS_EXCEPTION_IF_NULL(value_node);
      if (!AnfAlgo::OutputAddrExist(value_node, 0)) {
        MS_LOG(INFO) << "The device address is not exist: " << value_node->ToString();
        continue;
      }
      auto device_tensor = AnfAlgo::GetMutableOutputAddr(value_node, 0, false);
      const auto &front_node = FetchFrontNodeByBackendNode(value_node, graph);
      DeviceTensorStore::GetInstance().Insert(front_node.get(), device_tensor);
      UpdateRefCount(device_tensor.get(), true);
    }

    for (auto &input_node : graph->input_nodes()) {
      MS_EXCEPTION_IF_NULL(input_node);
      if (!IsPersistentDeviceTensor(input_node)) {
        continue;
      }
      auto device_tensor = AnfAlgo::GetMutableOutputAddr(input_node, 0, false);
      MS_EXCEPTION_IF_NULL(device_tensor);
      const auto &front_node = FetchFrontNodeByBackendNode(input_node, graph);
      DeviceTensorStore::GetInstance().Insert(front_node.get(), device_tensor);
      UpdateRefCount(device_tensor.get(), true);

      // If the device tensor store of this device type is not exist, then create the new device tensor of this type.
      if (DeviceTensorStore::GetInstance().Fetch(front_node.get(), device_context->GetDeviceAddressType()) == nullptr) {
        MS_LOG(INFO) << "Fetch no device tensor store by:" << front_node->fullname_with_scope()
                     << ", type:" << device_context->GetDeviceAddressType();
        auto other_type_device_tensor = device_context->CreateDeviceAddress(
          nullptr, device_tensor->GetSize(), device_tensor->format(), device_tensor->type_id());
        DeviceTensorStore::GetInstance().Insert(front_node.get(), other_type_device_tensor);
        UpdateRefCount(other_type_device_tensor.get(), true);
      }
    }
  }

  // In control flow, there may be some value nodes that is not in the kernel graph and needs to be placed
  // in the tensor store separately.
  for (const auto &value_node : graph_compiler_info.control_node_parser_->front_value_nodes_) {
    MS_EXCEPTION_IF_NULL(value_node.first);
    auto device_tensor = AnfAlgo::GetMutableOutputAddr(value_node.first, 0, false);
    DeviceTensorStore::GetInstance().Insert(value_node.first.get(), device_tensor);
    UpdateRefCount(device_tensor.get(), true);
  }
}

HostTensorQueue *GraphScheduler::FetchHostQueue(const ActorInfo &actor_info) const {
  const auto &iter = actor_to_host_queue_.find(actor_info);
  if (iter != actor_to_host_queue_.end()) {
    return iter->second.get();
  } else {
    return nullptr;
  }
}

void GraphScheduler::InsertActor(OpActor<DeviceTensor> *actor) {
  MS_EXCEPTION_IF_NULL(actor);
  if (actor_name_to_actor_.count(actor->GetAID().Name()) > 0) {
    MS_LOG(EXCEPTION) << "The actor already exists: " << actor->GetAID().Name();
  }
  actor_name_to_actor_[actor->GetAID().Name()] = actor;
}

OpActor<DeviceTensor> *GraphScheduler::FetchActor(const std::string &actor_name) const {
  const auto &iter = actor_name_to_actor_.find(actor_name);
  if (iter == actor_name_to_actor_.end()) {
    return nullptr;
  }
  return iter->second;
}

void GraphScheduler::DumpActor(const ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  const auto &context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (!save_graphs) {
    return;
  }
  auto save_graphs_path = context_ptr->get_param<std::string>(MS_CTX_SAVE_GRAPHS_PATH);
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }

  std::string filename = save_graphs_path + "/actor_set_" + actor_set->name_ + ".ir";
  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file [" << filename << "] failed!";
    return;
  }

  ofs << "[Device tensor stores]\n";
  DumpDeviceTensorStore(graph_compiler_info, ofs);

  ofs << "\n\n[Data source actors]\n";
  for (const auto &data_source_actor : actor_set->data_source_actors_) {
    DumpDSActor(data_source_actor.get(), ofs);
  }

  ofs << "\n\n[Kernel actors]\n";
  for (const auto &kernel_actor : actor_set->kernel_actors_) {
    DumpKernelActor(kernel_actor.get(), ofs);
  }

  ofs << "\n\n[No input kernel actors]\n";
  for (const auto &no_input_kernel_actor : actor_set->no_input_kernel_actors_) {
    DumpKernelActor(no_input_kernel_actor.get(), ofs);
  }

  ofs << "\n\n[Copy actors]\n";
  for (const auto &copy_actor : actor_set->copy_actors_) {
    DumpCopyActor(copy_actor.get(), ofs);
  }

  ofs << "\n\n[Loop count actor]\n";
  const auto &loop_count_actor = actor_set->loop_count_actor_;
  if (loop_count_actor != nullptr) {
    DumpLoopCountActor(loop_count_actor.get(), ofs);
  }

  ofs << "\n\n[Output actor]\n";
  const auto &output_actor = actor_set->output_actor_;
  if (output_actor != nullptr) {
    DumpOutputActor(output_actor.get(), ofs);
  }
}

void GraphScheduler::DumpBaseActor(const OpActor<DeviceTensor> *actor, std::ofstream &ofs) const {
  MS_EXCEPTION_IF_NULL(actor);

  const auto &output_data_arrows = actor->output_data_arrows();
  ofs << "\t\toutput_data_arrows:" << output_data_arrows.size() << "\n ";
  for (const auto &data_arrow : output_data_arrows) {
    MS_EXCEPTION_IF_NULL(data_arrow);
    ofs << "\t\t\tfrom_output_index:" << data_arrow->from_output_index_
        << "\tto_actor_name:" << data_arrow->to_op_id_.Name() << "\tto_input_index:" << data_arrow->to_input_index_
        << "\n";
  }

  const auto &output_control_arrows = actor->output_control_arrows();
  ofs << "\t\toutput_control_arrows:" << output_control_arrows.size() << "\n ";
  for (const auto &aid : output_control_arrows) {
    ofs << "\t\t\tto_actor_name:" << aid.Name() << "\n";
  }
}

void GraphScheduler::DumpDSActor(const DataSourceActor *actor, std::ofstream &ofs) const {
  MS_EXCEPTION_IF_NULL(actor);
  const auto &actor_name = actor->GetAID().Name();

  if (actor_name.find("_DeviceDSActor") != string::npos) {
    // Dump the member info of device queue data source actor.
    const auto &device_queue_ds_actor = dynamic_cast<const DeviceQueueDataSourceActor *>(actor);
    MS_EXCEPTION_IF_NULL(device_queue_ds_actor->device_context_);
    ofs << "\tactor_name:" << actor_name
        << "\tdevice_context:" << device_queue_ds_actor->device_context_->device_context_key().ToString() << "\n";
    const auto &data_kernel = device_queue_ds_actor->data_kernel_;
    MS_EXCEPTION_IF_NULL(data_kernel);
    ofs << "\t\tdata_kernel_name:" << data_kernel->fullname_with_scope()
        << "\tinput_number:" << AnfAlgo::GetInputTensorNum(data_kernel)
        << "\toutput_number:" << AnfAlgo::GetOutputTensorNum(data_kernel) << "\n";
    for (size_t i = 0; i < AnfAlgo::GetOutputTensorNum(data_kernel); ++i) {
      const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(data_kernel, i, false);
      MS_EXCEPTION_IF_NULL(device_tensor);
      ofs << "\t\t\toutput_index:" << i << "\tptr:" << device_tensor->GetPtr() << "\tsize:" << device_tensor->GetSize()
          << "\toriginal_ref_count:" << device_tensor->original_ref_count() << "\n ";
    }
  } else if (actor_name.find("_HostDSActor") != string::npos) {
    // Dump the member info of host queue data source actor.
    ofs << "\tactor_name:" << actor_name << "\n";
    const auto &host_queue_ds_actor = dynamic_cast<const HostQueueDataSourceActor *>(actor);
    ofs << "\t\tdata_nodes:" << host_queue_ds_actor->data_nodes_.size() << "\n";
    for (size_t i = 0; i < host_queue_ds_actor->data_nodes_.size(); ++i) {
      const auto &data_node = host_queue_ds_actor->data_nodes_[i];
      MS_EXCEPTION_IF_NULL(data_node);
      const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(data_node, 0, false);
      MS_EXCEPTION_IF_NULL(device_tensor);
      ofs << "\t\t\tnode_order_number:" << i << "\tnode_name:" << data_node->fullname_with_scope()
          << "\tptr:" << device_tensor->GetPtr() << "\tsize:" << device_tensor->GetSize()
          << "\toriginal_ref_count:" << device_tensor->original_ref_count()
          << "\tdevice_context:" << host_queue_ds_actor->device_contexts_[i]->device_context_key().ToString() << "\n";
    }
  }

  DumpBaseActor(actor, ofs);

  ofs << "\t\toutput_result_arrows:" << actor->output_result_arrows_.size() << "\n ";
  for (const auto &result_arrow : actor->output_result_arrows_) {
    MS_EXCEPTION_IF_NULL(result_arrow);
    ofs << "\t\t\tfrom_output_index:" << result_arrow->from_output_index_
        << "\tto_actor_name:" << result_arrow->to_op_id_.Name()
        << "\toutput_node_position:" << result_arrow->to_input_index_ << "\n";
  }
  ofs << "\n";
}

void GraphScheduler::DumpLoopCountActor(const LoopCountActor *actor, std::ofstream &ofs) const {
  MS_EXCEPTION_IF_NULL(actor);
  ofs << "\tactor_name:" << actor->GetAID().Name() << "\tloop_count:" << actor->loop_count_
      << "\tinput_controls_num:" << actor->branch_id_to_input_controls_num_.at(kMainBranchID) << "\n";

  ofs << "\t\toutput_control_arrows:" << (actor->data_source_aids_.size() + actor->no_input_kernel_aids_.size() + 1)
      << "\n ";
  for (const auto &aid : actor->data_source_aids_) {
    ofs << "\t\t\tto_actor_name:" << aid.Name() << "\n";
  }
  for (const auto &aid : actor->no_input_kernel_aids_) {
    ofs << "\t\t\tto_actor_name:" << aid.Name() << "\n";
  }
  ofs << "\t\t\tto_actor_name:" << actor->output_aid_.Name() << "\n";

  ofs << "\t\tcontinuous_memory_nodes:" << actor->continuous_memory_nodes_.size() << "\n ";
  for (const auto &iter : actor->continuous_memory_nodes_) {
    ofs << "\t\t\tnode_name:" << iter.first.first->fullname_with_scope()
        << "\tdevice_context:" << iter.first.second->device_context_key().ToString()
        << "\tis_input_need:" << iter.second.first << "\tis_output_need:" << iter.second.second << "\n";
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
        << "\toriginal_ref_count:" << device_tensor->original_ref_count() << "\n ";
  }

  ofs << "\t\tdevice_tensor_stores:" << actor->device_tensor_store_keys_.size() << "\n ";
  for (const auto &device_tensor_store_key : actor->device_tensor_store_keys_) {
    MS_EXCEPTION_IF_NULL(device_tensor_store_key.second);
    ofs << "\t\t\tto_input_index:" << device_tensor_store_key.first
        << "\tfrom_node_name:" << device_tensor_store_key.second->fullname_with_scope() << "\n";
  }

  DumpBaseActor(actor, ofs);

  ofs << "\t\toutput_result_arrows:" << actor->output_result_arrows_.size() << "\n ";
  for (const auto &result_arrow : actor->output_result_arrows_) {
    MS_EXCEPTION_IF_NULL(result_arrow);
    ofs << "\t\t\tfrom_output_index:" << result_arrow->from_output_index_
        << "\tto_actor_name:" << result_arrow->to_op_id_.Name()
        << "\toutput_node_position:" << result_arrow->to_input_index_ << "\n";
  }
  ofs << "\n";
}

void GraphScheduler::DumpOutputActor(const OutputActor *actor, std::ofstream &ofs) const {
  MS_EXCEPTION_IF_NULL(actor);
  ofs << "\tactor_name:" << actor->GetAID().Name() << "\tloop_count:" << actor->loop_count_
      << "\toutputs_num:" << actor->outputs_num_ << "\n";

  ofs << "\t\tdevice_tensor_store_keys:" << actor->device_tensor_store_keys_.at(kMainBranchID).size() << "\n ";
  for (const auto &device_tensor_store_key : actor->device_tensor_store_keys_.at(kMainBranchID)) {
    MS_EXCEPTION_IF_NULL(device_tensor_store_key.second);
    ofs << "\t\t\toutput_node_position:" << device_tensor_store_key.first
        << "\toutput_node_name:" << device_tensor_store_key.second->fullname_with_scope() << "\n";
  }
}

void GraphScheduler::DumpCopyActor(const CopyActor *actor, std::ofstream &ofs) const {
  MS_EXCEPTION_IF_NULL(actor);
  MS_EXCEPTION_IF_NULL(actor->input_device_context_);
  MS_EXCEPTION_IF_NULL(actor->output_device_context_);
  ofs << "\tactor_name:" << actor->GetAID().Name()
      << "\tinput_device_context:" << actor->input_device_context_->device_context_key().ToString()
      << "\toutput_device_context:" << actor->output_device_context_->device_context_key().ToString()
      << "\tinput_data_num:" << actor->input_datas_num_ << "\tinput_controls_num:" << actor->input_controls_num_
      << "\n";

  auto device_tensor = actor->output_;
  if (device_tensor != nullptr) {
    ofs << "\t\toutput_index:" << 0 << "\tptr:" << device_tensor->GetPtr() << "\tsize:" << device_tensor->GetSize()
        << "\toriginal_ref_count:" << device_tensor->original_ref_count() << "\n ";
  }

  if (actor->device_tensor_store_key_.second != nullptr) {
    ofs << "\t\tdevice_tensor_stores:" << 1 << "\n ";
    ofs << "\t\t\tto_input_index:" << actor->device_tensor_store_key_.first
        << "\tfrom_node_name:" << actor->device_tensor_store_key_.second->fullname_with_scope() << "\n";
  }

  DumpBaseActor(actor, ofs);
  ofs << "\n";
}

void GraphScheduler::DumpDeviceTensorStore(const GraphCompilerInfo &graph_compiler_info, std::ofstream &ofs) const {
  for (const auto &graph : graph_compiler_info.graphs_) {
    MS_EXCEPTION_IF_NULL(graph);
    ofs << "\tgraph id:" << graph->graph_id() << "\n";

    for (auto &value_node : graph->graph_value_nodes()) {
      MS_EXCEPTION_IF_NULL(value_node);
      if (!AnfAlgo::OutputAddrExist(value_node, 0)) {
        continue;
      }
      const auto &front_node = FetchFrontNodeByBackendNode(value_node, graph);
      const auto device_tensors = DeviceTensorStore::GetInstance().Fetch(front_node.get());
      ofs << "\t\tdevcie tensor key:" << front_node->fullname_with_scope() << "\tvalue size:" << device_tensors.size()
          << "\n";
      for (const auto &device_tensor : device_tensors) {
        ofs << "\t\t\tdevcie tensor value:" << device_tensor << "\tptr:" << device_tensor->GetPtr()
            << "\tsize:" << device_tensor->GetSize() << "\toriginal_ref_count:" << device_tensor->original_ref_count()
            << "\tdevice_type:" << device_tensor->DeviceType() << "\n ";
      }
    }

    for (auto &input_node : graph->input_nodes()) {
      MS_EXCEPTION_IF_NULL(input_node);
      if (!IsPersistentDeviceTensor(input_node)) {
        continue;
      }
      const auto &front_node = FetchFrontNodeByBackendNode(input_node, graph);
      const auto device_tensors = DeviceTensorStore::GetInstance().Fetch(front_node.get());
      ofs << "\t\tdevcie tensor key:" << front_node->fullname_with_scope() << "\tvalue size:" << device_tensors.size()
          << "\n";
      for (const auto &device_tensor : device_tensors) {
        ofs << "\t\t\tdevcie tensor value:" << device_tensor << "\tptr:" << device_tensor->GetPtr()
            << "\tsize:" << device_tensor->GetSize() << "\toriginal_ref_count:" << device_tensor->original_ref_count()
            << "\tdevice_type:" << device_tensor->DeviceType() << "\n ";
      }
    }
    ofs << "\n";
  }
}
}  // namespace runtime
}  // namespace mindspore
