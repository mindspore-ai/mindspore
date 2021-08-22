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
#if !defined(_WIN32) && !defined(_WIN64)
#include "utils/signal_util.h"
#endif
#include "common/trans.h"
#include "debug/data_dump/dump_json_parser.h"
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/recorder_manager.h"
#endif
#ifdef ENABLE_DEBUGGER
#include "debug/debugger/debugger.h"
#endif
#include "profiler/device/profiling.h"

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

void UpdateRefCount(DeviceTensor *const device_tensor, bool is_max_ref_count = false) {
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
      MS_LOG(EXCEPTION) << "Device(id:" << device_context->device_context_key().device_id_
                        << ") memory isn't enough and alloc failed, node name: " << node->fullname_with_scope()
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
      MS_LOG(EXCEPTION) << "Device(id:" << device_context->device_context_key().device_id_
                        << ") memory isn't enough and alloc failed, node name: " << node->fullname_with_scope()
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
    DeviceTensorStore::GetInstance().Insert(front_node.get(), host_tensor_address);
    if (host_tensor_address->DeviceType() == device_tensor->DeviceType()) {
      AnfAlgo::SetOutputAddr(host_tensor_address, 0, backend_node.get());
    } else {
      MS_LOG(INFO) << "The device type is not equal, host tensor type:" << host_tensor_address->DeviceType()
                   << ", device tensor type:" << device_tensor->DeviceType();
    }
  }

  // If the ptr of device tensor is not nullptr, it indicates that the device data has been prepared.
  if (host_tensor_address->GetPtr() == nullptr) {
    MS_LOG(INFO) << "Prepare device data for weight node:" << backend_node->fullname_with_scope()
                 << ", device type:" << host_tensor_address->DeviceType();
    // Allocate device memory and copy data from host tensor to device.
    if (!device_context->AllocateMemory(host_tensor_address.get(), host_tensor_address->GetSize())) {
      MS_LOG(EXCEPTION) << "Device(id:" << device_context->device_context_key().device_id_
                        << ") memory isn't enough and alloc failed, node name: " << backend_node->fullname_with_scope();
    }
    if (!host_tensor_address->SyncHostToDevice(trans::GetRuntimePaddingShape(backend_node, 0),
                                               LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                               tensor->data_c(), tensor->device_info().host_format_)) {
      MS_LOG(EXCEPTION) << "SyncHostToDevice failed, node name: " << backend_node->fullname_with_scope();
    }
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
    if (another_device_tensor->GetPtr() == nullptr) {
      if (!another_device_context->AllocateMemory(another_device_tensor.get(), another_device_tensor->GetSize())) {
        MS_LOG(EXCEPTION) << "Device(id:" << another_device_context->device_context_key().device_id_
                          << ") memory isn't enough and alloc failed, node name: "
                          << backend_node->fullname_with_scope();
      }
    }
    MS_LOG(INFO) << "Prepare device data for weight node:" << backend_node->fullname_with_scope()
                 << ", device type:" << another_device_type;
    if (!Copy(another_device_tensor.get(), host_tensor_address.get())) {
      MS_LOG(EXCEPTION) << "Sync data error.";
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
  bool need_update_device_tensor_store = (device_tensors.size() == 0) ? true : false;
  for (auto &device_tensor : device_tensors) {
    if (device_tensor->GetPtr() == nullptr) {
      need_update_device_tensor_store = true;
      break;
    }
  }
  if (need_update_device_tensor_store) {
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

void PrepareDataForHostDataSourceActor(const std::unordered_map<AnfNodePtr, size_t> &data_node_position_map,
                                       const AnfNodePtr &node, const TensorPtr &tensor,
                                       std::vector<TensorPtr> *host_tensors) {
  MS_EXCEPTION_IF_NULL(tensor);

  // Fill the host tensors for non weighted parameters.
  const auto &iter = data_node_position_map.find(node);
  if (iter == data_node_position_map.end()) {
    return;
  }

  (*host_tensors)[iter->second] = tensor;
  auto tensor_address = std::dynamic_pointer_cast<DeviceTensor>(tensor->device_address());
  auto device_address = AnfAlgo::GetMutableOutputAddr(node, 0, false);
  MS_EXCEPTION_IF_NULL(device_address);
  if ((tensor_address != nullptr) && (tensor_address->DeviceType() == device_address->DeviceType())) {
    AnfAlgo::SetOutputAddr(tensor_address, 0, node.get());
  }
}

void PrepareDataForInputData(const HostQueueDataSourceActor *host_data_source_actor, const AnfNodePtr &node,
                             const TensorPtr &tensor, const DeviceContext *device_context,
                             std::vector<TensorPtr> *const host_tensors) {
  MS_EXCEPTION_IF_NULL(tensor);
  // Fill the host tensors for non weighted parameters.
  if (host_data_source_actor != nullptr) {
    (*host_tensors)[host_data_source_actor->FetchDataNodePosition(node)] = tensor;
  }

  auto device_address = std::dynamic_pointer_cast<DeviceTensor>(tensor->device_address());
  if (device_address != nullptr) {
    AnfAlgo::SetOutputAddr(device_address, 0, node.get());
    return;
  }

  DeviceTensorPtr node_device_address = nullptr;
  if (!AnfAlgo::OutputAddrExist(node, 0, false)) {
    TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(node, 0);
    if (output_type_id == kTypeUnknown) {
      output_type_id = AnfAlgo::GetOutputInferDataType(node, 0);
    }

    size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(node, 0);
    auto new_device_address =
      device_context->CreateDeviceAddress(nullptr, tensor_size, AnfAlgo::GetOutputFormat(node, 0), output_type_id);
    AnfAlgo::SetOutputAddr(new_device_address, 0, node.get());
    node_device_address = new_device_address;
  } else {
    node_device_address = AnfAlgo::GetMutableOutputAddr(node, 0, false);
  }

  tensor->set_device_address(node_device_address);
  UpdateRefCount(node_device_address.get(), true);

  MS_EXCEPTION_IF_NULL(device_context);
  if (node_device_address->GetPtr() == nullptr &&
      !device_context->AllocateMemory(node_device_address.get(), node_device_address->GetSize())) {
    MS_LOG(EXCEPTION) << "Device(id:" << device_context->device_context_key().device_id_
                      << ") memory isn't enough and alloc failed, node name: " << node->fullname_with_scope();
  }

  if (!node_device_address->SyncHostToDevice(trans::GetRuntimePaddingShape(node, 0),
                                             LongToSize(tensor->data().nbytes()), tensor->data_type(), tensor->data_c(),
                                             tensor->device_info().host_format_)) {
    MS_LOG(EXCEPTION) << "SyncHostToDevice failed.";
  }
}

inline bool IsSingleOpActorSet(const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  return actor_set->kernel_actors_.size() == 1;
}

bool RunInStepMode(const ActorSet *actor_set, const std::vector<TensorPtr> *input_tensors) {
  OpContext<DeviceTensor> op_context;
  // Step mode does not need sequential number.
  op_context.sequential_num_ = nullptr;

  // Trigger kernel actor running in the step execution strategy.
  if (IsSingleOpActorSet(actor_set)) {
    MS_EXCEPTION_IF_NULL(input_tensors);
    for (auto &kernel_actor : actor_set->kernel_actors_) {
      MS_EXCEPTION_IF_NULL(kernel_actor);
      kernel_actor->RunOpControlWithInputTensor(nullptr, &op_context, input_tensors);
    }
    return true;
  }

  std::vector<Promise<int>> result(1);
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

  auto result_future = result[0].GetFuture();
  result_future.Wait();
  MsException::Instance().CheckException();
  return result_future.IsOK();
}

#if !defined(_WIN32) && !defined(_WIN64)
void IntHandler(int, siginfo_t *, void *) {
  int this_pid = getpid();
  MS_LOG(WARNING) << "Process " << this_pid << " receive KeyboardInterrupt signal.";
  (void)kill(this_pid, SIGTERM);
}
#endif
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
  thread_pool_ = ActorThreadPool::CreateThreadPool(actor_thread_num);
  MS_EXCEPTION_IF_NULL(thread_pool_);
  std::string OMP_env = std::to_string(OMP_thread_num);
  (void)common::SetEnv("OMP_NUM_THREADS", OMP_env.c_str(), 0);
  auto OMP_thread_num_used = common::GetEnv("OMP_NUM_THREADS");
  MS_LOG(INFO) << "The actor thread number: " << actor_thread_num
               << ", the computed OMP thread number : " << OMP_thread_num
               << ", the used OMP thread number : " << OMP_thread_num_used;

  BuildAndScheduleGlobalActor();
}

void GraphScheduler::BuildAndScheduleGlobalActor() {
  auto actorMgr = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actorMgr);

  // Create and schedule memory manager actor.
  auto memory_manager_actor = std::make_shared<MemoryManagerActor>();
  MS_EXCEPTION_IF_NULL(memory_manager_actor);
  memory_manager_aid_ = memory_manager_actor->GetAID();
  auto base_actor = static_cast<ActorReference>(memory_manager_actor);
  base_actor->set_thread_pool(thread_pool_);
  // Bind single thread to response to memory alloc and free quickly.
  (void)actorMgr->Spawn(base_actor, false);

  // Create and schedule recorder actor.
  auto recorder_actor = std::make_shared<RecorderActor>();
  MS_EXCEPTION_IF_NULL(recorder_actor);
  recorder_aid_ = &(recorder_actor->GetAID());
  auto base_recorder_actor = static_cast<ActorReference>(recorder_actor);
  base_recorder_actor->set_thread_pool(thread_pool_);
  (void)actorMgr->Spawn(base_recorder_actor, true);

  // Create and schedule debug actor.
  bool debugger_actor_need = DumpJsonParser::GetInstance().e2e_dump_enabled();
#ifdef ENABLE_DEBUGGER
  if (Debugger::GetInstance()->DebuggerBackendEnabled()) {
    debugger_actor_need = true;
  }
#endif
  if (debugger_actor_need) {
    auto debug_actor = std::make_shared<DebugActor>();
    MS_EXCEPTION_IF_NULL(debug_actor);
    debug_aid_ = &(debug_actor->GetAID());
    auto base_debug_actor = static_cast<ActorReference>(debug_actor);
    base_debug_actor->set_thread_pool(thread_pool_);
    (void)actorMgr->Spawn(base_debug_actor, true);
  }
}

ActorSet *GraphScheduler::Transform(const GraphCompilerInfo &graph_compiler_info) {
  MS_LOG(INFO) << "Graph(" << graph_compiler_info.name_ << ") transforms actor begin.";
  if (graph_compiler_info.graphs_.size() == 0) {
    MS_LOG(EXCEPTION) << "The number of graphs is zero.";
  }
  if (graph_compiler_info.graphs_.size() != graph_compiler_info.device_contexts_.size()) {
    MS_LOG(EXCEPTION) << "The number of graphs is not equal to the number of device contexts.";
  }

  PersistDeviceTensor(graph_compiler_info);
  auto strategy = graph_compiler_info.strategy_;
  const auto &actor_set = Build(graph_compiler_info);
  CacheGraphOutputToActor(graph_compiler_info);
  Link(actor_set.get(), graph_compiler_info);
  // The copy actors are built in the link, so need push into the actor set after link.
  actor_set->copy_actors_ = copy_actors_;

  (void)actors_.emplace(actor_set->name_, actor_set);

  DumpActor(actor_set.get(), graph_compiler_info);
  if (!CheckActorValid(actor_set.get(), strategy)) {
    MS_LOG(EXCEPTION) << "The actor set of " << graph_compiler_info.name_ << " is invalid.";
  }
  MS_LOG(INFO) << "Graph(" << graph_compiler_info.name_ << ") transforms actor end.";

  // Local maps and vectors clear.
  graph_output_to_actor_.clear();
  front_node_to_actor_.clear();
  copy_actors_.clear();

  return actor_set.get();
}

void GraphScheduler::Schedule(const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  std::vector<ActorReference> actors;

  // Collect actors.
  for (auto &data_source_actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(data_source_actor);
    (void)actors.emplace_back(static_cast<ActorReference>(data_source_actor));
  }
  for (auto &kernel_actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    (void)actors.emplace_back(static_cast<ActorReference>(kernel_actor));
  }
  for (auto &switch_actor : actor_set->switch_actors_) {
    MS_EXCEPTION_IF_NULL(switch_actor);
    (void)actors.emplace_back(static_cast<ActorReference>(switch_actor));
  }
  for (auto &gather_actor : actor_set->gather_actors_) {
    MS_EXCEPTION_IF_NULL(gather_actor);
    (void)actors.emplace_back(static_cast<ActorReference>(gather_actor));
  }
  for (auto &copy_actor : actor_set->copy_actors_) {
    MS_EXCEPTION_IF_NULL(copy_actor);
    (void)actors.emplace_back(static_cast<ActorReference>(copy_actor));
  }
  if (actor_set->loop_count_actor_ != nullptr) {
    (void)actors.emplace_back(static_cast<ActorReference>(actor_set->loop_count_actor_));
  }
  if (actor_set->output_actor_ != nullptr) {
    (void)actors.emplace_back(static_cast<ActorReference>(actor_set->output_actor_));
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
                                const std::vector<std::vector<TensorPtr>> &input_tensors) {
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
      } else if (IsHostQueueDSActor(input_node, graph, graph_compiler_info.origin_parameters_order_,
                                    graph_compiler_info.strategy_)) {
        MS_EXCEPTION_IF_NULL(host_data_source_actor);
        PrepareDataForHostDataSourceActor(host_data_source_actor->data_node_position_map_, input_node, input_tensor,
                                          &host_tensors);
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
  PrepareDataForControlNode(host_data_source_actor, graph_compiler_info.control_node_parser_,
                            graph_compiler_info.origin_parameters_order_, input_tensors.back(), &host_tensors);

  // 4.Prepare the data of host tensor queue(non weighted parameters of graph).
  if (host_data_source_actor != nullptr) {
    const auto &host_tensor_queue = FetchHostQueue(actor_set->name_);
    MS_EXCEPTION_IF_NULL(host_tensor_queue);
    host_tensor_queue->Push(host_tensors);
  }
}

void GraphScheduler::PrepareRunOp(const ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info,
                                  const std::vector<std::vector<TensorPtr>> &input_tensors) {
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

    // 2.Prepare the data of device tensor store(weights of graph), and fill host tensors for non weighted parameters.
    const auto &input_nodes = graph->input_nodes();
    const auto &tensors = input_tensors[i];
    for (size_t j = 0; j < input_nodes.size(); ++j) {
      const auto &input_node = input_nodes[j];
      const auto &input_tensor = tensors[j];
      MS_EXCEPTION_IF_NULL(input_node);
      if (IsPersistentDeviceTensor(input_node)) {
        // Prepare the device data for weights.
        PrepareDataForWeightNode(input_node, input_node, input_tensor, device_context);
      } else {
        PrepareDataForInputData(host_data_source_actor, input_node, input_tensor, device_context, &host_tensors);
      }
    }
  }

  // 3.Prepare the data of host tensor queue(non weighted parameters of graph).
  if (host_data_source_actor != nullptr) {
    const auto &host_tensor_queue = FetchHostQueue(actor_set->name_);
    MS_EXCEPTION_IF_NULL(host_tensor_queue);
    host_tensor_queue->Push(host_tensors);
  }
}

void GraphScheduler::PrepareDataForControlNode(HostQueueDataSourceActor *host_data_source_actor,
                                               const ControlNodeParserPtr &control_node_parser,
                                               const std::vector<AnfNodePtr> &origin_parameters,
                                               const std::vector<TensorPtr> &tensors,
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
      const auto &iter = host_data_source_actor->data_node_position_map_.find(input_node);
      if (iter == host_data_source_actor->data_node_position_map_.end()) {
        MS_LOG(EXCEPTION) << "Cannot find node" << AnfAlgo::GetNodeDebugString(input_node) << " in data source actor";
      }
      const size_t pos = iter->second;
      const AnfNodePtr &backend_node = host_data_source_actor->data_nodes_[pos];
      (*host_tensors)[pos] = input_tensor;
      auto device_address = std::dynamic_pointer_cast<DeviceTensor>(input_tensor->device_address());
      if (device_address != nullptr) {
        AnfAlgo::SetOutputAddr(device_address, 0, backend_node.get());
      }
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
#if !defined(_WIN32) && !defined(_WIN64)
  SignalGuard sg(IntHandler);
#endif
  if (strategy == GraphExecutionStrategy::kStep) {
    return RunInStepMode(actor_set, input_tensors);
  }

  // Construct OpContext.
  OpContext<DeviceTensor> op_context;
  uuids::uuid sequential_num;
  std::vector<Promise<int>> result(1);
  op_context.sequential_num_ = &sequential_num;
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

ActorSetPtr GraphScheduler::Build(const GraphCompilerInfo &graph_compiler_info) {
  auto actor_set = std::make_shared<ActorSet>(graph_compiler_info.name_);
  MS_EXCEPTION_IF_NULL(actor_set);

  auto host_queue = std::make_shared<HostTensorQueue>();
  (void)actor_to_host_queue_.emplace(actor_set->name_, host_queue);
  actor_set->data_source_actors_ = BuildDataSourceActor(graph_compiler_info, host_queue);
  actor_set->kernel_actors_ = BuildKernelActor(graph_compiler_info);
  actor_set->loop_count_actor_ = BuildLoopCountActor(graph_compiler_info);
  actor_set->output_actor_ = BuildOutputActor(graph_compiler_info);
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
      if (IsKernelActor(output_kernel, graph_compiler_info.strategy_)) {
        actor = FetchActor(output_kernel->fullname_with_scope());
      } else if (IsDeviceQueueDSActor(output_kernel, graph_compiler_info.strategy_)) {
        std::string actor_name = graph_compiler_info.name_ + "_DeviceDSActor" + "_" + std::to_string(graph->graph_id());
        actor = FetchActor(actor_name);
      } else if (IsHostQueueDSActor(output_kernel, graph, graph_compiler_info.origin_parameters_order_,
                                    graph_compiler_info.strategy_)) {
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
        MS_LOG(INFO) << "Ignore the internal parameter node:" << output_kernel->DebugString();
        continue;
      }

      MS_EXCEPTION_IF_NULL(actor);
      MS_LOG(INFO) << "Cache the graph " << graph->graph_id() << " output node:" << output_kernel->fullname_with_scope()
                   << " with index: " << output_with_index.second << " to actor:" << actor->GetAID().Name()
                   << " with index:" << actor_output_index;
      (void)graph_output_to_actor_.emplace(origin_output_with_index, GraphOutputPair(actor, actor_output_index));
    }
  }
}

void GraphScheduler::Link(ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info) {
  MS_EXCEPTION_IF_NULL(actor_set);
  std::vector<KernelActor *> auto_monad_actors;
  std::vector<CNodePtr> communication_nodes;
  const std::unordered_set<PrimitivePtr, PrimitiveHasher, PrimitiveEqual> auto_monad_prims = {
    prim::kPrimDepend, prim::kPrimUpdateState, prim::kPrimLoad};

  // Foreach the execution order to link the actors.
  for (size_t index = 0; index < graph_compiler_info.graphs_.size(); ++index) {
    const auto &graph = graph_compiler_info.graphs_[index];
    MS_EXCEPTION_IF_NULL(graph);
    auto execution_order = graph->execution_order();
    for (auto &kernel : execution_order) {
      if (AnfAlgo::IsCommunicationOp(kernel)) {
        (void)communication_nodes.emplace_back(kernel);
      }
      if (IsSkippedKernelActor(kernel) || (!IsKernelActor(kernel, graph_compiler_info.strategy_))) {
        continue;
      }
      const auto &kernel_actor = dynamic_cast<KernelActor *>(FetchActor(kernel->fullname_with_scope()));
      MS_EXCEPTION_IF_NULL(kernel_actor);

      for (size_t i = 0; i < AnfAlgo::GetInputNum(kernel); ++i) {
        auto input_node = AnfAlgo::GetInputNode(kernel, i);
        // Link the control arrows of kernel actor by the auto monad, the inputs include monad node.
        if (AnfAlgo::IsOneOfPrimitiveCNode(input_node, auto_monad_prims)) {
          LinkControlArrowByAutoMonad(kernel_actor, input_node, graph);
        }
        if (HasAbstractMonad(input_node)) {
          (void)auto_monad_actors.emplace_back(kernel_actor);
          continue;  // No data arrow for monad input.
        }

        KernelWithIndex from_kernel_with_output_idx = AnfAlgo::VisitKernelWithReturnType(input_node, 0, false);
        KernelWithIndex to_kernel_with_input_idx = std::make_pair(kernel, i);
        // The gather of linking data arrows of kernel by the different from kernel type.
        LinkDataArrow(kernel_actor, graph_compiler_info, graph, from_kernel_with_output_idx, to_kernel_with_input_idx);
      }
    }
    // Link the control arrows for allreduce kernel by the send/recv nodes in the kernel graph.
    LinkControlArrowBySendRecvNodes(graph);
  }

  // Link the control arrows by the communication nodes to ensure communication nodes running order.
  LinkControlArrowByCommunicationNode(communication_nodes, graph_compiler_info);

  if (graph_compiler_info.strategy_ == GraphExecutionStrategy::kPipeline) {
    // Link the arrow by control node.
    LinkArrowByControlNode(graph_compiler_info, actor_set);
  }

  // Auto monad actor may modify the device tensor store.
  LinkDeviceTensorStoreForAutoMonadActor(auto_monad_actors);

  // BuildNoInputKernelActor depends on whether kernel actors have input, so must be behind the link of kernel actors.
  actor_set->no_input_kernel_actors_ = BuildNoInputKernelActor(actor_set, graph_compiler_info.strategy_);

  // Link the control arrows of loop count actor, which depends on the no input kernel actors.
  LinkControlArrowForLoopCountActor(actor_set->loop_count_actor_.get(), actor_set,
                                    graph_compiler_info.control_node_parser_);

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

    for (size_t j = 0; j < input_nodes.size(); j++) {
      const auto &input_node = input_nodes[j];
      MS_EXCEPTION_IF_NULL(input_node);

      if (IsHostQueueDSActor(input_node, graph, graph_compiler_info.origin_parameters_order_,
                             graph_compiler_info.strategy_)) {
        if (host_queue_ds_actor == nullptr) {
          auto actor_name = graph_compiler_info.name_ + "_HostDSActor";
          MS_LOG(INFO) << "Create host queue data source actor: " << actor_name;
          host_queue_ds_actor = std::make_shared<HostQueueDataSourceActor>(actor_name, 1, memory_manager_aid_, nullptr,
                                                                           nullptr, host_queue);
          InsertActor(host_queue_ds_actor.get());
          (void)data_source_actors.emplace_back(host_queue_ds_actor);
        }

        const auto &front_node = FetchFrontNodeByBackendNode(input_node, graph);
        // In the scenario where multiple backend nodes correspond to the same front node, only the first backend node
        // is saved in the host queue data source actor.
        if (front_node_position_temp_map.count(front_node) > 0) {
          (void)host_queue_ds_actor->data_node_position_map_.emplace(input_node,
                                                                     front_node_position_temp_map[front_node]);
          continue;
        }
        (void)host_queue_ds_actor->data_nodes_.emplace_back(input_node);
        (void)host_queue_ds_actor->device_contexts_.emplace_back(device_context);
        (void)host_queue_ds_actor->data_node_position_map_.emplace(input_node, data_node_position);
        (void)front_node_position_temp_map.emplace(front_node, data_node_position);
        data_node_position++;
      }
    }

    // Build device queue data source actor.
    const auto &execution_order = graph->execution_order();
    const auto &iter =
      std::find_if(execution_order.begin(), execution_order.end(), [&graph_compiler_info](const CNodePtr &node) {
        return IsDeviceQueueDSActor(node, graph_compiler_info.strategy_);
      });
    if (iter != execution_order.end()) {
      auto actor_name = graph_compiler_info.name_ + "_DeviceDSActor" + "_" + std::to_string(graph->graph_id());
      MS_LOG(INFO) << "Create queue data source actor: " << actor_name;
      auto device_queue_ds_actor = std::make_shared<DeviceQueueDataSourceActor>(
        actor_name, 1, device_context, memory_manager_aid_, debug_aid_, recorder_aid_);
      MS_EXCEPTION_IF_NULL(device_queue_ds_actor);
      InsertActor(device_queue_ds_actor.get());
      (void)data_source_actors.emplace_back(device_queue_ds_actor);
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
      (void)data_source_actors.emplace_back(host_queue_ds_actor);
    }

    const auto &backend_node = backend_iter->second.first;
    auto iter = find(host_queue_ds_actor->data_nodes_.begin(), host_queue_ds_actor->data_nodes_.end(), backend_node);
    if (iter != host_queue_ds_actor->data_nodes_.end()) {
      (void)host_queue_ds_actor->data_node_position_map_.emplace(parameter,
                                                                 iter - host_queue_ds_actor->data_nodes_.begin());
    } else {
      (void)host_queue_ds_actor->data_node_position_map_.emplace(parameter, host_queue_ds_actor->data_nodes_.size());
      (void)host_queue_ds_actor->data_nodes_.emplace_back(backend_iter->second.first);
      (void)host_queue_ds_actor->device_contexts_.emplace_back(backend_iter->second.second);
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

    // Single op graph in step mode, kernel actor executes synchronously.
    bool is_single_op_graph = execution_order.size() == 1;
    GraphExecutionStrategy strategy = graph_compiler_info.strategy_;
    if (strategy == GraphExecutionStrategy::kStep) {
      strategy = (is_single_op_graph ? strategy : GraphExecutionStrategy::kPipeline);
    }

    for (auto &kernel : execution_order) {
      if (IsKernelActor(kernel, graph_compiler_info.strategy_) && (!IsSkippedKernelActor(kernel))) {
        auto kernel_actor = std::make_shared<KernelActor>(kernel->fullname_with_scope(), kernel, device_context,
                                                          memory_manager_aid_, debug_aid_, recorder_aid_, strategy);
        MS_EXCEPTION_IF_NULL(kernel_actor);
        InsertActor(kernel_actor.get());
        (void)kernel_actors.emplace_back(kernel_actor);
        auto front_node = graph->GetFrontAnfByBackendAnf(kernel);
        if (front_node != nullptr) {
          front_node_to_actor_[front_node] = kernel_actor;
        }
      }
    }
  }
  return kernel_actors;
}

LoopCountActorPtr GraphScheduler::BuildLoopCountActor(const GraphCompilerInfo &graph_compiler_info) {
  if (graph_compiler_info.strategy_ == GraphExecutionStrategy::kStep) {
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

OutputActorPtr GraphScheduler::BuildOutputActor(const GraphCompilerInfo &graph_compiler_info) {
  if (graph_compiler_info.strategy_ == GraphExecutionStrategy::kStep) {
    return nullptr;
  }

  auto loop_count = ConfigManager::GetInstance().iter_num();
  auto actor_name = graph_compiler_info.name_ + "_" + "OutputActor";
  bool need_loop_count = (graph_compiler_info.strategy_ == GraphExecutionStrategy::kPipeline) ? true : false;

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

      (void)no_input_kernel_actors.emplace_back(kernel_actor);
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

  // Build switch actor by switch node and switchlayer node.
  for (const auto &control_node : graph_compiler_info.control_nodes_) {
    if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitch) ||
        AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitchLayer)) {
      const auto func_graph = control_node->func_graph();
      const auto branch_id = graph_compiler_info.control_node_parser_->GetBranchIDByFuncGraph(func_graph);
      const auto &actor_name = control_node->DebugString();
      auto switch_actor = std::make_shared<SwitchActor>(actor_name, graph_compiler_info.device_contexts_[0],
                                                        control_node->cast<CNodePtr>(), branch_id, false);
      switch_actor->ParseInput(graph_compiler_info.control_node_parser_);

      // Fetch all the input nodes of switch actor.
      switch_actor->FetchInputNode(graph_compiler_info.control_node_parser_);
      InsertActor(switch_actor.get());
      (void)switch_actors.emplace_back(switch_actor);
    }
  }

  // Build switch actor by return node.
  const auto func_graphs_to_call_num = graph_compiler_info.control_node_parser_->func_graph_to_call_num_;
  for (const auto &func_graph_to_call_num : func_graphs_to_call_num) {
    const auto &return_node = func_graph_to_call_num.first->get_return();
    MS_EXCEPTION_IF_NULL(return_node);
    const auto &actor_name = return_node->DebugString();
    auto switch_actor = std::make_shared<SwitchActor>(actor_name, graph_compiler_info.device_contexts_[0],
                                                      return_node->cast<CNodePtr>(), kInvalidBranchID, true);
    switch_actor->ParseInput(graph_compiler_info.control_node_parser_);

    // Fetch all the input nodes of switch actor.
    switch_actor->FetchInputNode(graph_compiler_info.control_node_parser_);
    InsertActor(switch_actor.get());
    (void)switch_actors.emplace_back(switch_actor);
  }

  return switch_actors;
}

std::vector<GatherActorPtr> GraphScheduler::BuildGatherActor(const GraphCompilerInfo &graph_compiler_info) {
  std::vector<GatherActorPtr> gather_actors;

  const auto &loop_count_actor_name = graph_compiler_info.name_ + "_LoopCountActor";
  const auto &loop_count_actor = FetchActor(loop_count_actor_name);
  if (loop_count_actor == nullptr) {
    return gather_actors;
  }

  const auto &output_actor_name = graph_compiler_info.name_ + "_" + "OutputActor";
  const auto &output_actor = FetchActor(output_actor_name);
  MS_EXCEPTION_IF_NULL(output_actor);

  const auto parser = graph_compiler_info.control_node_parser_;

  bool is_main_return = true;
  // Each funcgraph has a return node, get the funcgraph from the return node, and create a gather actor.
  std::unordered_map<AnfNodePtr, AnfNodePtr> front_to_backend_kernel;
  for (const auto &pair : front_node_to_actor_) {
    front_to_backend_kernel[pair.first] = pair.second->kernel_;
  }

  for (const auto &control_node : graph_compiler_info.control_nodes_) {
    const auto &func_graph = control_node->func_graph();
    const auto &cnode = control_node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    const auto &return_node = func_graph->get_return();

    if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) {
      // Root funcgraph does not need to create a gather actor.
      if (is_main_return) {
        is_main_return = false;
        continue;
      }

      if (AnfAlgo::CheckPrimitiveType(inputs[kReturnInputPos], prim::kPrimPartial)) {
        continue;
      }
      auto actor_name = func_graph->ToString();
      std::vector<KernelWithIndex> parameters;
      for (const auto &parameter : func_graph->get_inputs()) {
        if (HasAbstractMonad(parameter) || HasAbstractRef(parameter)) {
          continue;
        }
        (void)parameters.emplace_back(parameter, 0);
      }

      const auto branch_id = parser->GetBranchIDByFuncGraph(func_graph);

      const auto &output_switch_actor = FetchActor(return_node->DebugString());
      MS_EXCEPTION_IF_NULL(output_switch_actor);
      const auto &output_switch_aid = output_switch_actor->GetAID();

      auto gather_actor =
        std::make_shared<GatherActor>(actor_name, parameters, true, output_switch_aid, AID(), branch_id);
      gather_actor->FetchBackendInputNode(func_graph, graph_compiler_info.control_node_parser_);
      InsertActor(gather_actor.get());
      (void)gather_actors.emplace_back(gather_actor);
    }
  }

  // Create gather actor for call node which input0 of call node is a funcgraph.
  for (const auto &control_node : graph_compiler_info.control_nodes_) {
    const auto &cnode = control_node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();

    if (inputs[0]->isa<ValueNode>() && IsValueNode<FuncGraph>(inputs[0])) {
      // Collect the parameters.
      std::vector<KernelWithIndex> parameters;
      for (size_t i = kCallInputStartPos; i < inputs.size(); ++i) {
        if (HasAbstractMonad(inputs[i]) || (inputs[i]->isa<Parameter>() && HasAbstractRef(inputs[i]))) {
          continue;
        }
        (void)parameters.emplace_back(inputs[i], 0);
      }

      auto func_graph = control_node->func_graph();
      auto actor_name = control_node->DebugString();
      const auto branch_id = parser->GetBranchIDByFuncGraph(func_graph);
      const auto &to_func_graph = GetValueNode<FuncGraphPtr>(inputs[0]);
      const auto &to_actor = FetchActor(to_func_graph->ToString());
      auto gather_actor =
        std::make_shared<GatherActor>(actor_name, parameters, false, AID(), to_actor->GetAID(), branch_id);
      gather_actor->FetchBackendInputNode(func_graph, graph_compiler_info.control_node_parser_);

      InsertActor(gather_actor.get());
      (void)gather_actors.emplace_back(gather_actor);
    }
  }

  // Create gather actor for kernel graph which has a call input.
  const auto &graph_with_device_contexts = graph_compiler_info.control_node_parser_->call_input_kernel_graphs_;
  for (const auto &graph_with_device_context : graph_with_device_contexts) {
    const auto &graph = graph_with_device_context.first;
    const auto &parameters = FetchParameterbyKernelGraph(graph);

    auto actor_name = graph->ToString();
    auto gather_actor = std::make_shared<GatherActor>(actor_name, parameters, false, AID(), AID(), kInvalidBranchID);
    InsertActor(gather_actor.get());
    (void)gather_actors.emplace_back(gather_actor);
  }

  return gather_actors;
}

void GraphScheduler::LinkDataArrow(KernelActor *to_actor, const GraphCompilerInfo &graph_compiler_info,
                                   const KernelGraphPtr &graph, KernelWithIndex from_kernel_with_output_idx,
                                   KernelWithIndex to_kernel_with_input_idx) {
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(graph);

  auto from_kernel = from_kernel_with_output_idx.first;
  auto front_node = GetFrontNodeByBackendNode(from_kernel);

  if (from_kernel->isa<Parameter>() && graph_compiler_info.control_node_parser_->IsCallInputKernelGraph(graph)) {
    const auto &kernel_with_index = GetFrontNodeByKernelGraph(from_kernel, graph);
    const auto &real_front_node_with_index =
      AnfAlgo::VisitKernelWithReturnType(kernel_with_index.first, SizeToInt(kernel_with_index.second));
    if (HasAbstractRef(real_front_node_with_index.first)) {
      (void)to_actor->device_tensor_store_keys_.emplace_back(to_kernel_with_input_idx.second,
                                                             real_front_node_with_index.first.get());
      return;
    }

    // When there is a call input in the kernel graph, all the inputs of the kernel graph needs to be sent by gather.
    const auto actor_name = graph->ToString();
    auto actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    LinkDataArrowForGatherActor(dynamic_cast<GatherActor *>(actor), to_actor, real_front_node_with_index,
                                to_kernel_with_input_idx);
    return;
  }

  if (IsDeviceQueueDSActor(from_kernel, graph_compiler_info.strategy_)) {
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
    if (HasAbstractRef(from_kernel)) {
      (void)to_actor->device_tensor_store_keys_.emplace_back(to_kernel_with_input_idx.second, front_node.get());
      return;
    }
    LinkDataArrowForGatherActor(from_actor, to_actor, {front_node, 0}, to_kernel_with_input_idx);
  } else if (IsHostQueueDSActor(from_kernel, graph, graph_compiler_info.origin_parameters_order_,
                                graph_compiler_info.strategy_)) {
    // Link the data arrows of host queue data source actor.
    std::string actor_name = graph_compiler_info.name_ + "_HostDSActor";
    const auto &from_actor = dynamic_cast<HostQueueDataSourceActor *>(FetchActor(actor_name));
    LinkDataArrowForHostDSActor(from_actor, to_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
  } else if (IsKernelActor(from_kernel, graph_compiler_info.strategy_)) {
    // Link the data arrows of kernel actor.
    const auto &from_actor = dynamic_cast<KernelActor *>(FetchActor(from_kernel->fullname_with_scope()));
    LinkDataArrowForKernelActor(from_actor, to_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
  } else if (IsInternalParameter(from_kernel, graph)) {
    // Link data arrow for internal parameter, convert internal parameter to actor by internal parameter cache to
    // link.
    LinkDataArrowForInternalParameter(from_kernel, graph_compiler_info.origin_parameters_order_, graph, to_actor,
                                      to_kernel_with_input_idx);
  } else if (IsPersistentDeviceTensor(from_kernel)) {
    const auto devcie_tensor_store_key = FetchFrontNodeByBackendNode(from_kernel, graph);
    (void)to_actor->device_tensor_store_keys_.emplace_back(to_kernel_with_input_idx.second,
                                                           devcie_tensor_store_key.get());
  } else {
    // May exist the from kernel that no need link in the pynative mode.
    MS_LOG(DEBUG) << "Invalid from kernel: " << from_kernel->fullname_with_scope();
  }
}

void GraphScheduler::LinkDataArrowForInternalParameter(const AnfNodePtr &internal_parameter,
                                                       const std::vector<AnfNodePtr> &host_parameters,
                                                       const KernelGraphPtr &graph, KernelActor *to_actor,
                                                       const KernelWithIndex &to_kernel_with_input_idx) {
  MS_EXCEPTION_IF_NULL(internal_parameter);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(to_actor);

  // Parameter ---> front node.
  auto front_output_with_index = graph->GetFrontNodeByInternalParameter(internal_parameter);
  auto front_output_node = front_output_with_index.first;
  MS_EXCEPTION_IF_NULL(front_output_node);
  if (IsSwitchActor(front_output_node)) {
    auto switch_actor = dynamic_cast<SwitchActor *>(FetchActor(front_output_node->DebugString()));
    MS_EXCEPTION_IF_NULL(switch_actor);
    LinkDataArrowForSwitchActor(switch_actor, 0, to_actor, to_kernel_with_input_idx.second);
    to_actor->input_datas_num_++;
    return;
  }
  if (IsPersistentDeviceTensor(front_output_node)) {
    (void)to_actor->device_tensor_store_keys_.emplace_back(to_kernel_with_input_idx.second, front_output_node.get());
    return;
  }

  // front node ---> actor.
  if (graph_output_to_actor_.count(front_output_with_index) == 0) {
    MS_LOG(EXCEPTION) << "Can't find actor by front node:" << AnfAlgo::GetNodeDebugString(front_output_node)
                      << ", internal parameter:" << AnfAlgo::GetNodeDebugString(internal_parameter);
  }
  auto actor_pair = graph_output_to_actor_[front_output_with_index];
  MS_EXCEPTION_IF_NULL(actor_pair.first);
  MS_LOG(INFO) << "Graph " << graph->graph_id() << " internal parameter:" << internal_parameter->DebugString()
               << ", corresponding front node:" << front_output_node->fullname_with_scope()
               << " with index:" << front_output_with_index.second
               << ", from actor:" << actor_pair.first->GetAID().Name() << " with index:" << actor_pair.second
               << ", to actor:" << to_actor->GetAID().Name() << " with index:" << to_kernel_with_input_idx.second;

  if (IsDeviceQueueDSActor(front_output_node)) {
    auto from_actor = dynamic_cast<DeviceQueueDataSourceActor *>(actor_pair.first);
    MS_EXCEPTION_IF_NULL(from_actor);
    auto from_kernel_with_output_idx = KernelWithIndex(from_actor->data_kernel_, actor_pair.second);
    LinkDataArrowForDeviceDSActor(from_actor, to_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
  } else if (IsKernelActor(front_output_node)) {
    auto from_actor = dynamic_cast<KernelActor *>(actor_pair.first);
    MS_EXCEPTION_IF_NULL(from_actor);
    auto from_kernel_with_output_idx = KernelWithIndex(from_actor->kernel_, actor_pair.second);
    LinkDataArrowForKernelActor(from_actor, to_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
  } else if (IsHostQueueDSActor(front_output_node, graph, host_parameters)) {
    auto from_actor = dynamic_cast<HostQueueDataSourceActor *>(actor_pair.first);
    MS_EXCEPTION_IF_NULL(from_actor);
    auto from_kernel_with_output_idx = KernelWithIndex(from_actor->data_nodes_[actor_pair.second], 0);
    LinkDataArrowForHostDSActor(from_actor, to_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
  } else {
    MS_LOG(EXCEPTION) << "Invalid internal parameter: " << internal_parameter->DebugString();
  }
}

void GraphScheduler::LinkDataArrowForDeviceDSActor(DeviceQueueDataSourceActor *const from_actor,
                                                   KernelActor *const to_actor,
                                                   const KernelWithIndex &from_kernel_with_output_idx,
                                                   const KernelWithIndex &to_kernel_with_input_idx) {
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
    (void)from_actor->output_data_arrows_.emplace_back(op_arrow);
    to_actor->input_datas_num_++;
    (void)to_actor->input_data_arrow_aids_.emplace_back(from_actor->GetAID());

    // Update the reference count of device tensor.
    UpdateRefCount(from_kernel, from_output_index);
  }
}

void GraphScheduler::LinkDataArrowForHostDSActor(HostQueueDataSourceActor *const from_actor,
                                                 KernelActor *const to_actor,
                                                 const KernelWithIndex &from_kernel_with_output_idx,
                                                 const KernelWithIndex &to_kernel_with_input_idx) {
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
    (void)from_actor->output_data_arrows_.emplace_back(op_arrow);
    to_actor->input_datas_num_++;
    (void)to_actor->input_data_arrow_aids_.emplace_back(from_actor->GetAID());

    // Update the reference count of device tensor.
    UpdateRefCount(from_actor->data_nodes_[position], from_output_index);
  }
}

void GraphScheduler::LinkDataArrowForKernelActor(KernelActor *from_actor, KernelActor *const to_actor,
                                                 KernelWithIndex from_kernel_with_output_idx,
                                                 const KernelWithIndex &to_kernel_with_input_idx) {
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
    (void)from_actor->output_data_arrows_.emplace_back(op_arrow);
    to_actor->input_datas_num_++;
    (void)to_actor->input_data_arrow_aids_.emplace_back(from_actor->GetAID());

    // Update the reference count of device tensor.
    UpdateRefCount(from_kernel, from_output_index);
  }
}

void GraphScheduler::LinkDataArrowForCopyActor(OpActor<DeviceTensor> *const from_actor, KernelActor *const to_actor,
                                               const KernelWithIndex &from_kernel_with_output_idx,
                                               const KernelWithIndex &to_kernel_with_input_idx) {
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
    (void)copy_actors_.emplace_back(copy_actor_shared_ptr);
    copy_actor = copy_actor_shared_ptr.get();
    MS_EXCEPTION_IF_NULL(copy_actor);
    InsertActor(copy_actor);

    // Link.
    const DeviceContext *from_devcie_context = nullptr;
    auto from_device_tensor = AnfAlgo::GetMutableOutputAddr(from_kernel, from_output_index, false);
    auto op_arrow_to_copy = std::make_shared<DataArrow>(from_output_index, copy_actor->GetAID(), 0);
    if (IsDeviceQueueDSActor(from_kernel)) {
      auto real_from_actor = dynamic_cast<DeviceQueueDataSourceActor *>(from_actor);
      MS_EXCEPTION_IF_NULL(real_from_actor);
      from_devcie_context = real_from_actor->device_context_;
      (void)real_from_actor->output_data_arrows_.emplace_back(op_arrow_to_copy);
    } else if (IsKernelActor(from_kernel)) {
      auto real_from_actor = dynamic_cast<KernelActor *>(from_actor);
      MS_EXCEPTION_IF_NULL(real_from_actor);
      from_devcie_context = real_from_actor->device_context_;
      (void)real_from_actor->output_data_arrows_.emplace_back(op_arrow_to_copy);
    } else if (IsHostQueueDSActor(from_kernel)) {
      auto real_from_actor = dynamic_cast<HostQueueDataSourceActor *>(from_actor);
      MS_EXCEPTION_IF_NULL(real_from_actor);
      auto position = real_from_actor->FetchDataNodePosition(from_kernel);
      from_devcie_context = real_from_actor->device_contexts_[position];
      op_arrow_to_copy->from_output_index_ = SizeToInt(position);
      (void)real_from_actor->output_data_arrows_.emplace_back(op_arrow_to_copy);
      from_device_tensor =
        AnfAlgo::GetMutableOutputAddr(real_from_actor->data_nodes_[position], from_output_index, false);
    }
    copy_actor->input_datas_num_++;

    // Set the member of the copy actor.
    MS_EXCEPTION_IF_NULL(from_device_tensor);
    auto to_kernel_mod = AnfAlgo::GetKernelMod(to_kernel_with_input_idx.first);
    MS_EXCEPTION_IF_NULL(to_kernel_mod);
    auto input_sizes = to_kernel_mod->GetInputSizeList();
    if (to_input_index >= input_sizes.size()) {
      MS_LOG(EXCEPTION) << "To input index(" << to_input_index << ") is out of size: " << input_sizes.size();
    }
    copy_actor->output_ = to_devcie_context->CreateDeviceAddress(
      nullptr, input_sizes[to_input_index], from_device_tensor->format(), from_device_tensor->type_id());
    MS_EXCEPTION_IF_NULL(from_devcie_context);
    copy_actor->input_device_context_ = from_devcie_context;
    copy_actor->output_device_context_ = to_devcie_context;

    // Update the reference count of device tensor.
    UpdateRefCount(from_device_tensor.get());
  }

  // If the copy actor already exists, only need link between copy actor and to actor.
  auto op_arrow_from_copy = std::make_shared<DataArrow>(0, to_actor->GetAID(), to_input_index);
  (void)copy_actor->output_data_arrows_.emplace_back(op_arrow_from_copy);
  to_actor->input_datas_num_++;
  UpdateRefCount(copy_actor->output_.get());
}

void GraphScheduler::LinkControlArrowByAutoMonad(KernelActor *to_actor, const AnfNodePtr &from_node,
                                                 const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(from_node);
  // Find the real input node, include the monad node and make tuple node.
  const std::vector<PrimitivePtr> return_types = {prim::kPrimDepend, prim::kPrimUpdateState, prim::kPrimLoad,
                                                  prim::kPrimMakeTuple};
  const auto &input_kernel_with_output_idx = AnfAlgo::VisitKernelWithReturnType(from_node, 0, false, return_types);
  MS_EXCEPTION_IF_NULL(input_kernel_with_output_idx.first);
  auto input_anfnode = input_kernel_with_output_idx.first;
  CNodePtr input_cnode = nullptr;
  if (input_anfnode->isa<CNode>()) {
    input_cnode = input_anfnode->cast<CNodePtr>();
  }
  // Make tuple node needs to be expanded.
  if (AnfAlgo::CheckPrimitiveType(input_anfnode, prim::kPrimMakeTuple)) {
    MS_EXCEPTION_IF_NULL(input_cnode);
    for (size_t i = 1; i < input_cnode->inputs().size(); ++i) {
      LinkControlArrowByAutoMonad(to_actor, input_cnode->input(i), graph);
    }
    return;
  }

  const std::unordered_set<PrimitivePtr, PrimitiveHasher, PrimitiveEqual> recursion_prims = {
    prim::kPrimDepend, prim::kPrimUpdateState, prim::kPrimLoad, prim::kPrimMakeTuple};
  // Get the real depend input by monad node which needs to link the control arrow.
  std::vector<AnfNodePtr> real_depend_inputs;
  if (AnfAlgo::CheckPrimitiveType(input_anfnode, prim::kPrimDepend) ||
      AnfAlgo::CheckPrimitiveType(input_anfnode, prim::kPrimLoad)) {
    MS_EXCEPTION_IF_NULL(input_cnode);
    real_depend_inputs.push_back(input_cnode->input(kDependAttachNodeIndex));
    // The real input may be this scene:  depend/load --> load/depend, so need add the control arrow for real input
    // node in this scene.
    if (AnfAlgo::IsOneOfPrimitiveCNode(input_cnode->input(kRealInputIndexInDepend), recursion_prims)) {
      real_depend_inputs.push_back(input_cnode->input(kRealInputIndexInDepend));
    }
  } else if (AnfAlgo::CheckPrimitiveType(input_anfnode, prim::kPrimUpdateState)) {
    MS_EXCEPTION_IF_NULL(input_cnode);
    for (size_t i = kUpdateStateRealInput; i < input_cnode->inputs().size(); ++i) {
      real_depend_inputs.push_back(input_cnode->input(i));
    }
  } else {
    real_depend_inputs.push_back(input_anfnode);
  }

  for (const auto &real_depend_input : real_depend_inputs) {
    auto real_depend_input_with_idx = AnfAlgo::VisitKernelWithReturnType(real_depend_input, 0, false, return_types);
    auto real_depend_kernel = real_depend_input_with_idx.first;
    // The monad node and make tuple node need recursion.
    if (AnfAlgo::IsOneOfPrimitiveCNode(real_depend_kernel, recursion_prims)) {
      LinkControlArrowByAutoMonad(to_actor, real_depend_kernel, graph);
      continue;
    }

    KernelActor *from_actor = nullptr;
    if (IsKernelActor(real_depend_kernel)) {
      from_actor = dynamic_cast<KernelActor *>(FetchActor(real_depend_kernel->fullname_with_scope()));
    } else if (IsInternalParameter(real_depend_kernel, graph)) {
      auto front_output_with_index = graph->GetFrontNodeByInternalParameter(real_depend_kernel);
      MS_EXCEPTION_IF_NULL(front_output_with_index.first);
      if (IsKernelActor(front_output_with_index.first)) {
        if (graph_output_to_actor_.count(front_output_with_index) == 0) {
          MS_LOG(EXCEPTION) << "Can't find actor by front node:" << front_output_with_index.first->DebugString();
        }
        from_actor = dynamic_cast<KernelActor *>(graph_output_to_actor_[front_output_with_index].first);
      }
    }
    if (from_actor == nullptr) {
      continue;
    }
    MS_LOG(INFO) << "Link control arrow by auto monad, from actor:  " << from_actor->GetAID().Name()
                 << ", to actor: " << to_actor->GetAID().Name();
    (void)from_actor->output_control_arrows_.emplace_back(to_actor->GetAID());
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
    (void)from_actor->output_control_arrows_.emplace_back(to_aid);
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
      if (input_actor != nullptr) {
        (void)input_actor->output_control_arrows_.emplace_back(from_send_actor->GetAID());
        from_send_actor->input_controls_num_++;
      }
    }

    // from_send_actor --> from_recv_actor
    (void)from_send_actor->output_control_arrows_.emplace_back(from_recv_actor->GetAID());
    from_recv_actor->input_controls_num_++;

    // from_recv_actor --> to_allreduce_actor
    (void)from_recv_actor->output_control_arrows_.emplace_back(to_allreduce_actor->GetAID());
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
    (void)from_allreduce_actor->output_control_arrows_.emplace_back(to_send_actor->GetAID());
    to_send_actor->input_controls_num_++;

    // to_send_actor --> to_recv_actor
    (void)to_send_actor->output_control_arrows_.emplace_back(to_recv_actor->GetAID());
    to_recv_actor->input_controls_num_++;

    // to_recv_actor --> outputs of from_allreduce_actor
    for (auto &output_data_arrow : from_allreduce_actor->output_data_arrows_) {
      auto output_actor = dynamic_cast<KernelActor *>(FetchActor(output_data_arrow->to_op_id_.Name()));
      if (output_actor != nullptr) {
        (void)to_recv_actor->output_control_arrows_.emplace_back(output_actor->GetAID());
        output_actor->input_controls_num_++;
      }
    }

    // In the scene of allreduce op and computing op parallel multi stream, the input memory of allreduce can be
    // reused only when the recv node runs finished, which is expressed by the reference count increased.
    for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(from_allreduce_node); ++i) {
      auto device_tensor = AnfAlgo::GetPrevNodeMutableOutputAddr(from_allreduce_node, i, false);
      MS_EXCEPTION_IF_NULL(device_tensor);
      UpdateRefCount(device_tensor.get());
      (void)to_recv_actor->external_reference_tensors_.emplace_back(device_tensor.get());
    }
  }
}

void GraphScheduler::LinkControlArrowByCommunicationNode(const std::vector<CNodePtr> &communication_nodes,
                                                         const GraphCompilerInfo &graph_compiler_info) {
  const size_t kCommunicationNodesMinNum = 2;
  if (communication_nodes.size() < kCommunicationNodesMinNum) {
    return;
  }

  // Ensure communication node to execute orderly.
  for (size_t i = 1; i < communication_nodes.size(); ++i) {
    auto from_actor = dynamic_cast<KernelActor *>(FetchActor(communication_nodes[i - 1]->fullname_with_scope()));
    auto to_actor = dynamic_cast<KernelActor *>(FetchActor(communication_nodes[i]->fullname_with_scope()));
    MS_EXCEPTION_IF_NULL(from_actor);
    MS_EXCEPTION_IF_NULL(to_actor);
    (void)from_actor->output_control_arrows_.emplace_back(to_actor->GetAID());
    to_actor->input_controls_num_++;
  }

  // Ensure all actors execute orderly to optimize the execution performance in the multi device scenario currently.
  // Using the multi stream to optimize the performance in the future.
  for (auto &graph : graph_compiler_info.graphs_) {
    auto &execution_order = graph->execution_order();
    for (size_t i = 1; i < execution_order.size(); ++i) {
      auto from_actor = dynamic_cast<KernelActor *>(FetchActor(execution_order[i - 1]->fullname_with_scope()));
      auto to_actor = dynamic_cast<KernelActor *>(FetchActor(execution_order[i]->fullname_with_scope()));
      if ((from_actor != nullptr) && (to_actor != nullptr)) {
        (void)from_actor->output_control_arrows_.emplace_back(to_actor->GetAID());
        to_actor->input_controls_num_++;
      }
    }
  }
}

void GraphScheduler::LinkControlArrowForLoopCountActor(LoopCountActor *loop_count_actor, const ActorSet *actor_set,
                                                       const ControlNodeParserPtr &parser) {
  MS_EXCEPTION_IF_NULL(actor_set);
  // There is no loop count actor in step mode.
  if (loop_count_actor == nullptr) {
    return;
  }

  // Collect the actors which have no output.
  std::vector<MemoryAwareActor *> no_output_actors;
  for (auto &kernel_actor : actor_set->kernel_actors_) {
    // The no output kernel control side in subgraph needs to be connected to the corresponding output switch actor.
    if ((kernel_actor->output_data_arrows_.size() == 0) && (kernel_actor->output_control_arrows_.size() == 0) &&
        parser->IsKernelInRootFuncGraph(kernel_actor->kernel_)) {
      MS_EXCEPTION_IF_NULL(kernel_actor->kernel_);
      MS_LOG(INFO) << kernel_actor->kernel_->fullname_with_scope() << " is not real used by other nodes.";
      (void)no_output_actors.emplace_back(kernel_actor.get());
    }
  }
  for (auto &data_actor : actor_set->data_source_actors_) {
    if ((data_actor->output_data_arrows_.size() == 0) && (data_actor->output_control_arrows_.size() == 0)) {
      (void)no_output_actors.emplace_back(data_actor.get());
    }
  }
  for (auto &copy_actor : copy_actors_) {
    if ((copy_actor->output_data_arrows_.size() == 0) && (copy_actor->output_control_arrows_.size() == 0)) {
      (void)no_output_actors.emplace_back(copy_actor.get());
    }
  }
  // No output actor --> loop count actor.
  for (auto &no_output_actor : no_output_actors) {
    (void)no_output_actor->output_control_arrows_.emplace_back(loop_count_actor->GetAID());
    loop_count_actor->input_controls_num_++;
  }

  // Loop count actor --> data source actor.
  for (auto &data_source_actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(data_source_actor);
    (void)loop_count_actor->data_source_aids_.emplace_back(data_source_actor->GetAID());
  }

  // Loop count actor --> no input kernel actor.
  for (auto &no_input_kernel_actor : actor_set->no_input_kernel_actors_) {
    MS_EXCEPTION_IF_NULL(no_input_kernel_actor);
    (void)loop_count_actor->no_input_kernel_aids_.emplace_back(no_input_kernel_actor->GetAID());
    no_input_kernel_actor->input_controls_num_++;
  }

  // Loop count actor --> output actor.
  MS_EXCEPTION_IF_NULL(actor_set->output_actor_);
  loop_count_actor->output_aid_ = actor_set->output_actor_->GetAID();
}

void GraphScheduler::LinkOutputResultArrowForOutputActor(OutputActor *to_actor,
                                                         const GraphCompilerInfo &graph_compiler_info) {
  if (graph_compiler_info.strategy_ == GraphExecutionStrategy::kStep) {
    return;
  }

  MS_EXCEPTION_IF_NULL(to_actor);

  size_t number = 0;
  for (const auto &graph : graph_compiler_info.graphs_) {
    MS_EXCEPTION_IF_NULL(graph);
    ++number;
    auto outputs = AnfAlgo::GetAllOutputWithIndex(graph->output());
    std::set<std::vector<size_t>> unique_output_positions;
    std::set<KernelWithIndex> unique_outputs;
    for (const auto &output : outputs) {
      if (IsInternalParameter(output.first, graph)) {
        MS_LOG(INFO) << "Ignore the internal parameter node:" << output.first->DebugString();
        continue;
      }
      (void)unique_outputs.insert(output);
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
      (void)unique_output_positions.insert(iter->second);
      for (auto &output_position : iter->second) {
        to_actor->device_contexts_[output_position] = graph_compiler_info.device_contexts_[number - 1];
        // The device tensor of graph out need be taken over by host tensor, so set the max reference count.
        UpdateRefCount(output_with_index.first, output_with_index.second, true);

        // The graph output is from device tensor store.
        if (IsPersistentDeviceTensor(output_with_index.first)) {
          (void)to_actor->device_tensor_store_keys_.emplace_back(output_position, output_with_index.first);
          continue;
        }

        // The graph output is from kernel actor.
        if (IsKernelActor(output_with_index.first)) {
          const auto &from_actor =
            dynamic_cast<KernelActor *>(FetchActor(output_with_index.first->fullname_with_scope()));
          MS_EXCEPTION_IF_NULL(from_actor);
          auto op_arrow = std::make_shared<DataArrow>(output_with_index.second, to_actor->GetAID(), output_position);
          (void)from_actor->output_result_arrows_.emplace_back(op_arrow);
          continue;
        }

        // The graph output is from data source actor.
        std::string actor_name;
        DataSourceActor *from_actor = nullptr;
        size_t from_actor_output_index = 0;
        if (IsHostQueueDSActor(output_with_index.first, graph, graph_compiler_info.origin_parameters_order_,
                               graph_compiler_info.strategy_)) {
          actor_name = graph_compiler_info.name_ + "_HostDSActor";
          const auto &host_queue_ds_actor = dynamic_cast<HostQueueDataSourceActor *>(FetchActor(actor_name));
          from_actor_output_index = host_queue_ds_actor->FetchDataNodePosition(output_with_index.first);
          UpdateRefCount(host_queue_ds_actor->data_nodes_[from_actor_output_index], output_with_index.second, true);
          from_actor = static_cast<DataSourceActor *>(host_queue_ds_actor);
        } else if (IsDeviceQueueDSActor(output_with_index.first, graph_compiler_info.strategy_)) {
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
        (void)from_actor->output_result_arrows_.emplace_back(op_arrow);
      }
    }
  }
}

void GraphScheduler::LinkOutputResultArrowForSwitchActor(const GraphCompilerInfo &graph_compiler_info,
                                                         const ActorSet *actor_set) {
  const auto &to_actor = actor_set->output_actor_;
  const auto &loop_count_actor = actor_set->loop_count_actor_;
  if (to_actor == nullptr || loop_count_actor == nullptr) {
    return;
  }

  // When there is a call node in the output, the output will be sent to the output actor by the switch actor of
  // the funcgraph called by the call node.
  const auto &outputs = graph_compiler_info.origin_outputs_order_;
  for (const auto &output : outputs) {
    const auto &output_node = output.first.first;
    const auto &output_index = output.first.second;
    const auto output_poses = output.second;

    if (IsCallNode(output_node)) {
      const auto &func_graphs = FetchFuncGraphbyCallNode(output_node);
      for (const auto func_graph : func_graphs) {
        const auto &actor_name = func_graph->get_return()->DebugString();
        auto actor = FetchActor(actor_name);
        MS_EXCEPTION_IF_NULL(actor);
        auto switch_actor = dynamic_cast<SwitchActor *>(actor);
        MS_EXCEPTION_IF_NULL(switch_actor);

        // Set branch index into switch actor.
        size_t branch_index = switch_actor->branch_id_to_index_.size();
        if (switch_actor->branch_id_to_index_.find(kMainBranchID) != switch_actor->branch_id_to_index_.end()) {
          branch_index = switch_actor->branch_id_to_index_[kMainBranchID];
        } else {
          switch_actor->branch_id_to_index_[kMainBranchID] = branch_index;
        }

        // Link output result arrow.
        for (const auto output_pos : output_poses) {
          auto op_arrow = std::make_shared<DataArrow>(output_index, to_actor->GetAID(), output_pos);
          to_actor->device_contexts_[output_pos] = switch_actor->device_context_;
          (void)switch_actor->output_branch_result_arrows_[branch_index].emplace_back(op_arrow);
        }
      }
    }
  }

  const auto &switch_actors = actor_set->switch_actors_;
  for (const auto &from_actor : switch_actors) {
    MS_EXCEPTION_IF_NULL(from_actor);
    auto origin_output_with_index = KernelWithIndex(from_actor->node_, 0);
    const auto &iter = graph_compiler_info.origin_outputs_order_.find(origin_output_with_index);
    if (iter == graph_compiler_info.origin_outputs_order_.end()) {
      continue;
    }

    // If the switch actor is in the output list, the output of switch actor should be sent to the output actor.
    // And need to link a control arrow to the loop count actor.
    for (const auto pos : iter->second) {
      to_actor->device_contexts_[pos] = from_actor->device_context_;
    }

    for (size_t i = 0; i < from_actor->branch_inputs_pos_.size(); ++i) {
      const auto &input_pos = from_actor->branch_inputs_pos_[i];
      if (input_pos.empty()) {
        MS_LOG(EXCEPTION) << "Invalid input num in switch actor:" << from_actor->GetAID();
      }

      for (const auto pos : iter->second) {
        auto op_arrow = std::make_shared<DataArrow>(0, to_actor->GetAID(), pos);
        (void)from_actor->output_branch_result_arrows_[i].emplace_back(op_arrow);
      }

      (void)from_actor->output_branch_control_arrows_[i].emplace_back(loop_count_actor->GetAID());
    }
    loop_count_actor->input_controls_num_++;
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
      if (FetchActor(name) != nullptr) {
        continue;
      }
      auto copy_actor = std::make_shared<CopyActor>(name, memory_manager_aid_);
      MS_EXCEPTION_IF_NULL(copy_actor);
      (void)copy_actors_.emplace_back(copy_actor);
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

      MS_LOG(INFO) << "The kernel actor: " << kernel_actor->GetAID().Name()
                   << "has control arrows number:" << kernel_actor->output_control_arrows_.size();
      // Link from copy actor to kernel actor users.
      for (auto &output_contorl : kernel_actor->output_control_arrows_) {
        (void)copy_actor->output_control_arrows_.emplace_back(output_contorl);
      }
      // Move the control arrows from kernel actor to kernel actor users.
      kernel_actor->output_control_arrows_.clear();

      // Link from kernel actor to copy actor.
      (void)kernel_actor->output_control_arrows_.emplace_back(copy_actor->GetAID());
      copy_actor->input_controls_num_++;
    }
  }
}

void GraphScheduler::PrepareInputNodeForSwitchActor(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &node : control_nodes) {
    CNodePtr cnode = node->cast<CNodePtr>();
    auto inputs = cnode->inputs();
    // Before link data arrow, parameters of the call node in switch-call need to be add to the switch actor.
    if (inputs[0]->isa<CNode>()) {
      auto actor = FetchActor(inputs[0]->DebugString());
      MS_EXCEPTION_IF_NULL(actor);
      auto switch_actor = dynamic_cast<SwitchActor *>(actor);
      MS_EXCEPTION_IF_NULL(switch_actor);

      for (size_t i = kCallInputStartPos; i < inputs.size(); ++i) {
        if (HasAbstractMonad(inputs[i])) {
          continue;
        }
        switch_actor->AddCommonInput(inputs[i]);
      }
    }
  }
}

void GraphScheduler::LinkArrowByControlNode(const GraphCompilerInfo &graph_compiler_info, ActorSet *actor_set) {
  PrepareInputNodeForSwitchActor(graph_compiler_info.control_nodes_);

  for (const auto &node : graph_compiler_info.control_nodes_) {
    CNodePtr cnode = node->cast<CNodePtr>();
    const auto &from_func_graph = node->func_graph();
    auto inputs = cnode->inputs();
    // Link data arrow for switch node.
    if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimSwitch) ||
        AnfAlgo::CheckPrimitiveType(node, prim::kPrimSwitchLayer)) {
      auto actor = actor_name_to_actor_[node->DebugString()];
      MS_EXCEPTION_IF_NULL(actor);
      auto switch_actor = dynamic_cast<SwitchActor *>(actor);
      MS_EXCEPTION_IF_NULL(switch_actor);
      LinkDataArrowForSwitchActor(graph_compiler_info, switch_actor);
    } else if (inputs[0]->isa<ValueNode>() && IsValueNode<FuncGraph>(inputs[0])) {
      // Link the data arrow for the input of the call node.
      const auto &actor_name = node->DebugString();
      auto actor = FetchActor(actor_name);
      MS_EXCEPTION_IF_NULL(actor);
      auto gather_actor = dynamic_cast<GatherActor *>(actor);
      MS_EXCEPTION_IF_NULL(gather_actor);

      const auto &func_graph = GetValueNode<FuncGraphPtr>(inputs[0]);
      MS_EXCEPTION_IF_NULL(func_graph);
      const auto &to_actor = FetchActor(func_graph->ToString());
      MS_EXCEPTION_IF_NULL(to_actor);

      size_t persist_input_num = 0;
      for (size_t i = kCallInputStartPos; i < inputs.size(); ++i) {
        MS_EXCEPTION_IF_NULL(actor);
        if (inputs[i]->isa<ValueNode>()) {
          const auto &node_value = inputs[i]->cast<ValueNodePtr>()->value();
          if (!node_value->isa<tensor::Tensor>()) {
            persist_input_num++;
            continue;
          }

          (void)gather_actor->device_tensor_store_keys_.emplace_back(i - kCallInputStartPos - persist_input_num,
                                                                     inputs[i].get());
          gather_actor->device_contexts_[i - kCallInputStartPos - persist_input_num] =
            graph_compiler_info.control_node_parser_->GetFrontValueNodeDeviceContext(inputs[i]);
        } else if ((inputs[i]->isa<Parameter>() && HasAbstractRef(inputs[i]->cast<ParameterPtr>())) ||
                   AnfAlgo::CheckPrimitiveType(inputs[i], prim::kPrimUpdateState) || HasAbstractMonad(inputs[i])) {
          persist_input_num++;
          continue;
        } else {
          const auto &input_with_index = AnfAlgo::VisitKernelWithReturnType(inputs[i], 0);
          LinkDataArrowByControlNode(graph_compiler_info, input_with_index, from_func_graph, actor,
                                     i - kCallInputStartPos - persist_input_num);
        }

        auto op_arrow = std::make_shared<DataArrow>(i - kCallInputStartPos - persist_input_num, to_actor->GetAID(),
                                                    i - kCallInputStartPos - persist_input_num);
        (void)gather_actor->output_data_arrows_.emplace_back(op_arrow);
      }
    }
  }

  // Link arrow for switch actor of subgraph output.
  for (const auto &func_graph_with_call_num : graph_compiler_info.control_node_parser_->func_graph_to_call_num_) {
    const auto &func_graph = func_graph_with_call_num.first;
    MS_EXCEPTION_IF_NULL(func_graph);
    auto actor = FetchActor(func_graph->get_return()->DebugString());
    MS_EXCEPTION_IF_NULL(actor);
    auto switch_actor = dynamic_cast<SwitchActor *>(actor);
    MS_EXCEPTION_IF_NULL(switch_actor);
    LinkDataArrowForSwitchActor(graph_compiler_info, switch_actor);
  }

  // Link arrow for gather actor for call input kernel graph.
  for (const auto &call_input_kernel_graph : graph_compiler_info.control_node_parser_->call_input_kernel_graphs_) {
    const auto &kernel_graph = call_input_kernel_graph.first;
    MS_EXCEPTION_IF_NULL(kernel_graph);
    auto actor = FetchActor(kernel_graph->ToString());
    MS_EXCEPTION_IF_NULL(actor);
    auto gather_actor = dynamic_cast<GatherActor *>(actor);
    MS_EXCEPTION_IF_NULL(gather_actor);

    for (size_t i = 0; i < gather_actor->data_nodes_.size(); ++i) {
      const auto &input_with_index = gather_actor->data_nodes_[i];
      const auto &from_func_graph = kernel_graph->GetFuncGraph();
      LinkDataArrowByControlNode(graph_compiler_info, input_with_index, from_func_graph, gather_actor, i);
    }
  }
  LinkBranchArrowForSwitchActor(graph_compiler_info);

  LinkBranchArrowForGatherActor(graph_compiler_info);

  LinkControlArrowForGatherActor(&(actor_set->kernel_actors_), graph_compiler_info.graphs_,
                                 graph_compiler_info.control_node_parser_);

  LinkControlArrowForSwitchActor(&(actor_set->switch_actors_), actor_set->loop_count_actor_.get(),
                                 graph_compiler_info.origin_outputs_order_);

  LinkOutputResultArrowForSwitchActor(graph_compiler_info, actor_set);
}

void GraphScheduler::LinkDataArrowForGatherActor(GatherActor *from_actor, KernelActor *to_actor,
                                                 const KernelWithIndex &front_node_with_index,
                                                 const KernelWithIndex &to_node_with_index) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(front_node_with_index.first);

  auto position = from_actor->FetchDataNodePosition(front_node_with_index);

  auto op_arrow = std::make_shared<DataArrow>(position, to_actor->GetAID(), to_node_with_index.second);
  (void)from_actor->output_data_arrows_.emplace_back(op_arrow);
  to_actor->input_datas_num_++;
}

void GraphScheduler::LinkDataArrowByCallInput(const KernelWithIndex &call_node_with_index,
                                              const ControlNodeParserPtr &parser, const FuncGraphPtr &from_func_graph,
                                              OpActor<DeviceTensor> *to_actor, const size_t to_index) {
  // Fetch all the funcgraph that call node would call.
  const auto cnode = call_node_with_index.first->cast<CNodePtr>();
  std::vector<FuncGraphPtr> func_graphs = FetchFuncGraphbyCallNode(cnode);

  // Collect the output of each funcgraph.
  for (const auto &func_graph : func_graphs) {
    const auto actor_name = func_graph->get_return()->DebugString();
    auto actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    auto switch_actor = dynamic_cast<SwitchActor *>(actor);
    MS_EXCEPTION_IF_NULL(switch_actor);
    const size_t branch_index = switch_actor->branch_id_to_index_.size();

    const auto &func_graph_to_branch_id = parser->func_graph_to_branch_id_;
    const auto &iter = func_graph_to_branch_id.find(from_func_graph);

    int branch_id = kMainBranchID;
    if (iter != func_graph_to_branch_id.end()) {
      branch_id = iter->second;
    }
    if (switch_actor->branch_id_to_index_.find(branch_id) != switch_actor->branch_id_to_index_.end()) {
      LinkDataArrowForSwitchActor(switch_actor, call_node_with_index.second, to_actor, to_index,
                                  switch_actor->branch_id_to_index_[branch_id]);
      continue;
    }
    LinkDataArrowForSwitchActor(switch_actor, call_node_with_index.second, to_actor, to_index, branch_index);
    switch_actor->branch_id_to_index_[branch_id] = branch_index;
  }
}

void GraphScheduler::LinkDataArrowForSwitchActor(SwitchActor *from_actor, const size_t from_index,
                                                 OpActor<DeviceTensor> *to_actor, const size_t to_index,
                                                 const size_t branch_index) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  size_t start_branch = 0;
  size_t max_branch = from_actor->output_branch_arrows_.size();
  if (branch_index != SIZE_MAX) {
    start_branch = branch_index;
    max_branch = branch_index + 1;
  }
  for (size_t i = start_branch; i < max_branch; ++i) {
    if (from_actor->branch_inputs_pos_[i].size() <= from_index) {
      MS_LOG(EXCEPTION) << "No input for switch actor:" << from_actor->GetAID() << " branch:" << i
                        << " from index:" << from_index << " output size:" << from_actor->branch_inputs_pos_[i].size()
                        << " to actor:" << to_actor->GetAID() << " to index:" << to_index;
    }
    auto op_arrow =
      std::make_shared<DataArrow>(from_actor->branch_inputs_pos_[i][from_index], to_actor->GetAID(), to_index);
    (void)from_actor->output_branch_arrows_[i].emplace_back(op_arrow);
  }
}

void GraphScheduler::LinkDataArrowByControlNode(const GraphCompilerInfo &graph_compiler_info,
                                                const KernelWithIndex &input_with_index,
                                                const FuncGraphPtr &from_func_graph, OpActor<DeviceTensor> *to_actor,
                                                const size_t to_index) {
  const auto &parameters = graph_compiler_info.origin_parameters_order_;
  const auto &front_to_backend_parameter = graph_compiler_info.control_node_parser_->front_to_backend_parameters_;
  const auto &input_node = input_with_index.first;

  if (IsCallNode(input_node)) {
    // The actor input is a call node.
    LinkDataArrowByCallInput(input_with_index, graph_compiler_info.control_node_parser_, from_func_graph, to_actor,
                             to_index);
  } else if (IsGatherActor(input_node, actor_name_to_actor_)) {
    // The actor input is a parameter in gather actor.
    auto from_actor = dynamic_cast<GatherActor *>(actor_name_to_actor_[input_node->func_graph()->ToString()]);
    auto position = from_actor->FetchDataNodePosition({input_node, 0});
    auto op_arrow = std::make_shared<DataArrow>(position, to_actor->GetAID(), to_index);
    (void)from_actor->output_data_arrows_.emplace_back(op_arrow);
  } else if (IsSwitchActor(input_node)) {
    const auto &actor_name = input_node->DebugString();
    auto actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    LinkDataArrowForSwitchActor(dynamic_cast<SwitchActor *>(actor), 0, to_actor, to_index);
  } else if (IsKernelActor(input_node, graph_compiler_info.strategy_)) {
    // The actor input is a cnode.
    if (front_node_to_actor_.find(input_node) == front_node_to_actor_.end()) {
      const auto &kernel_with_index = AnfAlgo::VisitKernelWithReturnType(input_node, 0);
      const auto &backend_node =
        graph_compiler_info.control_node_parser_->GetBackendKernelByFrontKernel(kernel_with_index);
      if (backend_node.first == nullptr) {
        MS_LOG(EXCEPTION) << "Cannot find actor:" << to_actor->GetAID()
                          << " input_node:" << AnfAlgo::GetNodeDebugString(input_node) << " addr:" << input_node;
      }
      const auto &actor_name = backend_node.first->fullname_with_scope();
      const auto &actor = FetchActor(actor_name);
      MS_EXCEPTION_IF_NULL(actor);
      auto from_actor = dynamic_cast<KernelActor *>(actor);
      MS_EXCEPTION_IF_NULL(from_actor);

      auto op_arrow = std::make_shared<DataArrow>(backend_node.second, to_actor->GetAID(), to_index);
      (void)from_actor->output_data_arrows_.emplace_back(op_arrow);
      auto device_tensor = AnfAlgo::GetMutableOutputAddr(from_actor->kernel_, backend_node.second, false);
      UpdateRefCount(device_tensor.get(), true);
      return;
    }

    auto op_arrow = std::make_shared<DataArrow>(input_with_index.second, to_actor->GetAID(), to_index);
    auto from_actor = front_node_to_actor_[input_node];
    (void)from_actor->output_data_arrows_.emplace_back(op_arrow);
    auto device_tensor = AnfAlgo::GetMutableOutputAddr(from_actor->kernel_, input_with_index.second, false);
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
      MS_LOG(EXCEPTION) << "Cannot find data node in data source actor, backend node:"
                        << AnfAlgo::GetNodeDebugString(backend_node)
                        << " front node:" << AnfAlgo::GetNodeDebugString(input_node);
    }

    auto op_arrow = std::make_shared<DataArrow>(iter->second, to_actor->GetAID(), to_index);
    (void)from_actor->output_data_arrows_.emplace_back(op_arrow);
    auto device_tensor = AnfAlgo::GetMutableOutputAddr(from_actor->data_nodes_[iter->second], 0, false);
    UpdateRefCount(device_tensor.get(), true);
  } else {
    MS_LOG(EXCEPTION) << "Cannot find actor of switch input_node:" << AnfAlgo::GetNodeDebugString(input_node)
                      << " to actor:" << to_actor->GetAID();
  }
}

void GraphScheduler::LinkDataArrowForSwitchActor(const GraphCompilerInfo &graph_compiler_info, SwitchActor *actor) {
  // Link switch input.
  const auto &inputs = actor->input_nodes_;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto input = inputs[i];
    if (input.first->isa<ValueNode>() || (input.first->isa<Parameter>() && HasAbstractRef(input.first))) {
      continue;
    }

    const FuncGraphPtr from_func_graph = actor->node_->func_graph();
    LinkDataArrowByControlNode(graph_compiler_info, input, from_func_graph, actor, i);
  }

  // Link switch output.
  for (size_t i = 0; i < actor->branch_func_graph_.size(); ++i) {
    auto func_graph = actor->branch_func_graph_[i];
    if (func_graph == nullptr) {
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
      (void)actor->output_branch_arrows_[i].emplace_back(op_arrow);
    }
  }
}

void GraphScheduler::LinkControlArrowForGatherActor(std::vector<KernelActorPtr> *kernel_actors,
                                                    const std::vector<KernelGraphPtr> &graphs,
                                                    const ControlNodeParserPtr &parser) {
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
          (void)gather_actor->output_control_arrows_.emplace_back(kernel_actor->GetAID());
          kernel_actor->input_controls_num_ = 1;
        }
      }
    }
  }

  for (auto &kernel_actor : *kernel_actors) {
    MS_EXCEPTION_IF_NULL(kernel_actor);

    if ((kernel_actor->output_data_arrows_.size() == 0) && (kernel_actor->output_control_arrows_.size() == 0) &&
        !parser->IsKernelInRootFuncGraph(kernel_actor->kernel_)) {
      // Check whether the kernel actor belongs to the root graph.
      // In general, all no output nodes belong to the root funcgraph, and the corresponding switch actor for output
      // should be empty. In control flow, the control arrow of the no output node in the sub funcgraph should be
      // sent to the output switch actor.
      const auto &graph = kernel_actor->kernel_->func_graph();
      OpActor<DeviceTensor> *actor = nullptr;

      if (graph != nullptr) {
        const auto &kernel_graph = dynamic_cast<KernelGraph *>(graph.get());
        const auto func_graph = kernel_graph->GetFuncGraph();
        if (func_graph != nullptr) {
          actor = FetchActor(func_graph->get_return()->DebugString());
          if (actor != nullptr) {
            auto switch_actor = dynamic_cast<SwitchActor *>(actor);
            MS_EXCEPTION_IF_NULL(switch_actor);

            (void)kernel_actor->output_control_arrows_.emplace_back(switch_actor->GetAID());
            switch_actor->input_controls_num_++;
          }
        }
      }
    }
  }

  // Link input auto monad control arrow from kernel actor to gather actor.
  const auto &monad_nodes = parser->kernel_to_call_nodes_;
  for (const auto node_pair : monad_nodes) {
    const auto &kernel_actor_name = node_pair.first->fullname_with_scope();
    const auto &gather_actor_name = node_pair.second->DebugString();
    auto kernel_op_actor = FetchActor(kernel_actor_name);
    auto gather_op_actor = FetchActor(gather_actor_name);
    if (kernel_op_actor == nullptr || gather_op_actor == nullptr) {
      continue;
    }
    auto kernel_actor = dynamic_cast<KernelActor *>(kernel_op_actor);
    auto gather_actor = dynamic_cast<GatherActor *>(gather_op_actor);
    (void)kernel_actor->output_control_arrows_.emplace_back(gather_actor->GetAID());
    gather_actor->input_controls_num_++;
  }
}

void GraphScheduler::LinkControlArrowForSwitchActor(std::vector<SwitchActorPtr> *switch_actors,
                                                    LoopCountActor *to_actor,
                                                    const KernelMapPosition &origin_outputs_order) {
  if (to_actor == nullptr || (*switch_actors).empty()) {
    return;
  }

  // If there is no output from the switch actor branch, it means that the subgraph has no input,
  // and need to connect a control arrow to the corresponding gather actor.
  for (auto &switch_actor : (*switch_actors)) {
    if (AnfAlgo::CheckPrimitiveType(switch_actor->node_, prim::kPrimReturn)) {
      const auto &func_graph = switch_actor->node_->func_graph();
      if (func_graph->output()->isa<ValueNode>()) {
        const auto &actor_name = func_graph->ToString();
        auto actor = FetchActor(actor_name);
        MS_EXCEPTION_IF_NULL(actor);
        auto gather_actor = dynamic_cast<GatherActor *>(actor);
        MS_EXCEPTION_IF_NULL(gather_actor);
        (void)gather_actor->output_control_arrows_.emplace_back(switch_actor->GetAID());
        switch_actor->input_controls_num_++;
      }
    }

    for (size_t i = 0; i < switch_actor->output_branch_arrows_.size(); ++i) {
      const auto &arrows = switch_actor->output_branch_arrows_[i];
      if (arrows.empty() && switch_actor->branch_func_graph_[i] != nullptr) {
        const auto &actor_name = switch_actor->branch_func_graph_[i]->ToString();
        const auto &actor = FetchActor(actor_name);
        if (actor != nullptr) {
          const auto &gather_actor = dynamic_cast<GatherActor *>(actor);
          MS_EXCEPTION_IF_NULL(gather_actor);
          (void)switch_actor->output_branch_control_arrows_[i].emplace_back(gather_actor->GetAID());
          gather_actor->input_controls_num_++;
        }
      }
    }
  }

  // Collect all the call node in outputs.
  std::set<AnfNodePtr> call_nodes;
  for (const auto &output : origin_outputs_order) {
    if (IsCallNode(output.first.first)) {
      (void)call_nodes.insert(output.first.first);
    }
  }
  to_actor->input_controls_num_ += call_nodes.size();

  // Link the output switch actor of the subgraph to the output actor.
  for (const auto &call_node : call_nodes) {
    const auto &func_graphs = FetchFuncGraphbyCallNode(call_node);
    for (const auto func_graph : func_graphs) {
      MS_EXCEPTION_IF_NULL(func_graph);
      const auto &actor_name = func_graph->get_return()->DebugString();
      auto actor = FetchActor(actor_name);
      MS_EXCEPTION_IF_NULL(actor);
      auto switch_actor = dynamic_cast<SwitchActor *>(actor);
      MS_EXCEPTION_IF_NULL(switch_actor);

      size_t branch_index = switch_actor->branch_id_to_index_.size();
      if (switch_actor->branch_id_to_index_.find(kMainBranchID) != switch_actor->branch_id_to_index_.end()) {
        branch_index = switch_actor->branch_id_to_index_[kMainBranchID];
      } else {
        switch_actor->branch_id_to_index_[kMainBranchID] = branch_index;
      }

      (void)switch_actor->output_branch_control_arrows_[branch_index].emplace_back(to_actor->GetAID());
    }
  }
}

void GraphScheduler::LinkBranchArrowForSwitchActor(const GraphCompilerInfo &graph_compiler_info) {
  for (const auto &control_node : graph_compiler_info.control_nodes_) {
    if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitch) ||
        AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitchLayer)) {
      const auto &actor_name = control_node->DebugString();
      auto actor = FetchActor(actor_name);
      MS_EXCEPTION_IF_NULL(actor);
      auto switch_actor = dynamic_cast<SwitchActor *>(actor);
      MS_EXCEPTION_IF_NULL(switch_actor);

      for (size_t i = 0; i < switch_actor->branch_func_graph_.size(); ++i) {
        const auto &func_graph = switch_actor->branch_func_graph_[i];
        if (func_graph == nullptr) {
          continue;
        }

        const auto &gather_actor = FetchActor(func_graph->ToString());
        MS_EXCEPTION_IF_NULL(gather_actor);
        (void)switch_actor->output_branch_branch_arrows_[i].emplace_back(gather_actor->GetAID());
      }
    }
  }
}

void GraphScheduler::LinkBranchArrowForGatherActor(const GraphCompilerInfo &graph_compiler_info) {
  if (graph_compiler_info.control_nodes_.empty()) {
    return;
  }

  // Link branch arrow from gather actor to gather actor.
  for (const auto &control_node : graph_compiler_info.control_nodes_) {
    const auto &cnode = control_node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    if (inputs[0]->isa<ValueNode>() && IsValueNode<FuncGraph>(inputs[0])) {
      const auto &actor_name = control_node->DebugString();
      auto actor = FetchActor(actor_name);
      MS_EXCEPTION_IF_NULL(actor);
      auto gather_actor = dynamic_cast<GatherActor *>(actor);
      MS_EXCEPTION_IF_NULL(gather_actor);
      (void)gather_actor->output_branch_arrows_.emplace_back(gather_actor->gather_aid_);
    }
  }

  // Link branch arrow from gather actor to switch actor.
  for (const auto &func_graph_with_call_num : graph_compiler_info.control_node_parser_->func_graph_to_call_num_) {
    const auto &actor_name = func_graph_with_call_num.first->ToString();
    auto actor = FetchActor(actor_name);
    MS_EXCEPTION_IF_NULL(actor);
    auto gather_actor = dynamic_cast<GatherActor *>(actor);
    MS_EXCEPTION_IF_NULL(gather_actor);
    (void)gather_actor->output_branch_arrows_.emplace_back(gather_actor->switch_aid_);
  }
}

bool GraphScheduler::CheckActorValid(const ActorSet *actor_set, GraphExecutionStrategy strategy) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  // Check the data source actors.
  for (const auto &data_source_actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(data_source_actor);
    if (data_source_actor->output_data_arrows_.size() + data_source_actor->output_result_arrows_.size() +
          data_source_actor->output_control_arrows_.size() ==
        0) {
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
      MS_LOG(ERROR) << "The input building of " << AnfAlgo::GetNodeDebugString(kernel_actor->kernel_)
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
    size_t device_tensor_store_num = (copy_actor->device_tensor_store_key_.second == nullptr) ? 0 : 1;
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
    if (loop_count_actor->input_controls_num_ == 0) {
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
      AnfNodePtr sub_front_node = nullptr;
      if (IsInternalParameter(input_node, graph)) {
        auto front_output_with_index = graph->GetFrontNodeByInternalParameter(input_node);
        sub_front_node = front_output_with_index.first;
      } else if (IsPersistentDeviceTensor(input_node) || HasAbstractRef(input_node)) {
        sub_front_node = FetchFrontNodeByBackendNode(input_node, graph);
      }
      if (sub_front_node == nullptr) {
        continue;
      }

      // The sub front nodes share the device tensor store with the root front node.
      auto front_node = sub_front_node;
      if (graph_compiler_info.control_node_parser_ != nullptr) {
        front_node = graph_compiler_info.control_node_parser_->FetchRootGraphFrontNodeBySubFrontNode(sub_front_node);
      }
      MS_LOG(DEBUG) << "Graph id:" << graph->graph_id() << ", sub front node:" << sub_front_node->DebugString()
                    << ", root front node:" << front_node->DebugString();
      auto device_tensor = AnfAlgo::GetMutableOutputAddr(input_node, 0, false);
      MS_EXCEPTION_IF_NULL(device_tensor);
      if (IsPersistentDeviceTensor(input_node)) {
        DeviceTensorStore::GetInstance().Insert(front_node.get(), device_tensor);
        UpdateRefCount(device_tensor.get(), true);
      }

      // Share the weight in the host and device, then input_node is internal parameter and front_node is weight.
      if (!IsPersistentDeviceTensor(front_node)) {
        continue;
      }
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

bool GraphScheduler::IsHostQueueDSActor(const AnfNodePtr &node, const KernelGraphPtr &graph,
                                        const std::vector<AnfNodePtr> &host_parameters,
                                        GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(node);

  bool is_parameter_data = node->isa<Parameter>() && (!AnfAlgo::IsParameterWeight(node->cast<ParameterPtr>()));
  if (!is_parameter_data) {
    return false;
  }

  if (strategy == GraphExecutionStrategy::kStep) {
    MS_EXCEPTION_IF_NULL(graph);
    return graph->execution_order().size() > 1;
  }

  if (graph == nullptr) {
    return true;
  }

  // In control flow, only the parameters of the root funcgraph are in the host data source.
  const auto &front_node = graph->GetFrontAnfByBackendAnf(node);
  bool is_host = ((front_node == nullptr) || host_parameters.empty() ||
                  find(host_parameters.begin(), host_parameters.end(), front_node) != host_parameters.end());

  //  Judge whether node is internal parameter.
  const auto &internal_front_node = graph->GetFrontNodeByInternalParameter(node);
  if (internal_front_node.first == nullptr && is_host) {
    return true;
  }

  return false;
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

  ofs << "\n\n[Gather actors]\n";
  for (const auto &gather_actor : actor_set->gather_actors_) {
    DumpGatherActor(gather_actor.get(), ofs);
  }

  ofs << "\n\n[Switch actors]\n";
  for (const auto &switch_actor : actor_set->switch_actors_) {
    DumpSwitchActor(switch_actor.get(), ofs);
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
      << "\tinput_controls_num:" << actor->input_controls_num_ << "\n";

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

  ofs << "\t\tdevice_tensor_store_keys:" << actor->device_tensor_store_keys_.size() << "\n ";
  for (const auto &device_tensor_store_key : actor->device_tensor_store_keys_) {
    MS_EXCEPTION_IF_NULL(device_tensor_store_key.second);
    ofs << "\t\t\toutput_node_position:" << device_tensor_store_key.first
        << "\toutput_node_name:" << device_tensor_store_key.second->fullname_with_scope() << "\n";
  }

  ofs << "\t\tdevice_contexts:" << actor->device_contexts_.size() << "\n ";
  for (const auto &device_context : actor->device_contexts_) {
    if (device_context == nullptr) {
      ofs << "\t\t\tdevice_context:" << device_context << "\n";
      continue;
    }
    ofs << "\t\t\tdevice_context:" << device_context->device_context_key().ToString() << "\n";
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
      MS_EXCEPTION_IF_NULL(front_node);
      const auto device_tensors = DeviceTensorStore::GetInstance().Fetch(front_node.get());
      ofs << "\t\tdevcie tensor key:" << front_node->DebugString() << "\tvalue size:" << device_tensors.size() << "\n";
      for (const auto &device_tensor : device_tensors) {
        MS_EXCEPTION_IF_NULL(device_tensor);
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
      const auto &sub_front_node = FetchFrontNodeByBackendNode(input_node, graph);
      // The sub front nodes share the device tensor store with the root front node.
      auto front_node = sub_front_node;
      if (graph_compiler_info.control_node_parser_ != nullptr) {
        front_node = graph_compiler_info.control_node_parser_->FetchRootGraphFrontNodeBySubFrontNode(sub_front_node);
      }
      const auto device_tensors = DeviceTensorStore::GetInstance().Fetch(front_node.get());
      MS_EXCEPTION_IF_NULL(front_node);
      ofs << "\t\tdevcie tensor key:" << front_node->DebugString() << "\tvalue size:" << device_tensors.size() << "\n";
      for (const auto &device_tensor : device_tensors) {
        MS_EXCEPTION_IF_NULL(device_tensor);
        ofs << "\t\t\tdevcie tensor value:" << device_tensor << "\tptr:" << device_tensor->GetPtr()
            << "\tsize:" << device_tensor->GetSize() << "\toriginal_ref_count:" << device_tensor->original_ref_count()
            << "\tdevice_type:" << device_tensor->DeviceType() << "\n ";
      }
    }
    ofs << "\n";
  }
}

void GraphScheduler::DumpGatherActor(const GatherActor *actor, std::ofstream &ofs) const {
  MS_EXCEPTION_IF_NULL(actor);
  ofs << "\tactor_name:" << actor->GetAID().Name() << '\n';

  ofs << "\t\tactor input num:" << actor->data_nodes_.size() << "\n";
  for (const auto &node : actor->data_nodes_) {
    ofs << "\t\t\t" << AnfAlgo::GetNodeDebugString(node.first) << "\tindex:" << node.second << '\n';
  }

  ofs << "\t\tactor front to backend node:\n";
  for (const auto &front_to_backend_parameter : actor->front_to_backend_parameter_) {
    ofs << "\t\t\tfront node:" << AnfAlgo::GetNodeDebugString(front_to_backend_parameter.first) << '\n';
    for (const auto node_with_index : front_to_backend_parameter.second) {
      ofs << "\t\t\t\tbackend node:" << AnfAlgo::GetNodeDebugString(node_with_index.first)
          << "\tindex:" << node_with_index.second << '\n';
    }
  }

  ofs << "\t\tactor output data arrow:\n";
  for (const auto &data_arrow : actor->output_data_arrows_) {
    MS_EXCEPTION_IF_NULL(data_arrow);
    ofs << "\t\t\tfrom_output_index:" << data_arrow->from_output_index_
        << "\tto_actor_name:" << data_arrow->to_op_id_.Name() << "\tto_input_index:" << data_arrow->to_input_index_
        << "\n";
  }

  ofs << "\t\tactor output result arrow:\n";
  for (const auto &result_arrow : actor->output_result_arrows_) {
    MS_EXCEPTION_IF_NULL(result_arrow);
    ofs << "\t\t\tfrom_output_index:" << result_arrow->from_output_index_
        << "\tto_actor_name:" << result_arrow->to_op_id_.Name() << "\tto_input_index:" << result_arrow->to_input_index_
        << "\n";
  }

  ofs << "\t\tactor output control arrow:\n";
  for (const auto &control_arrow : actor->output_control_arrows_) {
    ofs << "\t\t\tto_actor_name:" << control_arrow;
  }
}

void GraphScheduler::DumpSwitchActor(const SwitchActor *actor, std::ofstream &ofs) const {
  MS_EXCEPTION_IF_NULL(actor);
  ofs << "\tactor_name:" << actor->GetAID().Name() << '\n';

  ofs << "\t\tactor input num:" << actor->input_nodes_.size() << "\n";
  for (const auto &node : actor->input_nodes_) {
    ofs << "\t\t\t" << AnfAlgo::GetNodeDebugString(node.first) << '\t' << node.second << '\n';
  }

  ofs << "\t\tactor input pos:\n";
  for (size_t i = 0; i < actor->branch_inputs_pos_.size(); ++i) {
    ofs << "\t\t\tbranch " << i << " input pos:";
    for (const auto pos : actor->branch_inputs_pos_[i]) {
      ofs << pos << '\t';
    }
    ofs << '\n';
  }

  ofs << "\t\tactor output data arrow:\n";
  for (size_t i = 0; i < actor->output_branch_arrows_.size(); ++i) {
    ofs << "\t\t\tbranch " << i << " output data:\n";
    for (const auto arrow : actor->output_branch_arrows_[i]) {
      MS_EXCEPTION_IF_NULL(arrow);
      ofs << "\t\t\t\t from index:" << arrow->from_output_index_ << "\tto_actor_name:" << arrow->to_op_id_
          << "\tto_input_index:" << arrow->to_input_index_ << '\n';
    }
  }

  ofs << "\t\tactor output result arrow:\n";
  for (size_t i = 0; i < actor->output_branch_result_arrows_.size(); ++i) {
    ofs << "\t\t\tbranch " << i << " output result:\n";
    for (const auto arrow : actor->output_branch_result_arrows_[i]) {
      MS_EXCEPTION_IF_NULL(arrow);
      ofs << "\t\t\t\t from index:" << arrow->from_output_index_ << "\tto_actor_name:" << arrow->to_op_id_
          << "\tto_input_index:" << arrow->to_input_index_ << '\n';
    }
  }

  ofs << "\t\tactor output control arrow:\n";
  for (size_t i = 0; i < actor->output_branch_control_arrows_.size(); ++i) {
    ofs << "\t\t\tbranch " << i << " output control:\n";
    for (const auto arrow : actor->output_branch_control_arrows_[i]) {
      ofs << "\t\t\t\t from index:" << arrow << '\n';
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
