/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "backend/ge_backend/runtime/actor/super_kernel_actor.h"
#include "backend/ge_backend/runtime/scheduler_helper.h"
#include "backend/ge_backend/runtime/actor/output_actor.h"
#include "backend/ge_backend/runtime/actor/memory_manager_actor.h"
#include "backend/ge_backend/runtime/actor/debug_actor.h"
#include "include/runtime/hardware_abstract/stream/multi_stream_controller.h"
#include "async/async.h"
#include "utils/llm_manager.h"
#include "utils/log_adapter.h"
#include "primitive/framework_ops.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
namespace {
inline void UpdateShape(const AnfNodePtr &input_node, const KernelTensorPtr &node_device_kernel_tensor,
                        const KernelTensorPtr &input_kernel_tensor, const KernelTransformType &type) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(node_device_kernel_tensor);
  MS_EXCEPTION_IF_NULL(input_kernel_tensor);
  if (type != KernelTransformType::kSuperKernelActor || input_node->cast<ParameterPtr>()->has_dynamic_shape()) {
    // For dynamic shape in sub graph sink and any type parameter, the input size should be updated.
    node_device_kernel_tensor->device_address()->SetSize(input_kernel_tensor->device_address()->GetSize());
    // Update Shape.
    node_device_kernel_tensor->SetShape(input_kernel_tensor->GetShape()->Clone());
  }
}

inline bool InputDataNoNeedCopy(const AnfNodePtr &input_node, const KernelTensorPtr &input_kernel_tensor,
                                const KernelTensorPtr &node_kernel_tensor, const KernelTransformType &type) {
  if (input_kernel_tensor == nullptr) {
    return true;
  }
  auto node_device_tensor = node_kernel_tensor->device_address().get();
  auto input_device_tensor = input_kernel_tensor->device_address().get();
  MS_EXCEPTION_IF_NULL(node_device_tensor);
  if (input_device_tensor == nullptr) {
    return true;
  }

  if (input_device_tensor == node_device_tensor) {
    (void)input_kernel_tensor->TouchSyncHandler();
    return true;
  }

  UpdateShape(input_node, node_kernel_tensor, input_kernel_tensor, type);

  if (TEST_FLAG(node_kernel_tensor->flag(), device::kDeviceAddressFlagNotUsed) ||
      input_device_tensor->GetPtr() == node_device_tensor->GetPtr()) {
    return true;
  }

  return false;
}
}  // namespace

void SuperKernelActor::Init() {
  MS_EXCEPTION_IF_NULL(graph_);

  // Set the number of actor running dependent messages.
  running_dependent_msg_num_ = SizeToInt(input_datas_num_ + input_controls_num_);

  // Init the output data.
  InitOutputData();
  if (output_data_arrows_.size() != output_data_nodes_.size()) {
    MS_LOG(EXCEPTION) << "The size of output data arrows is not equal to the output data nodes.";
  }
  if (output_data_arrows_.size() != output_data_.size()) {
    MS_LOG(EXCEPTION) << "The size of output data arrows is not equal to the output data.";
  }
  for (size_t i = 0; i < output_data_arrows_.size(); ++i) {
    auto &data_arrow = output_data_arrows_[i];
    auto &output_node = output_data_nodes_[i];
    auto data = output_data_[i].first.get();
    MS_EXCEPTION_IF_NULL(data_arrow);
    MS_EXCEPTION_IF_NULL(output_node);
    MS_EXCEPTION_IF_NULL(data);
    auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(output_node, IntToSize(data_arrow->from_output_index_), false);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    data->data_ = kernel_tensor;
  }

  const auto &output_with_indexs = common::AnfAlgo::GetAllOutputWithIndex(graph_->output());
  for (const auto &origin_output_with_index : output_with_indexs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(origin_output_with_index);
    const auto &output_node = output_with_index.first;
    MS_EXCEPTION_IF_NULL(output_node);
    if (output_node->isa<CNode>() && (!HasAbstractMonad(output_node))) {
      auto device_address = AnfAlgo::GetMutableOutputAddr(output_node, output_with_index.second, false);
      MS_EXCEPTION_IF_NULL(device_address);
      auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(output_node, output_with_index.second, false);
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      if (kernel_tensor->is_ptr_persisted() || graph_->is_dynamic_shape()) {
        MS_LOG(DEBUG) << "Actor:" << GetAID() << " skip alloc memory for device address:" << device_address
                      << " is persist:" << kernel_tensor->is_ptr_persisted()
                      << " is dynamic shape:" << graph_->is_dynamic_shape()
                      << " output node:" << output_node->DebugString();
        continue;
      }
      // Free the ptr in device address of output node.
      if (device_address->GetPtr() != nullptr) {
        MS_LOG(INFO) << "Output node:" << output_node->DebugString() << " has a default ptr, maybe a mem leak.";
        device_address->set_ptr(nullptr);
      }
      if (IsSkippedLaunch()) {
        device_address_to_node_[device_address.get()] = {device_address->GetSize(), output_node->fullname_with_scope()};
      }
      memory_alloc_list_.emplace_back(kernel_tensor);
    }
  }

  // Check whether the parameter needs to be copied out.
  node_kernel_tensors_.resize(graph_->input_nodes().size());
  is_parameters_need_copy_.resize(graph_->input_nodes().size());
  copy_input_kernel_tensors_.resize(graph_->input_nodes().size());
  for (size_t i = 0; i < graph_->input_nodes().size(); ++i) {
    const auto &input_node = graph_->input_nodes()[i];
    MS_EXCEPTION_IF_NULL(input_node);
    node_kernel_tensors_[i] = AnfAlgo::GetOutputKernelTensor(input_node, 0, false);
    if (!common::AnfAlgo::HasAbstractRef(input_node)) {
      is_parameters_need_copy_[i] = false;
      continue;
    }
    // If the parameter has ref attribute and is directly used by the kernel in the graph, it needs to be copied.
    is_parameters_need_copy_[i] = true;
  }

  graph_executor_->InitGraphInfo(graph_);
}

size_t SuperKernelActor::FetchInputNodePosition(const AnfNodePtr &intput_node) {
  MS_EXCEPTION_IF_NULL(intput_node);
  MS_EXCEPTION_IF_NULL(graph_);

  auto &input_nodes = graph_->input_nodes();
  const auto &iter = find(input_nodes.begin(), input_nodes.end(), intput_node);
  if (iter == input_nodes.end()) {
    MS_LOG_WITH_NODE(EXCEPTION, intput_node) << "Invalid input node:" << intput_node->fullname_with_scope();
  }
  return iter - input_nodes.begin();
}

void SuperKernelActor::FetchInputDeviceTensor(OpContext<KernelTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, GetAID().Name());
  MS_EXCEPTION_IF_NULL(context);
  std::vector<KernelTensorPtr> memory_free_list;
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter != input_op_datas_.end()) {
    for (auto &input_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(input_data);
      MS_EXCEPTION_IF_NULL(input_data->data_);
      size_t index = IntToSize(input_data->index_);
      if (index >= input_kernel_tensors_.size()) {
        std::string error_info = "Invalid input index:" + std::to_string(index) +
                                 " total:" + std::to_string(input_kernel_tensors_.size()) +
                                 " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      input_kernel_tensors_[index] = input_data->data_;

      if (memory::mem_pool::IsNeedProfilieMemoryLog()) {
        auto output_address = reinterpret_cast<std::uintptr_t>(input_kernel_tensors_[index]->device_address().get());
        MS_LOG(WARNING) << "Need Profile Memory, Memory use, actor name: " << GetAID().Name()
                        << ", kernel graph: " << graph_->ToString() << ", device address class ptr: " << output_address
                        << ", device address size: " << input_kernel_tensors_[index]->GetSize()
                        << ", device address addr: " << input_kernel_tensors_[index]->device_address()->GetPtr()
                        << ", index: " << index;
      }
      if (input_data->data_->dynamic_ref_count() != INT32_MAX) {
        (void)memory_free_list.emplace_back(input_data->data_);
      }
    }
    memory_free_lists_.push(memory_free_list);
  }
}

void SuperKernelActor::Run(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(graph_);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, GetAID().Name(), "SuperKernelActor", graph_->ToString());

  MS_LOG(INFO) << "Super kernel actor(" << GetAID().Name()
               << ") launches graph: " << std::to_string(graph_->graph_id());
  if (memory::mem_pool::IsNeedProfilieMemoryLog()) {
    MS_LOG(WARNING) << "Need Profile Memory, launch actor name: " << GetAID().Name()
                    << ", kernel graph: " << graph_->ToString();
  }

  FetchInputDeviceTensor(context);
  FetchPersistentDeviceTensor();

  TrackInputMemory();

  if (memory_alloc_list_.size() > 0) {
    for (auto &kernel_tensor : memory_alloc_list_) {
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      auto &device_tensor = kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(device_tensor);
      if (kernel_tensor->IsNotNeedAlloc()) {
        continue;
      }
      if (memory::mem_pool::IsNeedProfilieMemoryLog()) {
        auto &info = device_address_to_node_[device_tensor.get()];
        auto output_address = reinterpret_cast<std::uintptr_t>(device_tensor.get());
        MS_LOG(WARNING) << "Need Profile Memory, Memory need allocated, actor name: " << GetAID().Name()
                        << ", kernel graph: " << graph_->ToString() << ", node: " << info.node_full_name
                        << ", device address class ptr: " << output_address << ", device address size: " << info.size;
      }
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, GetAID().Name(),
                                                     device::tracker::MemType::kGraphOutput, device_tensor->GetSize(),
                                                     device_tensor.get());
    }
    SendMemoryAllocReq(context);
  } else {
    OnMemoryAllocFinish(context);
  }
  if (memory::mem_pool::IsNeedProfilieMemoryLog()) {
    MS_LOG(WARNING) << "Need Profile Memory, end launch, actor name: " << GetAID().Name()
                    << ", kernel graph: " << graph_->ToString();
  }
}

void SuperKernelActor::FetchPersistentDeviceTensor() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_type = device::GetDeviceTypeByName(ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET));

  for (auto &device_tensor_store_key : device_tensor_store_keys_) {
    auto input_kernel_tensor =
      DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second.get(), device_type);
    // Ge backend maybe nullptr.
    if (input_kernel_tensor == nullptr) {
      MS_LOG(DEBUG) << "Failed get device tensor for node:" << device_tensor_store_key.second->DebugString()
                    << " index:" << device_tensor_store_key.first;
      continue;
    }

    size_t index = device_tensor_store_key.first;
    input_kernel_tensors_[index] = input_kernel_tensor;
  }
}

void SuperKernelActor::SendMemoryAllocReq(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);

  sort(memory_alloc_list_.begin(), memory_alloc_list_.end(), [](const KernelTensorPtr a, const KernelTensorPtr b) {
    MS_EXCEPTION_IF_NULL(a);
    MS_EXCEPTION_IF_NULL(b);
    return a->GetSize() > b->GetSize();
  });
  if (ActorDispatcher::is_memory_allocation_sync()) {
    ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &memory_alloc_list_, context,
                              GetAID());
    OnMemoryAllocFinish(context);
  } else {
    ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &memory_alloc_list_, context,
                          GetAID());
  }
}

void SuperKernelActor::OnMemoryAllocFinish(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(graph_);
  if (IsRunningFailed(context)) {
    MS_LOG(INFO) << "Running failed in actor:" << GetAID().Name();
    return;
  }
  {
    ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, GetAID().Name());
    if (!CopyInputData(context, graph_)) {
      std::string error_info = "Copy the input data failed, graph id: " + std::to_string(graph_->graph_id());
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);

  try {
    const std::vector<tensor::TensorPtr> inputs;
    std::vector<tensor::TensorPtr> outputs;
    const std::map<string, string> compile_options;

    MS_EXCEPTION_IF_NULL(graph_executor_);
    if (!IsSkippedLaunch(nullptr, graph_)) {
      ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kGraphLaunch, GetAID().Name());
      auto ret = graph_executor_->RunGraph(graph_, inputs, &outputs, compile_options);
      if (!ret) {
        std::string error_info = "Launch graph failed, graph id: " + std::to_string(graph_->graph_id());
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
    } else if (memory::mem_pool::IsNeedProfilieMemoryLog()) {
      auto memory_size = graph_executor_->GetGraphFeatureMemory(graph_);
      MS_LOG(WARNING) << "Need Profile Memory, graph: " << graph_->ToString() << ", feature memory: " << memory_size;
      MS_LOG(WARNING) << "Need Profile Memory, max used static memory: "
                      << host_context->device_res_manager_->GetMaxUsedMemorySize();
    }
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    std::string error_info = "Launch graph exception, graph id: " + std::to_string(graph_->graph_id());
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }

  {
    ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPostLaunch, GetAID().Name());
    for (auto item : ref_node_addr_map_) {
      MS_EXCEPTION_IF_NULL(item.first);
      MS_EXCEPTION_IF_NULL(item.second);
      MS_LOG(INFO) << "The input ref node copy back from address: " << item.first->ToString()
                   << " to address: " << item.second->ToString() << ".";
      if (!SyncCopy(item.second, item.first, kDefaultStreamIndex) ||
          !host_context->device_res_manager_->SyncAllStreams()) {
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "Copy data failed.");
      }
    }
    ref_node_addr_map_.clear();
  }

  // Debug actor is blocked, must wait debug actor callback message to process continue.
  if (debug_aid_ != nullptr) {
    SendDebugReq(context);
    return;
  }
  PostRun(context);
}

void SuperKernelActor::SendDebugReq(OpContext<KernelTensor> *const context) {
  running_dependent_msg_num_ = 1;
  OnDebugFinish(context);
}

bool SuperKernelActor::CopyInputDataPersistedHandle(const KernelTensorPtr &input_kernel_tensor,
                                                    const KernelTensorPtr &node_kernel_tensor, size_t i) {
  auto &input_device_tensor = input_kernel_tensor->device_address();
  auto &node_device_tensor = node_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(node_device_tensor);
  if ((input_device_tensor->GetDeviceType() == node_device_tensor->GetDeviceType()) &&
      AnfAlgo::IsEquivalentFormat(input_kernel_tensor->format(), node_kernel_tensor->format())) {
    MS_LOG(DEBUG) << "Not need copy for device tensor:" << node_device_tensor << " ptr:" << node_device_tensor->GetPtr()
                  << " index:" << i << " for actor:" << GetAID();
    // Set the ptr from input_device_tensor and set mem pool false to avoid memory double management for
    // supporting zero copy.
    if (type_ != KernelTransformType::kSuperKernelActor) {
      node_device_tensor->set_ptr(input_device_tensor->GetMutablePtr());
    } else {
      node_device_tensor->set_ptr(input_kernel_tensor->GetValidPtr(input_device_tensor->stream_id()));
    }
    MS_LOG(DEBUG) << "Actor:" << GetAID() << "set need sync flag from:" << input_device_tensor
                  << " to:" << node_device_tensor
                  << " sync user data handler:" << node_kernel_tensor->need_sync_user_data();
    node_device_tensor->set_from_mem_pool(false);
    // continue
    return true;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);

  if (device::GetDeviceTypeByName(device_name) != node_device_tensor->GetDeviceType()) {
    MS_LOG(EXCEPTION) << "GE backend only support Ascend, but got "
                      << device::GetDeviceNameByType(node_device_tensor->GetDeviceType());
  }

  if (copy_input_kernel_tensors_[i] == nullptr) {
    MS_EXCEPTION_IF_NULL(node_kernel_tensor);
    auto node_device_address = node_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(node_device_address);
    auto new_kernel_tensor = node_kernel_tensor->CloneKernelTensor();
    MS_EXCEPTION_IF_NULL(new_kernel_tensor);
    // create device address with correct context.
    auto new_device_address = std::make_shared<device::DeviceAddress>(
      node_device_address->device_pointer()->ptr(), node_device_address->size(),
      device::GetDeviceTypeByName(device_name), node_device_address->stream_id());
    new_kernel_tensor->set_device_address(new_device_address);
    new_kernel_tensor->set_tensor_storage_info(nullptr);
    new_kernel_tensor->SetShapeVector(node_kernel_tensor->GetShapeVector());
    new_kernel_tensor->SetDeviceType(node_device_tensor->GetDeviceType());
    new_kernel_tensor->set_device_ptr(nullptr);
    new_kernel_tensor->set_user_data(node_kernel_tensor->user_data());
    new_kernel_tensor->set_need_sync_user_data(node_kernel_tensor->need_sync_user_data());

    copy_input_kernel_tensors_[i] = new_kernel_tensor;
    MS_LOG(DEBUG) << "Create new kernel tensor:" << copy_input_kernel_tensors_[i] << " index:" << i
                  << " for actor:" << GetAID();
  }
  auto copy_kernel_tensor = copy_input_kernel_tensors_[i];
  MS_EXCEPTION_IF_NULL(copy_kernel_tensor);
  auto &copy_device_tensor = copy_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(copy_device_tensor);
  copy_kernel_tensor->set_user_data(node_kernel_tensor->user_data());
  copy_kernel_tensor->set_need_sync_user_data(node_kernel_tensor->need_sync_user_data());
  if ((copy_device_tensor->GetPtr() == nullptr) &&
      (!host_context->device_res_manager_->AllocateMemory(copy_device_tensor.get()))) {
    MS_LOG(ERROR) << "Device(id:" << std::to_string(node_device_tensor->device_id())
                  << ") memory isn't enough and alloc failed, kernel name: " << GetAID()
                  << ", alloc size: " + std::to_string(copy_device_tensor->GetSize()) << "B.";
    return true;
  }
  MS_LOG(DEBUG) << "Alloc memory for device tensor:" << copy_device_tensor << " ptr:" << copy_device_tensor->GetPtr()
                << " size:" << copy_device_tensor->GetSize() << " index:" << i << " for actor:" << GetAID();
  if (type_ != KernelTransformType::kSuperKernelActor) {
    node_device_tensor->set_ptr(copy_device_tensor->GetMutablePtr());
  } else {
    node_device_tensor->set_ptr(copy_kernel_tensor->GetValidPtr(copy_device_tensor->stream_id()));
  }
  node_device_tensor->set_from_mem_pool(false);
  return false;
}

bool SuperKernelActor::CopyInputData(const OpContext<KernelTensor> *context, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(graph);

  auto &input_nodes = graph->input_nodes();
  if (input_kernel_tensors_.size() != node_kernel_tensors_.size()) {
    MS_LOG(ERROR) << "The size of input_kernel_tensors_[" << input_kernel_tensors_.size()
                  << "] is not equal to the size of node_kernel_tensors_[" << node_kernel_tensors_.size() << "].";
    return false;
  }

  for (size_t i = 0; i < input_kernel_tensors_.size(); ++i) {
    auto &node_kernel_tensor = node_kernel_tensors_[i];
    MS_EXCEPTION_IF_NULL(node_kernel_tensor);
    auto &node_device_tensor = node_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(node_device_tensor);
    MS_EXCEPTION_IF_NULL(input_nodes[i]);
    auto &input_kernel_tensor = input_kernel_tensors_[i];
    if (InputDataNoNeedCopy(input_nodes[i], input_kernel_tensor, node_kernel_tensor, type_)) {
      MS_LOG(DEBUG) << "Actor:" << GetAID() << " input kernel tensor " << i << ":" << input_kernel_tensor
                    << " no need copy.";
      continue;
    }
    MS_EXCEPTION_IF_NULL(input_kernel_tensor);
    UpdateShape(input_nodes[i], node_kernel_tensor, input_kernel_tensor, type_);
    node_kernel_tensor->set_user_data(input_kernel_tensors_[i]->user_data());
    node_kernel_tensor->set_need_sync_user_data(input_kernel_tensors_[i]->need_sync_user_data());
    if (type_ != KernelTransformType::kSuperKernelActor) {
      node_kernel_tensor->SetValue(input_kernel_tensor->GetValueTrack());
    }

    // Copy.
    KernelTensor *copy_kernel_tensor = nullptr;
    // If the input is not a persist device address, in a heterogeneous scenario, a new device address needs to
    // be created. And set ptr to node device address to support the zero copy of graph input nodes.
    if (!node_kernel_tensor->is_ptr_persisted()) {
      if (CopyInputDataPersistedHandle(input_kernel_tensors_[i], node_kernel_tensor, i)) {
        continue;
      }
      copy_kernel_tensor = copy_input_kernel_tensors_[i].get();
    } else {
      if (node_device_tensor->GetPtr() == nullptr) {
        MS_LOG(INFO) << "The node device tensor, which shared with another graph, has no device memory and will skip "
                        "copy for actor:"
                     << GetAID();
        continue;
      }
      copy_kernel_tensor = node_kernel_tensor.get();
    }
    MS_EXCEPTION_IF_NULL(copy_kernel_tensor);
    MS_LOG(INFO) << "The input data of node:" << input_nodes[i]->DebugString()
                 << " need copy from kernel tensor:" << input_kernel_tensor->ToString()
                 << " to:" << copy_kernel_tensor->ToString()
                 << ", is ref node need copy back:" << is_parameters_need_copy_[i] << " for actor:" << GetAID();
    device::DeviceContextKey host_key = {copy_kernel_tensor->GetDeviceType(), copy_kernel_tensor->device_id()};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);

    if (!SyncCopy(copy_kernel_tensor, input_kernel_tensor.get(), kDefaultStreamIndex) ||
        !host_context->device_res_manager_->SyncAllStreams()) {
      MS_LOG(ERROR) << "Copy data failed for actor:" << GetAID() << " input index:" << i;
      continue;
    }
    if (is_parameters_need_copy_[i]) {
      ref_node_addr_map_[copy_kernel_tensor] = input_kernel_tensor.get();
    }
  }
  return true;
}

void SuperKernelActor::SendMemoryFreeReq(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(graph_);

  if (memory_free_lists_.size() > 0 && memory_free_lists_.back().size() > 0) {
    if (memory::mem_pool::IsNeedProfilieMemoryLog()) {
      for (auto data : memory_free_lists_.back()) {
        auto output_address = reinterpret_cast<std::uintptr_t>(data->device_address().get());
        MS_LOG(WARNING) << "Need Profile Memory, Memory need Decrease DynamicRefCount, actor name: " << GetAID().Name()
                        << ", kernel graph: " << graph_->ToString() << ", device address class ptr: " << output_address
                        << ", device address size: " << data->GetSize()
                        << ", device address addr: " << data->device_address()->GetPtr();
      }
    }

    if (ActorDispatcher::is_memory_free_sync()) {
      ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &(memory_free_lists_.back()),
                                context, GetAID());
    } else {
      ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &(memory_free_lists_.back()), context,
                            GetAID());
    }
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);

  // Free the address that is the temp store for kernel input copy.
  for (auto &copy_input_kernel_tensor : copy_input_kernel_tensors_) {
    if ((copy_input_kernel_tensor != nullptr) && (copy_input_kernel_tensor->device_address() != nullptr) &&
        (copy_input_kernel_tensor->device_address()->GetPtr() != nullptr)) {
      host_context->device_res_manager_->FreeMemory(copy_input_kernel_tensor->device_address().get());
    }
  }
}

void SuperKernelActor::TrackInputMemory() {
  if (!device::tracker::MemTrackerManager::GetInstance().IsEnabled()) {
    return;
  }

  for (auto &kernel_tensor : input_kernel_tensors_) {
    if (kernel_tensor == nullptr || kernel_tensor->device_address() == nullptr ||
        !kernel_tensor->device_address()->IsPtrValid()) {
      continue;
    }
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(UseMemBlock, GetAID().Name(),
                                                   kernel_tensor->device_address()->GetPtr());
  }
}
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore
