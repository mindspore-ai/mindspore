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

#include <set>
#include "runtime/graph_scheduler/actor/super_kernel_actor.h"
#include "runtime/graph_scheduler/actor/output_actor.h"
#include "runtime/graph_scheduler/actor/memory_manager_actor.h"
#include "runtime/graph_scheduler/actor/debug_actor.h"
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
void SuperKernelActor::Init() {
  MS_EXCEPTION_IF_NULL(graph_);
  // Check device contexts number.
  if (device_contexts_.size() != device::kDeviceContextsNumOne) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }

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
    auto device_address = AnfAlgo::GetMutableOutputAddr(output_node, IntToSize(data_arrow->from_output_index_), false);
    data->data_ = device_address.get();
  }

  const auto &output_with_indexs = common::AnfAlgo::GetAllOutputWithIndex(graph_->output());
  for (const auto &output_with_index : output_with_indexs) {
    const auto &output_node = output_with_index.first;
    if (output_node->isa<CNode>() && (!HasAbstractMonad(output_node))) {
      auto device_address = AnfAlgo::GetMutableOutputAddr(output_node, output_with_index.second, false);
      if (device_address->is_ptr_persisted()) {
        continue;
      }
      // Free the ptr in device address of output node.
      if (device_address->GetPtr() != nullptr) {
        MS_LOG(WARNING) << "Output node:" << output_node->DebugString() << " has a default ptr, maybe a mem leak.";
        device_address->set_ptr(nullptr);
      }
      memory_alloc_list_.emplace_back(device_address.get());
    }
  }

  // Check whether the parameter needs to be copied out.
  is_parameters_need_copy_.resize(graph_->input_nodes().size());
  copy_input_device_tensors_.resize(graph_->input_nodes().size());
  for (size_t i = 0; i < graph_->input_nodes().size(); ++i) {
    const auto &input_node = graph_->input_nodes()[i];
    MS_EXCEPTION_IF_NULL(input_node);
    if (!common::AnfAlgo::HasAbstractRef(input_node)) {
      is_parameters_need_copy_[i] = false;
      continue;
    }
    // If the parameter has ref attribute and is directly used by the kernel in the graph, it needs to be copied.
    is_parameters_need_copy_[i] = true;
  }
}

size_t SuperKernelActor::FetchInputNodePosition(const AnfNodePtr &intput_node) {
  MS_EXCEPTION_IF_NULL(intput_node);
  MS_EXCEPTION_IF_NULL(graph_);

  auto &input_nodes = graph_->input_nodes();
  const auto &iter = find(input_nodes.begin(), input_nodes.end(), intput_node);
  if (iter == input_nodes.end()) {
    MS_LOG(EXCEPTION) << "Invalid input node:" << intput_node->fullname_with_scope();
  }
  return iter - input_nodes.begin();
}

void SuperKernelActor::FetchInputDeviceTensor(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);
  std::vector<DeviceTensor *> memory_free_list;
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter != input_op_datas_.end()) {
    for (auto &input_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(input_data);
      MS_EXCEPTION_IF_NULL(input_data->data_);
      size_t index = IntToSize(input_data->index_);
      if (index >= input_device_tensors_.size()) {
        std::string error_info = "Invalid input index:" + std::to_string(index) +
                                 " total:" + std::to_string(input_device_tensors_.size()) +
                                 " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      input_device_tensors_[index] = input_data->data_;
      if (input_data->data_->dynamic_ref_count() != INT32_MAX) {
        memory_free_list.emplace_back(input_data->data_);
      }
    }
    memory_free_lists_.push(memory_free_list);
  }

  // Check device tensor store.
  for (auto &device_tensor_store_key : device_tensor_store_keys_) {
    auto input_device_tensor = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second.get(),
                                                                      device_contexts_[0]->GetDeviceType());
    // Ge backend maybe nullptr.
    if (input_device_tensor == nullptr) {
      MS_LOG(WARNING) << "Failed get device tensor for node:" << device_tensor_store_key.second->DebugString()
                      << " indx:" << device_tensor_store_key.first;
      continue;
    }

    size_t index = device_tensor_store_key.first;
    if (index >= input_device_tensors_.size()) {
      std::string error_info = "Invalid input index:" + std::to_string(index) +
                               " total:" + std::to_string(input_device_tensors_.size()) +
                               " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    input_device_tensors_[index] = input_device_tensor;
  }
}

void SuperKernelActor::Run(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(graph_);
  if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "Invalid device context for super kernel actor:" + GetAID().Name());
  }
  MS_LOG(INFO) << "Super kernel actor(" << GetAID().Name()
               << ") launches graph: " << std::to_string(graph_->graph_id());
  FetchInputDeviceTensor(context);
  if (memory_alloc_list_.size() > 0) {
    SendMemoryAllocReq(context);
  } else {
    OnMemoryAllocFinish(context);
  }
}

void SuperKernelActor::SendMemoryAllocReq(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  if (ActorDispatcher::is_memory_allocation_sync()) {
    ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &memory_alloc_list_,
                              device_contexts_[0], context, GetAID());
    OnMemoryAllocFinish(context);
  } else {
    ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &memory_alloc_list_,
                          device_contexts_[0], context, GetAID());
  }
}

void SuperKernelActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  if (!CopyInputData(context)) {
    std::string error_info = "Copy the input data failed, graph id: " + std::to_string(graph_->graph_id());
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }
  try {
    const auto input_nodes = graph_->input_nodes();
    for (size_t i = 0; i < input_device_tensors_.size(); ++i) {
      auto node_device_tensor = AnfAlgo::GetMutableOutputAddr(input_nodes[i], 0, false);
      if (node_device_tensor != nullptr && (!node_device_tensor->is_ptr_persisted()) &&
          input_device_tensors_[i] != nullptr) {
        node_device_tensor->set_ptr(input_device_tensors_[i]->GetMutablePtr());
        node_device_tensor->set_from_mem_pool(false);
      }
    }

    const std::vector<tensor::Tensor> inputs;
    std::vector<tensor::Tensor> outputs;
    const std::map<string, string> compile_options;
    MS_EXCEPTION_IF_NULL(device_contexts_[0]->graph_executor_);
    auto ret = device_contexts_[0]->graph_executor_->RunGraph(graph_, inputs, &outputs, compile_options);
    if (!ret) {
      std::string error_info = "Launch graph failed, graph id: " + std::to_string(graph_->graph_id());
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    std::string error_info = "Launch graph exception, graph id: " + std::to_string(graph_->graph_id());
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }

  for (auto item : ref_node_addr_map_) {
    MS_EXCEPTION_IF_NULL(item.first);
    MS_EXCEPTION_IF_NULL(item.second);
    auto formal_param_addr = AnfAlgo::GetMutableOutputAddr(item.first, 0, false);
    MS_EXCEPTION_IF_NULL(formal_param_addr);
    MS_LOG(INFO) << "The input ref_node: " << item.first->DebugString()
                 << " need copy back, from address: " << formal_param_addr->GetPtr()
                 << " to address: " << item.second->GetPtr() << ".";
    if (!Copy(item.second, formal_param_addr.get())) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "Copy data failed.");
    }
  }
  ref_node_addr_map_.clear();

  // Debug actor is blocked, must wait debug actor callback message to process continue.
  if (debug_aid_ != nullptr) {
    SendDebugReq(context);
    return;
  }

  PostRun(context);
}

void SuperKernelActor::SendDebugReq(OpContext<DeviceTensor> *const context) {
  running_dependent_msg_num_ = 1;
  ActorDispatcher::SendSync(*debug_aid_, &DebugActor::DebugForGraph, graph_, device_contexts_[0], context, &GetAID());
  OnDebugFinish(context);
}

bool SuperKernelActor::CopyInputData(const OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(graph_);
  for (size_t i = 0; i < input_device_tensors_.size(); ++i) {
    auto src_device_tensor = input_device_tensors_[i];
    if (src_device_tensor == nullptr) {
      continue;
    }

    auto &input_nodes = graph_->input_nodes();
    if (i >= input_nodes.size()) {
      MS_LOG(ERROR) << "The input index:" << i << "is out of range:" << input_nodes.size() << ".";
      return false;
    }
    auto dst_node = input_nodes[i];
    MS_EXCEPTION_IF_NULL(dst_node);

    auto dst_param = dst_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(dst_param);
    if (!dst_param->IsUsedByRealKernelInGraph(graph_->graph_id())) {
      continue;
    }
    auto dst_device_tensor = AnfAlgo::GetMutableOutputAddr(dst_node, 0, false);
    MS_EXCEPTION_IF_NULL(dst_device_tensor);
    if (src_device_tensor->GetPtr() == dst_device_tensor->GetPtr()) {
      continue;
    }

    // If the input is not a persist device address, in a heterogeneous scenario, a new device address needs to
    // be created.
    if (!dst_device_tensor->is_ptr_persisted()) {
      if (src_device_tensor->GetDeviceType() == dst_device_tensor->GetDeviceType()) {
        MS_LOG(DEBUG) << "Disable copy for device tensor:" << dst_device_tensor;
        continue;
      }

      if (copy_input_device_tensors_[i] == nullptr) {
        copy_input_device_tensors_[i] = device_contexts_[0]->device_res_manager_->CreateDeviceAddress(
          nullptr, dst_device_tensor->GetSize(), dst_device_tensor->format(), dst_device_tensor->type_id(),
          dst_device_tensor->host_shape());
        MS_LOG(DEBUG) << "Create new device tensor:" << copy_input_device_tensors_[i] << " index:" << i
                      << " for actor:" << GetAID();
      }
      dst_device_tensor = copy_input_device_tensors_[i];
      MS_EXCEPTION_IF_NULL(dst_device_tensor);
      MS_EXCEPTION_IF_NULL(device_contexts_[0]);
      if ((dst_device_tensor->GetPtr() == nullptr) &&
          (!device_contexts_[0]->device_res_manager_->AllocateMemory(dst_device_tensor.get()))) {
        MS_LOG(ERROR) << "Device(id:" << std::to_string((device_contexts_[0])->device_context_key().device_id_)
                      << ") memory isn't enough and alloc failed, kernel name: " << GetAID()
                      << ", alloc size: " + std::to_string(dst_device_tensor->GetSize()) << "B.";
        continue;
      }
      MS_LOG(DEBUG) << "Alloc memory for device tensor:" << dst_device_tensor << " ptr:" << dst_device_tensor->GetPtr()
                    << " index:" << i << " for actor:" << GetAID();
    }

    MS_LOG(INFO) << "The input data of node:" << dst_node->DebugString()
                 << " need copy from address:" << src_device_tensor->GetPtr()
                 << ", type:" << src_device_tensor->GetDeviceType() << " to address:" << dst_device_tensor->GetPtr()
                 << ", type:" << dst_device_tensor->GetDeviceType() << ".";
    if (!Copy(dst_device_tensor.get(), src_device_tensor)) {
      MS_LOG(ERROR) << "Copy data failed.";
      continue;
    }
    input_device_tensors_[i] = dst_device_tensor.get();
    if (is_parameters_need_copy_[i] && ref_node_addr_map_.count(dst_node) == 0) {
      ref_node_addr_map_[dst_node] = src_device_tensor;
    }
  }
  return true;
}

void SuperKernelActor::SendMemoryFreeReq(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  if (memory_free_lists_.size() > 0 && memory_free_lists_.back().size() > 0) {
    if (ActorDispatcher::is_memory_free_sync()) {
      ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &(memory_free_lists_.back()),
                                device_contexts_[0], context, GetAID());
    } else {
      ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &(memory_free_lists_.back()),
                            device_contexts_[0], context, GetAID());
    }
  }

  // Free the address that is the temp store for kernel input copy.
  for (auto &copy_input_device_tensor : copy_input_device_tensors_) {
    if ((copy_input_device_tensor != nullptr) && (copy_input_device_tensor->GetPtr() != nullptr)) {
      MS_EXCEPTION_IF_NULL(device_contexts_[0]);
      device_contexts_[0]->device_res_manager_->FreeMemory(copy_input_device_tensor.get());
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
