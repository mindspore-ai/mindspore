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

#include "runtime/framework/actor/super_kernel_actor.h"
#include "runtime/framework/actor/output_actor.h"
#include "runtime/framework/actor/memory_manager_actor.h"
#include "runtime/framework/actor/debug_actor.h"
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

  if (output_data_arrows_.size() != output_data_nodes_.size()) {
    MS_LOG(EXCEPTION) << "The size of output data arrows is not equal to the output data nodes.";
  }
  // Init the output data.
  for (size_t i = 0; i < output_data_arrows_.size(); ++i) {
    auto &data_arrow = output_data_arrows_[i];
    auto &output_node = output_data_nodes_[i];
    MS_EXCEPTION_IF_NULL(data_arrow);
    MS_EXCEPTION_IF_NULL(output_node);

    auto device_address = AnfAlgo::GetMutableOutputAddr(output_node, data_arrow->from_output_index_, false);
    auto data =
      std::make_unique<OpData<DeviceTensor>>(data_arrow->to_op_id_, device_address.get(), data_arrow->to_input_index_);
    (void)output_data_.emplace_back(std::move(data));
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

void SuperKernelActor::Run(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(graph_);
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);
  MS_LOG(INFO) << "Super kernel actor(" << GetAID().Name()
               << ") launches graph: " << std::to_string(graph_->graph_id());
  if (!CopyInputData(context)) {
    std::string error_info = "Copy the input data failed, graph id: " + std::to_string(graph_->graph_id());
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }

  try {
    auto ret = device_contexts_[0]->LaunchGraph(graph_);
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
  ActorDispatcher::Send(*debug_aid_, &DebugActor::DebugForGraph, graph_, device_contexts_[0], context, &GetAID());
}

bool SuperKernelActor::CopyInputData(const OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(graph_);
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter == input_op_datas_.end()) {
    return true;
  }

  auto &input_nodes = graph_->input_nodes();
  // Copy input data.
  for (auto &input_data : data_iter->second) {
    MS_EXCEPTION_IF_NULL(input_data);
    if (IntToSize(input_data->index_) >= input_nodes.size()) {
      MS_LOG(ERROR) << "The input index:" << input_data->index_ << "is out of range:" << input_nodes.size() << ".";
      return false;
    }
    auto input_node = input_nodes[input_data->index_];
    MS_EXCEPTION_IF_NULL(input_node);
    auto input_param = input_node->cast<ParameterPtr>();
    if (!input_param->IsUsedByRealKernelInGraph(graph_->graph_id())) {
      continue;
    }
    auto device_address = AnfAlgo::GetMutableOutputAddr(input_node, 0, false);
    MS_EXCEPTION_IF_NULL(device_address);
    auto &input_device_tensor = input_data->data_;
    MS_EXCEPTION_IF_NULL(input_device_tensor);
    if (input_device_tensor->GetPtr() == device_address->GetPtr()) {
      continue;
    }

    MS_LOG(INFO) << "The input data of node:" << input_node->DebugString()
                 << " need copy from address:" << input_device_tensor->GetPtr()
                 << ", type:" << input_device_tensor->DeviceType() << " to address:" << device_address->GetPtr()
                 << ", type:" << device_address->DeviceType() << ".";
    if (!Copy(device_address.get(), input_device_tensor)) {
      MS_LOG(ERROR) << "Copy data failed.";
      return false;
    }
    if (HasAbstractRef(input_node) && ref_node_addr_map_.count(input_node) == 0) {
      ref_node_addr_map_[input_node] = input_device_tensor;
    }
  }

  // Check device tensor store.
  for (auto &device_tensor_store_key : device_tensor_store_keys_) {
    auto input_device_tensor = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second.get(),
                                                                      device_contexts_[0]->GetDeviceAddressType());
    MS_EXCEPTION_IF_NULL(input_device_tensor);
    if (device_tensor_store_key.first >= input_nodes.size()) {
      MS_LOG(ERROR) << "The input index:" << device_tensor_store_key.first << "is out of range:" << input_nodes.size();
      return false;
    }
    auto input_node = input_nodes[device_tensor_store_key.first];
    MS_EXCEPTION_IF_NULL(input_node);

    auto input_param = input_node->cast<ParameterPtr>();
    if (!input_param->IsUsedByRealKernelInGraph(graph_->graph_id())) {
      continue;
    }

    auto device_address = AnfAlgo::GetMutableOutputAddr(input_node, 0, false);
    MS_EXCEPTION_IF_NULL(device_address);
    if (input_device_tensor->GetPtr() != device_address->GetPtr()) {
      MS_LOG(ERROR) << "The input data of node:" << input_node->DebugString()
                    << " device address:" << input_device_tensor->GetPtr()
                    << ", type:" << input_device_tensor->DeviceType()
                    << " is not equal to the graph node device address:" << device_address->GetPtr()
                    << ", type:" << device_address->DeviceType() << ".";
      return false;
    }
  }

  return true;
}

void SuperKernelActor::SendMemoryFreeReq(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  const auto &sequential_num = context->sequential_num_;

  // Collect the input device tensors.
  std::vector<DeviceTensor *> memory_free_list;
  if (input_op_datas_.count(sequential_num) > 0) {
    for (auto &input_data : input_op_datas_[sequential_num]) {
      MS_EXCEPTION_IF_NULL(input_data);
      MS_EXCEPTION_IF_NULL(input_data->data_);
      if (input_data->data_->dynamic_ref_count() != INT32_MAX) {
        (void)memory_free_list.emplace_back(input_data->data_);
      }
    }
  }

  if (memory_free_list.size() > 0) {
    memory_free_lists_.push(memory_free_list);
    ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &(memory_free_lists_.back()),
                          device_contexts_[0], context, GetAID());
  }
}
}  // namespace runtime
}  // namespace mindspore
