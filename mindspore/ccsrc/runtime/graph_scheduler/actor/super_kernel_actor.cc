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

  // Check whether the parameter needs to be copied out.
  is_parameters_need_copy_.resize(graph_->input_nodes().size());
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
    const std::vector<tensor::Tensor> inputs;
    std::vector<tensor::Tensor> outputs;
    const std::map<string, string> compile_options;
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
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter == input_op_datas_.end()) {
    return true;
  }

  auto check_and_copy_func = [this](DeviceTensor *src_device_tensor, size_t dst_index) {
    auto &input_nodes = graph_->input_nodes();
    if (IntToSize(dst_index) >= input_nodes.size()) {
      MS_LOG(ERROR) << "The input index:" << dst_index << "is out of range:" << input_nodes.size() << ".";
      return false;
    }
    auto dst_node = input_nodes[IntToSize(dst_index)];
    MS_EXCEPTION_IF_NULL(dst_node);

    auto dst_param = dst_node->cast<ParameterPtr>();
    if (!dst_param->IsUsedByRealKernelInGraph(graph_->graph_id())) {
      return true;
    }
    auto dst_device_tensor = AnfAlgo::GetMutableOutputAddr(dst_node, 0, false);
    MS_EXCEPTION_IF_NULL(dst_device_tensor);
    if (src_device_tensor->GetPtr() == dst_device_tensor->GetPtr()) {
      return true;
    }

    MS_LOG(INFO) << "The input data of node:" << dst_node->DebugString()
                 << " need copy from address:" << src_device_tensor->GetPtr()
                 << ", type:" << src_device_tensor->GetDeviceType() << " to address:" << dst_device_tensor->GetPtr()
                 << ", type:" << dst_device_tensor->GetDeviceType() << ".";
    if (!Copy(dst_device_tensor.get(), src_device_tensor)) {
      MS_LOG(ERROR) << "Copy data failed.";
      return false;
    }
    if (is_parameters_need_copy_[dst_index] && ref_node_addr_map_.count(dst_node) == 0) {
      ref_node_addr_map_[dst_node] = src_device_tensor;
    }
    return true;
  };

  // Copy input data.
  for (auto &input_data : data_iter->second) {
    MS_EXCEPTION_IF_NULL(input_data);
    MS_EXCEPTION_IF_NULL(input_data->data_);
    if (!check_and_copy_func(input_data->data_, input_data->index_)) {
      return false;
    }
  }

  // Check device tensor store.
  for (auto &device_tensor_store_key : device_tensor_store_keys_) {
    auto input_device_tensor = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second.get(),
                                                                      device_contexts_[0]->GetDeviceType());
    // Ge backend maybe nullptr.
    if (input_device_tensor == nullptr) {
      continue;
    }
    if (!check_and_copy_func(input_device_tensor, device_tensor_store_key.first)) {
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
    if (ActorDispatcher::is_memory_free_sync()) {
      ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &(memory_free_lists_.back()),
                                device_contexts_[0], context, GetAID());
    } else {
      ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &(memory_free_lists_.back()),
                            device_contexts_[0], context, GetAID());
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
