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

#include "runtime/framework/actor/kernel_actor.h"
#include "runtime/framework/actor/memory_manager_actor.h"
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
void KernelActor::RunOpData(OpDataPtr<DeviceTensor> input_data, OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  auto sequential_num = context->sequential_num_;
  input_op_datas_[sequential_num].emplace_back(input_data);
  // When all the input data are collected, then allocate memory and callback launch.
  if (CheckLaunchCondition(context)) {
    FetchInputDeviceTensor(context);
    FetchOutputDeviceTensor();
    FetchWorkspaceDeviceTensor();
    AllocateMemory(context);
  }
}

void KernelActor::RunOpControl(AID *input_control, OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  auto sequential_num = context->sequential_num_;
  input_op_controls_[sequential_num].emplace_back(input_control);
  // When all the input data are collected, then allocate memory and callback launch.
  if (CheckLaunchCondition(context)) {
    FetchInputDeviceTensor(context);
    FetchOutputDeviceTensor();
    FetchWorkspaceDeviceTensor();
    AllocateMemory(context);
  }
}

void KernelActor::AllocateMemory(OpContext<DeviceTensor> *context) {
  std::vector<DeviceTensor *> alloc_list(output_device_tensors_);
  alloc_list.insert(alloc_list.end(), workspace_device_tensors_.begin(), workspace_device_tensors_.end());
  Async(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, alloc_list, device_context_, context, GetAID());
}

void KernelActor::FreeMemory(OpContext<DeviceTensor> *context) {
  std::vector<DeviceTensor *> free_list(input_device_tensors_);
  free_list.insert(free_list.end(), output_device_tensors_.begin(), output_device_tensors_.end());
  free_list.insert(free_list.end(), workspace_device_tensors_.begin(), workspace_device_tensors_.end());
  Async(memory_manager_aid_, &MemoryManagerActor::FreeMemory, free_list, device_context_, context);
}

void KernelActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(kernel_);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel_);
  std::vector<AddressPtr> kernel_inputs;
  std::vector<AddressPtr> kernel_outputs;
  std::vector<AddressPtr> kernel_workspaces;
  FetchLaunchArgs(&kernel_inputs, &kernel_outputs, &kernel_workspaces);
  MS_EXCEPTION_IF_NULL(device_context_);
  auto ret = device_context_->LaunchKernel(kernel_mod, kernel_inputs, kernel_workspaces, kernel_outputs);
  if (!ret) {
    std::string error_info = "Launch kernel failed: " + kernel_->ToString();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }
  SendOutput(context);
  FreeMemory(context);
}

bool KernelActor::CheckLaunchCondition(OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  if (input_datas_num_ != 0) {
    auto data_iter = input_op_datas_.find(context->sequential_num_);
    if (data_iter == input_op_datas_.end()) {
      return false;
    }
    if (data_iter->second.size() != input_datas_num_) {
      return false;
    }
  }

  if (input_controls_num_ != 0) {
    auto control_iter = input_op_controls_.find(context->sequential_num_);
    if (control_iter == input_op_controls_.end()) {
      return false;
    }
    if (control_iter->second.size() != input_controls_num_) {
      return false;
    }
  }
  return true;
}

void KernelActor::FetchInputDeviceTensor(OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  auto input_size = input_datas_num_ + device_tensor_store_keys_.size();
  input_device_tensors_.resize(input_size);

  auto data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter != input_op_datas_.end()) {
    for (auto &input_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(input_data);
      input_device_tensors_[input_data->index_] = input_data->data_;
    }
  }

  for (auto &device_tensor_store_key : device_tensor_store_keys_) {
    auto device_tensor = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second);
    input_device_tensors_[device_tensor_store_key.first] = device_tensor.get();
  }
}

void KernelActor::FetchOutputDeviceTensor() {
  output_device_tensors_.clear();
  for (size_t i = 0; i < AnfAlgo::GetOutputTensorNum(kernel_); ++i) {
    auto device_address = AnfAlgo::GetMutableOutputAddr(kernel_, i, false);
    MS_EXCEPTION_IF_NULL(device_address);
    output_device_tensors_.emplace_back(device_address.get());
  }
}

void KernelActor::FetchWorkspaceDeviceTensor() {
  workspace_device_tensors_.clear();
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel_);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto workspace_sizes = kernel_mod->GetWorkspaceSizeList();
  for (size_t i = 0; i < workspace_sizes.size(); ++i) {
    if (workspace_sizes[i] != 0) {
      auto device_address = AnfAlgo::GetMutableWorkspaceAddr(kernel_, i);
      MS_EXCEPTION_IF_NULL(device_address);
      workspace_device_tensors_.emplace_back(device_address.get());
    }
  }
}

void KernelActor::FetchLaunchArgs(std::vector<AddressPtr> *kernel_inputs, std::vector<AddressPtr> *kernel_outputs,
                                  std::vector<AddressPtr> *kernel_workspaces) {
  MS_EXCEPTION_IF_NULL(kernel_inputs);
  MS_EXCEPTION_IF_NULL(kernel_outputs);
  MS_EXCEPTION_IF_NULL(kernel_workspaces);
  for (auto &input : input_device_tensors_) {
    MS_EXCEPTION_IF_NULL(input);
    kernel_inputs->emplace_back(std::make_shared<Address>(input->GetMutablePtr(), input->GetSize()));
  }

  for (auto &output : output_device_tensors_) {
    MS_EXCEPTION_IF_NULL(output);
    kernel_outputs->emplace_back(std::make_shared<Address>(output->GetMutablePtr(), output->GetSize()));
  }

  for (auto &workspace : workspace_device_tensors_) {
    MS_EXCEPTION_IF_NULL(workspace);
    kernel_workspaces->emplace_back(std::make_shared<Address>(workspace->GetMutablePtr(), workspace->GetSize()));
  }
}

void KernelActor::SendOutput(OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  // Send output data.
  for (auto &op_arrow : output_op_arrows_) {
    MS_EXCEPTION_IF_NULL(op_arrow);
    if (IntToSize(op_arrow->from_output_index_) >= output_device_tensors_.size()) {
      std::string error_info = "The output index is out of range: " + kernel_->ToString();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    auto device_address = output_device_tensors_[op_arrow->from_output_index_];
    auto data = std::make_shared<OpData<DeviceTensor>>(op_arrow->to_op_id_, device_address, op_arrow->to_input_index_);
    Async(op_arrow->to_op_id_, &KernelActor::RunOpData, data, context);
  }

  // Send output control.
  auto source_aid = const_cast<AID *>(&GetAID());
  for (auto &output_control : output_op_controls_) {
    Async(output_control, &OpActor::RunOpControl, source_aid, context);
  }
}

}  // namespace runtime
}  // namespace mindspore
