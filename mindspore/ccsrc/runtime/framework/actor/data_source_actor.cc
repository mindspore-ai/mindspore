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

#include "runtime/framework/actor/data_source_actor.h"
#include "runtime/framework/actor/kernel_actor.h"
#include "runtime/framework/actor/memory_manager_actor.h"
#include "mindrt/include/async/async.h"
#include "common/trans.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
void DataSourceActor::FetchData(OpContext<DeviceTensor> *context) {
  MS_LOG(INFO) << "Data source actor(" << GetAID().Name() << ") fetches data.";
  MS_EXCEPTION_IF_NULL(context);
  if (buffers_.size() == buffer_capacity_) {
    // Send output to trigger computing and free memory.
    SendOutput(context);
    FreeMemory(context);
    buffers_.pop();
    return;
  }

  // Construct device tensors and fill to the buffers from member nodes.
  FillDataBuffer();
  if (buffers_.size() == 0) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The data queue is empty.");
  }

  // Allocate memory for device tensors.
  AllocateMemory(context);
}

void DataSourceActor::AllocateMemory(OpContext<DeviceTensor> *context) {
  auto device_tensors = buffers_.back();
  Async(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, device_tensors, device_context_, context, GetAID());
}

void DataSourceActor::FreeMemory(OpContext<DeviceTensor> *context) {
  auto device_tensors = buffers_.front();
  Async(memory_manager_aid_, &MemoryManagerActor::FreeMemory, device_tensors, device_context_, context);
}

void DataSourceActor::SendOutput(OpContext<DeviceTensor> *context) {
  MS_LOG(INFO) << "Data source actor(" << GetAID().Name() << ") sends output data.";
  MS_EXCEPTION_IF_NULL(context);
  if (buffers_.size() == 0) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The data queue is empty.");
  }

  // Send output data.
  auto output_device_tensors = buffers_.front();
  for (auto &op_arrow : output_op_arrows_) {
    MS_EXCEPTION_IF_NULL(op_arrow);
    if (IntToSize(op_arrow->from_output_index_) >= output_device_tensors.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The output index is of range.");
    }
    auto device_address = output_device_tensors[op_arrow->from_output_index_];
    auto data = std::make_shared<OpData<DeviceTensor>>(op_arrow->to_op_id_, device_address, op_arrow->to_input_index_);
    Async(op_arrow->to_op_id_, &KernelActor::RunOpData, data, context);
  }
}

void DeviceQueueDataSourceActor::FillDataBuffer() {
  // Construct device tensors.
  std::vector<DeviceTensor *> device_tensors;
  for (size_t i = 0; i < AnfAlgo::GetOutputTensorNum(data_kernel_); ++i) {
    auto device_address = AnfAlgo::GetMutableOutputAddr(data_kernel_, i, false);
    MS_EXCEPTION_IF_NULL(device_address);
    device_tensors.emplace_back(device_address.get());
  }

  buffers_.push(device_tensors);
}

void DeviceQueueDataSourceActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(device_context_);
  if (buffers_.size() == 0) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The data queue is empty.");
  }

  // Construct outputs of data kernel launching.
  auto device_tensors = buffers_.back();
  std::vector<AddressPtr> kernel_outputs;
  for (auto &device_tensor : device_tensors) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    kernel_outputs.emplace_back(std::make_shared<Address>(device_tensor->GetMutablePtr(), device_tensor->GetSize()));
  }

  // Copy data from device queue by data kernel launching.
  std::vector<AddressPtr> empty_address;
  auto kernel_mod = AnfAlgo::GetKernelMod(data_kernel_);
  auto ret = device_context_->LaunchKernel(kernel_mod, empty_address, empty_address, kernel_outputs);
  if (!ret) {
    std::string error_info = "Launch kernel failed: " + data_kernel_->ToString();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }

  // Send output to trigger computing and free memory.
  SendOutput(context);
  FreeMemory(context);
  buffers_.pop();
}

void HostQueueDataSourceActor::FillDataBuffer() {
  // Construct device tensors.
  std::vector<DeviceTensor *> device_tensors;
  for (auto &data_node : data_nodes_) {
    auto device_address = AnfAlgo::GetMutableOutputAddr(data_node, 0, false);
    MS_EXCEPTION_IF_NULL(device_address);
    device_tensors.emplace_back(device_address.get());
  }

  buffers_.push(device_tensors);
}

void HostQueueDataSourceActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  if (buffers_.size() == 0) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The data queue is empty.");
  }

  // Get host tensors from host queue and get device tensors from buffers.
  MS_EXCEPTION_IF_NULL(host_queue_);
  auto host_tensors = host_queue_->PullData();
  auto device_tensors = buffers_.back();
  if (host_tensors.size() != device_tensors.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context),
                                      "The length of host tensors is not equal to the length of device tensors.");
  }

  // Copy data from host tensor to device tensor.
  for (size_t i = 0; i < host_tensors.size(); ++i) {
    auto host_tensor = host_tensors[i];
    auto device_tensor = device_tensors[i];
    MS_EXCEPTION_IF_NULL(host_tensor);
    MS_EXCEPTION_IF_NULL(device_tensor);
    if (!device_tensor->SyncHostToDevice(trans::GetRuntimePaddingShape(data_nodes_[i], 0),
                                         LongToSize(host_tensor->data().nbytes()), host_tensor->data_type(),
                                         host_tensor->data_c())) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "SyncHostToDevice failed.");
    }
  }

  // Send output to trigger computing and free memory.
  SendOutput(context);
  FreeMemory(context);
  buffers_.pop();
}
}  // namespace runtime
}  // namespace mindspore
