/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/rl/tensors_queue_cpu_kernel.h"
#include <memory>
#include <chrono>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_tensor_array.h"
#include "runtime/device/tensor_array_manager.h"

namespace mindspore {
namespace kernel {
constexpr size_t kSecondInputIndex = 2;
constexpr int kRetryNumber = 10;
using mindspore::device::TensorsQueueMgr;
using mindspore::device::cpu::CPUTensorsQueue;
using mindspore::device::cpu::CPUTensorsQueuePtr;

// Init static mutex in base.
std::mutex TensorsQueueCPUBaseMod::tq_mutex_;
std::condition_variable TensorsQueueCPUBaseMod::read_cdv_;
std::condition_variable TensorsQueueCPUBaseMod::write_cdv_;
// Create a TensorsQueue.
TensorsQueueCreateCpuKernelMod::TensorsQueueCreateCpuKernelMod() : size_(0), elements_num_(0), type_(nullptr) {}

int TensorsQueueCreateCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  shapes_ = GetValue<std::vector<std::vector<int64_t>>>(primitive_->GetAttr("shapes"));
  type_ = GetValue<TypePtr>(primitive_->GetAttr("dtype"));
  size_ = GetValue<int64_t>(primitive_->GetAttr("size"));
  elements_num_ = GetValue<int64_t>(primitive_->GetAttr("elements_num"));
  name_ = GetValue<std::string>(primitive_->GetAttr("name"));
  output_size_list_.clear();
  output_size_list_.push_back(sizeof(int64_t));
  return KRET_OK;
}

bool TensorsQueueCreateCpuKernelMod::Launch(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
                                            const std::vector<KernelTensor *> &outputs) {
  // Create a TensorsQueue, and generate an unique handle.
  int64_t tensors_queue_handle = TensorsQueueMgr::GetInstance().GetHandleCount();
  auto name = "TensorsQueue_" + name_ + "_" + std::to_string(tensors_queue_handle);
  CPUTensorsQueuePtr tensors_queue = std::make_shared<CPUTensorsQueue>(name, type_, size_, elements_num_, shapes_);
  MS_EXCEPTION_IF_NULL(tensors_queue);
  // Malloc mem ahead for tensors queue.
  tensors_queue->CreateTensorsQueue();
  auto out_addr = GetDeviceAddress<int64_t>(outputs, 0);
  // Set handle to out_addr.
  MS_EXCEPTION_IF_NULL(out_addr);
  out_addr[0] = tensors_queue_handle;
  MS_LOG(DEBUG) << "Create handle id " << tensors_queue_handle;
  // Put the TensorsQueue to a saved map : map<handle, TensorsQueue> in TensorsQueue manager.
  // And increase the handle count automatically in AddTensorsQueue function.
  TensorsQueueMgr::GetInstance().AddTensorsQueue(tensors_queue_handle, tensors_queue);
  return true;
}

// Put one element into a TensorsQueue
TensorsQueuePutCpuKernelMod::TensorsQueuePutCpuKernelMod() {}

int TensorsQueuePutCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return KRET_OK;
  }
  output_size_list_.clear();
  output_size_list_.push_back(sizeof(int64_t));
  return KRET_OK;
}

bool TensorsQueuePutCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                         const std::vector<KernelTensor *> &) {
  CPUTensorsQueuePtr tensors_q = GetTensorsQueue(inputs);
  std::unique_lock<std::mutex> lock_(tq_mutex_);
  int retry_times = 0;
  std::vector<AddressPtr> inputs_address;
  for (size_t i = 0; i < inputs.size(); ++i) {
    AddressPtr input_address = std::make_shared<Address>(inputs[i]->device_ptr(), inputs[i]->size());
    inputs_address.push_back(input_address);
  }
  // If the tensors_q is full, put data will failed. So the op will sleep and waited to be notified.
  // Else if put succeed, we will notify all the warit op in read_cdv_.
  while (true) {
    if (!tensors_q->Put(inputs_address)) {
      if (write_cdv_.wait_for(lock_, std::chrono::seconds(kRetryNumber),
                              [this, tensors_q] { return !tensors_q->IsFull(); })) {
        retry_times++;
        MS_LOG(WARNING) << "Retry put data into TensorsQueue [" << retry_times << "/" << kRetryNumber << "].";
      }
      if (retry_times > kRetryNumber) {
        MS_LOG(EXCEPTION) << "Failed to put data after retried for " << kRetryNumber << " times.";
      }
    } else {
      MS_LOG(DEBUG) << "Put data succeed.";
      read_cdv_.notify_one();
      break;
    }
  }
  return true;
}

// Get or Pop one element from a TensorsQueue
TensorsQueueGetCpuKernelMod::TensorsQueueGetCpuKernelMod() : elements_num_(0), pop_after_get_(false) {}

int TensorsQueueGetCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  // Current all the tensor in one element must have the same type.
  TypePtr type = GetValue<TypePtr>(primitive_->GetAttr("dtype"));
  elements_num_ = GetValue<int64_t>(primitive_->GetAttr("elements_num"));
  pop_after_get_ = GetValue<bool>(primitive_->GetAttr("pop_after_get"));
  auto shapes = GetValue<std::vector<std::vector<int64_t>>>(primitive_->GetAttr("shapes"));

  output_size_list_.clear();
  for (int64_t i = 0; i < elements_num_; i++) {
    size_t value_size = GetTypeByte(type);
    for (auto x : shapes[LongToSize(i)]) {
      value_size *= LongToSize(x);
    }
    output_size_list_.push_back(value_size);
  }
  return KRET_OK;
}

bool TensorsQueueGetCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                         const std::vector<KernelTensor *> &outputs) {
  CPUTensorsQueuePtr tensors_q = GetTensorsQueue(inputs);
  std::unique_lock<std::mutex> lock_(tq_mutex_);
  std::vector<AddressPtr> outputs_address;
  for (size_t i = 0; i < outputs.size(); ++i) {
    AddressPtr output_address = std::make_shared<Address>(outputs[i]->device_ptr(), outputs[i]->size());
    outputs_address.push_back(output_address);
  }
  // Get one element from the head of tensors_q, if `pop_after_get` is true, then pop the tensors_q.
  // If the tensors_q is empty, get/pop failed, retry for max kRetryNumber times.
  int retry_times = 0;
  while (true) {
    if (!tensors_q->Get(outputs_address, pop_after_get_)) {
      if (read_cdv_.wait_for(lock_, std::chrono::seconds(kRetryNumber)) == std::cv_status::timeout) {
        retry_times++;
        MS_LOG(WARNING) << "Retry get data from TensorsQueue [" << retry_times << "/" << kRetryNumber << "].";
      }
      if (retry_times > kRetryNumber) {
        MS_LOG(EXCEPTION) << "Failed to get data after retried for " << kRetryNumber << " times.";
      }
    } else {
      MS_LOG(DEBUG) << "Get data succeed.";
      write_cdv_.notify_one();
      break;
    }
  }
  return true;
}  // namespace kernel

// Clear the TensorsQueue
TensorsQueueClearCpuKernelMod::TensorsQueueClearCpuKernelMod() {}

int TensorsQueueClearCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  output_size_list_.clear();
  output_size_list_.push_back(sizeof(int64_t));
  return KRET_OK;
}

bool TensorsQueueClearCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &) {
  CPUTensorsQueuePtr tensors_q = GetTensorsQueue(inputs);
  std::unique_lock<std::mutex> lock_(tq_mutex_);
  // Return all the element addr back to store, and the tensors_q will be empty.
  tensors_q->Clear();
  return true;
}

// Get size of the TensorsQueue
TensorsQueueSizeCpuKernelMod::TensorsQueueSizeCpuKernelMod() {}

int TensorsQueueSizeCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  output_size_list_.clear();
  output_size_list_.push_back(sizeof(int64_t));
  return KRET_OK;
}

bool TensorsQueueSizeCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &,
                                          const std::vector<KernelTensor *> &outputs) {
  CPUTensorsQueuePtr tensors_q = GetTensorsQueue(inputs);
  std::unique_lock<std::mutex> lock_(tq_mutex_);
  auto out_addr = GetDeviceAddress<int64_t>(outputs, 0);
  int64_t host_size = SizeToLong(tensors_q->AvailableSize());
  MS_EXCEPTION_IF_NULL(out_addr);
  out_addr[0] = host_size;
  return true;
}

// Close the TensorsQueue
TensorsQueueCloseCpuKernelMod::TensorsQueueCloseCpuKernelMod() {}

int TensorsQueueCloseCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  output_size_list_.clear();
  output_size_list_.push_back(sizeof(int64_t));
  return KRET_OK;
}

bool TensorsQueueCloseCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &) {
  auto handle_addr = GetDeviceAddress<int64_t>(inputs, 0);
  MS_ERROR_IF_NULL(handle_addr);
  auto tensors_q =
    std::dynamic_pointer_cast<CPUTensorsQueue>(TensorsQueueMgr::GetInstance().GetTensorsQueue(handle_addr[0]));
  MS_EXCEPTION_IF_NULL(tensors_q);
  // Free memory
  tensors_q->Free();
  // Erase TensorsQueue from the map.
  if (!TensorsQueueMgr::GetInstance().EraseTensorsQueue(handle_addr[0])) {
    MS_LOG(EXCEPTION) << "Close TensorsQueue failed";
  }
  return true;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorsQueueCreate, TensorsQueueCreateCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorsQueuePut, TensorsQueuePutCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorsQueueGet, TensorsQueueGetCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorsQueueClear, TensorsQueueClearCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorsQueueClose, TensorsQueueCloseCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorsQueueSize, TensorsQueueSizeCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
