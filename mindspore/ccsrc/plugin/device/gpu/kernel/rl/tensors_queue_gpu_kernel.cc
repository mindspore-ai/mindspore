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

#include "plugin/device/gpu/kernel/rl/tensors_queue_gpu_kernel.h"
#include <memory>
#include <chrono>
#include <functional>
#include "kernel/common_utils.h"
#include "plugin/device/gpu/hal/device/gpu_tensor_array.h"
#include "runtime/device/tensor_array_manager.h"

namespace mindspore {
namespace kernel {
constexpr size_t kSecondInputIndex = 2;
constexpr int kRetryNumber = 10;
using mindspore::device::TensorsQueueMgr;
using mindspore::device::gpu::GPUTensorsQueue;
using mindspore::device::gpu::GPUTensorsQueuePtr;

// Init static mutex in base.
std::mutex TensorsQueueBaseMod::tq_mutex_;
std::condition_variable TensorsQueueBaseMod::read_cdv_;
std::condition_variable TensorsQueueBaseMod::write_cdv_;
// Create a TensorsQueue.
TensorsQueueCreateKernelMod::TensorsQueueCreateKernelMod() : size_(0), elements_num_(0), type_(nullptr) {}

bool TensorsQueueCreateKernelMod::Init(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_node_ = kernel_node;
  shapes_ = GetAttr<std::vector<std::vector<int64_t>>>(kernel_node, "shapes");
  type_ = GetAttr<TypePtr>(kernel_node, "dtype");
  size_ = GetAttr<int64_t>(kernel_node, "size");
  elements_num_ = GetAttr<int64_t>(kernel_node, "elements_num");
  name_ = GetAttr<std::string>(kernel_node, "name");

  output_size_list_.push_back(sizeof(int64_t));
  return true;
}

bool TensorsQueueCreateKernelMod::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                         const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  // Create a TensorsQueue, and generate an unique handle.
  int64_t tensors_queue_handle = TensorsQueueMgr::GetInstance().GetHandleCount();
  auto name = "TensorsQueue_" + name_ + "_" + std::to_string(tensors_queue_handle);
  GPUTensorsQueuePtr tensors_queue = std::make_shared<GPUTensorsQueue>(name, type_, size_, elements_num_, shapes_);
  MS_EXCEPTION_IF_NULL(tensors_queue);
  // Malloc mem ahead for tensors queue.
  tensors_queue->CreateTensorsQueue();
  auto out_addr = GetDeviceAddress<int64_t>(outputs, 0);
  // Set handle to out_addr.
  CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                             cudaMemcpyAsync(out_addr, &tensors_queue_handle, sizeof(int64_t), cudaMemcpyHostToDevice,
                                             reinterpret_cast<cudaStream_t>(stream_ptr)),
                             "Create TensorsQueue failed");
  MS_LOG(DEBUG) << "Create handle id " << tensors_queue_handle;
  // Put the TensorsQueue to a saved map : map<handle, TensorsQueue> in TensorsQueue manager.
  // And increase the handle count automatically in AddTensorsQueue function.
  TensorsQueueMgr::GetInstance().AddTensorsQueue(tensors_queue_handle, tensors_queue);
  return true;
}

// Put one element into a TensorsQueue
TensorsQueuePutKernelMod::TensorsQueuePutKernelMod() : elements_num_(0) {}

bool TensorsQueuePutKernelMod::Init(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_node_ = kernel_node;
  // Current all the tensor in one element should have the same type.
  type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kSecondInputIndex);
  elements_num_ = GetAttr<int64_t>(kernel_node, "elements_num");

  // Input[0] for handle.
  input_size_list_.push_back(sizeof(int64_t));
  // Input[1-n] : (Tensor(), Tensor(), ...)
  for (int64_t i = 0; i < elements_num_; i++) {
    auto shapes = AnfAlgo::GetInputDeviceShape(kernel_node, LongToSize(i + 1));
    auto value_size =
      std::accumulate(shapes.begin(), shapes.end(), GetTypeByte(TypeIdToType(type_)), std::multiplies<size_t>());
    input_size_list_.push_back(value_size);
  }
  output_size_list_.push_back(sizeof(int64_t));
  return true;
}

bool TensorsQueuePutKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                      const std::vector<AddressPtr> &, void *stream) {
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  GPUTensorsQueuePtr tensors_q = GetTensorsQueue(kernel_node_, inputs, cuda_stream);
  std::unique_lock<std::mutex> lock_(tq_mutex_);
  int retry_times = 0;
  // If the tensors_q is full, put data will failed. So the op will sleep and waited to be notified.
  // Else if put succeed, we will notify all the warit op in read_cdv_.
  while (true) {
    if (!tensors_q->Put(inputs, stream)) {
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
TensorsQueueGetKernelMod::TensorsQueueGetKernelMod() : elements_num_(0), pop_after_get_(false) {}

bool TensorsQueueGetKernelMod::Init(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_node_ = kernel_node;
  // Current all the tensor in one element should have the same type.
  TypePtr type = GetAttr<TypePtr>(kernel_node, "dtype");
  elements_num_ = GetAttr<int64_t>(kernel_node, "elements_num");
  pop_after_get_ = GetAttr<bool>(kernel_node, "pop_after_get");
  auto shapes = GetAttr<std::vector<std::vector<int64_t>>>(kernel_node, "shapes");

  // Input[0] for handle.
  input_size_list_.push_back(sizeof(int64_t));

  for (int64_t i = 0; i < elements_num_; i++) {
    size_t value_size = GetTypeByte(type) * SizeOf(shapes[i]);
    output_size_list_.push_back(value_size);
  }
  return true;
}

bool TensorsQueueGetKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                      const std::vector<AddressPtr> &outputs, void *stream) {
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  GPUTensorsQueuePtr tensors_q = GetTensorsQueue(kernel_node_, inputs, cuda_stream);
  std::unique_lock<std::mutex> lock_(tq_mutex_);
  // Get one element from the head of tensors_q, if `pop_after_get` is true, then pop the tensors_q.
  // If the tensors_q is empty, get/pop failed, retry for max kRetryNumber times.
  int retry_times = 0;
  while (true) {
    if (!tensors_q->Get(outputs, pop_after_get_, stream)) {
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
TensorsQueueClearKernelMod::TensorsQueueClearKernelMod() {}

bool TensorsQueueClearKernelMod::Init(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_node_ = kernel_node;
  // Input[0] for handle.
  input_size_list_.push_back(sizeof(int64_t));
  output_size_list_.push_back(sizeof(int64_t));
  return true;
}

bool TensorsQueueClearKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                        const std::vector<AddressPtr> &outputs, void *stream) {
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  GPUTensorsQueuePtr tensors_q = GetTensorsQueue(kernel_node_, inputs, cuda_stream);
  std::unique_lock<std::mutex> lock_(tq_mutex_);
  // Return all the element addr back to store, and the tensors_q will be empty.
  tensors_q->Clear();
  return true;
}

// Get size of the TensorsQueue
TensorsQueueSizeKernelMod::TensorsQueueSizeKernelMod() {}

bool TensorsQueueSizeKernelMod::Init(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_node_ = kernel_node;
  // Input[0] for handle.
  input_size_list_.push_back(sizeof(int64_t));
  output_size_list_.push_back(sizeof(int64_t));
  return true;
}

bool TensorsQueueSizeKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                       const std::vector<AddressPtr> &outputs, void *stream) {
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  GPUTensorsQueuePtr tensors_q = GetTensorsQueue(kernel_node_, inputs, cuda_stream);
  std::unique_lock<std::mutex> lock_(tq_mutex_);
  auto out_addr = GetDeviceAddress<int64_t>(outputs, 0);
  int64_t host_size = SizeToLong(tensors_q->AvailableSize());
  CHECK_CUDA_RET_WITH_EXCEPT(
    kernel_node_, cudaMemcpyAsync(out_addr, &host_size, sizeof(int64_t), cudaMemcpyHostToDevice, cuda_stream),
    "Set host size to device failed");
  return true;
}

// Close the TensorsQueue
TensorsQueueCloseKernelMod::TensorsQueueCloseKernelMod() {}

bool TensorsQueueCloseKernelMod::Init(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_node_ = kernel_node;
  // Input[0] for handle.
  input_size_list_.push_back(sizeof(int64_t));
  output_size_list_.push_back(sizeof(int64_t));
  return true;
}

bool TensorsQueueCloseKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                        const std::vector<AddressPtr> &outputs, void *stream) {
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  auto handle_addr = GetDeviceAddress<int64_t>(inputs, 0);
  MS_ERROR_IF_NULL(handle_addr);
  int64_t handle = 0;
  CHECK_CUDA_RET_WITH_EXCEPT(
    kernel_node_, cudaMemcpyAsync(&handle, handle_addr, sizeof(int64_t), cudaMemcpyDeviceToHost, cuda_stream),
    "Get handle to host failed");
  CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaStreamSynchronize(cuda_stream),
                             "TensorsQueueClose cudaStreamSynchronized failed");
  auto tensors_q = std::dynamic_pointer_cast<GPUTensorsQueue>(TensorsQueueMgr::GetInstance().GetTensorsQueue(handle));
  MS_EXCEPTION_IF_NULL(tensors_q);
  // Free memory
  tensors_q->Free();
  // Erase TensorsQueue from the map.
  if (!TensorsQueueMgr::GetInstance().EraseTensorsQueue(handle)) {
    MS_LOG(EXCEPTION) << "Close TensorsQueue failed";
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
