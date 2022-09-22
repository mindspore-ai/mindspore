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
#include "plugin/device/gpu/hal/device/gpu_data_queue.h"
#include <string>
#include <utility>
#include "include/backend/data_queue/data_queue_mgr.h"
#include "utils/ms_context.h"
#include "plugin/device/gpu/hal/device/queue_common.h"
#include "plugin/device/gpu/hal/device/gpu_device_manager.h"

namespace mindspore {
namespace device {
namespace {
void SetThreadToDeviceId(uint32_t device_id) {
  // Without cudaSetDevice cuda memory will allocate on GPU:0 as default
  // and will overload in distribute scenario.
  auto ret = cudaSetDevice(device_id);
  if (ret != cudaSuccess) {
    MS_LOG(EXCEPTION) << "cudaSetDevice failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
  }
}
}  // namespace
GpuDataQueueDynamic::GpuDataQueueDynamic(const std::string &channel_name, const size_t capacity)
    : DataQueue(channel_name, capacity), node_info_(nullptr) {
  node_info_ = std::make_unique<NodeInfo[]>(capacity);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const std::string &device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device_id_ = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  device_context_ = DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_target, device_id_});
  device_context_->Initialize();
  stream_ = reinterpret_cast<cudaStream_t>(gpu::GPUDeviceManager::GetInstance().default_stream());
}

DataQueueStatus GpuDataQueueDynamic::Push(std::vector<DataQueueItem> data) {
  if (data.empty()) {
    return DataQueueStatus::SUCCESS;
  }

  if (IsFull()) {
    return DataQueueStatus::TIMEOUT;
  }

  for (size_t i = 0; i < data.size(); i++) {
    auto &item = data[i];
    if (item.data_ptr == nullptr) {
      MS_LOG(ERROR) << "Invalid Input: ptr: " << item.data_ptr << ", len: " << item.data_len;
      return DataQueueStatus::ERROR_INPUT;
    }
    void *addr = device_context_->device_res_manager_->AllocateMemory(item.data_len);
    CHECK_CUDA_RET_WITH_ERROR(cudaMemcpyAsync(addr, item.data_ptr, item.data_len, cudaMemcpyHostToDevice, stream_),
                              "Cuda Memcpy Error");
    item.device_addr = addr;
  }

  node_info_[tail_].event_.reset(new cudaEvent_t());
  CHECK_CUDA_RET_WITH_ERROR(cudaEventCreate(&(*(node_info_[tail_].event_))), "Cuda Create Event Failed");
  CHECK_CUDA_RET_WITH_ERROR(cudaEventRecord(*(node_info_[tail_].event_), stream_), "Cuda Create Event Failed");
  node_info_[tail_].data_ = std::move(data);
  tail_ = (tail_ + 1) % (capacity_);
  ++size_;
  return DataQueueStatus::SUCCESS;
}

DataQueueStatus GpuDataQueueDynamic::Front(std::vector<DataQueueItem> *data) const {
  CHECK_CUDA_RET_WITH_ERROR(cudaEventSynchronize(*(node_info_[head_].event_)), "Cuda Event Syn Failed");
  CHECK_CUDA_RET_WITH_ERROR(cudaEventDestroy(*(node_info_[head_].event_)), "Cuda Destroy Event Failed");
  for (auto &item : node_info_[head_].data_) {
    host_release_(item.data_ptr, item.worker_id);
  }
  *data = node_info_[head_].data_;
  return DataQueueStatus::SUCCESS;
}

DataQueueStatus GpuDataQueueDynamic::Pop() {
  head_ = (head_ + 1) % (capacity_);
  --size_;
  return DataQueueStatus::SUCCESS;
}

void GpuDataQueueDynamic::SetThreadDevice() { SetThreadToDeviceId(device_id_); }

GpuQueue::GpuQueue(const std::string &channel_name, size_t capacity, const std::vector<size_t> &shape)
    : DataQueue(channel_name, capacity), shape_(shape), len_(0), stream_(0), node_info_(nullptr) {
  if (channel_name.empty()) {
    return;
  }
  CHECK_CUDA_RET_WITH_ERROR(cudaStreamCreate(&stream_), "Cuda Create Stream Failed");
  node_info_ = std::make_unique<NodeInfo[]>(capacity);
  for (auto item : shape) {
    len_ += item;
  }
  size_t buffer_size = len_ * capacity;
  if (!device::gpu::GPUMemoryAllocator::GetInstance().AllocBufferQueueMem(buffer_size, &buffer_)) {
    MS_LOG(EXCEPTION) << "Memory not enough: failed to allocate GPU buffer queue memory[" << buffer_size << "].";
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  device_id_ = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
}

GpuQueue::~GpuQueue() {
  if (buffer_ != nullptr && !device::gpu::GPUMemoryAllocator::GetInstance().FreeDeviceMem(buffer_)) {
    MS_LOG(ERROR) << "Free GPU data queue memory failed";
  }
  if (stream_ != nullptr && cudaStreamDestroy(stream_) != cudaSuccess) {
    MS_LOG(ERROR) << "Destroy the stream of GPU data queue failed";
  }
}

DataQueueStatus GpuQueue::Push(std::vector<DataQueueItem> data) {
  if (IsFull()) {
    return DataQueueStatus::TIMEOUT;
  }
  void *addr = reinterpret_cast<uint8_t *>(buffer_) + tail_ * len_;
  for (size_t i = 0; i < data.size(); i++) {
    auto &item = data[i];
    MS_EXCEPTION_IF_NULL(item.data_ptr);
    if (item.data_len != shape_[i]) {
      ds_detected_ = true;
      ++size_;
      MS_LOG(ERROR) << "Detected that dataset is dynamic shape, you need to call network.set_inputs() to "
                       "configure dynamic dims of input data before running the network";
      return DataQueueStatus::ERROR_INPUT;
    }
    CHECK_CUDA_RET_WITH_ERROR(cudaMemcpyAsync(addr, item.data_ptr, item.data_len, cudaMemcpyHostToDevice, stream_),
                              "Cuda Memcpy Error");
    item.device_addr = addr;
    addr = reinterpret_cast<uint8_t *>(addr) + item.data_len;
  }

  node_info_[tail_].event_.reset(new cudaEvent_t());
  CHECK_CUDA_RET_WITH_ERROR(cudaEventCreate(&(*(node_info_[tail_].event_))), "Cuda Create Event Failed");
  CHECK_CUDA_RET_WITH_ERROR(cudaEventRecord(*(node_info_[tail_].event_), stream_), "Cuda Create Event Failed");
  node_info_[tail_].data_ = std::move(data);
  tail_ = (tail_ + 1) % (capacity_);
  ++size_;
  return DataQueueStatus::SUCCESS;
}

DataQueueStatus GpuQueue::Front(std::vector<DataQueueItem> *data) const {
  if (ds_detected_) {
    return DataQueueStatus::ERROR_INPUT;
  }
  CHECK_CUDA_RET_WITH_ERROR(cudaEventSynchronize(*(node_info_[head_].event_)), "Cuda Event Syn Failed");
  CHECK_CUDA_RET_WITH_ERROR(cudaEventDestroy(*(node_info_[head_].event_)), "Cuda Destroy Event Failed");
  for (auto &item : node_info_[head_].data_) {
    host_release_(item.data_ptr, item.worker_id);
  }
  *data = node_info_[head_].data_;
  return DataQueueStatus::SUCCESS;
}

DataQueueStatus GpuQueue::Pop() {
  head_ = (head_ + 1) % (capacity_);
  --size_;
  return DataQueueStatus::SUCCESS;
}

void GpuQueue::SetThreadDevice() { SetThreadToDeviceId(device_id_); }

GpuDataQueue::GpuDataQueue(const std::string &channel_name, size_t capacity, const std::vector<size_t> &shape)
    : DataQueue(channel_name, capacity), shape_(shape), len_(0), stream_(0), node_info_(nullptr) {
  CHECK_CUDA_RET_WITH_ERROR(cudaStreamCreate(&stream_), "Cuda Create Stream Failed");
  node_info_ = std::make_unique<NodeInfo[]>(capacity);
  for (auto item : shape) {
    len_ += item;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  device_id_ = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
}

GpuDataQueue::~GpuDataQueue() {
  for (size_t i = 0; i < capacity_; ++i) {
    void *device_addr = node_info_[i].device_addr_;
    if (device_addr != nullptr && !device::gpu::GPUMemoryAllocator::GetInstance().FreeDeviceMem(device_addr)) {
      MS_LOG(ERROR) << "Free GPU data queue memory failed";
    }
  }
  if (stream_ != nullptr && cudaStreamDestroy(stream_) != cudaSuccess) {
    MS_LOG(ERROR) << "Destroy the stream of GPU data queue failed";
  }
}

DataQueueStatus GpuDataQueue::Push(std::vector<DataQueueItem> data) {
  if (data.empty()) {
    return DataQueueStatus::SUCCESS;
  }
  if (IsFull()) {
    return DataQueueStatus::TIMEOUT;
  }
  size_t data_len =
    std::accumulate(data.begin(), data.end(), 0, [](size_t accum, const auto &it) { return accum + it.data_len; });
  void *device_addr = nullptr;
  if (data_len > node_info_[tail_].data_len_) {
    auto &mem_allocator = device::gpu::GPUMemoryAllocator::GetInstance();
    if (node_info_[tail_].device_addr_ != nullptr) {
      MS_EXCEPTION_IF_CHECK_FAIL(mem_allocator.FreeDeviceMem(node_info_[tail_].device_addr_),
                                 "Cuda Free Memory Failed");
      node_info_[tail_].device_addr_ = nullptr;
    }
    MS_EXCEPTION_IF_CHECK_FAIL(mem_allocator.AllocBufferQueueMem(data_len, &device_addr),
                               "Allocate Data Queue Memory Failed");
    node_info_[tail_].device_addr_ = device_addr;
    node_info_[tail_].data_len_ = data_len;
  } else {
    device_addr = node_info_[tail_].device_addr_;
  }

  for (auto &item : data) {
    if (item.data_ptr == nullptr) {
      MS_LOG(ERROR) << "Invalid Input: ptr: " << item.data_ptr << ", len: " << item.data_len;
      return DataQueueStatus::ERROR_INPUT;
    }
    CHECK_CUDA_RET_WITH_ERROR(
      cudaMemcpyAsync(device_addr, item.data_ptr, item.data_len, cudaMemcpyHostToDevice, stream_), "Cuda Memcpy Error");
    item.device_addr = device_addr;
    device_addr = reinterpret_cast<uint8_t *>(device_addr) + item.data_len;
  }

  node_info_[tail_].event_.reset(new cudaEvent_t());
  CHECK_CUDA_RET_WITH_ERROR(cudaEventCreate(&(*(node_info_[tail_].event_))), "Cuda Create Event Failed");
  CHECK_CUDA_RET_WITH_ERROR(cudaEventRecord(*(node_info_[tail_].event_), stream_), "Cuda Create Event Failed");
  node_info_[tail_].data_ = std::move(data);
  tail_ = (tail_ + 1) % (capacity_);
  ++size_;
  return DataQueueStatus::SUCCESS;
}

DataQueueStatus GpuDataQueue::FrontAsync(std::vector<DataQueueItem> *data) const {
  *data = node_info_[head_].data_;
  return DataQueueStatus::SUCCESS;
}

DataQueueStatus GpuDataQueue::Front(std::vector<DataQueueItem> *data) const {
  CHECK_CUDA_RET_WITH_ERROR(cudaEventSynchronize(*(node_info_[head_].event_)), "Cuda Event Syn Failed");
  CHECK_CUDA_RET_WITH_ERROR(cudaEventDestroy(*(node_info_[head_].event_)), "Cuda Destroy Event Failed");
  for (auto &item : node_info_[head_].data_) {
    host_release_(item.data_ptr, item.worker_id);
  }
  *data = node_info_[head_].data_;
  return DataQueueStatus::SUCCESS;
}

DataQueueStatus GpuDataQueue::Pop() {
  head_ = (head_ + 1) % (capacity_);
  --size_;
  return DataQueueStatus::SUCCESS;
}

void GpuDataQueue::SetThreadDevice() { SetThreadToDeviceId(device_id_); }

namespace {
std::shared_ptr<DataQueue> CreateGpuDataQueue(const std::string &channel_name, bool dynamic_shape, size_t capacity,
                                              const std::vector<size_t> &shape) {
  return std::make_shared<GpuDataQueue>(channel_name, capacity, shape);
}

REGISTER_DATA_QUEUE_CREATOR(kGPUDevice, CreateGpuDataQueue);
}  // namespace
}  // namespace device
}  // namespace mindspore
