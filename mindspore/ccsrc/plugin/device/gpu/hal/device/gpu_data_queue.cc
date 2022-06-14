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
#include "plugin/device/gpu/hal/device/queue_common.h"
#include "utils/ms_context.h"
namespace mindspore {
namespace device {
GpuDataQueueDynamic::GpuDataQueueDynamic(const size_t capacity) : DataQueue(capacity), stream_(0), node_info_(nullptr) {
  CHECK_CUDA_RET_WITH_ERROR(cudaStreamCreate(&stream_), "Cuda Create Stream Failed");
  node_info_ = std::make_unique<NodeInfo[]>(capacity);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const std::string &device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  uint32_t device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  device_context_ = DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_target, device_id});
  device_context_->Initialize();
}

BlockQueueStatus_T GpuDataQueueDynamic::Push(std::vector<DataQueueItem> data) {
  for (size_t i = 0; i < data.size(); i++) {
    auto &item = data[i];
    if (item.data_ptr_ == nullptr) {
      MS_LOG(ERROR) << "Invalid Input: ptr: " << item.data_ptr_ << ", len: " << item.data_len_;
      return ERROR_INPUT;
    }
    void *addr = device_context_->AllocateMemory(item.data_len_);
    CHECK_CUDA_RET_WITH_ERROR(cudaMemcpyAsync(addr, item.data_ptr_, item.data_len_, cudaMemcpyHostToDevice, stream_),
                              "Cuda Memcpy Error");
    item.device_addr_ = addr;
  }

  node_info_[tail_].event_.reset(new cudaEvent_t());
  CHECK_CUDA_RET_WITH_ERROR(cudaEventCreate(&(*(node_info_[tail_].event_))), "Cuda Create Event Failed");
  CHECK_CUDA_RET_WITH_ERROR(cudaEventRecord(*(node_info_[tail_].event_), stream_), "Cuda Create Event Failed");
  node_info_[tail_].data_ = data;
  tail_ = (tail_ + 1) % (capacity_);
  ++size_;
  return SUCCESS;
}

BlockQueueStatus_T GpuDataQueueDynamic::Front(std::vector<DataQueueItem> *data) const {
  CHECK_CUDA_RET_WITH_ERROR(cudaEventSynchronize(*(node_info_[head_].event_)), "Cuda Event Syn Failed");
  CHECK_CUDA_RET_WITH_ERROR(cudaEventDestroy(*(node_info_[head_].event_)), "Cuda Destroy Event Failed");
  for (auto &item : node_info_[head_].data_) {
    host_release_(item.data_ptr_, item.worker_id_);
  }
  *data = node_info_[head_].data_;
  return SUCCESS;
}

BlockQueueStatus_T GpuDataQueueDynamic::Pop() {
  head_ = (head_ + 1) % (capacity_);
  --size_;
  return SUCCESS;
}

bool GpuDataQueueDynamic::Destroy() {
  if (stream_ != nullptr) {
    auto ret = cudaStreamDestroy(stream_);
    if (ret == cudaSuccess) {
      stream_ = nullptr;
      return true;
    } else {
      return false;
    }
  } else {
    return true;
  }
}

GpuQueue::GpuQueue(void *addr, const std::vector<size_t> &shape, const size_t &capacity)
    : DataQueue(capacity), buffer_(addr), shape_(shape), len_(0), stream_(0), node_info_(nullptr) {
  CHECK_CUDA_RET_WITH_ERROR(cudaStreamCreate(&stream_), "Cuda Create Stream Failed");
  node_info_ = std::make_unique<NodeInfo[]>(capacity);
  for (auto item : shape) {
    len_ += item;
  }
}

GpuQueue::~GpuQueue() { buffer_ = nullptr; }

BlockQueueStatus_T GpuQueue::Push(std::vector<DataQueueItem> data) {
  void *addr = reinterpret_cast<uint8_t *>(buffer_) + tail_ * len_;
  for (size_t i = 0; i < data.size(); i++) {
    auto &item = data[i];
    if (item.data_ptr_ == nullptr || item.data_len_ > shape_[i]) {
      MS_LOG(ERROR) << "Invalid Input: ptr: " << item.data_ptr_ << ", len: " << item.data_len_
                    << ", exceeds the max len: " << shape_[i];
      return ERROR_INPUT;
    }
    CHECK_CUDA_RET_WITH_ERROR(cudaMemcpyAsync(addr, item.data_ptr_, item.data_len_, cudaMemcpyHostToDevice, stream_),
                              "Cuda Memcpy Error");
    item.device_addr_ = addr;
    addr = reinterpret_cast<uint8_t *>(addr) + item.data_len_;
  }

  node_info_[tail_].event_.reset(new cudaEvent_t());
  CHECK_CUDA_RET_WITH_ERROR(cudaEventCreate(&(*(node_info_[tail_].event_))), "Cuda Create Event Failed");
  CHECK_CUDA_RET_WITH_ERROR(cudaEventRecord(*(node_info_[tail_].event_), stream_), "Cuda Create Event Failed");
  node_info_[tail_].data_ = data;
  tail_ = (tail_ + 1) % (capacity_);
  ++size_;
  return SUCCESS;
}

BlockQueueStatus_T GpuQueue::Front(std::vector<DataQueueItem> *data) const {
  CHECK_CUDA_RET_WITH_ERROR(cudaEventSynchronize(*(node_info_[head_].event_)), "Cuda Event Syn Failed");
  CHECK_CUDA_RET_WITH_ERROR(cudaEventDestroy(*(node_info_[head_].event_)), "Cuda Destroy Event Failed");
  for (auto &item : node_info_[head_].data_) {
    host_release_(item.data_ptr_, item.worker_id_);
  }
  *data = node_info_[head_].data_;
  return SUCCESS;
}

BlockQueueStatus_T GpuQueue::Pop() {
  head_ = (head_ + 1) % (capacity_);
  --size_;
  return SUCCESS;
}

bool GpuQueue::Destroy() {
  if (stream_ != nullptr) {
    auto ret = cudaStreamDestroy(stream_);
    if (ret == cudaSuccess) {
      return true;
    } else {
      return false;
    }
  } else {
    return true;
  }
}
}  // namespace device
}  // namespace mindspore
