/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "device/gpu/blocking_queue.h"
#include <chrono>
#include "device/gpu/gpu_common.h"
#include "common/utils.h"

namespace mindspore {
namespace device {
GpuQueue::GpuQueue(void *addr, const std::vector<size_t> &shape, const size_t &capacity)
    : buffer_(addr), head_(0), tail_(0), shape_(shape), len_(0), capacity_(capacity), stream_(0), node_info_(nullptr) {
  CHECK_CUDA_RET_WITH_ERROR(cudaStreamCreate(&stream_), "Cuda Create Stream Failed");
  node_info_ = std::make_unique<NodeInfo[]>(capacity);
  for (auto item : shape) {
    len_ += item;
  }
}

GpuQueue::~GpuQueue() { buffer_ = nullptr; }

BlockQueueStatus_T GpuQueue::Push(const std::vector<DataItemGpu> &data) {
  int offset = 0;
  for (size_t i = 0; i < data.size(); i++) {
    auto item = data[i];
    if (item.data_ptr_ == nullptr || item.data_len_ != shape_[i]) {
      MS_LOG(ERROR) << "Invalid Input: ptr: " << item.data_ptr_ << ", len: " << item.data_len_;
      return ERROR_INPUT;
    }

    void *addr = reinterpret_cast<unsigned char *>(buffer_) + tail_ * len_ + offset;
    CHECK_CUDA_RET_WITH_ERROR(cudaMemcpyAsync(addr, item.data_ptr_, item.data_len_, cudaMemcpyHostToDevice, stream_),
                              "Cuda Memcpy Error");

    offset += item.data_len_;
  }

  node_info_[tail_].event_.reset(new cudaEvent_t());
  CHECK_CUDA_RET_WITH_ERROR(cudaEventCreate(&(*(node_info_[tail_].event_))), "Cuda Create Event Failed");
  node_info_[tail_].data_ = data;
  tail_ = (tail_ + 1) % (capacity_);
  return SUCCESS;
}

BlockQueueStatus_T GpuQueue::Front(void **addr, size_t *len) const {
  CHECK_CUDA_RET_WITH_ERROR(cudaEventSynchronize(*(node_info_[head_].event_)), "Cuda Event Syn Failed");
  CHECK_CUDA_RET_WITH_ERROR(cudaEventDestroy(*(node_info_[head_].event_)), "Cuda Destroy Event Failed");
  *addr = (unsigned char *)buffer_ + head_ * len_;
  *len = len_;

  for (auto item : node_info_[head_].data_) {
    host_release_(item.data_ptr_);
  }
  return SUCCESS;
}

BlockQueueStatus_T GpuQueue::Pop() {
  head_ = (head_ + 1) % (capacity_);
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

BlockQueueStatus_T BlockingQueue::Create(void *addr, const std::vector<size_t> &shape, const size_t &capacity) {
  if (addr == nullptr) {
    MS_LOG(ERROR) << "addr is nullptr";
    return INTERNAL_ERROR;
  }
  queue_ = std::make_shared<GpuQueue>(addr, shape, capacity);
  return SUCCESS;
}

void BlockingQueue::RegisterRelease(const std::function<void(void *)> &func) { queue_->RegisterRelease(func); }

BlockQueueStatus_T BlockingQueue::Push(const std::vector<DataItemGpu> &data, unsigned int timeout_in_sec) {
  std::unique_lock<std::mutex> locker(mutex_);
  if (queue_->IsFull()) {
    if (not_full_cond_.wait_for(locker, std::chrono::seconds(timeout_in_sec)) == std::cv_status::timeout) {
      return TIMEOUT;
    }
  }
  auto ret = queue_->Push(data);
  if (ret) {
    return ret;
  }
  not_empty_cond_.notify_one();
  return SUCCESS;
}

BlockQueueStatus_T BlockingQueue::Front(void **addr, size_t *len) {
  std::unique_lock<std::mutex> locker(mutex_);
  bool timeout = not_empty_cond_.wait_for(locker, std::chrono::seconds(30), [this] { return !queue_->IsEmpty(); });
  if (!timeout) {
    return TIMEOUT;
  }

  return queue_->Front(addr, len);
}

BlockQueueStatus_T BlockingQueue::Pop() {
  std::unique_lock<std::mutex> locker(mutex_);
  not_empty_cond_.wait(locker, [this] { return !queue_->IsEmpty(); });
  auto ret = queue_->Pop();
  if (ret) {
    return ret;
  }
  not_full_cond_.notify_one();
  return SUCCESS;
}

bool BlockingQueue::Destroy() {
  if (queue_ != nullptr) {
    return queue_->Destroy();
  } else {
    return true;
  }
}
}  // namespace device
}  // namespace mindspore
