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

#include "plugin/device/gpu/hal/device/blocking_queue.h"
#include <chrono>
#include "plugin/device/gpu/hal/device/queue_common.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace device {
const size_t kTimeout = 100;

GpuQueue::GpuQueue(void *addr, const std::vector<size_t> &shape, const size_t &capacity)
    : buffer_(addr),
      head_(0),
      tail_(0),
      shape_(shape),
      len_(0),
      size_(0),
      capacity_(capacity),
      stream_(0),
      node_info_(nullptr) {
  CHECK_CUDA_RET_WITH_ERROR(cudaStreamCreate(&stream_), "Cuda Create Stream Failed");
  node_info_ = std::make_unique<NodeInfo[]>(capacity);
  for (auto item : shape) {
    len_ += item;
  }
}

GpuQueue::~GpuQueue() { buffer_ = nullptr; }

BlockQueueStatus_T GpuQueue::Push(std::vector<DataItemGpu> data) {
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

BlockQueueStatus_T GpuQueue::Front(std::vector<DataItemGpu> *data) const {
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

BlockQueueStatus_T BlockingQueue::Create(void *addr, const std::vector<size_t> &shape, const size_t &capacity) {
  if (addr == nullptr) {
    MS_LOG(ERROR) << "addr is nullptr";
    return INTERNAL_ERROR;
  }
  queue_ = std::make_shared<GpuQueue>(addr, shape, capacity);
  return SUCCESS;
}

void BlockingQueue::RegisterRelease(const std::function<void(void *, int32_t)> &func) { queue_->RegisterRelease(func); }

BlockQueueStatus_T BlockingQueue::Push(const std::vector<DataItemGpu> &data, unsigned int) {
  std::unique_lock<std::mutex> locker(mutex_);
  if (queue_->IsFull()) {
    if (not_full_cond_.wait_for(locker, std::chrono::microseconds(kTimeout)) == std::cv_status::timeout) {
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

BlockQueueStatus_T BlockingQueue::Front(std::vector<DataItemGpu> *data) {
  std::unique_lock<std::mutex> locker(mutex_);
  bool timeout = not_empty_cond_.wait_for(locker, std::chrono::seconds(30), [this] { return !queue_->IsEmpty(); });
  if (!timeout) {
    return TIMEOUT;
  }

  return queue_->Front(data);
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
