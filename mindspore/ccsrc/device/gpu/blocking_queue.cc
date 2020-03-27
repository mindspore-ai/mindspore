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
#include "dataset/util/make_unique.h"
#include "common/utils.h"

namespace mindspore {
namespace device {
GpuQueue::GpuQueue(void *addr, size_t feature_size, size_t label_size, size_t capacity)
    : buffer_(addr),
      head_(0),
      tail_(0),
      feature_size_(feature_size),
      label_size_(label_size),
      capacity_(capacity),
      stream_(0),
      node_info_(nullptr) {
  CHECK_CUDA_RET_WITH_ERROR(cudaStreamCreate(&stream_), "Cuda Create Stream Failed");
  node_info_ = mindspore::make_unique<NodeInfo[]>(capacity);
}

GpuQueue::~GpuQueue() { buffer_ = nullptr; }

BlockQueueStatus_T GpuQueue::Push(void *feature_addr, size_t feature_size, void *label_addr, size_t label_size) {
  if ((feature_addr == nullptr) || (label_addr == nullptr)) {
    MS_LOG(ERROR) << "input nullptr";
    return ERROR_INPUT;
  }
  if ((feature_size != feature_size_) || (label_size != label_size_)) {
    MS_LOG(ERROR) << "Data input error. Input data size: (" << feature_size << ", " << label_size << "), with ("
                  << feature_size_ << ", " << label_size_ << ") expect";
    return ERROR_INPUT;
  }
  void *feature_start_addr = reinterpret_cast<unsigned char *>(buffer_) + tail_ * (feature_size + label_size);
  if (feature_start_addr == nullptr) {
    MS_LOG(ERROR) << "feature start addr is nullptr";
    return INTERNAL_ERROR;
  }
  CHECK_CUDA_RET_WITH_ERROR(
    cudaMemcpyAsync(feature_start_addr, feature_addr, feature_size, cudaMemcpyHostToDevice, stream_),
    "Cuda Memcpy Error");
  void *label_start_addr = reinterpret_cast<unsigned char *>(feature_start_addr) + feature_size;
  if (label_start_addr == nullptr) {
    MS_LOG(ERROR) << "label start addr is nullptr";
    return INTERNAL_ERROR;
  }
  CHECK_CUDA_RET_WITH_ERROR(cudaMemcpyAsync(label_start_addr, label_addr, label_size, cudaMemcpyHostToDevice, stream_),
                            "Cuda Memcpy Error");
  node_info_[tail_].event_.reset(new cudaEvent_t());
  CHECK_CUDA_RET_WITH_ERROR(cudaEventCreate(&(*(node_info_[tail_].event_))), "Cuda Create Event Failed");
  node_info_[tail_].host_feature_addr_ = feature_addr;
  node_info_[tail_].host_label_addr_ = label_addr;
  tail_ = (tail_ + 1) % (capacity_);
  return SUCCESS;
}

BlockQueueStatus_T GpuQueue::Front(void **feature_addr, size_t *feature_size, void **label_addr,
                                   size_t *label_size) const {
  CHECK_CUDA_RET_WITH_ERROR(cudaEventSynchronize(*(node_info_[head_].event_)), "Cuda Event Syn Failed");
  CHECK_CUDA_RET_WITH_ERROR(cudaEventDestroy(*(node_info_[head_].event_)), "Cuda Destroy Event Failed");
  *feature_addr = (unsigned char *)buffer_ + head_ * (feature_size_ + label_size_);
  *feature_size = feature_size_;
  *label_addr = (unsigned char *)buffer_ + head_ * (feature_size_ + label_size_) + feature_size_;
  *label_size = label_size_;
  host_release_(node_info_[head_].host_feature_addr_);
  host_release_(node_info_[head_].host_label_addr_);
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

BlockQueueStatus_T BlockingQueue::Create(void *addr, size_t feature_size, size_t label_size, size_t capacity) {
  if (addr == nullptr) {
    MS_LOG(ERROR) << "addr is nullptr";
    return INTERNAL_ERROR;
  }
  queue_ = std::make_shared<GpuQueue>(addr, feature_size, label_size, capacity);
  return SUCCESS;
}

void BlockingQueue::RegisterRelease(const std::function<void(void *)> &func) { queue_->RegisterRelease(func); }

BlockQueueStatus_T BlockingQueue::Push(void *feature_addr, size_t feature_size, void *label_addr, size_t label_size,
                                       unsigned int timeout_in_sec) {
  std::unique_lock<std::mutex> locker(mutex_);
  if (queue_->IsFull()) {
    if (not_full_cond_.wait_for(locker, std::chrono::seconds(timeout_in_sec)) == std::cv_status::timeout) {
      return TIMEOUT;
    }
  }
  auto ret = queue_->Push(feature_addr, feature_size, label_addr, label_size);
  if (ret) {
    return ret;
  }
  not_empty_cond_.notify_one();
  return SUCCESS;
}

BlockQueueStatus_T BlockingQueue::Front(void **feature_addr, size_t *feature_size, void **label_addr,
                                        size_t *label_size) {
  std::unique_lock<std::mutex> locker(mutex_);
  bool timeout = not_empty_cond_.wait_for(locker, std::chrono::seconds(30), [this] { return !queue_->IsEmpty(); });
  if (!timeout) {
    return TIMEOUT;
  }

  return queue_->Front(feature_addr, feature_size, label_addr, label_size);
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
