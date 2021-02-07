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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_BLOCKING_QUEUE_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_BLOCKING_QUEUE_H_

#include <unistd.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <cstring>
#include <string>
#include <vector>
#include <condition_variable>
#include <functional>

namespace mindspore {
namespace device {
enum BlockQueueStatus_T : int { SUCCESS = 0, QUEUE_NOT_EXIST, HANDLE_NOT_EXIST, ERROR_INPUT, INTERNAL_ERROR, TIMEOUT };

struct DataItemGpu {
  int32_t worker_id_;
  std::string data_type_;
  size_t data_len_;
  void *data_ptr_;
};

class GpuQueue {
 public:
  GpuQueue(void *addr, const std::vector<size_t> &shape, const size_t &capacity);
  virtual ~GpuQueue();

  void RegisterRelease(const std::function<void(void *, int32_t)> &func) { host_release_ = func; }

  inline bool IsEmpty() const { return size_ == 0; }
  inline bool IsFull() const { return size_ == capacity_; }

  BlockQueueStatus_T Push(const std::vector<DataItemGpu> &data);
  BlockQueueStatus_T Front(void **ptr, size_t *len) const;
  BlockQueueStatus_T Pop();
  bool Destroy();
  size_t Size() { return size_; }
  size_t Capacity() { return capacity_; }

 private:
  struct NodeInfo {
    std::unique_ptr<cudaEvent_t> event_;
    std::vector<DataItemGpu> data_;
  };

  void *buffer_;
  size_t head_;
  size_t tail_;
  std::vector<size_t> shape_;
  size_t len_;
  size_t size_;
  size_t capacity_;
  cudaStream_t stream_;
  std::unique_ptr<NodeInfo[]> node_info_;
  std::function<void(void *, int32_t)> host_release_;

  GpuQueue(const GpuQueue &) = delete;
  GpuQueue &operator=(const GpuQueue &) = delete;
};

class BlockingQueue {
 public:
  BlockingQueue() : queue_(nullptr) {}
  ~BlockingQueue() = default;

  BlockQueueStatus_T Create(void *addr, const std::vector<size_t> &shape, const size_t &capacity);
  void RegisterRelease(const std::function<void(void *, int32_t)> &func);
  BlockQueueStatus_T Push(const std::vector<DataItemGpu> &data, unsigned int timeout_in_sec);
  BlockQueueStatus_T Front(void **ptr, size_t *len);
  BlockQueueStatus_T Pop();
  bool Destroy();
  size_t Size() { return queue_->Size(); }
  size_t Capacity() { return queue_->Capacity(); }

 private:
  std::mutex mutex_;
  std::condition_variable not_full_cond_;
  std::condition_variable not_empty_cond_;
  std::shared_ptr<GpuQueue> queue_;
};
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_BLOCKING_QUEUE_H_
