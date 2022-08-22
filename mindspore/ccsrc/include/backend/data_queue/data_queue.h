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

#ifndef MINDSPORE_CCSRC_INCLUDE_BACKEND_DATA_QUEUE_DATA_QUEUE_H
#define MINDSPORE_CCSRC_INCLUDE_BACKEND_DATA_QUEUE_DATA_QUEUE_H

#include <unistd.h>
#include <string>
#include <memory>
#include <vector>
#include <functional>

namespace mindspore {
namespace device {
class DeviceContext;

enum class DataQueueStatus : int { SUCCESS = 0, QUEUE_EXIST, QUEUE_NOT_EXIST, ERROR_INPUT, INTERNAL_ERROR, TIMEOUT };

struct DataQueueItem {
  int32_t worker_id_{0};
  std::string data_type_;
  size_t data_len_{0};
  void *data_ptr_{nullptr};
  std::vector<int64_t> shapes_;
  void *device_addr_{nullptr};
};

class DataQueue {
 public:
  explicit DataQueue(const size_t capacity);
  virtual ~DataQueue() = default;

  virtual void RegisterRelease(const std::function<void(void *, int32_t)> &func) { host_release_ = func; }

  virtual bool IsOpen() const { return true; }
  virtual bool IsEmpty() const { return size_ == 0; }
  virtual bool IsFull() const { return size_ == capacity_; }

  virtual DataQueueStatus Push(std::vector<DataQueueItem> data) = 0;
  virtual DataQueueStatus Front(std::vector<DataQueueItem> *data) const = 0;
  virtual DataQueueStatus Pop() = 0;
  virtual bool Destroy() = 0;
  virtual void SetThreadDevice() = 0;

  virtual size_t Size() { return size_; }
  virtual size_t Capacity() { return capacity_; }

  virtual std::shared_ptr<void> AllocHostMem(size_t size) { return std::shared_ptr<void>(::malloc(size), ::free); }

 protected:
  size_t head_;
  size_t tail_;
  size_t size_;
  size_t capacity_;

  std::function<void(void *, int32_t)> host_release_;
  DeviceContext *device_context_;

 private:
  DataQueue(const DataQueue &) = delete;
  DataQueue &operator=(const DataQueue &) = delete;
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_BACKEND_DATA_QUEUE_DATA_QUEUE_H
