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

#include <string>
#include <memory>
#include <vector>
#include <functional>
#include "include/backend/visible.h"

namespace mindspore {
namespace device {
class DeviceContext;

enum class DataQueueStatus : int { SUCCESS = 0, QUEUE_EXIST, QUEUE_NOT_EXIST, ERROR_INPUT, INTERNAL_ERROR, TIMEOUT };

struct DataQueueItem {
  int32_t worker_id{0};
  std::string data_type;
  size_t data_len{0};
  void *data_ptr{nullptr};
  std::vector<int64_t> shapes;
  void *device_addr{nullptr};
  // add tensor type when tdt need more types than data and end-of-sequence
};

class BACKEND_EXPORT DataQueue {
 public:
  explicit DataQueue(const std::string &channel_name, const size_t capacity);
  virtual ~DataQueue() = default;

  virtual void RegisterRelease(const std::function<void(void *, int32_t)> &func) { host_release_ = func; }
  virtual bool IsOpen() const { return !closed_; }
  virtual void Close() { closed_ = true; }
  virtual bool IsEmpty() const { return size_ == 0; }
  virtual bool IsFull() const { return size_ == capacity_; }
  virtual DataQueueStatus FrontAsync(std::vector<DataQueueItem> *data) const { return DataQueueStatus::SUCCESS; }
  virtual DataQueueStatus Push(std::vector<DataQueueItem> data) = 0;
  virtual DataQueueStatus Front(std::vector<DataQueueItem> *data) const = 0;
  virtual DataQueueStatus Pop() = 0;
  virtual void SetThreadDevice() {}
  virtual size_t Size() const { return size_; }
  virtual size_t Capacity() const { return capacity_; }

 protected:
  const std::string channel_name_;
  size_t head_;
  size_t tail_;
  size_t size_;
  size_t capacity_;
  bool closed_{false};
  std::function<void(void *, int32_t)> host_release_;
  DeviceContext *device_context_;

 private:
  DataQueue(const DataQueue &) = delete;
  DataQueue &operator=(const DataQueue &) = delete;
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_BACKEND_DATA_QUEUE_DATA_QUEUE_H
