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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_DATA_QUEUE_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_DATA_QUEUE_H_

#include <unistd.h>
#include <string>
#include <memory>
#include <vector>
#include <functional>
#include "runtime/hardware/device_context_manager.h"
#include "mindspore/core/utils/data_queue_handler.h"

namespace mindspore {
namespace device {
class DataQueue {
 public:
  explicit DataQueue(const size_t capacity);
  virtual ~DataQueue() = default;

  virtual void RegisterRelease(const std::function<void(void *, int32_t)> &func) { host_release_ = func; }

  virtual bool IsEmpty() const { return size_ == 0; }
  virtual bool IsFull() const { return size_ == capacity_; }

  virtual BlockQueueStatus_T Push(std::vector<DataQueueItem> data) = 0;
  virtual BlockQueueStatus_T Front(std::vector<DataQueueItem> *data) const = 0;
  virtual BlockQueueStatus_T Pop() = 0;
  virtual bool Destroy() = 0;
  virtual size_t Size() { return size_; }
  virtual size_t Capacity() { return capacity_; }

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

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_BLOCKING_QUEUE_H_
