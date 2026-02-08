/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_DATA_QUEUE_BLOCKING_QUEUE_H
#define MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_DATA_QUEUE_BLOCKING_QUEUE_H

#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <condition_variable>
#include <functional>
#include "include/runtime/hardware_abstract/data_queue/data_queue.h"
namespace mindspore {
namespace device {
class RUNTIME_HARDWARE_EXPORT BlockingQueue {
 public:
  BlockingQueue() : queue_(nullptr) {}
  ~BlockingQueue() = default;

  DataQueueStatus Create(const std::shared_ptr<DataQueue> &data_queue);
  void RegisterRelease(const std::function<void(void *, int32_t)> &func);
  DataQueueStatus Push(const std::vector<DataQueueItem> &data, unsigned int timeout = 0);
  DataQueueStatus Front(std::vector<DataQueueItem> *data);
  DataQueueStatus FrontAsync(std::vector<DataQueueItem> *data);
  DataQueueStatus Pop();
  DataQueueStatus Clear();
  void Close();
  bool IsOpen();
  size_t Size() { return queue_->Size(); }
  size_t Capacity() { return queue_->Capacity(); }
  const std::shared_ptr<DataQueue> &Queue() const { return queue_; }

 private:
  std::mutex mutex_;
  std::condition_variable not_full_cond_;
  std::condition_variable not_empty_cond_;
  std::shared_ptr<DataQueue> queue_;
  const size_t kPushTimeoutMicroseconds = 100;
  const size_t kPopTimeoutSeconds = 1;
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_DATA_QUEUE_BLOCKING_QUEUE_H
