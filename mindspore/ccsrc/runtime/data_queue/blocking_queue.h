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
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <condition_variable>
#include <functional>
#include "runtime/data_queue/data_queue.h"
namespace mindspore {
namespace device {
class BlockingQueue {
 public:
  BlockingQueue() : queue_(nullptr) {}
  ~BlockingQueue() = default;

  BlockQueueStatus_T Create(const std::shared_ptr<DataQueue> &data_queue);
  void RegisterRelease(const std::function<void(void *, int32_t)> &func);
  BlockQueueStatus_T Push(const std::vector<DataQueueItem> &data, unsigned int timeout_in_sec);
  BlockQueueStatus_T Front(std::vector<DataQueueItem> *data);
  BlockQueueStatus_T Pop();
  BlockQueueStatus_T Clear();
  bool Destroy();
  size_t Size() { return queue_->Size(); }
  size_t Capacity() { return queue_->Capacity(); }

 private:
  std::mutex mutex_;
  std::condition_variable not_full_cond_;
  std::condition_variable not_empty_cond_;
  std::shared_ptr<DataQueue> queue_;
};
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_BLOCKING_QUEUE_H_
