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

#include "include/backend/data_queue/blocking_queue.h"

namespace mindspore {
namespace device {
void BlockingQueue::RegisterRelease(const std::function<void(void *, int32_t)> &func) { queue_->RegisterRelease(func); }

DataQueueStatus BlockingQueue::Push(const std::vector<DataQueueItem> &data, unsigned int) {
  std::unique_lock<std::mutex> locker(mutex_);
  if (queue_->IsFull()) {
    if (not_full_cond_.wait_for(locker, std::chrono::microseconds(kPushTimeoutMicroseconds)) ==
        std::cv_status::timeout) {
      return DataQueueStatus::TIMEOUT;
    }
  }
  auto ret = queue_->Push(data);
  if (ret != DataQueueStatus::SUCCESS) {
    return ret;
  }
  not_empty_cond_.notify_one();
  return DataQueueStatus::SUCCESS;
}

DataQueueStatus BlockingQueue::Front(std::vector<DataQueueItem> *data) {
  std::unique_lock<std::mutex> locker(mutex_);
  bool timeout =
    not_empty_cond_.wait_for(locker, std::chrono::seconds(kPopTimeoutSeconds), [this] { return !queue_->IsEmpty(); });
  if (!timeout) {
    return DataQueueStatus::TIMEOUT;
  }
  return queue_->Front(data);
}

DataQueueStatus BlockingQueue::FrontAsync(std::vector<DataQueueItem> *data) {
  std::unique_lock<std::mutex> locker(mutex_);
  bool timeout =
    not_empty_cond_.wait_for(locker, std::chrono::seconds(kPopTimeoutSeconds), [this] { return !queue_->IsEmpty(); });
  if (!timeout) {
    return DataQueueStatus::TIMEOUT;
  }
  return queue_->FrontAsync(data);
}

DataQueueStatus BlockingQueue::Pop() {
  std::unique_lock<std::mutex> locker(mutex_);
  not_empty_cond_.wait(locker, [this] { return !queue_->IsEmpty(); });
  auto ret = queue_->Pop();
  if (ret != DataQueueStatus::SUCCESS) {
    return ret;
  }
  not_full_cond_.notify_one();
  return DataQueueStatus::SUCCESS;
}

DataQueueStatus BlockingQueue::Create(const std::shared_ptr<DataQueue> &data_queue) {
  this->queue_ = data_queue;
  return DataQueueStatus::SUCCESS;
}

void BlockingQueue::Close() {
  std::unique_lock<std::mutex> locker(mutex_);
  queue_->Close();
}

bool BlockingQueue::IsOpen() {
  std::unique_lock<std::mutex> locker(mutex_);
  return queue_->IsOpen();
}

DataQueueStatus BlockingQueue::Clear() {
  std::unique_lock<std::mutex> locker(mutex_);
  while (Size() > 0) {
    std::vector<DataQueueItem> data;
    auto ret = queue_->Front(&data);
    if (ret != DataQueueStatus::SUCCESS) {
      return ret;
    }
    ret = queue_->Pop();
    if (ret != DataQueueStatus::SUCCESS) {
      return ret;
    }
  }
  return DataQueueStatus::SUCCESS;
}
}  // namespace device
}  // namespace mindspore
