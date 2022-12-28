/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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

#ifndef AICPU_CONTEXT_COMMON_NOTIFICATION_H
#define AICPU_CONTEXT_COMMON_NOTIFICATION_H
#include <cassert>
#include <atomic>
#include <condition_variable>
#include <mutex>

namespace aicpu {

class Notification {
 public:
  Notification() : notified_(0) {}
  ~Notification() { std::unique_lock<std::mutex> l(mu_); }

  void Notify() {
    std::unique_lock<std::mutex> l(mu_);
    if (!HasBeenNotified()) {
      notified_.store(true, std::memory_order_release);
      cv_.notify_all();
    }
  }

  bool HasBeenNotified() const { return notified_.load(std::memory_order_acquire); }

  void WaitForNotification() {
    if (!HasBeenNotified()) {
      std::unique_lock<std::mutex> l(mu_);
      while (!HasBeenNotified()) {
        cv_.wait(l);
      }
    }
  }

 private:
  std::mutex mu_;               // protects mutations of notified_
  std::condition_variable cv_;  // signaled when notified_ becomes non-zero
  std::atomic<bool> notified_;  // mutations under mu_
};

}  // namespace aicpu
#endif  // AICPU_CONTEXT_COMMON_NOTIFICATION_H
