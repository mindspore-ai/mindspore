/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_MINDRT_RUNTIME_SEMAPHORE_H_
#define MINDSPORE_CORE_MINDRT_RUNTIME_SEMAPHORE_H_

#include <mutex>
#include <condition_variable>

namespace mindspore {
class Semaphore {
 public:
  explicit Semaphore(uint32_t count = 0) : count_(count) {}

  inline void Signal() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++count_;
    lock.unlock();
    cv_.notify_one();
  }

  inline void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return count_ > 0; });
    --count_;
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  uint32_t count_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_MINDRT_RUNTIME_SEMAPHORE_H_
