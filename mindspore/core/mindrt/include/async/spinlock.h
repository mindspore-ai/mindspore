/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_MINDRT_INCLUDE_ASYNC_SPINLOCK_H
#define MINDSPORE_CORE_MINDRT_INCLUDE_ASYNC_SPINLOCK_H

#include <atomic>

namespace mindspore {
class SpinLock {
 public:
  void lock() {
    while (locked.test_and_set(std::memory_order_acquire)) {
    }
  }

  void unlock() { locked.clear(std::memory_order_release); }

 private:
  std::atomic_flag locked = ATOMIC_FLAG_INIT;
};
}  // namespace mindspore

#endif
