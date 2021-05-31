/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_MINDRT_RUNTIME_HQUEUE_H_
#define MINDSPORE_CORE_MINDRT_RUNTIME_HQUEUE_H_
#include <atomic>
#include <vector>

namespace mindspore {
// implement a lock-free queue
template <class T>
class HQueue {
 public:
  HQueue(const HQueue &) = delete;
  HQueue &operator=(const HQueue &) = delete;
  explicit HQueue(size_t queue_size) : freeHead(0), usedHead(0) { cache.resize(queue_size); }
  virtual ~HQueue() {
    freeHead = 0;
    usedHead = 0;
  }

  bool Enqueue(T *t) {
    size_t curPos = freeHead.load(std::memory_order_relaxed);
    size_t nextPos = curPos + 1;
    if (nextPos == cache.size()) {
      nextPos = 0;
    }

    size_t usedIndex = usedHead.load(std::memory_order_acquire);
    if (nextPos != usedIndex) {
      cache[curPos] = t;
      // move free head to new position
      freeHead.store(nextPos, std::memory_order_release);
      return true;
    }

    // cache is full
    return false;
  }

  T *Dequeue() {
    size_t usedIndex = usedHead.load(std::memory_order_relaxed);
    size_t freeIndex = freeHead.load(std::memory_order_acquire);

    if (freeIndex == usedHead) {  // empty
      return nullptr;
    }

    T *ret = cache[usedIndex];
    usedIndex++;
    if (usedIndex == cache.size()) {
      usedIndex = 0;
    }
    usedHead.store(usedIndex, std::memory_order_release);
    return ret;
  }

 private:
  std::vector<T *> cache;
  std::atomic<size_t> freeHead;
  std::atomic<size_t> usedHead;
};

}  // namespace mindspore

#endif  // MINDSPORE_CORE_MINDRT_RUNTIME_HQUEUE_H_
