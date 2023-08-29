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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_BLOCKING_QUEUE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_BLOCKING_QUEUE_H_

#include <memory>
#include <mutex>
#include <condition_variable>
#include "utils/ms_utils.h"

namespace mindspore {
namespace distributed {
/**
 * @brief This class implements a generic blocking queue and could be used for the multi-producer, multi-consumer case.
 * For performance, the queue element is a pointer, and the user needs to do their own memory management
 * (pointer lifetime).
 */
template <typename T>
class BlockingQueue {
 public:
  explicit BlockingQueue(size_t capacity) : capacity_(capacity) { elements_ = std::make_unique<T *[]>(capacity_); }
  ~BlockingQueue() = default;

  /**
   * @brief Push new data to tail of queue.
   * @param[in] `data`: The pointer to new element to enqueue.
   */
  void Push(T *data) {
    std::unique_lock<std::mutex> lock(mtx_);
    while (Full()) {
      if (closed_) {
        return;
      }

      full_cv_.wait(lock);
    }

    elements_[tail_] = data;
    tail_ = (tail_ + 1) % capacity_;
    ++size_;

    empty_cv_.notify_one();
  }

  /**
   * @brief Get the first element(at head position in queue) of the queue and removes it from the queue.
   * @return The element which need to dequeue.
   */
  T *Pop() {
    std::unique_lock<std::mutex> lock(mtx_);
    while (Empty()) {
      if (closed_) {
        return nullptr;
      }

      empty_cv_.wait(lock);
    }

    auto pop_value = elements_[head_];
    head_ = (head_ + 1) % capacity_;
    --size_;

    full_cv_.notify_one();
    return pop_value;
  }

  /**
   * @brief Check whether there is no element in queue.
   * @return Whether there is no element in queue.
   */
  bool Empty() { return size_ == 0; }

  /**
   * @brief Check whether the number of queue elements reaches capacity.
   * @return Whether the number of queue elements reaches capacity.
   */
  bool Full() { return size_ == capacity_; }

  /**
   * @brief Close the queue and stop push and pop operations.
   */
  void Close() {
    std::unique_lock<std::mutex> lock(mtx_);
    if (!closed_) {
      closed_ = true;
      full_cv_.notify_all();
      empty_cv_.notify_all();
    }
  }

 private:
  DISABLE_COPY_AND_ASSIGN(BlockingQueue);

  // The maximum capacity of queue.
  size_t capacity_;
  // The element number in queue.
  size_t size_{0};

  // The buffer used to record elements in the queue.
  std::unique_ptr<T *[]> elements_;

  // The cursor used to point the head position.
  size_t head_{0};
  // The cursor used to point the tail position.
  size_t tail_{0};

  // The flag indicates whether the queue is closed.
  bool closed_{false};

  // A lock used to secure the access of queue elements.
  std::mutex mtx_;
  // Used to block the push operations when queue is full.
  std::condition_variable full_cv_;
  // Used to block the pop operations when queue is empty.
  std::condition_variable empty_cv_;
};
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_BLOCKING_QUEUE_H_
