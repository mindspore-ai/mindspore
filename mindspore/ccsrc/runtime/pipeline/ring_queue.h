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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_RING_QUEUE_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_RING_QUEUE_H_

#include <atomic>
#include <array>
#include <cstddef>
#include <thread>

namespace mindspore {
// A simple ring buffer (or circular queue) with atomic operations for
// thread-safe enqueue, dequeue, and check for emptiness.
// RingQueue is only applicable to single-producer and single-consumer scenarios.
template <typename T, std::size_t Capacity>
class RingQueue {
 public:
  RingQueue() : head_(0), tail_(0) {}

  void Enqueue(const T &value) {
    std::size_t current_tail = tail_.load(std::memory_order_relaxed);
    std::size_t next_tail = (current_tail + 1) % Capacity;

    while (next_tail == head_.load(std::memory_order_acquire)) {
    }

    buffer_[current_tail] = value;
    tail_.store(next_tail, std::memory_order_release);
  }

  void Dequeue() {
    std::size_t current_head = head_.load(std::memory_order_relaxed);
    while (current_head == tail_.load(std::memory_order_acquire)) {
    }

    // Free memory when task is finished.
    buffer_[current_head] = nullptr;
    head_.store((current_head + 1) % Capacity, std::memory_order_release);
  }

  const T &Head() {
    std::size_t current_head = head_.load(std::memory_order_acquire);
    while (current_head == tail_.load(std::memory_order_acquire)) {
    }
    return buffer_[current_head];
  }

  bool IsEmpty() const { return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire); }

 private:
  std::array<T, Capacity> buffer_;
  // CPU cache line size is 64.
  alignas(64) std::atomic<std::size_t> head_;
  alignas(64) std::atomic<std::size_t> tail_;
};
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_RING_QUEUE_H_
