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
#include <cstdint>
#include <thread>
#include <mutex>
#include <condition_variable>

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
    std::size_t spins = 0;

    while (true) {
      const std::size_t current_head = head_.load(std::memory_order_acquire);
      if (next_tail != current_head) {
        break;
      }

      if (spins++ < max_spin_count_) {
        continue;
      } else {
        std::unique_lock<std::mutex> lock(mtx_);
        not_full_.wait(lock, [this, next_tail] { return head_.load(std::memory_order_acquire) != next_tail; });
        break;
      }
    }

    buffer_[current_tail] = value;
    tail_.store(next_tail, std::memory_order_release);

    // Notify consumer (under mutex to avoid missed wakes)
    std::lock_guard<std::mutex> lock(mtx_);
    not_empty_.notify_one();
  }

  void Dequeue() {
    std::size_t current_head = head_.load(std::memory_order_relaxed);
    std::size_t spins = 0;

    while (true) {
      const std::size_t current_tail = tail_.load(std::memory_order_acquire);
      if (current_head != current_tail) {
        break;
      }

      if (spins++ < max_spin_count_) {
        continue;
      } else {
        std::unique_lock<std::mutex> lock(mtx_);
        not_empty_.wait(lock, [this, current_head] { return tail_.load(std::memory_order_acquire) != current_head; });
        break;
      }
    }

    buffer_[current_head] = nullptr;
    head_.store((current_head + 1) % Capacity, std::memory_order_release);

    // Notify producer (under mutex to avoid missed wakes)
    std::lock_guard<std::mutex> lock(mtx_);
    not_full_.notify_one();
  }

  const T &Head() {
    std::size_t current_head = head_.load(std::memory_order_acquire);
    std::size_t spins = 0;

    while (current_head == tail_.load(std::memory_order_acquire)) {
      if (spins++ < max_spin_count_) {
        continue;
      } else {
        std::unique_lock<std::mutex> lock(mtx_);
        not_empty_.wait(lock, [this, current_head] { return tail_.load(std::memory_order_acquire) != current_head; });
        break;
      }
    }
    return buffer_[current_head];
  }

  bool IsEmpty() const { return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire); }

  bool spin() { return false; }

  void set_spin(bool /* spin */) {}

 private:
  std::array<T, Capacity> buffer_;
  alignas(128) std::atomic<std::size_t> head_;
  alignas(128) std::atomic<std::size_t> tail_;
  std::mutex mtx_;
  std::condition_variable not_full_;
  std::condition_variable not_empty_;
  static const std::size_t max_spin_count_{1200000};
};
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_RING_QUEUE_H_
