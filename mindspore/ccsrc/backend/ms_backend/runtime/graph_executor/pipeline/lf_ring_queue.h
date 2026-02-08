/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_CORE_GRAPH_EXECUTOR_PIPELINE_LF_RING_QUEUE_H_
#define MINDSPORE_CCSRC_RUNTIME_CORE_GRAPH_EXECUTOR_PIPELINE_LF_RING_QUEUE_H_

#include <atomic>
#include <array>
#include <utility>
#include <cstddef>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
constexpr size_t kQueueElemAlignSize = 128;
// This is a lock-free queue implementation that supports multiple producers and a single consumer, improves performance
// through spinning, and supports inplacement construct element, pause and continue the queue, the queue is in pause
// status after creation.
template <typename T, uint64_t Capacity>
class LFRingQueue {
  static_assert(Capacity > 0, "Capacity must be greater than 0 for LFRingQueue");
  static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");
  static_assert(std::is_nothrow_move_constructible_v<T>, "The template type T must has nothrow move constructible");
  static_assert(std::is_nothrow_destructible_v<T>, "The template type T must has nothrow destructible");

 public:
  LFRingQueue() = default;
  ~LFRingQueue() = default;

  LFRingQueue(const LFRingQueue &) = delete;
  LFRingQueue &operator=(const LFRingQueue &) = delete;

  // The Element is the element type stored in buffer_.
  struct Element {
    std::atomic<bool> ready{false};
    typename std::aligned_storage<sizeof(T), alignof(T)>::type storage;
  };

  // Pause the queue, can not push element to a queue which is in pause status, the queue is in pause status after
  // creation.
  void Pause() { running_.store(false, std::memory_order_release); }

  // Check the queue is on pause state.
  bool IsPaused() const { return false == running_.load(std::memory_order_acquire); }

  // Continue the queue which is in pause status.
  void Continue() {
    running_.store(true);

    std::unique_lock<std::mutex> lock(mtx_);
    pause_cv_.notify_one();
  }

  void Finalize() noexcept {
    alive_.store(false);
    std::lock_guard<std::mutex> lock(mtx_);
    pause_cv_.notify_all();
  }

  // Check the queue is empty or not.
  bool Empty() const noexcept { return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire); }

  // Push element to lock free queue, the args parameter must be of type T or can construct type T object.
  // type. Push is multi thread safety and return until finishing push.
  template <typename... Args>
  bool Push(Args &&...args) {
    if (!alive_.load(std::memory_order_acquire)) return false;

    uint64_t current_tail;
    Element *current_element;

    while (true) {
      if (!alive_.load(std::memory_order_acquire)) return false;

      if (!running_.load(std::memory_order_acquire)) {
        MS_LOG(ERROR) << "The queue is in pause status, can not push task.";
        return false;
      }

      if (TryPush(current_tail, current_element, std::forward<Args>(args)...)) {
        return true;
      }
    }
  }

  // Pop a element from buffer_, return until finishing pop.
  bool Pop() {
    if (!alive_.load(std::memory_order_acquire)) return false;

    while (true) {
      if (!alive_.load(std::memory_order_acquire)) return false;

      if (TryPop()) {
        return true;
      }
    }
  }

  // Get the first element in queue, return until finishing get the first element.
  T *Front() noexcept {
    while (true) {
      if (!alive_.load(std::memory_order_acquire)) return nullptr;

      if (T *ptr = TryFront()) {
        return ptr;
      }
      if (!running_.load(std::memory_order_acquire)) {
        std::unique_lock<std::mutex> lock(mtx_);
        pause_cv_.wait(lock, [this] {
          return !Empty() || !alive_.load(std::memory_order_acquire) || running_.load(std::memory_order_acquire);
        });
      }
    }
  }

 private:
  template <typename... Args>
  bool TryPush(uint64_t &current_tail, Element *&element,  // NOLINT(runtime/references)
               Args &&...args) {
    current_tail = tail_.load(std::memory_order_relaxed);
    auto current_head = head_.load(std::memory_order_relaxed);
    if (current_tail < current_head || current_tail - current_head >= Capacity) {
      return false;
    }
    if (current_tail == UINT64_MAX) {
      // This is a safety check, it requires continuous pushing for millions of years to trigger an overflow.
      MS_LOG(EXCEPTION) << "The queue is overflow and push task failed.";
    }

    if (!tail_.compare_exchange_weak(current_tail, current_tail + 1, std::memory_order_acq_rel,
                                     std::memory_order_relaxed)) {
      return false;
    }

    element = &buffer_[current_tail & mask_];
    new (&element->storage) T(std::forward<Args>(args)...);
    element->ready.store(true, std::memory_order_release);
    return true;
  }

  bool TryPop() {
    const uint64_t current_head = head_.load(std::memory_order_relaxed);
    if (current_head == tail_.load(std::memory_order_acquire)) {
      return false;
    }

    Element &element = buffer_[current_head & mask_];
    if (!element.ready.load(std::memory_order_acquire)) {
      return false;
    }

    reinterpret_cast<T *>(&element.storage)->~T();
    element.ready.store(false, std::memory_order_release);
    head_.store(current_head + 1, std::memory_order_release);
    return true;
  }

  T *TryFront() noexcept {
    const uint64_t current_head = head_.load(std::memory_order_relaxed);
    if (current_head == tail_.load(std::memory_order_acquire)) {
      return nullptr;
    }

    Element &element = buffer_[current_head & mask_];
    if (!element.ready.load(std::memory_order_acquire)) {
      return nullptr;
    }

    return reinterpret_cast<T *>(&element.storage);
  }

  // Bitmask for fast modulo.
  static constexpr uint64_t mask_ = Capacity - 1;
  alignas(kQueueElemAlignSize) Element buffer_[Capacity];
  alignas(kQueueElemAlignSize) std::atomic<uint64_t> head_{0};
  alignas(kQueueElemAlignSize) std::atomic<uint64_t> tail_{0};

  // Indicates whether the queue is currently paused or in execution state.
  alignas(kQueueElemAlignSize) std::atomic<bool> running_{false};
  // Indicates whether the queue is currently in a closed state.
  alignas(kQueueElemAlignSize) std::atomic<bool> alive_{true};

  // Lock the status for pause or execution state.
  std::mutex mtx_;
  std::condition_variable pause_cv_;
};
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_CORE_GRAPH_EXECUTOR_PIPELINE_LF_RING_QUEUE_H_
