/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_RING_BUFFER_H
#define MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_RING_BUFFER_H
#include <array>
#include <atomic>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <vector>
#include <utility>
#include "include/utils/visible.h"
#include "utils/log_adapter.h"
namespace mindspore {
namespace profiler {
namespace ascend {

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__)
template <typename T>
class PROFILER_EXPORT RingBuffer {
 public:
  RingBuffer()
      : is_inited_(false),
        is_quit_(false),
        read_index_(0),
        write_index_(0),
        idle_write_index_(0),
        capacity_(0),
        mask_(0),
        cycles_exceed_cnt_(0),
        full_cnt_(0) {}

  ~RingBuffer() { UnInit(); }

  void Init(size_t capacity) {
    capacity_ = capacity;
    mask_ = capacity_ - 1;
    data_queue_.resize(capacity);
    is_inited_ = true;
    is_quit_ = false;
  }

  void UnInit() {
    if (is_inited_) {
      data_queue_.clear();
      read_index_ = 0;
      write_index_ = 0;
      idle_write_index_ = 0;
      capacity_ = 0;
      mask_ = 0;
      is_quit_ = true;
      is_inited_ = false;
      auto final_cycles_exceed_cnt = cycles_exceed_cnt_.load(std::memory_order_relaxed);
      if (final_cycles_exceed_cnt > 0) {
        MS_LOG(WARNING) << "Profiler RingBuffer uninit, final cycles exceed cnt: " << final_cycles_exceed_cnt;
        cycles_exceed_cnt_ = 0;
      }
      auto final_full_cnt = full_cnt_.load(std::memory_order_relaxed);
      if (final_full_cnt > 0) {
        MS_LOG(WARNING) << "Profiler RingBuffer uninit, final full cnt: " << final_full_cnt;
        full_cnt_ = 0;
      }
    }
  }

  bool Push(T data) {
    size_t curr_read_index = 0;
    size_t curr_write_index = 0;
    size_t next_write_index = 0;
    size_t cycles = 0;
    static const size_t cycle_limit = 1024;
    do {
      if (!is_inited_ || is_quit_) {
        return false;
      }
      ++cycles;
      if (cycles >= cycle_limit) {
        cycles_exceed_cnt_.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
      curr_read_index = read_index_.load(std::memory_order_relaxed);
      curr_write_index = idle_write_index_.load(std::memory_order_relaxed);
      next_write_index = curr_write_index + 1;
      if ((next_write_index & mask_) == (curr_read_index & mask_)) {
        full_cnt_.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
    } while (!idle_write_index_.compare_exchange_weak(curr_write_index, next_write_index));
    size_t index = curr_write_index & mask_;
    data_queue_[index] = std::move(data);
    ++write_index_;
    return true;
  }

  bool Pop(T &data) {  // NOLINT(runtime/references)
    if (!is_inited_) {
      return false;
    }
    size_t curr_read_index = read_index_.load(std::memory_order_relaxed);
    size_t curr_write_index = write_index_.load(std::memory_order_relaxed);
    if ((curr_read_index & mask_) == (curr_write_index & mask_) && !is_quit_) {
      return false;
    }
    size_t index = curr_read_index & mask_;
    data = std::move(data_queue_[index]);
    read_index_++;
    return true;
  }

  size_t Size() {
    size_t curr_read_index = read_index_.load(std::memory_order_relaxed);
    size_t curr_write_index = write_index_.load(std::memory_order_relaxed);
    if (curr_read_index > curr_write_index) {
      return capacity_ - (curr_read_index & mask_) + (curr_write_index & mask_);
    }
    return curr_write_index - curr_read_index;
  }

 private:
  bool is_inited_;
  volatile bool is_quit_;
  std::atomic<size_t> read_index_;
  std::atomic<size_t> write_index_;
  std::atomic<size_t> idle_write_index_;
  size_t capacity_;
  size_t mask_;
  std::vector<T> data_queue_;

  // Ringbuffer push failed info
  std::atomic<size_t> cycles_exceed_cnt_;
  std::atomic<size_t> full_cnt_;
};
#else
template <typename T>
class PROFILER_EXPORT RingBuffer {
 public:
  RingBuffer() {}
  ~RingBuffer() {}
  void Init(size_t capacity) { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }
  void UnInit() { MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows."; }
  bool Push(T data) {
    MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
    return true;
  }
  bool Pop(T &data) {  // NOLINT(runtime/references)
    MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
    return true;
  }
  size_t Size() {
    MS_LOG(INTERNAL_EXCEPTION) << "profiler not support cpu windows.";
    return 0;
  }
};
#endif
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_COMMON_DEBUG_PROFILER_RING_BUFFER_H
