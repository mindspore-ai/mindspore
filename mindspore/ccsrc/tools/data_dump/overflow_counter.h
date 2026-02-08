/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_TOOLS_DATA_DUMP_OVERFLOW_COUNTER_H_
#define MINDSPORE_CCSRC_TOOLS_DATA_DUMP_OVERFLOW_COUNTER_H_

#include <atomic>
#include <memory>
#include <mutex>

class OverflowCounter {
 private:
  std::atomic<uint32_t> count_{0};
  OverflowCounter() = default;

  inline static std::shared_ptr<OverflowCounter> instance_ = nullptr;
  inline static std::once_flag instance_mutex_;

 public:
  ~OverflowCounter() = default;
  static OverflowCounter &GetInstance() {
    std::call_once(instance_mutex_, []() {
      if (instance_ == nullptr) {
        instance_ = std::shared_ptr<OverflowCounter>(new OverflowCounter);
      }
    });
    return *instance_;
  }

  void addCount() { ++count_; }

  uint32_t getCount() const { return count_; }
};

#endif  // MINDSPORE_CCSRC_TOOLS_DATA_DUMP_OVERFLOW_COUNTER_H_
