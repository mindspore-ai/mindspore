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

#include "ps/core/communicator/task_executor.h"

namespace mindspore {
namespace ps {
namespace core {
TaskExecutor::TaskExecutor(size_t thread_num, size_t max_task_num, size_t submit_timeout)
    : running_(true),
      thread_num_(thread_num),
      idle_thread_num_(0),
      submit_timeout_(submit_timeout),
      max_task_num_(max_task_num),
      task_num_(0) {
  for (size_t i = 0; i < thread_num; i++) {
    working_threads_.emplace_back([this]() {
      std::function<void()> task;
      while (true) {
        std::unique_lock<std::mutex> lock(mtx_);
        // Idle thread number increases when the mtx_ is locked.
        idle_thread_num_++;

        if (!running_) {
          // To avoid thread from blocking after destructor.
          return;
        }

        cv_.wait(lock);

        if (!running_ || task_queue_.empty()) {
          return;
        }

        task = task_queue_.front();
        task_queue_.pop();
        if (lock.owns_lock()) {
          lock.unlock();
        }

        task();
      }
    });
  }
  notify_thread_ = std::thread([this]() {
    // If there is no idle thread, wait until the working thread is available.
    while (running_) {
      {
        std::unique_lock<std::mutex> lock(mtx_);
        if (idle_thread_num_ > 0 && task_num_ > 0) {
          idle_thread_num_--;
          task_num_--;
          lock.unlock();
          cv_.notify_one();
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(kSubmitTaskIntervalInMs));
    }
  });
}

TaskExecutor::~TaskExecutor() {
  {
    std::unique_lock<std::mutex> lock(mtx_);
    running_ = false;
  }
  cv_.notify_all();
  for (auto &t : working_threads_) {
    t.join();
  }
  notify_thread_.join();
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
