/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_COMMON_THREAD_POOL_H_
#define MINDSPORE_CCSRC_COMMON_THREAD_POOL_H_

#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <queue>
#include <string>
#include <atomic>
#include <memory>
#include <utility>
#include <functional>
#include <iostream>
#include "utils/log_adapter.h"

namespace mindspore {
namespace common {
const int kCoreThreadNum = 3;
const int kDefaultMaxThreadNum = 8;
enum Status { FAIL = -1, SUCCESS = 0 };
using Task = std::function<int()>;

class Queue {
 public:
  Queue() = default;
  ~Queue() = default;
  bool Enqueue(Task *task);
  bool Dequeue(Task **out);
  std::atomic_int task_size_ = {0};

 private:
  std::atomic_int head_ = {0};
  std::atomic_int tail_ = {0};
  Task *buffer_[2]{};
};

class ThreadPool {
 public:
  ~ThreadPool();
  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;
  static ThreadPool &GetInstance();
  bool SyncRun(const std::vector<Task> &tasks);
  size_t GetSyncRunThreadNum() { return max_thread_num_; }
  void ClearThreadPool();

 private:
  ThreadPool();
  bool SetThreadPool(int config_thread_num);
  void AddNewThread(int add_num);
  void AddRunThread(int num);
  void SubRunThread(int num);
  bool CheckResult();
  bool InnerSyncRun(const std::vector<Task> &tasks);
  void SyncRunLoop();

  int cur_thread_nums_{0};
  int cur_thread_run_nums_{0};
  int core_thread_num_{kCoreThreadNum};
  int max_thread_num_{kDefaultMaxThreadNum};
  std::mutex pool_mtx_;
  std::mutex thread_mtx_;
  std::condition_variable queue_ready_;
  std::atomic_bool exit_run_ = {false};
  std::vector<std::atomic_bool *> activate_list_{};
  std::vector<std::thread> thread_list_{};
  std::vector<std::shared_ptr<Queue>> queue_list_{};
  std::vector<std::pair<int, std::pair<bool, int>>> error_info_{};
  std::queue<Task> task_queue_;
  std::mutex task_mutex_;
  std::condition_variable task_cond_var_;
  int task_finished_count_{0};
  std::condition_variable finished_cond_var_;
  std::vector<std::thread> sync_run_threads_{};
};
}  // namespace common
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_COMMON_THREAD_POOL_H_
