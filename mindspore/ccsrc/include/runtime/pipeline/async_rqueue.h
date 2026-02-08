/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIME_PIPELINE_ASYNC_R_QUEUE_H_
#define MINDSPORE_CCSRC_INCLUDE_RUNTIME_PIPELINE_ASYNC_R_QUEUE_H_

#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <set>
#include <map>
#include <vector>
#include <string>
#include <unordered_map>
#include <condition_variable>
#include <utility>

#include "runtime/pipeline/visible.h"

#include "runtime/pipeline/ring_queue.h"

namespace mindspore {
namespace runtime {
class AsyncTask;
using AsyncTaskPtr = std::shared_ptr<AsyncTask>;

constexpr auto kQueueCapacity = 1024;
enum kThreadWaitLevel : int {
  kLevelUnknown = 0,
  kLevelPython,
  kLevelFrontend,
  kLevelGrad,
  kLevelBackend,
  kLevelDevice,
  kLevelStressDetect,
};

// Create a new thread to execute the tasks in the queue sequentially.
class RUNTIME_PIPELINE_EXPORT AsyncRQueue {
 public:
  explicit AsyncRQueue(std::string name, kThreadWaitLevel wait_level)
      : name_(std::move(name)),
        wait_level_(wait_level),
        tasks_queue_(std::make_unique<RingQueue<AsyncTaskPtr, kQueueCapacity>>()) {}
  virtual ~AsyncRQueue();

  // Add task to the end of the queue.
  virtual void Push(const AsyncTaskPtr &task);

  bool CanPush() const;

  // Wait for all async task finish executing.
  virtual void Wait();

  // Check if the queue is empty.
  virtual bool Empty();

  // clear tasks of queue, and wait last task.
  void Clear();

  // When an exception occurs, the state needs to be reset.
  void Reset();

  // Thread join before the process exit.
  virtual void WorkerJoin();

  // Reinit resources after fork occurs.
  void ChildAfterFork();

  // Call once before all ChildAfterFork
  static void ChildAfterForkPre();

  void ParentBeforeFork();

  bool Spin() { return tasks_queue_->spin(); }

  void SetSpin(bool spin);

  bool IsMultiThreadDisabled() const { return disable_multi_thread_; }

  void SetMultiThreadDisabled(bool disabled) { disable_multi_thread_ = disabled; }

 protected:
  void WorkerLoop();
  void SetThreadName() const;

  std::unique_ptr<std::thread> worker_{nullptr};
  std::string name_;
  kThreadWaitLevel wait_level_;
  inline static std::unordered_map<std::thread::id, kThreadWaitLevel> thread_id_to_wait_level_;
  inline static std::mutex level_mutex_;

 private:
  void ClearTaskWithException();

  void BindCoreForThread();

  std::unique_ptr<RingQueue<AsyncTaskPtr, kQueueCapacity>> tasks_queue_;
#if defined(__APPLE__)
  bool disable_multi_thread_{true};
#else
  bool disable_multi_thread_{false};
#endif

  std::map<std::string, int> thread_to_core_idx = {
    {"frontend_queue", 0}, {"backend_queue", 1}, {"launch_queue", 2}, {"bprop_queue", 3}};
};
}  // namespace runtime
using AsyncRQueuePtr = std::unique_ptr<runtime::AsyncRQueue>;
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIME_PIPELINE_ASYNC_R_QUEUE_H_
