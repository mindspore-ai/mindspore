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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_ASYNC_HQUEUE_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_ASYNC_HQUEUE_H_

#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <string>
#include <utility>
#include <unordered_map>
#include <condition_variable>

#include "include/backend/visible.h"
#include "runtime/pipeline/task/task.h"
#include "thread/hqueue.h"
#ifndef USE_HQUEUE
#define USE_HQUEUE
#endif

namespace mindspore {
namespace runtime {
/* Thread status */
constexpr int kThreadBusy = 0;  // busy, the thread is running task
constexpr int kThreadIdle = 1;  // idle, the thread is waiting
// Create a new thread to execute the tasks in the queue sequentially.
class BACKEND_EXPORT AsyncHqueue {
 public:
  explicit AsyncHqueue(std::string name) : name_(std::move(name)) {}
  virtual ~AsyncHqueue();

  // Init resource
  void Init();

  // Add task to the end of the queue.
  bool Push(AsyncTask *task);

  // Wait for all async task finish executing.
  void Wait();

  // Check if the queue is empty.
  bool Empty();

  // clear tasks of queue, and wait last task.
  void Clear();

  // When an exception occurs, the state needs to be reset.
  void Reset();

  // Thread join before the process exit.
  void WorkerJoin();

  // Reinit resources after fork occurs.
  void ChildAfterFork();

  // Check grad queue exception.
  void CheckException();

 protected:
  void WorkerLoop();
  void SetThreadName() const;
  std::unique_ptr<std::thread> worker_{nullptr};
  std::mutex task_mutex_;
  std::unique_ptr<std::condition_variable> task_cond_var_{nullptr};
  std::string name_;

 private:
  void ClearTaskWithException();
  HQueue<AsyncTask> tasks_hqueque_;
  bool init_{false};
  bool alive_{true};
  bool stop_{false};
  std::atomic_int status_{kThreadBusy};
  size_t spin_count_{0};
  std::exception_ptr e_ptr_{nullptr};
};
using AsyncHqueuePtr = std::unique_ptr<AsyncHqueue>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_ASYNC_QUEUE_H_
