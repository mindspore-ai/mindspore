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

#ifndef MINDSPORE_PI_JIT_ASYNC_TASK_MANAGERER_H_
#define MINDSPORE_PI_JIT_ASYNC_TASK_MANAGERER_H_

#include <functional>
#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "ir/anf.h"
#include "runtime/pipeline/async_rqueue.h"
#include "pybind11/stl.h"

namespace mindspore {
namespace pijit {
namespace py = pybind11;
using RecordFunc = std::function<void(const py::object &prim, const py::object &out, const py::list &inputs)>;
class AsyncTaskMultiWorker : public std::enable_shared_from_this<AsyncTaskMultiWorker> {
 public:
  explicit AsyncTaskMultiWorker(runtime::TaskType task_type) : comp_count_(0), task_type_(task_type), done_(false) {}
  virtual ~AsyncTaskMultiWorker() = default;
  virtual void Run() = 0;
  runtime::TaskType task_type() const { return task_type_; }
  void Depend(std::shared_ptr<AsyncTaskMultiWorker> task);
  void DependOn(std::vector<std::shared_ptr<AsyncTaskMultiWorker>> *tasks);
  void Notify();
  void NotifyTo(std::vector<std::shared_ptr<AsyncTaskMultiWorker>> *tasks);
  bool Available();
  void Reset();
  void RunWrapper();
  bool Done() const { return done_; }

 protected:
  std::vector<std::shared_ptr<AsyncTaskMultiWorker>> depends_;
  std::vector<std::shared_ptr<AsyncTaskMultiWorker>> notifies_;
  std::atomic<size_t> comp_count_;
  runtime::TaskType task_type_;
  bool done_;
};
#ifdef USE_ASYNC_SINGLE_WORKER
using AsyncTask = runtime::AsyncTask;
using AsyncTaskPtr = std::shared_ptr<runtime::AsyncTask>;
#else
using AsyncTask = AsyncTaskMultiWorker;
using AsyncTaskPtr = std::shared_ptr<AsyncTaskMultiWorker>;
#endif

class AsyncQueueMultiWorker {
 public:
  AsyncQueueMultiWorker(std::string name, runtime::kThreadWaitLevel wait_level, size_t worker_count = 8);
  virtual ~AsyncQueueMultiWorker();

  void Push(const AsyncTaskPtr &task);
  void Wait();
  bool Empty();
  void Clear();
  void WorkerJoin();

 protected:
  bool Available();
  AsyncTaskPtr PopAvailable();
  AsyncTaskPtr Pop();
  void WorkerLoop();
  std::vector<std::unique_ptr<std::thread>> workers_;
  std::mutex mutex_;
  std::condition_variable ready_cv_;
  std::condition_variable task_cv_;
  std::vector<AsyncTaskPtr> tasks_queue_;
  std::vector<AsyncTaskPtr> wait_queue_;
  std::string name_;
  runtime::kThreadWaitLevel wait_level_;
  size_t worker_cnt_;
  size_t ready_cnt_;
  bool terminate_;
};
#ifdef USE_ASYNC_SINGLE_WORKER
using AsyncQueue = runtime::AsyncRQueue;
using AsyncQueuePtr = AsyncRQueuePtr;
#else
using AsyncQueue = AsyncQueueMultiWorker;
using AsyncQueuePtr = std::shared_ptr<AsyncQueueMultiWorker>;
#endif

class RecordTask : public mindspore::pijit::AsyncTask {
 public:
  explicit RecordTask(RecordFunc task, const py::object &prim, const py::object &out, const py::list &inputs)
      : mindspore::pijit::AsyncTask(runtime::kBpropTask),
        run_task_(std::move(task)),
        prim_(prim),
        out_(out),
        inputs_(inputs) {}
  ~RecordTask() override = default;
  void Run() override;

 private:
  RecordFunc run_task_;
  py::object prim_;
  py::object out_;
  py::list inputs_;
};

using RecordTaskPtr = std::shared_ptr<RecordTask>;

class RunGenerateBpropTask : public mindspore::pijit::AsyncTask {
 public:
  explicit RunGenerateBpropTask(std::function<void()> task)
      : mindspore::pijit::AsyncTask(runtime::kBpropTask), run_task_(std::move(task)) {}
  ~RunGenerateBpropTask() override = default;
  void Run() override;

 private:
  std::function<void()> run_task_;
};

using RunGenerateBpropTaskPtr = std::shared_ptr<RunGenerateBpropTask>;

class RunBpropTask : public mindspore::pijit::AsyncTask {
 public:
  explicit RunBpropTask(std::function<void(const ValuePtr &value)> task, const ValuePtr &value)
      : mindspore::pijit::AsyncTask(runtime::kBpropTask), run_task_(std::move(task)), value_(value) {}
  ~RunBpropTask() override = default;
  void Run() override;

 private:
  std::function<void(const ValuePtr &value)> run_task_;
  ValuePtr value_;
};

using RunBpropTaskPtr = std::shared_ptr<RunBpropTask>;

using Level = runtime::kThreadWaitLevel;
class AsyncTaskManager {
 public:
  AsyncTaskManager()
      : record_task_queue_(std::make_shared<AsyncQueue>("record_task_queue", Level::kLevelGrad)),
        generate_task_queue_(std::make_shared<AsyncQueue>("generate_task_queue", Level::kLevelGrad)),
        run_task_queue_(std::make_shared<AsyncQueue>("run_task_queue", Level::kLevelGrad)) {}
  virtual ~AsyncTaskManager() = default;

  const AsyncQueuePtr &GetRecordTaskQueue() const { return record_task_queue_; }
  const AsyncQueuePtr &GetGenerateTaskQueue() const { return generate_task_queue_; }
  const AsyncQueuePtr &GetRunTaskQueue() const { return run_task_queue_; }
  void DispatchRecordTask(const AsyncTaskPtr &task) const { record_task_queue_->Push(task); }
  void DispatchGenerateTask(const AsyncTaskPtr &task) const { generate_task_queue_->Push(task); }
  void DispatchRunTask(const AsyncTaskPtr &task) const { run_task_queue_->Push(task); }

 private:
  AsyncQueuePtr record_task_queue_;
  AsyncQueuePtr generate_task_queue_;
  AsyncQueuePtr run_task_queue_;
};

using AsyncTaskManagerPtr = std::shared_ptr<AsyncTaskManager>;
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_ASYNC_TASK_MANAGERER_H_
