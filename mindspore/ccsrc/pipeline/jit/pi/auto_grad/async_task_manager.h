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
#include <memory>
#include <utility>
#include "runtime/pipeline/async_rqueue.h"
#include "pybind11/stl.h"

namespace mindspore {
namespace pijit {
namespace py = pybind11;
using AsyncTask = runtime::AsyncTask;
using AsyncQueue = runtime::AsyncRQueue;
using AsyncQueuePtr = AsyncRQueuePtr;
using RecordFunc = std::function<void(const py::object &prim, const py::object &out, const py::list &inputs)>;

class RecordTask : public AsyncTask {
 public:
  explicit RecordTask(RecordFunc task, const py::object &prim, const py::object &out, const py::list &inputs)
      : AsyncTask(runtime::kBpropTask), run_task_(std::move(task)), prim_(prim), out_(out), inputs_(inputs) {}
  ~RecordTask() override = default;
  void Run() override;

 private:
  RecordFunc run_task_;
  py::object prim_;
  py::object out_;
  py::list inputs_;
};

using RecordTaskPtr = std::shared_ptr<RecordTask>;

class RunBpropTask : public AsyncTask {
 public:
  explicit RunBpropTask(std::function<void()> task) : AsyncTask(runtime::kBpropTask), run_task_(std::move(task)) {}
  ~RunBpropTask() override = default;
  void Run() override;

 private:
  std::function<void()> run_task_;
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
  void DispatchRecordTask(const RecordTaskPtr &task) const { record_task_queue_->Push(task); }
  void DispatchGenerateTask(const RunBpropTaskPtr &task) const { generate_task_queue_->Push(task); }
  void DispatchRunTask(const RunBpropTaskPtr &task) const { run_task_queue_->Push(task); }

 private:
  AsyncQueuePtr record_task_queue_;
  AsyncQueuePtr generate_task_queue_;
  AsyncQueuePtr run_task_queue_;
};

using AsyncTaskManagerPtr = std::shared_ptr<AsyncTaskManager>;
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_ASYNC_TASK_MANAGERER_H_
