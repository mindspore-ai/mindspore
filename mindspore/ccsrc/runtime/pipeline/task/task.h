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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_TASK_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_TASK_H_

#include <cstdint>
#include <exception>
#include <atomic>

namespace mindspore {
namespace runtime {
enum TaskType {
  kUnknownTask = 0,
  kDeviceOpTask,
  kDeviceOpBuildTask,
  kPyBoostOpTask,
  kBpropTask,
  kFrontendTask,
  kBackendTask,
  kKernelTask,
  kExitTask,
  kWaitTask,
  kStressDetectTask,
  kRunGraphTask
};

enum class KernelTaskType { kNORMAL_VIEW_TASK = 0, kCONTIGUOUS_TASK, kCOPY_TASK };

class AsyncTask {
 public:
  explicit AsyncTask(TaskType task_type) : task_type_(task_type) {}
  virtual ~AsyncTask() = default;
  virtual void Run() = 0;
  virtual bool RunWithRet() { return false; }
  virtual void SetException(const std::exception_ptr & /* e */) {}

  TaskType task_type() const { return task_type_; }
  uint64_t task_id() const { return task_id_; }
  void set_task_id(uint64_t task_id) { task_id_ = task_id; }
  static uint64_t MakeId() { return last_id_.fetch_add(1, std::memory_order_relaxed); }

 protected:
  TaskType task_type_;
  uint64_t task_id_{MakeId()};
  inline static std::atomic<uint64_t> last_id_{1};
};

class ExitTask : public AsyncTask {
 public:
  ExitTask() : AsyncTask(kExitTask) {}
  ~ExitTask() override = default;
  void Run() override {}
};

class WaitTask : public AsyncTask {
 public:
  WaitTask() : AsyncTask(kWaitTask) {}
  ~WaitTask() override = default;
  void Run() override {}
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_TASK_H_
