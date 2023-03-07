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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_OP_EXECUTOR_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_OP_EXECUTOR_H_

#include <vector>
#include <memory>
#include <queue>
#include <map>
#include <string>
#include <set>
#include <utility>
#include "backend/common/session/kernel_graph.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "runtime/hardware/device_context.h"
#include "runtime/graph_scheduler/graph_scheduler.h"
#include "include/backend/visible.h"
#include "runtime/pynative/async/backend_op_task.h"
#include "runtime/pynative/async/async_queue.h"

namespace mindspore::runtime {
class BACKEND_EXPORT OpExecutor {
 public:
  static OpExecutor &GetInstance();

  class ExecuteGuard {
   public:
    ExecuteGuard() { OpExecutor::GetInstance().executing_ = true; }
    ~ExecuteGuard() { OpExecutor::GetInstance().executing_ = false; }
  };

  void RegisterForwardCallback(const std::function<void()> &callback);

  // Register build callback function
  void Register(const std::function<void()> &callback);

  void PushOpBuildTask(const std::shared_ptr<pynative::BackendOpBuildTask> &op_build_task);

  void PushOpRunTask(const std::shared_ptr<pynative::BackendOpRunTask> &op_run_task);

  const std::vector<std::shared_ptr<pynative::BackendOpBuildTask>> &GetOpBuildTasks() const { return op_build_tasks_; }

  bool BuildQueueEmpty();
  bool RunQueueEmpty();

  // If the build queue is full, we can compile the kernels in parallel.
  bool BuildQueueFull();

  // Clear the build tasks when batch build finished.
  void ClearOpBuildTasks();

  std::vector<std::shared_ptr<pynative::BackendOpBuildTask>> PopOpBuildTasks();

  // When an exception occurs, the state needs to be reset.
  // Because we may sometimes have to ignore the exception and continue to run other tasks
  void Reset();

  // Determine if there is another task with the same name in execution.
  // Tasks with the same name use the same CNode cache. So we need to wait.
  bool ActorInQueue(GraphId graph_id);

  // Wait for all OpRunTasks to finish executing.
  void Wait();

  void WaitAll();

  // Thread join before the process exit.
  void WorkerJoin();

 private:
  OpExecutor();
  ~OpExecutor();
  DISABLE_COPY_AND_ASSIGN(OpExecutor);

  void WaitForBuild();
  void WaitForRun();
  void ClearResources();

  pynative::AsyncQueue async_queue_;

  std::vector<std::shared_ptr<pynative::BackendOpBuildTask>> op_build_tasks_;

  std::set<GraphId> actor_in_queue_;
  std::function<void()> batch_build_callback_{nullptr};
  inline static size_t kMaxQueueSize = 20;
  bool executing_{false};
  std::mutex build_mutex_;
  std::function<void()> forward_callback_{nullptr};
};
}  // namespace mindspore::runtime
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_OP_EXECUTOR_H_
