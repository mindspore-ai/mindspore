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

#ifndef MINDSPORE_MINDSPORE_CCSRC_BACKEND_SESSION_PYNATIVE_TASK_MANAGER_H_
#define MINDSPORE_MINDSPORE_CCSRC_BACKEND_SESSION_PYNATIVE_TASK_MANAGER_H_

#include <vector>
#include <memory>
#include <queue>
#include <map>
#include <string>
#include <utility>
#include "backend/session/kernel_graph.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace session {
class RunOpContext {
 public:
  RunOpContext(std::string graph_info, bool is_dynamic_shape, KernelGraphPtr graph, std::vector<int64_t> tensors_mask,
               std::vector<tensor::TensorPtr> input_tensors,
               std::map<tensor::TensorPtr, KernelWithIndex> tensor_to_node)
      : graph_info_(std::move(graph_info)),
        is_dynamic_shape_(is_dynamic_shape),
        graph_(std::move(graph)),
        tensors_mask_(std::move(tensors_mask)),
        input_tensors_(std::move(input_tensors)),
        tensor_to_node_(std::move(tensor_to_node)) {}
  ~RunOpContext() = default;

  const KernelGraphPtr &graph() const { return graph_; }
  bool is_dynamic_shape() const { return is_dynamic_shape_; }
  const std::vector<int64_t> &tensor_mask() const { return tensors_mask_; }
  const std::vector<tensor::TensorPtr> &input_tensors() const { return input_tensors_; }
  const std::map<tensor::TensorPtr, KernelWithIndex> &tensor_to_node() const { return tensor_to_node_; }

 private:
  std::string graph_info_;
  bool is_dynamic_shape_;
  KernelGraphPtr graph_;
  std::vector<int64_t> tensors_mask_;
  std::vector<tensor::TensorPtr> input_tensors_;
  std::map<tensor::TensorPtr, session::KernelWithIndex> tensor_to_node_;
};

enum SessionTaskType {
  kUnknowTask = 0,
  kBuildTask,
  kLaunchTask,
};

class SessionTask {
 public:
  explicit SessionTask(SessionTaskType type, std::shared_ptr<RunOpContext> context)
      : type_(type), context_(std::move(context)) {}
  virtual ~SessionTask() = default;
  virtual void Run() = 0;
  const std::shared_ptr<RunOpContext> &context() { return context_; }

 protected:
  SessionTaskType type_;
  std::shared_ptr<RunOpContext> context_;
};

class BuildTask : public SessionTask {
 public:
  explicit BuildTask(std::shared_ptr<RunOpContext> context)
      : SessionTask(SessionTaskType::kBuildTask, std::move(context)) {}
  ~BuildTask() override = default;
  // Parallel build
  void Run() override {}
};

class LaunchTask : public SessionTask {
 public:
  explicit LaunchTask(std::shared_ptr<RunOpContext> context)
      : SessionTask(SessionTaskType::kLaunchTask, std::move(context)) {}
  ~LaunchTask() override = default;
  void Run() override {}
};

class PynativeTaskManager {
 public:
  static PynativeTaskManager &GetInstance() {
    static PynativeTaskManager instance;
    return instance;
  }

  class ExecuteGuard {
   public:
    ExecuteGuard() { PynativeTaskManager::GetInstance().executing_ = true; }
    ~ExecuteGuard() { PynativeTaskManager::GetInstance().executing_ = false; }
  };

  void Init(const std::function<void()> &execute_all) {
    execute_all_ = execute_all;
    inited_ = true;
  }
  const std::vector<std::shared_ptr<SessionTask>> &GetAllBuildTasks() const { return build_tasks_; }
  const std::queue<std::shared_ptr<SessionTask>> &GetAllLaunchTasks() const { return launch_tasks_; }
  void PopLaunchTask() { launch_tasks_.pop(); }
  void ClearAllBuildTasks() { build_tasks_.clear(); }
  void Reset() {
    ClearAllResources();
    execute_all_ = nullptr;
    inited_ = false;
  }
  void ClearAllResources() {
    build_tasks_.clear();
    std::queue<std::shared_ptr<SessionTask>> empty;
    std::swap(launch_tasks_, empty);
  }
  void ExecuteRemainingTasks() {
    if (!executing_) {
      ExecuteGuard guard;
      if (execute_all_ != nullptr) {
        execute_all_();
      }
    }
  }

  void PushBuildTask(const std::shared_ptr<SessionTask> &build_task) { build_tasks_.push_back(build_task); }
  void PushLaunchTask(const std::shared_ptr<SessionTask> &launch_task) { launch_tasks_.push(launch_task); }
  [[nodiscard]] bool QueueEmpty() const { return launch_tasks_.empty() && build_tasks_.empty(); }
  [[nodiscard]] bool QueueFull() const {
    return build_tasks_.size() > kMaxQueueSize || launch_tasks_.size() > kMaxQueueSize;
  }
  [[nodiscard]] bool inited() const { return inited_; }

 private:
  std::vector<std::shared_ptr<SessionTask>> build_tasks_;
  std::queue<std::shared_ptr<SessionTask>> launch_tasks_;
  std::function<void()> execute_all_{nullptr};
  inline static size_t kMaxQueueSize = 100;
  bool executing_{false};
  bool inited_{false};
};
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_BACKEND_SESSION_PYNATIVE_TASK_MANAGER_H_
