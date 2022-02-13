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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_OP_BUILDER_OP_LAZY_BUILDER_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_OP_BUILDER_OP_LAZY_BUILDER_H_

#include <vector>
#include <memory>
#include <queue>
#include <map>
#include <string>
#include <utility>
#include "backend/common/session/kernel_graph.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "runtime/hardware/device_context.h"
#include "runtime/graph_scheduler/graph_scheduler.h"

namespace mindspore::runtime {
class OpLazyBuilderContext {
 public:
  OpLazyBuilderContext(GraphCompilerInfo *graph_compiler_info, KernelGraphPtr graph,
                       std::vector<session::KernelWithIndex> output_nodes, const session::OpRunInfo &op_run_info,
                       device::DeviceContext *device_context, bool is_pynative_infer)
      : graph_compiler_info_(graph_compiler_info),
        graph_(std::move(graph)),
        output_nodes_(std::move(output_nodes)),
        op_run_info_(op_run_info),
        device_context_(device_context),
        is_pyantive_infer_(is_pynative_infer) {}
  ~OpLazyBuilderContext() = default;

  GraphCompilerInfo *graph_compiler_info() const { return graph_compiler_info_; }
  const KernelGraphPtr &graph() const { return graph_; }
  const std::vector<session::KernelWithIndex> &output_nodes() const { return output_nodes_; }
  const session::OpRunInfo &op_run_info() const { return op_run_info_; }
  device::DeviceContext *device_context() const { return device_context_; }
  bool is_pynative_infer() const { return is_pyantive_infer_; }

 private:
  GraphCompilerInfo *graph_compiler_info_;
  KernelGraphPtr graph_;
  std::vector<session::KernelWithIndex> output_nodes_;
  session::OpRunInfo op_run_info_;
  device::DeviceContext *device_context_;
  bool is_pyantive_infer_{false};
};

class OpTask {
 public:
  explicit OpTask(std::shared_ptr<OpLazyBuilderContext> context) : context_(std::move(context)) {}
  virtual ~OpTask() = default;
  const std::shared_ptr<OpLazyBuilderContext> &context() { return context_; }

 protected:
  std::shared_ptr<OpLazyBuilderContext> context_;
};

class OpBuildTask : public OpTask {
 public:
  explicit OpBuildTask(std::shared_ptr<OpLazyBuilderContext> context) : OpTask(std::move(context)) {}
  ~OpBuildTask() override = default;
};

class OpRunTask : public OpTask {
 public:
  explicit OpRunTask(std::shared_ptr<OpLazyBuilderContext> context) : OpTask(std::move(context)) {}
  ~OpRunTask() override = default;
};

class OpLazyBuilder {
 public:
  static OpLazyBuilder &GetInstance() {
    static OpLazyBuilder instance;
    return instance;
  }

  class ExecuteGuard {
   public:
    ExecuteGuard() { OpLazyBuilder::GetInstance().executing_ = true; }
    ~ExecuteGuard() { OpLazyBuilder::GetInstance().executing_ = false; }
  };

  void Register(const std::function<void()> &callback);
  const std::vector<std::shared_ptr<OpTask>> &GetOpBuildTasks() const { return op_build_tasks; }
  const std::queue<std::shared_ptr<OpTask>> &GetOpRunTasks() const { return op_run_tasks; }
  void ClearOpBuildTasks() { op_build_tasks.clear(); }
  void Reset();
  void ClearAllResources();
  void ExecuteRemainingTasks();

  void PushOpBuildTask(const std::shared_ptr<OpTask> &op_build_task) { op_build_tasks.push_back(op_build_task); }
  void PushOpRunTask(const std::shared_ptr<OpTask> &op_run_task) { op_run_tasks.push(op_run_task); }
  void PopOpRunTask() { op_run_tasks.pop(); }
  bool QueueEmpty() const { return op_run_tasks.empty() && op_build_tasks.empty(); }
  bool QueueFull() const { return op_build_tasks.size() > kMaxQueueSize || op_run_tasks.size() > kMaxQueueSize; }
  bool registered() const { return registered_; }

 private:
  OpLazyBuilder() = default;
  ~OpLazyBuilder() = default;
  DISABLE_COPY_AND_ASSIGN(OpLazyBuilder);
  std::vector<std::shared_ptr<OpTask>> op_build_tasks;
  std::queue<std::shared_ptr<OpTask>> op_run_tasks;
  std::function<void()> execute_callback_{nullptr};
  inline static size_t kMaxQueueSize = 20;
  bool executing_{false};
  bool registered_{false};
};
}  // namespace mindspore::runtime
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_OP_BUILDER_OP_LAZY_BUILDER_H_
