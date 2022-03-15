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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_OP_TASK_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_OP_TASK_H_

#include <vector>
#include <memory>
#include <queue>
#include <map>
#include <string>
#include <utility>
#include "backend/common/session/kernel_graph.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "runtime/hardware/device_context.h"
#include "runtime/graph_scheduler/graph_scheduler.h"

namespace mindspore::runtime {
class OpTaskContext {
 public:
  OpTaskContext(GraphCompilerInfo *graph_compiler_info, KernelGraphPtr graph,
                std::vector<session::KernelWithIndex> output_nodes, session::OpRunInfo op_run_info,
                device::DeviceContext *device_context, bool is_pynative_infer)
      : graph_compiler_info_(graph_compiler_info),
        graph_(std::move(graph)),
        output_nodes_(std::move(output_nodes)),
        op_run_info_(std::move(op_run_info)),
        device_context_(device_context),
        is_pyantive_infer_(is_pynative_infer) {}
  ~OpTaskContext() = default;

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

enum OpTaskType {
  kBuildTask,
  kRunTask,
  kExitTask,
};

class OpTask {
 public:
  OpTask(std::shared_ptr<OpTaskContext> context, OpTaskType task_type)
      : context_(std::move(context)), task_type_(task_type) {}
  virtual ~OpTask() = default;

  virtual void Run() = 0;
  OpTaskType task_type() const { return task_type_; }
  const std::shared_ptr<OpTaskContext> &context() { return context_; }

 protected:
  std::shared_ptr<OpTaskContext> context_;
  OpTaskType task_type_;
};

class OpBuildTask : public OpTask {
 public:
  OpBuildTask(std::shared_ptr<OpTaskContext> context, std::promise<bool> promise)
      : OpTask(std::move(context), kBuildTask), promise_(std::move(promise)) {}
  ~OpBuildTask() override = default;
  void Run() override {}
  void SetBuildReady(bool build_success) { promise_.set_value(build_success); }

 private:
  std::promise<bool> promise_;
};

class OpRunTask : public OpTask {
 public:
  OpRunTask(std::shared_ptr<OpTaskContext> context,
            std::function<void(const std::shared_ptr<runtime::OpTaskContext> &context)> run, std::future<bool> future)
      : OpTask(std::move(context), kRunTask), run_(std::move(run)), future_(std::move(future)) {}
  ~OpRunTask() override = default;
  void Run() override {
    MS_LOG(DEBUG) << "Wait for build";
    auto build_status = future_.get();
    if (!build_status) {
      MS_LOG(WARNING) << "Op build failed, no need to launch.";
      return;
    }
    MS_EXCEPTION_IF_NULL(run_);
    run_(context_);
  }

 private:
  std::function<void(const std::shared_ptr<runtime::OpTaskContext> &context)> run_;
  std::future<bool> future_;
};

class ExitOpTask : public OpTask {
 public:
  ExitOpTask() : OpTask(nullptr, kExitTask) {}
  ~ExitOpTask() override = default;
  void Run() override {}
};
}  // namespace mindspore::runtime
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_OP_TASK_H_
