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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_BACKEND_OP_TASK_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_BACKEND_OP_TASK_H_

#include <utility>
#include <vector>
#include <memory>
#include <future>

#include "runtime/pynative/async/task.h"
#include "backend/common/session/session_basic.h"

namespace mindspore {
namespace pynative {
class OpTaskContext {
 public:
  OpTaskContext(GraphId graph_id, KernelGraphPtr graph, std::vector<session::KernelWithIndex> output_nodes,
                session::BackendOpRunInfoPtr op_run_info, device::DeviceContext *device_context, bool is_pynative_infer)
      : graph_id_(graph_id),
        graph_(std::move(graph)),
        output_nodes_(std::move(output_nodes)),
        op_run_info_(std::move(op_run_info)),
        device_context_(device_context),
        is_pyantive_infer_(is_pynative_infer) {}
  ~OpTaskContext() = default;

  GraphId graph_id() const { return graph_id_; }
  const KernelGraphPtr &graph() const { return graph_; }
  const std::vector<session::KernelWithIndex> &output_nodes() const { return output_nodes_; }
  const session::BackendOpRunInfoPtr &op_run_info() const { return op_run_info_; }
  device::DeviceContext *device_context() const { return device_context_; }
  bool is_pynative_infer() const { return is_pyantive_infer_; }

 private:
  GraphId graph_id_;
  KernelGraphPtr graph_;
  std::vector<session::KernelWithIndex> output_nodes_;
  session::BackendOpRunInfoPtr op_run_info_;
  device::DeviceContext *device_context_;
  bool is_pyantive_infer_{false};
};

class BackendOpTask : public AsyncTask {
 public:
  BackendOpTask(std::shared_ptr<OpTaskContext> context, pynative::TaskType task_type)
      : AsyncTask(task_type), context_(std::move(context)) {}
  ~BackendOpTask() override = default;

  void Run() override {}

  const std::shared_ptr<OpTaskContext> &context() { return context_; }

 protected:
  std::shared_ptr<OpTaskContext> context_;
};

class BackendOpRunTask : public BackendOpTask {
 public:
  BackendOpRunTask(std::shared_ptr<OpTaskContext> context,
                   std::function<void(const std::shared_ptr<OpTaskContext> &context)> run_func,
                   std::future<bool> future)
      : BackendOpTask(std::move(context), kOpRunTask), run_func_(std::move(run_func)), future_(std::move(future)) {}
  ~BackendOpRunTask() override = default;
  void Run() override;

 private:
  std::function<void(const std::shared_ptr<OpTaskContext> &context)> run_func_;
  std::future<bool> future_;
};

class BackendOpBuildTask : public BackendOpTask {
 public:
  BackendOpBuildTask(std::shared_ptr<OpTaskContext> context, std::promise<bool> promise)
      : BackendOpTask(std::move(context), kOpBuildTask), promise_(std::move(promise)) {}
  ~BackendOpBuildTask() override = default;
  void Run() override {}
  void SetBuildReady(bool build_success) { promise_.set_value(build_success); }

 private:
  std::promise<bool> promise_;
};
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_BACKEND_OP_TASK_H_
