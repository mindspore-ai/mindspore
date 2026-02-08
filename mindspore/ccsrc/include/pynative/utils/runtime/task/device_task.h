/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_TASK_DEVICE_TASK_H_
#define MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_TASK_DEVICE_TASK_H_

#include <utility>
#include <vector>
#include <memory>
#include <future>

#include "runtime/pipeline/task/task.h"
#include "include/backend/common/kernel_graph/session_basic.h"
#include "include/pynative/utils/runtime/op_compiler.h"

namespace mindspore {
namespace runtime {
class PYNATIVE_UTILS_EXPORT OpTaskContext {
 public:
  OpTaskContext(GraphId graph_id, KernelGraphPtr graph, session::BackendOpRunInfoPtr op_run_info,
                OpCompilerInfoPtr op_compiler_info, bool is_pynative_infer)
      : graph_id_(graph_id),
        graph_(std::move(graph)),
        op_run_info_(std::move(op_run_info)),
        op_compiler_info_(std::move(op_compiler_info)),
        is_pyantive_infer_(is_pynative_infer) {}
  ~OpTaskContext() = default;

  GraphId graph_id() const { return graph_id_; }
  const KernelGraphPtr &graph() const { return graph_; }
  const session::BackendOpRunInfoPtr &op_run_info() const { return op_run_info_; }
  const device::DeviceContext *device_context() const { return op_compiler_info_->device_context_; }
  bool is_pynative_infer() const { return is_pyantive_infer_; }
  const OpCompilerInfoPtr &op_compiler_info() const { return op_compiler_info_; }

 private:
  GraphId graph_id_;
  KernelGraphPtr graph_;
  session::BackendOpRunInfoPtr op_run_info_;
  OpCompilerInfoPtr op_compiler_info_;
  bool is_pyantive_infer_;
};

class PYNATIVE_UTILS_EXPORT DeviceOpTask : public AsyncTask {
 public:
  DeviceOpTask(std::shared_ptr<OpTaskContext> context, TaskType task_type)
      : AsyncTask(task_type), context_(std::move(context)) {}
  ~DeviceOpTask() override = default;

  void Run() override {}

  const std::shared_ptr<OpTaskContext> &context() { return context_; }

 protected:
  std::shared_ptr<OpTaskContext> context_;
};

class PYNATIVE_UTILS_EXPORT DeviceOpRunTask : public DeviceOpTask {
 public:
  DeviceOpRunTask(std::shared_ptr<OpTaskContext> context,
                  std::function<void(const std::shared_ptr<OpTaskContext> &context)> run_func);
  ~DeviceOpRunTask() override;
  void Run() override;

 private:
  std::function<void(const std::shared_ptr<OpTaskContext> &context)> run_func_;
};

class PYNATIVE_UTILS_EXPORT PyBoostDeviceTask : public AsyncTask {
 public:
  explicit PyBoostDeviceTask(std::function<void()> run_func) : AsyncTask(kPyBoostOpTask), run_func_(run_func) {}
  ~PyBoostDeviceTask() = default;

  void Run() override;

 private:
  std::function<void()> run_func_;
};

class PYNATIVE_UTILS_EXPORT DeviceLaunchTask : public AsyncTask {
 public:
  explicit DeviceLaunchTask(std::function<void()> run_func) : AsyncTask(kKernelTask), run_func_(std::move(run_func)) {}
  ~DeviceLaunchTask() = default;

  void Run() override;

 private:
  std::function<void()> run_func_;
};

class PYNATIVE_UTILS_EXPORT PassthroughDeviceTask : public AsyncTask {
 public:
  explicit PassthroughDeviceTask(std::function<void(void)> run_func)
      : AsyncTask(kDeviceOpTask), run_func_(std::move(run_func)) {}
  ~PassthroughDeviceTask() override = default;
  void Run() override;

 private:
  std::function<void(void)> run_func_;
};

class PYNATIVE_UTILS_EXPORT PassthroughNoWaitDeviceTask : public AsyncTask {
 public:
  explicit PassthroughNoWaitDeviceTask(std::function<void(void)> run_func)
      : AsyncTask(kDeviceOpTask), run_func_(std::move(run_func)) {}
  ~PassthroughNoWaitDeviceTask() override = default;
  void Run() override;

 private:
  std::function<void(void)> run_func_;
};
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_TASK_DEVICE_TASK_H_
