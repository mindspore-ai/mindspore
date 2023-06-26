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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_DEVICE_TASK_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_DEVICE_TASK_H_

#include <utility>
#include <vector>
#include <memory>
#include <future>

#include "runtime/pynative/async/task.h"
#include "backend/common/session/session_basic.h"
#include "runtime/pynative/op_compiler.h"

namespace mindspore {
namespace pynative {
class OpTaskContext {
 public:
  OpTaskContext(GraphId graph_id, KernelGraphPtr graph, session::BackendOpRunInfoPtr op_run_info,
                device::DeviceContext *device_context, bool is_pynative_infer)
      : graph_id_(graph_id),
        graph_(std::move(graph)),
        op_run_info_(std::move(op_run_info)),
        device_context_(device_context),
        is_pyantive_infer_(is_pynative_infer) {}
  ~OpTaskContext() = default;

  GraphId graph_id() const { return graph_id_; }
  const KernelGraphPtr &graph() const { return graph_; }
  const session::BackendOpRunInfoPtr &op_run_info() const { return op_run_info_; }
  device::DeviceContext *device_context() const { return device_context_; }
  bool is_pynative_infer() const { return is_pyantive_infer_; }
  void set_op_compiler_info(const OpCompilerInfoPtr &op_compiler_info) { op_compiler_info_ = op_compiler_info; }
  OpCompilerInfoPtr op_compiler_info() const { return op_compiler_info_; }
  void set_device_address_list(const vector<device::DeviceAddressPtr> &device_address_list) {
    device_address_list_ = device_address_list;
  }
  vector<device::DeviceAddressPtr> &device_address_list() { return device_address_list_; }

 private:
  GraphId graph_id_;
  KernelGraphPtr graph_;
  std::vector<session::KernelWithIndex> output_nodes_;
  session::BackendOpRunInfoPtr op_run_info_;
  device::DeviceContext *device_context_;
  bool is_pyantive_infer_{false};
  OpCompilerInfoPtr op_compiler_info_;
  vector<device::DeviceAddressPtr> device_address_list_;
};

class DeviceOpTask : public AsyncTask {
 public:
  DeviceOpTask(std::shared_ptr<OpTaskContext> context, pynative::TaskType task_type)
      : AsyncTask(task_type), context_(std::move(context)) {}
  ~DeviceOpTask() override = default;

  void Run() override {}

  const std::shared_ptr<OpTaskContext> &context() { return context_; }

 protected:
  std::shared_ptr<OpTaskContext> context_;
};

class DeviceOpRunTask : public DeviceOpTask {
 public:
  DeviceOpRunTask(std::shared_ptr<OpTaskContext> context,
                  std::function<void(const std::shared_ptr<OpTaskContext> &context)> run_func, std::future<bool> future)
      : DeviceOpTask(std::move(context), kDeviceOpTask), run_func_(std::move(run_func)), future_(std::move(future)) {}
  ~DeviceOpRunTask() override = default;
  void Run() override;

 private:
  std::function<void(const std::shared_ptr<OpTaskContext> &context)> run_func_;
  std::future<bool> future_;
};

class DeviceOpBuildTask : public DeviceOpTask {
 public:
  DeviceOpBuildTask(std::shared_ptr<OpTaskContext> context, std::promise<bool> promise)
      : DeviceOpTask(std::move(context), kDeviceOpBuildTask), promise_(std::move(promise)), has_set_value_(false) {}
  ~DeviceOpBuildTask() override;
  void Run() override {}
  void SetBuildReady(bool build_success);

 private:
  std::promise<bool> promise_;
  bool has_set_value_;
};

class AllocViewMemDeviceTask : public AsyncTask {
 public:
  AllocViewMemDeviceTask(
    std::function<void(device::DeviceContext *device_context, const tensor::TensorPtr &tensor)> run_func,
    device::DeviceContext *device_context, const tensor::TensorPtr &tensor)
      : AsyncTask(kDeviceOpTask), run_func_(std::move(run_func)), device_context_(device_context), tensor_(tensor) {}
  ~AllocViewMemDeviceTask() override = default;
  void Run() override;

 private:
  std::function<void(device::DeviceContext *device_context, const tensor::TensorPtr &tensor)> run_func_;
  device::DeviceContext *device_context_;
  tensor::TensorPtr tensor_;
};

class KernelDeviceTask : public AsyncTask {
 public:
  KernelDeviceTask(
    std::function<void(const KernelTaskType &task_type, const device::DeviceAddressPtrList &input_addr_list,
                       const TensorStorageInfoPtrList &input_storage_list,
                       const device::DeviceAddressPtrList &output_addr_list, device::DeviceContext *device_context)>
      run_func,
    const KernelTaskType &task_type, device::DeviceContext *device_context,
    const device::DeviceAddressPtrList &input_addr_list, const TensorStorageInfoPtrList &input_storage_list,
    const device::DeviceAddressPtrList &output_addr_list)
      : AsyncTask(kDeviceOpTask),
        run_func_(std::move(run_func)),
        task_type_(task_type),
        device_context_(device_context),
        input_addr_list_(input_addr_list),
        input_storage_list_(input_storage_list),
        output_addr_list_(output_addr_list) {}
  ~KernelDeviceTask() override = default;
  void Run() override;

 private:
  std::function<void(const KernelTaskType &task_type, const device::DeviceAddressPtrList &input_addr_list,
                     const TensorStorageInfoPtrList &input_storage_list,
                     const device::DeviceAddressPtrList &output_addr_list, device::DeviceContext *device_context)>
    run_func_;
  KernelTaskType task_type_;
  device::DeviceContext *device_context_;
  device::DeviceAddressPtrList input_addr_list_;
  TensorStorageInfoPtrList input_storage_list_;
  device::DeviceAddressPtrList output_addr_list_;
};
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_ASYNC_DEVICE_TASK_H_
