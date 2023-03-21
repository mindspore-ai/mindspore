/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_EXECUTE_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_EXECUTE_H_

#include <memory>
#include <string>
#include <vector>
#include "pipeline/pynative/forward/forward.h"
#include "pipeline/pynative/grad/grad.h"

#include "pybind11/pybind11.h"
#include "frontend/operator/composite/composite.h"

namespace mindspore::pynative {
namespace py = pybind11;

class PyNativeExecutor : public std::enable_shared_from_this<PyNativeExecutor> {
 public:
  static std::shared_ptr<PyNativeExecutor> GetInstance() {
    std::lock_guard<std::mutex> i_lock(instance_lock_);
    if (executor_ == nullptr) {
      executor_ = std::shared_ptr<PyNativeExecutor>(new (std::nothrow) PyNativeExecutor());
      Init();
    }
    return executor_;
  }
  ~PyNativeExecutor() = default;
  static void Init();
  PyNativeExecutor(const PyNativeExecutor &) = delete;
  PyNativeExecutor &operator=(const PyNativeExecutor &) = delete;
  inline GradExecutorPtr grad_executor() const {
    MS_EXCEPTION_IF_NULL(grad_executor_);
    return grad_executor_;
  }
  inline ForwardExecutorPtr forward_executor() const {
    MS_EXCEPTION_IF_NULL(forward_executor_);
    return forward_executor_;
  }

  bool DisablePyTraceAsync(const FrontendOpRunInfoPtr &op_run_info) const;
  void StoreAsyncStatus(const FrontendOpRunInfoPtr &op_run_info) const;
  py::object RunOpAsync(const py::args &args) const;
  py::object RealRunOp(const py::args &args) const;
  py::object CallConstantFolding(const py::args &args) const;
  bool grad_flag() const;
  void set_grad_flag(bool flag) const;
  void set_py_exe_path(const py::object &py_exe_path) const;
  void set_kernel_build_server_dir(const py::object &kernel_build_server_dir) const;
  void SetHookChanged(const py::object &cell) const;
  void NewGraph(const py::object &obj, const py::args &args) const;
  void EndGraph(const py::object &obj, const py::object &out, const py::args &args) const;
  py::object Run() const;
  void GradNet(const prim::GradOperationPtr &grad, const py::object &cell, const py::object &weights,
               const py::object &grad_position, const py::args &args) const;
  py::object GradMsFunction(const py::object &out, const py::args &args) const;
  void SetDynamicInput(const py::object &cell) const;

  py::object CheckAlreadyRun(const prim::GradOperationPtr &grad, const py::object &obj, const py::object &weights,
                             const py::object &grad_hash_id, const py::args &args) const;
  void ClearRes() const;
  // Sync stream
  void Sync() const;
  void SetLazyBuild(bool enable) const;
  bool IsFirstCell() const;
  void WorkerJoin();
  void SetMsFunctionCompileStatus(bool is_compiling, const std::string &phase) const;

 private:
  PyNativeExecutor() = default;
  static std::shared_ptr<PyNativeExecutor> executor_;
  static std::mutex instance_lock_;
  static ForwardExecutorPtr forward_executor_;
  static GradExecutorPtr grad_executor_;
};
}  // namespace mindspore::pynative
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_EXECUTE_H_
