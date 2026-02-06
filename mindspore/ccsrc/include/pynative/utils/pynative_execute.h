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
#include "pynative/forward/op_graph/forward.h"
#include "pynative/backward/grad.h"

#include "pybind11/pybind11.h"
#include "include/frontend/operator/composite/grad_operation.h"
#include "ir/anf.h"
#include "include/fork_utils.h"
#include "include/utils/visible.h"

namespace mindspore::pynative {
namespace py = pybind11;

class PYNATIVE_EXPORT PyNativeExecutor : public std::enable_shared_from_this<PyNativeExecutor> {
 public:
  static const std::shared_ptr<PyNativeExecutor> &GetInstance() {
    std::lock_guard<std::mutex> i_lock(instance_lock_);
    if (executor_ == nullptr) {
      executor_ = std::shared_ptr<PyNativeExecutor>(new (std::nothrow) PyNativeExecutor());
      MS_EXCEPTION_IF_NULL(executor_);
      Init();
    }
    return executor_;
  }
  ~PyNativeExecutor() = default;
  static void Init();
  PyNativeExecutor(const PyNativeExecutor &) = delete;
  PyNativeExecutor &operator=(const PyNativeExecutor &) = delete;
  static inline const GradExecutorPtr &grad_executor() {
    MS_EXCEPTION_IF_NULL(grad_executor_);
    return grad_executor_;
  }
  static inline const ForwardExecutorPtr &forward_executor() {
    MS_EXCEPTION_IF_NULL(forward_executor_);
    return forward_executor_;
  }
  void StoreAsyncStatus(const PyboostOpRunInfoPtr &op_run_info) const;
  void StoreAsyncStatus(const FrontendOpRunInfoPtr &op_run_info) const;
  // Generate stub tensor and dispatch async task.
  py::object RunOpStub(const py::args &args) const;
  py::object RealRunOp(const py::args &args) const;
  void SetAsyncForGraph(bool flag) const;
  py::object CallConstantFolding(const py::args &args) const;
  bool grad_flag() const;
  void set_grad_flag(bool flag) const;
  bool enable_grad() const;
  void set_enable_grad(bool enable_grad) const;
  bool RequiresGrad() const;
  void set_py_exe_path(const py::object &py_exe_path) const;
  void set_kernel_build_server_dir(const py::object &kernel_build_server_dir) const;
  void SetHookCellId(const py::object &cell) const;
  void NewGraph(const py::object &obj, const py::args &args) const;
  void EndGraph(const py::object &obj, const py::object &out, const py::args &args) const;
  py::object RunGrad(const prim::GradOperationPtr &grad, const py::object &cell, const py::object &weights,
                     const py::object &grad_position, const py::object &has_aux, const py::args &args) const;
  py::object GradJit(const py::args &args) const;
  py::object CallCustomBprop(const py::object &cell_obj, const py::object &out, const py::args &args) const;
  void set_forward_use_dynamic_shape_process(bool flag) const;
  void SetDynamicInput(const py::object &obj, const py::args &args) const;
  py::object GetDynamicInput(const py::object &actual_input) const;
  bool IsHighOrder() const;
  py::object CheckAlreadyRun(const prim::GradOperationPtr &grad, const py::object &obj, const py::object &weights,
                             const py::object &grad_hash_id, const py::args &args, const py::kwargs &kwargs) const;
  void ClearRes() const;
  // Sync stream
  void Sync() const;
  void SetMixedPrecisionType(const MixedPrecisionType mix_type, bool is_push) const;
  void WorkerJoin();
  void SetJitCompilePhase(const std::string &phase) const;
  void SetIsRunRecompute(bool is_runing_recompute) const;
  void ParentBeforeFork();
  void ChildAfterFork();
  py::object RunSliceOpStub(const std::vector<ValuePtr> &input_v,
                            const std::vector<SliceOpInfoPtr> &slice_op_infos) const;
  void SetCreationType(const py::object &obj, autograd::CreationType creation_type);
  void QueueBackwardFinalCallback(const py::object &callback) const;
  void PushSavedTensorHook(const py::function &pack_hook, const py::function &unpack_hook);
  void PopSavedTensorHook();
  std::optional<std::string> DisableSavedTensorHook(const string &error_msg, bool is_error_on_outer_hook);
  void SetSavedTensorHookDisableErrorMessage(std::optional<std::string> error_msg);
  bool DisableFrontendAndBpropPipeline();
  void EnableFrontendAndBpropPipeline();
  bool IsSavedTensorHookActive();
  int64_t CurrentAutoDiffEngineId();

 private:
  PyNativeExecutor() = default;
  static std::shared_ptr<PyNativeExecutor> executor_;
  static std::mutex instance_lock_;
  static ForwardExecutorPtr forward_executor_;
  static GradExecutorPtr grad_executor_;
};
}  // namespace mindspore::pynative
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_EXECUTE_H_
