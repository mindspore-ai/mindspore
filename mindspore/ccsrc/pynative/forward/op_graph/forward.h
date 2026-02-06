/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_FORWARD_FORWARD_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_FORWARD_FORWARD_H_

#include <memory>
#include <string>
#include <map>
#include <utility>
#include <stack>
#include <vector>

#include "pynative/forward/op_graph/do_cast.h"
#include "pynative/forward/pyboost/do_pyboost_cast.h"
#include "pynative/forward/op_graph/do_infer.h"
#include "ir/cell.h"
#include "view/view_strides_calculator.h"
#include "include/runtime/pipeline/async_rqueue.h"
#include "include/pynative/utils/runtime/op_backend/op_backend.h"
#include "device_address/device_type.h"

namespace mindspore {
namespace pynative {
class GradExecutor;
using GradExecutorPtr = std::shared_ptr<GradExecutor>;
using GradExecutorWeakPtr = std::weak_ptr<GradExecutor>;

class ForwardExecutor {
 public:
  ForwardExecutor()
      : cast_operation_(std::make_shared<CastOperation>()),
        pyboost_cast_operation_(std::make_shared<PyBoostCastOperation>()),
        infer_operation_(std::make_shared<InferOperation>()) {}
  ~ForwardExecutor() = default;

  void Init();
  std::function<void(const FrontendOpRunInfoPtr &)> RunOpS = [this](auto &&PH1) {
    RunOpFrontend(std::forward<decltype(PH1)>(PH1));
  };

  void DispatchFrontendTask(const FrontendOpRunInfoPtr &op_run_info);
  void RunOpFrontend(const FrontendOpRunInfoPtr &op_run_info);
  FrontendOpRunInfoPtr GenerateOpRunInfo(const py::args &args, bool stub = false);
  ValuePtr RunSliceOpFrontend(const std::vector<ValuePtr> &input_values,
                              const std::vector<SliceOpInfoPtr> &slice_op_infos, bool requires_grad,
                              const stub::StubNodePtr &stub_output, size_t stream_id);
  void DispatchSilceOpFrontendTask(const std::vector<ValuePtr> &input_values,
                                   const std::vector<SliceOpInfoPtr> &slice_op_infos, bool requires_grad,
                                   const stub::StubNodePtr &stub_output, size_t stream_id);
  void set_grad_executor(const GradExecutorPtr &grad_executor) { grad_executor_ = GradExecutorWeakPtr(grad_executor); }
  void RefreshForwardCallback();
  void ClearNodeAbsMap() const;
  void SetNodeAbsMapByValue(const FrontendOpRunInfoPtr &op_run_info) const;
  void SetNodeAbsMapById(const std::string &id, const abstract::AbstractBasePtr &abs) const;
  AbstractBasePtr GetNodeAbsById(const std::string &id) const;
  void ClearRes();
  bool EnablePipeline(const std::string &op_name) const;
  bool enable_async() const;
  device::DeviceType device_target() const { return device_target_; }
  void set_mix_precision_type(const MixedPrecisionType mix_precision_type, bool is_push) {
    is_push ? mix_precision_type_stack_.push(mix_precision_type) : mix_precision_type_stack_.pop();
    MS_LOG(DEBUG) << "Set mix precision type " << mix_precision_type << ", is push " << is_push;
  }
  void ExecuteLazyTask() const;
  void Sync(const bool &);
  std::function<void(const bool &)> SyncData = [this](auto &&PH1) { return Sync(std::forward<decltype(PH1)>(PH1)); };
  bool CellNotSetMixedPrecision(const PyboostOpRunInfoPtr &op_run_info);
  bool CellNotSetMixedPrecision(const FrontendOpRunInfoPtr &op_run_info);
  inline InferOperationPtr infer_operation() const {
    MS_EXCEPTION_IF_NULL(infer_operation_);
    return infer_operation_;
  }

  void WaitForwardTask();
  device::DeviceType GetCurrentDeviceTarget(const PrimitivePtr &op_prim) const;
  void ReInit();
  void ForwardOpGradImpl(const OpGradInfoPtr &grad_info, const AsyncStatus &async_status) const;
  GradExecutorPtr grad() const;
  void InitOpRunInfo(const PyboostOpRunInfoPtr &op_run_info);
  void InitOpRunInfo(const FrontendOpRunInfoPtr &op_run_info);
  // Mix precision and Implicit transform
  void SetCastForInputs(const FrontendOpRunInfoPtr &op_run_info) const;
  inline const PyBoostCastOperationPtr &pyboost_cast_operation() const {
    MS_EXCEPTION_IF_NULL(pyboost_cast_operation_);
    return pyboost_cast_operation_;
  }
  void ChildAfterFork();

 private:
  inline CastOperationPtr cast_operation() const {
    MS_EXCEPTION_IF_NULL(cast_operation_);
    return cast_operation_;
  }
  ValuePtr RunOpInVM(const FrontendOpRunInfoPtr &op_run_info) const;
  ValuePtr RunOpInMs(const FrontendOpRunInfoPtr &op_run_info, const compile::BackendOpRunInfoPtr &backend_op_run_info);
  ValuePtr RunOpInMsInner(const FrontendOpRunInfoPtr &op_run_info,
                          const compile::BackendOpRunInfoPtr &backend_op_run_info);
  ValuePtr RunOpWithBackendPolicy(const FrontendOpRunInfoPtr &op_run_info,
                                  const compile::BackendOpRunInfoPtr &backend_op_run_info);
  void RunOpBackend(const FrontendOpRunInfoPtr &op_run_info, const compile::BackendOpRunInfoPtr &backend_op_run_info);
  void RunOpBackendSync(const FrontendOpRunInfoPtr &op_run_info);

  VectorRef RunOpBackendInner(const FrontendOpRunInfoPtr &op_run_info,
                              const compile::BackendOpRunInfoPtr &backend_op_run_info);
  // Infer output abstract
  void InferOutputAbstract(const FrontendOpRunInfoPtr &op_run_info) const;
  void PrepareOpInputs(const FrontendOpRunInfoPtr &op_run_info);
  void OpRunInfoUsePrimC(const FrontendOpRunInfoPtr &op_run_info) const;
  void CreateInputAddressForViewOp(const tensor::TensorPtr &input_tensor, const FrontendOpRunInfoPtr &op_run_info);
  void DispatchViewKernelTask(const FrontendOpRunInfoPtr &op_run_info, const runtime::KernelTaskType &task_type);
  void ForwardRunViewKernelTask(const FrontendOpRunInfoPtr &op_run_info, const runtime::KernelTaskType &task_type,
                                bool enable_async);

  bool ProcessViewOp(const FrontendOpRunInfoPtr &op_run_info, const ops::StridesCalcFunc &func_info,
                     bool is_tuple_output);
  device::DeviceAddressPtr TensorContiguousCallback(const tensor::TensorPtr &input_tensor);

  void CreateViewOutputTensor(const FrontendOpRunInfoPtr &op_run_info, const tensor::TensorPtr &input_tensor,
                              const TensorStorageInfoPtr &storage_info, runtime::KernelTaskType task_type,
                              bool is_multi_output);

  void DispatchAllocateMemTask(const FrontendOpRunInfoPtr &op_run_info, const tensor::TensorPtr &input_tensor,
                               const size_t &input_idx, bool need_wait = false);
  PrimitivePtr GetSlicePrimFromCache(const std::string &op_name);
  FrontendOpRunInfoPtr GenerateSliceOpRunInfo(const std::string &op_name, bool requires_grad,
                                              const stub::StubNodePtr &stub_output, size_t stream_id);
  void CreateViewOpOutputs(const FrontendOpRunInfoPtr &op_run_info, const tensor::TensorPtr &view_input_tensor,
                           runtime::KernelTaskType task_type, const TensorStorageInfoPtrList &storage_infos,
                           bool is_tuple_output);

 private:
  bool init_{false};
  bool enable_async_{true};
  std::stack<MixedPrecisionType> mix_precision_type_stack_;
  device::DeviceType device_target_{device::DeviceType::kUnknown};
  std::string last_target_{"Unknown"};
  GradExecutorWeakPtr grad_executor_;
  CastOperationPtr cast_operation_;
  PyBoostCastOperationPtr pyboost_cast_operation_;
  InferOperationPtr infer_operation_;
  compile::OpBackendPtr op_backend_{nullptr};
  mindspore::HashMap<std::string, PrimitivePtr> slice_prim_cache_;
};
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_FORWARD_FORWARD_H_
