/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_OP_BACKEND_H_
#define MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_OP_BACKEND_H_

#include <memory>
#include <map>
#include <vector>
#include <string>
#include "include/utils/visible.h"
#include "include/backend/common/kernel_graph/session_basic.h"
#include "include/pynative/utils/runtime/op_compiler.h"
#include "include/pynative/utils/runtime/task/device_task.h"

namespace mindspore::compile {
using BackendOpRunInfoPtr = session::BackendOpRunInfoPtr;
using KernelTensor = kernel::KernelTensor;
using KernelTensorPtr = kernel::KernelTensorPtr;

class PYNATIVE_UTILS_EXPORT ViewBackend {
 public:
  void RunViewKernelTask(const pynative::BaseOpRunInfo &base_op_run_info, const runtime::KernelTaskType &task_type,
                         bool enable_async) const;

  void RunAllocMemTask(DeviceContext *device_context, const tensor::TensorPtr &tensor, bool enable_async) const;

  void RunViewKernelTaskAsyncImpl(const runtime::KernelTaskType &task_type, DeviceContext *device_context,
                                  const tensor::TensorPtrList &input_tensors,
                                  const tensor::TensorPtrList &output_tensors, const size_t &stream_id) const;

  void AllocateMemForTensor(const tensor::TensorPtr &tensor, DeviceContext *device_context) const;

  static void ContiguousInputByRunInfo(const BackendOpRunInfoPtr &op_run_info);

  using ContiguousFunc = std::function<void(const BackendOpRunInfoPtr &)>;

  static void SetContiguousFunc(const ContiguousFunc &contiguous_func) { contiguous_func_ = contiguous_func; }

 private:
  inline static ContiguousFunc contiguous_func_;
};

class PYNATIVE_UTILS_EXPORT PostRunOp {
 public:
  void UpdateOutput(const std::vector<session::KernelWithIndex> &output_nodes, VectorRef *outputs) const;

  void ClearGraphDeviceAddress(const KernelGraphPtr &graph, const DeviceContext *device_context,
                               bool is_gradient_out) const;

  void ClearInputDeviceAddress(const KernelGraphPtr &graph, const DeviceContext *device_context) const;

  void ClearOpInputOutput(const OpCompilerInfoPtr &op_compiler_info) const;

  void UpdateOutputAbstract(const VectorRef &outputs, const session::BackendOpRunInfoPtr &op_run_info) const;

  void UpdateOutputDynamic(const session::BackendOpRunInfoPtr &op_run_info, const OpCompilerInfoPtr &op_compiler_info,
                           const std::vector<KernelTensorPtr> &kernel_tensor_list, VectorRef *outputs) const;

 private:
  tensor::TensorPtr CreateOutputTensor(const AnfNodePtr &output_node, size_t output_index) const;

  tensor::TensorPtr CreateOutputTensorDynamicImpl(const OpCompilerInfoPtr &op_compiler_info,
                                                  const AnfNodePtr &output_node, size_t output_index,
                                                  const KernelTensorPtr &kernel_tensor,
                                                  size_t idx_in_graph_outputs) const;
};

class PYNATIVE_UTILS_EXPORT OpBackend {
 public:
  OpBackend() = default;
  ~OpBackend() = default;
  // Run op on device.
  void Run(const BackendOpRunInfoPtr &op_run_info, device::DeviceType device_type, uint32_t device_id,
           VectorRef *outputs);

  void RunViewKernelTask(const pynative::BaseOpRunInfo &base_op_run_info, const runtime::KernelTaskType &task_type,
                         bool enable_async) const;

  void RunAllocMemTask(DeviceContext *device_context, const tensor::TensorPtr &tensor, bool enable_async) const;

 protected:
  void RunInner(const BackendOpRunInfoPtr &op_run_info, const std::string &device_name, uint32_t device_id,
                VectorRef *outputs);

  void RunOpImpl(bool single_op_cache_hit, const OpCompilerInfoPtr &op_compiler_info,
                 const session::BackendOpRunInfoPtr &op_run_info, VectorRef *outputs);

  void OpRunCallback(const std::shared_ptr<runtime::OpTaskContext> &context);

  void DispatchOpTask(bool single_op_cache_hit, VectorRef *outputs, const OpCompilerInfoPtr &op_compiler_info,
                      const session::BackendOpRunInfoPtr &op_run_info);

  void RunInnerDynamic(const BackendOpRunInfoPtr &op_run_info, const std::string &device_name, uint32_t device_id,
                       VectorRef *outputs);

  void RunOpImplDynamic(bool single_op_cache_hit, const OpCompilerInfoPtr &op_compiler_info,
                        const session::BackendOpRunInfoPtr &op_run_info, VectorRef *outputs);

  void DispatchOpTaskDynamic(VectorRef *outputs, const OpCompilerInfoPtr &op_compiler_info,
                             const session::BackendOpRunInfoPtr &op_run_info,
                             const std::vector<KernelTensorPtr> &kernel_tensor_list);

  void OpRunCallbackDynamic(const std::shared_ptr<runtime::OpTaskContext> &context);

  std::vector<KernelTensorPtr> GetOutputKernelTensor(const OpCompilerInfoPtr &op_compiler_info) const;

  PostRunOp post_run_;
  ViewBackend view_backend_;
};
using OpBackendPtr = std::unique_ptr<OpBackend>;
}  // namespace mindspore::compile

#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_OP_BACKEND_H_
