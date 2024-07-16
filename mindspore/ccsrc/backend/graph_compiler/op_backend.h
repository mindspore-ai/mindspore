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

#ifndef MINDSPORE_MINDSPORE_CCSRC_BACKEND_GRAPH_COMPILER_OP_BACKEND_H_
#define MINDSPORE_MINDSPORE_CCSRC_BACKEND_GRAPH_COMPILER_OP_BACKEND_H_

#include <memory>
#include <map>
#include <vector>
#include <string>
#include "include/backend/visible.h"
#include "backend/common/session/session_basic.h"
#include "runtime/pynative/op_compiler.h"
#include "runtime/pipeline/task/device_task.h"

namespace mindspore::compile {
using BackendOpRunInfoPtr = session::BackendOpRunInfoPtr;

class BACKEND_EXPORT ViewBackend {
 public:
  void RunViewKernelTask(const pynative::BaseOpRunInfo &base_op_run_info, const runtime::KernelTaskType &task_type,
                         bool enable_async) const;

  void RunAllocMemTask(DeviceContext *device_context, const tensor::BaseTensorPtr &tensor, bool enable_async,
                       bool is_cpu_address_exist) const;

  void RunViewKernelTaskAsyncImpl(const runtime::KernelTaskType &task_type, DeviceContext *device_context,
                                  const device::DeviceAddressPtrList &input_addr_list,
                                  const device::DeviceAddressPtrList &output_addr_list, const size_t &stream_id) const;

  void AllocateMemForTensor(const tensor::BaseTensorPtr &tensor, DeviceContext *device_context,
                            bool is_cpu_address_exist) const;
};

class BACKEND_EXPORT PostRunOp {
 public:
  void UpdateOutput(const std::vector<session::KernelWithIndex> &output_nodes, VectorRef *outputs) const;

  void ReleaseForwardOpOutput(const std::vector<ValuePtr> &input_tensors);

  void ClearGraphDeviceAddress(const KernelGraphPtr &graph, const DeviceContext *device_context,
                               bool is_gradient_out) const;

  void ClearInputDeviceAddress(const KernelGraphPtr &graph, const DeviceContext *device_context) const;

  void ClearOpInputOutput(const OpCompilerInfoPtr &op_compiler_info) const;

  void UpdateOutputAbstract(const VectorRef &outputs, const session::BackendOpRunInfoPtr &op_run_info) const;

  void UpdateOutputDynamic(const session::BackendOpRunInfoPtr &op_run_info, const OpCompilerInfoPtr &op_compiler_info,
                           const vector<device::DeviceAddressPtr> &device_address_list, VectorRef *outputs) const;

  void set_forward_tensor_ref_count(const std::map<std::string, size_t> &forward_tensor_ref_count) {
    forward_tensor_ref_count_ = forward_tensor_ref_count;
  }

 private:
  tensor::BaseTensorPtr CreateOutputTensor(const AnfNodePtr &output_node, size_t output_index) const;

  tensor::BaseTensorPtr CreateOutputTensorDynamicImpl(const OpCompilerInfoPtr &op_compiler_info,
                                                      const AnfNodePtr &output_node, size_t output_index,
                                                      const std::shared_ptr<device::DeviceAddress> &address,
                                                      size_t idx_in_graph_outputs) const;

  // Cache forward op output value node tensor ref count of kernels for back propagation graph in PyNative mode.
  std::map<std::string, size_t> forward_tensor_ref_count_;
};

class BACKEND_EXPORT OpBackend {
 public:
  OpBackend() = default;
  ~OpBackend() = default;
  // Run op on device.
  void Run(const BackendOpRunInfoPtr &op_run_info, const std::string &device_name, uint32_t device_id,
           VectorRef *outputs);

  void set_forward_tensor_ref_count(const std::map<std::string, size_t> &forward_tensor_ref_count) {
    post_run_.set_forward_tensor_ref_count(forward_tensor_ref_count);
  }

  void RunViewKernelTask(const pynative::BaseOpRunInfo &base_op_run_info, const runtime::KernelTaskType &task_type,
                         bool enable_async) const;

  void RunAllocMemTask(DeviceContext *device_context, const tensor::BaseTensorPtr &tensor, bool enable_async,
                       bool is_cpu_address_exist) const;

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
                             const vector<device::DeviceAddressPtr> &device_address_list);

  void OpRunCallbackDynamic(const std::shared_ptr<runtime::OpTaskContext> &context);

  device::DeviceAddressPtrList GetOutputDeviceAddress(const OpCompilerInfoPtr &op_compiler_info) const;

  PostRunOp post_run_;
  ViewBackend view_backend_;
};
using OpBackendPtr = std::unique_ptr<OpBackend>;
}  // namespace mindspore::compile

#endif  // MINDSPORE_MINDSPORE_CCSRC_BACKEND_GRAPH_COMPILER_OP_BACKEND_H_
