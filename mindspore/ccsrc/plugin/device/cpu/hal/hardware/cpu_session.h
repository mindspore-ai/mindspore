/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_HAL_HARDWARE_CPU_SESSION_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_HAL_HARDWARE_CPU_SESSION_H_
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "backend/common/session/session_basic.h"
#include "include/backend/kernel_graph.h"
#include "plugin/device/cpu/hal/device/cpu_kernel_runtime.h"
#include "backend/common/session/session_factory.h"
namespace mindspore {
namespace session {
class CPUSession : public SessionBasic {
 public:
  CPUSession() = default;
  ~CPUSession() override = default;
  void Init(uint32_t device_id) override;

 protected:
  void UnifyMindIR(const KernelGraphPtr &graph) override { SessionBasic::UnifyMindIR(graph); }
  void CreateOutputTensors(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &input_tensors, VectorRef *,
                           std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node,
                           KernelMapTensor *node_to_tensor) override;
  GraphId CompileGraphImpl(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) override;
  void PreExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph, const std::vector<tensor::TensorPtr> &inputs,
                       VectorRef *const outputs) override;
  void PostExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph, const std::vector<tensor::TensorPtr> &inputs,
                        VectorRef *const outputs) override;
  void ExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph) override;
  ParameterPtr CreateNewParameterFromParameter(const AnfNodePtr &anf, KernelGraph *graph) override;
  void GraphKernelOptimize(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void Optimize(const std::shared_ptr<KernelGraph> &kernel_graph);
  KernelGraphPtr BuildOpImpl(const BackendOpRunInfoPtr &op_run_info, const GraphInfo &graph_info,
                             const std::vector<tensor::TensorPtr> &input_tensors,
                             const std::vector<int64_t> &tensors_mask) override;
  void RunOpImpl(const GraphInfo &graph_info, const BackendOpRunInfoPtr &op_run_info,
                 std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs,
                 const std::vector<int64_t> &tensors_mask) override;
  void RunOpImplOrigin(const GraphInfo &graph_info, const BackendOpRunInfoPtr &op_run_info,
                       std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs,
                       const std::vector<int64_t> &tensors_mask) override;
  void LoadInputData(const std::shared_ptr<KernelGraph> &kernel_graph,
                     const std::vector<tensor::TensorPtr> &inputs_const) const override;

 private:
  void Reorder(std::vector<CNodePtr> *node_list) const;
  void SetKernelInfo(const KernelGraph *kernel_graph) const;
  void BuildKernel(const KernelGraph *kernel_graph) const;
  void SetOutputFlags(const VectorRef &base_ref);
  void UpdateDynamicOutputShape(const std::map<tensor::TensorPtr, KernelWithIndex> &tensor_to_node) const;
  device::cpu::CPUKernelRuntime runtime_;
};
MS_REG_SESSION(kCPUDevice, CPUSession);
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_HAL_HARDWARE_CPU_SESSION_H_
