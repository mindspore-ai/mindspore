/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_SESSION_GPU_SESSION_H
#define MINDSPORE_CCSRC_BACKEND_SESSION_GPU_SESSION_H

#include <vector>
#include <memory>
#include <algorithm>
#include <string>
#include <map>
#include "backend/common/session/session_basic.h"
#include "include/backend/kernel_graph.h"
#include "backend/common/session/session_factory.h"
using KernelGraph = mindspore::session::KernelGraph;

namespace mindspore {
namespace session {
namespace gpu {
class GPUSession : public SessionBasic {
 public:
  GPUSession() = default;
  ~GPUSession() override = default;
  void Init(uint32_t device_id) override;
  void SyncStream() const override;

 protected:
  void UnifyMindIR(const KernelGraphPtr &graph) override { SessionBasic::UnifyMindIR(graph); }
  GraphId CompileGraphImpl(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) override;
  GraphId CompileGraphImpl(NotNull<FuncGraphPtr> func_graph) override;
  void PreExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph, const std::vector<tensor::TensorPtr> &inputs,
                       VectorRef *const outputs) override;
  void PostExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph, const std::vector<tensor::TensorPtr> &inputs,
                        VectorRef *const outputs) override;
  void ExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph) override;
  KernelGraphPtr BuildOpImpl(const BackendOpRunInfoPtr &op_run_info, const GraphInfo &graph_info,
                             const std::vector<tensor::TensorPtr> &input_tensors,
                             const std::vector<int64_t> &tensors_mask) override;
  void RunOpImpl(const GraphInfo &graph_info, const BackendOpRunInfoPtr &op_run_info,
                 std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs,
                 const std::vector<int64_t> &tensors_mask) override;
  void RunOpImplOrigin(const GraphInfo &graph_info, const BackendOpRunInfoPtr &op_run_info,
                       std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs,
                       const std::vector<int64_t> &tensors_mask) override;
  std::string GetCommWorldGroup() override { return kNcclWorldGroup; }
  void LoadInputData(const std::shared_ptr<KernelGraph> &kernel_graph,
                     const std::vector<tensor::TensorPtr> &inputs_const) const override;
  void UpdateOutputTensors(const VectorRef *outputs,
                           const std::map<tensor::TensorPtr, session::KernelWithIndex> &tensor_to_node,
                           std::map<DeviceAddressPtr, DeviceAddressPtr> *new_to_old_device_address) override;

 private:
  void SelectKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const;

  void StartKernelRT() const;

  void Optimize(const std::shared_ptr<KernelGraph> &kernel_graph);

  void HardwareOptimize(const std::shared_ptr<KernelGraph> &kernel_graph);

  void RunOpOptimize(const std::shared_ptr<KernelGraph> &kernel_graph);

  void RunOpHardwareOptimize(const std::shared_ptr<KernelGraph> &kernel_graph);

  void GraphKernelOptimize(const std::shared_ptr<KernelGraph> &kernel_graph);

  void AssignStream(const std::shared_ptr<KernelGraph> &kernel_graph);

  void BuildKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const;

  void AllocateMemory(const KernelGraph *kernel_graph) const;

  void RunOpAllocateMemory(const std::vector<tensor::TensorPtr> &input_tensors, const KernelGraph *kernel_graph,
                           bool is_gradient_out) const;

  void RunOpClearMemory(const KernelGraph *kernel_graph) const;

  void RunOpGenKernelEvent(const KernelGraph *graph) const;

  void Execute(const std::shared_ptr<KernelGraph> &kernel_graph) const;

#ifdef ENABLE_DEBUGGER
  void Dump(const std::shared_ptr<KernelGraph> &kernel_graph) const;

  bool DumpDataEnabledIteration() const;
#endif

  GraphId CompileGraphImpl(const KernelGraphPtr &kernel_graph);
};
using GPUSessionPtr = std::shared_ptr<GPUSession>;
MS_REG_SESSION(kGPUDevice, GPUSession);
}  // namespace gpu
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_SESSION_GPU_SESSION_H
