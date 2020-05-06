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
#ifndef MINDSPORE_CCSRC_SESSION_GPU_SESSION_H
#define MINDSPORE_CCSRC_SESSION_GPU_SESSION_H

#include <vector>
#include <memory>
#include "session/session_basic.h"
#include "session/kernel_graph.h"
#include "session/session_factory.h"
using KernelGraph = mindspore::session::KernelGraph;

namespace mindspore {
namespace session {
namespace gpu {
class GPUSession : public SessionBasic {
 public:
  GPUSession() = default;
  ~GPUSession() override = default;

  void Init(uint32_t device_id) override {
    SessionBasic::Init(device_id);
    context_ = std::make_shared<Context>(kGPUDevice, device_id);
  }

  GraphId CompileGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) override;

  void RunGraph(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) override;
  void BuildOp(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
               const std::vector<tensor::TensorPtr> &input_tensors, const std::vector<bool> &tensors_mask) override;
  py::tuple RunOp(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
                  const std::vector<tensor::TensorPtr> &input_tensors) override;

 private:
  void SelectKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const;

  void StartKernelRT() const;

  void Optimize(const std::shared_ptr<KernelGraph> &kernel_graph);

  void AssignStream(const std::shared_ptr<KernelGraph> &kernel_graph);

  void BuildKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const;

  void AllocateMemory(KernelGraph *kernel_graph) const;

  void RunOpAllocateMemory(const std::vector<tensor::TensorPtr> &input_tensors, KernelGraph *kernel_graph) const;

  void Execute(const std::shared_ptr<KernelGraph> &kernel_graph) const;
};
using GPUSessionPtr = std::shared_ptr<GPUSession>;
MS_REG_SESSION(kGPUDevice, GPUSession);
}  // namespace gpu
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_SESSION_GPU_SESSION_H
