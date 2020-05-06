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
#ifndef MINDSPORE_CCSRC_SESSION_CPU_SESSION_H
#define MINDSPORE_CCSRC_SESSION_CPU_SESSION_H
#include <string>
#include <memory>
#include <vector>
#include "session/session_basic.h"
#include "session/kernel_graph.h"
#include "device/cpu/cpu_kernel_runtime.h"
#include "session/session_factory.h"
namespace mindspore {
namespace session {
class CPUSession : public SessionBasic {
 public:
  CPUSession() = default;
  ~CPUSession() override = default;
  void Init(uint32_t device_id) override {
    SessionBasic::Init(device_id);
    context_ = std::make_shared<Context>(kCPUDevice, device_id);
  }
  GraphId CompileGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) override;
  void RunGraph(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) override;

 private:
  void SetKernelInfo(const KernelGraph *kernel_graph);
  void BuildKernel(const KernelGraph *kernel_graph);
  device::cpu::CPUKernelRuntime runtime_;
};
MS_REG_SESSION(kCPUDevice, CPUSession);
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_SESSION_CPU_SESSION_H
