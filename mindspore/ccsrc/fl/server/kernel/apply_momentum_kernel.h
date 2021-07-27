/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_APPLY_MOMENTUM_KERNEL_H_
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_APPLY_MOMENTUM_KERNEL_H_

#include <vector>
#include <memory>
#include <utility>
#include "backend/kernel_compiler/cpu/apply_momentum_cpu_kernel.h"
#include "fl/server/kernel/optimizer_kernel.h"
#include "fl/server/kernel/optimizer_kernel_factory.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
using mindspore::kernel::ApplyMomentumCPUKernel;
template <typename T>
class ApplyMomentumKernel : public ApplyMomentumCPUKernel, public OptimizerKernel {
 public:
  ApplyMomentumKernel() = default;
  ~ApplyMomentumKernel() override = default;

  void InitKernel(const CNodePtr &cnode) override {
    MS_EXCEPTION_IF_NULL(cnode);
    ApplyMomentumCPUKernel::InitKernel(cnode);
    InitServerKernelInputOutputSize(cnode);
    GenerateReuseKernelNodeInfo();
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return ApplyMomentumCPUKernel::Launch(inputs, workspace, outputs);
  }

  void GenerateReuseKernelNodeInfo() override {
    MS_LOG(INFO) << "FedAvg reuse 'weight', 'accumulation', 'learning rate' and 'momentum' of the kernel node.";
    reuse_kernel_node_inputs_info_.insert(std::make_pair(kWeight, 0));
    reuse_kernel_node_inputs_info_.insert(std::make_pair(kAccumulation, 1));
    reuse_kernel_node_inputs_info_.insert(std::make_pair(kLearningRate, 2));
    reuse_kernel_node_inputs_info_.insert(std::make_pair(kMomentum, 4));
    return;
  }
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_APPLY_MOMENTUM_KERNEL_H_
