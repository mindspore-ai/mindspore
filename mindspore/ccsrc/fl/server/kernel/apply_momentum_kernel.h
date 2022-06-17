/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/kernel/apply_momentum_cpu_kernel.h"
#include "fl/server/kernel/optimizer_kernel.h"
#include "fl/server/kernel/optimizer_kernel_factory.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
using mindspore::kernel::ApplyMomentumCpuKernelMod;
size_t kWeightIndex = 0;
size_t kAccumulationIndex = 1;
size_t kLearningRateIndex = 2;
size_t kMomentumIndex = 4;
template <typename T>
class ApplyMomentumKernel : public ApplyMomentumCpuKernelMod, public OptimizerKernelMod {
 public:
  ApplyMomentumKernel() = default;
  ~ApplyMomentumKernel() override = default;

  void InitKernel(const CNodePtr &cnode) override {
    MS_EXCEPTION_IF_NULL(cnode);
    ApplyMomentumCpuKernelMod::InitKernel(cnode);
    InitServerKernelInputOutputSize(cnode);
    GenerateReuseKernelNodeInfo();
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return ApplyMomentumCpuKernelMod::Launch(inputs, workspace, outputs);
  }

  void GenerateReuseKernelNodeInfo() override {
    MS_LOG(INFO) << "FedAvg reuse 'weight', 'accumulation', 'learning rate' and 'momentum' of the kernel node.";
    (void)reuse_kernel_node_inputs_info_.insert(std::make_pair(kWeight, kWeightIndex));
    (void)reuse_kernel_node_inputs_info_.insert(std::make_pair(kAccumulation, kAccumulationIndex));
    (void)reuse_kernel_node_inputs_info_.insert(std::make_pair(kLearningRate, kLearningRateIndex));
    (void)reuse_kernel_node_inputs_info_.insert(std::make_pair(kMomentum, kMomentumIndex));
    return;
  }
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_APPLY_MOMENTUM_KERNEL_H_
