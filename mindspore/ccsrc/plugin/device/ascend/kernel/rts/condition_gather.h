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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_CONDITION_GATHER_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_CONDITION_GATHER_H
#include "plugin/device/ascend/kernel/rts/rt_kernel.h"
#include <memory>
#include <vector>

namespace mindspore {
namespace kernel {
class ConditionGatherKernel : public RtKernel {
 public:
  ConditionGatherKernel() = default;
  ~ConditionGatherKernel() override;
  bool Init(const AnfNodePtr &anf_node) override;
  bool Launch(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &, void *) override;
  std::vector<KernelAttr> GetOpSupport() override { MS_LOG(EXCEPTION) << "This interface is not support in RtKernel."; }
};

MS_REG_RTKERNEL(conditiongather, ConditionGatherKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_CONDITION_GATHER_H
