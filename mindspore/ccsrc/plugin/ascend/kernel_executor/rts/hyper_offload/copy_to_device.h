/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_COPY_TO_DEVICE_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_COPY_TO_DEVICE_H
#include <memory>
#include <vector>

#include "plugin/ascend/kernel_executor/rts/rt_kernel.h"

namespace mindspore {
namespace kernel {
class CopyToDeviceKernel : public RtKernel {
 public:
  CopyToDeviceKernel() = default;
  ~CopyToDeviceKernel() override {}
  bool Init(const AnfNodePtr &anf_node) override;
  bool Launch(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &, void *) override;
  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override { return {kIndex0}; }
};

MS_REG_RTKERNEL(copytodevice, CopyToDeviceKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_COPY_TO_DEVICE_H
