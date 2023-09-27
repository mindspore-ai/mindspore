/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DYN_BROADCAST_GRAD_ARGS_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DYN_BROADCAST_GRAD_ARGS_KERNEL_H_

#include <vector>
#include <map>
#include <memory>
#include <algorithm>
#include <utility>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class DynamicBroadcastGradientArgsCpuKernelMod : public NativeCpuKernelMod,
                                                 public MatchKernelHelper<DynamicBroadcastGradientArgsCpuKernelMod> {
 public:
  DynamicBroadcastGradientArgsCpuKernelMod() : r0_size_(0), r1_size_(0) {}
  ~DynamicBroadcastGradientArgsCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  void SyncOutputShape() override {
    ShapeVector r0_shape{SizeToLong(r0_size_)};
    ShapeVector r1_shape{SizeToLong(r1_size_)};

    outputs_[0]->SetShapeVector(r0_shape);
    outputs_[1]->SetShapeVector(r1_shape);

    return;
  }

 protected:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &workspace,
                    const std::vector<kernel::KernelTensor *> &outputs);

  size_t r0_size_;
  size_t r1_size_;
  bool is_null_input0_{false};
  bool is_null_input1_{false};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DYN_BROADCAST_GRAD_ARGS_KERNEL_H_
