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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_INSTANCE_NORM_V2_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_INSTANCE_NORM_V2_GRAD_CPU_KERNEL_H_

#include <set>
#include <vector>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/eigen/eigen_common_utils.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class InstanceNormV2GradCpuKernelMod : public NativeCpuKernelMod {
 public:
  InstanceNormV2GradCpuKernelMod() = default;
  ~InstanceNormV2GradCpuKernelMod() override = default;
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

  TypeId in_type_{kTypeUnknown};
  bool is_training_ = true;
  float epsilon_ = 0.00001;
  std::vector<int64_t> dy_shape_4d_;
  std::vector<int64_t> batch_channels_2d_;
  bool dy_is_4d_ = true;
  int64_t instance_num = 0;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_INSTANCE_NORM_V2_GRAD_CPU_KERNEL_H_
