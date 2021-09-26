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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MIRROR_PAD_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MIRROR_PAD_GRAD_CPU_KERNEL_H_

#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
namespace mindspore {
namespace kernel {
class MirrorPadGradCPUKernel : public CPUKernel {
 public:
  MirrorPadGradCPUKernel() = default;
  ~MirrorPadGradCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  void InitInputOutputSize(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  template <typename T>
  void InitWorkspaceSize();

  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs) const;

  template <typename T>
  void MirrorPadGrad_Width_Height(const size_t size, const T *interim_dy, const int64_t dx_height,
                                  const int64_t dx_width, const int64_t dy_height, const int64_t dy_width,
                                  const int64_t padd_dim, const int64_t *paddings_arg, int64_t mode, T *dx) const;

  template <typename T>
  void MirrorPadGradBatchChannel(const size_t size, T *dy, T *interim_dy, const int64_t dx_batches,
                                 const int64_t dx_channels, const int64_t dy_height, const int64_t dy_width,
                                 const int64_t padd_dim, const int64_t *paddings_arg, int64_t mode) const;

  TypeId dtype_{kTypeUnknown};
  size_t tensor_size_{1};
  size_t shape_size_{1};
  size_t output_size_{1};
  size_t workspace_size_{1};
  int mode_{0};
  int64_t num_paddings_{0};
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
};

MS_REG_CPU_KERNEL(
  MirrorPadGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
  MirrorPadGradCPUKernel);

MS_REG_CPU_KERNEL(
  MirrorPadGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
  MirrorPadGradCPUKernel);

MS_REG_CPU_KERNEL(
  MirrorPadGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
  MirrorPadGradCPUKernel);

MS_REG_CPU_KERNEL(
  MirrorPadGrad,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
  MirrorPadGradCPUKernel);

MS_REG_CPU_KERNEL(
  MirrorPadGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  MirrorPadGradCPUKernel);

MS_REG_CPU_KERNEL(
  MirrorPadGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  MirrorPadGradCPUKernel);

MS_REG_CPU_KERNEL(
  MirrorPadGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  MirrorPadGradCPUKernel);

MS_REG_CPU_KERNEL(
  MirrorPadGrad,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  MirrorPadGradCPUKernel);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MIRROR_PAD_CPU_KERNEL_H_
