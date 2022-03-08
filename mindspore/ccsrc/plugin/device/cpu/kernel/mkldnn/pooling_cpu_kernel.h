/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_POOLING_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_POOLING_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <unordered_map>

#include "plugin/device/cpu/kernel/mkldnn/mkl_cpu_kernel.h"

namespace mindspore {
namespace kernel {
class PoolingCpuKernelMod : public MKLCpuKernelMod {
 public:
  PoolingCpuKernelMod() = default;
  ~PoolingCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  void EliminateInvalidPadding(float *output);
  void ReComputeDivisor(float *output);

  static std::unordered_map<void *, std::vector<unsigned char>> pooling_max_workspace_;
  dnnl::algorithm algorithm_{dnnl::algorithm::pooling_max};
  bool ceil_mode_{false};
  float divisor_override_{0.0};
  std::vector<size_t> dst_shape_;
  std::vector<float> padding_invalid_;
  std::vector<float> kernel_;

 private:
  void InitFields(const CNodePtr &kernel_node);
  void InitInputOutputSize(const CNodePtr &kernel_node) override;

  size_t workspace_size_{0};
};

MS_REG_CPU_KERNEL(MaxPool, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  PoolingCpuKernelMod)
MS_REG_CPU_KERNEL(MaxPool3D, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  PoolingCpuKernelMod)
MS_REG_CPU_KERNEL(AvgPool, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  PoolingCpuKernelMod)
MS_REG_CPU_KERNEL(AvgPool3D, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  PoolingCpuKernelMod)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_POOLING_CPU_KERNEL_H_
