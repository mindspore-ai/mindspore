/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_POOLING_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_POOLING_GRAD_CPU_KERNEL_H_

#include <string>
#include <vector>
#include <memory>
#include <utility>

#include "plugin/device/cpu/kernel/mkldnn/pooling_cpu_kernel.h"

namespace mindspore {
namespace kernel {
class PoolingGradCpuKernelMod : public PoolingCpuKernelMod {
 public:
  PoolingGradCpuKernelMod() = default;
  ~PoolingGradCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void InitFields(const CNodePtr &kernel_node);
  void ComputeMaxValueIndex(void *src, void *dst, unsigned char *work_array) const;

  dnnl::memory::desc src_desc_{};
  dnnl::memory::desc dst_desc_{};
  dnnl::memory::desc workspace_desc_{};
  dnnl::pooling_forward::primitive_desc forward_prim_desc_{};
  size_t grad_index_{0};
};

MS_REG_CPU_KERNEL(AvgPoolGrad,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32),
                  PoolingGradCpuKernelMod)

MS_REG_CPU_KERNEL(AvgPool3DGrad, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  PoolingGradCpuKernelMod)

MS_REG_CPU_KERNEL(MaxPoolGrad,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32),
                  PoolingGradCpuKernelMod)

MS_REG_CPU_KERNEL(MaxPool3DGrad,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32),
                  PoolingGradCpuKernelMod)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_POOLING_GRAD_CPU_KERNEL_H_
