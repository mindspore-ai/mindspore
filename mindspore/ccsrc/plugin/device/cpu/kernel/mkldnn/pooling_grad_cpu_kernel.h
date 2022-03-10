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
#include <map>

#include "plugin/device/cpu/kernel/mkldnn/pooling_cpu_kernel.h"

namespace mindspore {
namespace kernel {
constexpr auto kAvgPoolGrad = "AvgPoolGrad";
constexpr auto kAvgPool3DGrad = "AvgPool3DGrad";
constexpr auto kMaxPoolGrad = "MaxPoolGrad";
constexpr auto kMaxPool3DGrad = "MaxPool3DGrad";
constexpr auto kUnknown = "Unknown";
class PoolingGradCpuKernelMod : public PoolingCpuKernelMod {
 public:
  PoolingGradCpuKernelMod() = default;
  explicit PoolingGradCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~PoolingGradCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override {
    static std::map<std::string, std::vector<KernelAttr>> support_list = {
      {kAvgPoolGrad,
       {{KernelAttr()
           .AddInputAttr(kNumberTypeFloat32)
           .AddInputAttr(kNumberTypeFloat32)
           .AddInputAttr(kNumberTypeFloat32)
           .AddOutputAttr(kNumberTypeFloat32)}}},
      {kAvgPool3DGrad, {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}}},
      {kMaxPoolGrad,
       {{KernelAttr()
           .AddInputAttr(kNumberTypeFloat32)
           .AddInputAttr(kNumberTypeFloat32)
           .AddInputAttr(kNumberTypeFloat32)
           .AddOutputAttr(kNumberTypeFloat32)}}},
      {kMaxPool3DGrad,
       {{KernelAttr()
           .AddInputAttr(kNumberTypeFloat32)
           .AddInputAttr(kNumberTypeFloat32)
           .AddInputAttr(kNumberTypeFloat32)
           .AddOutputAttr(kNumberTypeFloat32)}}}};

    auto iter = support_list.find(kernel_type_);
    if (iter == support_list.end()) {
      MS_LOG(EXCEPTION) << "PoolingGrad does not support kernel type: " << kernel_type_;
    }
    return iter->second;
  }

 private:
  void InitFields(const CNodePtr &kernel_node);
  void InitInputOutputSize(const CNodePtr &kernel_node) override;
  void ComputeMaxValueIndex(void *src, void *dst, void *work_array) const;

  dnnl::memory::desc src_desc_{};
  dnnl::memory::desc dst_desc_{};
  dnnl::memory::desc workspace_desc_{};
  dnnl::pooling_forward::primitive_desc forward_prim_desc_{};
  size_t grad_index_{0};
  std::string kernel_type_{kUnknown};
  size_t workspace_size_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_POOLING_GRAD_CPU_KERNEL_H_
