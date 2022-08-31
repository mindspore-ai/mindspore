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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MEDIAN_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MEDIAN_GRAD_CPU_KERNEL_H_

#include <algorithm>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class MedianGradCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  MedianGradCpuKernelMod() = default;
  ~MedianGradCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  TypeId input0_type_;
  TypeId input1_type_;
  std::vector<int64_t> input0_shape_;
  std::vector<int64_t> input1_shape_;
  std::vector<int64_t> input2_shape_;
  bool global_median_;
  int axis_{0};
  size_t input0_dim_;
  size_t input1_dim_;
  size_t input2_dim_;
  size_t input0_num_elements_;
  size_t input1_num_elements_;
  template <typename T1, typename T2>
  bool GlobalMedianGradCompute(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) const;
  template <typename T1, typename T2>
  bool MedianGradCompute(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MEDIAN_GRAD_CPU_KERNEL_H_
