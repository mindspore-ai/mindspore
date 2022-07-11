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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAXUNPOOL3D_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAXUNPOOL3D_CPU_KERNEL_H_
#include <functional>
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class MaxUnpool3DCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  MaxUnpool3DCpuKernelMod() = default;
  ~MaxUnpool3DCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  };

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename DATA_T, typename INDICES_T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  using MaxUnpool3DFunc = std::function<bool(MaxUnpool3DCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                             const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, MaxUnpool3DFunc>> func_list_;
  MaxUnpool3DFunc kernel_func_;

  template <typename DATA_T>
  void OutPutInitKernel(DATA_T *rawOutput, size_t length);
  CNodeWeakPtr node_wpt_;
  ShapeVector input_shape_;
  ShapeVector indices_shape_;
  ShapeVector output_shape_;
  std::string data_format_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAXUNPOOL3D_CPU_KERNEL_H_
