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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SCATTER_ND_MIN_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SCATTER_ND_MIN_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <unordered_map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
#define ADD_KERNEL(t1, t2, t3, t4) \
  KernelAttr()                     \
    .AddInputAttr(kNumberType##t1) \
    .AddInputAttr(kNumberType##t2) \
    .AddInputAttr(kNumberType##t3) \
    .AddOutputAttr(kNumberType##t4)

template <typename T, typename S>
struct ComputeParams {
  T *x_{nullptr};
  S *indices_{nullptr};
  T *updates_{nullptr};
  int unit_size_{0};
  int indices_unit_rank_{0};
  std::vector<int> *out_strides_{nullptr};
  size_t x_mem_size_{0};
  int indices_mem_size_{0};
};

class ScatterMinCPUKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  ScatterMinCPUKernelMod() = default;
  ~ScatterMinCPUKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  virtual void *ScatterMinRealData(const std::vector<AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &outputs) = 0;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  template <typename T>
  bool DoComputeWithIndicesType(const std::vector<kernel::AddressPtr> &inputs,
                                const std::vector<kernel::AddressPtr> &outputs, TypeId indices_dtype_);

  TypeId dtype_{kTypeUnknown};
  TypeId indices_dtype_{kTypeUnknown};
  int unit_size_{0};
  size_t num_units_{0};
  int indices_unit_rank_{0};
  std::vector<int> out_strides_;
};

class ScatterNdMinCpuKernelMod : public ScatterMinCPUKernelMod {
 protected:
  void *ScatterMinRealData(const std::vector<AddressPtr> &inputs,
                           const std::vector<kernel::AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {
      ADD_KERNEL(Int8, Int32, Int8, Int8),          ADD_KERNEL(Int16, Int32, Int16, Int16),
      ADD_KERNEL(Int32, Int32, Int32, Int32),       ADD_KERNEL(Int64, Int32, Int64, Int64),
      ADD_KERNEL(UInt8, Int32, UInt8, UInt8),       ADD_KERNEL(UInt16, Int32, UInt16, UInt16),
      ADD_KERNEL(UInt32, Int32, UInt32, UInt32),    ADD_KERNEL(UInt64, Int32, UInt64, UInt64),
      ADD_KERNEL(Float16, Int32, Float16, Float16), ADD_KERNEL(Float32, Int32, Float32, Float32),
      ADD_KERNEL(Float64, Int32, Float64, Float64), ADD_KERNEL(Int8, Int64, Int8, Int8),
      ADD_KERNEL(Int16, Int64, Int16, Int16),       ADD_KERNEL(Int32, Int64, Int32, Int32),
      ADD_KERNEL(Int64, Int64, Int64, Int64),       ADD_KERNEL(UInt8, Int64, UInt8, UInt8),
      ADD_KERNEL(UInt16, Int64, UInt16, UInt16),    ADD_KERNEL(UInt32, Int64, UInt32, UInt32),
      ADD_KERNEL(UInt64, Int64, UInt64, UInt64),    ADD_KERNEL(Float16, Int64, Float16, Float16),
      ADD_KERNEL(Float32, Int64, Float32, Float32), ADD_KERNEL(Float64, Int64, Float64, Float64)};
    return support_list;
  }
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SCATTER_ND_MIN_CPU_KERNEL_H_
