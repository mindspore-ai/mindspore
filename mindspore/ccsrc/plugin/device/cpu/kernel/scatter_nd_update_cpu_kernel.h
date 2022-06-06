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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SCATTER_ND_UPDATE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SCATTER_ND_UPDATE_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <unordered_map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
struct ComputeParams {
  T *x_{nullptr};
  S *indices_{nullptr};
  T *updates_{nullptr};
  size_t unit_size_{0};
  size_t indices_unit_rank_{0};
  std::vector<size_t> *out_strides_{nullptr};
  size_t x_mem_size_{0};
};

#define ADD_KERNEL(t1, t2)         \
  KernelAttr()                     \
    .AddInputAttr(kNumberType##t1) \
    .AddInputAttr(kNumberType##t2) \
    .AddInputAttr(kNumberType##t1) \
    .AddOutputAttr(kNumberType##t1)

class ScatterUpdateCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  ScatterUpdateCpuKernelMod() = default;
  ~ScatterUpdateCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  virtual void *ScatterUpdateRealData(const std::vector<AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &outputs) = 0;

 private:
  template <typename T, typename S>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  void LaunchTypeChoose(const std::vector<AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

  TypeId dtype_value{kTypeUnknown};
  TypeId dtype_shape{kTypeUnknown};
  int unit_size_{0};
  size_t num_units_{0};
  size_t indices_unit_rank_{0};
  std::vector<size_t> out_strides_;
};

class ScatterNdUpdateCpuKernelMod : public ScatterUpdateCpuKernelMod {
 public:
  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {
      ADD_KERNEL(Float32, Int32),    ADD_KERNEL(Float16, Int32),   ADD_KERNEL(Float64, Int32),
      ADD_KERNEL(Int8, Int32),       ADD_KERNEL(Int16, Int32),     ADD_KERNEL(Int32, Int32),
      ADD_KERNEL(Int64, Int32),      ADD_KERNEL(UInt8, Int32),     ADD_KERNEL(UInt16, Int32),
      ADD_KERNEL(UInt32, Int32),     ADD_KERNEL(UInt64, Int32),    ADD_KERNEL(Complex64, Int32),
      ADD_KERNEL(Complex128, Int32), ADD_KERNEL(Float32, Int64),   ADD_KERNEL(Float16, Int64),
      ADD_KERNEL(Float64, Int64),    ADD_KERNEL(Int8, Int64),      ADD_KERNEL(Int16, Int64),
      ADD_KERNEL(Int32, Int64),      ADD_KERNEL(Int64, Int64),     ADD_KERNEL(UInt8, Int64),
      ADD_KERNEL(UInt16, Int64),     ADD_KERNEL(UInt32, Int64),    ADD_KERNEL(UInt64, Int64),
      ADD_KERNEL(Complex64, Int64),  ADD_KERNEL(Complex128, Int64)};
    return support_list;
  }

 protected:
  void *ScatterUpdateRealData(const std::vector<AddressPtr> &inputs,
                              const std::vector<kernel::AddressPtr> &outputs) override;
};

class TensorScatterUpdateCpuKernelMod : public ScatterUpdateCpuKernelMod {
 public:
  std::vector<KernelAttr> GetOpSupport() override {
    static const std::vector<KernelAttr> support_list = {KernelAttr()
                                                           .AddInputAttr(kNumberTypeFloat32)
                                                           .AddInputAttr(kNumberTypeInt32)
                                                           .AddInputAttr(kNumberTypeFloat32)
                                                           .AddOutputAttr(kNumberTypeFloat32),
                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeFloat64)
                                                           .AddInputAttr(kNumberTypeInt32)
                                                           .AddInputAttr(kNumberTypeFloat64)
                                                           .AddOutputAttr(kNumberTypeFloat64),

                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeInt32)
                                                           .AddInputAttr(kNumberTypeInt32)
                                                           .AddInputAttr(kNumberTypeInt32)
                                                           .AddOutputAttr(kNumberTypeInt32),

                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeInt32)
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddOutputAttr(kNumberTypeInt64),

                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeBool)
                                                           .AddInputAttr(kNumberTypeInt32)
                                                           .AddInputAttr(kNumberTypeBool)
                                                           .AddOutputAttr(kNumberTypeBool),

                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeBool)
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeBool)
                                                           .AddOutputAttr(kNumberTypeBool)};
    return support_list;
  }

 protected:
  void *ScatterUpdateRealData(const std::vector<AddressPtr> &inputs,
                              const std::vector<kernel::AddressPtr> &outputs) override;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SCATTER_ND_UPDATE_CPU_KERNEL_H_
