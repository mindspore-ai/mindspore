/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_ISNAN_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_ISNAN_CPU_KERNEL_H_

#include <vector>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class IsNanCpuKernelMod : public NativeCpuKernelMod {
 public:
  IsNanCpuKernelMod() = default;
  ~IsNanCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override {
    static const std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
      KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool),
      KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool),
      KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
      KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
      KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
      KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
      KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
      KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool),
      KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeBool),
      KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeBool),
      KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeBool)};
    return support_list;
  }

 private:
  template <typename T>
  void LaunchKernelFloat(const std::vector<AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs) const;

  void LaunchKernelOther(const std::vector<AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs) const;

  void LaunchKernelFloat16(const std::vector<AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs) const;

  std::map<TypeId, size_t> dtype_map_ = {{kNumberTypeBool, sizeof(bool)},       {kNumberTypeInt8, sizeof(int8_t)},
                                         {kNumberTypeInt16, sizeof(int16_t)},   {kNumberTypeInt32, sizeof(int32_t)},
                                         {kNumberTypeInt64, sizeof(int64_t)},   {kNumberTypeFloat16, sizeof(float16)},
                                         {kNumberTypeFloat32, sizeof(float)},   {kNumberTypeFloat64, sizeof(double)},
                                         {kNumberTypeUInt8, sizeof(uint8_t)},   {kNumberTypeUInt16, sizeof(uint16_t)},
                                         {kNumberTypeUInt32, sizeof(uint32_t)}, {kNumberTypeUInt64, sizeof(uint64_t)}};
  TypeId input_dtype_{kTypeUnknown};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_ISNAN_CPU_KERNEL_H_
