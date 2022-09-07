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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_ISFINITE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_ISFINITE_CPU_KERNEL_H_

#include <vector>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class IsFiniteCpuKernelMod : public NativeCpuKernelMod {
 public:
  IsFiniteCpuKernelMod() = default;
  ~IsFiniteCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

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

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_ISFINITE_CPU_KERNEL_H_
