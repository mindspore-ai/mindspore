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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_IDENTITYN_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_IDENTITYN_CPU_KERNEL_H_

#include <vector>
#include <set>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "kernel/common_utils.h"
#include "plugin/factory/ms_factory.h"
#include "mindspore/core/ops/identity_n.h"

namespace mindspore {
namespace kernel {
class IdentityNCpuKernelMod : public NativeCpuKernelMod {
 public:
  IdentityNCpuKernelMod() = default;
  ~IdentityNCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  const std::set<TypeId> support_types_ = {
    kNumberTypeBool,  kNumberTypeUInt8,  kNumberTypeInt8,  kNumberTypeUInt16,  kNumberTypeInt16,   kNumberTypeUInt32,
    kNumberTypeInt32, kNumberTypeUInt64, kNumberTypeInt64, kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64};
  bool CheckType(TypeId idx_type, size_t idx);
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_IDENTITYN_CPU_KERNEL_H_
