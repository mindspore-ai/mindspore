/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CAST_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CAST_CPU_KERNEL_H_
#include <functional>
#include <memory>
#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class CastCPUKernel : public CPUKernel {
 public:
  CastCPUKernel() = default;
  ~CastCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  TypeId source_dtype{kTypeUnknown};
  TypeId target_dtype{kTypeUnknown};
};

MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool), CastCPUKernel);

MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeUInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeUInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeUInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeUInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool), CastCPUKernel);

MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool), CastCPUKernel);

MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool), CastCPUKernel);

MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool), CastCPUKernel);

MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool), CastCPUKernel);

MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool), CastCPUKernel);

MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool), CastCPUKernel);

MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool), CastCPUKernel);

MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeFloat16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeFloat32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeFloat64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeBool), CastCPUKernel);

MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeFloat16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeFloat32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeFloat64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeBool), CastCPUKernel);

MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeFloat16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeFloat32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeFloat64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt8), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt16), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt32), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64), CastCPUKernel);
MS_REG_CPU_KERNEL(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeBool), CastCPUKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CAST_CPU_KERNEL_H_
