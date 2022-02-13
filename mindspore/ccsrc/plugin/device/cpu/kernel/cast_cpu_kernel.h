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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CAST_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CAST_CPU_KERNEL_H_

#include <functional>
#include <memory>
#include <vector>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename S, typename T>
class CastCpuKernelMod : public NativeCpuKernelMod {
 public:
  CastCpuKernelMod() = default;
  ~CastCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  TypeId source_dtype_{kTypeUnknown};
  TypeId target_dtype_{kTypeUnknown};
};

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint8_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint8_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint8_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint8_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint8_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint8_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint8_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint8_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint8_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint8_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint8_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint8_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint16_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint16_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint16_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint16_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint16_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint16_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint16_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint16_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint16_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint16_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint16_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint16_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint32_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint32_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint32_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint32_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint32_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint32_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint32_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint32_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint32_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint32_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint32_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint32_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint64_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint64_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint64_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint64_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint64_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint64_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint64_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint64_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint64_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint64_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint64_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, uint64_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int8_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int8_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int8_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int8_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int8_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int8_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int8_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int8_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int8_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int8_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int8_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int8_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int16_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int16_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int16_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int16_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int16_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int16_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int16_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int16_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int16_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int16_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int16_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int16_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int32_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int32_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int32_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int32_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int32_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int32_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int32_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int32_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int32_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int32_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int32_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int32_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int64_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int64_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int64_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int64_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int64_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int64_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int64_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int64_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int64_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int64_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int64_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, int64_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float16, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float16, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float16, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float16, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float16, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float16, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float16, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float16, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float16, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float16, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float16, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float16, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, float, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, double, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, double, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, double, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, double, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, double, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, double, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, double, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, double, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, double, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, double, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, double, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, double, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, bool, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, bool, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, bool, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, bool, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, bool, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, bool, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, bool, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, bool, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, bool, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, bool, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, bool, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCpuKernelMod, bool, bool);

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CAST_CPU_KERNEL_H_
