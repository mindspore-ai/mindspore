/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
template <typename S, typename T>
class CastCPUKernel : public CPUKernel {
 public:
  CastCPUKernel() = default;
  ~CastCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  TypeId source_dtype_{kTypeUnknown};
  TypeId target_dtype_{kTypeUnknown};
};

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint8_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint8_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint8_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint8_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint8_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint8_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint8_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint8_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint8_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint8_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint8_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint8_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint16_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint16_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint16_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint16_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint16_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint16_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint16_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint16_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint16_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint16_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint16_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint16_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint32_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint32_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint32_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint32_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint32_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint32_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint32_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint32_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint32_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint32_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint32_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint32_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint64_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint64_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint64_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint64_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint64_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint64_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint64_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint64_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint64_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint64_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint64_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, uint64_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int8_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int8_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int8_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int8_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int8_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int8_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int8_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int8_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int8_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int8_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int8_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int8_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int16_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int16_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int16_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int16_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int16_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int16_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int16_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int16_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int16_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int16_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int16_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int16_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int32_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int32_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int32_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int32_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int32_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int32_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int32_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int32_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int32_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int32_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int32_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int32_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int64_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int64_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int64_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int64_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int64_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int64_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int64_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int64_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int64_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int64_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int64_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, int64_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float16, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float16, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float16, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float16, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float16, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float16, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float16, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float16, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float16, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float16, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float16, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float16, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, float, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, double, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, double, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, double, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, double, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, double, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, double, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, double, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, double, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, double, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, double, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, double, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, double, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, bool, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, bool, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, bool, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, bool, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, bool, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, bool, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, bool, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, bool, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, bool, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, bool, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, bool, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr(), CastCPUKernel, bool, bool);

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CAST_CPU_KERNEL_H_
