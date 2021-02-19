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
template <typename S, typename T>
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

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat16), CastCPUKernel,
                      bool, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat32), CastCPUKernel,
                      bool, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat64), CastCPUKernel,
                      bool, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt8), CastCPUKernel,
                      bool, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt16), CastCPUKernel,
                      bool, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt32), CastCPUKernel,
                      bool, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt64), CastCPUKernel,
                      bool, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt8), CastCPUKernel,
                      bool, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt16), CastCPUKernel,
                      bool, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt32), CastCPUKernel,
                      bool, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt64), CastCPUKernel,
                      bool, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool), CastCPUKernel,
                      bool, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      CastCPUKernel, float16, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat32),
                      CastCPUKernel, float16, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat64),
                      CastCPUKernel, float16, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt8), CastCPUKernel,
                      float16, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt16),
                      CastCPUKernel, float16, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
                      CastCPUKernel, float16, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
                      CastCPUKernel, float16, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeUInt8),
                      CastCPUKernel, float16, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeUInt16),
                      CastCPUKernel, float16, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeUInt32),
                      CastCPUKernel, float16, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeUInt64),
                      CastCPUKernel, float16, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool), CastCPUKernel,
                      float16, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat16),
                      CastCPUKernel, float, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      CastCPUKernel, float, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat64),
                      CastCPUKernel, float, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt8), CastCPUKernel,
                      float, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt16),
                      CastCPUKernel, float, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
                      CastCPUKernel, float, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
                      CastCPUKernel, float, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt8),
                      CastCPUKernel, float, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt16),
                      CastCPUKernel, float, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt32),
                      CastCPUKernel, float, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt64),
                      CastCPUKernel, float, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool), CastCPUKernel,
                      float, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat16),
                      CastCPUKernel, double, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat32),
                      CastCPUKernel, double, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      CastCPUKernel, double, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt8), CastCPUKernel,
                      double, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt16),
                      CastCPUKernel, double, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
                      CastCPUKernel, double, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
                      CastCPUKernel, double, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt8),
                      CastCPUKernel, double, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt16),
                      CastCPUKernel, double, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt32),
                      CastCPUKernel, double, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt64),
                      CastCPUKernel, double, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool), CastCPUKernel,
                      double, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat16), CastCPUKernel,
                      int8_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat32), CastCPUKernel,
                      int8_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat64), CastCPUKernel,
                      int8_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8), CastCPUKernel,
                      int8_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt16), CastCPUKernel,
                      int8_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32), CastCPUKernel,
                      int8_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64), CastCPUKernel,
                      int8_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt8), CastCPUKernel,
                      int8_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt16), CastCPUKernel,
                      int8_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt32), CastCPUKernel,
                      int8_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt64), CastCPUKernel,
                      int8_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool), CastCPUKernel,
                      int8_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat16),
                      CastCPUKernel, int16_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat32),
                      CastCPUKernel, int16_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat64),
                      CastCPUKernel, int16_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt8), CastCPUKernel,
                      int16_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16), CastCPUKernel,
                      int16_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32), CastCPUKernel,
                      int16_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64), CastCPUKernel,
                      int16_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt8), CastCPUKernel,
                      int16_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt16), CastCPUKernel,
                      int16_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt32), CastCPUKernel,
                      int16_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt64), CastCPUKernel,
                      int16_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool), CastCPUKernel,
                      int16_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
                      CastCPUKernel, int32_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
                      CastCPUKernel, int32_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
                      CastCPUKernel, int32_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8), CastCPUKernel,
                      int32_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16), CastCPUKernel,
                      int32_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), CastCPUKernel,
                      int32_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64), CastCPUKernel,
                      int32_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8), CastCPUKernel,
                      int32_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16), CastCPUKernel,
                      int32_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32), CastCPUKernel,
                      int32_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64), CastCPUKernel,
                      int32_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool), CastCPUKernel,
                      int32_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
                      CastCPUKernel, int64_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
                      CastCPUKernel, int64_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
                      CastCPUKernel, int64_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8), CastCPUKernel,
                      int64_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16), CastCPUKernel,
                      int64_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32), CastCPUKernel,
                      int64_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), CastCPUKernel,
                      int64_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8), CastCPUKernel,
                      int64_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16), CastCPUKernel,
                      int64_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32), CastCPUKernel,
                      int64_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64), CastCPUKernel,
                      int64_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool), CastCPUKernel,
                      int64_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat16),
                      CastCPUKernel, uint8_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat32),
                      CastCPUKernel, uint8_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat64),
                      CastCPUKernel, uint8_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt8), CastCPUKernel,
                      uint8_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt16), CastCPUKernel,
                      uint8_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32), CastCPUKernel,
                      uint8_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64), CastCPUKernel,
                      uint8_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8), CastCPUKernel,
                      uint8_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt16), CastCPUKernel,
                      uint8_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt32), CastCPUKernel,
                      uint8_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt64), CastCPUKernel,
                      uint8_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool), CastCPUKernel,
                      uint8_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeFloat16),
                      CastCPUKernel, uint16_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeFloat32),
                      CastCPUKernel, uint16_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeFloat64),
                      CastCPUKernel, uint16_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt8), CastCPUKernel,
                      uint16_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt16), CastCPUKernel,
                      uint16_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32), CastCPUKernel,
                      uint16_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64), CastCPUKernel,
                      uint16_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt8), CastCPUKernel,
                      uint16_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                      CastCPUKernel, uint16_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt32),
                      CastCPUKernel, uint16_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt64),
                      CastCPUKernel, uint16_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeBool), CastCPUKernel,
                      uint16_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeFloat16),
                      CastCPUKernel, uint32_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeFloat32),
                      CastCPUKernel, uint32_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeFloat64),
                      CastCPUKernel, uint32_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt8), CastCPUKernel,
                      uint32_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt16), CastCPUKernel,
                      uint32_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32), CastCPUKernel,
                      uint32_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64), CastCPUKernel,
                      uint32_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt8), CastCPUKernel,
                      uint32_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt16),
                      CastCPUKernel, uint32_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      CastCPUKernel, uint32_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt64),
                      CastCPUKernel, uint32_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeBool), CastCPUKernel,
                      uint32_t, bool);

MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeFloat16),
                      CastCPUKernel, uint64_t, float16);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeFloat32),
                      CastCPUKernel, uint64_t, float);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeFloat64),
                      CastCPUKernel, uint64_t, double);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt8), CastCPUKernel,
                      uint64_t, int8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt16), CastCPUKernel,
                      uint64_t, int16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32), CastCPUKernel,
                      uint64_t, int32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64), CastCPUKernel,
                      uint64_t, int64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt8), CastCPUKernel,
                      uint64_t, uint8_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt16),
                      CastCPUKernel, uint64_t, uint16_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt32),
                      CastCPUKernel, uint64_t, uint32_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                      CastCPUKernel, uint64_t, uint64_t);
MS_REG_CPU_KERNEL_T_S(Cast, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeBool), CastCPUKernel,
                      uint64_t, bool);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CAST_CPU_KERNEL_H_
