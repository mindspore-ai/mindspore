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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MINIMUM_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MINIMUM_CPU_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class MinimumCPUKernel : public CPUKernel {
 public:
  MinimumCPUKernel() = default;
  ~MinimumCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void CheckParam(const CNodePtr &kernel_node);

  bool IsBroadcast();

  size_t Index(const size_t &index, const size_t &dim);

  void InitTensorBroadcastShape();

  void InitInputTensorAndScalar(size_t max_input_shape_size);

  void InitInputTensors(TypeId input_x_dtype, TypeId input_y_dtype);

  // Broadcast Arithmetic
  void BroadcastArithKernel(const size_t l0, const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                            const size_t l5, const size_t l6, const size_t r0, const size_t r1, const size_t r2,
                            const size_t r3, const size_t r4, const size_t r5, const size_t r6, const size_t d0,
                            const size_t d1, const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                            const size_t d6, const T *input_x, const T *input_y, T *output);

  T MinimumFunc(const T &lhs, const T &rhs) { return lhs < rhs ? lhs : rhs; }

  void BroadcastArithOneScalarOneTensor(const T *input_x, const T *input_y, T *output);

  void BroadcastArithTensors(const T *input_x, const T *input_y, T *output);

  void BroadcastArith(const T *input_x, const T *input_y, T *output);

 private:
  bool need_broadcast_{false};
  size_t input_x_num_{1};
  size_t input_y_num_{1};
  size_t output_num_{1};
  std::vector<size_t> input_x_shape_;
  std::vector<size_t> input_y_shape_;
  std::vector<size_t> output_shape_;
  std::vector<size_t> broadcast_input_x_shape_;
  std::vector<size_t> broadcast_input_y_shape_;
  std::vector<size_t> broadcast_output_shape_;
  const size_t max_dims{7};
};

MS_REG_CPU_KERNEL_T(
  Minimum, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  MinimumCPUKernel, int32_t);

MS_REG_CPU_KERNEL_T(
  Minimum,
  KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
  MinimumCPUKernel, uint32_t);

MS_REG_CPU_KERNEL_T(
  Minimum,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  MinimumCPUKernel, float);

MS_REG_CPU_KERNEL_T(
  Minimum, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  MinimumCPUKernel, int64_t);

MS_REG_CPU_KERNEL_T(
  Minimum,
  KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
  MinimumCPUKernel, uint64_t);

MS_REG_CPU_KERNEL_T(
  Minimum,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  MinimumCPUKernel, double);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UPDATE_CACHE_CPU_KERNEL_H_
