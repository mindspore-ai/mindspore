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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CONJUGATE_TRANSPOSE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CONJUGATE_TRANSPOSE_CPU_KERNEL_H_

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "nnacl/transpose.h"

namespace mindspore {
namespace kernel {
class ConjugateTransposeCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  ConjugateTransposeCpuKernelMod() = default;
  ~ConjugateTransposeCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
  template <typename T>
  static void ConjComplexFunc(T *input, T *output, size_t start, size_t end);

 protected:
  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
      KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
      KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
      KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
      KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
      KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
      KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
      KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
      KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
      KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
      KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
      KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
      KernelAttr()
        .AddInputAttr(kNumberTypeComplex64)
        .AddInputAttr(kNumberTypeInt32)
        .AddOutputAttr(kNumberTypeComplex64),
      KernelAttr()
        .AddInputAttr(kNumberTypeComplex128)
        .AddInputAttr(kNumberTypeInt32)
        .AddOutputAttr(kNumberTypeComplex128),
      KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
      KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
      KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
      KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
      KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
      KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
      KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
      KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
      KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
      KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
      KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
      KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
      KernelAttr()
        .AddInputAttr(kNumberTypeComplex64)
        .AddInputAttr(kNumberTypeInt64)
        .AddOutputAttr(kNumberTypeComplex64),
      KernelAttr()
        .AddInputAttr(kNumberTypeComplex128)
        .AddInputAttr(kNumberTypeInt64)
        .AddOutputAttr(kNumberTypeComplex128)};
    return support_list;
  }

 private:
  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  template <typename T>
  void LaunchComplexKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  template <typename T>
  void ParallelRun(const T *input_addr, T *output_addr, const int *output_shape, size_t count,
                   const TransposeParameter *transpose_param);
  template <typename T>
  int DoTranspose(const T *in_data, T *out_data, const int *output_shape, const TransposeParameter *transpose_param);
  template <typename T>
  void TransposeDim2(const T *in_data, T *out_data, const int *strides, const int *out_strides, const int *perm,
                     const int *output_shape);
  template <typename T>
  void TransposeDim3(const T *in_data, T *out_data, const int *strides, const int *out_strides, const int *perm,
                     const int *output_shape);
  template <typename T>
  void TransposeDim4(const T *in_data, T *out_data, const int *strides, const int *out_strides, const int *perm,
                     const int *output_shape);
  template <typename T>
  void TransposeDim5(const T *in_data, T *out_data, const int *strides, const int *out_strides, const int *perm,
                     const int *output_shape);
  template <typename T>
  void TransposeDim6(const T *in_data, T *out_data, const int *strides, const int *out_strides, const int *perm,
                     const int *output_shape);
  template <typename T>
  void TransposeDim7(const T *in_data, T *out_data, const int *strides, const int *out_strides, const int *perm,
                     const int *output_shape);
  template <typename T>
  void TransposeDims(const T *in_data, T *out_data, const int *output_shape, const TransposeParameter *transpose_param,
                     const int task_id, const int thread_num);

  TransposeParameter transpose_param_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<size_t> axes_;
  TypeId dtype_{kTypeUnknown};
  TypeId perm_type_{kTypeUnknown};
  using TypeKernel = std::function<void(ConjugateTransposeCpuKernelMod *, const std::vector<AddressPtr> &,
                                        const std::vector<AddressPtr> &)>;
  std::unordered_map<TypeId, TypeKernel> launch_map_;
  TypeKernel launch_func_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CONJUGATE_TRANSPOSE_CPU_KERNEL_H_
