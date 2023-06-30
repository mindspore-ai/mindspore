/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SEQUENCE_UNSTACK_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SEQUENCE_UNSTACK_CPU_KERNEL_H_

#include <map>
#include <algorithm>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <utility>
#include <complex>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "nnacl/base/sequence_unstack_base.h"

namespace mindspore {
namespace kernel {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

class SequenceUnstackCpuKernelMod : public NativeCpuKernelMod {
 public:
  SequenceUnstackCpuKernelMod() = default;
  ~SequenceUnstackCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
      KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt32),
      KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt16),
      KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt8),
      KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kObjectTypeTuple, kNumberTypeUInt32),
      KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kObjectTypeTuple, kNumberTypeUInt16),
      KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kObjectTypeTuple, kNumberTypeUInt8),
      KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kObjectTypeTuple, kNumberTypeUInt64),
      KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat32),
      KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat16),
      KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kObjectTypeTuple, kNumberTypeFloat64),
      KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kObjectTypeTuple, kNumberTypeComplex64),
      KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kObjectTypeTuple, kNumberTypeComplex128),
      KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kObjectTypeTuple, kNumberTypeBool)};
    return support_list;
  }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);

  using SequenceUnstackFunc =
    std::function<bool(SequenceUnstackCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, SequenceUnstackFunc>> func_list_;
  SequenceUnstackFunc kernel_func_;

  SequenceUnstackParameter sequence_unstack_param_{};
  std::vector<int64_t> input_shape_;
  size_t output_num_{0};
  size_t input_size_{1};
  size_t input_num_{1};
  size_t tuple_num_{1};
  int64_t origin_axis_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SEQUENCE_UNSTACK_CPU_KERNEL_H_
