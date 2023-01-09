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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NN_BINARY_CROSS_ENTROPY_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NN_BINARY_CROSS_ENTROPY_KERNEL_H

#include <vector>
#include <string>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
enum ReductionType { kNone, kMean, kSum };

class BinaryCrossEntropyCpuKernelMod : public NativeCpuKernelMod {
 public:
  BinaryCrossEntropyCpuKernelMod() = default;
  ~BinaryCrossEntropyCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  void LaunchToScalar(const int &input_size, const ReductionType &reduction, T *loss, T *tmp_loss) const;
  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  TypeId dtype_{kTypeUnknown};
  size_t input_size_{1};
  ReductionType reduction_{kNone};
  bool weight_defined_{false};  // true: there are 3 inputs, false: there are 2 inputs(no [weight])
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NN_BINARY_CROSS_ENTROPY_KERNEL_H
