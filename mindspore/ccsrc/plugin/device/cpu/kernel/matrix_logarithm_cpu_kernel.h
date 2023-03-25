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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_LOGARITHM_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_LOGARITHM_CPU_KERNEL_H_

#include <vector>
#include <utility>
#include <map>
#include <algorithm>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class MatrixLogarithmCpuKernelMod : public NativeCpuKernelMod {
 public:
  MatrixLogarithmCpuKernelMod() = default;
  ~MatrixLogarithmCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    kernel_func_(this, inputs, outputs);
    return true;
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  using MatrixLogarithmLaunchFunc = std::function<void(MatrixLogarithmCpuKernelMod *, const std::vector<AddressPtr> &,
                                                       const std::vector<AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, MatrixLogarithmLaunchFunc>> func_list_;
  MatrixLogarithmLaunchFunc kernel_func_;

  template <typename T>
  void LaunchMatrixLogarithm(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  ShapeVector shape_x_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_LOGARITHM_CPU_KERNEL_H_
