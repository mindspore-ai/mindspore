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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_EXP_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_EXP_CPU_KERNEL_H_

#include <algorithm>
#include <complex>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/eigen/eigen_common_utils.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class MatrixExpCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<MatrixExpCpuKernelMod> {
 public:
  MatrixExpCpuKernelMod() = default;
  ~MatrixExpCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  bool CheckInputShape() const;

  template <typename Derived1, typename Derived2, typename Derived3>
  void MTaylorApproximant(const Eigen::MatrixBase<Derived1> &A, const Eigen::MatrixBase<Derived2> &I, int order,
                          Eigen::MatrixBase<Derived3> *E) const;

  template <typename Derived1, typename Derived2>
  void MexpImpl(const Eigen::MatrixBase<Derived1> &A, const Eigen::MatrixBase<Derived2> &I,
                Eigen::MatrixBase<Derived1> *mexp) const;

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);

  void TyepChangeForFp16(int64_t i, int64_t m, int64_t size_mm, const mindspore::Float16 *input_x,
                         mindspore::Float16 *output_y) const;

  template <typename T>
  bool LaunchKernelFP16(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                        const std::vector<kernel::AddressPtr> &outputs);

  std::vector<size_t> input_shape_;
  TypeId data_type_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_EXP_CPU_KERNEL_H_
