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

#include "plugin/device/cpu/kernel/eigen/matrix_solve_cpu_kernel.h"
#include <Eigen/Dense>
#include <map>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/matrix_solve.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kInputNum = 3;
constexpr auto kOutputNum = 1;
using KernelRunFunc = MatrixSolveCpuKernelMod::KernelRunFunc;
using Eigen::Map;
using Eigen::PartialPivLU;
template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
}  // namespace
bool MatrixSolveCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();

  auto kernel_ptr = std::make_shared<ops::MatrixSolve>(base_operator->GetPrim());
  adjoint_ = kernel_ptr->get_adjoint();

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

int MatrixSolveCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  const auto matrix_shape = inputs.at(kIndex0)->GetShapeVector();
  const auto rhs_shape = inputs.at(kIndex1)->GetShapeVector();
  batch_num_ = std::accumulate(matrix_shape.begin(), matrix_shape.end() - kIndex2, int64_t(1), std::multiplies{});
  m_ = matrix_shape.back();
  k_ = rhs_shape.back();

  return KRET_OK;
}

template <typename T>
bool MatrixSolveCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  auto matrix_ptr = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  auto rhs_ptr = reinterpret_cast<T *>(inputs[kIndex1]->addr);
  auto output_ptr = reinterpret_cast<T *>(outputs[kIndex0]->addr);

  const size_t matrix_batch_size = LongToSize(m_ * m_);
  const size_t rhs_batch_size = LongToSize(m_ * k_);

  bool invertible = true;

  auto task = [this, matrix_ptr, rhs_ptr, output_ptr, matrix_batch_size, rhs_batch_size, &invertible](size_t start,
                                                                                                      size_t end) {
    for (size_t i = start; i < end; i++) {
      Map<Matrix<T>> matrix(matrix_ptr + i * matrix_batch_size, m_, m_);
      Map<Matrix<T>> rhs(rhs_ptr + i * rhs_batch_size, m_, k_);
      Map<Matrix<T>> output(output_ptr + i * rhs_batch_size, m_, k_);

      PartialPivLU<Matrix<T>> lu(m_);
      if (adjoint_) {
        (void)lu.compute(matrix.adjoint());
      } else {
        (void)lu.compute(matrix);
      }

      if (lu.matrixLU().diagonal().cwiseAbs().minCoeff() <= 0) {
        invertible = false;
        break;
      } else {
        output.noalias() = lu.solve(rhs);
      }
    }
  };

  ParallelLaunch(task, LongToSize(batch_num_), 0, this, pool_);

  if (!invertible) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the input 'matrix' is not invertible.";
  }

  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &MatrixSolveCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &MatrixSolveCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &MatrixSolveCpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &MatrixSolveCpuKernelMod::LaunchKernel<std::complex<float>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &MatrixSolveCpuKernelMod::LaunchKernel<std::complex<double>>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MatrixSolve, MatrixSolveCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
