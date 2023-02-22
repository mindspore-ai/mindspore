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

#include "plugin/device/cpu/kernel/matrix_triangular_solve_cpu_kernel.h"
#include <Eigen/Dense>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>

namespace mindspore {
namespace kernel {
using Eigen::ColMajor;
using Eigen::Dynamic;
using Eigen::Lower;
using Eigen::Map;
using Eigen::MatrixBase;
using Eigen::RowMajor;
using Eigen::UnitLower;
using Eigen::UnitUpper;
using Eigen::Upper;
template <typename T, int Major>
using Matrix = Eigen::Matrix<T, Dynamic, Dynamic, Major>;
constexpr auto kSolveTriangularInputsNum = 2;
constexpr auto kSolveTriangularOutputsNum = 1;
constexpr auto kAVectorxDimNum = 1;
constexpr auto kAMatrixDimNum = 2;
constexpr size_t kRowIndex = 2;
constexpr size_t kColIndex = 1;

bool MatrixTriangularSolveCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSolveTriangularInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSolveTriangularOutputsNum, kernel_name_);

  trans_ = GetValue<bool>(prim->GetAttr(ADJOINT));
  lower_ = GetValue<bool>(prim->GetAttr(LOWER));

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int MatrixTriangularSolveCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs,
                                              const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto a_shape = inputs[0]->GetShapeVector();
  auto b_shape = inputs[1]->GetShapeVector();
  // Since the shape check is done in frontend, we can suppose that the shape of a, b here is valid.
  size_t a_dims = a_shape.size();
  size_t aRowIndex = a_dims - kRowIndex;
  m_ = a_shape[aRowIndex];
  size_t b_sims = b_shape.size();
  bool vector_b = b_sims == a_dims - 1;
  if (vector_b) {
    n_ = 1;
  } else {
    n_ = b_shape[b_sims - 1];
  }
  batch_ = 1;
  for (size_t batch = 0; batch < a_dims - kRowIndex; ++batch) {
    batch_ *= a_shape[batch];
  }
  return KRET_OK;
}

template <typename Derived_a, typename Derived_b, typename T>
inline void solve(const MatrixBase<Derived_a> &a, const MatrixBase<Derived_b> &b, T *output_addr, int m, int n,
                  bool lower, bool unit_diagonal) {
  Map<Matrix<T, RowMajor>> output(output_addr, m, n);
  if (unit_diagonal) {
    if (lower) {
      output.noalias() = a.template triangularView<UnitLower>().solve(b);
    } else {
      output.noalias() = a.template triangularView<UnitUpper>().solve(b);
    }
  } else {
    if (lower) {
      output.noalias() = a.template triangularView<Lower>().solve(b);
    } else {
      output.noalias() = a.template triangularView<Upper>().solve(b);
    }
  }
}

template <typename T>
bool MatrixTriangularSolveCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                     const std::vector<AddressPtr> &,
                                                     const std::vector<AddressPtr> &outputs) {
  auto a_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto b_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);

  size_t a_batch_size = m_ * m_;
  size_t b_batch_size = m_ * n_;
  size_t output_batch_size = m_ * n_;

  for (size_t i = 0; i < batch_; ++i) {
    T *a_batch_addr = a_addr + i * a_batch_size;
    T *b_batch_addr = b_addr + i * b_batch_size;
    T *output_batch_addr = output_addr + i * output_batch_size;

    Map<Matrix<T, RowMajor>> b(b_batch_addr, m_, n_);
    if (trans_) {
      Map<Matrix<T, ColMajor>> a(a_batch_addr, m_, m_);
      auto a_conj = a.conjugate();
      solve(a_conj, b, output_batch_addr, m_, n_, !lower_, unit_diagonal_);
    } else {
      Map<Matrix<T, RowMajor>> a(a_batch_addr, m_, m_);
      solve(a, b, output_batch_addr, m_, n_, lower_, unit_diagonal_);
    }
  }

  return true;
}

std::vector<std::pair<KernelAttr, MatrixTriangularSolveCpuKernelMod::MatrixTriangularSolveFunc>>
  MatrixTriangularSolveCpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &MatrixTriangularSolveCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &MatrixTriangularSolveCpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &MatrixTriangularSolveCpuKernelMod::LaunchKernel<std::complex<float>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &MatrixTriangularSolveCpuKernelMod::LaunchKernel<std::complex<double>>}};

std::vector<KernelAttr> MatrixTriangularSolveCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MatrixTriangularSolveFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MatrixTriangularSolve, MatrixTriangularSolveCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
