/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/eigen/solve_triangular_cpu_kernel.h"
#include <Eigen/Dense>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <map>
#include "mindspore/core/ops/solve_triangular.h"

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
using KernelRunFunc = SolveTriangularCpuKernelMod::KernelRunFunc;
constexpr auto kSolveTriangularInputsNum = 2;
constexpr auto kSolveTriangularOutputsNum = 1;
int SolveTriangularCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto a_shape = LongVecToSizeVec(inputs.at(kIndex0)->GetShapeVector());
  auto b_shape = LongVecToSizeVec(inputs.at(kIndex1)->GetShapeVector());
  // Since the shape check is done in frontend, we can suppose that the shape of a, b here is valid.
  size_t a_dims = a_shape.size();
  size_t b_dims = b_shape.size();
  m_ = a_shape[a_dims - kIndex2];
  n_ = (b_dims == a_dims - 1) ? 1 : b_shape[b_dims - 1];
  batch_ = std::accumulate(a_shape.begin(), a_shape.end() - kIndex2, int64_t(1), std::multiplies{});
  return KRET_OK;
}

bool SolveTriangularCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();

  auto kernel_ptr = std::make_shared<ops::SolveTriangular>(base_operator->GetPrim());
  lower_ = kernel_ptr->get_lower();
  unit_diagonal_ = kernel_ptr->get_unit_diagonal();
  const std::string trans = kernel_ptr->get_trans();
  if (trans == "N") {
    trans_ = false;
  } else if (trans == "T") {
    trans_ = true;
  } else if (trans == "C") {
    // currently does not support complex.
    trans_ = true;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', 'trans' must be in ['N', 'T', 'C'], but got [" << trans << "].";
  }

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
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
bool SolveTriangularCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                               const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSolveTriangularInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSolveTriangularOutputsNum, kernel_name_);

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
      solve(a, b, output_batch_addr, m_, n_, !lower_, unit_diagonal_);
    } else {
      Map<Matrix<T, RowMajor>> a(a_batch_addr, m_, m_);
      solve(a, b, output_batch_addr, m_, n_, lower_, unit_diagonal_);
    }
  }

  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &SolveTriangularCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SolveTriangularCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &SolveTriangularCpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SolveTriangular, SolveTriangularCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
