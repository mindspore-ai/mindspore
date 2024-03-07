/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <map>

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
constexpr size_t kSolveTriangularInputsNum = 5;
constexpr size_t kSolveTriangularOutputsNum = 1;
constexpr size_t kIndexA = 0;
constexpr size_t kIndexB = 1;
constexpr size_t kIndexX = 0;
constexpr size_t kIndexTrans = 2;
constexpr size_t kIndexLower = 3;
constexpr size_t kIndexUnitDiagonal = 4;
constexpr size_t kSquareSize = 2;
constexpr int64_t kTransN = 0;
constexpr int64_t kTransT = 1;
constexpr int64_t kTransC = 2;
int SolveTriangularCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto a_shape = LongVecToSizeVec(inputs.at(kIndexA)->GetShapeVector());
  auto b_shape = LongVecToSizeVec(inputs.at(kIndexB)->GetShapeVector());
  size_t a_dims = a_shape.size();
  size_t b_dims = b_shape.size();
  m_ = a_shape[a_dims - kSquareSize];
  n_ = (b_dims == a_dims - 1) ? 1 : b_shape[b_dims - 1];
  batch_ = std::accumulate(a_shape.begin(), a_shape.end() - kSquareSize, int64_t(1), std::multiplies{});
  return KRET_OK;
}

bool SolveTriangularCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  lower_ = inputs[kIndexLower]->GetValueWithCheck<bool>();
  int64_t trans = inputs[kIndexTrans]->GetValueWithCheck<int64_t>();
  unit_diagonal_ = inputs[kIndexUnitDiagonal]->GetValueWithCheck<bool>();
  if (trans == kTransN) {
    trans_ = false;
    conj_ = false;
  } else if (trans == kTransT) {
    trans_ = true;
    conj_ = false;
  } else if (trans == kTransC) {
    trans_ = true;
    conj_ = true;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', 'trans' must be in [0, 1, 2, 'N', 'T', 'C'], but got [" << trans
                      << "].";
  }

  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }

  return true;
}

template <typename Derived_a, typename Derived_b, typename T>
void SolveTriangularCpuKernelMod::solve(const MatrixBase<Derived_a> &a, const MatrixBase<Derived_b> &b, T *output_addr,
                                        bool lower) {
  Map<Matrix<T, RowMajor>> output(output_addr, m_, n_);
  if (unit_diagonal_) {
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

void SolveTriangularCpuKernelMod::SolveTriangularCheck(const std::vector<KernelTensor *> &inputs,
                                                       const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSolveTriangularInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSolveTriangularOutputsNum, kernel_name_);
  auto a_shape = LongVecToSizeVec(inputs.at(kIndexA)->GetShapeVector());
  auto b_shape = LongVecToSizeVec(inputs.at(kIndexB)->GetShapeVector());
  auto a_rank = a_shape.size();
  auto b_rank = b_shape.size();
  const size_t expected_b_dim = (b_shape.size() == a_shape.size() - 1) ? 1 : kSquareSize;
  if (a_rank < kSquareSize) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', dim of matrix a must greater or equal to 2, but got a at " << a_rank
                             << "-dimensional ";
  }
  if (a_rank != b_rank && a_rank != b_rank + 1) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the dimension of `b` should be 'a.dim' or 'a.dim' - 1, which is " << a_rank
                             << " or " << (a_rank - 1) << ", but got " << b_rank << "-dimensions.";
  }
  if (a_shape[a_rank - kIndex1] != a_shape[a_rank - kSquareSize]) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the last two dimensions of `a` should be the same, but got shape of " << a_shape
                             << ". Please make sure that the shape of `a` be like [..., N, N].";
  }
  if (a_shape[a_rank - kSquareSize] != b_shape[b_rank - expected_b_dim]) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the last two dimensions of `a` and `b` should be matched, but got shape of "
                             << a_shape << " and " << b_shape
                             << ". Please make sure that the shape of `a` and `b` be like [..., N, N] X [..., N, M] or "
                                "[..., N, N ] X[..., N].";
  }
  if (!std::equal(a_shape.begin(), a_shape.begin() + (a_rank - kSquareSize), b_shape.begin(),
                  b_shape.begin() + (b_rank - expected_b_dim))) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the batch dimensions of `a` and `b` should all be the same, but got shape of "
                             << a_shape << " and " << b_shape
                             << ". Please make sure that the shape of `a` and `b` be like [a, b, c, ..., N, N] X [a, "
                                "b, c, ..., N, M] or [a, b, c, ..., N, N] X [a, b, c, ..., N].";
  }
}

template <typename T_in, typename T_out>
bool SolveTriangularCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &,
                                               const std::vector<KernelTensor *> &outputs) {
  SolveTriangularCheck(inputs, outputs);
  auto a_addr = reinterpret_cast<T_in *>(inputs[kIndexA]->device_ptr());
  auto b_addr = reinterpret_cast<T_in *>(inputs[kIndexB]->device_ptr());
  auto output_addr = reinterpret_cast<T_out *>(outputs[kIndexX]->device_ptr());
  size_t a_batch_size = m_ * m_;
  size_t b_batch_size = m_ * n_;
  size_t output_batch_size = b_batch_size;
  T_out *casted_a_addr = static_cast<T_out *>(malloc(sizeof(T_out) * a_batch_size));
  if (casted_a_addr == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', malloc [casted_a_addr] memory failed.";
    return false;
  }
  T_out *casted_b_addr = static_cast<T_out *>(malloc(sizeof(T_out) * b_batch_size));
  if (casted_b_addr == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', malloc [casted_b_addr] memory failed.";
    return false;
  }
  for (size_t i = 0; i < batch_; ++i) {
    T_in *a_batch_addr = a_addr + i * a_batch_size;
    T_in *b_batch_addr = b_addr + i * b_batch_size;
    T_out *output_batch_addr = output_addr + i * output_batch_size;
    for (size_t j = 0; j < a_batch_size; j++) {
      casted_a_addr[j] = static_cast<T_out>(a_batch_addr[j]);
    }
    for (size_t j = 0; j < b_batch_size; j++) {
      casted_b_addr[j] = static_cast<T_out>(b_batch_addr[j]);
    }
    Map<Matrix<T_out, RowMajor>> b(casted_b_addr, m_, n_);
    if (trans_) {
      Map<Matrix<T_out, ColMajor>> a(casted_a_addr, m_, m_);
      if (conj_) {
        solve(a.conjugate(), b, output_batch_addr, !lower_);
      } else {
        solve(a, b, output_batch_addr, !lower_);
      }
    } else {
      Map<Matrix<T_out, RowMajor>> a(casted_a_addr, m_, m_);
      solve(a, b, output_batch_addr, lower_);
    }
  }
  free(casted_a_addr);
  free(casted_b_addr);
  return true;
}

#define SOLVE_TRIANGULAR_CPU_REG(T1, T2, T3, T4)                           \
  KernelAttr()                                                             \
    .AddInputAttr(T1)                                  /* a */             \
    .AddInputAttr(T1)                                  /* b */             \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) /* trans */         \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)  /* lower */         \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)  /* unit_diagonal */ \
    .AddOutputAttr(T2),                                /* x */             \
    &SolveTriangularCpuKernelMod::LaunchKernel<T3, T4>

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &SolveTriangularCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {SOLVE_TRIANGULAR_CPU_REG(kNumberTypeFloat32, kNumberTypeFloat32, float, float)},
    {SOLVE_TRIANGULAR_CPU_REG(kNumberTypeFloat64, kNumberTypeFloat64, double, double)},
    {SOLVE_TRIANGULAR_CPU_REG(kNumberTypeInt8, kNumberTypeFloat32, int8_t, float)},
    {SOLVE_TRIANGULAR_CPU_REG(kNumberTypeInt16, kNumberTypeFloat32, int16_t, float)},
    {SOLVE_TRIANGULAR_CPU_REG(kNumberTypeInt32, kNumberTypeFloat32, int32_t, float)},
    {SOLVE_TRIANGULAR_CPU_REG(kNumberTypeInt64, kNumberTypeFloat64, int64_t, double)},
    {SOLVE_TRIANGULAR_CPU_REG(kNumberTypeComplex64, kNumberTypeComplex64, std::complex<float>, std::complex<float>)},
    {SOLVE_TRIANGULAR_CPU_REG(kNumberTypeComplex128, kNumberTypeComplex128, std::complex<double>,
                              std::complex<double>)},
    {SOLVE_TRIANGULAR_CPU_REG(kNumberTypeFloat16, kNumberTypeFloat16, Eigen::half, Eigen::half)},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SolveTriangular, SolveTriangularCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
