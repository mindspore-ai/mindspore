/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/eigen/solve_triangular_grad_cpu_kernel.h"
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <map>
#include "mindspore/core/ops/grad/solve_triangular_grad.h"

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
using KernelRunFunc = SolveTriangularGradCpuKernelMod::KernelRunFunc;
constexpr size_t kSolveTriangularGradInputsNum = 6;
constexpr size_t kSolveTriangularGradOutputsNum = 2;
constexpr size_t kIndexA = 0;
constexpr size_t kIndexX = 1;
constexpr size_t kIndexDX = 2;
constexpr size_t kIndexTrans = 3;
constexpr size_t kIndexLower = 4;
constexpr size_t kIndexUnitDiagonal = 5;
constexpr size_t kIndexDA = 0;
constexpr size_t kIndexDB = 1;
constexpr size_t kSquareSize = 2;
constexpr int64_t kTransN = 0;
constexpr int64_t kTransT = 1;
constexpr int64_t kTransC = 2;
int SolveTriangularGradCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto a_shape = LongVecToSizeVec(inputs.at(kIndexA)->GetShapeVector());
  auto x_shape = LongVecToSizeVec(inputs.at(kIndexX)->GetShapeVector());

  size_t a_dims = a_shape.size();
  size_t x_dims = x_shape.size();
  m_ = a_shape[a_dims - kSquareSize];
  n_ = (x_dims == a_dims - 1) ? 1 : x_shape[x_dims - 1];
  batch_ = std::accumulate(a_shape.begin(), a_shape.end() - kSquareSize, int64_t(1), std::multiplies{});
  a_batch_size_ = m_ * m_;
  x_batch_size_ = m_ * n_;
  dx_batch_size_ = x_batch_size_;
  da_batch_size_ = a_batch_size_;
  db_batch_size_ = x_batch_size_;
  return KRET_OK;
}

bool SolveTriangularGradCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  return true;
}

template <typename Derived_a, typename Derived_b, typename T>
void SolveTriangularGradCpuKernelMod::solve(const MatrixBase<Derived_a> &a, const MatrixBase<Derived_b> &b,
                                            T *output_addr, bool lower) {
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

template <typename T>
void mat_tri_view(T *mat_addr, size_t m, bool lower, bool unit_diagonal) {
  T zero = static_cast<T>(0);
  if (unit_diagonal) {
    for (size_t i = 0; i < m; i++) {
      mat_addr[i * m + i] = zero;
    }
  }
  if (lower) {
    for (size_t i = 0; i < m; i++) {
      for (size_t j = i + 1; j < m; j++) {
        mat_addr[i * m + j] = zero;
      }
    }
  } else {
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < i; j++) {
        mat_addr[i * m + j] = zero;
      }
    }
  }
}

void SolveTriangularGradCpuKernelMod::set_attr(const std::vector<KernelTensor *> &inputs) {
  int64_t trans = *reinterpret_cast<int64_t *>(inputs[kIndexTrans]->device_ptr());
  lower_ = *reinterpret_cast<bool *>(inputs[kIndexLower]->device_ptr());
  unit_diagonal_ = *reinterpret_cast<bool *>(inputs[kIndexUnitDiagonal]->device_ptr());

  if (trans == kTransN) {
    trans_ = true;
    conj_ = false;
  } else if (trans == kTransT) {
    trans_ = false;
    conj_ = false;
  } else if (trans == kTransC) {
    trans_ = false;
    conj_ = true;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', 'trans' must be in [0, 1, 2, 'N', 'T', 'C'], but got [" << trans
                      << "].";
  }
}

template <typename T>
void SolveTriangularGradCpuKernelMod::calculate_db(T *a_addr, T *dx_addr, T *db_addr) {
  Map<Matrix<T, RowMajor>> dx(dx_addr, m_, n_);
  if (trans_) {
    Map<Matrix<T, ColMajor>> a(a_addr, m_, m_);
    solve(a, dx, db_addr, !lower_);
  } else {
    Map<Matrix<T, RowMajor>> a(a_addr, m_, m_);
    if (conj_) {
      solve(a.conjugate(), dx, db_addr, lower_);
    } else {
      solve(a, dx, db_addr, lower_);
    }
  }
}

template <typename T>
void SolveTriangularGradCpuKernelMod::calculate_da(T *x_addr, T *da_addr, T *db_addr) {
  Map<Matrix<T, RowMajor>> x(x_addr, m_, n_);
  Map<Matrix<T, RowMajor>> db(db_addr, m_, n_);
  Map<Matrix<T, RowMajor>> da(da_addr, m_, m_);
  if (!trans_) {
    da = x * db.transpose();
  } else {
    da = db * x.transpose();
  }
  da = -da;
  if (conj_) {
    Map<Matrix<T, RowMajor>>(da_addr, m_, m_) = da.conjugate();
  }
  mat_tri_view(da_addr, m_, lower_, unit_diagonal_);
}

template <typename T_in, typename T_out, typename T_grad>
bool SolveTriangularGradCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &,
                                                   const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSolveTriangularGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSolveTriangularGradOutputsNum, kernel_name_);
  auto a_addr = reinterpret_cast<T_in *>(inputs[kIndexA]->device_ptr());
  auto x_addr = reinterpret_cast<T_out *>(inputs[kIndexX]->device_ptr());
  auto dx_addr = reinterpret_cast<T_out *>(inputs[kIndexDX]->device_ptr());
  auto da_addr = reinterpret_cast<T_grad *>(outputs[kIndexDA]->device_ptr());
  auto db_addr = reinterpret_cast<T_grad *>(outputs[kIndexDB]->device_ptr());
  set_attr(inputs);
  T_grad *casted_a_addr = static_cast<T_grad *>(malloc(sizeof(T_grad) * a_batch_size_));
  if (casted_a_addr == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', malloc [casted_a_addr] memory failed.";
    return false;
  }
  T_grad *casted_x_addr = static_cast<T_grad *>(malloc(sizeof(T_grad) * x_batch_size_));
  if (casted_x_addr == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', malloc [casted_x_addr] memory failed.";
    return false;
  }
  T_grad *casted_dx_addr = static_cast<T_grad *>(malloc(sizeof(T_grad) * dx_batch_size_));
  if (casted_dx_addr == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', malloc [casted_dx_addr] memory failed.";
    return false;
  }
  for (size_t i = 0; i < batch_; ++i) {
    T_in *a_batch_addr = a_addr + i * a_batch_size_;
    T_out *x_batch_addr = x_addr + i * x_batch_size_;
    T_out *dx_batch_addr = dx_addr + i * dx_batch_size_;
    T_grad *da_batch_addr = da_addr + i * da_batch_size_;
    T_grad *db_batch_addr = db_addr + i * db_batch_size_;
    for (size_t j = 0; j < a_batch_size_; j++) {
      casted_a_addr[j] = static_cast<T_grad>(a_batch_addr[j]);
    }
    for (size_t j = 0; j < x_batch_size_; j++) {
      casted_x_addr[j] = static_cast<T_grad>(x_batch_addr[j]);
    }
    for (size_t j = 0; j < dx_batch_size_; j++) {
      casted_dx_addr[j] = static_cast<T_grad>(dx_batch_addr[j]);
    }
    calculate_db<T_grad>(casted_a_addr, casted_dx_addr, db_batch_addr);
    calculate_da<T_grad>(casted_x_addr, da_batch_addr, db_batch_addr);
  }
  free(casted_a_addr);
  free(casted_x_addr);
  free(casted_dx_addr);
  return true;
}

#define SOLVE_TRIANGULAR_GRAD_CPU_REG(T1, T2, T3, T4, T5, T6)              \
  KernelAttr()                                                             \
    .AddInputAttr(T1)                                  /* a */             \
    .AddInputAttr(T2)                                  /* x */             \
    .AddInputAttr(T2)                                  /* dx */            \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) /* trans */         \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)  /* lower */         \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)  /* unit_diagonal */ \
    .AddOutputAttr(T3)                                 /* da */            \
    .AddOutputAttr(T3),                                /* db */            \
    &SolveTriangularGradCpuKernelMod::LaunchKernel<T4, T5, T6>

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &SolveTriangularGradCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {SOLVE_TRIANGULAR_GRAD_CPU_REG(kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32, float, float, float)},
    {SOLVE_TRIANGULAR_GRAD_CPU_REG(kNumberTypeFloat64, kNumberTypeFloat64, kNumberTypeFloat64, double, double, double)},
    {SOLVE_TRIANGULAR_GRAD_CPU_REG(kNumberTypeInt8, kNumberTypeFloat32, kNumberTypeFloat32, int8_t, float, float)},
    {SOLVE_TRIANGULAR_GRAD_CPU_REG(kNumberTypeInt16, kNumberTypeFloat32, kNumberTypeFloat32, int16_t, float, float)},
    {SOLVE_TRIANGULAR_GRAD_CPU_REG(kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32, int32_t, float, float)},
    {SOLVE_TRIANGULAR_GRAD_CPU_REG(kNumberTypeInt64, kNumberTypeFloat64, kNumberTypeFloat32, int64_t, double, float)},
    {SOLVE_TRIANGULAR_GRAD_CPU_REG(kNumberTypeComplex64, kNumberTypeComplex64, kNumberTypeComplex64,
                                   std::complex<float>, std::complex<float>, std::complex<float>)},
    {SOLVE_TRIANGULAR_GRAD_CPU_REG(kNumberTypeComplex128, kNumberTypeComplex128, kNumberTypeComplex128,
                                   std::complex<double>, std::complex<double>, std::complex<double>)},
    {SOLVE_TRIANGULAR_GRAD_CPU_REG(kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat32, Eigen::half, Eigen::half,
                                   float)},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SolveTriangularGrad, SolveTriangularGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
