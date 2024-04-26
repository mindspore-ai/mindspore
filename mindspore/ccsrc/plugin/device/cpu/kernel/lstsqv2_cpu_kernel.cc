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

#include "plugin/device/cpu/kernel/lstsqv2_cpu_kernel.h"
#include <Eigen/Dense>
#include <algorithm>

namespace mindspore {
namespace kernel {
namespace {
using Eigen::ComputeThinU;
using Eigen::ComputeThinV;
using Eigen::Dynamic;
using Eigen::Map;
using Eigen::MatrixBase;
using Eigen::RowMajor;
template <typename T, int Major>
using Matrix = Eigen::Matrix<T, Dynamic, Dynamic, Major>;
constexpr size_t kLstsqV2InputsNum = 3;
constexpr size_t kLstsqV2OutputsNum = 4;
constexpr size_t kIndexA = 0;
constexpr size_t kIndexB = 1;
constexpr size_t kIndexDriver = 2;
constexpr size_t kIndexSolution = 0;
constexpr size_t kIndexResidual = 1;
constexpr size_t kIndexRank = 2;
constexpr size_t kIndexSingularValue = 3;
constexpr size_t kIndexTemp = 0;
constexpr size_t kMatrixSize = 2;
constexpr size_t kVectorSize = 1;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

bool LstsqV2CpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " valid cpu kernel does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  data_unit_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndexSolution).dtype);
  return true;
}

int LstsqV2CpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto a_shape = LongVecToSizeVec(inputs.at(kIndexA)->GetShapeVector());
  auto b_shape = LongVecToSizeVec(inputs.at(kIndexB)->GetShapeVector());
  size_t a_dims = a_shape.size();
  size_t b_dims = b_shape.size();
  size_t batch_dims = a_dims - 2;
  batch_ = 1;
  a_batch_ = 1;
  // broadcast batch dim of A and B
  for (size_t idx = 0; idx < batch_dims; idx++) {
    size_t broadcast_dim = std::max(a_shape[idx], b_shape[idx]);
    batch_ = batch_ * broadcast_dim;
    a_batch_ = a_batch_ * a_shape[idx];
    a_batch_shape_.emplace_back(a_shape[idx]);
    b_batch_shape_.emplace_back(b_shape[idx]);
    broadcast_batch_shape_.emplace_back(broadcast_dim);
  }
  m_ = a_shape[a_dims - 2];
  n_ = a_shape[a_dims - 1];
  k_ = a_dims == b_dims ? b_shape[b_dims - 1] : 1;
  a_mat_size_ = m_ * n_;
  b_mat_size_ = m_ * k_;
  solution_mat_size_ = n_ * k_;
  res_vec_size_ = k_;
  singular_value_vec_size_ = m_ < n_ ? m_ : n_;
  auto driver_opt = inputs[kIndexDriver]->GetOptionalValueWithCheck<int64_t>();
  if (driver_opt.has_value()) {
    driver_ = static_cast<DriverName>(driver_opt.value());
  } else {
    driver_ = DriverName::GELSY;
  }
  (void)workspace_size_list_.emplace_back(b_mat_size_ * data_unit_size_);
  return KRET_OK;
}

void LstsqV2CpuKernelMod::LstsqV2Check(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kLstsqV2InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kLstsqV2OutputsNum, kernel_name_);
  auto a_shape = LongVecToSizeVec(inputs.at(kIndexA)->GetShapeVector());
  auto b_shape = LongVecToSizeVec(inputs.at(kIndexB)->GetShapeVector());
  auto a_rank = a_shape.size();
  auto b_rank = b_shape.size();
  const size_t b_unit_size = (b_shape.size() == a_shape.size() - 1) ? kVectorSize : kMatrixSize;
  if (a_rank < kMatrixSize) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', dim of matrix a must greater or equal to 2, but got a at " << a_rank
                             << "-dimensional ";
  }
  if (a_rank != b_rank && a_rank != b_rank + 1) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the dimension of `b` should be 'a.dim' or 'a.dim' - 1, which is " << a_rank
                             << " or " << (a_rank - 1) << ", but got " << b_rank << "-dimensions.";
  }
  if (a_shape[a_rank - kMatrixSize] != b_shape[b_rank - b_unit_size]) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the last two dimensions of `a` and `b` should be matched, but got shape of "
                             << a_shape << " and " << b_shape
                             << ". Please make sure that the shape of `a` and `b` be like [..., N, M] X [..., N, K] or "
                                "[..., N, M] X [..., N].";
  }
}

template <typename T>
void LstsqFrobeniusNorm(T *temp_addr, T *res_addr, size_t col_size, size_t row_size) {
  for (size_t row_idx = 0; row_idx < row_size; row_idx++) {
    res_addr[row_idx] = 0;
    for (size_t col_idx = 0; col_idx < col_size; col_idx++) {
      size_t idx = col_idx * row_size + row_idx;
      res_addr[row_idx] += temp_addr[idx] * temp_addr[idx];
    }
  }
}

void LstsqFrobeniusNorm(complex64 *temp_addr, float *res_addr, size_t col_size, size_t row_size) {
  for (size_t row_idx = 0; row_idx < row_size; row_idx++) {
    res_addr[row_idx] = 0;
    for (size_t col_idx = 0; col_idx < col_size; col_idx++) {
      size_t idx = col_idx * row_size + row_idx;
      res_addr[row_idx] +=
        temp_addr[idx].real() * temp_addr[idx].real() + temp_addr[idx].imag() * temp_addr[idx].imag();
    }
  }
}

void LstsqFrobeniusNorm(complex128 *temp_addr, double *res_addr, size_t col_size, size_t row_size) {
  for (size_t row_idx = 0; row_idx < row_size; row_idx++) {
    res_addr[row_idx] = 0;
    for (size_t col_idx = 0; col_idx < col_size; col_idx++) {
      size_t idx = col_idx * row_size + row_idx;
      res_addr[row_idx] +=
        temp_addr[idx].real() * temp_addr[idx].real() + temp_addr[idx].imag() * temp_addr[idx].imag();
    }
  }
}

template <typename T1, typename T2>
bool LstsqV2CpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                       const std::vector<kernel::KernelTensor *> &workspace,
                                       const std::vector<kernel::KernelTensor *> &outputs) {
  LstsqV2Check(inputs, outputs);
  T1 *input_a_addr = reinterpret_cast<T1 *>(inputs[kIndexA]->device_ptr());
  T1 *input_b_addr = reinterpret_cast<T1 *>(inputs[kIndexB]->device_ptr());

  T1 *output_solution_addr = reinterpret_cast<T1 *>(outputs[kIndexSolution]->device_ptr());
  T2 *output_residual_addr = reinterpret_cast<T2 *>(outputs[kIndexResidual]->device_ptr());
  int64_t *output_rank_addr = reinterpret_cast<int64_t *>(outputs[kIndexRank]->device_ptr());
  T2 *output_singular_value_addr = reinterpret_cast<T2 *>(outputs[kIndexSingularValue]->device_ptr());

  T1 *temp_addr = reinterpret_cast<T1 *>(workspace[kIndexTemp]->device_ptr());
  if (a_mat_size_ == 0 || b_mat_size_ == 0) {
    output_rank_addr[0] = 0;
    return true;
  }

  // calculate rank and singular value with A batch dim
  for (size_t i = 0; i < a_batch_; i++) {
    T1 *a_batch_addr = input_a_addr + i * a_mat_size_;
    Map<Matrix<T1, RowMajor>> a(a_batch_addr, m_, n_);
    if (driver_ == DriverName::GELS) {
      if (m_ >= n_) {
        output_rank_addr[i] = a.fullPivHouseholderQr().rank();
      } else {
        output_rank_addr[i] = a.fullPivLu().rank();
      }
    } else {
      output_rank_addr[i] = a.completeOrthogonalDecomposition().rank();
    }
    if (driver_ == DriverName::GELSS || driver_ == DriverName::GELSD) {
      T2 *singular_value_batch_addr = output_singular_value_addr + i * singular_value_vec_size_;
      Matrix<T2, RowMajor> singular_value = a.bdcSvd(ComputeThinU | ComputeThinV).singularValues();
      Map<Matrix<T2, RowMajor>>(singular_value_batch_addr, singular_value.rows(), singular_value.cols()) =
        singular_value;
    }
  }
  BroadcastIterator batch_broadcast_iter(a_batch_shape_, b_batch_shape_, broadcast_batch_shape_);
  batch_broadcast_iter.SetPos(0);
  // calculate solution and residual with broadcast batch dim
  for (size_t i = 0; i < batch_; i++) {
    T1 *a_batch_addr = input_a_addr + batch_broadcast_iter.GetInputPosA() * a_mat_size_;
    T1 *b_batch_addr = input_b_addr + batch_broadcast_iter.GetInputPosB() * b_mat_size_;
    T1 *solution_batch_addr = output_solution_addr + i * solution_mat_size_;
    T2 *residual_batch_addr = output_residual_addr + i * res_vec_size_;
    Map<Matrix<T1, RowMajor>> a(a_batch_addr, m_, n_);
    Map<Matrix<T1, RowMajor>> b(b_batch_addr, m_, k_);
    Map<Matrix<T1, RowMajor>> solution(solution_batch_addr, n_, k_);
    Map<Matrix<T1, RowMajor>> temp(temp_addr, m_, k_);
    if (driver_ == DriverName::GELSS || driver_ == DriverName::GELSD) {
      solution = a.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
    } else {
      solution = a.completeOrthogonalDecomposition().solve(b);
    }
    bool compute_res = driver_ != DriverName::GELSY && m_ > n_;
    if (compute_res) {
      temp = b - a * solution;
      LstsqFrobeniusNorm(temp_addr, residual_batch_addr, m_, k_);
    }
    batch_broadcast_iter.GenNextPos();
  }
  return true;
}

#define LSTSQV2_CPU_REG(T1, T2, T3, T4)                           \
  KernelAttr()                                                    \
    .AddInputAttr(T1)                       /* a */               \
    .AddInputAttr(T1)                       /* b */               \
    .AddOptionalInputAttr(kNumberTypeInt64) /* driver */          \
    .AddOutputAttr(T1)                      /* solution */        \
    .AddOutputAttr(T2)                      /* residuals */       \
    .AddOutputAttr(kNumberTypeInt64)        /* rank */            \
    .AddOutputAttr(T2),                     /* singular_values */ \
    &LstsqV2CpuKernelMod::LaunchKernel<T3, T4>

std::vector<std::pair<KernelAttr, LstsqV2CpuKernelMod::LstsqV2Func>> LstsqV2CpuKernelMod::func_list_ = {
  {LSTSQV2_CPU_REG(kNumberTypeFloat32, kNumberTypeFloat32, float, float)},
  {LSTSQV2_CPU_REG(kNumberTypeFloat64, kNumberTypeFloat64, double, double)},
  {LSTSQV2_CPU_REG(kNumberTypeComplex64, kNumberTypeFloat32, complex64, float)},
  {LSTSQV2_CPU_REG(kNumberTypeComplex128, kNumberTypeFloat64, complex128, double)},
};

std::vector<KernelAttr> LstsqV2CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LstsqV2Func> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LstsqV2, LstsqV2CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
