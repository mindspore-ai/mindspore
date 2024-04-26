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

#include "cpu_kernel/ms_kernel/lstsqv2.h"

#include <Eigen/Dense>
#include <iostream>

#include "context/inc/cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "utils/eigen_tensor.h"
#include "utils/bcast.h"

namespace {
const uint32_t kOutputNum = 4;
const uint32_t kInputNum = 2;
constexpr size_t kIndexA = 0;
constexpr size_t kIndexB = 1;
constexpr size_t kIndexDriver = 2;
constexpr size_t kIndexSolution = 0;
constexpr size_t kIndexResidual = 1;
constexpr size_t kIndexRank = 2;
constexpr size_t kIndexSingularValue = 3;
constexpr size_t kMatrixSize = 2;
constexpr size_t kVectorSize = 1;
constexpr int64_t kDriverGELS = 0;
constexpr int64_t kDriverGELSY = 1;
constexpr int64_t kDriverGELSD = 2;
constexpr int64_t kDriverGELSS = 3;
const char *kLstsqV2 = "LstsqV2";
const char *const kInputDriverName = "driver";
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
}  // namespace
// namespace aicpu
namespace aicpu {
uint32_t LstsqV2CpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "LstsqV2 check input and output number failed.");
  CUST_KERNEL_HANDLE_ERROR(ctx, LstsqV2Check(ctx), "[%s] check params failed.", kLstsqV2);
  DataType data_type = ctx.Input(kIndexA)->GetDataType();
  switch (data_type) {
    case DT_FLOAT:
      return LstsqV2Compute<float, float>(ctx);
    case DT_DOUBLE:
      return LstsqV2Compute<double, double>(ctx);
    case DT_COMPLEX64:
      return LstsqV2Compute<complex64, float>(ctx);
    case DT_COMPLEX128:
      return LstsqV2Compute<complex128, double>(ctx);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "LstsqV2 kernel data type [%u] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t LstsqV2CpuKernel::LstsqV2Check(CpuKernelContext &ctx) {
  std::vector<int64_t> a_shape = ctx.Input(kIndexA)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> b_shape = ctx.Input(kIndexB)->GetTensorShape()->GetDimSizes();
  auto a_rank = a_shape.size();
  auto b_rank = b_shape.size();
  const size_t b_unit_size = (b_shape.size() == a_shape.size() - 1) ? kVectorSize : kMatrixSize;
  if (a_rank < kMatrixSize) {
    CUST_KERNEL_LOG_ERROR(ctx, "For [%s], dim of matrix a must greater or equal to 2, but got a at [%lld]-dimensional.",
                          kLstsqV2, a_rank);
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (a_rank != b_rank && a_rank != b_rank + 1) {
    CUST_KERNEL_LOG_ERROR(
      ctx,
      "For [%s], the dimension of `b` should be 'a.dim' or 'a.dim' - 1, which is [%lld] or [%lld], but got "
      "[%lld]-dimensions.",
      kLstsqV2, a_rank, a_rank - 1, b_rank);
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (a_shape[a_rank - kMatrixSize] != b_shape[b_rank - b_unit_size]) {
    CUST_KERNEL_LOG_ERROR(
      ctx,
      "For [%s], the last two dimensions of `a` and `b` should be matched, but got shape of [%s] and [%s]. Please make "
      "sure that the shape of `a` and `b` be like [..., N, N] X [..., N, M] or [..., N, N ] X[..., N].",
      kLstsqV2, VectorToString(a_shape), VectorToString(b_shape));
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
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
uint32_t LstsqV2CpuKernel::LstsqV2Compute(CpuKernelContext &ctx) {
  std::vector<int64_t> shape_a = ctx.Input(kIndexA)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_b = ctx.Input(kIndexB)->GetTensorShape()->GetDimSizes();
  size_t a_dims = shape_a.size();
  size_t b_dims = shape_b.size();
  size_t batch_dims = a_dims - 2;
  int64_t batch = 1;
  int64_t batch_a = 1;
  std::vector<int64_t> batch_shape_a(shape_a.begin(), shape_a.begin() + batch_dims);
  std::vector<int64_t> batch_shape_b(shape_b.begin(), shape_b.begin() + batch_dims);
  std::vector<int64_t> broadcast_batch_shape;
  for (size_t idx = 0; idx < batch_dims; idx++) {
    int64_t broadcast_dim = std::max(shape_a[idx], shape_b[idx]);
    batch = batch * broadcast_dim;
    batch_a = batch_a * shape_a[idx];
    broadcast_batch_shape.emplace_back(broadcast_dim);
  }

  int64_t m = shape_a[a_dims - 2];
  int64_t n = shape_a[a_dims - 1];
  int64_t k = a_dims == b_dims ? shape_b[b_dims - 1] : 1;
  int64_t a_mat_size = m * n;
  int64_t b_mat_size = m * k;
  int64_t solution_mat_size = n * k;
  int64_t res_vec_size = k;
  int64_t singular_value_vec_size = m < n ? m : n;
  int64_t *rank_addr = reinterpret_cast<int64_t *>(ctx.Output(kIndexRank)->GetData());
  if (a_mat_size == 0 || b_mat_size == 0) {
    rank_addr[0] = 0;
    return KERNEL_STATUS_OK;
  }
  T1 *a_addr = reinterpret_cast<T1 *>(ctx.Input(kIndexA)->GetData());
  T1 *b_addr = reinterpret_cast<T1 *>(ctx.Input(kIndexB)->GetData());
  T1 *solution_addr = reinterpret_cast<T1 *>(ctx.Output(kIndexSolution)->GetData());
  T2 *residual_addr = reinterpret_cast<T2 *>(ctx.Output(kIndexResidual)->GetData());
  T2 *singular_value_addr = reinterpret_cast<T2 *>(ctx.Output(kIndexSingularValue)->GetData());

  Bcast bcast_a(ctx, batch_shape_a, broadcast_batch_shape);
  Bcast bcast_b(ctx, batch_shape_b, broadcast_batch_shape);

  bool driver_is_none = (ctx.Input(kIndexDriver) == nullptr) ||
                        (CpuKernelUtils::GetTensorName(ctx.Input(kIndexDriver)) != kInputDriverName);

  int64_t driver;

  if (driver_is_none) {
    driver = kDriverGELSY;
  } else {
    driver = reinterpret_cast<int64_t *>(ctx.Input(kIndexDriver)->GetData())[0];
  }
  T1 *temp_addr = static_cast<T1 *>(malloc(sizeof(T1) * b_mat_size));
  CUST_KERNEL_CHECK_NULLPTR(ctx, temp_addr, KERNEL_STATUS_INNER_ERROR, "[LstsqV2] Malloc memory [temp_addr] failed!")
  // calculate rank and singular value with A batch dim
  for (int64_t i = 0; i < batch_a; i++) {
    T1 *a_batch_addr = a_addr + i * a_mat_size;
    Eigen::Map<Matrix<T1>> a(a_batch_addr, m, n);
    if (driver == kDriverGELS) {
      if (m >= n) {
        rank_addr[i] = a.fullPivHouseholderQr().rank();
      } else {
        rank_addr[i] = a.fullPivLu().rank();
      }
    } else {
      rank_addr[i] = a.completeOrthogonalDecomposition().rank();
    }
    if (driver == kDriverGELSS || driver == kDriverGELSD) {
      T2 *singular_value_batch_addr = singular_value_addr + i * singular_value_vec_size;
      Matrix<T2> singular_value = a.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).singularValues();
      Eigen::Map<Matrix<T2>>(singular_value_batch_addr, singular_value.rows(), singular_value.cols()) = singular_value;
    }
  }
  // calculate solution and residual with broadcast batch dim
  for (int64_t i = 0; i < batch; i++) {
    T1 *a_batch_addr = a_addr + bcast_a.GetBroadcastXIndex(i) * a_mat_size;
    T1 *b_batch_addr = b_addr + bcast_b.GetBroadcastXIndex(i) * b_mat_size;
    T1 *solution_batch_addr = solution_addr + i * solution_mat_size;
    T2 *residual_batch_addr = residual_addr + i * res_vec_size;
    Eigen::Map<Matrix<T1>> a(a_batch_addr, m, n);
    Eigen::Map<Matrix<T1>> b(b_batch_addr, m, k);
    Eigen::Map<Matrix<T1>> solution(solution_batch_addr, n, k);
    Eigen::Map<Matrix<T1>> temp(temp_addr, m, k);
    if (driver == kDriverGELSS || driver == kDriverGELSD) {
      solution = a.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    } else {
      solution = a.completeOrthogonalDecomposition().solve(b);
    }
    bool compute_res = driver != kDriverGELSY && m > n;
    if (compute_res) {
      temp = b - a * solution;
      LstsqFrobeniusNorm(temp_addr, residual_batch_addr, m, k);
    }
  }
  free(temp_addr);
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kLstsqV2, LstsqV2CpuKernel);
}  // namespace aicpu
