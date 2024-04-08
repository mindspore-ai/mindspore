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
#include "solve_triangular_grad.h"
#include <cstdint>
#include <string.h>
#include "securec.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "solve_triangular.h"

namespace {
const char *kSolveTriangularGrad = "SolveTriangularGrad";
constexpr uint32_t kSolveTriangularGradInputsNum = 6;
constexpr uint32_t kSolveTriangularGradOutputsNum = 2;
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
}  // namespace

namespace aicpu {

uint32_t SolveTriangularGradCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kSolveTriangularGradInputsNum, kSolveTriangularGradOutputsNum),
                           "[%s] check input and output failed.", kSolveTriangularGrad);
  auto a_type = ctx.Input(0)->GetDataType();
  uint32_t ret = KERNEL_STATUS_OK;
  switch (a_type) {
    case DT_FLOAT:
      ret = SolveTriangularGradCompute<float, float, float>(ctx);
      break;
    case DT_DOUBLE:
      ret = SolveTriangularGradCompute<double, double, double>(ctx);
      break;
    case DT_FLOAT16:
      ret = SolveTriangularGradCompute<Eigen::half, Eigen::half, float>(ctx);
      break;
    case DT_INT8:
      ret = SolveTriangularGradCompute<int8_t, float, float>(ctx);
      break;
    case DT_INT16:
      ret = SolveTriangularGradCompute<int16_t, float, float>(ctx);
      break;
    case DT_INT32:
      ret = SolveTriangularGradCompute<int32_t, float, float>(ctx);
      break;
    case DT_INT64:
      ret = SolveTriangularGradCompute<int64_t, double, float>(ctx);
      break;
    case DT_COMPLEX64:
      ret = SolveTriangularGradCompute<std::complex<float>, std::complex<float>, std::complex<float>>(ctx);
      break;
    case DT_COMPLEX128:
      ret = SolveTriangularGradCompute<std::complex<double>, std::complex<double>, std::complex<double>>(ctx);
      break;
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "[%s] Data type of input is not support, input data type is [%s].",
                            ctx.GetOpType().c_str(), DTypeStr(a_type).c_str());
      ret = KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
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

template <typename T_in, typename T_out, typename T_grad>
uint32_t SolveTriangularGradCpuKernel::SolveTriangularGradCompute(CpuKernelContext &ctx) {
  auto input_a_addr = reinterpret_cast<T_in *>(ctx.Input(kIndexA)->GetData());
  auto input_x_addr = reinterpret_cast<T_out *>(ctx.Input(kIndexX)->GetData());
  auto input_dx_addr = reinterpret_cast<T_out *>(ctx.Input(kIndexDX)->GetData());
  auto output_da_addr = reinterpret_cast<T_grad *>(ctx.Output(kIndexDA)->GetData());
  auto output_db_addr = reinterpret_cast<T_grad *>(ctx.Output(kIndexDB)->GetData());

  int64_t *trans_ptr = reinterpret_cast<int64_t *>(ctx.Input(kIndexTrans)->GetData());
  int64_t trans = *trans_ptr;
  bool *lower_ptr = reinterpret_cast<bool *>(ctx.Input(kIndexLower)->GetData());
  bool lower = *lower_ptr;
  bool *unit_diagonal_ptr = reinterpret_cast<bool *>(ctx.Input(kIndexUnitDiagonal)->GetData());
  bool unit_diagonal = *unit_diagonal_ptr;

  std::vector<int64_t> shape_a = ctx.Input(kIndexA)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_dx = ctx.Input(kIndexDX)->GetTensorShape()->GetDimSizes();
  size_t a_dims = shape_a.size();
  size_t dx_dims = shape_dx.size();
  size_t m = shape_a[a_dims - kSquareSize];
  size_t n = (dx_dims == a_dims - 1) ? 1 : shape_dx[dx_dims - 1];
  size_t batch = std::accumulate(shape_a.begin(), shape_a.end() - kSquareSize, size_t(1), std::multiplies{});

  size_t a_mat_size = m * m;
  size_t x_mat_size = m * n;
  size_t dx_mat_size = x_mat_size;
  size_t da_mat_size = a_mat_size;
  size_t db_mat_size = x_mat_size;

  T_grad *casted_a_addr = static_cast<T_grad *>(malloc(sizeof(T_grad) * a_mat_size));
  CUST_KERNEL_CHECK_NULLPTR(ctx, casted_a_addr, KERNEL_STATUS_PARAM_INVALID,
                            "[Solve_triangular] Malloc memory [casted_a_array] failed!")
  T_grad *casted_x_addr = static_cast<T_grad *>(malloc(sizeof(T_grad) * x_mat_size));
  CUST_KERNEL_CHECK_NULLPTR(ctx, casted_x_addr, KERNEL_STATUS_PARAM_INVALID,
                            "[Solve_triangular] Malloc memory [casted_x_array] failed!")
  T_grad *casted_dx_addr = static_cast<T_grad *>(malloc(sizeof(T_grad) * dx_mat_size));
  CUST_KERNEL_CHECK_NULLPTR(ctx, casted_dx_addr, KERNEL_STATUS_PARAM_INVALID,
                            "[Solve_triangular] Malloc memory [casted_dx_array] failed!")

  for (size_t i = 0; i < batch; ++i) {
    T_in *a_batch_addr = input_a_addr + i * a_mat_size;
    T_out *x_batch_addr = input_x_addr + i * x_mat_size;
    T_out *dx_batch_addr = input_dx_addr + i * dx_mat_size;
    T_grad *da_batch_addr = output_da_addr + i * da_mat_size;
    T_grad *db_batch_addr = output_db_addr + i * db_mat_size;

    for (size_t i = 0; i < a_mat_size; i++) {
      casted_a_addr[i] = static_cast<T_grad>(a_batch_addr[i]);
    }
    for (size_t i = 0; i < x_mat_size; i++) {
      casted_x_addr[i] = static_cast<T_grad>(x_batch_addr[i]);
    }
    for (size_t i = 0; i < dx_mat_size; i++) {
      casted_dx_addr[i] = static_cast<T_grad>(dx_batch_addr[i]);
    }

    Eigen::Map<Eigen::Matrix<T_grad, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dx(casted_dx_addr, m, n);
    if (trans == kTransT) {
      Eigen::Map<Eigen::Matrix<T_grad, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> a(casted_a_addr, m, m);
      SolveTriangularCpuKernel::solve(a, dx, db_batch_addr, m, n, lower, unit_diagonal);
    } else if (trans == kTransN) {
      Eigen::Map<Eigen::Matrix<T_grad, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> a(casted_a_addr, m, m);
      SolveTriangularCpuKernel::solve(a, dx, db_batch_addr, m, n, !lower, unit_diagonal);
    } else {
      Eigen::Map<Eigen::Matrix<T_grad, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> a(casted_a_addr, m, m);
      SolveTriangularCpuKernel::solve(a.conjugate(), dx, db_batch_addr, m, n, lower, unit_diagonal);
    }
    Eigen::Map<Eigen::Matrix<T_grad, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> x(casted_x_addr, m, n);
    Eigen::Map<Eigen::Matrix<T_grad, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> db(db_batch_addr, m, n);
    Eigen::Map<Eigen::Matrix<T_grad, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> da(da_batch_addr, m, m);
    if (trans == kTransT || trans == kTransC) {
      da = x * db.transpose();
    } else {
      da = db * x.transpose();
    }
    da = -da;
    if (trans == kTransC) {
      da = da.conjugate();
    }
    mat_tri_view(da_batch_addr, m, lower, unit_diagonal);
  }
  free(casted_a_addr);
  free(casted_x_addr);
  free(casted_dx_addr);
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kSolveTriangularGrad, SolveTriangularGradCpuKernel);
}  // namespace aicpu