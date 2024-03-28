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
#include "solve_triangular.h"
#include <cstdint>
#include <string.h>
#include "securec.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kSolveTriangular = "SolveTriangular";
constexpr uint32_t kSolveTriangularInputsNum = 5;
constexpr uint32_t kSolveTriangularOutputsNum = 1;
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
}  // namespace

namespace aicpu {

uint32_t SolveTriangularCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kSolveTriangularInputsNum, kSolveTriangularOutputsNum),
                           "[%s] check input and output failed.", kSolveTriangular);
  CUST_KERNEL_HANDLE_ERROR(ctx, SolveTriangularCheck(ctx), "[%s] check params failed.", kSolveTriangular);
  auto a_type = ctx.Input(0)->GetDataType();
  uint32_t ret = KERNEL_STATUS_OK;
  switch (a_type) {
    case DT_FLOAT:
      ret = SolveTriangularCompute<float, float>(ctx);
      break;
    case DT_DOUBLE:
      ret = SolveTriangularCompute<double, double>(ctx);
      break;
    case DT_FLOAT16:
      ret = SolveTriangularCompute<Eigen::half, Eigen::half>(ctx);
      break;
    case DT_INT8:
      ret = SolveTriangularCompute<int8_t, float>(ctx);
      break;
    case DT_INT16:
      ret = SolveTriangularCompute<int16_t, float>(ctx);
      break;
    case DT_INT32:
      ret = SolveTriangularCompute<int32_t, float>(ctx);
      break;
    case DT_INT64:
      ret = SolveTriangularCompute<int64_t, double>(ctx);
      break;
    case DT_COMPLEX64:
      ret = SolveTriangularCompute<std::complex<float>, std::complex<float>>(ctx);
      break;
    case DT_COMPLEX128:
      ret = SolveTriangularCompute<std::complex<double>, std::complex<double>>(ctx);
      break;
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "[%s] Data type of input is not support, input data type is [%s].",
                            ctx.GetOpType().c_str(), DTypeStr(a_type).c_str());
      ret = KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
}

uint32_t SolveTriangularCpuKernel::SolveTriangularCheck(CpuKernelContext &ctx) {
  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.Input(kIndexA)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                            "Get input 'a' data failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.Input(kIndexB)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                            "Get input 'b' data failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.Input(kIndexTrans)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                            "Get input 'trans' data failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.Input(kIndexLower)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                            "Get input 'lower' data failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.Input(kIndexUnitDiagonal)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                            "Get input UnitDiagonal data failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.Output(kIndexX)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                            "Get output 'x' data failed.")

  std::vector<int64_t> a_shape = ctx.Input(kIndexA)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> b_shape = ctx.Input(kIndexB)->GetTensorShape()->GetDimSizes();
  auto a_rank = a_shape.size();
  auto b_rank = b_shape.size();
  const size_t expected_b_dim = (b_shape.size() == a_shape.size() - 1) ? 1 : kSquareSize;
  if (a_rank < kSquareSize) {
    CUST_KERNEL_LOG_ERROR(ctx, "For [%s], dim of matrix a must greater or equal to 2, but got a at [%lld]-dimensional.",
                          kSolveTriangular, a_rank);
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (a_rank != b_rank && a_rank != b_rank + 1) {
    CUST_KERNEL_LOG_ERROR(
      ctx,
      "For [%s], the dimension of `b` should be 'a.dim' or 'a.dim' - 1, which is [%lld] or [%lld], but got "
      "[%lld]-dimensions.",
      kSolveTriangular, a_rank, a_rank - 1, b_rank);
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (a_shape[a_rank - 1] != a_shape[a_rank - kSquareSize]) {
    CUST_KERNEL_LOG_ERROR(
      ctx,
      "For [%s], the last two dimensions of `a` should be the same, but got shape of [%s]. Please make sure that the "
      "shape of `a` be like [..., N, N].",
      kSolveTriangular, VectorToString(a_shape));
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (a_shape[a_rank - kSquareSize] != b_shape[b_rank - expected_b_dim]) {
    CUST_KERNEL_LOG_ERROR(
      ctx,
      "For [%s], the last two dimensions of `a` and `b` should be matched, but got shape of [%s] and [%s]. Please make "
      "sure that the shape of `a` and `b` be like [..., N, N] X [..., N, M] or [..., N, N ] X[..., N].",
      kSolveTriangular, VectorToString(a_shape), VectorToString(b_shape));
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (!std::equal(a_shape.begin(), a_shape.begin() + (a_rank - kSquareSize), b_shape.begin(),
                  b_shape.begin() + (b_rank - expected_b_dim))) {
    CUST_KERNEL_LOG_ERROR(
      ctx,
      "For [%s], the batch dimensions of `a` and `b` should all be the same, but got shape of [%s] and [%s]. Please "
      "make sure that the shape of `a` and `b` be like [a, b, c, ..., N, N] X [a, b, c, ..., N, M] or [a, b, c, ..., "
      "N, N] X [a, b, c, ..., N].",
      kSolveTriangular, VectorToString(a_shape), VectorToString(b_shape));
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

template <typename T_in, typename T_out>
uint32_t SolveTriangularCpuKernel::SolveTriangularCompute(CpuKernelContext &ctx) {
  auto input_a_addr = reinterpret_cast<T_in *>(ctx.Input(kIndexA)->GetData());
  auto input_b_addr = reinterpret_cast<T_in *>(ctx.Input(kIndexB)->GetData());
  auto output_x_addr = reinterpret_cast<T_out *>(ctx.Output(kIndexX)->GetData());
  const std::map<std::string, DataType> trans_type = {{"trans", ctx.Input(kIndexTrans)->GetDataType()}};

  int64_t *trans_ptr = reinterpret_cast<int64_t *>(ctx.Input(kIndexTrans)->GetData());
  int64_t trans = *trans_ptr;
  bool *lower_ptr = reinterpret_cast<bool *>(ctx.Input(kIndexLower)->GetData());
  bool lower = *lower_ptr;
  bool *unit_diagonal_ptr = reinterpret_cast<bool *>(ctx.Input(kIndexUnitDiagonal)->GetData());
  bool unit_diagonal = *unit_diagonal_ptr;

  std::vector<int64_t> shape_a = ctx.Input(kIndexA)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_b = ctx.Input(kIndexB)->GetTensorShape()->GetDimSizes();
  size_t a_dims = shape_a.size();
  size_t b_dims = shape_b.size();
  size_t m = shape_a[a_dims - kSquareSize];
  size_t n = (b_dims == a_dims - 1) ? 1 : shape_b[b_dims - 1];
  size_t batch = std::accumulate(shape_a.begin(), shape_a.end() - kSquareSize, size_t(1), std::multiplies{});

  size_t a_mat_size = m * m;
  size_t b_mat_size = m * n;
  size_t output_mat_size = b_mat_size;

  T_out *casted_a_addr = static_cast<T_out *>(malloc(sizeof(T_out) * a_mat_size));
  CUST_KERNEL_CHECK_NULLPTR(ctx, casted_a_addr, KERNEL_STATUS_PARAM_INVALID,
                            "[Solve_triangular] Malloc memory [casted_a_array] failed!")
  T_out *casted_b_addr = static_cast<T_out *>(malloc(sizeof(T_out) * b_mat_size));
  CUST_KERNEL_CHECK_NULLPTR(ctx, casted_b_addr, KERNEL_STATUS_PARAM_INVALID,
                            "[Solve_triangular] Malloc memory [casted_b_array] failed!")

  for (size_t i = 0; i < batch; ++i) {
    T_in *a_batch_addr = input_a_addr + i * a_mat_size;
    T_in *b_batch_addr = input_b_addr + i * b_mat_size;
    T_out *output_batch_addr = output_x_addr + i * output_mat_size;

    for (size_t j = 0; j < a_mat_size; j++) {
      casted_a_addr[j] = static_cast<T_out>(a_batch_addr[j]);
    }
    for (size_t j = 0; j < b_mat_size; j++) {
      casted_b_addr[j] = static_cast<T_out>(b_batch_addr[j]);
    }

    Eigen::Map<Eigen::Matrix<T_out, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> b(casted_b_addr, m, n);
    if (trans == kTransT) {
      Eigen::Map<Eigen::Matrix<T_out, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> a(casted_a_addr, m, m);
      solve(a, b, output_batch_addr, m, n, !lower, unit_diagonal);
    } else if (trans == kTransN) {
      Eigen::Map<Eigen::Matrix<T_out, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> a(casted_a_addr, m, m);
      solve(a, b, output_batch_addr, m, n, lower, unit_diagonal);
    } else if (trans == kTransC) {
      Eigen::Map<Eigen::Matrix<T_out, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> a(casted_a_addr, m, m);
      solve(a.conjugate(), b, output_batch_addr, m, n, !lower, unit_diagonal);
    } else {
      CUST_KERNEL_LOG_ERROR(ctx, "For 'SolveTirangular' 'trans' must be in [0, 1, 2, 'N', 'T', 'C'], but got [%d].",
                            trans);
    }
  }

  free(casted_a_addr);
  free(casted_b_addr);

  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kSolveTriangular, SolveTriangularCpuKernel);
}  // namespace aicpu