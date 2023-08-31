/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "cpu_kernel/ms_kernel/lu.h"

#include <algorithm>
#include <vector>

#include "Eigen/LU"
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 2;
const uint32_t kInputNum = 1;
const char *kLu = "Lu";
constexpr int64_t kParallelDataNums = 32 * 1024;
constexpr int64_t kParallelDataNumsMid = 256 * 1024;

#define LU_COMPUTE_CASE(IN_DTYPE, IN_TYPE, OUT_DTYPE, CTX)                                                 \
  case (IN_DTYPE): {                                                                                       \
    switch (OUT_DTYPE) {                                                                                   \
      case (DT_INT32): {                                                                                   \
        uint32_t result = LuCompute<IN_TYPE, int32_t>(CTX);                                                \
        if (result != KERNEL_STATUS_OK) {                                                                  \
          KERNEL_LOG_ERROR("Lu kernel compute failed.");                                                   \
          return result;                                                                                   \
        }                                                                                                  \
        break;                                                                                             \
      }                                                                                                    \
      case (DT_INT64): {                                                                                   \
        uint32_t result = LuCompute<IN_TYPE, int64_t>(CTX);                                                \
        if (result != KERNEL_STATUS_OK) {                                                                  \
          KERNEL_LOG_ERROR("Lu kernel compute failed.");                                                   \
          return result;                                                                                   \
        }                                                                                                  \
        break;                                                                                             \
      }                                                                                                    \
      default:                                                                                             \
        KERNEL_LOG_ERROR("Lu kernel output p data type [%s] not support.", DTypeStr(output_type).c_str()); \
        return KERNEL_STATUS_PARAM_INVALID;                                                                \
    }                                                                                                      \
    break;                                                                                                 \
  }
}  // namespace

namespace aicpu {
uint32_t LuCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kLu);
  KERNEL_HANDLE_ERROR(LuCheck(ctx), "[%s] check params failed.", kLu);
  DataType input_type = ctx.Input(0)->GetDataType();
  DataType output_type = ctx.Output(1)->GetDataType();
  switch (input_type) {
    LU_COMPUTE_CASE(DT_FLOAT, float, output_type, ctx)
    LU_COMPUTE_CASE(DT_DOUBLE, double, output_type, ctx)
    LU_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, output_type, ctx)
    LU_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, output_type, ctx)
    default:
      KERNEL_LOG_ERROR("Lu kernel input data type [%s] not support.", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t LuCpuKernel::LuCheck(const CpuKernelContext &ctx) {
  auto input_0 = ctx.Input(0);
  auto output_0 = ctx.Output(0);
  auto output_1 = ctx.Output(1);
  KERNEL_CHECK_NULLPTR(input_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input data failed.");
  KERNEL_CHECK_NULLPTR(output_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output 0 data failed");
  KERNEL_CHECK_NULLPTR(output_1->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output 1 data failed");

  KERNEL_CHECK_NULLPTR(input_0->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID, "Get input tensor shape failed.");
  std::vector<int64_t> shape_x = input_0->GetTensorShape()->GetDimSizes();
  size_t shape_size = shape_x.size();
  KERNEL_CHECK_FALSE((shape_size > 1), KERNEL_STATUS_PARAM_INVALID, "Input must be at least rank 2, got [%zu].",
                     shape_x.size());
  KERNEL_CHECK_FALSE((shape_x[shape_size - 2] == shape_x[shape_size - 1]), KERNEL_STATUS_PARAM_INVALID,
                     "Dimensions must be equal, but are [%zu] and [%zu].", shape_x[shape_size - 2],
                     shape_x[shape_size - 1]);

  return KERNEL_STATUS_OK;
}

template <typename Scalar, typename Tidx>
uint32_t LuCpuKernel::LuCompute(const CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<Scalar *>(ctx.Input(0)->GetData());
  auto output_lu = reinterpret_cast<Scalar *>(ctx.Output(0)->GetData());
  auto output_p = reinterpret_cast<Tidx *>(ctx.Output(1)->GetData());

  std::vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  size_t shape_size = shape_x.size();
  int64_t n = shape_x[shape_size - 1];
  int64_t size_nn = n * n;

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  typedef Eigen::Matrix<Tidx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Indices;

  if (size_nn > 0) {
    size_t martix_num = ctx.Input(0)->NumElements() / size_nn;
    int64_t data_size = ctx.Input(0)->NumElements() * sizeof(Scalar);
    if (data_size <= kParallelDataNums) {
      Eigen::PartialPivLU<Matrix> lu;
      for (size_t i = 0; i < martix_num; i++) {
        Eigen::Map<Matrix> martix_x(input_x + i * size_nn, n, n);
        Eigen::Map<Matrix> martix_lu(output_lu + i * size_nn, n, n);
        Eigen::Map<Indices> martix_p(output_p + i * n, n, 1);
        lu.compute(martix_x);
        martix_lu = lu.matrixLU();
        Eigen::PermutationMatrix<-1, -1, Tidx> permutation = lu.permutationP().transpose();
        martix_p = permutation.indices();
        RealScalar min_abs_pivot = martix_lu.diagonal().cwiseAbs().minCoeff();
        KERNEL_CHECK_FALSE((min_abs_pivot > RealScalar(0)), KERNEL_STATUS_PARAM_INVALID, "Input is not invertible.");
      }
    } else {
      uint32_t min_core_num = 1;
      uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
      if (data_size <= kParallelDataNumsMid) {
        max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
      }
      if (max_core_num > martix_num) {
        max_core_num = martix_num;
      }
      auto shard_lu = [&](size_t start, size_t end) {
        Eigen::PartialPivLU<Matrix> lu;
        for (size_t i = start; i < end; i++) {
          Eigen::Map<Matrix> martix_x(input_x + i * size_nn, n, n);
          Eigen::Map<Matrix> martix_lu(output_lu + i * size_nn, n, n);
          Eigen::Map<Indices> martix_p(output_p + i * n, n, 1);
          lu.compute(martix_x);
          martix_lu = lu.matrixLU();
          Eigen::PermutationMatrix<-1, -1, Tidx> permutation = lu.permutationP().transpose();
          martix_p = permutation.indices();
          RealScalar min_abs_pivot = martix_lu.diagonal().cwiseAbs().minCoeff();
          KERNEL_CHECK_FALSE((min_abs_pivot > RealScalar(0)), KERNEL_STATUS_PARAM_INVALID, "Input is not invertible.");
        }
        return KERNEL_STATUS_OK;
      };
      KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, martix_num, martix_num / max_core_num, shard_lu),
                          "Lu Compute failed.");
    }
  }

  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kLu, LuCpuKernel);
}  // namespace aicpu
