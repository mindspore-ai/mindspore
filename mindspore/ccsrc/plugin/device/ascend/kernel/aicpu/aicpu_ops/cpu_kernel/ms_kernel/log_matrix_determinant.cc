/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include "cpu_kernel/ms_kernel/log_matrix_determinant.h"

#include <vector>
#include <algorithm>

#include "Eigen/LU"
#include "context/inc/cpu_kernel_utils.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 2;
const uint32_t kInputNum = 1;
const uint32_t kIndexTwo = 2;
const char *const kLogMatrixDeterminant = "LogMatrixDeterminant";
constexpr int64_t kParallelDataNums = 8 * 1024;

#define LOG_MATRIX_DETERMINANT_COMPUTE_CASE(DTYPE, TYPE, CTX)                    \
  case (DTYPE): {                                                                \
    uint32_t result = LogMatrixDeterminantCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                                            \
      CUST_KERNEL_LOG_ERROR(ctx, "LogMatrixDeterminant kernel compute failed."); \
      return result;                                                             \
    }                                                                            \
    break;                                                                       \
  }
}  // namespace

namespace aicpu {
uint32_t LogMatrixDeterminantCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.",
                           kLogMatrixDeterminant);
  CUST_KERNEL_HANDLE_ERROR(ctx, LogMatrixDeterminantCheck(ctx), "[%s] check params failed.", kLogMatrixDeterminant);
  DataType data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    LOG_MATRIX_DETERMINANT_COMPUTE_CASE(DT_FLOAT, float, ctx)
    LOG_MATRIX_DETERMINANT_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    LOG_MATRIX_DETERMINANT_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    LOG_MATRIX_DETERMINANT_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "LogMatrixDeterminant kernel data type [%s] not support.",
                            DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t LogMatrixDeterminantCpuKernel::LogMatrixDeterminantCheck(CpuKernelContext &ctx) {
  auto input_0 = ctx.Input(0);
  auto output_0 = ctx.Output(0);
  auto output_1 = ctx.Output(1);
  CUST_KERNEL_CHECK_NULLPTR(ctx, input_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input x data failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, output_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output sign data failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, output_1->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output y data failed.")

  CUST_KERNEL_CHECK_NULLPTR(ctx, input_0->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID,
                            "Get input x tensor shape failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, output_0->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID,
                            "Get output sign tensor shape failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, output_1->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID,
                            "Get output y tensor shape failed.")
  std::vector<int64_t> shape_x = input_0->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_sign = output_0->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_y = output_1->GetTensorShape()->GetDimSizes();
  size_t shape_size_x = shape_x.size();
  size_t shape_size_sign = shape_sign.size();
  size_t shape_size_y = shape_y.size();
  CUST_KERNEL_CHECK_FALSE(ctx, (shape_size_x > 1), KERNEL_STATUS_PARAM_INVALID,
                          "Input x must be at least rank 2, got [%zu].", shape_size_x)
  CUST_KERNEL_CHECK_FALSE(ctx, (shape_x[shape_size_x - 1] > 0), KERNEL_STATUS_PARAM_INVALID,
                          "Input x last dimension must be at least 1.")
  CUST_KERNEL_CHECK_FALSE(ctx, (shape_x[shape_size_x - kIndexTwo] == shape_x[shape_size_x - 1]),
                          KERNEL_STATUS_PARAM_INVALID, "Input x dimensions must be equal, but are [%lld] and [%lld].",
                          shape_x[shape_size_x - kIndexTwo], shape_x[shape_size_x - 1])

  CUST_KERNEL_CHECK_FALSE(ctx, (shape_size_sign <= shape_size_x - kIndexTwo + 1), KERNEL_STATUS_PARAM_INVALID,
                          "Output sign must be rank [%zu], got [%zu].", shape_size_x - kIndexTwo, shape_size_sign)
  CUST_KERNEL_CHECK_FALSE(ctx, (shape_size_y <= shape_size_x - kIndexTwo + 1), KERNEL_STATUS_PARAM_INVALID,
                          "Output y must be rank [%zu], got [%zu].", shape_size_x - kIndexTwo, shape_size_y)
  for (size_t i = 0; i < shape_size_x - kIndexTwo; i++) {
    CUST_KERNEL_CHECK_FALSE(ctx, (shape_sign[i] == shape_x[i]), KERNEL_STATUS_PARAM_INVALID,
                            "Output sign and Input x dimension [%zu] must be equal, got [%lld] and [%lld].", i,
                            shape_sign[i], shape_x[i])
    CUST_KERNEL_CHECK_FALSE(ctx, (shape_y[i] == shape_x[i]), KERNEL_STATUS_PARAM_INVALID,
                            "Output y and Input x dimension [%zu] must be equal, got [%lld] and [%lld].", i, shape_y[i],
                            shape_x[i])
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t LogMatrixDeterminantCpuKernel::LogMatrixDeterminantCompute(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_sign = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(1)->GetData());

  std::vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  size_t shape_size = shape_x.size();
  int64_t m = shape_x[shape_size - 1];
  int64_t size_mm = m * m;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MartixXd;
  using RealT = typename Eigen::NumTraits<T>::Real;
  if (size_mm > 0) {
    int64_t martix_num = ctx.Input(0)->NumElements() / size_mm;
    int64_t data_size = ctx.Input(0)->NumElements() * static_cast<int64_t>(sizeof(T));
    if (data_size <= kParallelDataNums) {
      for (int64_t i = 0; i < martix_num; i++) {
        RealT log_abs_det = 0;
        T sign = 1;
        Eigen::Map<MartixXd> martix_x(input_x + i * m * m, m, m);
        if (martix_x.size() > 0) {
          Eigen::PartialPivLU<MartixXd> lu(martix_x);
          MartixXd LU = lu.matrixLU();
          sign = lu.permutationP().determinant();
          auto diag = LU.diagonal().array().eval();
          auto abs_diag = diag.cwiseAbs().eval();
          log_abs_det += abs_diag.log().sum();
          sign *= (diag / abs_diag).prod();
        }
        if (!Eigen::numext::isfinite(log_abs_det)) {
          sign = 0;
          log_abs_det = log_abs_det > 0 ? -std::log(RealT(0)) : std::log(RealT(0));
        }
        *(output_sign + i) = sign;
        *(output_y + i) = log_abs_det;
      }
    } else {
      uint32_t min_core_num = 1;
      int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
      if (max_core_num > martix_num) {
        max_core_num = martix_num;
      }
      auto shard_work = [&](size_t start, size_t end) {
        RealT log_abs_det = 0;
        for (size_t i = start; i < end; i++) {
          log_abs_det = 0;
          T sign = 1;
          Eigen::Map<MartixXd> martix_x(input_x + i * m * m, m, m);
          if (martix_x.size() > 0) {
            Eigen::PartialPivLU<MartixXd> lu(martix_x);
            MartixXd LU = lu.matrixLU();
            sign = static_cast<T>(lu.permutationP().determinant());
            auto diag = LU.diagonal().array().eval();
            auto abs_diag = diag.cwiseAbs().eval();
            log_abs_det += abs_diag.log().sum();
            sign *= (diag / abs_diag).prod();
          }
          if (!Eigen::numext::isfinite(log_abs_det)) {
            sign = 0;
            log_abs_det = log_abs_det > 0 ? -std::log(RealT(0)) : std::log(RealT(0));
          }
          *(output_sign + i) = sign;
          *(output_y + i) = log_abs_det;
        }
      };
      CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, martix_num, martix_num / max_core_num, shard_work),
                               "LogMatrixDeterminant Compute failed.");
    }
  }
  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kLogMatrixDeterminant, LogMatrixDeterminantCpuKernel);
}  // namespace aicpu
