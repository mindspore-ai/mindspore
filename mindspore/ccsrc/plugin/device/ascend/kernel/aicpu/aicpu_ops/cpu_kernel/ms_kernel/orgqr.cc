/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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

#include "orgqr.h"

#include "Eigen/Dense"

#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include <numeric>
#include <iostream>

using namespace Eigen;
namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kOrgqr = "Orgqr";
const double ZERO = 0.;
const uint32_t kTWO = 2;
constexpr int64_t kParallelDataNums = 18 * 1024;
constexpr int64_t kParallelDataNumsMid = 32 * 1024;

#define ORGQR_COMPUTE(DTYPE, TYPE, CTX)                           \
  case (DTYPE): {                                                 \
    uint32_t result = OrgqrCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                             \
      CUST_KERNEL_LOG_ERROR(ctx, "Orgqr kernel compute failed."); \
      return result;                                              \
    }                                                             \
    break;                                                        \
  }
#define ORGQR_COMPUTE_COMPLEX(DTYPE, TYPE, CTX)                   \
  case (DTYPE): {                                                 \
    uint32_t result = OrgqrComputeComplex<TYPE>(CTX);             \
    if (result != KERNEL_STATUS_OK) {                             \
      CUST_KERNEL_LOG_ERROR(ctx, "Orgqr kernel compute failed."); \
      return result;                                              \
    }                                                             \
    break;                                                        \
  }
}  // namespace

namespace aicpu {
uint32_t OrgqrCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "Orgqr check input and output number failed.");
  CUST_KERNEL_HANDLE_ERROR(ctx, OrgqrCheck(ctx), "[%s] check params failed.", kOrgqr);
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    ORGQR_COMPUTE(DT_FLOAT, float, ctx)
    ORGQR_COMPUTE(DT_DOUBLE, double, ctx)
    ORGQR_COMPUTE_COMPLEX(DT_COMPLEX64, std::complex<float_t>, ctx)
    ORGQR_COMPUTE_COMPLEX(DT_COMPLEX128, std::complex<double_t>, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Orgqr kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t OrgqrCpuKernel::OrgqrCheck(CpuKernelContext &ctx) {
  std::vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  size_t shape_size = shape_x.size();
  CUST_KERNEL_CHECK_FALSE(ctx, (shape_size > 1), KERNEL_STATUS_PARAM_INVALID, "Input x must be at least rank 2.")
  CUST_KERNEL_CHECK_FALSE(ctx, (shape_x[shape_size - kTWO] > 0), KERNEL_STATUS_PARAM_INVALID,
                          "Dimension [%zu] of input x must be at least 1, but [%zu].", shape_size - kTWO,
                          shape_x[shape_size - kTWO])
  CUST_KERNEL_CHECK_FALSE(ctx, (shape_x[shape_size - 1] > 0), KERNEL_STATUS_PARAM_INVALID,
                          "Dimension [%zu] of input x must be at least 1, but [%zu].", shape_size - 1,
                          shape_x[shape_size - 1])
  CUST_KERNEL_CHECK_FALSE(
    ctx, (shape_x[shape_size - kTWO] >= shape_x[shape_size - 1]), KERNEL_STATUS_PARAM_INVALID,
    "Dimension [%zu] of input x must be bigger than dimension [%zu], when input x has rank [%zu].", shape_size - kTWO,
    shape_size - 1, shape_size)
  std::vector<int64_t> shape_tau = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  size_t shape_tau_size = shape_tau.size();
  CUST_KERNEL_CHECK_FALSE(ctx, (shape_x[shape_size - 1] >= shape_tau[shape_tau_size - 1]), KERNEL_STATUS_PARAM_INVALID,
                          "Dimension [%zu] of input tau must be less than [%zu], but [%zu].", shape_tau_size - 1,
                          shape_x[shape_size - 1], shape_tau[shape_tau_size - 1])
  if (shape_size > kTWO) {
    CUST_KERNEL_CHECK_FALSE(ctx, (shape_x[0] == shape_tau[0]), KERNEL_STATUS_PARAM_INVALID,
                            "Dimension 0 of input tau must equal Dimension 0 of input x when input has batch")
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t OrgqrCpuKernel::OrgqrCompute(CpuKernelContext &ctx) {
  auto *x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto *tau = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto *y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  std::vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  size_t shape_size = shape_x.size();
  size_t m = shape_x[shape_size - kTWO];
  size_t n = shape_x[shape_size - 1];
  std::vector<int64_t> shape_tau = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  size_t p = *(shape_tau.end() - 1);
  size_t size_mn = m * n;
  size_t matrix_num = ctx.Input(0)->NumElements() / size_mn;
  int64_t data_size = ctx.Input(0)->NumElements() * sizeof(T);
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MartrixXd;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXd;
  if (data_size <= kParallelDataNums) {
    for (size_t i = 0; i < matrix_num; i++) {
      Eigen::Map<MartrixXd> martrix_y(y + i * m * n, m, n);
      Eigen::Map<MartrixXd> martrix_x(x + i * m * n, m, n);
      MartrixXd tmp = MartrixXd::Identity(m, m);
      Eigen::Map<VectorXd> vector_tau(tau + i * p, p, 1);
      for (size_t k = 0; k < p; k++) {
        VectorXd vector_v = martrix_x.block(k, k, m - k, 1);
        vector_v[0] = 1;
        tmp.rightCols(m - k) =
          tmp.rightCols(m - k) - vector_tau(k) * (tmp.rightCols(m - k) * vector_v) * vector_v.transpose();
      }
      martrix_y = tmp.leftCols(n);
    }
  } else {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (data_size <= kParallelDataNumsMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }
    if (max_core_num > matrix_num) {
      max_core_num = matrix_num;
    }
    auto shard_qr = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        Eigen::Map<MartrixXd> martrix_y(y + i * m * n, m, n);
        Eigen::Map<MartrixXd> martrix_x(x + i * m * n, m, n);
        MartrixXd tmp = MartrixXd::Identity(m, m);
        Eigen::Map<VectorXd> vector_tau(tau + i * p, p, 1);
        for (size_t k = 0; k < p; k++) {
          VectorXd vector_v = martrix_x.block(k, k, m - k, 1);
          vector_v[0] = 1;
          tmp.rightCols(m - k) =
            tmp.rightCols(m - k) - vector_tau(k) * (tmp.rightCols(m - k) * vector_v) * vector_v.transpose();
        }
        martrix_y = tmp.leftCols(n);
      }
    };
    if (max_core_num == 0) {
      CUST_KERNEL_LOG_ERROR(ctx, "max_core_num could not be 0.");
    }
    CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, matrix_num, matrix_num / max_core_num, shard_qr),
                             "Orgqr Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t OrgqrCpuKernel::OrgqrComputeComplex(CpuKernelContext &ctx) {
  auto *x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto *tau = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto *y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  std::vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  size_t shape_size = shape_x.size();
  size_t m = shape_x[shape_size - kTWO];
  size_t n = shape_x[shape_size - 1];
  std::vector<int64_t> shape_tau = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  size_t p = *(shape_tau.end() - 1);
  size_t size_mn = m * n;
  size_t matrix_num = ctx.Input(0)->NumElements() / size_mn;
  int64_t data_size = ctx.Input(0)->NumElements() * sizeof(T);
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MartrixXd;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXd;
  if (data_size <= kParallelDataNums) {
    for (size_t i = 0; i < matrix_num; i++) {
      Eigen::Map<MartrixXd> martrix_y(y + i * m * n, m, n);
      Eigen::Map<MartrixXd> martrix_x(x + i * m * n, m, n);
      MartrixXd tmp = MartrixXd::Identity(m, m);
      Eigen::Map<VectorXd> vector_tau(tau + i * p, p, 1);
      for (size_t k = 0; k < p; k++) {
        VectorXd vector_v = martrix_x.block(k, k, m - k, 1);
        vector_v[0] = 1;
        tmp.rightCols(m - k) =
          tmp.rightCols(m - k) - vector_tau(k) * (tmp.rightCols(m - k) * vector_v) * vector_v.adjoint();
      }
      martrix_y = tmp.leftCols(n);
    }
  } else {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (data_size <= kParallelDataNumsMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }
    if (max_core_num > matrix_num) {
      max_core_num = matrix_num;
    }
    auto shard_qr = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        Eigen::Map<MartrixXd> martrix_y(y + i * m * n, m, n);
        Eigen::Map<MartrixXd> martrix_x(x + i * m * n, m, n);
        MartrixXd tmp = MartrixXd::Identity(m, m);
        Eigen::Map<VectorXd> vector_tau(tau + i * p, p, 1);
        for (size_t k = 0; k < p; k++) {
          VectorXd vector_v = martrix_x.block(k, k, m - k, 1);
          vector_v[0] = 1;
          tmp.rightCols(m - k) =
            tmp.rightCols(m - k) - vector_tau(k) * (tmp.rightCols(m - k) * vector_v) * vector_v.adjoint();
        }
        martrix_y = tmp.leftCols(n);
      }
    };
    if (max_core_num == 0) {
      CUST_KERNEL_LOG_ERROR(ctx, "max_core_num could not be 0.");
    }
    CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, matrix_num, matrix_num / max_core_num, shard_qr),
                             "Orgqr Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kOrgqr, OrgqrCpuKernel);
}  // namespace aicpu
