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
#include "ms_kernel/qr.h"
#include <algorithm>
#include <vector>
#include <utility>
#include "Eigen/Dense"
#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 2;
const uint32_t kInputNum = 1;
const char *kQr = "Qr";
constexpr int64_t kParallelDataNums = 16 * 1024;
constexpr int64_t kParallelDataNumsMid = 128 * 1024;

#define QR_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                              \
    uint32_t result = QrCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                          \
      CUST_KERNEL_LOG_ERROR(ctx, "Qr kernel compute failed."); \
      return result;                                           \
    }                                                          \
    break;                                                     \
  }
}  // namespace

namespace aicpu {
uint32_t QrCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kQr);
  CUST_KERNEL_HANDLE_ERROR(ctx, QrCheck(ctx), "[%s] check params failed.", kQr);
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    QR_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    QR_COMPUTE_CASE(DT_FLOAT, float, ctx)
    QR_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    QR_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    QR_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Qr kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t QrCpuKernel::QrCheck(CpuKernelContext &ctx) {
  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input data failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output 0 data failed")
  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.Output(1)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output 1 data failed")
  auto attr_full_matrices = ctx.GetAttr("full_matrices");
  CUST_KERNEL_CHECK_NULLPTR(ctx, attr_full_matrices, KERNEL_STATUS_PARAM_INVALID, "Get full_matrices attr failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.Input(0)->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID,
                            "Get input tensor shape failed.")
  std::vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  size_t shape_size = shape_x.size();
  CUST_KERNEL_CHECK_FALSE(ctx, (shape_size > 1), KERNEL_STATUS_PARAM_INVALID, "Input x must be at least rank 2.")
  CUST_KERNEL_CHECK_FALSE(ctx, (shape_x[shape_size - 2] > 0), KERNEL_STATUS_PARAM_INVALID,
                          "Dimension [%zu] must be at least 1, but [%zu].", shape_size - 2, shape_x[shape_size - 2])
  CUST_KERNEL_CHECK_FALSE(ctx, (shape_x[shape_size - 1] > 0), KERNEL_STATUS_PARAM_INVALID,
                          "Dimension [%zu] must be at least 1, but [%zu].", shape_size - 1, shape_x[shape_size - 1])

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t QrCpuKernel::QrCompute(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_q = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto output_r = reinterpret_cast<T *>(ctx.Output(1)->GetData());
  auto attr_full_matrices = ctx.GetAttr("full_matrices");
  bool full_matrices = attr_full_matrices->GetBool();

  std::vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  size_t shape_size = shape_x.size();
  int64_t m = shape_x[shape_size - 2];
  int64_t n = shape_x[shape_size - 1];
  int64_t p = std::min(m, n);
  int64_t size_mn = m * n;
  int64_t size_mm = m * m;
  int64_t size_mp = m * p;
  int64_t size_pn = p * n;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MartixXd;
  if (size_mn > 0) {
    size_t martix_num = ctx.Input(0)->NumElements() / size_mn;
    int64_t data_size = ctx.Input(0)->NumElements() * sizeof(T);
    if (data_size <= kParallelDataNums) {
      for (size_t i = 0; i < martix_num; i++) {
        Eigen::Map<MartixXd> martix_x(input_x + i * size_mn, m, n);
        Eigen::HouseholderQR<MartixXd> qr(martix_x);
        if (full_matrices) {
          Eigen::Map<MartixXd> martix_q(output_q + i * size_mm, m, m);
          Eigen::Map<MartixXd> martix_r(output_r + i * size_mn, m, n);
          martix_q = qr.householderQ();
          martix_r = qr.matrixQR().template triangularView<Eigen::Upper>();
        } else {
          Eigen::Map<MartixXd> martix_q(output_q + i * size_mp, m, p);
          Eigen::Map<MartixXd> martix_r(output_r + i * size_pn, p, n);
          MartixXd tmp = MartixXd::Identity(m, p);
          martix_q = qr.householderQ() * tmp;
          auto qr_top = qr.matrixQR().block(0, 0, p, n);
          martix_r = qr_top.template triangularView<Eigen::Upper>();
        }
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
      auto shard_qr = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
          Eigen::Map<MartixXd> martix_x(input_x + i * size_mn, m, n);
          Eigen::HouseholderQR<MartixXd> qr(martix_x);
          if (full_matrices) {
            Eigen::Map<MartixXd> martix_q(output_q + i * size_mm, m, m);
            Eigen::Map<MartixXd> martix_r(output_r + i * size_mn, m, n);
            martix_q = qr.householderQ();
            martix_r = qr.matrixQR().template triangularView<Eigen::Upper>();
          } else {
            Eigen::Map<MartixXd> martix_q(output_q + i * size_mp, m, p);
            Eigen::Map<MartixXd> martix_r(output_r + i * size_pn, p, n);
            MartixXd tmp = MartixXd::Identity(m, p);
            martix_q = qr.householderQ() * tmp;
            auto qr_top = qr.matrixQR().block(0, 0, p, n);
            martix_r = qr_top.template triangularView<Eigen::Upper>();
          }
        }
      };
      CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, martix_num, martix_num / max_core_num, shard_qr),
                               "Qr Compute failed.");
    }
  }
  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kQr, QrCpuKernel);
}  // namespace aicpu
