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

#include "cpu_kernel/ms_kernel/lstsqv2_grad.h"

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <iostream>

#include "securec/include/securec.h"
#include "context/inc/cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "utils/eigen_tensor.h"
#include "utils/bcast.h"

namespace {
const uint32_t kOutputNum = 2;
const uint32_t kInputNum = 3;
constexpr size_t kIndexGX = 0;
constexpr size_t kIndexA = 1;
constexpr size_t kIndexB = 2;
constexpr size_t kIndexGA = 0;
constexpr size_t kIndexGB = 1;
const char *kLstsqV2Grad = "LstsqV2Grad";
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace
// namespace aicpu
namespace aicpu {
uint32_t LstsqV2GradCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "Lstsq check input and output numberfailed.");
  DataType data_type = ctx.Input(kIndexA)->GetDataType();
  switch (data_type) {
    case DT_FLOAT:
      return LstsqV2GradCompute<float>(ctx);
    case DT_DOUBLE:
      return LstsqV2GradCompute<double>(ctx);
    case DT_COMPLEX64:
      return LstsqV2GradCompute<complex64>(ctx);
    case DT_COMPLEX128:
      return LstsqV2GradCompute<complex128>(ctx);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "LstsqV2Grad kernel data type [%u] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
void Pinv(T *input_addr, T *output_addr, size_t row, size_t col) {
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> in(input_addr, row, col);
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> out(output_addr, col, row);
  auto svd = in.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> s = svd.singularValues();
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> s_inv(col, row);
  s_inv.setZero();
  size_t s_size = row < col ? row : col;
  for (size_t i = 0; i < s_size; i++) {
    s_inv(i, i) = static_cast<T>(1) / s(i);
  }
  out = svd.matrixV() * s_inv * svd.matrixU().transpose().conjugate();
}

template <typename T>
void pinv_backward(T *g_pinv_addr, T *pinv_a_addr, T *a_addr, T *g_a_addr, size_t row, size_t col) {
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> g_pinv_a(g_pinv_addr, col, row);
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> pinv_a(pinv_a_addr, col, row);
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> a(a_addr, row, col);
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> g_a(g_a_addr, row, col);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pinv_a_h = pinv_a.transpose().conjugate();
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> g_pinv_a_h = g_pinv_a.transpose().conjugate();
  if (row <= col) {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> K = g_pinv_a_h * pinv_a;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> K_pinv_a_h = K * pinv_a_h;
    g_a = -(pinv_a * K).transpose().conjugate() + K_pinv_a_h - (a * pinv_a) * K_pinv_a_h +
          (pinv_a_h * pinv_a) * (g_pinv_a_h - K * a);
  } else {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> K = pinv_a * g_pinv_a_h;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pinv_a_h_K = pinv_a_h * K;
    g_a = -(K * pinv_a).transpose().conjugate() + (g_pinv_a_h - a * K) * pinv_a * pinv_a_h + pinv_a_h_K -
          pinv_a_h_K * pinv_a * a;
  }
}

template <typename T>
void memadd(T *source_addr, T *target_addr, size_t len) {
  for (size_t i = 0; i < len; i++) {
    target_addr[i] += source_addr[i];
  }
}

template <typename T>
uint32_t LstsqV2GradCpuKernel::LstsqV2GradCompute(CpuKernelContext &ctx) {
  T *gx_addr = reinterpret_cast<T *>(ctx.Input(kIndexGX)->GetData());
  T *a_addr = reinterpret_cast<T *>(ctx.Input(kIndexA)->GetData());
  T *b_addr = reinterpret_cast<T *>(ctx.Input(kIndexB)->GetData());
  T *ga_addr = reinterpret_cast<T *>(ctx.Output(kIndexGA)->GetData());
  T *gb_addr = reinterpret_cast<T *>(ctx.Output(kIndexGB)->GetData());

  std::vector<int64_t> shape_gx = ctx.Input(kIndexGX)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_a = ctx.Input(kIndexA)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_b = ctx.Input(kIndexB)->GetTensorShape()->GetDimSizes();
  size_t a_dims = shape_a.size();
  size_t b_dims = shape_b.size();
  int64_t batch = 1;
  int64_t size_a = 1;
  int64_t size_b = 1;
  size_t batch_dims = a_dims - 2;
  std::vector<int64_t> batch_shape_a(batch_dims);
  std::vector<int64_t> batch_shape_b(batch_dims);
  std::vector<int64_t> broadcast_batch_shape(batch_dims);
  for (size_t i = 0; i < batch_dims; i++) {
    batch_shape_a[i] = shape_a[i];
    batch_shape_b[i] = shape_b[i];
    broadcast_batch_shape[i] = shape_gx[i];
    batch *= shape_gx[i];
    size_a *= shape_a[i];
    size_b *= shape_b[i];
  }

  int64_t m = shape_a[a_dims - 2];
  int64_t n = shape_a[a_dims - 1];
  int64_t k = a_dims == b_dims ? shape_b[b_dims - 1] : 1;
  int64_t a_mat_size = m * n;
  int64_t b_mat_size = m * k;
  int64_t gx_mat_size = n * k;
  size_a = a_mat_size * size_a;
  size_b = b_mat_size * size_b;
  if (size_a == 0 || size_b == 0) return KERNEL_STATUS_OK;

  Bcast bcast_a(ctx, batch_shape_a, broadcast_batch_shape);
  Bcast bcast_b(ctx, batch_shape_b, broadcast_batch_shape);

  T *pinv_a_addr = static_cast<T *>(malloc(sizeof(T) * a_mat_size));
  CUST_KERNEL_CHECK_NULLPTR(ctx, pinv_a_addr, KERNEL_STATUS_INNER_ERROR,
                            "[LstsqV2] Malloc memory [pinv_a_addr] failed!")
  T *g_pinv_a_addr = static_cast<T *>(malloc(sizeof(T) * a_mat_size));
  CUST_KERNEL_CHECK_NULLPTR(ctx, g_pinv_a_addr, KERNEL_STATUS_INNER_ERROR,
                            "[LstsqV2] Malloc memory [g_pinv_a_addr] failed!")
  T *ga_temp_addr = static_cast<T *>(malloc(sizeof(T) * a_mat_size));
  CUST_KERNEL_CHECK_NULLPTR(ctx, ga_temp_addr, KERNEL_STATUS_INNER_ERROR,
                            "[LstsqV2] Malloc memory [ga_temp_addr] failed!")
  T *gb_temp_addr = static_cast<T *>(malloc(sizeof(T) * b_mat_size));
  CUST_KERNEL_CHECK_NULLPTR(ctx, gb_temp_addr, KERNEL_STATUS_INNER_ERROR,
                            "[LstsqV2] Malloc memory [gb_temp_addr] failed!")
  int ret = memset_s(ga_addr, sizeof(T) * size_a, 0, sizeof(T) * size_a);
  if (ret != EOK) {
    CUST_KERNEL_LOG_ERROR(ctx, "'LstsqV2Grad' memset[ga_addr] failed, ret is [%d]", ret);
    return KERNEL_STATUS_INNER_ERROR;
  }
  ret = memset_s(gb_addr, sizeof(T) * size_b, 0, sizeof(T) * size_b);
  if (ret != EOK) {
    CUST_KERNEL_LOG_ERROR(ctx, "'LstsqV2Grad' memset[gb_addr] failed, ret is [%d]", ret);
    return KERNEL_STATUS_INNER_ERROR;
  }

  for (int64_t i = 0; i < batch; i++) {
    T *a_batch_addr = a_addr + bcast_a.GetBroadcastXIndex(i) * a_mat_size;
    T *b_batch_addr = b_addr + bcast_b.GetBroadcastXIndex(i) * b_mat_size;
    T *gx_batch_addr = gx_addr + i * gx_mat_size;
    T *ga_batch_addr = ga_addr + bcast_a.GetBroadcastXIndex(i) * a_mat_size;
    T *gb_batch_addr = gb_addr + bcast_b.GetBroadcastXIndex(i) * b_mat_size;
    Pinv(a_batch_addr, pinv_a_addr, m, n);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> gX(gx_batch_addr, n, k);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> b(b_batch_addr, m, k);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> pinvA(pinv_a_addr, n, m);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> gPinvA(g_pinv_a_addr, n, m);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> gB(gb_temp_addr, m, k);
    gPinvA = gX * b.transpose().conjugate();
    gB = pinvA.transpose().conjugate() * gX;
    pinv_backward(g_pinv_a_addr, pinv_a_addr, a_batch_addr, ga_temp_addr, m, n);
    memadd(ga_temp_addr, ga_batch_addr, a_mat_size);
    memadd(gb_temp_addr, gb_batch_addr, b_mat_size);
  }
  free(pinv_a_addr);
  free(g_pinv_a_addr);
  free(ga_temp_addr);
  free(gb_temp_addr);
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kLstsqV2Grad, LstsqV2GradCpuKernel);
}  // namespace aicpu
