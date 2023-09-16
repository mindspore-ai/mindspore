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
#include "geqrf.h"
#include <cmath>
#include <complex>
#include "cpu_kernel_utils.h"
#include "kernel_log.h"
#include "status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

using namespace std;

namespace {
const char *kGeqrf = "Geqrf";
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 2;
}  // namespace

namespace aicpu {
uint32_t GeqrfCpuKernel::Compute(CpuKernelContext &ctx) {
  if (NormalCheck(ctx, kInputNum, kOutputNum) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  DataType input0_data_type = ctx.Input(0)->GetDataType();
  bool ret = KERNEL_STATUS_PARAM_INVALID;
  switch (input0_data_type) {
    case DT_FLOAT16:
      ret = DoCompute<Eigen::half>(ctx);
      break;
    case DT_FLOAT:
      ret = DoCompute<float>(ctx);
      break;
    case DT_DOUBLE:
      ret = DoCompute<double>(ctx);
      break;
    case DT_COMPLEX64:
      ret = DoComputeC<float>(ctx);
      break;
    case DT_COMPLEX128:
      ret = DoComputeC<double>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("Unsupported input data type[%s]", DTypeStr(input0_data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
}

template <typename T>
void GeqrfCpuKernel::Larfg(int n, int vm, int vn, T **A, T *tau) {
  T zero = static_cast<T>(0);
  if (n <= 1) {
    *tau = zero;
    return;
  }
  T xnorm = zero;
  for (int i = vm + 1; i < vm + n; i++) {
    xnorm = xnorm + A[i][vn] * A[i][vn];
  }
  xnorm = sqrt(xnorm);
  if (xnorm == zero) {
    *tau = zero;
    return;
  } else {
    T beta = sqrt(A[vm][vn] * A[vm][vn] + xnorm * xnorm);
    if (A[vm][vn] > zero) {
      beta = -beta;
    }
    *tau = (beta - (A[vm][vn])) / beta;
    auto scal = (A[vm][vn]) - beta;
    for (int i = vm + 1; i < vm + n; i++) {
      A[i][vn] /= scal;
    }
    A[vm][vn] = beta;
  }
}

template <typename T>
void GeqrfCpuKernel::Larf(int m, int n, T **A, T *tau, int cm, int cn) {
  if (m <= 0 || n <= 0) {
    return;
  }
  T *work = new T[n]();
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      work[j] += A[cm + i][cn - 1] * A[cm + i][cn + j];
    }
  }

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      A[i + cm][j + cn] -= (*tau) * A[cm + i][cn - 1] * work[j];
    }
  }
  delete[] work;
}

template <typename T>
void GeqrfCpuKernel::Geqrf(int m, int n, T **A, T *tau) {
  if (m < 0 || n < 0) {
    return;
  }
  int k = std::min(m, n);
  T one = static_cast<T>(1);
  for (int i = 0; i < k; i++) {
    Larfg<T>(m - i, i, i, A, tau + i);
    T aii = A[i][i];
    A[i][i] = one;
    Larf<T>(m - i, n - i - 1, A, tau + i, i, i + 1);
    A[i][i] = aii;
  }
}

template <typename T>
void GeqrfCpuKernel::CLarfg(int n, int vm, int vn, complex<T> **A, complex<T> *tau) {
  complex<T> one = complex<T>(1, 0);
  complex<T> zero = complex<T>(0, 0);
  if (n <= 0) {
    *tau = zero;
    return;
  }
  T xnorm = 0;
  for (int i = vm + 1; i < vm + n; i++) {
    xnorm = xnorm + norm(A[i][vn]);
  }
  xnorm = sqrt(xnorm);
  T alphr = A[vm][vn].real();
  T alphi = A[vm][vn].imag();
  if (xnorm == 0 && alphi == 0) {
    *tau = zero;
  } else {
    T beta;
    beta = sqrt(alphr * alphr + alphi * alphi + xnorm * xnorm);
    if (A[vm][vn].real() > 0) {
      beta = -beta;
    }
    *tau = complex<T>((beta - alphr) / beta, -alphi / beta);
    A[vm][vn] = one / (A[vm][vn] - beta);
    for (int i = vm + 1; i < vm + n; i++) {
      A[i][vn] *= A[vm][vn];
    }
    A[vm][vn] = beta;
  }
}

template <typename T>
void GeqrfCpuKernel::CLarf(int m, int n, complex<T> **A, complex<T> *tau, int cm, int cn) {
  if (m <= 0 || n <= 0) {
    return;
  }
  complex<T> zero = complex<T>(0, 0);
  complex<T> *work = new complex<T>[n];
  complex<T> temp = zero;
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      temp = temp + conj(A[i + cm][j + cn]) * A[cm + i][cn - 1];
    }
    work[j] = temp;
    temp = zero;
  }
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      A[i + cm][j + cn] = A[i + cm][j + cn] - conj(*tau) * A[cm + i][cn - 1] * conj(work[j]);
    }
  }
  delete[] work;
}

template <typename T>
void GeqrfCpuKernel::CGeqrf(int m, int n, complex<T> **A, complex<T> *tau) {
  if (m < 0 || n < 0) {
    return;
  }
  int k = std::min(m, n);
  complex<T> one = complex<T>(1, 0);
  complex<T> aii;
  for (int i = 0; i < k; i++) {
    CLarfg<T>(m - i, i, i, A, (tau + i));
    aii = A[i][i];
    A[i][i] = one;
    CLarf<T>(m - i, n - i - 1, A, tau + i, i, i + 1);
    A[i][i] = aii;
  }
}

template <typename T>
uint32_t GeqrfCpuKernel::DoCompute(CpuKernelContext &ctx) {
  auto input0_tensor = ctx.Input(0);
  auto input0_tensor_shape = input0_tensor->GetTensorShape();
  KERNEL_CHECK_NULLPTR(input0_tensor_shape, KERNEL_STATUS_PARAM_INVALID, "For Geqrf, input0_tensor_shape is null.");
  int32_t dim = input0_tensor_shape->GetDims();
  if (dim != kOutputNum) {
    KERNEL_LOG_ERROR("The input matrix must have dimension = 2");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  std::vector<int64_t> input0_dims = input0_tensor_shape->GetDimSizes();
  KERNEL_CHECK_FALSE(!input0_dims.empty(), KERNEL_STATUS_PARAM_INVALID, "For Geqrf, input0_dims is empty.");
  const int32_t m = input0_dims[0];
  const int32_t n = input0_dims[1];
  auto input_m = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_r = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto output_tau = reinterpret_cast<T *>(ctx.Output(1)->GetData());

  T **A = new T *[m];
  for (int i = 0; i < m; i++) {
    A[i] = new T[n];
  }
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      A[i][j] = *(input_m + i * n + j);
    }
  }
  Geqrf<T>(m, n, A, output_tau);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      *(output_r + i * n + j) = A[i][j];
    }
  }
  for (int i = 0; i < m; i++) {
    delete[] A[i];
  }
  delete[] A;
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t GeqrfCpuKernel::DoComputeC(CpuKernelContext &ctx) {
  auto input0_tensor = ctx.Input(0);
  auto input0_tensor_shape = input0_tensor->GetTensorShape();
  KERNEL_CHECK_NULLPTR(input0_tensor_shape, KERNEL_STATUS_PARAM_INVALID, "For Geqrf, input0_tensor_shape is null.");
  int32_t dim = input0_tensor_shape->GetDims();
  if (dim != kOutputNum) {
    KERNEL_LOG_ERROR("The input matrix must have dimension = 2");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  std::vector<int64_t> input0_dims = input0_tensor_shape->GetDimSizes();
  KERNEL_CHECK_FALSE(!input0_dims.empty(), KERNEL_STATUS_PARAM_INVALID, "For Geqrf, input0_dims is empty.");
  const int32_t m = input0_dims[0];
  const int32_t n = input0_dims[1];
  auto input_m = reinterpret_cast<complex<T> *>(ctx.Input(0)->GetData());
  auto output_r = reinterpret_cast<complex<T> *>(ctx.Output(0)->GetData());
  auto output_tau = reinterpret_cast<complex<T> *>(ctx.Output(1)->GetData());

  complex<T> **A = new complex<T> *[m];
  for (int i = 0; i < m; i++) {
    A[i] = new complex<T>[n];
  }
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      A[i][j] = *(input_m + i * n + j);
    }
  }
  CGeqrf<T>(m, n, A, output_tau);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      *(output_r + i * n + j) = A[i][j];
    }
  }
  for (int i = 0; i < m; i++) {
    delete[] A[i];
  }
  delete[] A;
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kGeqrf, GeqrfCpuKernel);
}  // namespace aicpu
