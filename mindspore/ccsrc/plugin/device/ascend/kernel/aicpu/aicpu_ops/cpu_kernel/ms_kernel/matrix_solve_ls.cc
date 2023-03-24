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
#include "matrix_solve_ls.h"

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 3;
const uint32_t kOutputNum = 1;
const char *MatrixSolveLs = "MatrixSolveLs";
const int64_t kNum2 = 2;
}  // namespace

namespace aicpu {
uint32_t MatrixSolveLsCpuKernel::Compute(CpuKernelContext &ctx) {
  bool qr_chole = (ctx.GetAttr("fast") == nullptr) ? true : ctx.GetAttr("fast")->GetBool();
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "MatrixSolveLs check input and output number failed.");

  Tensor *matrix = ctx.Input(kFirstInputIndex);
  Tensor *b = ctx.Input(kSecondInputIndex);
  Tensor *l2 = ctx.Input(2);
  Tensor *x = ctx.Output(0);
  if ((matrix->GetDataSize() == 0) || (b->GetDataSize() == 0)) {
    KERNEL_LOG_ERROR("[%s] Input is empty tensor.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto shapea = matrix->GetTensorShape();
  auto shapeb = b->GetTensorShape();
  auto shapel2 = l2->GetTensorShape();
  auto shapex = x->GetTensorShape();
  auto dims = shapea->GetDims();

  if (ctx.Input(1)->GetTensorShape()->GetDims() == 1) {
    if (shapea->GetDimSize(dims - kNum2) != shapeb->GetDimSize(0)) {
      KERNEL_LOG_ERROR(
        "[%s] #Rows mismatch between A and rhs."
        "#Rows of A = [%llu], #Rows of rhs = [%llu]",
        ctx.GetOpType().c_str(), shapea->GetDimSize(dims - kNum2), shapeb->GetDimSize(0));
      return KERNEL_STATUS_PARAM_INVALID;
    }
  } else {
    if (shapea->GetDimSize(dims - kNum2) != shapeb->GetDimSize(dims - kNum2)) {
      KERNEL_LOG_ERROR(
        "[%s] #Rows mismatch between A and rhs."
        "#Rows of A = [%llu], #Rows of rhs = [%llu]",
        ctx.GetOpType().c_str(), shapea->GetDimSize(dims - kNum2), shapeb->GetDimSize(dims - kNum2));
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  if (shapel2->GetDims() != 0 && !(shapel2->GetDims() == 1 && shapel2->GetDimSize(0) == 1)) {
    KERNEL_LOG_ERROR("[%s] Tensor l2 should be a scalar or a single 1-dimension number.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(1)->GetTensorShape()->GetDims() == 1) {
    if (shapex->GetDims() != shapeb->GetDims() || shapea->GetDimSize(dims - 1) != shapex->GetDimSize(0) ||
        shapex->GetDimSize(shapex->GetDims() - 1) != shapeb->GetDimSize(0)) {
      KERNEL_LOG_ERROR("[%s] Tensor y shape mismatch.", ctx.GetOpType().c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  } else {
    if (shapex->GetDims() != shapeb->GetDims() ||
        shapea->GetDimSize(dims - 1) != shapex->GetDimSize(shapex->GetDims() - kNum2) ||
        shapex->GetDimSize(shapex->GetDims() - 1) != shapeb->GetDimSize(shapeb->GetDims() - 1)) {
      KERNEL_LOG_ERROR("[%s] Tensor y shape mismatch.", ctx.GetOpType().c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  auto a_data_type = matrix->GetDataType();
  auto b_data_type = b->GetDataType();
  if (a_data_type != b_data_type) {
    KERNEL_LOG_ERROR("[%s] Tensor data type mismatch.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (a_data_type != DT_FLOAT && a_data_type != DT_DOUBLE && a_data_type != DT_COMPLEX64 &&
      a_data_type != DT_COMPLEX128) {
    KERNEL_LOG_ERROR("MatrixSolveLs kernel data type [%s] not support.", DTypeStr(a_data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (qr_chole) {
    if (a_data_type == DT_COMPLEX64) {
      return ComplexCholesky<float>(ctx);
    }
    if (a_data_type == DT_COMPLEX128) {
      return ComplexCholesky<double>(ctx);
    }
    if (a_data_type == DT_DOUBLE) {
      return RealCholesky<double>(ctx);
    }
    if (a_data_type == DT_FLOAT) {
      return RealCholesky<float>(ctx);
    }
  } else {
    if (a_data_type == DT_COMPLEX64) {
      return ComplexQr<float>(ctx);
    }
    if (a_data_type == DT_COMPLEX128) {
      return ComplexQr<double>(ctx);
    }
    if (a_data_type == DT_DOUBLE) {
      return RealQr<double>(ctx);
    }
    if (a_data_type == DT_FLOAT) {
      return RealQr<float>(ctx);
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(MatrixSolveLs, MatrixSolveLsCpuKernel);

template <typename T>
void MatrixSolveLsCpuKernel::RealCholeskySingleCompute(T *aptr, T *bptr, T *xptr, double *l2, int64_t m, int64_t k,
                                                       int64_t n) {
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a(m, k);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x(k, n);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> b(m, n);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a_copy;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a_b;

  for (int i = 0; i < m * k; i++) {
    *(a.data() + i) = *(aptr + i);
  }
  for (int i = 0; i < m * n; i++) {
    *(b.data() + i) = *(bptr + i);
  }

  if (m >= k) {
    a_copy =
      a.transpose() * a + ((T)*l2) * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Identity(k, k);
    a_b = a.transpose() * b;
  } else {
    a_copy = a * a.transpose();
    a_b = b;
  }
  for (int64_t i = 0; i < n; i++) {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> xi = a_copy.ldlt().solve(a_b.col(i));
    if (m < k) {
      xi = a.transpose() * xi;
    }
    x.col(i) = xi;
  }
  for (int64_t i = 0; i < k * n; i++) {
    *(xptr + i) = *(x.data() + i);
  }
}

template <typename T>
uint32_t MatrixSolveLsCpuKernel::RealCholesky(CpuKernelContext &ctx) {
  auto dims = ctx.Input(0)->GetTensorShape()->GetDims();
  auto aptr = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto bptr = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto xptr = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto l2 = reinterpret_cast<double *>(ctx.Input(2)->GetData());
  int64_t m = ctx.Input(0)->GetTensorShape()->GetDimSize(dims - 2);
  int64_t k = ctx.Input(0)->GetTensorShape()->GetDimSize(dims - 1);
  int64_t n = 1;
  if (ctx.Input(1)->GetTensorShape()->GetDims() > 1) {
    n = ctx.Input(1)->GetTensorShape()->GetDimSize(dims - 1);
  }
  int64_t data_num = ctx.Input(0)->NumElements();
  const int64_t mat_size = m * k;
  const int64_t rhs_size = m * n;
  const int64_t res_size = n * k;
  const int64_t batch = data_num / mat_size;
  const int64_t kParallelDataNum = 16 * mat_size;
  const int64_t kParallelDataNumMid = 72 * mat_size;
  if (data_num >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

    if (data_num <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }
    auto sharder_matrix_solve_ls = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        RealCholeskySingleCompute(aptr + i * mat_size, bptr + i * rhs_size, xptr + i * res_size, l2, m, k, n);
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, batch, batch / max_core_num, sharder_matrix_solve_ls),
                        "MatrixSolveLs Compute failed.");
  } else {
    for (int64_t i = 0; i < batch; i++) {
      RealCholeskySingleCompute(aptr + i * mat_size, bptr + i * rhs_size, xptr + i * res_size, l2, m, k, n);
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
void MatrixSolveLsCpuKernel::ComplexCholeskySingleCompute(std::complex<T> *aptr, std::complex<T> *bptr,
                                                          std::complex<T> *xptr, double *l2, int64_t m, int64_t k,
                                                          int64_t n) {
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(kNum2 * m, kNum2 * k);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x(kNum2 * k, n);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> b(kNum2 * m, n);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a_copy;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a_b;
  auto l2value = abs(*l2);

  for (int64_t i = 0; i < k; i++) {
    for (int64_t j = 0; j < m; j++) {
      *(A.data() + i + j * kNum2 * k) = std::real(*(aptr + i + j * k));
    }
    for (int64_t j = 0; j < m; j++) {
      *(A.data() + (i + k) + (j + m) * kNum2 * k) = std::real(*(aptr + i + j * k));
    }
    for (int64_t j = 0; j < m; j++) {
      *(A.data() + (i + k) + j * kNum2 * k) = -std::imag(*(aptr + i + j * k));
    }
    for (int64_t j = 0; j < m; j++) {
      *(A.data() + i + (j + m) * kNum2 * k) = std::imag(*(aptr + i + j * k));
    }
  }
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = 0; j < m; j++) {
      *(b.data() + i + j * n) = std::real(*(bptr + i + j * n));
      *(b.data() + i + (j + m) * n) = std::imag(*(bptr + i + j * n));
    }
  }

  if (m >= k) {
    a_copy =
      A.transpose() * A +
      ((T)l2value) * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Identity(kNum2 * k, kNum2 * k);
    a_b = A.transpose() * b;
  } else {
    a_copy =
      A * A.transpose() +
      ((T)l2value) * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Identity(kNum2 * m, kNum2 * m);
    a_b = b;
  }

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> xi;
  for (int64_t i = 0; i < n; i++) {
    xi = a_copy.ldlt().solve(a_b.col(i));
    if (m < k) {
      xi = A.transpose() * xi;
    }
    x.col(i) = xi;
    for (int64_t j = 0; j < k; j++) {
      (xptr + i + j * n)->real(*(x.data() + i + j * n));
      (xptr + i + j * n)->imag(*(x.data() + i + (j + k) * n));
    }
  }
}

template <typename T>
uint32_t MatrixSolveLsCpuKernel::ComplexCholesky(CpuKernelContext &ctx) {
  auto dims = ctx.Input(0)->GetTensorShape()->GetDims();
  auto l2 = reinterpret_cast<double *>(ctx.Input(2)->GetData());
  auto aptr = reinterpret_cast<std::complex<T> *>(ctx.Input(0)->GetData());
  auto bptr = reinterpret_cast<std::complex<T> *>(ctx.Input(1)->GetData());
  auto xptr = reinterpret_cast<std::complex<T> *>(ctx.Output(0)->GetData());
  int64_t m = ctx.Input(0)->GetTensorShape()->GetDimSize(dims - 2);
  int64_t k = ctx.Input(0)->GetTensorShape()->GetDimSize(dims - 1);
  int64_t n = 1;
  if (ctx.Input(1)->GetTensorShape()->GetDims() > 1) {
    n = ctx.Input(1)->GetTensorShape()->GetDimSize(dims - 1);
  }
  int64_t data_num = ctx.Input(0)->NumElements();
  const int64_t mat_size = m * k;
  const int64_t rhs_size = m * n;
  const int64_t res_size = n * k;
  const int64_t batch = data_num / mat_size;
  const int64_t kParallelDataNum = 16 * mat_size;
  const int64_t kParallelDataNumMid = 72 * mat_size;
  if (data_num >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (data_num <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }
    auto sharder_matrix_solve_ls = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        ComplexCholeskySingleCompute(aptr + i * mat_size, bptr + i * rhs_size, xptr + i * res_size, l2, m, k, n);
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, batch, batch / max_core_num, sharder_matrix_solve_ls),
                        "MatrixSolveLs Compute failed.");
  } else {
    for (int64_t i = 0; i < batch; i++) {
      ComplexCholeskySingleCompute(aptr + i * mat_size, bptr + i * rhs_size, xptr + i * res_size, l2, m, k, n);
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
void MatrixSolveLsCpuKernel::RealQrSingleCompute(T *aptr, T *bptr, T *xptr, int64_t m, int64_t k, int64_t n) {
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a(m, k);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x(k, n);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> b(m, n);

  for (int i = 0; i < m * k; i++) {
    *(a.data() + i) = *(aptr + i);
  }
  for (int i = 0; i < m * n; i++) {
    *(b.data() + i) = *(bptr + i);
  }

  Eigen::ColPivHouseholderQR<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> qr_solve(a);

  for (int64_t i = 0; i < n; i++) {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> xi = qr_solve.solve(b.col(i));
    x.col(i) = xi;
  }
  for (int64_t i = 0; i < k * n; i++) {
    *(xptr + i) = *(x.data() + i);
  }
}

template <typename T>
uint32_t MatrixSolveLsCpuKernel::RealQr(CpuKernelContext &ctx) {
  auto dims = ctx.Input(0)->GetTensorShape()->GetDims();
  auto aptr = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto bptr = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto xptr = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t m = ctx.Input(0)->GetTensorShape()->GetDimSize(dims - 2);
  int64_t k = ctx.Input(0)->GetTensorShape()->GetDimSize(dims - 1);
  int64_t n = 1;
  if (ctx.Input(1)->GetTensorShape()->GetDims() > 1) {
    n = ctx.Input(1)->GetTensorShape()->GetDimSize(dims - 1);
  }
  int64_t data_num = ctx.Input(0)->NumElements();
  const int64_t mat_size = m * k;
  const int64_t rhs_size = m * n;
  const int64_t res_size = n * k;
  const int64_t batch = data_num / mat_size;
  const int64_t kParallelDataNum = 16 * mat_size;
  const int64_t kParallelDataNumMid = 72 * mat_size;
  if (data_num >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (data_num <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }
    auto sharder_matrix_solve_ls = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        RealQrSingleCompute(aptr + i * mat_size, bptr + i * rhs_size, xptr + i * res_size, m, k, n);
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, batch, batch / max_core_num, sharder_matrix_solve_ls),
                        "MatrixSolveLs Compute failed.");
  } else {
    for (int64_t i = 0; i < batch; i++) {
      RealQrSingleCompute(aptr + i * mat_size, bptr + i * rhs_size, xptr + i * res_size, m, k, n);
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
void MatrixSolveLsCpuKernel::ComplexQrSingleCompute(std::complex<T> *aptr, std::complex<T> *bptr, std::complex<T> *xptr,
                                                    int64_t m, int64_t k, int64_t n) {
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(kNum2 * m, kNum2 * k);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x(kNum2 * k, n);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> b(kNum2 * m, n);
  for (int64_t i = 0; i < k; i++) {
    for (int64_t j = 0; j < m; j++) {
      *(A.data() + i + j * kNum2 * k) = std::real(*(aptr + i + j * k));
    }
    for (int64_t j = 0; j < m; j++) {
      *(A.data() + (i + k) + (j + m) * kNum2 * k) = std::real(*(aptr + i + j * k));
    }
    for (int64_t j = 0; j < m; j++) {
      *(A.data() + (i + k) + j * kNum2 * k) = -std::imag(*(aptr + i + j * k));
    }
    for (int64_t j = 0; j < m; j++) {
      *(A.data() + i + (j + m) * kNum2 * k) = std::imag(*(aptr + i + j * k));
    }
  }
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = 0; j < m; j++) {
      *(b.data() + i + j * n) = std::real(*(bptr + i + j * n));
      *(b.data() + i + (j + m) * n) = std::imag(*(bptr + i + j * n));
    }
  }

  Eigen::ColPivHouseholderQR<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> qr_solve(A);

  for (int64_t i = 0; i < n; i++) {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> xi = qr_solve.solve(b.col(i));
    x.col(i) = xi;

    for (int64_t j = 0; j < k; j++) {
      (xptr + i + j * n)->real(*(x.data() + i + j * n));
      (xptr + i + j * n)->imag(*(x.data() + i + (j + k) * n));
    }
  }
}

template <typename T>
uint32_t MatrixSolveLsCpuKernel::ComplexQr(CpuKernelContext &ctx) {
  auto dims = ctx.Input(0)->GetTensorShape()->GetDims();
  int64_t m = ctx.Input(0)->GetTensorShape()->GetDimSize(dims - 2);
  int64_t k = ctx.Input(0)->GetTensorShape()->GetDimSize(dims - 1);
  int64_t n = 1;
  if (ctx.Input(1)->GetTensorShape()->GetDims() > 1) {
    n = ctx.Input(1)->GetTensorShape()->GetDimSize(dims - 1);
  }
  int64_t data_num = ctx.Input(0)->NumElements();
  const int64_t mat_size = m * k;
  const int64_t rhs_size = m * n;
  const int64_t res_size = n * k;
  const int64_t batch = data_num / mat_size;
  const int64_t kParallelDataNum = 16 * mat_size;
  const int64_t kParallelDataNumMid = 72 * mat_size;
  auto aptr = reinterpret_cast<std::complex<T> *>(ctx.Input(0)->GetData());
  auto bptr = reinterpret_cast<std::complex<T> *>(ctx.Input(1)->GetData());
  auto xptr = reinterpret_cast<std::complex<T> *>(ctx.Output(0)->GetData());
  if (data_num >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (data_num <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }
    auto sharder_matrix_solve_ls = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        ComplexQrSingleCompute(aptr + i * mat_size, bptr + i * rhs_size, xptr + i * res_size, m, k, n);
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, batch, batch / max_core_num, sharder_matrix_solve_ls),
                        "MatrixSolveLs Compute failed.");
  } else {
    for (int64_t i = 0; i < batch; i++) {
      ComplexQrSingleCompute(aptr + i * mat_size, bptr + i * rhs_size, xptr + i * res_size, m, k, n);
    }
  }
  return KERNEL_STATUS_OK;
}

}  // namespace aicpu