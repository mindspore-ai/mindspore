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
#include "eig.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <complex>
#include <iostream>
#include <map>
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 2;
const char *Eig = "Eig";
}  // namespace

namespace aicpu {
uint32_t EigCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "Eig check input and output failed.");
  Tensor *input = ctx.Input(0);
  auto input_dtype = static_cast<DataType>(input->GetDataType());
  switch (input_dtype) {
    case DT_FLOAT:
      return ComputeKernel<float, std::complex<float>>(ctx);
    case DT_DOUBLE:
      return ComputeKernel<double, std::complex<double>>(ctx);
    case DT_COMPLEX64:
      return ComputeKernel<std::complex<float>, std::complex<float>>(ctx);
    case DT_COMPLEX128:
      return ComputeKernel<std::complex<double>, std::complex<double>>(ctx);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Eig kernel data type [%s] not support.", DTypeStr(input_dtype).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

REGISTER_MS_CPU_KERNEL(Eig, EigCpuKernel);

template <typename T, typename C>
uint32_t EigCpuKernel::ComputeKernel(CpuKernelContext &ctx) {
  auto xptr = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto valptr = reinterpret_cast<C *>(ctx.Output(0)->GetData());
  auto vecptr = reinterpret_cast<C *>(ctx.Output(1)->GetData());
  std::vector<int64_t> dims = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  int64_t rank = ctx.Input(0)->GetTensorShape()->GetDims();
  int64_t x_dim = ctx.Input(0)->GetTensorShape()->GetDimSize(rank - 1);
  int64_t batch_size = 1;
  if (rank > 2) {
    for (int64_t i = 0; i < rank - 2; i++) {
      batch_size *= dims[i];
    }
  }
  AttrValue *compute_v = ctx.GetAttr("compute_v");
  bool compute_v_ = (compute_v == nullptr) ? false : compute_v->GetBool();
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(x_dim, x_dim);
  for (int64_t k = 0; k < batch_size; k++) {
    for (int64_t i = 0; i < x_dim * x_dim; i++) {
      A.data()[i] = xptr[k * x_dim * x_dim + i];
    }
    if (!compute_v_) {
      Eigen::ComplexEigenSolver<Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> es(A, false);
      Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> D = es.eigenvalues();
      for (int64_t i = 0; i < x_dim; i++) {
        valptr[k * x_dim + i] = D.data()[i];
      }
    } else {
      Eigen::ComplexEigenSolver<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> es(A);
      Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> D = es.eigenvalues();
      Eigen::Matrix<C, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V = es.eigenvectors();
      for (int64_t i = 0; i < x_dim; i++) {
        valptr[k * x_dim + i] = D.data()[i];
      }
      for (int64_t i = 0; i < x_dim * x_dim; i++) {
        vecptr[k * x_dim * x_dim + i] = V.data()[i];
      }
    }
  }
  return KERNEL_STATUS_OK;
}
}  // namespace aicpu
