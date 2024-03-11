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
#include "cpu_kernel/ms_kernel/cholesky_solve.h"

#include <Eigen/Dense>

#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;
const char *CholeskySolve = "CholeskySolve";
}  // namespace

namespace aicpu {
uint32_t CholeskySolveCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "CholeskySolve check input and output failed.");

  Tensor *input_x1 = ctx.Input(0);
  AttrValue *upper = ctx.GetAttr("upper");
  bool upperinfo = (upper == nullptr) ? false : upper->GetBool();
  auto data_type_x1 = input_x1->GetDataType();

  switch (data_type_x1) {
    case DT_FLOAT:
      return ComputeKernel<float>(ctx, upperinfo);
    case DT_DOUBLE:
      return ComputeKernel<double>(ctx, upperinfo);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "CholeskySolve kernel data type [%s] not support.", DTypeStr(data_type_x1).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

REGISTER_MS_CPU_KERNEL(CholeskySolve, CholeskySolveCpuKernel);

template <typename T>
uint32_t CholeskySolveCpuKernel::ComputeKernel(CpuKernelContext &ctx, const bool &upper) {
  auto rhsptr = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto lhsptr = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto outptr = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  size_t batch_size = 1;
  std::vector<int64_t> dims = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  size_t dimsnum = ctx.Input(0)->GetTensorShape()->GetDims();
  size_t dim = dims[dimsnum - 2];
  size_t rhs_dim = dims[dimsnum - 1];
  if (dimsnum == 3) {
    batch_size = dims[dimsnum - 3];
  }
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RHS(dim, rhs_dim);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> LHS(dim, dim);
  for (size_t k = 0; k < batch_size; k++) {
    for (size_t i = 0; i < dim * rhs_dim; i++) {
      RHS.data()[i] = rhsptr[k * dim * rhs_dim + i];
    }
    for (size_t i = 0; i < dim * dim; i++) {
      LHS.data()[i] = lhsptr[k * dim * dim + i];
    }
    if (!upper) {
      LHS.template triangularView<Eigen::Lower>().solveInPlace(RHS);
      LHS.adjoint().template triangularView<Eigen::Upper>().solveInPlace(RHS);
    } else {
      LHS.adjoint().template triangularView<Eigen::Lower>().solveInPlace(RHS);
      LHS.template triangularView<Eigen::Upper>().solveInPlace(RHS);
    }
    for (size_t i = 0; i < dim * rhs_dim; i++) {
      outptr[k * dim * rhs_dim + i] = RHS.data()[i];
    }
  }
  return KERNEL_STATUS_OK;
}
}  // namespace aicpu
