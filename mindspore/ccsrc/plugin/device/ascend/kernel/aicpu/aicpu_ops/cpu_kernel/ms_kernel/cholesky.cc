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
#include "cholesky.h"
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <map>
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cpu_kernel_utils.h"

namespace {
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 1;
const char *Cholesky = "Cholesky";
}  // namespace

namespace aicpu {
uint32_t CholeskyCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Cholesky check input and output failed.");

  Tensor *input = ctx.Input(0);
  AttrValue *upper = ctx.GetAttr("upper");
  bool upperinfo = (upper == nullptr) ? false : upper->GetBool();
  auto data_type_in = input->GetDataType();

  switch (data_type_in) {
    case DT_FLOAT:
      return ComputeKernel<float>(ctx, upperinfo);
    case DT_DOUBLE:
      return ComputeKernel<double>(ctx, upperinfo);
    default:
      KERNEL_LOG_ERROR("Cholesky kernel data type [%s] not support.", DTypeStr(data_type_in).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

REGISTER_CPU_KERNEL(Cholesky, CholeskyCpuKernel);

template <typename T>
uint32_t CholeskyCpuKernel::ComputeKernel(CpuKernelContext &ctx, const bool &upper) {
  auto inputptr = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto outputptr = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  std::vector<int64_t> dims = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  int64_t dimsnum = ctx.Input(0)->GetTensorShape()->GetDims();
  int64_t dim = ctx.Input(0)->GetTensorShape()->GetDimSize(dimsnum - 1);
  int64_t n = dim;
  int64_t count = 1;
  int64_t no_batch = 2;

  if (dimsnum > no_batch) {
    for (int64_t m = 0; m < dimsnum - no_batch; m++) {
      count *= dims[m];
    }
  }

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(n, n);

  for (int64_t k = 0; k < count; k++) {
    for (int64_t i = 0; i < n * n; i++) {
      A.data()[i] = inputptr[k * n * n + i];
    }

    Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_llt(A);

    if (!A.isApprox(A.transpose()) || A_llt.info() == Eigen::NumericalIssue) {
      KERNEL_LOG_ERROR("There exists non semi-positive definitie matrix!");
      return KERNEL_STATUS_INNER_ERROR;
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> L = A_llt.matrixL();

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U = A_llt.matrixU();

    if (!upper) {
      for (int64_t i = 0; i < n * n; i++) {
        outputptr[k * n * n + i] = L.data()[i];
      }
    } else {
      for (int64_t i = 0; i < n * n; i++) {
        outputptr[k * n * n + i] = U.data()[i];
      }
    }
  }
  return KERNEL_STATUS_OK;
}
}  // namespace aicpu