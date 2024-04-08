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

#include "cpu_kernel/ms_kernel/cholesky_inverse.h"

#include <Eigen/Dense>

#include <iostream>

#include "context/inc/cpu_kernel_utils.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const uint32_t dimension = 2;
const char *kCholeskyInverse = "CholeskyInverse";
}  // namespace

namespace aicpu {
uint32_t CholeskyInverseCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "Check CholeskyInverse params failed.");
  Tensor *input = ctx.Input(0);
  CUST_KERNEL_CHECK_NULLPTR(ctx, input->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input data failed.");
  Tensor *output = ctx.Output(0);
  auto inputShape = input->GetTensorShape();
  CUST_KERNEL_CHECK_NULLPTR(ctx, inputShape, KERNEL_STATUS_PARAM_INVALID, "Get inputShape failed.");
  AttrValue *upper = ctx.GetAttr("upper");
  CUST_KERNEL_CHECK_NULLPTR(ctx, upper, KERNEL_STATUS_PARAM_INVALID, "Get upper failed.");
  CUST_KERNEL_LOG_DEBUG(ctx,
                        "CholeskyInverseCpuKernel[%s], input: size[%llu];"
                        "output: size[%llu].",
                        ctx.GetOpType().c_str(), input->GetDataSize(), output->GetDataSize());
  auto input_dims = inputShape->GetDims();
  if (input_dims != dimension) {
    CUST_KERNEL_LOG_ERROR(ctx, "CholeskyInverse input dim must be 2!");
    return KERNEL_STATUS_PARAM_INVALID;
  } else if (inputShape->GetDimSize(input_dims - 2) != inputShape->GetDimSize(input_dims - 1)) {
    CUST_KERNEL_LOG_ERROR(ctx, "CholeskyInverse input matrix must be square matrix!");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  DataType data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    case DT_FLOAT:
      return CholeskyInverseCompute<float>(ctx);
    case DT_DOUBLE:
      return CholeskyInverseCompute<double>(ctx);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "CholeskyInverse kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
template <typename T>
uint32_t CholeskyInverseCpuKernel::CholeskyInverseCompute(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto inputShape = ctx.Input(0)->GetTensorShape();
  int64_t n = inputShape->GetDimSize(0);
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXd;
  Eigen::Map<MatrixXd> A(input_x, n, n);
  MatrixXd result;
  AttrValue *upper = ctx.GetAttr("upper");
  bool val = upper->GetBool();
  if (val) {
    result = (A.transpose() * A).inverse();
  } else {
    result = (A * A.transpose()).inverse();
  }
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = 0; j < n; j++) {
      *(output_y + i * n + j) = result(i, j);
    }
  }
  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kCholeskyInverse, CholeskyInverseCpuKernel);
}  // namespace aicpu
