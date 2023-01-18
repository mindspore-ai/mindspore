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

#include "lstsq.h"
#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "utils/eigen_tensor.h"
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <iostream>
namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kLstsq = "Lstsq";
}  // namespace
// namespace aicpu
namespace aicpu {
uint32_t LstsqCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Lstsq check input and output number failed.");
  Tensor *input_x0 = ctx.Input(0);
  Tensor *input_x1 = ctx.Input(1);
  Tensor *output = ctx.Output(0);
  auto dims_0 = input_x0->GetTensorShape()->GetDims();
  auto dims_1 = input_x1->GetTensorShape()->GetDims();
  KERNEL_CHECK_FALSE((dims_0 == 2), KERNEL_STATUS_PARAM_INVALID, "Dimension of input[0] must be 2, but got[%zu].",
                     dims_0);
  KERNEL_CHECK_FALSE(((dims_1 == 2) || (dims_1 == 1)), KERNEL_STATUS_PARAM_INVALID,
                     "Dimension of input[1] must be 2 or 1, but got[%zu].", dims_1);
  auto shape_0 = input_x0->GetTensorShape();
  auto shape_1 = input_x1->GetTensorShape();
  KERNEL_CHECK_FALSE((shape_0->GetDimSize(0) == shape_1->GetDimSize(0)), KERNEL_STATUS_PARAM_INVALID,
                     "Lstsq shape_0[0] and shape_1[0] not equal.", shape_0->GetDimSize(0), shape_0->GetDimSize(1));
  AttrValue *I2_regularizer = ctx.GetAttr("l2_regularizer");
  AttrValue *fast = ctx.GetAttr("fast");
  KERNEL_CHECK_NULLPTR(I2_regularizer, KERNEL_STATUS_PARAM_INVALID, "Get l2_regularizer failed.");
  KERNEL_CHECK_NULLPTR(fast, KERNEL_STATUS_PARAM_INVALID, "Get fast failed.");
  KERNEL_LOG_DEBUG(
    "LstsqCpuKernel[%s], inputx0: size[%llu];"
    "inputx1: size[%llu], output: size[%llu].",
    ctx.GetOpType().c_str(), input_x0->GetDataSize(), input_x1->GetDataSize(), output->GetDataSize());
  DataType data_type1 = ctx.Input(0)->GetDataType();
  DataType data_type2 = ctx.Input(1)->GetDataType();
  KERNEL_CHECK_FALSE((data_type1 == data_type2), KERNEL_STATUS_PARAM_INVALID,
                     "Lstsq input_0_dtype must be equal to input_1_dtype.", data_type1, data_type2);
  switch (data_type1) {
    case DT_FLOAT16:
      return LstsqCompute<float, Eigen::half>(ctx);
    case DT_FLOAT:
      return LstsqCompute<float, float>(ctx);
    case DT_DOUBLE:
      return LstsqCompute<double, double>(ctx);
    default:
      KERNEL_LOG_ERROR("Lstsq kernel data type [%u] not support.", DTypeStr(data_type1).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T1, typename T2>
uint32_t LstsqCpuKernel::LstsqCompute(CpuKernelContext &ctx) {
  Eigen::Index m = ctx.Input(0)->GetTensorShape()->GetDimSize(0);
  Eigen::Index n = ctx.Input(0)->GetTensorShape()->GetDimSize(1);
  Eigen::Index k = 1;
  if (ctx.Input(1)->GetTensorShape()->GetDims() == 2) {
    k = ctx.Input(1)->GetTensorShape()->GetDimSize(1);
  }

  typedef Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MartixXd;
  MartixXd A(m, n);
  MartixXd B(m, k);

  auto aptr = reinterpret_cast<T2 *>(ctx.Input(0)->GetData());
  auto bptr = reinterpret_cast<T2 *>(ctx.Input(1)->GetData());

  for (int i = 0; i < m * n; i++) {
    *(A.data() + i) = static_cast<T1>(*(aptr + i));
  }
  for (int i = 0; i < m * k; i++) {
    *(B.data() + i) = static_cast<T1>(*(bptr + i));
  }

  MartixXd result(n, k);
  if (m >= n) {
    result = A.colPivHouseholderQr().solve(B);
  } else {
    MartixXd A_Transpose = A.transpose();
    MartixXd temp = A * A_Transpose;
    MartixXd tempI = temp.inverse();
    MartixXd x = A_Transpose * tempI;
    MartixXd output = x * B;
    result = output;
  }
  auto output_addr = reinterpret_cast<T2 *>(ctx.Output(0)->GetData());
  for (int i = 0; i < n * k; i++) {
    *(output_addr + i) = static_cast<T2>(*(result.data() + i));
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kLstsq, LstsqCpuKernel);
}  // namespace aicpu
