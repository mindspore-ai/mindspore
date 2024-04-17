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
#include "diag_part.h"
#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *kDiagPart = "DiagPart";

#define DIAGPART_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                                    \
    uint32_t result = DiagPartCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                                \
      CUST_KERNEL_LOG_ERROR(ctx, "DiagPart kernel compute failed."); \
      return result;                                                 \
    }                                                                \
    break;                                                           \
  }
}  // namespace

namespace aicpu {
uint32_t DiagPartCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.",
                           kDiagPart);
  CUST_KERNEL_HANDLE_ERROR(ctx, DiagPartCheck(ctx), "[%s] check params failed.", kDiagPart);
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    DIAGPART_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    DIAGPART_COMPUTE_CASE(DT_FLOAT, float, ctx)
    DIAGPART_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    DIAGPART_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    DIAGPART_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    DIAGPART_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    DIAGPART_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "DiagPart kernel data type [%s] not supports.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t DiagPartCpuKernel::DiagPartCheck(CpuKernelContext &ctx) {
  std::vector<int64_t> shape_input = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_output = ctx.Output(0)->GetTensorShape()->GetDimSizes();
  CUST_KERNEL_CHECK_FALSE(ctx, (shape_input.size() % 2 == 0), KERNEL_STATUS_PARAM_INVALID,
                          "The rank of the tensor should be even and positive.");
  for (size_t i = 0; i < shape_output.size(); i++) {
    CUST_KERNEL_CHECK_FALSE(ctx, (shape_input[i] == shape_input[i + shape_output.size()]), KERNEL_STATUS_PARAM_INVALID,
                            "Invalid shape: the input dimension [%zu] size [%zu] does not match "
                            "the input dimension [%zu] size [%zu].",
                            i, shape_input[i], i + shape_output.size(), shape_input[i + shape_output.size()]);
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t DiagPartCpuKernel::DiagPartCompute(CpuKernelContext &ctx) {
  auto input = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  uint64_t size = ctx.Output(0)->NumElements();
  for (size_t index = 0; index < size; index++) {
    *(output + index) = *(input + (1 + size) * index);
  }

  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kDiagPart, DiagPartCpuKernel);
}  // namespace aicpu
