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
#include "diag.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *kDiag = "Diag";
constexpr int64_t kParallelDataNums = 80 * 32;
constexpr int64_t kParallelDataNumsMid = 8 * 1024;

#define DIAG_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                      \
    uint32_t result = DiagCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                  \
      KERNEL_LOG_ERROR("Diag kernel compute failed."); \
      return result;                                   \
    }                                                  \
    break;                                             \
  }
}  // namespace

namespace aicpu {
uint32_t DiagCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kDiag);
  KERNEL_HANDLE_ERROR(DiagCheck(ctx), "[%s] check params failed.", kDiag);
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    DIAG_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    DIAG_COMPUTE_CASE(DT_FLOAT, float, ctx)
    DIAG_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    DIAG_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    DIAG_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    DIAG_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    DIAG_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      KERNEL_LOG_ERROR("Diag kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t DiagCpuKernel::DiagCheck(CpuKernelContext &ctx) {
  KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input data failed.")
  KERNEL_CHECK_NULLPTR(ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed.")
  KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID, "Get input tensor shape failed.")
  KERNEL_CHECK_NULLPTR(ctx.Output(0)->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID, "Get output tensor shape failed.")

  std::vector<int64_t> shape_input = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_output = ctx.Output(0)->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((shape_input.size() != 0), KERNEL_STATUS_PARAM_INVALID,
                     "Input must be at least rank 1, got [%zu].", shape_input.size())
  KERNEL_CHECK_FALSE((shape_input.size() != shape_output.size() * 2), KERNEL_STATUS_PARAM_INVALID,
                     "The output shape size should be twice the output shape size, "
                     "but the input shape size is [%zu] and the output shape size is [%zu].",
                     shape_input.size(), shape_output.size())
  for (size_t i = 0; i < shape_output.size(); ++i) {
    KERNEL_CHECK_FALSE((shape_input[i % shape_input.size()] == shape_output[i]), KERNEL_STATUS_PARAM_INVALID,
                       "Invalid shape: the input dimension [%zu] size [%zu] does not match "
                       "the output dimension [%zu] size [%zu].",
                       i % shape_input.size(), shape_input[i % shape_input.size()], i, shape_output[i])
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t DiagCpuKernel::DiagCompute(CpuKernelContext &ctx) {
  auto input = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  int64_t size = ctx.Input(0)->NumElements();
  int64_t data_size = size * sizeof(T);

  if (data_size <= kParallelDataNums) {
    std::fill(output, output + size * size, T());
    for (int64_t index = 0; index < size; index++) {
      *(output + (1 + size) * index) = *(input + index);
    }
  } else {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (data_size <= kParallelDataNumsMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }
    if (max_core_num > size) {
      max_core_num = size;
    }
    auto shard_diag = [&](int64_t start, int64_t end) {
      std::fill(output + size * start, output + size * end, T());
      for (int64_t index = start; index < end; index++) {
        *(output + (1 + size) * index) = *(input + index);
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, size, size / max_core_num, shard_diag),
                        "Diag Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kDiag, DiagCpuKernel);
}  // namespace aicpu
