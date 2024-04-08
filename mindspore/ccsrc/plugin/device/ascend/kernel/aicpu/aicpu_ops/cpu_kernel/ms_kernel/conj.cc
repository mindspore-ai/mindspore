/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
#include "cpu_kernel/ms_kernel/conj.h"

#include <algorithm>
#include <complex>
#include <functional>

#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *const kConj = "Conj";
constexpr int64_t kParallelDataNums = 512 * 1024;

#define CONJ_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                                \
    uint32_t result = ConjCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                            \
      CUST_KERNEL_LOG_ERROR(ctx, "Conj kernel compute failed."); \
      return result;                                             \
    }                                                            \
    break;                                                       \
  }
}  // namespace

namespace aicpu {
uint32_t ConjCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(const_cast<CpuKernelContext &>(ctx), kInputNum, kOutputNum),
                           "[%s] check input and output failed.", kConj);
  DataType dataType = ctx.Input(0)->GetDataType();
  switch (dataType) {
    CONJ_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    CONJ_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    CONJ_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    CONJ_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    CONJ_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    CONJ_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    CONJ_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    CONJ_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    CONJ_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    CONJ_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    CONJ_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    CONJ_COMPUTE_CASE(DT_FLOAT, float_t, ctx)
    CONJ_COMPUTE_CASE(DT_DOUBLE, double_t, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Conj kernel data type [%s] not support.", DTypeStr(dataType).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t ConjCpuKernel::ConjCheck(CpuKernelContext &ctx) const {
  auto input = ctx.Input(0);
  auto output = ctx.Output(0);
  CUST_KERNEL_CHECK_NULLPTR(ctx, input->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input data failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, output->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed")
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ConjCpuKernel::ConjCompute(CpuKernelContext &ctx) const {
  auto inputX = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto outputY = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t dataNum = ctx.Input(0)->NumElements();
  int64_t dataSize = dataNum * static_cast<int64_t>(sizeof(T));

  std::function<void(T *, T *, T *)> conj_compute;
  if constexpr ((std::is_same_v<T, std::complex<float>>) || (std::is_same_v<T, std::complex<double>>)) {
    conj_compute = [](T *input1, T *last1, T *d_first) {
      std::transform(input1, last1, d_first, [](T x) { return std::conj(x); });
    };
  } else {
    conj_compute = [](T *input1, T *last1, T *d_first) { std::copy(input1, last1, d_first); };
  }

  if (dataSize <= kParallelDataNums) {
    conj_compute(inputX, inputX + dataNum, outputY);
  } else {
    uint32_t minCoreNum = 1;
    int64_t maxCoreNum = std::max(minCoreNum, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (maxCoreNum > dataNum) {
      maxCoreNum = dataNum;
    }
    auto shardConj = [&inputX, &outputY, conj_compute](size_t start, size_t end) {
      conj_compute(inputX + start, inputX + end, outputY + start);
    };
    CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, dataNum, dataNum / maxCoreNum, shardConj),
                             "Conj Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kConj, ConjCpuKernel);
}  // namespace aicpu
