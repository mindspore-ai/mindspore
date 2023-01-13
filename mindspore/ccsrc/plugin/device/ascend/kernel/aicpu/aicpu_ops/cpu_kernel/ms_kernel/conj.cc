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
#include "conj.h"

#include <complex>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *const kConj = "Conj";
constexpr int64_t kParallelDataNums = 512 * 1024;

#define CONJ_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                      \
    uint32_t result = ConjCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                  \
      KERNEL_LOG_ERROR("Conj kernel compute failed."); \
      return result;                                   \
    }                                                  \
    break;                                             \
  }
}  // namespace

namespace aicpu {
uint32_t ConjCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kConj);
  KERNEL_HANDLE_ERROR(ConjCheck(ctx), "[%s] check params failed.", kConj);
  DataType dataType = ctx.Input(0)->GetDataType();
  switch (dataType) {
    CONJ_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    CONJ_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      KERNEL_LOG_ERROR("Conj kernel data type [%s] not support.", DTypeStr(dataType).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t ConjCpuKernel::ConjCheck(const CpuKernelContext &ctx) const {
  auto input = ctx.Input(0);
  auto output = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(input->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input data failed.")
  KERNEL_CHECK_NULLPTR(output->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed")
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ConjCpuKernel::ConjCompute(const CpuKernelContext &ctx) const {
  auto inputX = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto outputY = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t dataNum = ctx.Input(0)->NumElements();
  int64_t dataSize = dataNum * static_cast<int64_t>(sizeof(T));
  if (dataSize <= kParallelDataNums) {
    for (int64_t i = 0; i < dataNum; i++) {
      *(outputY + i) = std::conj(*(inputX + i));
    }
  } else {
    uint32_t minCoreNum = 1;
    int64_t maxCoreNum = std::max(minCoreNum, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (maxCoreNum > dataNum) {
      maxCoreNum = dataNum;
    }
    auto shardConj = [&inputX, &outputY](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        *(outputY + i) = std::conj(*(inputX + i));
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, dataNum, dataNum / maxCoreNum, shardConj),
                        "Conj Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kConj, ConjCpuKernel);
}  // namespace aicpu
