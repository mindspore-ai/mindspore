/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All right reserved.
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

#include "polar.h"

#include "complex"
#include "context/inc/cpu_kernel_utils.h"
#include "iostream"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;
const char *kPolar = "Polar";
const int64_t kParallelDataNumMid = 35 * 1024;
const int64_t kParallelDataNum = 7 * 1024;
}  // namespace

namespace aicpu {
uint32_t PolarCpuKernel::Compute(CpuKernelContext &ctx) {
  DataType abs_type = ctx.Input(0)->GetDataType();
  DataType angle_type = ctx.Input(1)->GetDataType();
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "Polar check input and output number failed.");
  if (abs_type == DT_FLOAT && angle_type == DT_FLOAT) {
    return PolarCompute<float>(ctx);
  } else if (abs_type == DT_DOUBLE && angle_type == DT_DOUBLE) {
    return PolarCompute<double>(ctx);
  } else {
    CUST_KERNEL_LOG_ERROR(ctx, "Polar kernel data type [%s],[%s] not support.", DTypeStr(abs_type).c_str(),
                          DTypeStr(angle_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t PolarCpuKernel::PolarCompute(CpuKernelContext &ctx) {
  auto abs = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto angle = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto output = reinterpret_cast<std::complex<T> *>(ctx.Output(0)->GetData());
  auto input_shape = ctx.Input(0)->GetTensorShape();
  int64_t elements = input_shape->NumElements();
  auto sharder_polar = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
      output[i].real(abs[i] * cos(angle[i]));
      output[i].imag(abs[i] * sin(angle[i]));
    }
  };
  if (elements > kParallelDataNum) {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (elements <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, static_cast<int64_t>(4));  // up to 4 cpu cores
    }

    if (max_core_num > elements) {
      max_core_num = elements;
    }
    if (max_core_num > 0) {
      CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, elements, elements / max_core_num, sharder_polar),
                               "Polar Compute failed.");
    }
  } else {
    sharder_polar(0, elements);
  }
  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kPolar, PolarCpuKernel);
}  // namespace aicpu
