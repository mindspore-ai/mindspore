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
#include "ms_kernel/reciprocal_grad.h"

#include <float.h>
#include <complex>
#include <math.h>
#include <algorithm>

#include "context/inc/cpu_kernel_utils.h"
#include "cpu_types.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kReciprocalGrad = "ReciprocalGrad";
const size_t kReciprocalGradInputNum = 2;
const size_t kReciprocalGradOutputNum = 1;
constexpr int64_t kParallelDataNums = 64 * 1024;
constexpr int64_t kParallelComplexDataNums = 16 * 1024;
}  // namespace

namespace aicpu {
uint32_t ReciprocalGradCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kReciprocalGradInputNum, kReciprocalGradOutputNum),
                           "Check ReciprocalGrad params failed.");
  Tensor *y = ctx.Input(0);
  Tensor *dy = ctx.Input(1);
  Tensor *z = ctx.Output(0);
  if (y->GetDataType() != dy->GetDataType()) {
    CUST_KERNEL_LOG_ERROR(ctx, "The data type of the input2 [%s] need be the same as the input1 [%s]",
                          DTypeStr(dy->GetDataType()).c_str(), DTypeStr(y->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (y->GetDataSize() != dy->GetDataSize()) {
    CUST_KERNEL_LOG_ERROR(ctx,
                          "The data size of the input2 [%llu] need be the same as the input1 "
                          "[%llu]",
                          dy->GetDataSize(), y->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  uint64_t data_num = y->NumElements();
  DataType data_type = y->GetDataType();
  uint32_t res = KERNEL_STATUS_OK;

  switch (data_type) {
    case DT_FLOAT16:
      res = ReciprocalGradCompute<Eigen::half>(y, dy, z, data_num, ctx);
      break;
    case DT_FLOAT:
      res = ReciprocalGradCompute<float>(y, dy, z, data_num, ctx);
      break;
    case DT_DOUBLE:
      res = ReciprocalGradCompute<double>(y, dy, z, data_num, ctx);
      break;
    case DT_COMPLEX64:
      res = ReciprocalGradComputeComplex<std::complex<float>>(y, dy, z, data_num, ctx);
      break;
    case DT_COMPLEX128:
      res = ReciprocalGradComputeComplex<std::complex<double>>(y, dy, z, data_num, ctx);
      break;
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "ReciprocalGrad invalid input type [%s]", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (res != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ReciprocalGradCpuKernel::ReciprocalGradCompute(Tensor *y, Tensor *dy, Tensor *z, uint64_t data_num,
                                                        CpuKernelContext &ctx) {
  auto input_y = reinterpret_cast<T *>(y->GetData());
  auto input_dy = reinterpret_cast<T *>(dy->GetData());
  auto output_z = reinterpret_cast<T *>(z->GetData());
  if (data_num <= kParallelDataNums) {
    for (size_t i = 0; i < data_num; i++) {
      output_z[i] = static_cast<T>(-1) * input_dy[i] * input_y[i] * input_y[i];
    }
  } else {
    uint32_t min_core_num = 1;
    uint64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shard_ReciprocalGrad = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        output_z[i] = static_cast<T>(-1) * input_dy[i] * input_y[i] * input_y[i];
      }
    };
    if (max_core_num == 0) {
      CUST_KERNEL_LOG_ERROR(ctx, "max_core_num could not be 0.");
    }
    uint32_t ret = CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_ReciprocalGrad);
    if (ret != KERNEL_STATUS_OK) {
      CUST_KERNEL_LOG_ERROR(ctx, "CpuKernelUtils::ParallelFor failed");
      return KERNEL_STATUS_INNER_ERROR;
    }
    CUST_KERNEL_HANDLE_ERROR(ctx,
                             CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_ReciprocalGrad),
                             "ReciprocalGrad Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ReciprocalGradCpuKernel::ReciprocalGradComputeComplex(Tensor *y, Tensor *dy, Tensor *z, uint64_t data_num,
                                                               CpuKernelContext &ctx) {
  auto input_y = reinterpret_cast<T *>(y->GetData());
  auto input_dy = reinterpret_cast<T *>(dy->GetData());
  auto output_z = reinterpret_cast<T *>(z->GetData());
  if (data_num <= kParallelComplexDataNums) {
    for (size_t i = 0; i < data_num; i++) {
      output_z[i] = static_cast<T>(-1) * input_dy[i] * conj(input_y[i] * input_y[i]);
    }
  } else {
    uint32_t min_core_num = 1;
    uint64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shard_ReciprocalGrad = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        output_z[i] = static_cast<T>(-1) * input_dy[i] * conj(input_y[i] * input_y[i]);
      }
    };
    if (max_core_num == 0) {
      CUST_KERNEL_LOG_ERROR(ctx, "max_core_num could not be 0.");
    }
    uint32_t ret = CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_ReciprocalGrad);
    if (ret != KERNEL_STATUS_OK) {
      CUST_KERNEL_LOG_ERROR(ctx, "CpuKernelUtils::ParallelFor failed");
      return KERNEL_STATUS_INNER_ERROR;
    }
    CUST_KERNEL_HANDLE_ERROR(ctx,
                             CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_ReciprocalGrad),
                             "ReciprocalGrad Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kReciprocalGrad, ReciprocalGradCpuKernel);
}  // namespace aicpu
