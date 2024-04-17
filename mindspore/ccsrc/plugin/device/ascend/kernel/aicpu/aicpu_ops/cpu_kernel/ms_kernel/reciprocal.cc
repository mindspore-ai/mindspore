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

#include "ms_kernel/reciprocal.h"

#include <float.h>
#include <complex>
#include <algorithm>

#include "context/inc/cpu_kernel_utils.h"
#include "cpu_types.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kReciprocal = "Reciprocal";
const size_t kReciprocalInputNum = 1;
const size_t kReciprocalOutputNum = 1;
constexpr int64_t kParallelDataNums = 32 * 1024;
}  // namespace

namespace aicpu {
uint32_t ReciprocalCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *x = ctx.Input(0);
  Tensor *y = ctx.Output(0);
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kReciprocalOutputNum, kReciprocalInputNum),
                           "Check Reciprocal params failed.");
  if (x->GetDataType() != y->GetDataType()) {
    CUST_KERNEL_LOG_ERROR(ctx, "The data type of the input [%s] need be the same as the output [%s]",
                          DTypeStr(x->GetDataType()).c_str(), DTypeStr(y->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (x->GetDataSize() != y->GetDataSize()) {
    CUST_KERNEL_LOG_ERROR(ctx,
                          "The data size of the input [%llu] need be the same as the output "
                          "[%llu]",
                          x->GetDataSize(), y->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  uint64_t data_num = x->NumElements();
  DataType data_type = x->GetDataType();
  uint32_t res = KERNEL_STATUS_OK;

  switch (data_type) {
    case DT_FLOAT:
      res = ReciprocalCompute<float>(x, y, data_num, ctx);
      break;
    case DT_DOUBLE:
      res = ReciprocalCompute<double>(x, y, data_num, ctx);
      break;
    case DT_FLOAT16:
      res = ReciprocalCompute<Eigen::half>(x, y, data_num, ctx);
      break;
    case DT_COMPLEX64:
      res = ReciprocalComputeComplex<std::complex<float>>(x, y, data_num, ctx);
      break;
    case DT_COMPLEX128:
      res = ReciprocalComputeComplex<std::complex<double>>(x, y, data_num, ctx);
      break;
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Reciprocal kernel data type [%s] not support", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (res != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ReciprocalCpuKernel::ReciprocalCompute(Tensor *x, Tensor *y, uint64_t data_num, CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(x->GetData());
  auto output_y = reinterpret_cast<T *>(y->GetData());
  if (data_num <= kParallelDataNums) {
    for (size_t i = 0; i < data_num; i++) {
      if (input_x[i] == static_cast<T>(0)) {
        CUST_KERNEL_LOG_ERROR(ctx, "Reciprocal kernel input[%d] cannot be 0", i);
        return KERNEL_STATUS_INNER_ERROR;
      }
      output_y[i] = static_cast<T>(1) / (input_x[i]);
    }
  } else {
    uint32_t min_core_num = 1;
    uint64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto shared_reciprocal = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        if (input_x[i] == static_cast<T>(0)) {
          CUST_KERNEL_LOG_ERROR(ctx, "Reciprocal kernel input[%d] cannot be 0", i);
          return KERNEL_STATUS_INNER_ERROR;
        }
        output_y[i] = static_cast<T>(1) / (input_x[i]);
      }
      return KERNEL_STATUS_OK;
    };
    if (max_core_num == 0) {
      CUST_KERNEL_LOG_ERROR(ctx, "max_core_num could not be 0.");
    }
    uint32_t ret = CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shared_reciprocal);
    if (ret != KERNEL_STATUS_OK) {
      CUST_KERNEL_LOG_ERROR(ctx, "CpuKernelUtils::ParallelFor failed");
      return KERNEL_STATUS_INNER_ERROR;
    }
    CUST_KERNEL_HANDLE_ERROR(ctx,
                             CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shared_reciprocal),
                             "Reciprocal Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ReciprocalCpuKernel::ReciprocalComputeComplex(Tensor *x, Tensor *y, uint64_t data_num, CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(x->GetData());
  auto output_y = reinterpret_cast<T *>(y->GetData());
  if (data_num <= kParallelDataNums) {
    for (size_t i = 0; i < data_num; i++) {
      output_y[i] = conj(input_x[i]) / (input_x[i].real() * input_x[i].real() + input_x[i].imag() * input_x[i].imag());
    }
  } else {
    uint32_t min_core_num = 1;
    uint64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shared_reciprocal = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        output_y[i] =
          conj(input_x[i]) / (input_x[i].real() * input_x[i].real() + input_x[i].imag() * input_x[i].imag());
      }
    };
    if (max_core_num == 0) {
      CUST_KERNEL_LOG_ERROR(ctx, "max_core_num could not be 0.");
    }
    uint32_t ret = CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shared_reciprocal);
    if (ret != KERNEL_STATUS_OK) {
      CUST_KERNEL_LOG_ERROR(ctx, "CpuKernelUtils::ParallelFor failed");
      return KERNEL_STATUS_INNER_ERROR;
    }
    CUST_KERNEL_HANDLE_ERROR(ctx,
                             CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shared_reciprocal),
                             "Reciprocal Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kReciprocal, ReciprocalCpuKernel);
}  // namespace aicpu
