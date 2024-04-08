/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "cpu_kernel/ms_kernel/sinc.h"
#include <algorithm>
#include <complex>
#include <set>
#include <vector>
#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
constexpr double kPI = 3.14159265358979323846L;
constexpr uint32_t kSincInputNum = 1;
constexpr uint32_t kSincOutputNum = 1;
const int64_t paralled_data_size = 64 * 1024;
const char *kSinc = "Sinc";
}  // namespace

namespace aicpu {
template <typename T>
uint32_t SincCpuKernel::SincTypeSameCompute(CpuKernelContext &ctx) {
  T *x_addr = static_cast<T *>(ctx.Input(0)->GetData());
  auto y_addr = static_cast<T *>(ctx.Output(0)->GetData());
  size_t x_size = ctx.Input(0)->NumElements();
  size_t date_size = x_size * sizeof(T);
  if (date_size <= paralled_data_size) {
    for (size_t i = 0; i < x_size; i++) {
      if (x_addr[i] == T(0.0f)) {
        y_addr[i] = T(1.0f);
      } else {
        T product = T(kPI) * x_addr[i];
        y_addr[i] = sin(product) / product;
      }
    }
  } else {
    auto shard_sinc = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        if (x_addr[i] == T(0.0f)) {
          y_addr[i] = T(1.0f);
        } else {
          T product = T(kPI) * x_addr[i];
          y_addr[i] = sin(product) / product;
        }
      }
    };
    uint32_t min_core_num = 1;
    size_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num == 0) {
      return KERNEL_STATUS_PARAM_INVALID;
    }
    if (max_core_num > date_size) {
      max_core_num = date_size;
    }
    CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, x_size, x_size / max_core_num, shard_sinc),
                             "Sinc Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SincCpuKernel::SincTypeChangeCompute(CpuKernelContext &ctx) {
  T *x_addr = static_cast<T *>(ctx.Input(0)->GetData());
  auto y_addr = static_cast<float *>(ctx.Output(0)->GetData());
  size_t x_size = ctx.Input(0)->NumElements();
  size_t date_size = x_size * sizeof(T);
  if (date_size <= paralled_data_size) {
    for (size_t i = 0; i < x_size; i++) {
      if (x_addr[i] == T(0.0f)) {
        y_addr[i] = static_cast<float>(1.0f);
      } else {
        float product = static_cast<float>(kPI) * x_addr[i];
        y_addr[i] = sin(product) / product;
      }
    }
  } else {
    auto shard_sinc = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        if (x_addr[i] == T(0.0f)) {
          y_addr[i] = static_cast<float>(1.0f);
        } else {
          float product = static_cast<float>(kPI) * x_addr[i];
          y_addr[i] = sin(product) / product;
        }
      }
    };
    uint32_t min_core_num = 1;
    size_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num == 0) {
      return KERNEL_STATUS_PARAM_INVALID;
    }
    if (max_core_num > date_size) {
      max_core_num = date_size;
    }
    CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, x_size, x_size / max_core_num, shard_sinc),
                             "Sinc Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SincCpuKernel::SincBoolCompute(CpuKernelContext &ctx) {
  bool *x_addr = static_cast<bool *>(ctx.Input(0)->GetData());
  auto y_addr = static_cast<float *>(ctx.Output(0)->GetData());
  size_t x_size = ctx.Input(0)->NumElements();
  size_t date_size = x_size * sizeof(T);
  if (date_size <= paralled_data_size) {
    for (size_t i = 0; i < x_size; i++) {
      float tmp;
      if (x_addr[i] == true) {
        tmp = 1.0f;
      } else {
        tmp = 0.0f;
      }
      float product = static_cast<float>(kPI) * tmp;
      y_addr[i] = sin(product) / product;
    }
  } else {
    auto shard_sinc = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        float tmp;
        if (x_addr[i] == true) {
          tmp = 1.0f;
        } else {
          tmp = 0.0f;
        }
        float product = static_cast<float>(kPI) * tmp;
        y_addr[i] = sin(product) / product;
      }
    };
    uint32_t min_core_num = 1;
    size_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num == 0) {
      return KERNEL_STATUS_PARAM_INVALID;
    }
    if (max_core_num > date_size) {
      max_core_num = date_size;
    }
    CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, x_size, x_size / max_core_num, shard_sinc),
                             "Sinc Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

inline std::uint32_t SincExtraCheck(CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetData() == nullptr) {
    CUST_KERNEL_LOG_ERROR(ctx, "Get input data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Output(0)->GetData() == nullptr) {
    CUST_KERNEL_LOG_ERROR(ctx, "Get output data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  DataType in_dtype = ctx.Input(0)->GetDataType();
  DataType out_dtype = ctx.Output(0)->GetDataType();
  std::set<DataType> dtypes;
  dtypes.insert(DT_FLOAT16);
  dtypes.insert(DT_FLOAT);
  dtypes.insert(DT_DOUBLE);
  dtypes.insert(DT_COMPLEX64);
  dtypes.insert(DT_COMPLEX128);
  if (dtypes.count(in_dtype) == 1) {
    if (out_dtype != in_dtype) {
      CUST_KERNEL_LOG_ERROR(
        ctx, "The data type of the output need be the same as the input when input is [%s], but got [%s].",
        DTypeStr(in_dtype).c_str(), DTypeStr(out_dtype).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  } else {
    if (out_dtype != DT_FLOAT) {
      CUST_KERNEL_LOG_ERROR(
        ctx, "The data type of the output must be float32 when the dtype of input is [%s], but got [%s].",
        DTypeStr(in_dtype).c_str(), DTypeStr(out_dtype).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  std::vector<int64_t> input_dims = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> output_dims = ctx.Output(0)->GetTensorShape()->GetDimSizes();
  if (input_dims.size() != output_dims.size()) {
    CUST_KERNEL_LOG_ERROR(ctx,
                          "The data dim size of the input [%llu] need be the same as the output "
                          "[%llu].",
                          input_dims.size(), output_dims.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (size_t index = 0; index < input_dims.size(); index++) {
    if (input_dims[index] != output_dims[index]) {
      CUST_KERNEL_LOG_ERROR(ctx, "The data dim of the input need be the same as the output.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t SincCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kSincInputNum, kSincOutputNum), "[%s] check params failed.", kSinc);
  uint32_t res = KERNEL_STATUS_OK;
  res = SincExtraCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    return res;
  }
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    case DT_FLOAT16:
      res = SincTypeSameCompute<Eigen::half>(ctx);
      break;
    case DT_FLOAT:
      res = SincTypeSameCompute<float>(ctx);
      break;
    case DT_DOUBLE:
      res = SincTypeSameCompute<double>(ctx);
      break;
    case DT_INT8:
      res = SincTypeChangeCompute<int8_t>(ctx);
      break;
    case DT_UINT8:
      res = SincTypeChangeCompute<uint8_t>(ctx);
      break;
    case DT_INT16:
      res = SincTypeChangeCompute<int16_t>(ctx);
      break;
    case DT_UINT16:
      res = SincTypeChangeCompute<uint16_t>(ctx);
      break;
    case DT_INT32:
      res = SincTypeChangeCompute<int32_t>(ctx);
      break;
    case DT_UINT32:
      res = SincTypeChangeCompute<uint32_t>(ctx);
      break;
    case DT_INT64:
      res = SincTypeChangeCompute<int64_t>(ctx);
      break;
    case DT_UINT64:
      res = SincTypeChangeCompute<uint64_t>(ctx);
      break;
    case DT_COMPLEX64:
      res = SincTypeSameCompute<std::complex<float>>(ctx);
      break;
    case DT_COMPLEX128:
      res = SincTypeSameCompute<std::complex<double>>(ctx);
      break;
    case DT_BOOL:
      res = SincBoolCompute<bool>(ctx);
      break;
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Sinc invalid input type [%s]", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (res != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kSinc, SincCpuKernel);
}  // namespace aicpu
