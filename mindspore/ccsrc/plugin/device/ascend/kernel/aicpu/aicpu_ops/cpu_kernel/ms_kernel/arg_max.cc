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
#include "cpu_kernel/ms_kernel/arg_max.h"
#include <vector>
#include <algorithm>
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kArgMax = "ArgMax";
const uint32_t kDataSize = 4 * 1024;
#define ARGMAX_COMPUTE_CASE(DTYPE, TYPE1, TYPE2, TYPE3, CTX)   \
  case (DTYPE): {                                              \
    uint32_t result = ArgMaxCompute<TYPE1, TYPE2, TYPE3>(CTX); \
    if (result != KERNEL_STATUS_OK) {                          \
      KERNEL_LOG_ERROR("ArgMax kernel compute failed.");       \
      return result;                                           \
    }                                                          \
    break;                                                     \
  }

#define ARGMAX_COMPUTE_CASE_ALL(TYPE1, TYPE2, CTX)                \
  ARGMAX_COMPUTE_CASE(DT_DOUBLE, double, TYPE1, TYPE2, CTX)       \
  ARGMAX_COMPUTE_CASE(DT_FLOAT, float, TYPE1, TYPE2, CTX)         \
  ARGMAX_COMPUTE_CASE(DT_FLOAT16, Eigen::half, TYPE1, TYPE2, CTX) \
  ARGMAX_COMPUTE_CASE(DT_INT8, int8_t, TYPE1, TYPE2, CTX)         \
  ARGMAX_COMPUTE_CASE(DT_INT16, int16_t, TYPE1, TYPE2, CTX)       \
  ARGMAX_COMPUTE_CASE(DT_INT32, int32_t, TYPE1, TYPE2, CTX)       \
  ARGMAX_COMPUTE_CASE(DT_INT64, int64_t, TYPE1, TYPE2, CTX)       \
  ARGMAX_COMPUTE_CASE(DT_UINT8, uint8_t, TYPE1, TYPE2, CTX)       \
  ARGMAX_COMPUTE_CASE(DT_UINT16, uint16_t, TYPE1, TYPE2, CTX)     \
  ARGMAX_COMPUTE_CASE(DT_UINT32, uint32_t, TYPE1, TYPE2, CTX)     \
  ARGMAX_COMPUTE_CASE(DT_UINT64, uint64_t, TYPE1, TYPE2, CTX)

#define LOG_ERROR_DTYPE(STR1, STR2)                                    \
  default:                                                             \
    KERNEL_LOG_ERROR("[%s] data type[%s] not supported.", STR1, STR2); \
    return KERNEL_STATUS_PARAM_INVALID;
}  // namespace

namespace aicpu {
uint32_t ArgMaxCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "ArgMax check input and output number failed.");
  Tensor *input_data = ctx.Input(0);
  Tensor *axes_data = ctx.Input(1);
  Tensor *output_data = ctx.Output(0);
  auto data_type = input_data->GetDataType();
  auto axes_type = axes_data->GetDataType();
  auto output_type = output_data->GetDataType();
  switch (output_type) {
    case DT_INT32:
      switch (axes_type) {
        case DT_INT32:
          switch (data_type) {
            ARGMAX_COMPUTE_CASE_ALL(int32_t, int32_t, ctx)
            LOG_ERROR_DTYPE("Input[0]", DTypeStr(data_type).c_str())
          }
          break;
        case DT_INT64:
          switch (data_type) {
            ARGMAX_COMPUTE_CASE_ALL(int64_t, int32_t, ctx)
            LOG_ERROR_DTYPE("Input[0]", DTypeStr(data_type).c_str())
          }
          break;
          LOG_ERROR_DTYPE("Input[1]", DTypeStr(axes_type).c_str())
      }
      break;
    case DT_INT64:
      switch (axes_type) {
        case DT_INT32:
          switch (data_type) {
            ARGMAX_COMPUTE_CASE_ALL(int32_t, int64_t, ctx)
            LOG_ERROR_DTYPE("Input[0]", DTypeStr(data_type).c_str())
          }
          break;
        case DT_INT64:
          switch (data_type) {
            ARGMAX_COMPUTE_CASE_ALL(int64_t, int64_t, ctx)
            LOG_ERROR_DTYPE("Input[0]", DTypeStr(data_type).c_str())
          }
          break;
          LOG_ERROR_DTYPE("Input[1]", DTypeStr(axes_type).c_str())
      }
      break;
      LOG_ERROR_DTYPE("Output[0]", DTypeStr(output_type).c_str())
  }
  return KERNEL_STATUS_OK;
}

template <typename T1, typename T2, typename T3>
uint32_t ArgMaxCpuKernel::ArgMaxCompute(const CpuKernelContext &ctx) {
  // get x
  Tensor *input_data = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(input_data->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 0 data failed.")
  auto input_data_addr = reinterpret_cast<T1 *>(input_data->GetData());
  auto input_shape = input_data->GetTensorShape();
  std::vector<int64_t> dims = input_shape->GetDimSizes();
  const int32_t kDimsNum = input_shape->GetDims();
  int64_t dims_addr[kDimsNum];
  int64_t tmp0 = 1;
  for (int32_t i = kDimsNum - 1; i > -1; i--) {
    dims_addr[i] = tmp0;
    tmp0 *= dims[i];
  }
  // get dimension
  Tensor *axes_data = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(axes_data->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 1 data failed.")
  auto axes_data_addr = reinterpret_cast<T2 *>(axes_data->GetData());
  if (axes_data_addr[0] > kDimsNum - 1 || axes_data_addr[0] < -kDimsNum) {
    KERNEL_LOG_ERROR("The value of axes must be in the range [[%d], [%d]], but got [%d]", -kDimsNum, kDimsNum - 1,
                     axes_data_addr[0]);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (axes_data_addr[0] < 0) {
    axes_data_addr[0] += kDimsNum;
  }
  // get y
  Tensor *output_data = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output_data->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output 0 data failed.")
  auto output_data_addr = reinterpret_cast<T3 *>(output_data->GetData());
  int64_t output_data_num = output_data->NumElements();
  if (output_data_num * sizeof(T3) < kDataSize) {
    int64_t output_seq[kDimsNum];
    output_seq[axes_data_addr[0]] = 0;
    for (int64_t i = 0; i < output_data_num; i++) {
      int64_t tmp = i;
      int64_t addr_base = 0;
      for (int64_t j = kDimsNum - 1; j > -1; j--) {
        if (j == axes_data_addr[0]) {
          continue;
        }
        output_seq[j] = tmp % dims[j];
        addr_base += output_seq[j] * dims_addr[j];
        tmp /= dims[j];
      }
      T1 max_value = input_data_addr[addr_base];
      T3 max_loc = 0;
      for (int64_t j = 1; j < dims[axes_data_addr[0]]; j++) {
        int64_t get_addr = addr_base + j * dims_addr[axes_data_addr[0]];
        T1 get_data = input_data_addr[get_addr];
        if (max_value < get_data) {
          max_value = get_data;
          max_loc = j;
        }
      }
      output_data_addr[i] = max_loc;
    }
  } else {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (max_core_num > output_data_num) {
      max_core_num = output_data_num;
    }
    auto shard_compute = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        int64_t output_seq[kDimsNum];
        output_seq[axes_data_addr[0]] = 0;
        int64_t tmp = i;
        int64_t addr_base = 0;
        for (int64_t j = kDimsNum - 1; j > -1; j--) {
          if (j == axes_data_addr[0]) {
            continue;
          }
          output_seq[j] = tmp % dims[j];
          addr_base += output_seq[j] * dims_addr[j];
          tmp /= dims[j];
        }
        T1 max_value = input_data_addr[addr_base];
        T3 max_loc = 0;
        for (int64_t j = 1; j < dims[axes_data_addr[0]]; j++) {
          int64_t get_addr = addr_base + j * dims_addr[axes_data_addr[0]];
          T1 get_data = input_data_addr[get_addr];
          if (max_value < get_data) {
            max_value = get_data;
            max_loc = j;
          }
        }
        output_data_addr[i] = max_loc;
      }
    };
    KERNEL_HANDLE_ERROR(
      CpuKernelUtils::ParallelFor(ctx, output_data_num, output_data_num / max_core_num, shard_compute),
      "ArgMax Compute failed.");
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kArgMax, ArgMaxCpuKernel);
}  // namespace aicpu
