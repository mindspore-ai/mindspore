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
#include "cpu_kernel/ms_kernel/logspace.h"

#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace aicpu {
namespace {
constexpr uint32_t kLogSpaceInputNum = 2;
constexpr uint32_t kLogSpaceOutputNum = 1;
const char *kLogSpace = "LogSpace";

#define LOGSPACE_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                                    \
    uint32_t result = LogSpaceCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                                \
      CUST_KERNEL_LOG_ERROR(ctx, "LogSpace kernel compute failed."); \
      return result;                                                 \
    }                                                                \
    break;                                                           \
  }

template <typename S, typename T>
void KernelCompute(CpuKernelContext &ctx) {
  auto *input_start_ = reinterpret_cast<S *>(ctx.Input(0)->GetData());
  auto *input_end_ = reinterpret_cast<S *>(ctx.Input(1)->GetData());
  auto input_start = static_cast<double>(input_start_[0]);
  auto input_end = static_cast<double>(input_end_[0]);
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  AttrValue *steps_data = ctx.GetAttr("steps");
  AttrValue *base_data = ctx.GetAttr("base");
  int64_t steps_value = 100;
  int base_value = 10;
  if (steps_data) {
    steps_value = steps_data->GetInt();
  }
  if (base_data) {
    base_value = base_data->GetInt();
  }
  if (steps_value != 1) {
    double b = (input_end - input_start) / (steps_value - 1);
    double q = pow(base_value, b);
    double input_start_value = input_start;
    for (int64_t i = 0; i < steps_value; i++) {
      double end_num = pow(base_value, input_start_value) * pow(q, i);
      *(output_y + i) = static_cast<T>(end_num);
    }
  }
  if (steps_value == 1) {
    double end_num = pow(base_value, static_cast<double>(input_start));
    *(output_y) = static_cast<T>(end_num);
  }
}
}  // namespace

uint32_t LogSpaceCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kLogSpaceInputNum, kLogSpaceOutputNum),
                           "[%s] check input and output failed.", kLogSpace);
  CUST_KERNEL_HANDLE_ERROR(ctx, LogSpaceCheck(ctx), "[%s] check params failed.", kLogSpace);
  DataType data_type = ctx.Output(0)->GetDataType();
  switch (data_type) {
    LOGSPACE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    LOGSPACE_COMPUTE_CASE(DT_FLOAT, float, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "LogSpace kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t LogSpaceCpuKernel::LogSpaceCheck(CpuKernelContext &ctx) {
  // get Attr steps_attr
  AttrValue *steps_attr_ptr = ctx.GetAttr("steps");
  if (steps_attr_ptr) {
    int64_t steps_data = steps_attr_ptr->GetInt();
    CUST_KERNEL_CHECK_FALSE(ctx, (steps_data >= 0), KERNEL_STATUS_PARAM_INVALID,
                            "Attr [steps] data has to be greater than or equal to 0.");
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t LogSpaceCpuKernel::LogSpaceCompute(CpuKernelContext &ctx) {
  DataType data_type_in = ctx.Input(0)->GetDataType();
  DataType data_type = ctx.Output(0)->GetDataType();
  if (data_type_in == data_type) {
    KernelCompute<T, T>(ctx);
  } else if (data_type_in == DT_FLOAT) {
    KernelCompute<float, T>(ctx);
  } else if (data_type_in == DT_FLOAT16) {
    KernelCompute<Eigen::half, T>(ctx);
  }
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kLogSpace, LogSpaceCpuKernel);
}  // namespace aicpu
