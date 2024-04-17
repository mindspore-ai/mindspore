/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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

#include "cpu_kernel/ms_kernel/randperm.h"

#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include <array>

#include "context/inc/cpu_kernel_utils.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *const kRandperm = "Randperm";
#define RANDPERM_COMPUTE_CASE(DTYPE, TYPE) \
  case (DTYPE): {                          \
    ret = RandpermCompute<TYPE>(ctx);      \
    break;                                 \
  }
}  // namespace

namespace aicpu {
uint32_t RandpermCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "randperm check input and output number failed.");
  auto output_type = ctx.Output(0)->GetDataType();
  uint32_t ret;
  switch (output_type) {
    RANDPERM_COMPUTE_CASE(DT_INT8, int8_t)
    RANDPERM_COMPUTE_CASE(DT_INT16, int16_t)
    RANDPERM_COMPUTE_CASE(DT_INT32, int32_t)
    RANDPERM_COMPUTE_CASE(DT_INT64, int64_t)
    RANDPERM_COMPUTE_CASE(DT_UINT8, uint8_t)
    RANDPERM_COMPUTE_CASE(DT_UINT16, uint16_t)
    RANDPERM_COMPUTE_CASE(DT_UINT32, uint32_t)
    RANDPERM_COMPUTE_CASE(DT_UINT64, uint64_t)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Output data type [%s] not support.", DTypeStr(output_type).c_str());
      ret = KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
}

template <typename T>
uint32_t RandpermCpuKernel::RandpermCompute(CpuKernelContext &ctx) {
  auto input_n = ctx.Input(0);
  auto n_dtype = input_n->GetDataType();
  auto n_data = input_n->GetData();
  CUST_KERNEL_CHECK_FALSE(ctx, (n_dtype == DT_INT32 || n_dtype == DT_INT64), KERNEL_STATUS_INNER_ERROR,
                          "Only support int32_t and int64_t for input 'n'.");
  auto n = n_dtype == DT_INT32 ? *static_cast<int32_t *>(n_data) : *static_cast<int64_t *>(n_data);

  auto max_length_attr = ctx.GetAttr("max_length");
  CUST_KERNEL_CHECK_NULLPTR(ctx, max_length_attr, KERNEL_STATUS_INNER_ERROR, "Get Attr [max_length] failed.");
  auto max_length = max_length_attr->GetInt();
  auto pad_attr = ctx.GetAttr("pad");
  CUST_KERNEL_CHECK_NULLPTR(ctx, pad_attr, KERNEL_STATUS_INNER_ERROR, "Get Attr [pad] failed.");
  auto pad = pad_attr->GetInt();

  CUST_KERNEL_CHECK_FALSE(ctx, n <= max_length, KERNEL_STATUS_INNER_ERROR, "'n'[%d] is greater than 'max_length'[%d].",
                          n, max_length);

  auto output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  std::iota(output_data, output_data + n, 0);
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(output_data, output_data + n, g);

  for (int i = n; i < max_length; ++i) {
    output_data[i] = static_cast<T>(pad);
  }
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kRandperm, RandpermCpuKernel);
}  // namespace aicpu
