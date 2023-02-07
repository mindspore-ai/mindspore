/*
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
#include "non_deterministic_ints.h"

#include <cmath>
#include <ctime>
#include <iostream>
#include <random>

#include "cpu_ops_kernel.h"
#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"

namespace {
const char *kNonDeterministicInts = "NonDeterministicInts";
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 1;
const uint32_t kInputDims = 1;
const uint32_t kInputSizes = 2;
constexpr int64_t kParallelDataNums = 7 * 1024;
}  // namespace

namespace aicpu {
template <typename T1, typename T2>
uint32_t NonDeterministicIntsCpuKernel::DoCompute(CpuKernelContext &ctx) {
  Tensor *input = ctx.Input(0);
  Tensor *output = ctx.Output(0);
  auto input_nums = input->NumElements();
  auto input_data = reinterpret_cast<T2 *>(input->GetData());
  auto output_data = reinterpret_cast<T1 *>(output->GetData());
  auto output_nums = ctx.Output(0)->NumElements();
  auto max_data = std::numeric_limits<T1>::max();
  std::vector<int64_t> out_put_dims;
  for (auto i = 0; i < input_nums; i++) {
    if (*(input_data + i) <= 0) {
      KERNEL_LOG_ERROR("Shape elements must be > 0.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    out_put_dims.push_back(input_data[i]);
  }
  if (output_nums <= kParallelDataNums) {
    std::default_random_engine seed(time(0));
    std::uniform_int_distribution<T1> u(-max_data, max_data);
    for (auto j = 0; j < output_nums; j++) {
      *(output_data + j) = u(seed);
    }
  } else {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (max_core_num > output_nums) {
      max_core_num = output_nums;
    }
    auto shard_non_deterministic_ints = [&](int64_t start, int64_t end) {
      std::default_random_engine seed(time(0));
      std::uniform_int_distribution<T1> u(-max_data, max_data);
      for (auto j = start; j < end; j++) {
        *(output_data + j) = u(seed);
      }
    };
    KERNEL_HANDLE_ERROR(
      CpuKernelUtils::ParallelFor(ctx, output_nums, output_nums / max_core_num, shard_non_deterministic_ints),
      "NonDeterministicInts compute failed.");
  }
  output->GetTensorShape()->SetDimSizes(out_put_dims);
  return KERNEL_STATUS_OK;
}

uint32_t NonDeterministicIntsCpuKernel::DataAndTypeCheck(CpuKernelContext &ctx) {
  // the non null of input and output has been verified in NormalCheck
  Tensor *input = ctx.Input(0);
  auto input_data_nums = input->NumElements();
  auto data_type = input->GetDataType();
  KERNEL_CHECK_FALSE((data_type == DT_INT32 || data_type == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                     " Input type must be one of int32 or int64.");
  KERNEL_CHECK_FALSE((input_data_nums >= kInputSizes), KERNEL_STATUS_PARAM_INVALID, "Input data elements must >= 2.");
  KERNEL_CHECK_FALSE((input->GetTensorShape()->GetDimSizes().size() == kInputDims), KERNEL_STATUS_PARAM_INVALID,
                     "Input tensor must be a 1-D tensor.");
  return KERNEL_STATUS_OK;
}

uint32_t NonDeterministicIntsCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check params failed.", kNonDeterministicInts);
  KERNEL_HANDLE_ERROR(DataAndTypeCheck(ctx), " data or type check failed.");
  auto output_data_type = ctx.Output(0)->GetDataType();
  auto input_data_type = ctx.Input(0)->GetDataType();
  uint32_t ret = KERNEL_STATUS_OK;
  switch (output_data_type) {
    case DT_INT32: {
      if (input_data_type == DT_INT32) {
        ret = DoCompute<int32_t, int32_t>(ctx);
      } else {
        ret = DoCompute<int32_t, int64_t>(ctx);
      }
      break;
    }
    case DT_INT64: {
      if (input_data_type == DT_INT32) {
        ret = DoCompute<int64_t, int32_t>(ctx);
      } else {
        ret = DoCompute<int64_t, int64_t>(ctx);
      }
      break;
    }
    default: {
      KERNEL_LOG_ERROR("NonDeterministicInts kernel data type [%s] not support.", DTypeStr(output_data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), ret, "Compute failed.");
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kNonDeterministicInts, NonDeterministicIntsCpuKernel);
}  // namespace aicpu