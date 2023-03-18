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

#include "igammagrada.h"
#include "igamma_utils.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kigammagrada = "IgammaGradA";
constexpr int64_t kParallelDataNums = 128;

#define SWITCH_PARALLEL(SHARD, end_num)                                                                 \
  if (data_num <= kParallelDataNums) {                                                                  \
    SHARD(0, end_num);                                                                                  \
  } else {                                                                                              \
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, (end_num), (end_num) / (max_core_num), SHARD), \
                        "IgammaGradA SHARD Compute failed.");                                           \
  }

#define IGAMMA_COMPUTE_CASE(DTYPE, TYPE, CTX, CALCINFO)        \
  case (DTYPE): {                                              \
    uint32_t result = IgammaGradACompute<TYPE>(CTX, CALCINFO); \
    if (result != KERNEL_STATUS_OK) {                          \
      KERNEL_LOG_ERROR("IgammaGradA kernel compute failed.");  \
      return result;                                           \
    }                                                          \
    break;                                                     \
  }
}  // namespace

namespace aicpu {
uint32_t IgammaGradACpuKernel::Compute(CpuKernelContext &ctx) {
  // check param number
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "IgammaGradA check input and output number failed.");

  BCalcInfo calc_info;
  KERNEL_HANDLE_ERROR(IgammaGradACheckAndBroadCast(ctx, calc_info), "IgammaGradA check params or bcast failed.");

  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    IGAMMA_COMPUTE_CASE(DT_FLOAT, float, ctx, calc_info)
    IGAMMA_COMPUTE_CASE(DT_DOUBLE, double, ctx, calc_info)
    default:
      KERNEL_LOG_ERROR("IgammaGradA kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t IgammaGradACpuKernel::IgammaGradACheckAndBroadCast(CpuKernelContext &ctx, BCalcInfo &calc_info) {
  calc_info.input_0 = ctx.Input(kFirstInputIndex);
  calc_info.input_1 = ctx.Input(kSecondInputIndex);
  calc_info.output = ctx.Output(0);

  // check input datatype
  DataType input0_datatype = calc_info.input_0->GetDataType();
  KERNEL_CHECK_FALSE((input0_datatype == DT_DOUBLE || input0_datatype == DT_FLOAT), KERNEL_STATUS_PARAM_INVALID,
                     "Input[0] data type must DT_FLOAT or DT_DOUBLE,"
                     "but got data type[%s].",
                     DTypeStr(input0_datatype).c_str());

  DataType input1_datatype = calc_info.input_1->GetDataType();
  KERNEL_CHECK_FALSE((input0_datatype == input1_datatype), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input1 [%s] need be same with "
                     "input0 [%s].",
                     DTypeStr(input1_datatype).c_str(), DTypeStr(input0_datatype).c_str())

  // check output dtype
  DataType output_datatype = calc_info.output->GetDataType();
  KERNEL_CHECK_FALSE((input0_datatype == output_datatype), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of output [%s] need be same with "
                     "input0 [%s].",
                     DTypeStr(output_datatype).c_str(), DTypeStr(input0_datatype).c_str())

  KERNEL_LOG_DEBUG(
    "IgammaGradACpuKernel[%s], input0: size[%llu];"
    "input1: size[%llu], output: size[%llu].",
    ctx.GetOpType().c_str(), calc_info.input_0->GetDataSize(), calc_info.input_1->GetDataSize(),
    calc_info.output->GetDataSize());

  Bcast bcast;
  KERNEL_HANDLE_ERROR(bcast.GenerateBcastInfo(calc_info), "Generate broadcast info failed.");
  (void)bcast.BCastIndexes(calc_info.x_indexes, calc_info.y_indexes);
  (void)bcast.GetBcastVec(calc_info);

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t IgammaGradACpuKernel::IgammaGradACompute(CpuKernelContext &ctx, BCalcInfo &calc_info) {
  auto input_x1 = reinterpret_cast<T *>(calc_info.input_0->GetData());
  auto input_x2 = reinterpret_cast<T *>(calc_info.input_1->GetData());
  auto output_y = reinterpret_cast<T *>(calc_info.output->GetData());

  int64_t data_num = calc_info.x_indexes.size();
  uint32_t min_core_num = 1;
  int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);

  if (max_core_num > data_num) {
    max_core_num = data_num;
  }

  if (max_core_num == 0) {
    max_core_num = 1;
  }

  auto shard_igammagrada = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      T *x1_index = input_x1 + calc_info.x_indexes[i];  // i-th value of input0
      T *x2_index = input_x2 + calc_info.y_indexes[i];  // i-th value of input1
      *(output_y + i) = IgammaGradASingle<T>(*x1_index, *x2_index);
    }
  };

  SWITCH_PARALLEL(shard_igammagrada, data_num);
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kigammagrada, IgammaGradACpuKernel);
}  // namespace aicpu