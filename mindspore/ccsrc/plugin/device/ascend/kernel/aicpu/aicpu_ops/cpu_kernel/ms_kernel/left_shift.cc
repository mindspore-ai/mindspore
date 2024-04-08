/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

#include "cpu_kernel/ms_kernel/left_shift.h"

#include <bits/stdc++.h>
#include <algorithm>

#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kLeftShift = "LeftShift";
const int64_t kParallelDataNum = 2 * 1024;
const int64_t kParallelDataNumMid = 16 * 1024;
const int64_t kParallelDataNumSameShape = 7 * 1024;
const int64_t kParallelDataNumSameShapeMid = 35 * 1024;

#define LEFT_SHIFT_COMPUTE_CASE(DTYPE, TYPE, CTX)                     \
  case (DTYPE): {                                                     \
    uint32_t result = LeftShiftCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                                 \
      CUST_KERNEL_LOG_ERROR(ctx, "LeftShift kernel compute failed."); \
      return result;                                                  \
    }                                                                 \
    break;                                                            \
  }
}  // namespace

namespace aicpu {
uint32_t LeftShiftCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "LeftShift check input and output number failed.");
  CUST_KERNEL_HANDLE_ERROR(ctx, LeftShiftParamCheck(ctx), "LeftShift check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    LEFT_SHIFT_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    LEFT_SHIFT_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    LEFT_SHIFT_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    LEFT_SHIFT_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    LEFT_SHIFT_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    LEFT_SHIFT_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    LEFT_SHIFT_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    LEFT_SHIFT_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)

    default:
      CUST_KERNEL_LOG_ERROR(ctx, "LeftShift kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t LeftShiftCpuKernel::LeftShiftParamCheck(CpuKernelContext &ctx) {
  Tensor *input_0 = ctx.Input(0);
  Tensor *input_1 = ctx.Input(1);
  Tensor *output = ctx.Output(0);
  DataType input0_type = input_0->GetDataType();
  DataType input1_type = input_1->GetDataType();
  CUST_KERNEL_CHECK_FALSE(ctx, (input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID,
                          "The data type of input0 [%s] need be same with "
                          "input1 [%s].",
                          DTypeStr(input0_type).c_str(), DTypeStr(input1_type).c_str())
  CUST_KERNEL_LOG_DEBUG(ctx,
                        "LeftShiftCpuKernel[%s], input0: size[%llu];"
                        "input1: size[%llu], output: size[%llu].",
                        ctx.GetOpType().c_str(), input_0->GetDataSize(), input_1->GetDataSize(), output->GetDataSize());

  return KERNEL_STATUS_OK;
}

template <typename T>
void LeftShiftCpuKernel::SpecialCompute(BcastShapeType type, int64_t start, int64_t end, CpuKernelContext &ctx) {
  auto input1 = static_cast<T *>(ctx.Input(0)->GetData());
  auto input2 = static_cast<T *>(ctx.Input(1)->GetData());
  auto output = static_cast<T *>(ctx.Output(0)->GetData());
  switch (type) {
    case BcastShapeType::SAME_SHAPE: {
      for (int64_t i = start; i < end; ++i) {
        T mid = *(input2 + i);
        T flag = sizeof(T) * 8 > 32 ? sizeof(T) * 8 : 32;
        if (mid > flag || mid < -flag) {
          mid = mid % flag;
        }
        *(output + i) = *(input1 + i) << mid;
      }
      break;
    }
    case BcastShapeType::X_ONE_ELEMENT: {
      for (int64_t i = start; i < end; ++i) {
        T mid = *(input2 + i);
        T flag = sizeof(T) * 8 > 32 ? sizeof(T) * 8 : 32;
        if (mid > flag || mid < -flag) {
          mid = mid % flag;
        }
        *(output + i) = *input1 << mid;
      }
      break;
    }
    case BcastShapeType::Y_ONE_ELEMENT: {
      T mid = *input2;
      T flag = sizeof(T) * 8 > 32 ? sizeof(T) * 8 : 32;
      if (mid > flag || mid < -flag) {
        mid = mid % flag;
      }
      for (int64_t i = start; i < end; ++i) {
        *(output + i) = *(input1 + i) << mid;
      }
      break;
    }
    default:
      CUST_KERNEL_LOG_WARN(ctx, "Invalid type [%d]", static_cast<int32_t>(type));
  }
}

template <typename T>
uint32_t LeftShiftCpuKernel::NoBcastCompute(CpuKernelContext &ctx) {
  int64_t input_0_elements_nums = ctx.Input(0)->NumElements();
  int64_t input_1_elements_nums = ctx.Input(1)->NumElements();
  int64_t data_num = ctx.Output(0)->NumElements();
  BcastShapeType type =
    input_0_elements_nums == input_1_elements_nums
      ? BcastShapeType::SAME_SHAPE
      : (input_0_elements_nums == 1 ? BcastShapeType::X_ONE_ELEMENT : BcastShapeType::Y_ONE_ELEMENT);

  if (data_num >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));

    if (data_num <= kParallelDataNumSameShapeMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    auto sharder_left_shift = [&](int64_t start, int64_t end) { SpecialCompute<T>(type, start, end, ctx); };

    CUST_KERNEL_HANDLE_ERROR(ctx,
                             CpuKernelUtils::ParallelFor(
                               ctx, data_num, data_num / ((max_core_num > 0) ? max_core_num : 1), sharder_left_shift),
                             "LeftShift Compute failed.");
  } else {
    SpecialCompute<T>(type, 0, data_num, ctx);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t LeftShiftCpuKernel::BcastCompute(CpuKernelContext &ctx, const Bcast &bcast) {
  auto input_0 = static_cast<T *>(ctx.Input(0)->GetData());
  auto input_1 = static_cast<T *>(ctx.Input(1)->GetData());
  auto output = static_cast<T *>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Output(0)->NumElements();
  if (data_num >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));

    if (data_num <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);
    }

    auto sharder_left_shift = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; ++i) {
        T mid = *(input_1 + bcast.GetBroadcastYIndex(i));
        T flag = sizeof(T) * 8 > 32 ? sizeof(T) * 8 : 32;
        if (mid > flag || mid < -flag) {
          mid = mid % flag;
        }
        *(output + i) = *(input_0 + bcast.GetBroadcastXIndex(i)) << mid;
      }
    };

    if (max_core_num == 0) {
      max_core_num = 1;
    }

    CUST_KERNEL_HANDLE_ERROR(ctx,
                             CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_left_shift),
                             "LeftShift Compute failed.");
  } else {
    for (int64_t i = 0; i < data_num; ++i) {
      T mid = *(input_1 + bcast.GetBroadcastYIndex(i));
      T flag = sizeof(T) * 8 > 32 ? sizeof(T) * 8 : 32;
      if (mid > flag || mid < -flag) {
        mid = mid % flag;
      }
      *(output + i) = *(input_0 + bcast.GetBroadcastXIndex(i)) << mid;
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t LeftShiftCpuKernel::LeftShiftCompute(CpuKernelContext &ctx) {
  Tensor *input0_tensor = ctx.Input(0);
  auto input0_shape = input0_tensor->GetTensorShape()->GetDimSizes();
  int64_t input0_elements_nums = input0_tensor->NumElements();

  Tensor *input1_tensor = ctx.Input(1);
  auto input1_shape = input1_tensor->GetTensorShape()->GetDimSizes();
  int64_t input1_elements_nums = input1_tensor->NumElements();

  bool isNeedBcast = (input0_shape == input1_shape) || (input0_elements_nums == 1) || (input1_elements_nums == 1);
  if (isNeedBcast) {
    return NoBcastCompute<T>(ctx);
  } else {
    Bcast bcast(ctx, input0_shape, input1_shape);
    if (!bcast.IsValid()) {
      CUST_KERNEL_LOG_ERROR(ctx, "[%s] broadcast failed.", ctx.GetOpType().c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }

    return BcastCompute<T>(ctx, bcast);
  }
}

REGISTER_MS_CPU_KERNEL(kLeftShift, LeftShiftCpuKernel);
}  // namespace aicpu
