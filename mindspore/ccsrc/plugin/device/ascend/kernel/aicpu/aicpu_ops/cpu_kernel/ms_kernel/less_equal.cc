/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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

#include "less_equal.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kLessEqual = "LessEqual";
// when input data size is more than kParallelDataNum, use Parallel func
const int64_t kParallelDataNum = 2 * 1024;
const int64_t kParallelDataNumMid = 16 * 1024;
const int64_t kParallelDataNumSameShape = 7 * 1024;
const int64_t kParallelDataNumSameShapeMid = 35 * 1024;

#define LESS_EQUAL_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                            \
    uint32_t result = LessEqualCompute<TYPE>(CTX);           \
    if (result != KERNEL_STATUS_OK) {                        \
      KERNEL_LOG_ERROR("Less Equal kernel compute failed."); \
      return result;                                         \
    }                                                        \
    break;                                                   \
  }
}  // namespace

namespace aicpu {
uint32_t LessEqualCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Less Equal check input and output number failed.");
  KERNEL_HANDLE_ERROR(LessEqualParamCheck(ctx), "Less Equal check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    LESS_EQUAL_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    LESS_EQUAL_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    LESS_EQUAL_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    LESS_EQUAL_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    LESS_EQUAL_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    LESS_EQUAL_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    LESS_EQUAL_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    LESS_EQUAL_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    LESS_EQUAL_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    LESS_EQUAL_COMPUTE_CASE(DT_FLOAT, float, ctx)
    LESS_EQUAL_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("Less Equal kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t LessEqualCpuKernel::LessEqualParamCheck(CpuKernelContext &ctx) {
  // the non null of input_0, input_1, output has been verified in NormalCheck
  Tensor *input_0 = ctx.Input(0);
  Tensor *input_1 = ctx.Input(1);
  Tensor *output = ctx.Output(0);
  DataType input0_type = input_0->GetDataType();
  DataType input1_type = input_1->GetDataType();
  KERNEL_CHECK_FALSE((input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input0 [%s] need be same with "
                     "input1 [%s].",
                     DTypeStr(input0_type).c_str(), DTypeStr(input1_type).c_str())
  KERNEL_LOG_DEBUG(
    "LessEqualCpuKernel[%s], input0: size[%llu];"
    "input1: size[%llu], output: size[%llu].",
    ctx.GetOpType().c_str(), input_0->GetDataSize(), input_1->GetDataSize(), output->GetDataSize());

  return KERNEL_STATUS_OK;
}

// special compute is used in the following situations.
// 1. the shapes of input1 and input2 are the same
// 2. input1 is a 1D tensor with only one element or input1 is scalar
// 3. input2 is a 1D tensor with only one element or input2 is scalar
// 4. the shapes of input1 and input2 are different
template <typename T>
void LessEqualCpuKernel::SpecialCompute(BcastShapeType type, int64_t start, int64_t end, const T *input1,
                                        const T *input2, bool *output) {
  switch (type) {
    case BcastShapeType::SAME_SHAPE:
      for (int64_t i = start; i < end; ++i) {
        *(output + i) = *(input1 + i) <= *(input2 + i);
      }
      break;
    case BcastShapeType::X_ONE_ELEMENT:
      for (int64_t i = start; i < end; ++i) {
        *(output + i) = *input1 <= *(input2 + i);
      }
      break;
    case BcastShapeType::Y_ONE_ELEMENT:
      for (int64_t i = start; i < end; ++i) {
        *(output + i) = *(input1 + i) <= *input2;
      }
      break;
    default:
      KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(type));
      break;
  }
}

template <typename T>
uint32_t LessEqualCpuKernel::NoBcastCompute(CpuKernelContext &ctx) {
  auto in0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto in1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto out = reinterpret_cast<bool *>(ctx.Output(0)->GetData());
  int64_t in0_elements_nums = ctx.Input(0)->NumElements();
  int64_t in1_elements_nums = ctx.Input(1)->NumElements();
  int64_t data_num = ctx.Output(0)->NumElements();
  BcastShapeType type = in0_elements_nums == in1_elements_nums
                          ? BcastShapeType::SAME_SHAPE
                          : (in0_elements_nums == 1 ? BcastShapeType::X_ONE_ELEMENT : BcastShapeType::Y_ONE_ELEMENT);

  if (data_num >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

    if (data_num <= kParallelDataNumSameShapeMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto sharder_less = [&](int64_t start, int64_t end) { SpecialCompute<T>(type, start, end, in0, in1, out); };

    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num can not be zero");
    } else {
      KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_less),
                          "Less Equal Compute failed.");
    }
  } else {
    SpecialCompute<T>(type, 0, data_num, in0, in1, out);
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t LessEqualCpuKernel::BcastCompute(CpuKernelContext &ctx, Bcast &bcast) {
  auto in0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto in1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto out = reinterpret_cast<bool *>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Output(0)->NumElements();
  if (data_num >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

    if (data_num <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto sharder_less = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; ++i) {
        *(out + i) = *(in0 + bcast.GetBroadcastXIndex(i)) <= *(in1 + bcast.GetBroadcastYIndex(i)) ? true : false;
      }
    };

    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num can not be zero");
    } else {
      KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_less),
                          "Less Equal Compute failed.");
    }
  } else {
    for (int64_t i = 0; i < data_num; ++i) {
      *(out + i) = *(in0 + bcast.GetBroadcastXIndex(i)) <= *(in1 + bcast.GetBroadcastYIndex(i)) ? true : false;
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t LessEqualCpuKernel::LessEqualCompute(CpuKernelContext &ctx) {
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
    Bcast bcast(input0_shape, input1_shape);
    if (!bcast.IsValid()) {
      KERNEL_LOG_ERROR("[%s] broadcast failed.", ctx.GetOpType().c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }

    return BcastCompute<T>(ctx, bcast);
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kLessEqual, LessEqualCpuKernel);
}  // namespace aicpu
