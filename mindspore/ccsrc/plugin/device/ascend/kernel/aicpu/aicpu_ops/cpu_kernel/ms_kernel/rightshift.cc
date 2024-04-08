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

#include "ms_kernel/rightshift.h"
#include <algorithm>
#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *const kRightShift = "RightShift";
// when input data size is more than kParallelDataNum, use Parallel func
const int64_t kParallelDataNum = 2 * 1024;
const int64_t kParallelDataNumMid = 16 * 1024;
const int64_t kParallelDataNumSameShape = 7 * 1024;
const int64_t kParallelDataNumSameShapeMid = 35 * 1024;

#define RIGHTSHIFT_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                                      \
    uint32_t result = RightShiftCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                                  \
      CUST_KERNEL_LOG_ERROR(ctx, "RightShift kernel compute failed."); \
      return result;                                                   \
    }                                                                  \
    break;                                                             \
  }
}  // namespace

namespace aicpu {
uint32_t RightShiftCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "RightShift check input and output number failed.");
  CUST_KERNEL_HANDLE_ERROR(ctx, RightShiftParamCheck(ctx), "RightShift check params or bcast failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    RIGHTSHIFT_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    RIGHTSHIFT_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    RIGHTSHIFT_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    RIGHTSHIFT_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    RIGHTSHIFT_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    RIGHTSHIFT_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    RIGHTSHIFT_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    RIGHTSHIFT_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "RightShift kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }

  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

uint32_t RightShiftCpuKernel::RightShiftParamCheck(CpuKernelContext &ctx) {
  Tensor *input_0 = ctx.Input(0);
  Tensor *input_1 = ctx.Input(1);
  Tensor *output = ctx.Output(0);
  CUST_KERNEL_CHECK_NULLPTR(ctx, input_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 0 data failed.");
  CUST_KERNEL_CHECK_NULLPTR(ctx, input_1->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 1 data failed.");
  CUST_KERNEL_CHECK_NULLPTR(ctx, output->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed");
  DataType input0_type = input_0->GetDataType();
  DataType input1_type = input_1->GetDataType();
  CUST_KERNEL_CHECK_FALSE(ctx, (input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID,
                          "The data type of input1 [%d] need be same with input0 [%d].",
                          static_cast<int32_t>(input0_type), static_cast<int32_t>(input1_type))
  CUST_KERNEL_LOG_INFO(ctx,
                       "RightShiftCpuKernel[%s], input0: size[%llu];"
                       "input1: size[%llu], output: size[%llu].",
                       ctx.GetOpType().c_str(), input_0->GetDataSize(), input_1->GetDataSize(), output->GetDataSize());

  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

// special compute is used in the following situations.
// 1. the shapes of input1 and input2 are the same
// 2. input1 is a 1D tensor with only one element or input1 is scalar
// 3. input2 is a 1D tensor with only one element or input2 is scalar
// 4. the shapes of input1 and input2 are different
template <typename T>
void RightShiftCpuKernel::SpecialCompute(CpuKernelContext &ctx, BcastShapeType type, int64_t start, int64_t end,
                                         const T *input1, const T *input2, T *output) {
  switch (type) {
    case BcastShapeType::SAME_SHAPE:
      for (int64_t i = start; i < end; ++i) {
        *(output + i) = *(input1 + i) >> *(input2 + i);
      }
      break;
    case BcastShapeType::X_ONE_ELEMENT:
      for (int64_t i = start; i < end; ++i) {
        *(output + i) = *input1 >> *(input2 + i);
      }
      break;
    case BcastShapeType::Y_ONE_ELEMENT:
      for (int64_t i = start; i < end; ++i) {
        *(output + i) = *(input1 + i) >> *input2;
      }
      break;
    default:
      CUST_KERNEL_LOG_WARN(ctx, "Invalid type [%d]", static_cast<int32_t>(type));
      break;
  }
}

template <typename T>
uint32_t RightShiftCpuKernel::NoBcastCompute(CpuKernelContext &ctx) {
  auto in0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto in1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto out = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t in0_elements_nums = ctx.Input(0)->NumElements();
  int64_t in1_elements_nums = ctx.Input(1)->NumElements();
  int64_t data_num = ctx.Output(0)->NumElements();
  BcastShapeType type = in0_elements_nums == in1_elements_nums
                          ? BcastShapeType::SAME_SHAPE
                          : (in0_elements_nums == 1 ? BcastShapeType::X_ONE_ELEMENT : BcastShapeType::Y_ONE_ELEMENT);

  T *in1_clamped = new T[in1_elements_nums];
  for (int64_t i = 0; i < in1_elements_nums; i++) {
    in1_clamped[i] = in1[i];
    T char_bit = static_cast<T>(sizeof(T) * CHAR_BIT > 32 ? sizeof(T) * CHAR_BIT : 32);
    if (in1_clamped[i] > char_bit || in1_clamped[i] < -char_bit) {
      in1_clamped[i] = in1_clamped[i] % char_bit;
    }
  }

  if (data_num >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

    if (data_num <= kParallelDataNumSameShapeMid) {
      max_core_num = std::min(max_core_num, static_cast<int64_t>(4));  // up to 4 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto sharder_less = [this, &ctx, &type, &in0, &in1_clamped, &out](int64_t start, int64_t end) {
      SpecialCompute<T>(ctx, type, start, end, in0, in1_clamped, out);
    };
    if (max_core_num == 0) {
      CUST_KERNEL_LOG_ERROR(ctx, "max_core_num could not be 0.");
    }
    uint32_t flag = CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_less);
    delete[] in1_clamped;
    CUST_KERNEL_HANDLE_ERROR(ctx, flag, "RightShift Compute failed.");
  } else {
    SpecialCompute<T>(ctx, type, 0, data_num, in0, in1_clamped, out);
    delete[] in1_clamped;
  }
  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

template <typename T>
uint32_t RightShiftCpuKernel::BcastCompute(CpuKernelContext &ctx, const Bcast &bcast) {
  auto in0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto in1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto out = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Output(0)->NumElements();

  int64_t in1_elements_nums = ctx.Input(1)->NumElements();
  T *in1_clamped = new T[in1_elements_nums];
  for (int64_t i = 0; i < in1_elements_nums; i++) {
    in1_clamped[i] = in1[i];
    T char_bit = static_cast<T>(sizeof(T) * CHAR_BIT > 32 ? sizeof(T) * CHAR_BIT : 32);
    if (in1_clamped[i] > char_bit || in1_clamped[i] < -char_bit) {
      in1_clamped[i] = in1_clamped[i] % char_bit;
    }
  }

  if (data_num >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

    if (data_num <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, static_cast<int64_t>(4));  // up to 4 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto sharder_less = [&in0, &in1_clamped, &bcast, &out](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; ++i) {
        *(out + i) = *(in0 + bcast.GetBroadcastXIndex(i)) >> *(in1_clamped + bcast.GetBroadcastYIndex(i));
      }
    };
    if (max_core_num == 0) {
      CUST_KERNEL_LOG_ERROR(ctx, "max_core_num could not be 0.");
    }
    uint32_t flag = CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_less);
    delete[] in1_clamped;
    CUST_KERNEL_HANDLE_ERROR(ctx, flag, "RightShift Compute failed.");
  } else {
    for (int64_t i = 0; i < data_num; ++i) {
      *(out + i) = *(in0 + bcast.GetBroadcastXIndex(i)) >> *(in1_clamped + bcast.GetBroadcastYIndex(i));
    }
    delete[] in1_clamped;
  }
  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

template <typename T>
uint32_t RightShiftCpuKernel::RightShiftCompute(CpuKernelContext &ctx) {
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

REGISTER_MS_CPU_KERNEL(kRightShift, RightShiftCpuKernel);
}  // namespace aicpu
