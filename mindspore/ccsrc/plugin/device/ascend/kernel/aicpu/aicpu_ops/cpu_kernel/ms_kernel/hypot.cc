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
#include "hypot.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kHypot = "Hypot";
const int64_t kParallelDataNum = 2 * 1024;
const int64_t kParallelDataNumMid = 16 * 1024;
const int64_t kParallelDataNumSameShape = 7 * 1024;
const int64_t kParallelDataNumSameShapeMid = 35 * 1024;

#define HYPOT_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                       \
    uint32_t result = HypotCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                   \
      KERNEL_LOG_ERROR("Hypot kernel compute failed."); \
      return result;                                    \
    }                                                   \
    break;                                              \
  }
}  // namespace

namespace aicpu {
template <typename T>
T hypot(T a, T b) {
  return std::hypot(a, b);
}

uint32_t HypotCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Hypot check input and output number failed.");
  KERNEL_HANDLE_ERROR(HypotParamCheck(ctx), "Hypot check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    HYPOT_COMPUTE_CASE(DT_FLOAT, float_t, ctx)
    HYPOT_COMPUTE_CASE(DT_DOUBLE, double_t, ctx)
    default:
      KERNEL_LOG_ERROR("Hypot kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t HypotCpuKernel::HypotParamCheck(CpuKernelContext &ctx) {
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
    "HypotCpuKernel[%s], input0: size[%llu];"
    "input1: size[%llu], output: size[%llu].",
    ctx.GetOpType().c_str(), input_0->GetDataSize(), input_1->GetDataSize(), output->GetDataSize());

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t HypotCpuKernel::NoBcastCompute(CpuKernelContext &ctx) {
  auto in0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto in1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto out = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t in0_elements_nums = ctx.Input(0)->NumElements();
  int64_t in1_elements_nums = ctx.Input(1)->NumElements();
  int64_t data_num = ctx.Output(0)->NumElements();
  BcastShapeType type;

  if (in0_elements_nums == in1_elements_nums) {
    type = BcastShapeType::SAME_SHAPE;
  } else {
    type = (in0_elements_nums == 1 ? BcastShapeType::X_ONE_ELEMENT : BcastShapeType::Y_ONE_ELEMENT);
  }

  if (data_num >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

    if (data_num <= kParallelDataNumSameShapeMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto sharder_hypot = [&](int64_t start, int64_t end) {
      switch (type) {
        case BcastShapeType::SAME_SHAPE:
          for (int64_t i = start; i < end; ++i) {
            *(out + i) = hypot(*(in0 + i), *(in1 + i));
          }
          break;
        case BcastShapeType::X_ONE_ELEMENT:
          for (int64_t i = start; i < end; ++i) {
            *(out + i) = hypot(*in0, *(in1 + i));
          }
          break;
        case BcastShapeType::Y_ONE_ELEMENT:
          for (int64_t i = start; i < end; ++i) {
            *(out + i) = hypot(*(in0 + i), *in1);
          }
          break;
        default:
          KERNEL_LOG_ERROR("Invalid type [%d]", static_cast<int32_t>(type));
          break;
      }
    };
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_hypot),
                        "Hypot Compute failed.");
  } else {
    switch (type) {
      case BcastShapeType::SAME_SHAPE:
        for (int64_t i = static_cast<int64_t>(0); i < data_num; ++i) {
          *(out + i) = hypot(*(in0 + i), *(in1 + i));
        }
        break;
      case BcastShapeType::X_ONE_ELEMENT:
        for (int64_t i = static_cast<int64_t>(0); i < data_num; ++i) {
          *(out + i) = hypot(*in0, *(in1 + i));
        }
        break;
      case BcastShapeType::Y_ONE_ELEMENT:
        for (int64_t i = static_cast<int64_t>(0); i < data_num; ++i) {
          *(out + i) = hypot(*(in0 + i), *in1);
        }
        break;
      default:
        KERNEL_LOG_ERROR("Invalid type [%d]", static_cast<int32_t>(type));
        break;
    }
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t HypotCpuKernel::BcastCompute(CpuKernelContext &ctx, Bcast &bcast) {
  T *in0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  T *in1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  T *out = reinterpret_cast<T *>(ctx.Output(0)->GetData());
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

    auto sharder_hypot = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; ++i) {
        *(out + i) = hypot<T>(*(in0 + bcast.GetBroadcastXIndex(i)), *(in1 + bcast.GetBroadcastYIndex(i)));
      }
    };

    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_hypot),
                        "Hypot Compute failed.");
  } else {
    for (int64_t i = 0; i < data_num; ++i) {
      *(out + i) = hypot<T>(*(in0 + bcast.GetBroadcastXIndex(i)), *(in1 + bcast.GetBroadcastYIndex(i)));
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t HypotCpuKernel::HypotCompute(CpuKernelContext &ctx) {
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

REGISTER_CPU_KERNEL(kHypot, HypotCpuKernel);
}  // namespace aicpu
