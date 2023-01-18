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
#include "xlogy.h"

#include "cmath"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kXlogy = "Xlogy";

const int64_t kParallelDataNum = 8 * 1024;
const int64_t kParallelDataNumMid = 16 * 1024;
const int64_t kParallelDataNumSameShape = 7 * 1024;
const int64_t kParallelDataNumSameShapeMid = 35 * 1024;

#define XLOGY_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                       \
    uint32_t result = XlogyCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                   \
      KERNEL_LOG_ERROR("Xlogy kernel compute failed."); \
      return result;                                    \
    }                                                   \
    break;                                              \
  }
}  // namespace

namespace aicpu {
uint32_t XlogyCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kXlogy);
  BCalcInfo calc_info;
  KERNEL_HANDLE_ERROR(XlogyParamCheck(ctx), "Xlogy check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    XLOGY_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    XLOGY_COMPUTE_CASE(DT_FLOAT, float, ctx)
    XLOGY_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("Xlogy kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t XlogyCpuKernel::XlogyParamCheck(CpuKernelContext &ctx) {
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
    "XlogyCpuKernel[%s], input0: size[%llu];"
    "input1: size[%llu], output: size[%llu].",
    ctx.GetOpType().c_str(), input_0->GetDataSize(), input_1->GetDataSize(), output->GetDataSize());

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t XlogyCpuKernel::SpecialCompute(BcastShapeType type, int64_t start, int64_t end, const T *input1,
                                        const T *input2, T *output) {
  auto zero = T(0);
  switch (type) {
    case BcastShapeType::SAME_SHAPE:
      for (int64_t i = start; i < end; ++i) {
        if (*(input1 + i) == zero) {
          *(output + i) = zero;
          continue;
        }
        if (*(input2 + i) < zero) {
          *(output + i) = std::numeric_limits<T>::quiet_NaN();
          continue;
        }
        *(output + i) = *(input1 + i) * log(*(input2 + i));
      }
      break;
    case BcastShapeType::X_ONE_ELEMENT:
      for (int64_t i = start; i < end; ++i) {
        if (*(input1) == zero) {
          *(output + i) = zero;
          continue;
        }
        if (*(input2 + i) < zero) {
          *(output + i) = std::numeric_limits<T>::quiet_NaN();
          continue;
        }
        *(output + i) = (*input1) * log(*(input2 + i));
      }
      break;
    case BcastShapeType::Y_ONE_ELEMENT:
      for (int64_t i = start; i < end; ++i) {
        if (*(input1 + i) == zero) {
          *(output + i) = zero;
          continue;
        }
        if (*(input2) < zero) {
          *(output + i) = std::numeric_limits<T>::quiet_NaN();
          continue;
        }
        *(output + i) = *(input1 + i) * log(*(input2));
      }
      break;
    default:
      KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(type));
      break;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t XlogyCpuKernel::NoBcastCompute(CpuKernelContext &ctx) {
  auto in0 = static_cast<T *>(ctx.Input(0)->GetData());
  auto in1 = static_cast<T *>(ctx.Input(1)->GetData());
  auto out = static_cast<T *>(ctx.Output(0)->GetData());
  int64_t in0_elements_nums = ctx.Input(0)->NumElements();
  int64_t in1_elements_nums = ctx.Input(1)->NumElements();
  int64_t data_num = ctx.Output(0)->NumElements();
  BcastShapeType type = in0_elements_nums == in1_elements_nums
                          ? BcastShapeType::SAME_SHAPE
                          : (in0_elements_nums == 1 ? BcastShapeType::X_ONE_ELEMENT : BcastShapeType::Y_ONE_ELEMENT);
  if (data_num >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (data_num <= kParallelDataNumSameShapeMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }
    auto sharder_div = [&](int64_t start, int64_t end) { SpecialCompute<T>(type, start, end, in0, in1, out); };
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_div),
                        "Xlogy Compute failed.");
  } else {
    SpecialCompute<T>(type, 0, data_num, in0, in1, out);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t XlogyCpuKernel::BcastCompute(CpuKernelContext &ctx, Bcast &bcast) {
  auto in0 = static_cast<T *>(ctx.Input(0)->GetData());
  auto in1 = static_cast<T *>(ctx.Input(1)->GetData());
  auto out = static_cast<T *>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Output(0)->NumElements();
  auto zero = T(0);
  if (data_num >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (data_num <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }
    auto sharder_div = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; ++i) {
        *(out + i) = *(in1 + i) >= zero
                       ? *(in0 + bcast.GetBroadcastXIndex(i)) * log(*(in1 + bcast.GetBroadcastYIndex(i)))
                       : std::numeric_limits<T>::quiet_NaN();
      }
    };
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_div),
                        "Xlogy Compute failed.");
  } else {
    for (int64_t i = 0; i < data_num; ++i) {
      *(out + i) = *(in1 + i) >= zero ? *(in0 + bcast.GetBroadcastXIndex(i)) * log(*(in1 + bcast.GetBroadcastYIndex(i)))
                                      : std::numeric_limits<T>::quiet_NaN();
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t XlogyCpuKernel::XlogyCompute(CpuKernelContext &ctx) {
  Tensor *input0_tensor = ctx.Input(0);
  auto input0_shape = input0_tensor->GetTensorShape()->GetDimSizes();
  int64_t input0_elements_nums = input0_tensor->NumElements();
  Tensor *input1_tensor = ctx.Input(1);
  auto input1_shape = input1_tensor->GetTensorShape()->GetDimSizes();
  int64_t input1_elements_nums = input1_tensor->NumElements();
  bool noNeedBcast = (input0_shape == input1_shape) || (input0_elements_nums == 1) || (input1_elements_nums == 1);
  if (noNeedBcast) {
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
REGISTER_CPU_KERNEL(kXlogy, XlogyCpuKernel);
}  // namespace aicpu
