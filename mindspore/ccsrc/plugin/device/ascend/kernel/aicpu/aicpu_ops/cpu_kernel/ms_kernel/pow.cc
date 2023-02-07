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
#include "pow.h"
#include <math.h>
#include <stdint.h>
#include "Eigen/Dense"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "utils/kernel_util.h"
#include "securec.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kPow = "Pow";
const int64_t kParallelDataNum = 2 * 1024;
const int64_t kParallelDataNumMid = 16 * 1024;
const int64_t kParallelDataNumSameShape = 7 * 1024;
const int64_t kParallelDataNumSameShapeMid = 35 * 1024;

#define POW_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                     \
    uint32_t result = PowCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                 \
      KERNEL_LOG_ERROR("Pow kernel compute failed."); \
      return result;                                  \
    }                                                 \
    break;                                            \
  }
}  // namespace

namespace aicpu {
uint32_t PowCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Pow check input and output number failed.");
  KERNEL_HANDLE_ERROR(PowParamCheck(ctx), "Pow check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    POW_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    POW_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    POW_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    POW_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    POW_COMPUTE_CASE(DT_FLOAT, float, ctx)
    POW_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    POW_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    POW_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      KERNEL_LOG_ERROR("Pow kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t PowCpuKernel::PowParamCheck(CpuKernelContext &ctx) {
  // the non null of input_0, input_1, output has been verified in NormalCheck
  Tensor *input_0 = ctx.Input(0);
  Tensor *input_1 = ctx.Input(1);
  Tensor *output = ctx.Output(0);
  DataType input0_type = input_0->GetDataType();
  DataType input1_type = input_1->GetDataType();
  auto input0_Shape = input_0->GetTensorShape();
  auto input1_Shape = input_1->GetTensorShape();
  KERNEL_CHECK_NULLPTR(input0_Shape, KERNEL_STATUS_PARAM_INVALID, "Get input0_Shape failed.")
  KERNEL_CHECK_NULLPTR(input1_Shape, KERNEL_STATUS_PARAM_INVALID, "Get input1_Shape failed.")
  KERNEL_CHECK_FALSE((input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input0 [%s] need be same with "
                     "input1 [%s].",
                     DTypeStr(input0_type).c_str(), DTypeStr(input1_type).c_str())
  KERNEL_LOG_DEBUG(
    "PowCpuKernel[%s], input0: size[%llu];"
    "input1: size[%llu], output: size[%llu].",
    ctx.GetOpType().c_str(), input_0->GetDataSize(), input_1->GetDataSize(), output->GetDataSize());

  return KERNEL_STATUS_OK;
}

template <typename T>
void PowCpuKernel::SpecialCompute(BcastShapeType type, int64_t start, int64_t end, T *input1, T *input2, T *output) {
  switch (type) {
    case BcastShapeType::SAME_SHAPE:
      for (int64_t i = start; i < end; ++i) {
        *(output + i) = pow(*(input1 + i), *(input2 + i));
      }
      break;
    case BcastShapeType::X_ONE_ELEMENT:
      for (int64_t i = start; i < end; ++i) {
        *(output + i) = pow(*(input1), *(input2 + i));
      }
      break;
    default:
      for (int64_t i = start; i < end; ++i) {
        *(output + i) = pow(*(input1 + i), *(input2));
      }
      break;
  }
}

template <typename T>
uint32_t PowCpuKernel::NoBcastCompute(CpuKernelContext &ctx) {
  auto in0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto in1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto out = reinterpret_cast<T *>(ctx.Output(0)->GetData());
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

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto sharder_pow = [&](size_t start, size_t end) { SpecialCompute<T>(type, start, end, in0, in1, out); };

    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_pow),
                        "Pow Compute failed.");
  } else {
    SpecialCompute<T>(type, 0, data_num, in0, in1, out);
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t PowCpuKernel::BcastCompute(CpuKernelContext &ctx, Bcast &bcast) {
  auto in0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto in1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto out = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Output(0)->NumElements();

  if (data_num >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);

    if (data_num <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto sharder_pow = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        auto input1 = in0 + bcast.GetBroadcastXIndex(i);  // i-th value of input0
        auto input2 = in1 + bcast.GetBroadcastYIndex(i);  // i-th value of input1
        *(out + i) = pow((*input1), (*input2));
      }
    };

    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_pow),
                        "Pow Compute failed.");
  } else {
    for (int64_t i = 0; i < data_num; i++) {
      auto input1 = in0 + bcast.GetBroadcastXIndex(i);  // i-th value of input0
      auto input2 = in1 + bcast.GetBroadcastYIndex(i);  // i-th value of input1
      *(out + i) = pow((*input1), (*input2));
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t PowCpuKernel::PowCompute(CpuKernelContext &ctx) {
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

    return BcastCompute<T>(ctx, bcast);
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kPow, PowCpuKernel);
}  // namespace aicpu
