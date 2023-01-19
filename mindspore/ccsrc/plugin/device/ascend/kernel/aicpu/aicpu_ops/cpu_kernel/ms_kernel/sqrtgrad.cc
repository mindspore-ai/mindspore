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
#include "sqrtgrad.h"

#include <complex>
#include <cstdint>
#include <typeinfo>
#include "Eigen/Dense"

#include <iostream>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "utils/kernel_util.h"
#include "kernel_log.h"
#include "securec.h"
#include "status.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kSqrtGrad = "SqrtGrad";
const int64_t kParallelDataNum = 2 * 1024;
const int64_t kParallelDataNumMid = 16 * 1024;
const int64_t kParallelDataNumSameShape = 7 * 1024;
const int64_t kParallelDataNumSameShapeMid = 35 * 1024;

#define SQRTGRAD_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                          \
    uint32_t result = SqrtGradCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                      \
      KERNEL_LOG_ERROR("SqrtGrad kernel compute failed."); \
      return result;                                       \
    }                                                      \
    break;                                                 \
  }

#define SQRTGRAD_COMPUTE_COMPLEX_CASE(DTYPE, TYPE, CTX)    \
  case (DTYPE): {                                          \
    uint32_t result = SqrtGradComputeComplex<TYPE>(CTX);   \
    if (result != KERNEL_STATUS_OK) {                      \
      KERNEL_LOG_ERROR("SqrtGrad kernel compute failed."); \
      return result;                                       \
    }                                                      \
    break;                                                 \
  }
}  // namespace

namespace aicpu {
uint32_t SqrtGradCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kSqrtGrad);
  KERNEL_HANDLE_ERROR(SqrtGradParamCheck(ctx), "[%s] check params failed.", kSqrtGrad);
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    SQRTGRAD_COMPUTE_COMPLEX_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    SQRTGRAD_COMPUTE_COMPLEX_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    SQRTGRAD_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    SQRTGRAD_COMPUTE_CASE(DT_FLOAT, float, ctx)
    SQRTGRAD_COMPUTE_CASE(DT_DOUBLE, double, ctx)

    default:
      KERNEL_LOG_ERROR("SqrtGrad kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t SqrtGradCpuKernel::SqrtGradParamCheck(CpuKernelContext &ctx) {
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
    "SqrtGradCpuKernel[%s], input0: size[%llu];"
    "input1: size[%llu], output: size[%llu].",
    ctx.GetOpType().c_str(), input_0->GetDataSize(), input_1->GetDataSize(), output->GetDataSize());

  return KERNEL_STATUS_OK;
}

/**
special compute is used in the following situations.
1. the shapes of input1 and input2 are the same
2. input1 is a 1D tensor with only one element or input1 is scalar
3. input2 is a 1D tensor with only one element or input2 is scalar
4. the shapes of input1 and input2 are different
*/
template <typename T>
void SqrtGradCpuKernel::SpecialCompute(int64_t start, int64_t end, T *input1, T *input2, T *output) {
  int flag = 0;
  for (int64_t i = start; i < end; ++i) {
    if (*(input2 + i) == static_cast<T>(0)) {
      flag = 1;
      break;
    }
  }
  for (int64_t i = start; i < end; ++i) {
    *(output + i) = *(input2 + i) * static_cast<T>(0.5) / *(input1 + i);
  }

  if (flag == 1) KERNEL_LOG_WARN("divide by zero encountered");
}

template <typename T>
void SqrtGradCpuKernel::SpecialComputeComplex(int64_t start, int64_t end, T *input1, T *input2, T *output) {
  int flag = 0;
  for (int64_t i = start; i < end; ++i) {
    if (*(input2 + i) == static_cast<T>(0)) {
      flag = 1;
      break;
    }
  }

  for (int64_t i = start; i < end; ++i) {
    T in1 = *(input1 + i);
    T in1_conj = std::conj(in1);
    if (in1_conj == static_cast<T>(0)) {
      *(output + i) = INFINITY;
    } else {
      *(output + i) = *(input2 + i) * static_cast<T>(0.5) / in1_conj;
    }
  }
  if (flag == 1) KERNEL_LOG_WARN("divide by zero encountered");
}

template <typename T>
uint32_t SqrtGradCpuKernel::NoBcastCompute(CpuKernelContext &ctx) {
  auto in0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto in1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto out = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t in0_elements_nums = ctx.Input(0)->NumElements();
  int64_t data_num = in0_elements_nums;

  if (data_num >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

    if (data_num <= kParallelDataNumSameShapeMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto sharder_sqrtgrad = [&](size_t start, size_t end) { SpecialCompute<T>(0, data_num, in0, in1, out); };

    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_sqrtgrad),
                        "SqrtGrad Compute failed.");
  } else {
    SpecialCompute<T>(0, data_num, in0, in1, out);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SqrtGradCpuKernel::NoBcastComputeComplex(CpuKernelContext &ctx) {
  auto in0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto in1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto out = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t in0_elements_nums = ctx.Input(0)->NumElements();
  int64_t data_num = in0_elements_nums;

  if (data_num >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

    if (data_num <= kParallelDataNumSameShapeMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto sharder_sqrtgrad = [&](size_t start, size_t end) { SpecialComputeComplex<T>(0, data_num, in0, in1, out); };

    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_sqrtgrad),
                        "SqrtGrad Compute failed.");
  } else {
    SpecialComputeComplex<T>(0, data_num, in0, in1, out);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SqrtGradCpuKernel::SqrtGradCompute(CpuKernelContext &ctx) {
  Tensor *input0_tensor = ctx.Input(0);
  auto input0_shape = input0_tensor->GetTensorShape()->GetDimSizes();
  int64_t input0_elements_nums = input0_tensor->NumElements();

  Tensor *input1_tensor = ctx.Input(1);
  auto input1_shape = input1_tensor->GetTensorShape()->GetDimSizes();
  int64_t input1_elements_nums = input1_tensor->NumElements();

  if (input0_elements_nums != input1_elements_nums) {
    KERNEL_LOG_WARN("Invalid element numbers, got[%d] and [%d]", static_cast<int32_t>(input0_elements_nums),
                    static_cast<int32_t>(input1_elements_nums));
    return KERNEL_STATUS_PARAM_INVALID;
  } else {
    return NoBcastCompute<T>(ctx);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SqrtGradCpuKernel::SqrtGradComputeComplex(CpuKernelContext &ctx) {
  Tensor *input0_tensor = ctx.Input(0);
  auto input0_shape = input0_tensor->GetTensorShape()->GetDimSizes();
  int64_t input0_elements_nums = input0_tensor->NumElements();

  Tensor *input1_tensor = ctx.Input(1);
  auto input1_shape = input1_tensor->GetTensorShape()->GetDimSizes();
  int64_t input1_elements_nums = input1_tensor->NumElements();

  if (input0_elements_nums != input1_elements_nums) {
    KERNEL_LOG_WARN("Invalid element numbers, got[%d] and [%d]", static_cast<int32_t>(input0_elements_nums),
                    static_cast<int32_t>(input1_elements_nums));
    return KERNEL_STATUS_PARAM_INVALID;
  } else {
    return NoBcastComputeComplex<T>(ctx);
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kSqrtGrad, SqrtGradCpuKernel);
}  // namespace aicpu
