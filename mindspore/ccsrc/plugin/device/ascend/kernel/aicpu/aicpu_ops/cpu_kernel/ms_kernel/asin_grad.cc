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

#include "cpu_kernel/ms_kernel/asin_grad.h"
#include <limits>
#include <algorithm>
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kAsinGrad = "AsinGrad";
// when input data size is more than kParallelDataNum, use Parallel func
const int64_t kParallelDataNum = 2 * 1024;
const int64_t kParallelDataNumMid = 16 * 1024;
uint32_t max_core_num = 1;
int64_t data_num = 1;
}  // namespace
namespace aicpu {
uint32_t AsinGradCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "AsinGrad check input and output number failed.");
  KERNEL_HANDLE_ERROR(AsinGradParamCheck(ctx), "AsinGrad check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  data_num = ctx.Output(0)->NumElements();
  if (data_num >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);

    if (data_num <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
  }
  switch (data_type) {
    case DT_FLOAT16:
      return AsinGradComputeFP16<Eigen::half>(ctx);
    case DT_FLOAT:
      return AsinGradComputeRealType<float>(ctx);
    case DT_DOUBLE:
      return AsinGradComputeRealType<double>(ctx);
    default:
      KERNEL_LOG_ERROR("AsinGrad kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t AsinGradCpuKernel::AsinGradParamCheck(const CpuKernelContext &ctx) {
  // the non null of input_0, input_1, output has been verified in NormalCheck
  Tensor *input_y = ctx.Input(0);
  Tensor *input_dy = ctx.Input(1);
  Tensor *output = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(input_y->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 'y' failed.")
  KERNEL_CHECK_NULLPTR(input_dy->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 'dy' data failed.")
  KERNEL_CHECK_NULLPTR(output->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed.")
  DataType input_y_type = input_y->GetDataType();
  DataType input_dy_type = input_dy->GetDataType();
  KERNEL_CHECK_FALSE((input_y_type == input_dy_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of y [%s] need be same with "
                     "dy [%s].",
                     DTypeStr(input_y_type).c_str(), DTypeStr(input_dy_type).c_str())
  KERNEL_LOG_DEBUG(
    "AsinGradCpuKernel[%s], y: size[%llu];"
    "dy: size[%llu], z: size[%llu].",
    ctx.GetOpType().c_str(), input_y->GetDataSize(), input_dy->GetDataSize(), output->GetDataSize());
  return KERNEL_STATUS_OK;
}

template <typename T>
void AsinGradCpuKernel::SpecialCompute(int64_t start, int64_t end, const T *input1, const T *input2, T *output) {
  for (int64_t i = start; i < end; i++) {
    T dividend = input2[i];
    T divisor = static_cast<T>(sqrt(static_cast<T>(1) - input1[i] * input1[i]));
    if (divisor == static_cast<T>(0)) {
      if (dividend == static_cast<T>(0)) {
        output[i] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
      if (std::numeric_limits<T>::has_infinity) {
        output[i] = dividend > 0 ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      } else {
        output[i] = dividend > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
      }
      continue;
    }
    output[i] = static_cast<T>(dividend / divisor);
  }
}

template <typename T>
uint32_t AsinGradCpuKernel::AsinGradComputeRealType(const CpuKernelContext &ctx) {
  auto in0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto in1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto out = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  if (data_num >= kParallelDataNum) {
    auto sharder_asin_grad = [&](int64_t start, int64_t end) { SpecialCompute<T>(start, end, in0, in1, out); };
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max core num could not be 0");
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_asin_grad),
                        "AsinGrad Compute failed.");
  } else {
    SpecialCompute<T>(0, data_num, in0, in1, out);
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
void AsinGradCpuKernel::SpecialComputeFP16(int64_t start, int64_t end, const T *input1, const T *input2, T *output) {
  for (int64_t i = start; i < end; i++) {
    float dividend = static_cast<float>(input2[i]);
    float divisor =
      static_cast<float>(sqrt(static_cast<float>(1) - static_cast<float>(input1[i]) * static_cast<float>(input1[i])));
    if (divisor == 0) {
      if (dividend == 0) {
        output[i] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
      if (std::numeric_limits<T>::has_infinity) {
        output[i] = dividend > 0 ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      } else {
        output[i] = dividend > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
      }
      continue;
    }
    output[i] = static_cast<T>(dividend / divisor);
  }
}

template <typename T>
uint32_t AsinGradCpuKernel::AsinGradComputeFP16(const CpuKernelContext &ctx) {
  auto in0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto in1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto out = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  if (data_num >= kParallelDataNum) {
    auto sharder_asin_grad = [&](int64_t start, int64_t end) { SpecialComputeFP16<T>(start, end, in0, in1, out); };
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max core num could not be 0");
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_asin_grad),
                        "AsinGrad Compute failed.");
  } else {
    SpecialComputeFP16<T>(0, data_num, in0, in1, out);
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kAsinGrad, AsinGradCpuKernel);
}  // namespace aicpu
