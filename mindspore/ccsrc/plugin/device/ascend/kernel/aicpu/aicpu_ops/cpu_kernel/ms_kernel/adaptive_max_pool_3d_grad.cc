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
#include "cpu_kernel/ms_kernel/adaptive_max_pool_3d_grad.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cpu_kernel/inc/cpu_context.h"

namespace {
const char *kAdaptiveMaxPool3dGrad = "AdaptiveMaxPool3dGrad";
constexpr uint32_t kInputNum = 3;
constexpr uint32_t kOutputNum = 1;
constexpr int32_t kDtypeNum = 20430;
constexpr uint64_t kParallelDataSize = 64 * 1024;

#define CREATE_COMPUTE_CASE(DTYPE1, TYPE1, DTYPE2, TYPE2, CTX)          \
  case (DTYPE1 * kDtypeNum + DTYPE2): {                                 \
    uint32_t result = AdaptiveMaxPool3dGradCompute<TYPE1, TYPE2>(CTX);  \
    if (result != KERNEL_STATUS_OK) {                                   \
      KERNEL_LOG_ERROR("AdaptiveMaxPool3dGrad kernel compute failed."); \
      return result;                                                    \
    }                                                                   \
    break;                                                              \
  }
}  // namespace

namespace aicpu {
uint32_t AdaptiveMaxPool3dGradCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(AdaptiveMaxPool3dGradCheck(ctx), "AdaptiveMaxPool3dGrad check params failed.");
  auto input_grad_dtype = ctx.Input(0)->GetDataType();
  auto output_dtype = ctx.Output(0)->GetDataType();
  switch (input_grad_dtype * kDtypeNum + output_dtype) {
    CREATE_COMPUTE_CASE(DT_INT8, int8_t, DT_INT8, int8_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT8, int8_t, DT_INT16, int16_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT8, int8_t, DT_INT32, int32_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT8, int8_t, DT_INT64, int64_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT8, int8_t, DT_UINT8, uint8_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT8, int8_t, DT_UINT16, uint16_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT8, int8_t, DT_UINT32, uint32_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT8, int8_t, DT_UINT64, uint64_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT8, int8_t, DT_FLOAT16, Eigen::half, ctx)
    CREATE_COMPUTE_CASE(DT_INT8, int8_t, DT_FLOAT, float, ctx)
    CREATE_COMPUTE_CASE(DT_INT8, int8_t, DT_DOUBLE, double, ctx)
    CREATE_COMPUTE_CASE(DT_INT16, int16_t, DT_INT8, int8_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT16, int16_t, DT_INT16, int16_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT16, int16_t, DT_INT32, int32_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT16, int16_t, DT_INT64, int64_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT16, int16_t, DT_UINT8, uint8_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT16, int16_t, DT_UINT16, uint16_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT16, int16_t, DT_UINT32, uint32_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT16, int16_t, DT_UINT64, uint64_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT16, int16_t, DT_FLOAT16, Eigen::half, ctx)
    CREATE_COMPUTE_CASE(DT_INT16, int16_t, DT_FLOAT, float, ctx)
    CREATE_COMPUTE_CASE(DT_INT16, int16_t, DT_DOUBLE, double, ctx)
    CREATE_COMPUTE_CASE(DT_INT32, int32_t, DT_INT8, int8_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT32, int32_t, DT_INT16, int16_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT32, int32_t, DT_INT32, int32_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT32, int32_t, DT_INT64, int64_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT32, int32_t, DT_UINT8, uint8_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT32, int32_t, DT_UINT16, uint16_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT32, int32_t, DT_UINT32, uint32_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT32, int32_t, DT_UINT64, uint64_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT32, int32_t, DT_FLOAT16, Eigen::half, ctx)
    CREATE_COMPUTE_CASE(DT_INT32, int32_t, DT_FLOAT, float, ctx)
    CREATE_COMPUTE_CASE(DT_INT32, int32_t, DT_DOUBLE, double, ctx)
    CREATE_COMPUTE_CASE(DT_INT64, int64_t, DT_INT8, int8_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT64, int64_t, DT_INT16, int16_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT64, int64_t, DT_INT32, int32_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT64, int64_t, DT_INT64, int64_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT64, int64_t, DT_UINT8, uint8_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT64, int64_t, DT_UINT16, uint16_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT64, int64_t, DT_UINT32, uint32_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT64, int64_t, DT_UINT64, uint64_t, ctx)
    CREATE_COMPUTE_CASE(DT_INT64, int64_t, DT_FLOAT16, Eigen::half, ctx)
    CREATE_COMPUTE_CASE(DT_INT64, int64_t, DT_FLOAT, float, ctx)
    CREATE_COMPUTE_CASE(DT_INT64, int64_t, DT_DOUBLE, double, ctx)
    CREATE_COMPUTE_CASE(DT_UINT8, uint8_t, DT_INT8, int8_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT8, uint8_t, DT_INT16, int16_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT8, uint8_t, DT_INT32, int32_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT8, uint8_t, DT_INT64, int64_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT8, uint8_t, DT_UINT8, uint8_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT8, uint8_t, DT_UINT16, uint16_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT8, uint8_t, DT_UINT32, uint32_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT8, uint8_t, DT_UINT64, uint64_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT8, uint8_t, DT_FLOAT16, Eigen::half, ctx)
    CREATE_COMPUTE_CASE(DT_UINT8, uint8_t, DT_FLOAT, float, ctx)
    CREATE_COMPUTE_CASE(DT_UINT8, uint8_t, DT_DOUBLE, double, ctx)
    CREATE_COMPUTE_CASE(DT_UINT16, uint16_t, DT_INT8, int8_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT16, uint16_t, DT_INT16, int16_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT16, uint16_t, DT_INT32, int32_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT16, uint16_t, DT_INT64, int64_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT16, uint16_t, DT_UINT8, uint8_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT16, uint16_t, DT_UINT16, uint16_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT16, uint16_t, DT_UINT32, uint32_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT16, uint16_t, DT_UINT64, uint64_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT16, uint16_t, DT_FLOAT16, Eigen::half, ctx)
    CREATE_COMPUTE_CASE(DT_UINT16, uint16_t, DT_FLOAT, float, ctx)
    CREATE_COMPUTE_CASE(DT_UINT16, uint16_t, DT_DOUBLE, double, ctx)
    CREATE_COMPUTE_CASE(DT_UINT32, uint32_t, DT_INT8, int8_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT32, uint32_t, DT_INT16, int16_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT32, uint32_t, DT_INT32, int32_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT32, uint32_t, DT_INT64, int64_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT32, uint32_t, DT_UINT8, uint8_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT32, uint32_t, DT_UINT16, uint16_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT32, uint32_t, DT_UINT32, uint32_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT32, uint32_t, DT_UINT64, uint64_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT32, uint32_t, DT_FLOAT16, Eigen::half, ctx)
    CREATE_COMPUTE_CASE(DT_UINT32, uint32_t, DT_FLOAT, float, ctx)
    CREATE_COMPUTE_CASE(DT_UINT32, uint32_t, DT_DOUBLE, double, ctx)
    CREATE_COMPUTE_CASE(DT_UINT64, uint64_t, DT_INT8, int8_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT64, uint64_t, DT_INT16, int16_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT64, uint64_t, DT_INT32, int32_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT64, uint64_t, DT_INT64, int64_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT64, uint64_t, DT_UINT8, uint8_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT64, uint64_t, DT_UINT16, uint16_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT64, uint64_t, DT_UINT32, uint32_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT64, uint64_t, DT_UINT64, uint64_t, ctx)
    CREATE_COMPUTE_CASE(DT_UINT64, uint64_t, DT_FLOAT16, Eigen::half, ctx)
    CREATE_COMPUTE_CASE(DT_UINT64, uint64_t, DT_FLOAT, float, ctx)
    CREATE_COMPUTE_CASE(DT_UINT64, uint64_t, DT_DOUBLE, double, ctx)
    CREATE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, DT_INT8, int8_t, ctx)
    CREATE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, DT_INT16, int16_t, ctx)
    CREATE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, DT_INT32, int32_t, ctx)
    CREATE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, DT_INT64, int64_t, ctx)
    CREATE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, DT_UINT8, uint8_t, ctx)
    CREATE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, DT_UINT16, uint16_t, ctx)
    CREATE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, DT_UINT32, uint32_t, ctx)
    CREATE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, DT_UINT64, uint64_t, ctx)
    CREATE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, DT_FLOAT16, Eigen::half, ctx)
    CREATE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, DT_FLOAT, float, ctx)
    CREATE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, DT_DOUBLE, double, ctx)
    CREATE_COMPUTE_CASE(DT_FLOAT, float, DT_INT8, int8_t, ctx)
    CREATE_COMPUTE_CASE(DT_FLOAT, float, DT_INT16, int16_t, ctx)
    CREATE_COMPUTE_CASE(DT_FLOAT, float, DT_INT32, int32_t, ctx)
    CREATE_COMPUTE_CASE(DT_FLOAT, float, DT_INT64, int64_t, ctx)
    CREATE_COMPUTE_CASE(DT_FLOAT, float, DT_UINT8, uint8_t, ctx)
    CREATE_COMPUTE_CASE(DT_FLOAT, float, DT_UINT16, uint16_t, ctx)
    CREATE_COMPUTE_CASE(DT_FLOAT, float, DT_UINT32, uint32_t, ctx)
    CREATE_COMPUTE_CASE(DT_FLOAT, float, DT_UINT64, uint64_t, ctx)
    CREATE_COMPUTE_CASE(DT_FLOAT, float, DT_FLOAT16, Eigen::half, ctx)
    CREATE_COMPUTE_CASE(DT_FLOAT, float, DT_FLOAT, float, ctx)
    CREATE_COMPUTE_CASE(DT_FLOAT, float, DT_DOUBLE, double, ctx)
    CREATE_COMPUTE_CASE(DT_DOUBLE, double, DT_INT8, int8_t, ctx)
    CREATE_COMPUTE_CASE(DT_DOUBLE, double, DT_INT16, int16_t, ctx)
    CREATE_COMPUTE_CASE(DT_DOUBLE, double, DT_INT32, int32_t, ctx)
    CREATE_COMPUTE_CASE(DT_DOUBLE, double, DT_INT64, int64_t, ctx)
    CREATE_COMPUTE_CASE(DT_DOUBLE, double, DT_UINT8, uint8_t, ctx)
    CREATE_COMPUTE_CASE(DT_DOUBLE, double, DT_UINT16, uint16_t, ctx)
    CREATE_COMPUTE_CASE(DT_DOUBLE, double, DT_UINT32, uint32_t, ctx)
    CREATE_COMPUTE_CASE(DT_DOUBLE, double, DT_UINT64, uint64_t, ctx)
    CREATE_COMPUTE_CASE(DT_DOUBLE, double, DT_FLOAT16, Eigen::half, ctx)
    CREATE_COMPUTE_CASE(DT_DOUBLE, double, DT_FLOAT, float, ctx)
    CREATE_COMPUTE_CASE(DT_DOUBLE, double, DT_DOUBLE, double, ctx)

    default:
      const std::vector<int8_t> support_dtype{DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,  DT_INT32, DT_INT64,
                                              DT_INT8,   DT_UINT16, DT_UINT32,  DT_UINT64, DT_UINT8};
      if (std::find(support_dtype.begin(), support_dtype.end(), output_dtype) == support_dtype.end()) {
        KERNEL_LOG_ERROR("AdaptiveMaxPool3dGrad kernel input data type [%s] not support.",
                         DTypeStr(output_dtype).c_str());
      }
      if (std::find(support_dtype.begin(), support_dtype.end(), input_grad_dtype) == support_dtype.end()) {
        KERNEL_LOG_ERROR(
          "AdaptiveMaxPool3dGrad kernel input grad data type [%s] not "
          "support.",
          DTypeStr(input_grad_dtype).c_str());
      }
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t AdaptiveMaxPool3dGradCpuKernel::AdaptiveMaxPool3dGradCheck(const CpuKernelContext &ctx) {
  auto input_grad = ctx.Input(0);
  auto input_x = ctx.Input(1);
  auto input_argmax = ctx.Input(2);
  auto output = ctx.Output(0);
  KERNEL_HANDLE_ERROR(NormalCheck(const_cast<CpuKernelContext &>(ctx), kInputNum, kOutputNum),
                      "AdaptiveMaxPool3dGrad check params failed.");

  const int64_t dim_num = input_x->GetTensorShape()->GetDims();
  KERNEL_CHECK_FALSE(dim_num == 4 || dim_num == 5, KERNEL_STATUS_PARAM_INVALID,
                     "Input dimensions must be equal to 4 or 5")
  KERNEL_CHECK_FALSE(input_grad->GetTensorShape()->GetDims() == dim_num, KERNEL_STATUS_PARAM_INVALID,
                     "Input grad dimensions must be same as input dimensions")
  KERNEL_CHECK_FALSE(input_argmax->GetTensorShape()->GetDims() == dim_num, KERNEL_STATUS_PARAM_INVALID,
                     "Input indice dimensions must be same as input dimensions")
  KERNEL_CHECK_FALSE(output->GetTensorShape()->GetDims() == dim_num, KERNEL_STATUS_PARAM_INVALID,
                     "output dimensions must be same as input dimensions")
  KERNEL_CHECK_FALSE(input_argmax->GetTensorShape()->GetDimSizes() == input_grad->GetTensorShape()->GetDimSizes(),
                     KERNEL_STATUS_PARAM_INVALID, "Input grad shape must be same as input argmax")
  KERNEL_LOG_DEBUG(
    "AdaptiveMaxPool3dGradCpuKernel[%s], input_grad: size[%llu] dtype[%s]; "
    "input_x: size[%llu] dtype[%s], input_argmax: size[%llu] dtype[%s]; "
    "output: size[%llu] dtype[%s]. ",
    ctx.GetOpType().c_str(), input_grad->GetDataSize(), DTypeStr(input_grad->GetDataType()).c_str(),
    input_x->GetDataSize(), DTypeStr(input_x->GetDataType()).c_str(), input_argmax->GetDataSize(),
    DTypeStr(input_argmax->GetDataType()).c_str(), output->GetDataSize(), DTypeStr(output->GetDataType()).c_str());

  return KERNEL_STATUS_OK;
}

template <typename T1, typename T2>
uint32_t AdaptiveMaxPool3dGradCpuKernel::AdaptiveMaxPool3dGradCompute(const CpuKernelContext &ctx) {
  KERNEL_CHECK_NULLPTR(ctx.Input(kFirstInputIndex)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input 'grad' data failed.")
  KERNEL_CHECK_NULLPTR(ctx.Input(kThirdInputIndex)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input 'argmax' data failed.")
  KERNEL_CHECK_NULLPTR(ctx.Output(kFirstInputIndex)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed.")
  auto input_grad = reinterpret_cast<T1 *>(ctx.Input(0)->GetData());
  auto input_argmax = reinterpret_cast<int32_t *>(ctx.Input(2)->GetData());
  auto output = reinterpret_cast<T2 *>(ctx.Output(0)->GetData());
  const int64_t output_num_data = ctx.Output(0)->NumElements();
  const T2 data_zero = static_cast<T2>(0);
  for (int64_t i = 0; i < output_num_data; ++i) {
    output[i] = data_zero;
  }

  const std::vector<int64_t> output_shape = ctx.Output(0)->GetTensorShape()->GetDimSizes();
  const std::vector<int64_t> argmax_shape = ctx.Input(2)->GetTensorShape()->GetDimSizes();
  const int64_t output_stride = output_shape.cend()[-1] * output_shape.cend()[-2] * output_shape.cend()[-3];
  const int64_t argmax_stride = argmax_shape.cend()[-1] * argmax_shape.cend()[-2] * argmax_shape.cend()[-3];

  KERNEL_CHECK_FALSE(argmax_stride == 0, KERNEL_STATUS_PARAM_INVALID, "argmax shape can not contain 0.")
  auto shard_adaptive_max_pool_3d_grad = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      output[input_argmax[i] + i / argmax_stride * output_stride] += static_cast<T2>(input_grad[i]);
    }
  };
  const int64_t input_num_data = ctx.Input(2)->NumElements();
  const int64_t max_core_num =
    std::max(static_cast<int64_t>(1), static_cast<int64_t>(aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2));
  const int64_t per_unit_size = input_num_data / max_core_num;
  const bool enable_parallel = ctx.Output(0)->GetDataSize() > kParallelDataSize;
  if (enable_parallel) {
    KERNEL_HANDLE_ERROR(
      CpuKernelUtils::ParallelFor(ctx, input_num_data, per_unit_size, shard_adaptive_max_pool_3d_grad),
      "AdaptiveMaxPool3dGrad compute failed.");
  } else {
    shard_adaptive_max_pool_3d_grad(0, input_num_data);
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kAdaptiveMaxPool3dGrad, AdaptiveMaxPool3dGradCpuKernel);
}  // namespace aicpu
