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

#include "ms_kernel/rsqrt_grad.h"

#include <algorithm>
#include <complex>
#include <iostream>

#include "utils/eigen_tensor.h"

namespace {
const char *kRsqrtGrad = "RsqrtGrad";
constexpr uint32_t kOutputNum = 1;
constexpr uint32_t kInputNum = 2;
}  // namespace

namespace aicpu {
uint32_t RsqrtGradCpuKernel::Compute(CpuKernelContext &ctx) {
  if (NormalCheck(ctx, kInputNum, kOutputNum) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *input_0 = ctx.Input(kFirstInputIndex);
  Tensor *input_1 = ctx.Input(kSecondInputIndex);
  if ((input_0->GetDataSize() == 0) || (input_1->GetDataSize() == 0)) {
    CUST_KERNEL_LOG_INFO(ctx, "[%s] Input is empty tensor.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_OK;
  }
  // choose compute function depend on dataType
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    case DT_FLOAT16:
      return RsqrtGradComputeFP16<Eigen::half>(ctx);
    case DT_FLOAT:
      return RsqrtGradCompute<float>(ctx);
    case DT_DOUBLE:
      return RsqrtGradCompute<double>(ctx);
    case DT_INT8:
      return RsqrtGradCompute<int8_t>(ctx);
    case DT_INT32:
      return RsqrtGradCompute<int32_t>(ctx);
    case DT_COMPLEX128:
      return RsqrtGradComputeComplex<std::complex<double>>(ctx);
    case DT_COMPLEX64:
      return RsqrtGradComputeComplex<std::complex<float>>(ctx);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "[%s] Data type of input is not support, input data type is [%s].",
                            ctx.GetOpType().c_str(), aicpu::DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T>
uint32_t RsqrtGradCpuKernel::RsqrtGradComputeFP16(CpuKernelContext &ctx) {
  Tensor *y = ctx.Input(0);
  Tensor *dy = ctx.Input(1);
  Tensor *z = ctx.Output(0);
  auto y_ptr = reinterpret_cast<T *>(y->GetData());
  auto dy_ptr = reinterpret_cast<T *>(dy->GetData());
  auto z_ptr = reinterpret_cast<T *>(z->GetData());
  int32_t input_0_num = y->GetTensorShape()->NumElements();
  int32_t input_1_num = dy->GetTensorShape()->NumElements();

  if (input_0_num >= input_1_num) {
    for (int32_t i = 0; i < input_1_num; i++) {
      z_ptr[i] =
        static_cast<T>((static_cast<double>(y_ptr[i]) * static_cast<double>(y_ptr[i]) * static_cast<double>(y_ptr[i])) *
                       (static_cast<double>(dy_ptr[i]) / (static_cast<double>(-2))));
    }
    for (int32_t i = input_1_num; i < input_0_num; i++) {
      z_ptr[i] = (T)(0);
    }
  } else {
    for (int32_t i = 0; i < input_0_num; i++) {
      z_ptr[i] =
        static_cast<T>((static_cast<double>(y_ptr[i]) * static_cast<double>(y_ptr[i]) * static_cast<double>(y_ptr[i])) *
                       (static_cast<double>(dy_ptr[i]) / (static_cast<double>(-2))));
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t RsqrtGradCpuKernel::RsqrtGradCompute(CpuKernelContext &ctx) {
  Tensor *y = ctx.Input(0);
  Tensor *dy = ctx.Input(1);
  Tensor *z = ctx.Output(0);

  CUST_KERNEL_CHECK_NULLPTR(ctx, z->GetData(), KERNEL_STATUS_PARAM_INVALID, "[%s] Get output data failed",
                            ctx.GetOpType().c_str())
  CUST_KERNEL_LOG_INFO(ctx,
                       "[%s] Input[0] data size is [%llu], input[1] data size is [%llu], output "
                       "data size is [%llu].",
                       ctx.GetOpType().c_str(), y->GetDataSize(), dy->GetDataSize(), z->GetDataSize());
  auto y_ptr = reinterpret_cast<T *>(y->GetData());
  auto dy_ptr = reinterpret_cast<T *>(dy->GetData());
  auto z_ptr = reinterpret_cast<T *>(z->GetData());
  int32_t input_0_num = y->GetTensorShape()->NumElements();
  int32_t input_1_num = dy->GetTensorShape()->NumElements();

  if (input_0_num >= input_1_num) {
    for (int32_t i = 0; i < input_1_num; i++) {
      z_ptr[i] = (dy_ptr[i] * y_ptr[i] * y_ptr[i] * y_ptr[i]) / (static_cast<T>(-2));
    }
    for (int32_t i = input_1_num; i < input_0_num; i++) {
      z_ptr[i] = (T)(0);
    }
  } else {
    for (int32_t i = 0; i < input_0_num; i++) {
      z_ptr[i] = (dy_ptr[i] * y_ptr[i] * y_ptr[i] * y_ptr[i]) / (static_cast<T>(-2));
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t RsqrtGradCpuKernel::RsqrtGradComputeComplex(CpuKernelContext &ctx) {
  Tensor *y = ctx.Input(0);
  Tensor *dy = ctx.Input(1);
  Tensor *z = ctx.Output(0);

  CUST_KERNEL_CHECK_NULLPTR(ctx, z->GetData(), KERNEL_STATUS_PARAM_INVALID, "[%s] Get output data failed",
                            ctx.GetOpType().c_str())
  CUST_KERNEL_LOG_INFO(ctx,
                       "[%s] Input[0] data size is [%llu], input[1] data size is [%llu], output "
                       "data size is [%llu].",
                       ctx.GetOpType().c_str(), y->GetDataSize(), dy->GetDataSize(), z->GetDataSize());
  auto y_ptr = reinterpret_cast<T *>(y->GetData());
  auto dy_ptr = reinterpret_cast<T *>(dy->GetData());
  auto z_ptr = reinterpret_cast<T *>(z->GetData());
  int32_t input_0_num = y->GetTensorShape()->NumElements();
  int32_t input_1_num = dy->GetTensorShape()->NumElements();
  if (input_0_num >= input_1_num) {
    for (int32_t i = 0; i < input_1_num; i++) {
      z_ptr[i] = (dy_ptr[i] * conj(y_ptr[i]) * conj(y_ptr[i]) * conj(y_ptr[i])) * (static_cast<T>(-0.5));
    }
    for (int32_t i = input_1_num; i < input_0_num; i++) {
      z_ptr[i] = static_cast<T>(0);
    }
  } else {
    for (int32_t i = 0; i < input_0_num; i++) {
      z_ptr[i] = (dy_ptr[i] * conj(y_ptr[i]) * conj(y_ptr[i]) * conj(y_ptr[i])) * (static_cast<T>(-0.5));
    }
  }

  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kRsqrtGrad, RsqrtGradCpuKernel);
}  // namespace aicpu
