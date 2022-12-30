/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All right reserved.
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
#include "logit_grad.h"

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/LU"
#include "cmath"
#include "cpu_context.h"
#include "cpu_kernel_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const int64_t kParallelDataNumSameShape = 7 * 1024;
const int64_t kParallelDataNumSameShapeMid = 16 * 1024;
const char *kLogitGrad = "LogitGrad";

#define LOGITGRAD_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                           \
    uint32_t result = LogitGradCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                       \
      KERNEL_LOG_ERROR("LogitGrad kernel compute failed."); \
      return result;                                        \
    }                                                       \
    break;                                                  \
  }
}  // namespace

namespace aicpu {
uint32_t LogitGradCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kLogitGrad);
  DataType data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    LOGITGRAD_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    LOGITGRAD_COMPUTE_CASE(DT_FLOAT, float, ctx)
    LOGITGRAD_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("LogitGrad kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t LogitGradCpuKernel::LogitGradCompute(CpuKernelContext &ctx) {
  auto input_y_grad_tensor = ctx.Input(0);
  auto input_x_tensor = ctx.Input(1);
  auto output_x_grad_tensor = ctx.Output(0);
  auto input_y_grad = reinterpret_cast<T *>(input_y_grad_tensor->GetData());
  auto input_x = reinterpret_cast<T *>(input_x_tensor->GetData());
  auto output_x_grad = reinterpret_cast<T *>(output_x_grad_tensor->GetData());
  auto input_shape = input_x_tensor->GetTensorShape();
  int64_t data_num = input_shape->NumElements();
  float eps = -1.0;
  AttrValue *attr = ctx.GetAttr("eps");
  if (attr != nullptr) {
    eps = attr->GetFloat();
  }
  if (data_num >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (data_num <= kParallelDataNumSameShapeMid) {
      max_core_num = std::min(max_core_num, 4U);
    }
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shared_less = [&](size_t start, size_t end) {
      T one = T(1);
      T zero = T(0);
      T up_bound = static_cast<T>(1) - static_cast<T>(eps);
      if (eps < 0) {
        for (size_t i = start; i < end; i++) {
          T y_grad = input_y_grad[i];
          T x = input_x[i];
          output_x_grad[i] = (x < zero || x > one) ? std::numeric_limits<T>::quiet_NaN() : (y_grad / (x * (one - x)));
        }
      } else {
        for (size_t i = start; i < end; i++) {
          T y_grad = input_y_grad[i];
          T x = input_x[i];
          output_x_grad[i] =
            static_cast<float>(x) < static_cast<float>(eps) || static_cast<float>(x) > static_cast<float>(up_bound)
              ? zero
              : (y_grad / (x * (one - x)));
        }
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shared_less),
                        "LogitGrad Compute failed.");
  } else {
    T one = T(1);
    T zero = T(0);
    T up_bound = static_cast<T>(1) - static_cast<T>(eps);
    if (eps < 0) {
      for (int64_t i = 0; i < data_num; i++) {
        T y_grad = input_y_grad[i];
        T x = input_x[i];
        output_x_grad[i] = (x < zero || x > one) ? std::numeric_limits<T>::quiet_NaN() : (y_grad / (x * (one - x)));
      }
    } else {
      for (int64_t i = 0; i < data_num; i++) {
        T y_grad = input_y_grad[i];
        T x = input_x[i];
        output_x_grad[i] =
          static_cast<float>(x) < static_cast<float>(eps) || static_cast<float>(x) > static_cast<float>(up_bound)
            ? zero
            : (y_grad / (x * (one - x)));
      }
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kLogitGrad, LogitGradCpuKernel);
}  // namespace aicpu
