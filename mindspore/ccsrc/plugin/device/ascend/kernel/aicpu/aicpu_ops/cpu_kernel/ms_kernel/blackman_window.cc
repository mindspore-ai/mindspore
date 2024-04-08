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

// 引入声明算子类的头文件
#include "cpu_kernel/ms_kernel/blackman_window.h"

#include <math.h>
#include <iostream>
#include <vector>

#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#define _USE_MATH_DEFINES

namespace {
// 输入输出的个数
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *kBlackmanWindow = "BlackmanWindow";
}  // namespace
// 定义命名空间aicpu
namespace aicpu {
// 实现自定义算子类的Compute函数
uint32_t BlackmanWindowCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "BlackmanWindow check input and output number failed.");
  // input
  Tensor *input = ctx.Input(0);

  auto inputShape = input->GetTensorShape();  // 获取输入的shape信息
  DataType inputType = input->GetDataType();  // 获取输入的DataType信息

  // output
  Tensor *output = ctx.Output(0);
  CUST_KERNEL_CHECK_NULLPTR(ctx, output, KERNEL_STATUS_PARAM_INVALID, "Get output failed.")

  auto outputShape = output->GetTensorShape();  // 获取输出的shape信息
  DataType outputType = output->GetDataType();  // 获取输出的DataType信息

  // attr
  AttrValue *periodic = ctx.GetAttr("periodic");
  CUST_KERNEL_CHECK_NULLPTR(ctx, periodic, KERNEL_STATUS_PARAM_INVALID, "Get periodic failed.")

  uint32_t result;
  switch (inputType) {
    case DT_INT32:
      switch (outputType) {
        case DT_FLOAT:
          result = BlackmanWindowCompute<int32_t, float>(ctx);
          return result;
        case DT_DOUBLE:
          result = BlackmanWindowCompute<int32_t, double>(ctx);
          return result;
        case DT_FLOAT16:
          result = BlackmanWindowCompute2<int32_t, Eigen::half>(ctx);
          return result;
        default:
          CUST_KERNEL_LOG_ERROR(ctx, "BlackmanWinow kernel output type not support.");
          return KERNEL_STATUS_PARAM_INVALID;
      }
      break;
    case DT_INT64:
      switch (outputType) {
        case DT_FLOAT:
          result = BlackmanWindowCompute<int64_t, float>(ctx);
          return result;
        case DT_DOUBLE:
          result = BlackmanWindowCompute<int64_t, double>(ctx);
          return result;
          break;
        case DT_FLOAT16:
          result = BlackmanWindowCompute2<int64_t, Eigen::half>(ctx);
          return result;
          break;
        default:
          CUST_KERNEL_LOG_ERROR(ctx, "BlackmanWinow kernel output type not support.");
          return KERNEL_STATUS_PARAM_INVALID;
      }
      break;
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "BlackmanWinow kernel input type not support.");
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename T2>
uint32_t BlackmanWindowCpuKernel::BlackmanWindowCompute(CpuKernelContext &ctx) {
  // 输入数据
  auto input = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto window_length = *input;
  // 输出数据
  auto output = reinterpret_cast<T2 *>(ctx.Output(0)->GetData());
  auto y = ctx.Output(0);
  // 属性 periodic
  auto periodic = ctx.GetAttr("periodic");

  bool periodic_value = periodic->GetBool();
  double pre_window_length = window_length;

  // 更新输出的动态shape
  auto output_shape = y->GetTensorShape();
  std::vector<int64_t> dim_vector;
  dim_vector.push_back(window_length);
  output_shape->SetDimSizes(dim_vector);

  if (window_length == 0) {
    T2 output_temp = (T2)0;
    *output = output_temp;
    return KERNEL_STATUS_OK;
  }
  if (window_length == 1) {
    T2 output_temp = (T2)1;
    *output = output_temp;
    return KERNEL_STATUS_OK;
  }
  if (periodic_value) {
    window_length += 1;
  }

  T2 end = (T2)0.42;

  for (int i = 0; i < pre_window_length; i++) {
    T2 temp = (T2)0.08 * (T2)cos((T2)((T2)4 * (T2)M_PI * (T2)i) / ((T2)window_length - (T2)1)) -
              (T2)0.5 * (T2)cos((T2)((T2)2 * (T2)M_PI * (T2)i) / ((T2)window_length - (T2)1)) + end;
    *(output + i) = temp;
  }

  return KERNEL_STATUS_OK;
}

template <typename T, typename T2>
uint32_t BlackmanWindowCpuKernel::BlackmanWindowCompute2(CpuKernelContext &ctx) {
  // 输入数据
  auto input = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto window_length = *input;
  // 输出数据
  auto output = reinterpret_cast<T2 *>(ctx.Output(0)->GetData());
  // float* output_data;
  auto y = ctx.Output(0);
  // 属性 periodic
  auto periodic = ctx.GetAttr("periodic");

  bool periodic_value = periodic->GetBool();
  double pre_window_length = window_length;

  // 更新输出的动态shape
  auto output_shape = y->GetTensorShape();
  std::vector<int64_t> dim_vector;
  dim_vector.push_back(window_length);
  output_shape->SetDimSizes(dim_vector);

  if (window_length == 0) {
    *output = (Eigen::half)0;
    return KERNEL_STATUS_OK;
  }
  if (window_length == 1) {
    *output = (Eigen::half)1;
    return KERNEL_STATUS_OK;
  }
  if (periodic_value) {
    window_length += 1;
  }

  float end = 0.42;
  for (int i = 0; i < pre_window_length; i++) {
    float temp =
      0.08 * cos((4 * M_PI * i) / (window_length - 1)) - 0.5 * cos((2 * M_PI * i) / (window_length - 1)) + end;
    *(output + i) = (Eigen::half)temp;
  }

  return KERNEL_STATUS_OK;
}

// 注册该算子实现
REGISTER_MS_CPU_KERNEL(kBlackmanWindow, BlackmanWindowCpuKernel);
}  // namespace aicpu
