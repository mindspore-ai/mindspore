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
#include "trace_grad.h"
#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "utils/eigen_tensor.h"
#include "Eigen/Core"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kTraceGrad = "TraceGrad";

}  // namespace

// 定义命名空间aicpu
namespace aicpu {
// 实现自定义算子类的Compute函数
uint32_t TraceGradCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Tracegrad check input and output number failed.");
  Tensor *y_grad = ctx.Input(0);
  Tensor *x_shape = ctx.Input(1);
  Tensor *x_grad = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(y_grad->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 0 data failed.");
  KERNEL_CHECK_NULLPTR(x_shape->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 1 data failed.");
  KERNEL_CHECK_NULLPTR(x_grad->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed");
  KERNEL_CHECK_FALSE(ctx.Input(1)->NumElements() == 2, KERNEL_STATUS_PARAM_INVALID, "Expected matrix input.",
                     ctx.Input(1)->NumElements());
  KERNEL_LOG_DEBUG(
    "TraceGradCpuKernel[%s], y_grad: size[%llu];"
    "x_shape: size[%llu], x_grad: size[%llu].",
    ctx.GetOpType().c_str(), y_grad->GetDataSize(), x_shape->GetDataSize(), x_grad->GetDataSize());
  DataType data_type = ctx.Input(0)->GetDataType();
  DataType shape_type = ctx.Input(1)->GetDataType();

  switch (data_type) {
    case DT_INT8:
      switch (shape_type) {
        case DT_INT32:
          return TraceGradCompute<int8_t, int32_t>(ctx);
        case DT_INT64:
          return TraceGradCompute<int8_t, int64_t>(ctx);
        default:
          break;
      }
    case DT_INT16:
      switch (shape_type) {
        case DT_INT32:
          return TraceGradCompute<int16_t, int32_t>(ctx);
        case DT_INT64:
          return TraceGradCompute<int16_t, int64_t>(ctx);
        default:
          break;
      }
    case DT_INT32:
      switch (shape_type) {
        case DT_INT32:
          return TraceGradCompute<int32_t, int32_t>(ctx);
        case DT_INT64:
          return TraceGradCompute<int32_t, int64_t>(ctx);
        default:
          break;
      }
    case DT_INT64:
      switch (shape_type) {
        case DT_INT32:
          return TraceGradCompute<int64_t, int32_t>(ctx);
        case DT_INT64:
          return TraceGradCompute<int64_t, int64_t>(ctx);
        default:
          break;
      }
    case DT_UINT8:
      switch (shape_type) {
        case DT_INT32:
          return TraceGradCompute<uint8_t, int32_t>(ctx);
        case DT_INT64:
          return TraceGradCompute<uint8_t, int64_t>(ctx);
        default:
          break;
      }
    case DT_UINT16:
      switch (shape_type) {
        case DT_INT32:
          return TraceGradCompute<uint16_t, int32_t>(ctx);
        case DT_INT64:
          return TraceGradCompute<uint16_t, int64_t>(ctx);
        default:
          break;
      }
    case DT_UINT32:
      switch (shape_type) {
        case DT_INT32:
          return TraceGradCompute<uint32_t, int32_t>(ctx);
        case DT_INT64:
          return TraceGradCompute<uint32_t, int64_t>(ctx);
        default:
          break;
      }
    case DT_UINT64:
      switch (shape_type) {
        case DT_INT32:
          return TraceGradCompute<uint64_t, int32_t>(ctx);
        case DT_INT64:
          return TraceGradCompute<uint64_t, int64_t>(ctx);
        default:
          break;
      }
    case DT_FLOAT16:
      switch (shape_type) {
        case DT_INT32:
          return TraceGradCompute<Eigen::half, int32_t>(ctx);
        case DT_INT64:
          return TraceGradCompute<Eigen::half, int64_t>(ctx);
        default:
          break;
      }
    case DT_FLOAT:
      switch (shape_type) {
        case DT_INT32:
          return TraceGradCompute<float, int32_t>(ctx);
        case DT_INT64:
          return TraceGradCompute<float, int64_t>(ctx);
        default:
          break;
      }
    case DT_DOUBLE:
      switch (shape_type) {
        case DT_INT32:
          return TraceGradCompute<double, int32_t>(ctx);
        case DT_INT64:
          return TraceGradCompute<double, int64_t>(ctx);
        default:
          break;
      }
    default:
      KERNEL_LOG_ERROR("TraceGrad kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T1, typename T2>
uint32_t TraceGradCpuKernel::TraceGradCompute(CpuKernelContext &ctx) {
  auto input_x1 = reinterpret_cast<T1 *>(ctx.Input(0)->GetData());
  auto input_x2 = reinterpret_cast<T2 *>(ctx.Input(1)->GetData());
  auto output_x = reinterpret_cast<T1 *>(ctx.Output(0)->GetData());

  T2 m = *(input_x2);
  T2 n = *(input_x2 + 1);
  T1 *grad_input = new T1[m * n];
  for (T2 i = 0; i < m * n; i++) *(grad_input + i) = (T1)0;
  for (T2 i = 0; i < m; i++)
    for (T2 j = 0; j < n; j++) {
      if (i == j) *(grad_input + i * n + j) = *(input_x1);
    }
  for (T2 i = 0; i < m * n; i++) *(output_x + i) = *(grad_input + i);

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kTraceGrad, TraceGradCpuKernel);
}  // namespace aicpu
