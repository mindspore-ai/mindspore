/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
#include "ms_kernel/reduce_sum.h"
#include <vector>
#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kReduceSumInputNum = 2;
const uint32_t kReduceSumOutputNum = 1;
const char *const kReduceSum = "ReduceSum";
#define REDUCESUM_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                                     \
    uint32_t result = ReduceSumCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                                 \
      CUST_KERNEL_LOG_ERROR(ctx, "ReduceSum kernel compute failed."); \
      return result;                                                  \
    }                                                                 \
    break;                                                            \
  }
#define REDUCESUM_COMPUTE_CASE_COMPLEX(DTYPE, TYPE, IN_TYPE, CTX)     \
  case (DTYPE): {                                                     \
    uint32_t result = ReduceSumCompute2<TYPE, IN_TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                                 \
      CUST_KERNEL_LOG_ERROR(ctx, "ReduceSum kernel compute failed."); \
      return result;                                                  \
    }                                                                 \
    break;                                                            \
  }
#define REDUCESUM_DEDUP_AXES(DTYPE, TYPE, CTX)                                 \
  case (DTYPE): {                                                              \
    uint32_t result = ReduceSumDedupAxes<TYPE>(CTX);                           \
    if (result != KERNEL_STATUS_OK) {                                          \
      CUST_KERNEL_LOG_ERROR(ctx, "ReduceSum kernel deduplicate axes failed."); \
      return result;                                                           \
    }                                                                          \
    break;                                                                     \
  }
}  // namespace

namespace aicpu {
uint32_t ReduceSumCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kReduceSumInputNum, kReduceSumOutputNum),
                           "[%s] check input and output failed.", kReduceSum);
  CUST_KERNEL_HANDLE_ERROR(ctx, ReduceSumCheck(ctx), "[%s] check params failed.", kReduceSum);

  auto axes_type = ctx.Input(1)->GetDataType();
  switch (axes_type) {
    REDUCESUM_DEDUP_AXES(DT_INT32, int32_t, ctx)
    REDUCESUM_DEDUP_AXES(DT_INT64, int64_t, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "ReduceSum kernel axes data type not support.", DTypeStr(axes_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  auto input_data_type = ctx.Input(0)->GetDataType();
  switch (input_data_type) {
    REDUCESUM_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    REDUCESUM_COMPUTE_CASE(DT_FLOAT, float, ctx)
    REDUCESUM_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    REDUCESUM_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    REDUCESUM_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    REDUCESUM_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    REDUCESUM_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    REDUCESUM_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    REDUCESUM_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    REDUCESUM_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    REDUCESUM_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    REDUCESUM_COMPUTE_CASE_COMPLEX(DT_COMPLEX64, std::complex<float>, float, ctx)
    REDUCESUM_COMPUTE_CASE_COMPLEX(DT_COMPLEX128, std::complex<double>, double, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "ReduceSum kernel data type [%s] not support.", DTypeStr(input_data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
uint32_t ReduceSumCpuKernel::ReduceSumCheck(CpuKernelContext &ctx) const {
  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "get input failed.");
  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.Input(0)->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID,
                            "Get input tensor shape failed.");
  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "get output failed.");
  if (ctx.Input(1)->GetData() != nullptr) {
    CUST_KERNEL_CHECK_FALSE(ctx, (ctx.Input(1)->GetDataType() == DT_INT32 || ctx.Input(1)->GetDataType() == DT_INT64),
                            KERNEL_STATUS_PARAM_INVALID, "Data type of axis is not support, axis data type is [%u].",
                            ctx.Input(1)->GetDataType());
  }
  return KERNEL_STATUS_OK;
}
template <typename T>
uint32_t ReduceSumCpuKernel::ReduceSumCompute(CpuKernelContext &ctx) {
  std::vector<int64_t> input_shape = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  auto input_data = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  if (input_shape.size() == 0) {
    output_data[0] = input_data[0];
    return KERNEL_STATUS_OK;
  }

  if (axes_.empty()) {
    int64_t data_num = ctx.Input(0)->NumElements();
    auto accumulator = static_cast<T>(0);
    for (int64_t i = 0; i < data_num; i++) {
      accumulator += input_data[i];
    }
    output_data[0] = accumulator;
    return KERNEL_STATUS_OK;
  }

  int64_t output_num = ctx.Output(0)->NumElements();
  uint32_t axes_idx = 0;
  CUST_KERNEL_HANDLE_ERROR(ctx,
                           ReduceSumOneAxes<T>(ctx, input_data, input_shape, output_data, output_num, axes_, axes_idx),
                           "Reduce sum compute failed.");
  return KERNEL_STATUS_OK;
}
template <typename T>
uint32_t ReduceSumCpuKernel::ReduceSumOneAxes(CpuKernelContext &ctx, const T *input_data,
                                              std::vector<int64_t> &input_shape, T *output_data, int64_t output_num,
                                              std::vector<int64_t> &axes, uint32_t &axes_idx) {
  if (axes_idx >= axes.size()) {
    for (int64_t i = 0; i < output_num; i++) {
      output_data[i] = input_data[i];
    }
    return KERNEL_STATUS_OK;
  }
  int64_t inner = 1;
  int64_t outer = 1;
  int64_t depth = 1;
  CUST_KERNEL_HANDLE_ERROR(ctx, ReduceSumParseAxes(input_shape, axes, axes_idx, inner, outer, depth),
                           "parse axes failed.");
  auto output_data_temp = new (std::nothrow) T[inner * outer];
  CUST_KERNEL_CHECK_NULLPTR(ctx, output_data_temp, KERNEL_STATUS_INNER_ERROR, "apply memory failed.");
  for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
    for (int64_t inner_index = 0; inner_index < inner; inner_index++) {
      auto accumulator = static_cast<T>(0);
      for (int64_t depth_index = 0; depth_index < depth; depth_index++) {
        int64_t index = outer_index;
        index += depth_index * outer;
        index += inner_index * depth * outer;
        accumulator += input_data[index];
      }
      int64_t output_index = outer_index;
      output_index += inner_index * outer;
      output_data_temp[output_index] = accumulator;
    }
  }
  uint32_t result = ReduceSumOneAxes<T>(ctx, output_data_temp, input_shape, output_data, output_num, axes, axes_idx);
  if (output_data_temp != nullptr) {
    delete[] output_data_temp;
  }
  return result;
}
template <typename T, typename T2>
uint32_t ReduceSumCpuKernel::ReduceSumCompute2(CpuKernelContext &ctx) {
  std::vector<int64_t> input_shape = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  auto input_data = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  if (input_shape.size() == 0) {
    output_data[0] = std::complex<T2>(input_data[0].real(), input_data[0].imag());
    return KERNEL_STATUS_OK;
  }

  int64_t input_num = ctx.Input(0)->NumElements();
  if (axes_.empty()) {
    auto accumulator_real = static_cast<T2>(0);
    auto accumulator_imag = static_cast<T2>(0);
    for (int64_t i = 0; i < input_num; i++) {
      accumulator_real += input_data[i].real();
      accumulator_imag += input_data[i].imag();
    }
    output_data[0] = std::complex<T2>(accumulator_real, accumulator_imag);
    return KERNEL_STATUS_OK;
  }

  int64_t output_num = ctx.Output(0)->NumElements();
  uint32_t axes_idx = 0;
  CUST_KERNEL_HANDLE_ERROR(
    ctx, (ReduceSumOneAxes2<T, T2>(ctx, input_data, input_num, input_shape, output_data, output_num, axes_, axes_idx)),
    "Reduce sum compute failed.");
  return KERNEL_STATUS_OK;
}
template <typename T, typename T2>
uint32_t ReduceSumCpuKernel::ReduceSumOneAxes2(CpuKernelContext &ctx, const T *input_data, int64_t input_num,
                                               std::vector<int64_t> input_shape, T *output_data, int64_t output_num,
                                               std::vector<int64_t> &axes, uint32_t &axes_idx) {
  if (axes_idx >= axes.size()) {
    auto accumulator_real = static_cast<T2>(0);
    auto accumulator_imag = static_cast<T2>(0);
    for (int64_t i = 0; i < output_num; i++) {
      accumulator_real = input_data[i].real();
      accumulator_imag = input_data[i].imag();
      output_data[i] = std::complex<T2>(accumulator_real, accumulator_imag);
    }
    return KERNEL_STATUS_OK;
  }
  int64_t inner = 1;
  int64_t outer = 1;
  int64_t depth = 1;
  CUST_KERNEL_HANDLE_ERROR(ctx, ReduceSumParseAxes(input_shape, axes, axes_idx, inner, outer, depth),
                           "parse axes failed.");
  std::vector<T2> input_data_real(input_num);
  std::vector<T2> input_data_imag(input_num);
  for (int64_t i = 0; i < input_num; i++) {
    input_data_real[i] = input_data[i].real();
    input_data_imag[i] = input_data[i].imag();
  }
  int64_t output_num_temp = inner * outer;
  auto *output_data_temp = new (std::nothrow) T[output_num_temp];
  CUST_KERNEL_CHECK_NULLPTR(ctx, output_data_temp, KERNEL_STATUS_INNER_ERROR, "apply memory failed.");
  for (int64_t outer_index = 0; outer_index < outer; outer_index++) {
    for (int64_t inner_index = 0; inner_index < inner; inner_index++) {
      auto accumulator_real = static_cast<T2>(0);
      auto accumulator_imag = static_cast<T2>(0);
      for (int64_t depth_index = 0; depth_index < depth; depth_index++) {
        int64_t index = outer_index;
        index += inner_index * depth * outer;
        index += depth_index * outer;
        accumulator_real += input_data_real[index];
        accumulator_imag += input_data_imag[index];
      }
      int64_t output_index = outer_index;
      output_index += inner_index * outer;
      output_data_temp[output_index] = std::complex<T2>(accumulator_real, accumulator_imag);
    }
  }
  uint32_t result = ReduceSumOneAxes2<T, T2>(ctx, output_data_temp, output_num_temp, input_shape, output_data,
                                             output_num, axes, axes_idx);
  if (output_data_temp != nullptr) {
    delete[] output_data_temp;
  }
  return result;
}

template <typename T1>
uint32_t ReduceSumCpuKernel::ReduceSumDedupAxes(CpuKernelContext &ctx) {
  int32_t rank = ctx.Input(0)->GetTensorShape()->GetDims();
  auto axes_data = reinterpret_cast<T1 *>(ctx.Input(1)->GetData());
  int64_t axes_num = ctx.Input(1)->NumElements();
  for (int64_t i = 0; i < axes_num; i++) {
    int32_t axis = axes_data[i];
    CUST_KERNEL_CHECK_FALSE(ctx, (axis < rank) && (axis >= -rank), KERNEL_STATUS_PARAM_INVALID,
                            "axes[%d] is out of input dims rank[%d]", axis, rank);
    if (axis < 0) {
      axis += rank;
    }
    axes_.push_back(axis);
  }
  int64_t j = 1;
  while (j < axes_num) {
    std::vector<int64_t>::iterator iter = find(axes_.begin(), axes_.begin() + j, axes_[j]);
    if (iter != axes_.begin() + j) {
      axes_.erase(iter);
      axes_num--;
    } else {
      j++;
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t ReduceSumCpuKernel::ReduceSumParseAxes(std::vector<int64_t> &input_shape, std::vector<int64_t> &axes,
                                                uint32_t &axes_idx, int64_t &inner, int64_t &outer,
                                                int64_t &depth) const {
  int64_t axis = axes[axes_idx];
  axes_idx++;
  int64_t rank = input_shape.size();
  for (int64_t i = 0; i < rank; i++) {
    if (i < axis) {
      inner *= input_shape[i];
    } else if (i > axis) {
      outer *= input_shape[i];
    } else {
      depth = input_shape[i];
      input_shape[i] = 1;
    }
  }
  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kReduceSum, ReduceSumCpuKernel);
}  // namespace aicpu
