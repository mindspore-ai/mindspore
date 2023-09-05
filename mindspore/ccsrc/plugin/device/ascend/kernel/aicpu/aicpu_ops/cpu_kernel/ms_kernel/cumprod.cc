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
#include "cpu_kernel/ms_kernel/cumprod.h"

#include <algorithm>
#include <vector>

#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kCumprodInputNum = 2;
const uint32_t kCumprodOutputNum = 1;
const int64_t paralled_data_size = 512 * 1024;
const char *kCumprod = "Cumprod";
#define CUMPROD_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                         \
    uint32_t result = CumprodCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                     \
      KERNEL_LOG_ERROR("Cumprod kernel compute failed."); \
      return result;                                      \
    }                                                     \
    break;                                                \
  }
}  // namespace

namespace aicpu {
uint32_t CumprodCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kCumprodInputNum, kCumprodOutputNum), "[%s] check input and output failed.",
                      kCumprod);
  // parse params
  KERNEL_HANDLE_ERROR(CumprodCheck(ctx), "[%s] check params failed.", kCumprod);
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    CUMPROD_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    CUMPROD_COMPUTE_CASE(DT_FLOAT, float, ctx)
    CUMPROD_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    CUMPROD_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    CUMPROD_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    CUMPROD_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    CUMPROD_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    CUMPROD_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    CUMPROD_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    CUMPROD_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    CUMPROD_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    CUMPROD_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    CUMPROD_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      KERNEL_LOG_ERROR("Cumprod kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
uint32_t CumprodCpuKernel::CumprodCheck(const CpuKernelContext &ctx) {
  KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "get input failed.");
  KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID, "Get input tensor shape failed.")
  KERNEL_CHECK_NULLPTR(ctx.GetAttr("exclusive"), KERNEL_STATUS_PARAM_INVALID, "get exclusive failed.");
  KERNEL_CHECK_NULLPTR(ctx.GetAttr("reverse"), KERNEL_STATUS_PARAM_INVALID, "get reverse failed.");
  KERNEL_CHECK_FALSE((ctx.Input(1)->GetDataType() == DT_INT32 || ctx.Input(1)->GetDataType() == DT_INT64),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Data type of axis is not support, axis data type is [%u], only support int32 or int64.",
                     ctx.Input(1)->GetDataType());
  KERNEL_CHECK_FALSE(ctx.Input(1)->NumElements() == 1, KERNEL_STATUS_PARAM_INVALID, "axis is out of shape")
  auto axis_data = reinterpret_cast<int32_t *>(ctx.Input(1)->GetData());
  int32_t axis = *axis_data;
  KERNEL_CHECK_FALSE((axis < ctx.Input(0)->GetTensorShape()->GetDims()), KERNEL_STATUS_PARAM_INVALID,
                     "axis is larger than input dims - 1");
  KERNEL_CHECK_FALSE((axis >= -ctx.Input(0)->GetTensorShape()->GetDims()), KERNEL_STATUS_PARAM_INVALID,
                     "axis is lower than -input dims");
  std::vector<int64_t> shape_input = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_output = ctx.Output(0)->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((shape_input.size() != 0), KERNEL_STATUS_PARAM_INVALID,
                     "Input must be at least rank 1, got [%zu].", shape_input.size())
  KERNEL_CHECK_FALSE((shape_input.size() == shape_output.size()), KERNEL_STATUS_PARAM_INVALID,
                     "The output shape size should be same as the output shape size")
  return KERNEL_STATUS_OK;
}
template <typename T>
uint32_t CumprodCpuKernel::CumprodCompute(const CpuKernelContext &ctx) {
  auto input_data = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto axis_data = reinterpret_cast<int32_t *>(ctx.Input(1)->GetData());
  int32_t axis = *axis_data;
  bool exclusive = ctx.GetAttr("exclusive")->GetBool();
  bool reverse = ctx.GetAttr("reverse")->GetBool();
  auto output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto shape = ctx.Input(0)->GetTensorShape();
  const int64_t rank = shape->GetDims();
  if (axis < 0) {
    axis += shape->GetDims();
  }
  size_t inner = 1;
  size_t outer = 1;
  size_t depth = 1;
  for (int32_t i = 0; i < rank; ++i) {
    if (i < axis) {
      inner *= shape->GetDimSize(i);
    } else if (i > axis) {
      outer *= shape->GetDimSize(i);
    } else {
      depth = shape->GetDimSize(i);
    }
  }
  int64_t data_num = ctx.Input(0)->NumElements();
  int64_t data_size = data_num * sizeof(T);
  if (data_size <= paralled_data_size) {
    for (size_t outer_index = 0; outer_index < outer; ++outer_index) {
      size_t outer_index_adj;
      if (reverse)
        outer_index_adj = (outer - 1) - outer_index;
      else
        outer_index_adj = outer_index;
      for (size_t inner_index = 0; inner_index < inner; inner_index++) {
        auto multiplier = static_cast<T>(1);
        size_t inner_index_adj;
        if (reverse)
          inner_index_adj = (inner - 1) - inner_index;
        else
          inner_index_adj = inner_index;
        for (size_t depth_index = 0; depth_index < depth; depth_index++) {
          size_t depth_index_adj;
          if (reverse)
            depth_index_adj = (depth - 1) - depth_index;
          else
            depth_index_adj = depth_index;
          size_t index = outer_index_adj;
          index += inner_index_adj * depth * outer;
          index += depth_index_adj * outer;
          if (exclusive) {
            output_data[index] = multiplier;
            multiplier *= input_data[index];
          } else {
            multiplier *= input_data[index];
            output_data[index] = multiplier;
          }
        }
      }
    }
  } else {
    auto shard_cumprod = [&](size_t start, size_t ene) {
      for (size_t outer_index = 0; outer_index < outer; ++outer_index) {
        size_t outer_index_adj;
        if (reverse) {
          outer_index_adj = (outer - 1) - outer_index;
        } else {
          outer_index_adj = outer_index;
        }
        for (size_t inner_index = 0; inner_index < inner; inner_index++) {
          auto multiplier = static_cast<T>(1);
          size_t inner_index_adj;
          if (reverse) {
            inner_index_adj = (inner - 1) - inner_index;
          } else {
            inner_index_adj = inner_index;
          }
          for (size_t depth_index = 0; depth_index < depth; depth_index++) {
            size_t depth_index_adj;
            if (reverse) {
              depth_index_adj = (depth - 1) - depth_index;
            } else {
              depth_index_adj = depth_index;
            }
            size_t index = outer_index_adj;
            index += inner_index_adj * depth * outer;
            index += depth_index_adj * outer;
            if (exclusive) {
              output_data[index] = multiplier;
              multiplier *= input_data[index];
            } else {
              multiplier *= input_data[index];
              output_data[index] = multiplier;
            }
          }
        }
      }
    };
    uint32_t min_core_num = 1;
    size_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (max_core_num > outer) {
      max_core_num = outer;
    }
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0");
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, outer, outer / max_core_num, shard_cumprod),
                        "Cumprod Compute failed.");
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kCumprod, CumprodCpuKernel);
}  // namespace aicpu
