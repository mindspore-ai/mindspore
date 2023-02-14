/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
 **/
#include "cumulativelogsumexp.h"

#include "cmath"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t KCumulativeLogsumexpInputNum = 2;
const uint32_t KCumulativeLogsumexpOutputNum = 1;
const float float16_exclusive_data = -65504e+0;
const float float_exclusive_data = -3.4028235e+38;
const double double_exclusive_data = -1.7976931348623157e+308;
const int64_t ParallelFor_size_float16 = 16 * 1024;
const int64_t ParallelFor_size_float32 = 32 * 1024;
const int64_t ParallelFor_size_double = 64 * 1024;
const char *KCumulativeLogsumexp = "CumulativeLogsumexp";
#define CUMULATIVELOGSUMEXP_COMPUTE_CASE(DTYPE, IN_TYPE, CTX)         \
  case (DTYPE): {                                                     \
    uint32_t result = CumulativeLogsumexpCompute<IN_TYPE>(CTX);       \
    if (result != KERNEL_STATUS_OK) {                                 \
      KERNEL_LOG_ERROR("CumulativeLogsumexp kernel compute failed."); \
      return result;                                                  \
    }                                                                 \
    break;                                                            \
  }
}  // namespace
namespace aicpu {
uint32_t CumulativeLogsumexpCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, KCumulativeLogsumexpInputNum, KCumulativeLogsumexpOutputNum),
                      "[%s] check input and output failed,", KCumulativeLogsumexp);
  KERNEL_HANDLE_ERROR(CumulativeLogsumexpCheck(ctx), "[%s] check params failed.", KCumulativeLogsumexp);
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    CUMULATIVELOGSUMEXP_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    CUMULATIVELOGSUMEXP_COMPUTE_CASE(DT_FLOAT, float, ctx)
    CUMULATIVELOGSUMEXP_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("CumulativeLogsumexp kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
uint32_t CumulativeLogsumexpCpuKernel::CumulativeLogsumexpCheck(CpuKernelContext &ctx) {
  KERNEL_CHECK_FALSE((ctx.Input(1)->GetDataType() == DT_INT16 || ctx.Input(1)->GetDataType() == DT_INT32),
                     KERNEL_STATUS_PARAM_INVALID, "Data type of axis is not support, axis data type is [%u].",
                     ctx.Input(1)->GetDataType())
  KERNEL_CHECK_FALSE(ctx.Input(1)->NumElements() == 1, KERNEL_STATUS_PARAM_INVALID, "axis is out of shape");
  int64_t axis;
  if (ctx.Input(1)->GetDataType() == DT_INT16) {
    axis = static_cast<int64_t>(*reinterpret_cast<int16_t *>(ctx.Input(1)->GetData()));
  } else {
    axis = static_cast<int64_t>(*reinterpret_cast<int32_t *>(ctx.Input(1)->GetData()));
  }
  KERNEL_CHECK_FALSE((axis < ctx.Input(0)->GetTensorShape()->GetDims()), KERNEL_STATUS_PARAM_INVALID,
                     "axis is larger than input dims - 1")
  KERNEL_CHECK_FALSE((axis >= -ctx.Input(0)->GetTensorShape()->GetDims()), KERNEL_STATUS_PARAM_INVALID,
                     "axis is lower than -input dims")
  std::vector<int64_t> shape_input = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_output = ctx.Output(0)->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((shape_input.size() != 0), KERNEL_STATUS_PARAM_INVALID,
                     "Input must be at least rank 1, got [%zu].", shape_input.size())
  KERNEL_CHECK_FALSE((shape_input.size() == shape_output.size()), KERNEL_STATUS_PARAM_INVALID,
                     "The output shape size should be same as the output shape size")
  DataType input0_type = ctx.Input(0)->GetDataType();
  DataType output0_type = ctx.Output(0)->GetDataType();
  KERNEL_CHECK_FALSE((input0_type == output0_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input0 [%s] need be same with output0 [%s] ", DTypeStr(input0_type).c_str(),
                     DTypeStr(output0_type).c_str())
  return KERNEL_STATUS_OK;
}
template <typename t>
void CumulativeProcess(uint32_t outer, uint32_t inner, uint32_t depth, bool reverse, bool exclusive, t *input_data,
                       t *output_data, DataType data_type) {
  for (size_t outer_index = 0; outer_index < outer; ++outer_index) {
    size_t outer_index_adj;
    if (reverse) {
      outer_index_adj = (outer - 1) - outer_index;
    } else {
      outer_index_adj = outer_index;
    }
    for (size_t inner_index = 0; inner_index < inner; ++inner_index) {
      double one = 1;
      double temp = 0;
      size_t inner_index_adj;
      if (reverse) {
        inner_index_adj = (inner - 1) - inner_index;
      } else {
        inner_index_adj = inner_index;
      }
      for (size_t depth_index = 0; depth_index < depth; ++depth_index) {
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
          if (depth_index == 0) {
            if (data_type == DT_FLOAT16) {
              output_data[index] = static_cast<t>(float16_exclusive_data);
            } else if (data_type == DT_FLOAT) {
              output_data[index] = static_cast<t>(float_exclusive_data);
            } else {
              output_data[index] = static_cast<t>(double_exclusive_data);
            }
            temp = static_cast<double>(input_data[index]);
          } else {
            output_data[index] = static_cast<t>(temp);
            double a = temp;
            double b, min0, max0;
            b = static_cast<double>(input_data[index]);
            min0 = (a < b) ? a : b;
            max0 = (a > b) ? a : b;
            temp = log(one + exp(min0 - max0)) + max0;
          }
        } else {
          if (depth_index == 0) {
            output_data[index] = input_data[index];
            temp = static_cast<double>(input_data[index]);
          } else {
            double a = temp;
            double b, min0, max0;
            b = static_cast<double>(input_data[index]);
            min0 = (a < b) ? a : b;
            max0 = (a > b) ? a : b;
            output_data[index] = static_cast<t>(log(one + exp(min0 - max0)) + max0);
            temp = log(one + exp(min0 - max0)) + max0;
          }
        }
      }
    }
  }
}
template <typename T>
uint32_t CumulativeLogsumexpCpuKernel::CumulativeLogsumexpCompute(CpuKernelContext &ctx) {
  auto input_data = static_cast<T *>(ctx.Input(0)->GetData());
  bool exclusive = false;
  bool reverse = false;
  AttrValue *exclusive_attr = ctx.GetAttr("exclusive");
  if (exclusive_attr != nullptr) {
    exclusive = exclusive_attr->GetBool();
  }
  AttrValue *reverse_attr = ctx.GetAttr("reverse");
  if (reverse_attr != nullptr) {
    reverse = reverse_attr->GetBool();
  }
  int64_t axis;
  if (ctx.Input(1)->GetDataType() == DT_INT16) {
    axis = static_cast<int64_t>(*reinterpret_cast<int16_t *>(ctx.Input(1)->GetData()));
  } else {
    axis = static_cast<int64_t>(*reinterpret_cast<int32_t *>(ctx.Input(1)->GetData()));
  }
  auto output_data = static_cast<T *>(ctx.Output(0)->GetData());
  auto shape = ctx.Input(0)->GetTensorShape();
  const int64_t rank = shape->GetDims();
  if (axis < 0) {
    axis += shape->GetDims();
  }
  uint32_t inner = 1;
  uint32_t outer = 1;
  uint32_t depth = 1;
  for (int32_t i = 0; i < rank; ++i) {
    if (i < axis) {
      inner *= shape->GetDimSize(i);
    } else if (i > axis) {
      outer *= shape->GetDimSize(i);
    } else {
      depth = shape->GetDimSize(i);
    }
  }  // end for
  auto data_type = ctx.Input(0)->GetDataType();
  int64_t data_num = ctx.Input(0)->NumElements();
  int64_t data_size = data_num * sizeof(T);
  if ((data_type == DT_FLOAT16 && data_size <= ParallelFor_size_float16) ||
      (data_type == DT_FLOAT && data_size <= ParallelFor_size_float32) ||
      (data_type == DT_DOUBLE && data_size <= ParallelFor_size_double)) {
    CumulativeProcess<T>(outer, inner, depth, reverse, exclusive, input_data, output_data, data_type);
  } else {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (max_core_num > outer) {
      max_core_num = outer;
    }
    auto shard_cumulativelogsumexp = [&](size_t start, size_t end) {
      CumulativeProcess<T>(outer, inner, depth, reverse, exclusive, input_data, output_data, data_type);
    };
    if (max_core_num == 0) {
      return KERNEL_STATUS_PARAM_INVALID;
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, outer, outer / max_core_num, shard_cumulativelogsumexp),
                        "CumulativeLogsumexp Compute failed.");
  }  // end else
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(KCumulativeLogsumexp, CumulativeLogsumexpCpuKernel);
}  // namespace aicpu
