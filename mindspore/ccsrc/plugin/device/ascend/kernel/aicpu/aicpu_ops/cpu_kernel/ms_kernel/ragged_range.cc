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
#include "ms_kernel/ragged_range.h"

#include <vector>
#include <cmath>
#include <type_traits>
#include <algorithm>

#include "context/inc/cpu_kernel_utils.h"
#include "inc/kernel_log.h"
#include "context/common/status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 2;
const uint32_t kInputNum = 3;
const char *kRaggedRange = "RaggedRange";
constexpr int64_t kParallelDataNums = 16 * 1024;

#define RAGGEDRANGE_COMPUTE_CASE(DTYPE, TYPE, TSPLITS, NROWS, STARTS, LIMITS, DELTAS, BROADCAST_START,      \
                                 BROADCAST_LIMITS, BROADCAST_DELTAS, RT_NESTED_SPLITS, RT_DENSE_VALUE, CTX) \
  case (DTYPE): {                                                                                           \
    uint32_t result =                                                                                       \
      RaggedRangeCompute<TYPE, TSPLITS>(NROWS, STARTS, LIMITS, DELTAS, BROADCAST_START, BROADCAST_LIMITS,   \
                                        BROADCAST_DELTAS, RT_NESTED_SPLITS, RT_DENSE_VALUE, CTX);           \
    if (result != KERNEL_STATUS_OK) {                                                                       \
      CUST_KERNEL_LOG_ERROR(ctx, "RaggedRange kernel compute failed.");                                     \
      return result;                                                                                        \
    }                                                                                                       \
    break;                                                                                                  \
  }

}  // namespace

namespace aicpu {
uint32_t RaggedRange::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "RaggedRange check params failed.");
  Tensor *starts = ctx.Input(0);
  auto starts_shape = starts->GetTensorShape();
  int32_t starts_dim = starts_shape->GetDims();

  Tensor *limits = ctx.Input(1);
  auto limits_shape = limits->GetTensorShape();
  int32_t limits_dim = limits_shape->GetDims();

  Tensor *deltas = ctx.Input(2);
  auto deltas_shape = deltas->GetTensorShape();
  int32_t deltas_dim = deltas_shape->GetDims();

  CUST_KERNEL_CHECK_FALSE(ctx, (starts_dim <= 1), KERNEL_STATUS_PARAM_INVALID, "starts must be a scalar or vector.");
  CUST_KERNEL_CHECK_FALSE(ctx, (limits_dim <= 1), KERNEL_STATUS_PARAM_INVALID, "limits must be a scalar or vector.");
  CUST_KERNEL_CHECK_FALSE(ctx, (deltas_dim <= 1), KERNEL_STATUS_PARAM_INVALID, "deltas must be a scalar or vector.");

  bool broadcast_starts = starts_dim == 0;
  bool broadcast_limits = limits_dim == 0;
  bool broadcast_deltas = deltas_dim == 0;

  std::vector<int> in_sizes;
  if (!broadcast_starts) in_sizes.push_back(starts_shape->GetDimSize(0));
  if (!broadcast_limits) in_sizes.push_back(limits_shape->GetDimSize(0));
  if (!broadcast_deltas) in_sizes.push_back(deltas_shape->GetDimSize(0));
  for (uint32_t i = 1; i < in_sizes.size(); ++i) {
    CUST_KERNEL_CHECK_FALSE(ctx, (in_sizes[i] == in_sizes[i - 1]), KERNEL_STATUS_PARAM_INVALID,
                            "starts, limits, and deltas must have the same shape.");
  }

  uint32_t nrows = in_sizes.empty() ? 1 : in_sizes[0];

  AttrValue *attr = ctx.GetAttr("Tsplits");
  CUST_KERNEL_CHECK_NULLPTR(ctx, attr, KERNEL_STATUS_PARAM_INVALID, "Get attr[Tsplits] failed.");
  DataType Tsplits = attr->GetDataType();
  CUST_KERNEL_CHECK_FALSE(ctx, (Tsplits == DT_INT32 || Tsplits == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                          "The attr Tsplits must be int32 or int64.");

  Tensor *rt_nested_splits = ctx.Output(0);
  Tensor *rt_dense_values = ctx.Output(1);

  auto starts_type = starts->GetDataType();
  auto limits_type = limits->GetDataType();
  auto deltas_type = deltas->GetDataType();
  CUST_KERNEL_CHECK_FALSE(ctx, (starts_type == limits_type && limits_type == deltas_type), KERNEL_STATUS_PARAM_INVALID,
                          "starts, limits and deltas must have the same type.");

  if (Tsplits == DT_INT32) {
    switch (starts_type) {
      RAGGEDRANGE_COMPUTE_CASE(DT_FLOAT, float, int32_t, nrows, starts, limits, deltas, broadcast_starts,
                               broadcast_limits, broadcast_deltas, rt_nested_splits, rt_dense_values, ctx)
      RAGGEDRANGE_COMPUTE_CASE(DT_DOUBLE, double, int32_t, nrows, starts, limits, deltas, broadcast_starts,
                               broadcast_limits, broadcast_deltas, rt_nested_splits, rt_dense_values, ctx)
      RAGGEDRANGE_COMPUTE_CASE(DT_INT32, int32_t, int32_t, nrows, starts, limits, deltas, broadcast_starts,
                               broadcast_limits, broadcast_deltas, rt_nested_splits, rt_dense_values, ctx)
      RAGGEDRANGE_COMPUTE_CASE(DT_INT64, int64_t, int32_t, nrows, starts, limits, deltas, broadcast_starts,
                               broadcast_limits, broadcast_deltas, rt_nested_splits, rt_dense_values, ctx)
      default:
        CUST_KERNEL_LOG_ERROR(ctx, "[%s] Data type of input is not support, input data type is [%s].",
                              ctx.GetOpType().c_str(), DTypeStr(starts_type).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
  } else {
    switch (starts_type) {
      RAGGEDRANGE_COMPUTE_CASE(DT_FLOAT, float, int64_t, nrows, starts, limits, deltas, broadcast_starts,
                               broadcast_limits, broadcast_deltas, rt_nested_splits, rt_dense_values, ctx)
      RAGGEDRANGE_COMPUTE_CASE(DT_DOUBLE, double, int64_t, nrows, starts, limits, deltas, broadcast_starts,
                               broadcast_limits, broadcast_deltas, rt_nested_splits, rt_dense_values, ctx)
      RAGGEDRANGE_COMPUTE_CASE(DT_INT32, int32_t, int64_t, nrows, starts, limits, deltas, broadcast_starts,
                               broadcast_limits, broadcast_deltas, rt_nested_splits, rt_dense_values, ctx)
      RAGGEDRANGE_COMPUTE_CASE(DT_INT64, int64_t, int64_t, nrows, starts, limits, deltas, broadcast_starts,
                               broadcast_limits, broadcast_deltas, rt_nested_splits, rt_dense_values, ctx)
      default:
        CUST_KERNEL_LOG_ERROR(ctx, "[%s] Data type of input is not support, input data type is [%s].",
                              ctx.GetOpType().c_str(), DTypeStr(starts_type).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  return KERNEL_STATUS_OK;
}

template <typename T, typename TSPLITS>
uint32_t RaggedRange::RaggedRangeCompute(const uint32_t nrows, Tensor *starts, Tensor *limits, Tensor *deltas,
                                         bool broadcast_starts, bool broadcast_limits, bool broadcast_deltas,
                                         Tensor *rt_nested_splits, Tensor *rt_dense_values, CpuKernelContext &ctx) {
  T *starts_addr = reinterpret_cast<T *>(starts->GetData());
  T *limits_addr = reinterpret_cast<T *>(limits->GetData());
  T *deltas_addr = reinterpret_cast<T *>(deltas->GetData());

  TSPLITS *rt_nested_splits_addr = reinterpret_cast<TSPLITS *>(rt_nested_splits->GetData());
  rt_nested_splits_addr[0] = 0;
  for (uint32_t row = 0; row < nrows; ++row) {
    T start = broadcast_starts ? starts_addr[0] : starts_addr[row];
    T limit = broadcast_limits ? limits_addr[0] : limits_addr[row];
    T delta = broadcast_deltas ? deltas_addr[0] : deltas_addr[row];
    CUST_KERNEL_CHECK_FALSE(ctx, (delta != 0), KERNEL_STATUS_PARAM_INVALID, "Requires delta != 0.");
    rt_nested_splits_addr[row + 1] = rt_nested_splits_addr[row] + RangeSize<T, TSPLITS>(start, limit, delta);
  }

  T *rt_dense_values_addr = reinterpret_cast<T *>(rt_dense_values->GetData());
  if (nrows <= kParallelDataNums) {
    int value_index = 0;
    for (uint32_t row = 0; row < nrows; ++row) {
      TSPLITS row_size = rt_nested_splits_addr[row + 1] - rt_nested_splits_addr[row];
      T value = broadcast_starts ? starts_addr[0] : starts_addr[row];
      T delta = broadcast_deltas ? deltas_addr[0] : deltas_addr[row];
      for (TSPLITS i = 0; i < row_size; ++i) {
        rt_dense_values_addr[value_index++] = value;
        value += delta;
      }
    }
  } else {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num > nrows) {
      max_core_num = nrows;
    }
    auto shared_rtvalues = [&](size_t start, size_t end) {
      for (size_t row = start; row < end; row++) {
        TSPLITS row_size = rt_nested_splits_addr[row + 1] - rt_nested_splits_addr[row];
        T value = broadcast_starts ? starts_addr[0] : starts_addr[row];
        T delta = broadcast_deltas ? deltas_addr[0] : deltas_addr[row];
        TSPLITS y_offset = rt_nested_splits_addr[row];
        for (TSPLITS i = 0; i < row_size; ++i) {
          rt_dense_values_addr[y_offset++] = value;
          value += delta;
        }
      }
    };
    uint32_t ret = CpuKernelUtils::ParallelFor(ctx, nrows, nrows / max_core_num, shared_rtvalues);
    if (ret != KERNEL_STATUS_OK) {
      CUST_KERNEL_LOG_ERROR(ctx, "CpuKernelUtils::ParallelFor failed.");
      return KERNEL_STATUS_INNER_ERROR;
    }
  }

  return KERNEL_STATUS_OK;
}

template <typename T, typename TSPLITS>
TSPLITS RaggedRange::RangeSize(T start, T limit, T delta) {
  if (((delta > 0) && (limit < start)) || ((delta < 0) && (limit > start))) {
    return 0;
  }
  return (std::is_integral<T>::value ? ((std::abs(limit - start) + std::abs(delta) - 1) / std::abs(delta))
                                     : std::ceil(std::abs((limit - start) / delta)));
}

REGISTER_MS_CPU_KERNEL(kRaggedRange, RaggedRange);
}  // namespace aicpu
