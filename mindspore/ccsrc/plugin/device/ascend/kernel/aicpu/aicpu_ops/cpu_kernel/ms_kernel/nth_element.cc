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
#include "nth_element.h"

#include <vector>
#include <algorithm>
#include "context/inc/cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "utils/eigen_tensor.h"
#include "inc/kernel_log.h"
#include "context/common/status.h"
using namespace std;

namespace {
const char *kNthElement = "NthElement";
constexpr uint64_t kParallelDataNums = 32 * 1024;

#define NTHELEMENT_COMPUTE_CASE(DTYPE, TYPE, X, Y, N, LAST_DIM, CTX)   \
  case (DTYPE): {                                                      \
    uint32_t result = NthElementCompute<TYPE>(X, Y, N, LAST_DIM, CTX); \
    if (result != KERNEL_STATUS_OK) {                                  \
      CUST_KERNEL_LOG_ERROR(ctx, "NthElement kernel compute failed."); \
      return result;                                                   \
    }                                                                  \
    break;                                                             \
  }
}  // namespace

namespace aicpu {
uint32_t NthElement::Compute(CpuKernelContext &ctx) {
  Tensor *input_n = ctx.Input(1);
  auto shape_n = input_n->GetTensorShape();
  CUST_KERNEL_CHECK_FALSE(ctx,
                          (shape_n->GetDimSizes().empty() || (shape_n->GetDims() == 1 && shape_n->GetDimSize(0) == 1)),
                          KERNEL_STATUS_PARAM_INVALID, "Input n must be a scalar or a single 1-dimension number.");
  DataType n_type = input_n->GetDataType();
  CUST_KERNEL_CHECK_FALSE(ctx, (n_type == DT_INT32), KERNEL_STATUS_PARAM_INVALID, "The type of input n must be int32.");
  CUST_KERNEL_CHECK_NULLPTR(ctx, input_n->GetData(), KERNEL_STATUS_PARAM_INVALID, "NthElement Get input n failed.");
  int32_t *n_data = reinterpret_cast<int32_t *>(input_n->GetData());
  int32_t n = *n_data;
  CUST_KERNEL_CHECK_FALSE(ctx, (n >= 0), KERNEL_STATUS_PARAM_INVALID, "Input n must be non-negative but is [%d].", n);

  Tensor *x = ctx.Input(0);
  CUST_KERNEL_CHECK_NULLPTR(ctx, x, KERNEL_STATUS_PARAM_INVALID, "NthElement Get input x failed.");
  auto x_shape = x->GetTensorShape();
  int32_t dims = x_shape->GetDims();
  CUST_KERNEL_CHECK_FALSE(ctx, (dims >= 1), KERNEL_STATUS_PARAM_INVALID,
                          "Input x must be at least rank 1 but is rank [%d]", dims);
  const int32_t last_dim = x_shape->GetDimSize(dims - 1);
  CUST_KERNEL_CHECK_FALSE(ctx, (last_dim > n), KERNEL_STATUS_PARAM_INVALID,
                          "Input x must have last dimension = [%d] > n = [%d]", last_dim, n);

  AttrValue *reverse_attr = ctx.GetAttr("reverse");
  CUST_KERNEL_CHECK_NULLPTR(ctx, reverse_attr, KERNEL_STATUS_PARAM_INVALID, "NthElement get attr reverse failed.");
  bool reverse = reverse_attr->GetBool();
  if (reverse) {
    n = last_dim - n - 1;
  }

  Tensor *y = ctx.Output(0);

  auto x_type = x->GetDataType();
  switch (x_type) {
    NTHELEMENT_COMPUTE_CASE(DT_FLOAT, float, x, y, n, last_dim, ctx)
    NTHELEMENT_COMPUTE_CASE(DT_FLOAT16, Eigen::half, x, y, n, last_dim, ctx)
    NTHELEMENT_COMPUTE_CASE(DT_UINT8, uint8_t, x, y, n, last_dim, ctx)
    NTHELEMENT_COMPUTE_CASE(DT_UINT16, uint16_t, x, y, n, last_dim, ctx)
    NTHELEMENT_COMPUTE_CASE(DT_INT8, int8_t, x, y, n, last_dim, ctx)
    NTHELEMENT_COMPUTE_CASE(DT_INT16, int16_t, x, y, n, last_dim, ctx)
    NTHELEMENT_COMPUTE_CASE(DT_INT32, int32_t, x, y, n, last_dim, ctx)
    NTHELEMENT_COMPUTE_CASE(DT_INT64, int64_t, x, y, n, last_dim, ctx)
    NTHELEMENT_COMPUTE_CASE(DT_DOUBLE, double, x, y, n, last_dim, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "[%s] Data type of input is not support, input data type is [%s].",
                            ctx.GetOpType().c_str(), DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t NthElement::NthElementCompute(Tensor *x, Tensor *y, const int32_t n, const int32_t last_dim,
                                       CpuKernelContext &ctx) {
  T *x_addrs = reinterpret_cast<T *>(x->GetData());
  T *y_addrs = reinterpret_cast<T *>(y->GetData());

  const uint64_t num_rows = y->NumElements();
  const uint64_t num = x->NumElements();

  if (num <= kParallelDataNums) {
    std::vector<T> buf(last_dim);
    for (size_t i = 0; i < num_rows; i++) {
      const T *input_start = x_addrs + i * last_dim;
      const T *input_end = input_start + last_dim;
      std::copy(input_start, input_end, buf.begin());
      std::nth_element(buf.begin(), buf.begin() + n, buf.end());
      y_addrs[i] = buf[n];
    }
  } else {
    uint32_t min_core_num = 1;
    uint64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (max_core_num > num_rows) {
      max_core_num = num_rows;
    }
    auto shard_nth_element = [&](size_t start, size_t end) {
      std::vector<T> buf(last_dim);
      for (size_t i = start; i < end; ++i) {
        const T *input_start = x_addrs + i * last_dim;
        const T *input_end = input_start + last_dim;
        std::copy(input_start, input_end, buf.begin());
        std::nth_element(buf.begin(), buf.begin() + n, buf.end());
        y_addrs[i] = buf[n];
      }
    };
    if (max_core_num == 0) {
      CUST_KERNEL_LOG_ERROR(ctx, "max_core_num could not be 0");
    }
    CUST_KERNEL_HANDLE_ERROR(ctx,
                             CpuKernelUtils::ParallelFor(ctx, num_rows, num_rows / max_core_num, shard_nth_element),
                             "NthElement Parallel Compute failed.");
  }

  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kNthElement, NthElement);
}  // namespace aicpu
