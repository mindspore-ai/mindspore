/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
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
#include "cpu_kernel/ms_kernel/sparse_segment_sqrt_n.h"
#include <math.h>
#include <vector>
#include "Eigen/Core"
#include "utils/kernel_util.h"

namespace aicpu {
const uint32_t kInputNum = 3;
const uint32_t kOutputNum = 1;
const char *SparseSegmentSqrtN = "SparseSegmentSqrtN";

#define COMPUTE_CASE(DTYPE, TYPE, DTYPE_1, DTYPE_2, CTX)   \
  case (DTYPE):                                            \
    if ((DTYPE_1) == DT_INT32) {                           \
      if ((DTYPE_2) == DT_INT32) {                         \
        return ComputeKernel<TYPE, int32_t, int32_t>(CTX); \
      } else {                                             \
        return ComputeKernel<TYPE, int32_t, int64_t>(CTX); \
      }                                                    \
    } else {                                               \
      if ((DTYPE_2) == DT_INT32) {                         \
        return ComputeKernel<TYPE, int64_t, int32_t>(CTX); \
      } else {                                             \
        return ComputeKernel<TYPE, int64_t, int64_t>(CTX); \
      }                                                    \
    }                                                      \
    break;
}  // namespace aicpu

namespace aicpu {
uint32_t SparseSegmentSqrtNCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "SparseSegmentSqrtN normalcheck failed.");
  Tensor *x = ctx.Input(0);
  Tensor *indices = ctx.Input(1);
  Tensor *segment_ids = ctx.Input(2);

  auto x_shape = x->GetTensorShape();
  auto indices_shape = indices->GetTensorShape();
  auto segment_ids_shape = segment_ids->GetTensorShape();

  if (x_shape->GetDims() < 1) {
    KERNEL_LOG_ERROR("[%s] Tensor input0's rank less than 1.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (indices_shape->NumElements() != segment_ids_shape->NumElements()) {
    KERNEL_LOG_ERROR("[%s] Tensor input1&input2's ranks mismatch.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto x_data_type = x->GetDataType();
  auto indices_data_type = indices->GetDataType();
  auto segment_ids_data_type = segment_ids->GetDataType();

  if (x_data_type != DT_FLOAT && x_data_type != DT_DOUBLE && x_data_type != DT_FLOAT16) {
    KERNEL_LOG_ERROR("SparseSegmentSqrtN kernel data type [%s] not support.", DTypeStr(x_data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if ((indices_data_type != DT_INT32 && indices_data_type != DT_INT64) ||
      (segment_ids_data_type != DT_INT32 && segment_ids_data_type != DT_INT64)) {
    KERNEL_LOG_ERROR("SparseSegmentSqrtN kernel data type [%s] not support.", DTypeStr(indices_data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  switch (x_data_type) {
    COMPUTE_CASE(DT_FLOAT16, Eigen::half, indices_data_type, segment_ids_data_type, ctx)
    COMPUTE_CASE(DT_FLOAT, float, indices_data_type, segment_ids_data_type, ctx)
    COMPUTE_CASE(DT_DOUBLE, double, indices_data_type, segment_ids_data_type, ctx)
    default:
      KERNEL_LOG_ERROR("SparseSegmentSqrtN kernel data type [%s] not support.", DTypeStr(x_data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

namespace {
template <typename T1, typename T2>
uint32_t CheckParamValidation(const CpuKernelContext &ctx) {
  size_t m = ctx.Input(2)->GetTensorShape()->NumElements();
  auto indices_addr = reinterpret_cast<T1 *>(ctx.Input(1)->GetData());
  auto segment_ids_addr = reinterpret_cast<T2 *>(ctx.Input(2)->GetData());
  if (segment_ids_addr[0] != 0) {
    KERNEL_LOG_ERROR("segment_ids can't miss ids.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (size_t i = 1; i < m; i++) {
    if (segment_ids_addr[i] < segment_ids_addr[i - 1]) {
      KERNEL_LOG_ERROR("segment_ids should be sorted.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    if (segment_ids_addr[i] - segment_ids_addr[i - 1] > 1) {
      KERNEL_LOG_ERROR("segment_ids can't miss ids.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  for (size_t i = 0; i < m; i++) {
    if (indices_addr[i] >= ctx.Input(0)->GetTensorShape()->GetDimSize(0)) {
      KERNEL_LOG_ERROR("indices out of range.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}
}  // namespace

template <typename T1, typename T2, typename T3>
uint32_t SparseSegmentSqrtNCpuKernel::ComputeKernel(const CpuKernelContext &ctx) {
  size_t n = ctx.Input(0)->GetTensorShape()->NumElements() / ctx.Input(0)->GetTensorShape()->GetDimSize(0);
  size_t m = ctx.Input(2)->GetTensorShape()->NumElements();
  size_t k = ctx.Output(0)->GetTensorShape()->NumElements();
  auto x_addr = reinterpret_cast<T1 *>(ctx.Input(0)->GetData());
  auto indices_addr = reinterpret_cast<T2 *>(ctx.Input(1)->GetData());
  auto segment_ids_addr = reinterpret_cast<T3 *>(ctx.Input(2)->GetData());
  auto y_addr = reinterpret_cast<T1 *>(ctx.Output(0)->GetData());
  std::vector<int64_t> x_shape_list = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  x_shape_list[0] = segment_ids_addr[m - 1] + 1;
  ctx.Output(0)->GetTensorShape()->SetDimSizes(x_shape_list);
  for (size_t i = 0; i < k; i++) {
    y_addr[i] = (T1)0;
  }
  auto ret = CheckParamValidation<T2, T3>(ctx);
  if (ret != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int oldindex = -1;
  int countnum = 0;
  for (size_t i = 0; i < m; i++) {
    if (oldindex == segment_ids_addr[i]) {
      countnum++;
    } else if (countnum != 0) {
      for (size_t j = 0; j < n; j++) {
        y_addr[j + oldindex * n] /= (T1)(sqrt(countnum));
      }
      countnum = 1;
      oldindex = segment_ids_addr[i];
    } else {
      countnum = 1;
      oldindex = segment_ids_addr[i];
    }
    for (size_t j = 0; j < n; j++) {
      y_addr[j + oldindex * n] += x_addr[j + indices_addr[i] * n];
    }
  }
  if (countnum != 0) {
    for (size_t j = 0; j < n; j++) {
      y_addr[j + oldindex * n] /= (T1)(sqrt(countnum));
    }
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(SparseSegmentSqrtN, SparseSegmentSqrtNCpuKernel);
}  // namespace aicpu
