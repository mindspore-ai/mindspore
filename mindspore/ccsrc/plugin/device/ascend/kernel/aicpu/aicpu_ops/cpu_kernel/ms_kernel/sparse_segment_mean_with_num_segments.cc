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
 */
#include "sparse_segment_mean_with_num_segments.h"

#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 4;
const uint32_t kOutputNum = 1;
const char *SparseSegmentMeanWithNumSegments = "SparseSegmentMeanWithNumSegments";

#define COMPUTE_CASE(DTYPE, TYPE, DTYPE_1, DTYPE_2, DTYPE_3, CTX)   \
  case (DTYPE):                                                     \
    if ((DTYPE_1) == DT_INT32) {                                    \
      if ((DTYPE_2) == DT_INT32 && (DTYPE_3) == DT_INT32) {         \
        return ComputeKernel<TYPE, int32_t, int32_t, int32_t>(CTX); \
      } else if ((DTYPE_2) == DT_INT32 && (DTYPE_3) != DT_INT32) {  \
        return ComputeKernel<TYPE, int32_t, int32_t, int64_t>(CTX); \
      } else if ((DTYPE_2) != DT_INT32 && (DTYPE_3) == DT_INT32) {  \
        return ComputeKernel<TYPE, int32_t, int64_t, int32_t>(CTX); \
      } else if ((DTYPE_2) != DT_INT32 && (DTYPE_3) != DT_INT32) {  \
        return ComputeKernel<TYPE, int32_t, int64_t, int64_t>(CTX); \
      }                                                             \
    } else {                                                        \
      if ((DTYPE_2) == DT_INT32 && (DTYPE_3) == DT_INT32) {         \
        return ComputeKernel<TYPE, int64_t, int32_t, int32_t>(CTX); \
      } else if ((DTYPE_2) == DT_INT32 && (DTYPE_3) != DT_INT32) {  \
        return ComputeKernel<TYPE, int64_t, int32_t, int64_t>(CTX); \
      } else if ((DTYPE_2) != DT_INT32 && (DTYPE_3) == DT_INT32) {  \
        return ComputeKernel<TYPE, int64_t, int64_t, int32_t>(CTX); \
      } else if ((DTYPE_2) != DT_INT32 && (DTYPE_3) != DT_INT32) {  \
        return ComputeKernel<TYPE, int64_t, int64_t, int64_t>(CTX); \
      }                                                             \
    }                                                               \
    break;
}  // namespace

namespace aicpu {
uint32_t SparseSegmentMeanWithNumSegmentsCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "SparseSegmentMeanWithNumSegments normalcheck failed.");
  Tensor *x = ctx.Input(0);
  Tensor *indices = ctx.Input(1);
  Tensor *segment_ids = ctx.Input(2);
  Tensor *num_segments = ctx.Input(3);

  if (x->GetDataSize() == 0 || indices->GetDataSize() == 0 || segment_ids->GetDataSize() == 0 ||
      num_segments->GetDataSize() == 0) {
    KERNEL_LOG_ERROR("[%s] Input is empty tensor.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto x_shape = x->GetTensorShape();
  auto indices_shape = indices->GetTensorShape();
  auto segment_ids_shape = segment_ids->GetTensorShape();
  auto num_segments_shape = num_segments->GetTensorShape();

  if (x_shape->GetDims() < 1) {
    KERNEL_LOG_ERROR("[%s] Tensor x's rank less than 1.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (indices_shape->NumElements() != segment_ids_shape->NumElements()) {
    KERNEL_LOG_ERROR("[%s] Tensor indices&segment_ids's ranks mismatch.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto x_data_type = x->GetDataType();
  auto indices_data_type = indices->GetDataType();
  auto segment_ids_data_type = segment_ids->GetDataType();
  auto num_segments_data_type = num_segments->GetDataType();

  if (indices_data_type != DT_INT32 && indices_data_type != DT_INT64) {
    KERNEL_LOG_ERROR("SparseSegmentMeanWithNumSegments kernel data type [%s] not support.",
                     DTypeStr(indices_data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (segment_ids_data_type != DT_INT32 && segment_ids_data_type != DT_INT64) {
    KERNEL_LOG_ERROR("SparseSegmentMeanWithNumSegments kernel data type [%s] not support.",
                     DTypeStr(segment_ids_data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (num_segments_data_type != DT_INT32 && num_segments_data_type != DT_INT64) {
    KERNEL_LOG_ERROR("SparseSegmentMeanWithNumSegments kernel data type [%s] not support.",
                     DTypeStr(num_segments_data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  switch (x_data_type) {
    COMPUTE_CASE(DT_FLOAT16, Eigen::half, indices_data_type, segment_ids_data_type, num_segments_data_type, ctx)
    COMPUTE_CASE(DT_FLOAT, float, indices_data_type, segment_ids_data_type, num_segments_data_type, ctx)
    COMPUTE_CASE(DT_DOUBLE, double, indices_data_type, segment_ids_data_type, num_segments_data_type, ctx)
    default:
      KERNEL_LOG_ERROR("SparseSegmentMeanWithNumSegments kernel data type [%s] not support.",
                       DTypeStr(x_data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(SparseSegmentMeanWithNumSegments, SparseSegmentMeanWithNumSegmentsCpuKernel);

template <typename T1, typename T2, typename T3, typename T4>
uint32_t SparseSegmentMeanWithNumSegmentsCpuKernel::ComputeKernel(CpuKernelContext &ctx) {
  int n = ctx.Input(0)->GetTensorShape()->NumElements() / ctx.Input(0)->GetTensorShape()->GetDimSize(0);
  int m = ctx.Input(2)->GetTensorShape()->NumElements();
  auto x_ptr = reinterpret_cast<T1 *>(ctx.Input(0)->GetData());
  auto indices_ptr = reinterpret_cast<T2 *>(ctx.Input(1)->GetData());
  auto segment_ids_ptr = reinterpret_cast<T3 *>(ctx.Input(2)->GetData());
  auto num_segments_ptr = reinterpret_cast<T4 *>(ctx.Input(3)->GetData());
  auto y_ptr = reinterpret_cast<T1 *>(ctx.Output(0)->GetData());

  std::vector<int64_t> y_shape_values = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  y_shape_values[0] = num_segments_ptr[0];
  ctx.Output(0)->GetTensorShape()->SetDimSizes(y_shape_values);

  for (int64_t i = 1; i < m; i++) {
    if (segment_ids_ptr[i] < segment_ids_ptr[i - 1]) {
      KERNEL_LOG_ERROR("segment_ids should be sorted.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  for (int64_t i = 0; i < m; i++) {
    if (indices_ptr[i] >= ctx.Input(0)->GetTensorShape()->GetDimSize(0)) {
      KERNEL_LOG_ERROR("indices out of range.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    if (segment_ids_ptr[i] >= num_segments_ptr[0]) {
      KERNEL_LOG_ERROR("segment_ids out of range.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  for (int64_t i = 0; i < ctx.Output(0)->GetTensorShape()->NumElements(); i++) {
    y_ptr[i] = (T1)0;
  }

  int oldindex = -1;
  int countnum = 0;
  for (int64_t i = 0; i < m; i++) {
    if (oldindex == segment_ids_ptr[i]) {
      countnum++;
    } else if (countnum != 0) {
      for (int64_t j = 0; j < n; j++) {
        y_ptr[j + oldindex * n] /= static_cast<T1>(countnum);
      }
      countnum = 1;
      oldindex = segment_ids_ptr[i];
    } else {
      countnum = 1;
      oldindex = segment_ids_ptr[i];
    }
    for (int64_t j = 0; j < n; j++) {
      y_ptr[j + oldindex * n] += x_ptr[j + indices_ptr[i] * n];
    }
  }
  if (countnum != 0) {
    for (int64_t j = 0; j < n; j++) {
      y_ptr[j + oldindex * n] /= static_cast<T1>(countnum);
    }
  }
  return KERNEL_STATUS_OK;
};
}  // namespace aicpu
