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
#include "sparse_segment_sqrt_n_grad.h"

#include "Eigen/Core"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 4;
const uint32_t kOutputNum = 1;
const char *SparseSegmentSqrtNGrad = "SparseSegmentSqrtNGrad";
}  // namespace

namespace aicpu {
uint32_t SparseSegmentSqrtNGradCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "SparseSegmentSqrtNGrad check input and output number failed.");
  Tensor *inputx = ctx.Input(0);
  Tensor *input_indices = ctx.Input(1);
  Tensor *input_segment_ids = ctx.Input(2);
  Tensor *input_output_dim = ctx.Input(3);

  auto data_type0 = inputx->GetDataType();
  auto data_type1 = input_indices->GetDataType();
  auto data_type2 = input_segment_ids->GetDataType();
  auto data_type3 = input_output_dim->GetDataType();

  if (data_type0 != DT_FLOAT && data_type0 != DT_DOUBLE && data_type0 != DT_FLOAT16) {
    KERNEL_LOG_ERROR("SparseSegmentSqrtNGrad kernel data type [%u] not support.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (data_type1 != data_type2 || data_type1 != data_type3 || data_type1 != DT_INT32) {
    KERNEL_LOG_ERROR("SparseSegmentSqrtNGrad kernel data type [%u] not support.", data_type1);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto shape0 = inputx->GetTensorShape();
  auto shape1 = input_indices->GetTensorShape();
  auto shape2 = input_segment_ids->GetTensorShape();
  auto scalarshape = input_output_dim->GetTensorShape();
  if (shape0->GetDims() < 1) {
    KERNEL_LOG_ERROR("[%s] Tensor input0's rank less than 1.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (shape1->NumElements() != shape2->NumElements()) {
    KERNEL_LOG_ERROR("[%s] Tensor input1&input2's ranks mismatch.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (data_type0 == DT_FLOAT) {
    return ComputeKernal<float>(ctx);
  } else if (data_type0 == DT_DOUBLE) {
    return ComputeKernal<double>(ctx);
  } else {
    return ComputeKernal<Eigen::half>(ctx);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SparseSegmentSqrtNGradCpuKernel::ComputeKernal(CpuKernelContext &ctx) {
  size_t n = ctx.Input(0)->GetTensorShape()->NumElements() / ctx.Input(0)->GetTensorShape()->GetDimSize(0);
  size_t m = ctx.Input(2)->GetTensorShape()->NumElements();
  int l = ctx.Output(0)->GetTensorShape()->GetDimSize(0);
  auto x_addr = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto indices_addr = reinterpret_cast<int32_t *>(ctx.Input(1)->GetData());
  auto segment_ids_addr = reinterpret_cast<int32_t *>(ctx.Input(2)->GetData());
  int k = *reinterpret_cast<int32_t *>(ctx.Input(3)->GetData());
  auto y_addr = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  std::vector<int64_t> y_shape_values = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  y_shape_values[0] = k;
  ctx.Output(0)->GetTensorShape()->SetDimSizes(y_shape_values);

  const size_t tensor_dim = 2;
  Eigen::TensorMap<Eigen::Tensor<T, tensor_dim>, Eigen::Aligned> res_map(y_addr, l, n);
  res_map.setZero();

  for (size_t i = 1; i < m; i++) {
    if (segment_ids_addr[i] < segment_ids_addr[i - 1]) {
      KERNEL_LOG_ERROR("Segment_ids should be sorted.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  for (size_t i = 0; i < m; i++) {
    if (indices_addr[i] >= ctx.Input(0)->GetTensorShape()->GetDimSize(0)) {
      KERNEL_LOG_ERROR("Indices out of range.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    if (segment_ids_addr[i] >= k) {
      KERNEL_LOG_ERROR("Segment_ids out of range.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  int beginindex = segment_ids_addr[0];
  size_t countnum = 1;
  for (size_t i = 1; i < m; i++) {
    if (segment_ids_addr[i] == beginindex) {
      countnum++;
      continue;
    }
    for (size_t j = 1; j <= countnum; j++) {
      for (size_t l = 0; l < n; l++) {
        y_addr[indices_addr[i - j] * n + l] += x_addr[beginindex * n + l] / (T)(sqrt(countnum));
      }
      beginindex = segment_ids_addr[i];
      countnum = 1;
    }
  }

  int i = m;
  for (size_t j = 1; j <= countnum; j++) {
    for (size_t l = 0; l < n; l++) {
      y_addr[indices_addr[i - j] * n + l] += x_addr[beginindex * n + l] / (T)(sqrt(countnum));
    }
  }
  return KERNEL_STATUS_OK;
};

REGISTER_CPU_KERNEL(SparseSegmentSqrtNGrad, SparseSegmentSqrtNGradCpuKernel);
}  // namespace aicpu
