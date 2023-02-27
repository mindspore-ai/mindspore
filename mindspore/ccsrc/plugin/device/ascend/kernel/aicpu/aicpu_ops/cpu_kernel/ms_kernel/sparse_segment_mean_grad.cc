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
#include <algorithm>
#include <iostream>
#include <map>
#include "utils/kernel_util.h"
#include "utils/eigen_tensor.h"
#include "sparse_segment_mean_grad.h"
#include "cpu_kernel_utils.h"

namespace {
const uint32_t kInputNum = 4;
const uint32_t kOutputNum = 1;
const char *const SparseSegmentMeanGrad = "SparseSegmentMeanGrad";
const uint32_t dim_2 = 2;
}  // namespace

namespace aicpu {
KernelStatus SparseSegmentMeanGradCpuKernel::CheckDataPara(const CpuKernelContext &ctx) const {
  Tensor *inputx = ctx.Input(0);
  Tensor *input_indices = ctx.Input(1);
  Tensor *input_segment_ids = ctx.Input(2);
  Tensor *input_output_dim = ctx.Input(3);
  Tensor *output0 = ctx.Output(0);
  if (inputx->GetDataSize() == 0 || input_indices->GetDataSize() == 0 || input_segment_ids->GetDataSize() == 0 ||
      input_output_dim->GetDataSize() == 0) {
    KERNEL_LOG_ERROR("[%s] Input is empty tensor.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto data_type0 = inputx->GetDataType();
  auto data_type4 = output0->GetDataType();
  if (data_type0 != data_type4) {
    KERNEL_LOG_ERROR("[%s] Tensor data type mismatch.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

KernelStatus SparseSegmentMeanGradCpuKernel::CheckShapePara(const CpuKernelContext &ctx) const {
  auto shape0 = ctx.Input(0)->GetTensorShape();
  auto shape1 = ctx.Input(1)->GetTensorShape();
  auto shape2 = ctx.Input(2)->GetTensorShape();
  auto scalarshape = ctx.Input(3)->GetTensorShape();
  auto output_dim0 = reinterpret_cast<int32_t *>(ctx.Input(3)->GetData());
  auto shape3 = ctx.Output(0)->GetTensorShape();
  if (shape0->GetDims() < 1) {
    KERNEL_LOG_ERROR("[%s] Tensor input0's rank less than 1.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (scalarshape->GetDims() > 1) {  // Scalar input becomes 0-D Tensor in the dynamic shape scenerio, has dim 1
    KERNEL_LOG_ERROR("[%s] Tensor outputdim0 should be a scalar.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (shape3->GetDims() < 1) {
    KERNEL_LOG_ERROR("[%s] Tensor output0's rank less than 1.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (shape1->NumElements() != shape2->NumElements()) {
    KERNEL_LOG_ERROR("[%s] segment _ids and indices should have same size.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (shape0->NumElements() / shape0->GetDimSize(0) != shape3->NumElements() / shape3->GetDimSize(0)) {
    KERNEL_LOG_ERROR("[%s] Tensor input0&output0's ranks mismatch.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (shape3->GetDimSize(0) != *(output_dim0)) {
    KERNEL_LOG_ERROR("[%s] Tensor output0's dim(0) mismatch with output_dim0.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
uint32_t SparseSegmentMeanGradCpuKernel::Compute(CpuKernelContext &ctx) {
  if ((NormalCheck(ctx, kInputNum, kOutputNum) != KERNEL_STATUS_OK) || (CheckDataPara(ctx) != KERNEL_STATUS_OK) ||
      (CheckShapePara(ctx) != KERNEL_STATUS_OK)) {
    return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }

  KernelStatus result = KERNEL_STATUS_OK;
  auto data_type0 = ctx.Input(0)->GetDataType();
  switch (data_type0) {
    case (DT_FLOAT):
      result = ComputeKernel<float>(ctx);
      break;
    case (DT_DOUBLE):
      result = ComputeKernel<double>(ctx);
      break;
    case (DT_FLOAT16):
      result = ComputeKernel<Eigen::half>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("SparseSegmentMeanGrad kernel data type [%s] not support.", DTypeStr(data_type0).c_str());
      result = KERNEL_STATUS_PARAM_INVALID;
  }
  return static_cast<uint32_t>(result);
}

template <typename T, typename T1, typename T2>
KernelStatus SparseSegmentMeanGradCpuKernel::ComputeKernelWithType(const CpuKernelContext &ctx) {
  int64_t segment_ids_num = ctx.Input(1)->GetTensorShape()->NumElements();
  int64_t num_segments = ctx.Input(0)->GetTensorShape()->GetDimSize(0);
  int64_t column = ctx.Input(0)->GetTensorShape()->NumElements() / ctx.Input(0)->GetTensorShape()->GetDimSize(0);
  auto dataptr = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto indicesptr = reinterpret_cast<T1 *>(ctx.Input(1)->GetData());
  auto segment_idsptr = reinterpret_cast<T2 *>(ctx.Input(2)->GetData());
  int32_t output_dim0 = *reinterpret_cast<int32_t *>(ctx.Input(3)->GetData());
  auto resultptr = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  std::vector<double> segment(num_segments, 0.0);
  for (int64_t i = 0; i < segment_ids_num; i++) {
    T2 index = segment_idsptr[i];
    if ((index >= num_segments) || (index < 0)) {
      KERNEL_LOG_ERROR("Segment id %d out of range [0, %d).", index, num_segments);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    segment[static_cast<size_t>(index)]++;
  }

  for (int64_t i = 0; i < num_segments; i++) {
    segment[i] = 1.0 / std::max(segment[i], 1.0);
  }

  std::vector<bool> is_modified(output_dim0, false);
  Eigen::TensorMap<Eigen::Tensor<T, dim_2, Eigen::RowMajor>> input_flat(dataptr, num_segments, column);
  Eigen::TensorMap<Eigen::Tensor<T, dim_2, Eigen::RowMajor>> output_flat(resultptr, output_dim0, column);
  output_flat.setZero();

  for (int64_t i = 0; i < segment_ids_num; i++) {
    T1 output_idx = indicesptr[i];
    if ((output_idx >= output_dim0) || (output_idx < 0)) {
      KERNEL_LOG_ERROR("Index %lld out of range [0, %d).", output_idx, output_dim0);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    size_t idx = static_cast<size_t>(segment_idsptr[i]);
    const T scale = static_cast<T>(segment[idx]);
    if (is_modified[static_cast<size_t>(output_idx)]) {
      output_flat.template chip<0>(output_idx) += input_flat.template chip<0>(idx) * scale;
    } else {
      output_flat.template chip<0>(output_idx) = input_flat.template chip<0>(idx) * scale;
    }
    is_modified[static_cast<size_t>(output_idx)] = true;
  }
  return KERNEL_STATUS_OK;
};

template <typename T>
KernelStatus SparseSegmentMeanGradCpuKernel::ComputeKernel(const CpuKernelContext &ctx) {
  auto indices_data_type = ctx.Input(1)->GetDataType();
  auto segment_ids_dtype = ctx.Input(2)->GetDataType();
  if (indices_data_type == DT_INT32) {
    if (segment_ids_dtype == DT_INT32) {
      return ComputeKernelWithType<T, int32_t, int32_t>(ctx);
    } else if (segment_ids_dtype == DT_INT64) {
      return ComputeKernelWithType<T, int32_t, int64_t>(ctx);
    } else {
      KERNEL_LOG_ERROR("SparseSegmentMeanGrad kernel data type [%s] not support.", DTypeStr(indices_data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  } else if (indices_data_type == DT_INT64) {
    if (segment_ids_dtype == DT_INT32) {
      return ComputeKernelWithType<T, int64_t, int32_t>(ctx);
    } else if (segment_ids_dtype == DT_INT64) {
      return ComputeKernelWithType<T, int64_t, int64_t>(ctx);
    } else {
      KERNEL_LOG_ERROR("SparseSegmentMeanGrad kernel data type [%s] not support.", DTypeStr(indices_data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  } else {
    KERNEL_LOG_ERROR("SparseSegmentMeanGrad kernel data type [%s] not support.", DTypeStr(indices_data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
}

REGISTER_CPU_KERNEL(SparseSegmentMeanGrad, SparseSegmentMeanGradCpuKernel);
}  // namespace aicpu