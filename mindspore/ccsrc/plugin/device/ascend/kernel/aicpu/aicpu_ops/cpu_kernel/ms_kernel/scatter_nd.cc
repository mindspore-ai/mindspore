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

#include "scatter_nd.h"

#include <complex>

#include "eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 3;
const uint32_t kOutputNum = 1;
const char *kScatterNd = "ScatterNd";
}  // namespace

namespace aicpu {
uint32_t ScatterNdCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Check ScatterNd Input and Output failed.");

  Tensor *input_indices = ctx.Input(0);
  Tensor *input_x = ctx.Input(1);
  Tensor *input_shape = ctx.Input(2);

  auto shape_x = input_x->GetTensorShape();
  auto shape_indices = input_indices->GetTensorShape();
  auto shape_shape = input_shape->GetTensorShape();
  int64_t indices_shape_m = shape_indices->GetDimSize(shape_indices->GetDims() - 1);

  if (shape_x->GetDims() < 1) {
    KERNEL_LOG_ERROR("[%s] Tensor input_x's rank less than 1.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (shape_indices->GetDims() < 1) {
    KERNEL_LOG_ERROR("[%s] Tensor input_indices's rank less than 1.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (shape_shape->GetDims() < 1) {
    KERNEL_LOG_ERROR("[%s] Tensor input_shape's rank less than 1.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (indices_shape_m > shape_shape->NumElements()) {
    KERNEL_LOG_ERROR("[%s] Tensor input_shape&input_indices ranks mismatch.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  for (int64_t i = 0; i < shape_indices->GetDims() - 1; i++) {
    if (shape_indices->GetDimSize(i) != shape_x->GetDimSize(i)) {
      KERNEL_LOG_ERROR("[%s], shape_indices and shape_updates mismatch.", ctx.GetOpType().c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  auto data_type_x = input_x->GetDataType();
  auto data_type_indices = input_indices->GetDataType();
  auto data_type_shape = input_shape->GetDataType();
  if (data_type_shape != DT_INT32 && data_type_shape != DT_INT64) {
    KERNEL_LOG_ERROR("ScatterNd kernel data type [%s] not support.", DTypeStr(data_type_shape).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (data_type_indices != DT_INT32 && data_type_indices != DT_INT64) {
    KERNEL_LOG_ERROR("ScatterNd kernel data type [%s] not support.", DTypeStr(data_type_indices).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (data_type_indices != data_type_shape) {
    KERNEL_LOG_ERROR("Indices and shape must have the same type.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  switch (data_type_x) {
    case DT_INT8:
      return DTYPE_CHOOSE<int8_t>(ctx);
    case DT_INT16:
      return DTYPE_CHOOSE<int16_t>(ctx);
    case DT_INT32:
      return DTYPE_CHOOSE<int32_t>(ctx);
    case DT_INT64:
      return DTYPE_CHOOSE<int64_t>(ctx);
    case DT_UINT8:
      return DTYPE_CHOOSE<uint8_t>(ctx);
    case DT_UINT16:
      return DTYPE_CHOOSE<uint16_t>(ctx);
    case DT_UINT32:
      return DTYPE_CHOOSE<uint32_t>(ctx);
    case DT_UINT64:
      return DTYPE_CHOOSE<uint64_t>(ctx);
    case DT_FLOAT16:
      return DTYPE_CHOOSE<Eigen::half>(ctx);
    case DT_FLOAT:
      return DTYPE_CHOOSE<float>(ctx);
    case DT_DOUBLE:
      return DTYPE_CHOOSE<double>(ctx);
    case DT_COMPLEX64:
      return DTYPE_CHOOSE<std::complex<float>>(ctx);
    case DT_COMPLEX128:
      return DTYPE_CHOOSE<std::complex<double>>(ctx);
    default:
      KERNEL_LOG_ERROR("ScatterNd kernel data type [%s] not support.", DTypeStr(data_type_x).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename data_type_x>
uint32_t ScatterNdCpuKernel::DTYPE_CHOOSE(CpuKernelContext &ctx) {
  auto indices_type = static_cast<DataType>(ctx.Input(0)->GetDataType());
  switch (indices_type) {
    case DT_INT32:
      return ScatterNdComputeRealKernel<int32_t, data_type_x>(ctx);
    case DT_INT64:
      return ScatterNdComputeRealKernel<int64_t, data_type_x>(ctx);
    default:
      KERNEL_LOG_ERROR("[%s] Data type of input is not supported, input data type is [%s].", ctx.GetOpType().c_str(),
                       DTypeStr(indices_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename indices_type, typename data_type_x>
uint32_t ScatterNdCpuKernel::ScatterNdComputeRealKernel(CpuKernelContext &ctx) {
  int64_t n_slices = 1;
  int64_t slice_size = 1;

  const int64_t outer_dims = ctx.Input(0)->GetTensorShape()->GetDims() - 1;
  const int64_t indices_nd = ctx.Input(0)->GetTensorShape()->GetDimSize(outer_dims);
  const int64_t updates_dims = ctx.Input(1)->GetTensorShape()->GetDims();

  auto shape_indices = ctx.Input(0)->GetTensorShape();
  auto data_shape = reinterpret_cast<indices_type *>(ctx.Input(2)->GetData());
  auto dims_shape = ctx.Input(2)->GetTensorShape()->NumElements();
  auto updates_shape = ctx.Input(1)->GetTensorShape();
  for (int64_t i = 0; i < dims_shape - indices_nd; i++) {
    if (updates_shape->GetDimSize(i + shape_indices->GetDims() - 1) != data_shape[i + indices_nd]) {
      KERNEL_LOG_ERROR("[%s], shape_indices and shape_updates mismatch.", ctx.GetOpType().c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  for (int64_t i = 0; i < outer_dims; ++i) {
    n_slices *= ctx.Input(0)->GetTensorShape()->GetDimSize(i);
  }
  for (int64_t i = outer_dims; i < updates_dims; ++i) {
    slice_size *= ctx.Input(1)->GetTensorShape()->GetDimSize(i);
  }
  const int kNumberInputTwo = 2;
  int64_t output_flat_size = 1;
  int64_t num_shape = ctx.Input(kNumberInputTwo)->NumElements();
  for (int64_t i = 0; i < num_shape; i++) {
    output_flat_size *= data_shape[i];
  }
  int64_t remain_flat_size = output_flat_size;
  std::vector<int64_t> dims_to_count(indices_nd, 0);
  for (int64_t i = 0; i < indices_nd; ++i) {
    dims_to_count[i] = remain_flat_size / data_shape[i];
    remain_flat_size = dims_to_count[i];
  }

  auto Indices_data = reinterpret_cast<indices_type *>(ctx.Input(0)->GetData());
  auto Updates_data = reinterpret_cast<data_type_x *>(ctx.Input(1)->GetData());
  auto Output_data = reinterpret_cast<data_type_x *>(ctx.Output(0)->GetData());

  memset(Output_data, 0, sizeof(data_type_x) * output_flat_size);
  for (int64_t i = 0; i < n_slices; ++i) {
    int64_t to_pos = 0;
    for (int64_t j = 0; j < indices_nd; ++j) {
      int64_t idx = Indices_data[i * indices_nd + j];

      if (idx < 0 || idx >= data_shape[j]) {
        KERNEL_LOG_ERROR("The indices[%d] is so big or small", idx);
        return KERNEL_STATUS_PARAM_INVALID;
      }

      to_pos += idx * dims_to_count[j];
    }
    for (int64_t j = 0; j < slice_size; j++) {
      Output_data[to_pos + j] += Updates_data[i * slice_size + j];
    }
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kScatterNd, ScatterNdCpuKernel);
}  // namespace aicpu