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

#include "scatter_nd_update.h"

#include <string.h>

#include <algorithm>
#include <complex>
#include <iostream>
#include <map>

#include "eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 3;
const uint32_t kOutputNum = 1;
const char *kScatterNdUpdate = "ScatterNdUpdate";
}  // namespace

namespace aicpu {
uint32_t ScatterNdUpdateCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Check ScatterNdUpdate Input and Output failed.");

  Tensor *input_var = ctx.Input(0);
  Tensor *input_indices = ctx.Input(1);
  Tensor *input_updates = ctx.Input(2);

  auto shape_var = input_var->GetTensorShape();
  auto shape_indices = input_indices->GetTensorShape();
  auto shape_updates = input_updates->GetTensorShape();

  if (shape_var->GetDims() < 1) {
    KERNEL_LOG_ERROR("[%s] Tensor input_var's rank less than 1.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (shape_indices->GetDims() < 2) {
    KERNEL_LOG_ERROR("[%s] Tensor input_indices's rank less than 2.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (shape_updates->GetDims() < 1) {
    KERNEL_LOG_ERROR("[%s] Tensor input_updates's rank less than 1.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto index_size = shape_indices->GetDims() - 1;
  auto index_depth = shape_indices->GetDimSize(index_size);

  if (index_depth > shape_var->GetDims()) {
    KERNEL_LOG_ERROR("[%s] Tensor input_var&input_indices ranks mismatch.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  std::vector<int64_t> batch_shape;
  for (int64_t i = 0; i < index_size; ++i) {
    batch_shape.push_back(shape_indices->GetDimSize(i));
  }

  for (int64_t i = index_depth; i <= shape_var->GetDims() - 1; ++i) {
    batch_shape.push_back(shape_var->GetDimSize(i));
  }

  if (batch_shape != shape_updates->GetDimSizes()) {
    KERNEL_LOG_ERROR("[%s] Tensor indices's & updates' and var's shape are dismatch .", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  for (int64_t i = 0; i < index_size; i++) {
    if (shape_indices->GetDimSize(i) != shape_updates->GetDimSize(i)) {
      KERNEL_LOG_ERROR("[%s], Tensor indices and updates should have the same batch number.", ctx.GetOpType().c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  auto data_type_var = input_var->GetDataType();
  auto data_type_indices = input_indices->GetDataType();

  if (data_type_indices != DT_INT32 && data_type_indices != DT_INT64) {
    KERNEL_LOG_ERROR("ScatterNdUpdate kernel data type [%s] not support.", DTypeStr(data_type_indices).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  switch (data_type_var) {
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
      KERNEL_LOG_ERROR("ScatterNdUpdate kernel data type [%s] not support.", DTypeStr(data_type_var).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

template <typename var_type>
uint32_t ScatterNdUpdateCpuKernel::DTYPE_CHOOSE(CpuKernelContext &ctx) {
  auto indices_type = static_cast<DataType>(ctx.Input(1)->GetDataType());
  switch (indices_type) {
    case DT_INT32:
      return ScatterNdUpdateComputeRealKernel<var_type, int32_t>(ctx);
    case DT_INT64:
      return ScatterNdUpdateComputeRealKernel<var_type, int64_t>(ctx);
    default:
      KERNEL_LOG_ERROR("[%s] Data type of input is not supported, input data type is [%s].", ctx.GetOpType().c_str(),
                       DTypeStr(indices_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename var_type, typename indices_type>
uint32_t ScatterNdUpdateCpuKernel::ScatterNdUpdateComputeRealKernel(CpuKernelContext &ctx) {
  int64_t n_slices = 1;
  int64_t slice_size = 1;

  const int64_t indices_dims = ctx.Input(1)->GetTensorShape()->GetDims() - 1;
  const int64_t indices_nd = ctx.Input(1)->GetTensorShape()->GetDimSize(indices_dims);
  const int64_t updates_dims = ctx.Input(2)->GetTensorShape()->GetDims();

  auto shape_var = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  auto shape_indices = ctx.Input(1)->GetTensorShape();
  auto dims_shape = ctx.Input(0)->GetTensorShape()->GetDims();
  for (int64_t i = 0; i < dims_shape - indices_nd; i++) {
    if (ctx.Input(2)->GetTensorShape()->GetDimSize(i + shape_indices->GetDims() - 1) != shape_var[i + indices_nd]) {
      KERNEL_LOG_ERROR("[%s] shape_indices and shape_updates mismatch.", ctx.GetOpType().c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  for (int64_t i = 0; i < indices_dims; ++i) {
    n_slices *= ctx.Input(1)->GetTensorShape()->GetDimSize(i);
  }
  for (int i = indices_dims; i < updates_dims; ++i) {
    slice_size *= ctx.Input(2)->GetTensorShape()->GetDimSize(i);
  }

  const int64_t var_flat_size = ctx.Input(0)->GetTensorShape()->NumElements();
  std::vector<int64_t> output_shape = ctx.Input(0)->GetTensorShape()->GetDimSizes();

  int64_t remain_flat_size = var_flat_size;
  std::vector<int64_t> dims_to_count(indices_nd, 0);
  for (int64_t i = 0; i < indices_nd; ++i) {
    dims_to_count[i] = remain_flat_size / output_shape[i];
    remain_flat_size = dims_to_count[i];
  }

  auto Var_data = reinterpret_cast<var_type *>(ctx.Input(0)->GetData());
  auto Indices_data = reinterpret_cast<indices_type *>(ctx.Input(1)->GetData());
  auto Updates_data = reinterpret_cast<var_type *>(ctx.Input(2)->GetData());
  auto Output_data = reinterpret_cast<var_type *>(ctx.Output(0)->GetData());

  for (int64_t i = 0; i < var_flat_size; ++i) {
    Output_data[i] = Var_data[i];
  }
  for (int64_t i = 0; i < n_slices; ++i) {
    int64_t to_pos = 0;
    for (int64_t j = 0; j < indices_nd; ++j) {
      int64_t idx = Indices_data[i * indices_nd + j];

      if (idx < 0 || idx >= output_shape[j]) {
        KERNEL_LOG_ERROR("The indices[%d] is so big or small", idx);
        return KERNEL_STATUS_PARAM_INVALID;
      }

      to_pos += idx * dims_to_count[j];
    }
    for (int64_t j = 0; j < slice_size; j++) {
      Output_data[to_pos + j] = Updates_data[i * slice_size + j];
    }
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kScatterNdUpdate, ScatterNdUpdateCpuKernel);
}  // namespace aicpu
