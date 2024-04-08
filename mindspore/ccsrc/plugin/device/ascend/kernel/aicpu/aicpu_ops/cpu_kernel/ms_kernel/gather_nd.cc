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

#include "gather_nd.h"

#include <string.h>

#include <algorithm>
#include <complex>
#include <iostream>
#include <map>

#include "utils/eigen_tensor.h"
#include "securec.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;
const char *kGatherNd = "GatherNd";
}  // namespace

namespace aicpu {
uint32_t GatherNdCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "Check GatherNd Input and Output failed.");

  Tensor *input_x = ctx.Input(0);
  Tensor *input_indices = ctx.Input(1);

  auto shape_x = input_x->GetTensorShape();
  auto shape_indices = input_indices->GetTensorShape();
  auto indices_rank = shape_indices->GetDims();
  auto indices_nd = shape_indices->GetDimSize(indices_rank - 1);

  if (shape_x->GetDims() < 1) {
    CUST_KERNEL_LOG_ERROR(ctx, "[%s] Tensor input_x's rank is less than 1.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (indices_rank < 1) {
    CUST_KERNEL_LOG_ERROR(ctx, "[%s] Tensor input_indices's rank is less than 1.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (indices_nd > shape_x->GetDims()) {
    CUST_KERNEL_LOG_ERROR(ctx, "[%s] Slice's  length must be less than x rank. ", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto data_type0 = input_x->GetDataType();
  auto data_type1 = input_indices->GetDataType();

  if (data_type1 != DT_INT32 && data_type1 != DT_INT64) {
    CUST_KERNEL_LOG_ERROR(ctx, "GatherNd kernel data type [%s] not support.", DTypeStr(data_type1).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  switch (data_type0) {
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
      CUST_KERNEL_LOG_ERROR(ctx, "GatherNd kernel data type [%s] not support.", DTypeStr(data_type0).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename data_type>
uint32_t GatherNdCpuKernel::DTYPE_CHOOSE(CpuKernelContext &ctx) {
  auto indices_type = static_cast<DataType>(ctx.Input(1)->GetDataType());
  switch (indices_type) {
    case DT_INT32:
      return GatherNdComputeRealKernel<int32_t, data_type>(ctx);
    case DT_INT64:
      return GatherNdComputeRealKernel<int64_t, data_type>(ctx);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "[%s] Data type of input is not supported, input data type is [%s].",
                            ctx.GetOpType().c_str(), DTypeStr(indices_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename indices_type, typename data_type>
uint32_t GatherNdCpuKernel::GatherNdComputeRealKernel(CpuKernelContext &ctx) {
  auto x_shape = ctx.Input(0)->GetTensorShape();
  auto indices_shape = ctx.Input(1)->GetTensorShape();

  int64_t n_slices = 1;
  int64_t slice_size = 1;
  const int64_t indices_dims = indices_shape->GetDims();
  int64_t indices_nd = indices_shape->GetDimSize(indices_dims - 1);

  const int64_t params_dims = x_shape->GetDims();

  for (int64_t i = 0; i < indices_dims - 1; ++i) {
    n_slices *= indices_shape->GetDimSize(i);
  }
  for (int64_t i = indices_nd; i < params_dims; ++i) {
    slice_size *= x_shape->GetDimSize(i);
  }

  int64_t remain_flat_size = x_shape->NumElements();
  std::vector<int64_t> dims_to_count = std::vector<int64_t>(indices_nd, 0);
  for (int64_t i = 0; i < indices_nd; ++i) {
    dims_to_count[i] = remain_flat_size / x_shape->GetDimSize(i);
    remain_flat_size = dims_to_count[i];
  }

  auto indices_data = reinterpret_cast<indices_type *>(ctx.Input(1)->GetData());
  auto x_data = reinterpret_cast<data_type *>(ctx.Input(0)->GetData());
  auto output_data = reinterpret_cast<data_type *>(ctx.Output(0)->GetData());
  auto output_size = ctx.Output(0)->GetDataSize();

  for (int64_t i = 0; i < n_slices; ++i) {
    int64_t from_pos = 0;
    for (int64_t j = 0; j < indices_nd; ++j) {
      if (indices_data[i * indices_nd + j] < 0) {
        CUST_KERNEL_LOG_ERROR(ctx, "For 'GatherNd', indices can't contain negative value.");
        return KERNEL_STATUS_INNER_ERROR;
      }
      from_pos += indices_data[i * indices_nd + j] * dims_to_count[j];
    }
    auto offset = i * slice_size;
    auto ret = memcpy_s(output_data + offset, output_size - offset * sizeof(data_type), x_data + from_pos,
                        slice_size * sizeof(data_type));
    if (ret != EOK) {
      CUST_KERNEL_LOG_ERROR(ctx, "For 'GatherNd', memcpy_s failed, ret=%d.", ret);
      return KERNEL_STATUS_INNER_ERROR;
    }
  }

  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kGatherNd, GatherNdCpuKernel);

}  // namespace aicpu
