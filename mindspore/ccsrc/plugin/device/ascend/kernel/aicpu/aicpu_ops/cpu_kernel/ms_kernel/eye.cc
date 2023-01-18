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
#include "eye.h"

#include <string.h>
#include "Eigen/Dense"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kEye = "Eye";

#define EYE_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                     \
    uint32_t result = EyePartCompute<TYPE>(CTX);      \
    if (result != KERNEL_STATUS_OK) {                 \
      KERNEL_LOG_ERROR("Eye kernel compute failed."); \
      return result;                                  \
    }                                                 \
    break;                                            \
  }
}  // namespace

namespace aicpu {
uint32_t EyeCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *output = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_PARAM_INVALID, "Get output failed")
  auto data_type = ctx.Output(0)->GetDataType();
  switch (data_type) {
    EYE_COMPUTE_CASE(DT_FLOAT, float, ctx)
    EYE_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    EYE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    EYE_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    EYE_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    EYE_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    EYE_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    EYE_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    EYE_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    EYE_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    EYE_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    EYE_COMPUTE_CASE(DT_COMPLEX64, std::complex<std::float_t>, ctx)
    EYE_COMPUTE_CASE(DT_COMPLEX128, std::complex<std::double_t>, ctx)
    default:
      KERNEL_LOG_ERROR("Eye kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t EyeCpuKernel::EyePartCompute(CpuKernelContext &ctx) {
  int64_t num_rows_value1 = 0;
  int64_t num_columns_value = -1;
  int64_t dim_value = 1;
  int32_t out_size_size = 0;
  AttrValue *num_rows = ctx.GetAttr("num_rows");
  KERNEL_CHECK_NULLPTR(num_rows, KERNEL_STATUS_PARAM_INVALID, "get num_rows failed.");
  num_rows_value1 = num_rows->GetInt();
  int64_t min_value = num_rows_value1;
  int64_t max_value = -1;
  int64_t num_col = num_rows_value1;
  AttrValue *num_columns = ctx.GetAttr("num_columns");
  if (num_columns) {
    num_columns_value = num_columns->GetInt();
    min_value = num_columns_value < num_rows_value1 ? num_columns_value : num_rows_value1;
    max_value = num_columns_value > num_rows_value1 ? num_columns_value : num_rows_value1;
    num_col = num_columns_value;
  }
  if (max_value == -1) {
    max_value = num_rows_value1;
  }
  AttrValue *batch_shape = ctx.GetAttr("batch_shape");
  if (batch_shape) {
    std::vector<int64_t> output_size = ctx.GetAttr("batch_shape")->GetListInt();
    out_size_size = output_size.size();
    int64_t batch_shape_value = 1;
    for (int32_t t = 0; t < out_size_size; t++) {
      batch_shape_value = output_size[t];
      dim_value = dim_value * batch_shape_value;
    }
  }
  KERNEL_CHECK_NULLPTR(ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed.")
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Output(0)->NumElements();
  int64_t data_size = data_num * sizeof(T);
  Tensor *y = ctx.Output(0);
  auto y_addr = y->GetData();
  memset(y_addr, 0.0, data_size);
  T num = static_cast<T>(1);
  int32_t block_size = min_value * max_value;
  for (int32_t dim = 0; dim < dim_value; dim++) {
    for (int32_t i = 0; i < min_value; i++) {
      *(output_y + (dim * block_size) + (num_col + 1) * i) = num;
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kEye, EyeCpuKernel);
}  // namespace aicpu