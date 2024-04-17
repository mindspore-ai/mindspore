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

#include <cstdint>
#include <string.h>
#include "Eigen/Dense"
#include "context/inc/cpu_kernel_utils.h"
#include "securec.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kEye = "Eye";
constexpr size_t kValue2 = 2;
#define EYE_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                               \
    uint32_t result = EyePartCompute<TYPE>(CTX);                \
    if (result != KERNEL_STATUS_OK) {                           \
      CUST_KERNEL_LOG_ERROR(ctx, "Eye kernel compute failed."); \
      return result;                                            \
    }                                                           \
    break;                                                      \
  }
}  // namespace

namespace aicpu {
uint32_t EyeCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *output = ctx.Output(0);
  CUST_KERNEL_CHECK_NULLPTR(ctx, output, KERNEL_STATUS_PARAM_INVALID, "Get output failed")
  auto data_type = ctx.Output(0)->GetDataType();
  switch (data_type) {
    EYE_COMPUTE_CASE(DT_BOOL, bool, ctx)
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
      CUST_KERNEL_LOG_ERROR(ctx, "Eye kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t EyeCpuKernel::EyePartCompute(CpuKernelContext &ctx) {
  auto out_shape = ctx.Output(0)->GetTensorShape()->GetDimSizes();
  size_t out_dims = out_shape.size();
  int64_t dim_value = 1;
  for (size_t i = 0; i < out_dims - kValue2; ++i) {
    dim_value *= out_shape[i];
  }
  int64_t num_rows = out_shape[out_dims - 2];
  int64_t num_cols = out_shape[out_dims - 1];
  int64_t min_value = num_rows > num_cols ? num_cols : num_rows;

  Tensor *y = ctx.Output(0);
  auto output_size = y->GetDataSize();
  auto y_addr = reinterpret_cast<char *>(y->GetData());
  while (output_size > 0) {
    auto copy_size = std::min(output_size, static_cast<uint64_t>(INT32_MAX));
    auto ret = memset_s(y_addr, output_size, 0, copy_size);
    if (ret != EOK) {
      CUST_KERNEL_LOG_ERROR(ctx, "For 'Eye', memset_s failed, ret=%d.", ret);
      return KERNEL_STATUS_INNER_ERROR;
    }
    output_size -= copy_size;
    y_addr += copy_size;
  }

  CUST_KERNEL_CHECK_NULLPTR(ctx, y->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed.")
  auto output_y = reinterpret_cast<T *>(y->GetData());
  T num = static_cast<T>(1);
  int64_t inner_size = num_rows * num_cols;
  for (int64_t dim = 0; dim < dim_value; dim++) {
    for (int64_t i = 0; i < min_value; i++) {
      *(output_y + (dim * inner_size) + (num_cols + 1) * i) = num;
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kEye, EyeCpuKernel);
}  // namespace aicpu