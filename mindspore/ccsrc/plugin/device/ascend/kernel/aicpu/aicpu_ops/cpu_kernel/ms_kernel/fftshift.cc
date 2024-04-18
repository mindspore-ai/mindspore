/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "cpu_kernel/ms_kernel/fftshift.h"
#include <securec.h>
#include "unsupported/Eigen/CXX11/Tensor"
#include "context/inc/cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "base/bfloat16.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const uint32_t kIndex0 = 0;
const uint32_t kDimIndex = 1;
const char *kFFTShift = "FFTShift";
const char *kIFFTShift = "IFFTShift";

#define FFTSHIFT_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                                    \
    uint32_t result = FFTShiftCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                                \
      CUST_KERNEL_LOG_ERROR(ctx, "FFTShift kernel compute failed."); \
      return result;                                                 \
    }                                                                \
    break;                                                           \
  }
}  // namespace

namespace aicpu {
uint32_t FFTShiftCpuKernel::Compute(CpuKernelContext &ctx) {
  op_name = ctx.GetOpType();
  op_name.erase(op_name.begin(), op_name.begin() + std::size("Cust") - 1);
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.",
                           op_name.c_str());
  auto x_type = ctx.Input(kIndex0)->GetDataType();
  switch (x_type) {
    FFTSHIFT_COMPUTE_CASE(DT_BOOL, bool, ctx)
    FFTSHIFT_COMPUTE_CASE(DT_BFLOAT16, bfloat16, ctx)
    FFTSHIFT_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    FFTSHIFT_COMPUTE_CASE(DT_FLOAT, float, ctx)
    FFTSHIFT_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    FFTSHIFT_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    FFTSHIFT_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    FFTSHIFT_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    FFTSHIFT_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    FFTSHIFT_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    FFTSHIFT_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    FFTSHIFT_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    FFTSHIFT_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    FFTSHIFT_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    FFTSHIFT_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "[%s] kernel data type [%s] not support.", op_name.c_str(), DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t FFTShiftCpuKernel::FFTShiftCompute(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(kIndex0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(kIndex0)->GetData());
  int64_t x_rank = ctx.Input(kIndex0)->GetTensorShape()->GetDims();
  // step 1: No need to process when input is empty tensor.
  if (x_rank == 0) {
    output_y[0] = input_x[0];
    return KERNEL_STATUS_OK;
  }

  // step 2: Get or set attribute dim
  std::vector<int64_t> dim;
  if (ctx.Input(kDimIndex) == nullptr) {
    for (int64_t i = 0; i < x_rank; ++i) {
      (void)dim.emplace_back(i);
    }
  } else {
    const std::map<std::string, DataType> types = {{"dim", ctx.Input(kDimIndex)->GetDataType()}};
    CUST_KERNEL_HANDLE_ERROR(ctx, CheckTensorTypeSame(ctx, types, DT_INT64, op_name),
                             "[%s] check dim data type failed.", op_name.c_str());
    int64_t *dim_ptr = reinterpret_cast<int64_t *>(ctx.Input(kDimIndex)->GetData());
    int64_t dim_size = ctx.Input(kDimIndex)->NumElements();
    for (int64_t i = 0; i < dim_size; ++i) {
      int64_t tmp_pos = dim_ptr[i] < 0 ? x_rank + dim_ptr[i] : dim_ptr[i];
      (void)dim.emplace_back(tmp_pos);
    }
  }
  bool forward = (op_name == kFFTShift);

  std::vector<int64_t> tensor_shape = ctx.Input(kIndex0)->GetTensorShape()->GetDimSizes();
  std::int64_t element_nums{ctx.Input(kIndex0)->NumElements()};

  // step 3: Calculate the offset of input[i]
  std::vector<int64_t> offsets(element_nums, 0);
  for (size_t j = 0; j < dim.size(); j++) {
    int64_t size_j = tensor_shape[dim[j]];
    int64_t size_back =
      std::accumulate(tensor_shape.begin() + dim[j] + 1, tensor_shape.end(), 1, std::multiplies<int64_t>());
    int64_t size_tmp1 = size_j * size_back;
    int64_t size_tmp2 = size_j / 2 * size_back;

    for (int64_t i = 0; i < element_nums; i++) {
      if (forward) {
        if ((i + offsets[i]) % size_tmp1 >= size_tmp1 - size_tmp2) {
          offsets[i] -= size_tmp1 - size_tmp2;
        } else {
          offsets[i] += size_tmp2;
        }
      } else {
        if ((i + offsets[i]) % size_tmp1 < size_tmp2) {
          offsets[i] += size_tmp1 - size_tmp2;
        } else {
          offsets[i] -= size_tmp2;
        }
      }
    }
  }

  // step 4: update output according to offset
  for (int64_t i = 0; i < element_nums; i++) {
    output_y[i + offsets[i]] = input_x[i];
  }

  return KERNEL_STATUS_OK;
};

REGISTER_MS_CPU_KERNEL(kFFTShift, FFTShiftCpuKernel);
REGISTER_MS_CPU_KERNEL(kIFFTShift, FFTShiftCpuKernel);
}  // namespace aicpu
