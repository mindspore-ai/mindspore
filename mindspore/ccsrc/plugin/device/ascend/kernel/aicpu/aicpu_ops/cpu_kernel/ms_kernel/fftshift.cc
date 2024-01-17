/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const uint32_t kIndex0 = 0;
const char *kFFTShift = "FFTShift";

#define FFTSHIFT_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                          \
    uint32_t result = FFTShiftCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                      \
      KERNEL_LOG_ERROR("FFTShift kernel compute failed."); \
      return result;                                       \
    }                                                      \
    break;                                                 \
  }
}  // namespace

namespace aicpu {
uint32_t FFTShiftCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kFFTShift);
  auto x_type = ctx.Input(0)->GetDataType();
  switch (x_type) {
    FFTSHIFT_COMPUTE_CASE(DT_BOOL, bool, ctx)
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
      KERNEL_LOG_ERROR("FFTShift kernel data type [%s] not support.", DTypeStr(x_type).c_str());
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

  // step 2: Get or set attribute axes
  auto axes = ctx.GetAttr("axes")->GetListInt();
  if (axes.size() == 0) {
    // Process all dimensions.
    axes.clear();
    for (int64_t i = 0; i < x_rank; ++i) {
      (void)axes.emplace_back(i);
    }
  } else {
    (void)std::for_each(axes.begin(), axes.end(), [x_rank](auto &axis) { axis = axis < 0 ? x_rank + axis : axis; });
  }

  std::vector<int64_t> tensor_shape = ctx.Input(kIndex0)->GetTensorShape()->GetDimSizes();
  std::int64_t element_nums{ctx.Input(kIndex0)->NumElements()};
  bool forward = ctx.GetAttr("forward")->GetBool();

  // step 3: Calculate the offset of input[i]
  std::vector<int64_t> offsets(element_nums, 0);
  for (size_t j = 0; j < axes.size(); j++) {
    int64_t size_j = tensor_shape[axes[j]];
    int64_t size_back =
      std::accumulate(tensor_shape.begin() + axes[j] + 1, tensor_shape.end(), 1, std::multiplies<int64_t>());
    int64_t size_tmp1 = size_j * size_back;
    int64_t size_tmp2 = size_j / 2 * size_back;

    for (int64_t i = 0; i < element_nums; i++) {
      if (forward == true) {
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

REGISTER_CPU_KERNEL(kFFTShift, FFTShiftCpuKernel);
}  // namespace aicpu
