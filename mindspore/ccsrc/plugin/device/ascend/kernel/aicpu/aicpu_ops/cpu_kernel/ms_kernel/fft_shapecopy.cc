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

#include "cpu_kernel/ms_kernel/fft_shapecopy.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "context/inc/cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "cpu_kernel/utils/fft_helper.h"
#include "base/bfloat16.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const uint32_t kIndex0 = 0;
const uint32_t kFftShapeIndex = 1;
const char *kFFTShapeCopy = "FFTShapeCopy";

#define FFTSHAPECOPY_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                                        \
    uint32_t result = FFTShapeCopyCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                                    \
      CUST_KERNEL_LOG_ERROR(ctx, "FFTShapeCopy kernel compute failed."); \
      return result;                                                     \
    }                                                                    \
    break;                                                               \
  }
}  // namespace

namespace aicpu {
uint32_t FFTShapeCopyCpuKernel::Compute(CpuKernelContext &ctx) {
  op_name_ = GetOpName(ctx);
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.",
                           op_name_.c_str());
  auto x_type = ctx.Input(kIndex0)->GetDataType();
  switch (x_type) {
    FFTSHAPECOPY_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    FFTSHAPECOPY_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    FFTSHAPECOPY_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    FFTSHAPECOPY_COMPUTE_CASE(DT_BFLOAT16, bfloat16, ctx)
    FFTSHAPECOPY_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    FFTSHAPECOPY_COMPUTE_CASE(DT_FLOAT, float, ctx)
    FFTSHAPECOPY_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    FFTSHAPECOPY_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    FFTSHAPECOPY_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "[%s] kernel data type [%s] not support.", op_name_.c_str(), DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t FFTShapeCopyCpuKernel::FFTShapeCopyCompute(CpuKernelContext &ctx) {
  auto input_ptr = reinterpret_cast<T *>(ctx.Input(kIndex0)->GetData());
  auto output_ptr = reinterpret_cast<T *>(ctx.Output(kIndex0)->GetData());

  int64_t x_rank = ctx.Input(kIndex0)->GetTensorShape()->GetDims();
  std::vector<int64_t> dout_shape = ctx.Input(kIndex0)->GetTensorShape()->GetDimSizes();

  std::vector<int64_t> shape;
  auto shape_tensor = reinterpret_cast<int64_t *>(ctx.Input(kFftShapeIndex)->GetData());
  for (int64_t i = 0; i < x_rank; i++) {
    (void)shape.emplace_back(shape_tensor[i]);
  }

  auto output_nums = ctx.Output(kIndex0)->NumElements();
  auto ret = memset_s(output_ptr, output_nums * sizeof(T), 0, output_nums * sizeof(T));
  CUST_KERNEL_CHECK_FALSE(ctx, (ret == EOK), KERNEL_STATUS_PARAM_INVALID, "FFTShapeCopy memset_s failed.");

  ShapeCopy<T, T>(input_ptr, output_ptr, dout_shape, shape);
  return KERNEL_STATUS_OK;
};

REGISTER_MS_CPU_KERNEL(kFFTShapeCopy, FFTShapeCopyCpuKernel);
}  // namespace aicpu
