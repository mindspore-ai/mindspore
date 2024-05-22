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

#include "cpu_kernel/ms_kernel/fftfreq.h"
#include "common/kernel_errcode.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/kernel_util.h"
#include "context/inc/cpu_kernel_utils.h"
#include "cpu_kernel/utils/fft_helper.h"
#include "mindspore/core/mindapi/base/types.h"
#include "base/bfloat16.h"

namespace aicpu {
const char *kFFTFreq = "FFTFreq";
const char *kRFFTFreq = "RFFTFreq";

template <typename T>
uint32_t FFTFreqCompute(CpuKernelContext &ctx) {
  auto op_name = GetOpName(ctx);
  int64_t n = reinterpret_cast<int64_t *>(ctx.Input(kNIndex)->GetData())[0];
  double d = reinterpret_cast<float *>(ctx.Input(kDIndex)->GetData())[0];
  auto *output_y = reinterpret_cast<T *>(ctx.Output(kIndex0)->GetData());

  int64_t index = 0;
  int64_t mid = (n + 1) / 2;
  int64_t r = (n % 2) ? 1 : 0;
  double weight = 1.0 / (d * n);
  while (index < mid) {
    output_y[index] = static_cast<T>(index * weight);
    index++;
  }
  if (op_name == kRFFTFreq) {
    output_y[index] = static_cast<T>(index * weight);
  } else {
    int64_t k = 0;
    while (k < n - mid) {
      output_y[index + k] = static_cast<T>((-index + r + k) * weight);
      k++;
    }
  }
  return KERNEL_STATUS_OK;
};

uint32_t FFTFreqCpuKernel::Compute(CpuKernelContext &ctx) {
  auto op_name = GetOpName(ctx);
  auto output_type = ctx.Output(kIndex0)->GetDataType();
  switch (output_type) {
    case (DT_BFLOAT16):
      return FFTFreqCompute<bfloat16>(ctx);
    case (DT_FLOAT16):
      return FFTFreqCompute<Eigen::half>(ctx);
    case (DT_FLOAT):
      return FFTFreqCompute<float>(ctx);
    case (DT_DOUBLE):
      return FFTFreqCompute<double>(ctx);
    case (DT_COMPLEX64):
      return FFTFreqCompute<complex64>(ctx);
    case (DT_COMPLEX128):
      return FFTFreqCompute<complex128>(ctx);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "[%s] kernel data type [%s] not support.", op_name.c_str(),
                            DTypeStr(output_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

REGISTER_MS_CPU_KERNEL(kFFTFreq, FFTFreqCpuKernel);
REGISTER_MS_CPU_KERNEL(kRFFTFreq, FFTFreqCpuKernel);
}  // namespace aicpu
