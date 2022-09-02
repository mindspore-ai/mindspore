/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/convolution_winograd_fp32.h"
#include "src/litert/kernel/cpu/fp32/convolution_winograd_base_fp32.h"
#if defined(ENABLE_AVX)
#include "src/litert/kernel/cpu/fp32/convolution_winograd_avx_fp32.h"
#endif

#if defined(ENABLE_SSE)
#include "src/litert/kernel/cpu/fp32/convolution_winograd_sse_fp32.h"
#endif

#if defined(ENABLE_ARM32)
#include "src/litert/kernel/cpu/fp32/convolution_winograd_arm32_fp32.h"
#endif

#if defined(ENABLE_ARM64)
#include "src/litert/kernel/cpu/fp32/convolution_winograd_arm64_fp32.h"
#endif
#include "nnacl/intrinsics/ms_simd_cpu_info.h"

namespace mindspore::kernel {
LiteKernel *CreateConvolutionWinogradCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs,
                                               const lite::InnerContext *ctx, int out_unit, float *origin_weight,
                                               float *origin_bias) {
  LiteKernel *kernel = nullptr;

#if defined(ENABLE_AVX)
  if (kernel == nullptr) {
    kernel = new (std::nothrow)
      kernel::ConvolutionWinogradAVXCPUKernel(parameter, inputs, outputs, ctx, out_unit, origin_weight, origin_bias);
  }
#endif

#if defined(ENABLE_SSE)
  if (kernel == nullptr) {
    kernel = new (std::nothrow)
      kernel::ConvolutionWinogradSSECPUKernel(parameter, inputs, outputs, ctx, out_unit, origin_weight, origin_bias);
  }
#endif

#if defined(ENABLE_ARM64)
  if (kernel == nullptr) {
    kernel = new (std::nothrow)
      kernel::ConvolutionWinogradARM64CPUKernel(parameter, inputs, outputs, ctx, out_unit, origin_weight, origin_bias);
  }
#elif defined(ENABLE_ARM32)
  if (kernel == nullptr) {
    kernel = new (std::nothrow)
      kernel::ConvolutionWinogradARM32CPUKernel(parameter, inputs, outputs, ctx, out_unit, origin_weight, origin_bias);
  }
#endif

  if (kernel == nullptr) {
    kernel = new (std::nothrow)
      kernel::ConvolutionWinogradBaseCPUKernel(parameter, inputs, outputs, ctx, out_unit, origin_weight, origin_bias);
  }
  return kernel;
}
}  // namespace mindspore::kernel
