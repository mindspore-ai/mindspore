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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either convolutionress or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnacl/kernel/convolution_im2col.h"
#include "nnacl/kernel/convolution_im2col_base.h"
#ifdef ENABLE_ARM32
#include "nnacl/kernel/convolution_im2col_arm32.h"
#endif
#ifdef ENABLE_ARM64
#include "nnacl/kernel/convolution_im2col_arm64.h"
#endif
#ifdef ENABLE_SSE
#include "nnacl/kernel/convolution_im2col_sse.h"
#endif
#ifdef ENABLE_AVX
#include "nnacl/kernel/convolution_im2col_avx.h"
#endif
#ifdef ENABLE_AVX512
#include "nnacl/intrinsics/ms_simd_cpu_info.h"
#include "nnacl/kernel/convolution_im2col_avx512.h"
#endif

ConvolutionBaseStruct *CreateConvolutionIm2Col(KernelBase *base, ConvParameter *conv_param) {
  ConvolutionBaseStruct *kernel = NULL;

#ifdef ENABLE_AVX512
  FormatC out_format = base->out_[OUTPUT_INDEX]->format_;
  if (out_format != Format_NC4HW4) {
    AVX512_HARDWARE_SELF_AWARENESS_BEGIN;
    kernel = CreateConvIm2ColAVX512(conv_param);
    if (kernel != NULL) {
      return kernel;
    }
    AVX512_HARDWARE_SELF_AWARENESS_END;
  }
#endif

#ifdef ENABLE_AVX
  kernel = CreateConvIm2ColAVX(conv_param);
  if (kernel != NULL) {
    return kernel;
  }
#endif

#ifdef ENABLE_SSE
  kernel = CreateConvIm2ColSSE(conv_param);
  if (kernel != NULL) {
    return kernel;
  }
#endif

#ifdef ENABLE_ARM64
  kernel = CreateConvIm2ColARM64(conv_param);
  if (kernel != NULL) {
    return kernel;
  }
#endif

#ifdef ENABLE_ARM32
  kernel = CreateConvIm2ColARM32(conv_param);
  if (kernel != NULL) {
    return kernel;
  }
#endif

  kernel = CreateConvIm2ColBase(conv_param);
  return kernel;
}
