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

#include "nnacl/kernel/convolution_winograd.h"
#include "nnacl/kernel/convolution_winograd_base.h"
#ifdef ENABLE_AVX
#include "nnacl/kernel/convolution_winograd_avx.h"
#endif
#ifdef ENABLE_SSE
#include "nnacl/kernel/convolution_winograd_sse.h"
#endif
#ifdef ENABLE_ARM64
#include "nnacl/kernel/convolution_winograd_arm64.h"
#endif
#ifdef ENABLE_ARM32
#include "nnacl/kernel/convolution_winograd_arm32.h"
#endif

ConvolutionWinogradBaseStruct *SelectConvolutionWinograd(ConvParameter *conv_param) {
  ConvolutionWinogradBaseStruct *kernel = NULL;

#ifdef ENABLE_AVX
  kernel = CreateConvWinogradAVX(conv_param);
  if (kernel != NULL) {
    return kernel;
  }
#endif

#ifdef ENABLE_SSE
  kernel = CreateConvWinogradSSE(conv_param);
  if (kernel != NULL) {
    return kernel;
  }
#endif

#ifdef ENABLE_ARM64
  kernel = CreateConvWinogradARM64(conv_param);
  if (kernel != NULL) {
    return kernel;
  }
#endif

#ifdef ENABLE_ARM32
  kernel = CreateConvWinogradARM32(conv_param);
  if (kernel != NULL) {
    return kernel;
  }
#endif

  kernel = CreateConvWinogradBase(conv_param);
  return kernel;
}

ConvolutionBaseStruct *CreateConvolutionWinograd(ConvParameter *conv_param, int out_unit) {
  ConvolutionWinogradBaseStruct *kernel = SelectConvolutionWinograd(conv_param);
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(kernel);

  kernel->output_unit_ = out_unit;
  kernel->conv_.malloc_weight_bias_ = ConvWinoBaseMallocWeightBiasData;
  kernel->conv_.run_impl_ = ConvWinoBaseRunImpl;
  kernel->conv_.pack_weight_ = ConvWinoBasePackWeight;
  return (ConvolutionBaseStruct *)kernel;
}
