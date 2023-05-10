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

#ifdef ENABLE_AVX
#include "nnacl/kernel/convolution_sw_avx.h"

ConvolutionBaseStruct *CreateConvolutionSWAVX(ConvParameter *conv_param) {
  ConvolutionSWAVXStruct *sw = (ConvolutionSWAVXStruct *)malloc(sizeof(ConvolutionSWAVXStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(sw);
  memset(sw, 0, sizeof(ConvolutionSWAVXStruct));

  return (ConvolutionBaseStruct *)sw;
}
#endif
