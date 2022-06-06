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

#include "nnacl/base/cast_base.h"
#include "nnacl/intrinsics/ms_simd_instructions.h"
#ifdef ENABLE_AVX512
#include "nnacl/avx512/cast_base_avx512.h"
#endif

#ifdef ENABLE_AVX
#include "nnacl/avx/cast_base_avx.h"
#endif

#ifdef ENABLE_SSE
#include "nnacl/sse/cast_base_sse.h"
#endif

#ifdef ENABLE_ARM
#include "nnacl/neon/cast_base_neon.h"
#endif

void Int32ToFloat32(const int32_t *input, float *output, int number) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(Int32ToFloat32, index, input, output, number);

  for (; index < number; ++index) {
    output[index] = (float)input[index];
  }
}

void Float32ToInt32(const float *input, int32_t *output, int number) {
  int index = 0;

  SIMD_RUN_X86_NO_SCALAR(Float32ToInt32, index, input, output, number);

  for (; index < number; ++index) {
    output[index] = (int32_t)input[index];
  }
}
