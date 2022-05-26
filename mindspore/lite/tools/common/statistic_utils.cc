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

#include "tools/common/statistic_utils.h"
#if defined(ENABLE_AVX) && defined(__linux__)
#include "nnacl/intrinsics/ms_simd_cpu_info.h"
#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif
#endif

namespace mindspore::lite {
#if defined(ENABLE_AVX) && defined(__linux__)
std::pair<float, float> GetFloatMinMaxValueWithSSE(const float *data, int size) {
  MS_ASSERT(data != nullptr);
  MS_ASSERT(size > 0);
  const int block_size = 4;

  float min_output[4];
  float max_output[4];
  __m128 load_data;
  const float *p = data;

  __m128 min_vals = _mm_set1_ps(FLT_MAX);
  __m128 max_vals = _mm_set1_ps(-FLT_MAX);

  int index = 0;
  int block_max_size = size - block_size + 1;
  for (; index < block_max_size; index += block_size) {
    load_data = _mm_load_ps(p + index);
    min_vals = _mm_min_ps(min_vals, load_data);
    max_vals = _mm_max_ps(max_vals, load_data);
  }
  _mm_store_ps(min_output, min_vals);
  _mm_store_ps(max_output, max_vals);

  float min_val = min_output[0];
  float max_val = max_output[0];
  for (int i = 1; i < block_size; i++) {
    min_val = min_val < min_output[i] ? min_val : min_output[i];
    max_val = max_val > max_output[i] ? max_val : max_output[i];
  }
  for (; index < size; index++) {
    min_val = min_val < p[index] ? min_val : p[index];
    max_val = max_val > p[index] ? max_val : p[index];
  }
  return {min_val, max_val};
}
#endif

std::pair<float, float> GetFloatMinMaxValue(const float *data, int size) {
#if defined(ENABLE_AVX) && defined(__linux__)
  if (IntelX86CpuInfoInit() == RET_OK && X86_Sse_Support() && size > 64) {
    return GetFloatMinMaxValueWithSSE(data, size);
  } else {
    return GetMinMaxValue(data, size);
  }
#else
  return GetMinMaxValue(data, size);
#endif
}
}  // namespace mindspore::lite
