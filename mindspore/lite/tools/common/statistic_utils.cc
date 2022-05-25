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
#ifdef ENABLE_SSE
#include "nnacl/intrinsics/ms_simd_cpu_info.h"
#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif
#endif

namespace mindspore::lite {
#ifdef ENABLE_SSE
std::pair<float, float> GetFloatMinMaxValueWithSSE(const float *data, int size) {
  MS_ASSERT(data != nullptr);
  MS_ASSERT(size > 0);
  const int block_width = 4;
  const int cnt_block = size / block_width;
  const int cnt_res = size % block_width;

  float min_output[4];
  float max_output[4];
  __m128 load_data;
  const float *p = data;

  __m128 min_vals = _mm_load_ps(p);
  __m128 max_vals = _mm_load_ps(p);
  p += block_width;

  for (int i = 1; i < cnt_block; i++) {
    load_data = _mm_load_ps(p);
    min_vals = _mm_min_ps(min_vals, load_data);
    max_vals = _mm_max_ps(max_vals, load_data);

    p += block_width;
  }
  _mm_store_ps(min_output, min_vals);
  _mm_store_ps(max_output, max_vals);

  float min_val = min_output[0];
  float max_val = max_output[0];
  for (int i = 1; i < block_width; i++) {
    min_val = min_val < min_output[i] ? min_val : min_output[i];
    max_val = max_val > max_output[i] ? max_val : max_output[i];
  }
  for (int i = 0; i < cnt_res; i++) {
    min_val = min_val < p[i] ? min_val : p[i];
    max_val = max_val > p[i] ? max_val : p[i];
  }
  return {min_val, max_val};
}
#endif

std::pair<float, float> GetFloatMinMaxValue(const float *data, int size) {
#ifdef ENABLE_SSE
  if (IntelX86CpuInfoInit() == RET_OK && X86_Sse_Support()) {
    return GetFloatMinMaxValueWithSSE(data, size);
  } else {
    return GetMinMaxValue(data, size);
  }
#else
  return GetMinMaxValue(data, size);
#endif
}
}  // namespace mindspore::lite
