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
#include "nnacl/fp32/cdist_fp32.h"
#include "nnacl/intrinsics/ms_simd_instructions.h"
#include "nnacl/cdist_fp32_simd.h"

void CdistTwoNormalOpt(const float *a, const float *b, float *dst, int64_t m, float p) {
  float result = 0;
  int64_t i = 0;

  SIMD_RUN_NO_SCALAR(CdistTwoNormalOpt, i, a, b, &result, m);

  for (; i < m; i++) {
    float x = fabsf(a[i] - b[i]);
    result += x * x;
  }
  result = sqrtf(result);
  *dst = result;

  return;
}

void CdistPNormalOpt(const float *a, const float *b, float *dst, int64_t m, float p) {
  float result = 0;
  int64_t i = 0;

  SIMD_RUN_NO_SCALAR(CdistPNormalOpt, i, a, b, &result, m, p);

  for (; i < m; i++) {
    float x = fabsf(a[i] - b[i]);
    result += powf(x, p);
  }
  result = powf(result, 1.0 / p);
  *dst = result;

  return;
}

void CdistZeroNormalOpt(const float *a, const float *b, float *c, int64_t m, float p) {
  float result = 0;
  for (int64_t i = 0; i < m; i++) {
    float x = fabsf(a[i] - b[i]);
    result += MSMIN(ceilf(x), 1.0f);
  }
  *c = result;
}

void CdistOneNormalOpt(const float *a, const float *b, float *c, int64_t m, float p) {
  float result = 0;
  for (int64_t i = 0; i < m; i++) {
    float x = fabsf(a[i] - b[i]);
    result += x;
  }
  *c = result;
}

void CdistInfNormalOpt(const float *a, const float *b, float *c, int64_t m, float p) {
  float result = 0;
  for (int64_t i = 0; i < m; i++) {
    float x = fabsf(a[i] - b[i]);
    result = MSMAX(result, x);
  }
  *c = result;
}
