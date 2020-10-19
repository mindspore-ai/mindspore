/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "nnacl/fp16/matrix_fp16.h"

void MatrixMultiplyFp16(const float16_t *matrix_a, const float16_t *matrix_b, float16_t *matrix_c, int m, int k,
                        int n) {
  int count = 0;
  for (int h = 0; h < m; h++) {
    int h_offset = h * k;
    for (int w = 0; w < n; w++) {
      float16_t res = 0;
      for (int i = 0; i < k; i++) {
        res += *(matrix_a + h_offset + i) * *(matrix_b + w + i * n);
      }
      *(matrix_c + count) = res;
      count++;
    }
  }
}

#ifndef ENABLE_ARM64
void MatrixMultiplyWinogradFp16(const float16_t *matix_a, const float16_t *matrix_b, float16_t *matrix_c, int m, int k,
                                int n, int in_channel) {
  int cnt = 0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int y = 0; y < in_channel; ++y) {
        float16_t tmp = 0;
        for (int z = 0; z < k; ++z) {
          tmp += matix_a[z * in_channel + y + i * in_channel * k] * matrix_b[j + z * n];
        }
        matrix_c[cnt++] = tmp;
      }
    }
  }
}
#endif

void MatrixMultiplyVecFp16(const float16x8_t *matrix_a, const float16x8_t *matrix_b, float16x8_t *matrix_c,
                           const float16_t *bias, int m, int k, int n) {
  if (bias == NULL) {
    int count = 0;
    for (int h = 0; h < m; h++) {
      int h_offset = h * k;
      for (int w = 0; w < n; w++) {
        float16x8_t res = vmovq_n_f16(0);
        for (int i = 0; i < k; i++) {
          res = vaddq_f16(res, vmulq_f16(matrix_a[h_offset + i], matrix_b[w + i * n]));
        }
        matrix_c[count] = res;
        count++;
      }
    }
  } else {
    int count = 0;
    float16x8_t bias_ptr = vld1q_f16(bias);
    for (int h = 0; h < m; h++) {
      int h_offset = h * k;
      for (int w = 0; w < n; w++) {
        float16x8_t res = vmovq_n_f16(0);
        for (int i = 0; i < k; i++) {
          res = vaddq_f16(res, vmulq_f16(matrix_a[h_offset + i], matrix_b[w + i * n]));
        }
        matrix_c[count] = vaddq_f16(res, bias_ptr);
        count++;
      }
    }
  }
}

void WinogradMatrixProductLeftFp16(const float16_t *S, const float16_t *B, float16_t *M, size_t w, size_t h, size_t k,
                                   size_t length) {
  int unitStep = 4 * length;
  for (int y = 0; y < h; ++y) {
    float16_t *dstY = M + y * w * unitStep;
    for (int x = 0; x < w; ++x) {
      float16_t *dstX = dstY + x * unitStep;
      const float16_t *srcX = S + x * unitStep;
      memset(dstX, 0, unitStep * sizeof(float16_t));
      for (int i = 0; i < k; ++i) {
        float16_t b = B[i * h + y];
        const float16_t *srcY = srcX + i * w * unitStep;
        if (0.0f == b) {
          continue;
        }
        for (int j = 0; j < unitStep; ++j) {
          dstX[j] += srcY[j] * b;
        }
      }
    }
  }
}

// M = S * B , M = w*h * l, S = k*h * l, B = w*k
void WinogradMatrixProductRightFp16(const float16_t *S, const float16_t *B, float16_t *M, size_t w, size_t h, size_t k,
                                    size_t length) {
  int unitStep = 4 * length;
  for (int y = 0; y < h; ++y) {
    float16_t *dstY = M + y * w * unitStep;
    const float16_t *srcY = S + y * k * unitStep;

    for (int x = 0; x < w; ++x) {
      float16_t *dstX = dstY + x * unitStep;
      memset(dstX, 0, unitStep * sizeof(float16_t));
      for (int i = 0; i < k; ++i) {
        const float16_t *srcX = srcY + i * unitStep;
        float16_t b = B[i * h + x];
        if (0.0f == b) {
          continue;
        }
        for (int j = 0; j < unitStep; ++j) {
          dstX[j] += srcX[j] * b;
        }
      }
    }
  }
}
