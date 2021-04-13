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
        float tmp = 0;
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
