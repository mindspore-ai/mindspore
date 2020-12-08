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

#ifdef ENABLE_SSE
#include <x86intrin.h>
#include "nnacl/minimal_filtering_generator.h"
#include "nnacl/op_base.h"

void MatrixMultiplyWinograd(const float *matix_a, const float *matrix_b, float *matrix_c, int m, int k, int n,
                            int in_channel, int c4_channel) {
  const float *src1 = matix_a;
  int c16 = DOWN_DIV(in_channel, C16NUM) * C16NUM;
  int c8 = DOWN_DIV(in_channel, C8NUM) * C8NUM;
  for (int i = 0; i < m; ++i) {
    const float *src1_n = src1;
    const float *src2_n = matrix_b;
    for (int j = 0; j < n; ++j) {
      const float *src1_j = src1_n;
      int y = 0;
      // 16 channel
      for (; y < c16; y += C16NUM) {
        __m128 dst1 = _mm_setzero_ps();
        __m128 dst2 = _mm_setzero_ps();
        __m128 dst3 = _mm_setzero_ps();
        __m128 dst4 = _mm_setzero_ps();
        const float *src2_y = src2_n;
        for (int z = 0; z < k; ++z) {
          __m128 ma1 = _mm_loadu_ps(src1_j);
          __m128 ma2 = _mm_loadu_ps(src1_j + 4);
          __m128 ma3 = _mm_loadu_ps(src1_j + 8);
          __m128 ma4 = _mm_loadu_ps(src1_j + 12);

          __m128 mb = _mm_load_ps1(src2_y);
          __m128 tmp1 = _mm_mul_ps(ma1, mb);
          __m128 tmp2 = _mm_mul_ps(ma2, mb);
          __m128 tmp3 = _mm_mul_ps(ma3, mb);
          __m128 tmp4 = _mm_mul_ps(ma4, mb);
          dst1 = _mm_add_ps(dst1, tmp1);
          dst2 = _mm_add_ps(dst2, tmp2);
          dst3 = _mm_add_ps(dst3, tmp3);
          dst4 = _mm_add_ps(dst4, tmp4);
          src1_j += in_channel;
          src2_y += n;
        }
        _mm_storeu_ps(matrix_c, dst1);
        _mm_storeu_ps(matrix_c + 4, dst2);
        _mm_storeu_ps(matrix_c + 8, dst3);
        _mm_storeu_ps(matrix_c + 12, dst4);
        src1_j -= in_channel * k;
        src1_j += C16NUM;
        matrix_c += C16NUM;
      }
      // 8 channel
      for (; y < c8; y += C8NUM) {
        __m128 dst1 = _mm_setzero_ps();
        __m128 dst2 = _mm_setzero_ps();
        const float *src2_y = src2_n;
        for (int z = 0; z < k; ++z) {
          __m128 ma1 = _mm_loadu_ps(src1_j);
          __m128 ma2 = _mm_loadu_ps(src1_j + 4);

          __m128 mb = _mm_load_ps1(src2_y);
          __m128 tmp1 = _mm_mul_ps(ma1, mb);
          __m128 tmp2 = _mm_mul_ps(ma2, mb);
          dst1 = _mm_add_ps(dst1, tmp1);
          dst2 = _mm_add_ps(dst2, tmp2);
          src1_j += in_channel;
          src2_y += n;
        }
        _mm_storeu_ps(matrix_c, dst1);
        _mm_storeu_ps(matrix_c + 4, dst2);
        src1_j -= in_channel * k;
        src1_j += C8NUM;
        matrix_c += C8NUM;
      }
      // remain chann
      for (; y < in_channel; ++y) {
        float tmp = 0;
        for (int z = 0; z < k; ++z) {
          tmp += matix_a[z * in_channel + y + i * in_channel * k] * matrix_b[j + z * n];
        }
        *matrix_c++ = tmp;
      }
      src2_n += 1;
    }
    src1 += k * in_channel;
  }
}

void MatmulFloatSse64Opt(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                         int col, int stride, int write_mode) {
  int C8Steps = row * C8NUM;
  int WinoSteps1 = stride * col;
  int WinoSteps2 = stride * C8NUM;
  for (int r = row; r > 0; r -= C4NUM) {
    const float *srcb_d = b;
    const float *bias_d = bias;
    float *dst = NULL;
    for (int cc = col; cc > 0; cc -= C8NUM) {
      if (write_mode != 0) {  // writec8
        dst = c;
      }
      const float *srca_d = a;
      __m128 dst1 = _mm_setzero_ps();
      __m128 dst2 = _mm_setzero_ps();
      __m128 dst3 = _mm_setzero_ps();
      __m128 dst4 = _mm_setzero_ps();
      __m128 dst5 = _mm_setzero_ps();
      __m128 dst6 = _mm_setzero_ps();
      __m128 dst7 = _mm_setzero_ps();
      __m128 dst8 = _mm_setzero_ps();
      for (int d = depth; d > 0; --d) {
        __m128 b1 = _mm_loadu_ps(srcb_d);
        __m128 b2 = _mm_loadu_ps(srcb_d + 4);
        __m128 a1 = _mm_load_ps1(srca_d);
        __m128 a2 = _mm_load_ps1(srca_d + 1);
        __m128 tmp1 = _mm_mul_ps(b1, a1);
        __m128 tmp2 = _mm_mul_ps(b2, a1);
        __m128 tmp3 = _mm_mul_ps(b1, a2);
        __m128 tmp4 = _mm_mul_ps(b2, a2);
        a1 = _mm_load_ps1(srca_d + 2);
        dst1 = _mm_add_ps(dst1, tmp1);
        dst2 = _mm_add_ps(dst2, tmp2);
        a2 = _mm_load_ps1(srca_d + 3);
        dst3 = _mm_add_ps(dst3, tmp3);
        dst4 = _mm_add_ps(dst4, tmp4);
        tmp1 = _mm_mul_ps(b1, a1);
        tmp2 = _mm_mul_ps(b2, a1);
        tmp3 = _mm_mul_ps(b1, a2);
        tmp4 = _mm_mul_ps(b2, a2);
        dst5 = _mm_add_ps(dst5, tmp1);
        dst6 = _mm_add_ps(dst6, tmp2);
        dst7 = _mm_add_ps(dst7, tmp3);
        dst8 = _mm_add_ps(dst8, tmp4);
        srcb_d += C8NUM;
        srca_d += C4NUM;
      }
      if (bias != NULL) {
        __m128 bias1 = _mm_loadu_ps(bias_d);
        __m128 bias2 = _mm_loadu_ps(bias_d + C4NUM);
        dst1 = _mm_add_ps(dst1, bias1);
        dst2 = _mm_add_ps(dst2, bias2);
        dst3 = _mm_add_ps(dst3, bias1);
        dst4 = _mm_add_ps(dst4, bias2);
        dst5 = _mm_add_ps(dst5, bias1);
        dst6 = _mm_add_ps(dst6, bias2);
        dst7 = _mm_add_ps(dst7, bias1);
        dst8 = _mm_add_ps(dst8, bias2);
        bias_d += C8NUM;
      }
      if (act_type == 3) {
        __m128 relu6 = _mm_set_ps(6.0, 6.0, 6.0, 6.0);
        dst1 = _mm_min_ps(dst1, relu6);
        dst2 = _mm_min_ps(dst2, relu6);
        dst3 = _mm_min_ps(dst3, relu6);
        dst4 = _mm_min_ps(dst4, relu6);
        dst5 = _mm_min_ps(dst5, relu6);
        dst6 = _mm_min_ps(dst6, relu6);
        dst7 = _mm_min_ps(dst7, relu6);
        dst8 = _mm_min_ps(dst8, relu6);
      }
      if (act_type == 1 || act_type == 3) {
        __m128 zero = _mm_setzero_ps();
        dst1 = _mm_max_ps(dst1, zero);
        dst2 = _mm_max_ps(dst2, zero);
        dst3 = _mm_max_ps(dst3, zero);
        dst4 = _mm_max_ps(dst4, zero);
        dst5 = _mm_max_ps(dst5, zero);
        dst6 = _mm_max_ps(dst6, zero);
        dst7 = _mm_max_ps(dst7, zero);
        dst8 = _mm_max_ps(dst8, zero);
      }
      if (write_mode == 2) {  // WriteWino
        c = dst + WinoSteps2;
        _mm_storeu_ps(dst, dst1);
        _mm_storeu_ps(dst + 4, dst2);
        dst += WinoSteps1;
        _mm_storeu_ps(dst, dst3);
        _mm_storeu_ps(dst + 4, dst4);
        dst += WinoSteps1;
        _mm_storeu_ps(dst, dst5);
        _mm_storeu_ps(dst + 4, dst6);
        dst += WinoSteps1;
        _mm_storeu_ps(dst, dst7);
        _mm_storeu_ps(dst + 4, dst8);
      } else if (write_mode == 0) {  // WriteC8
        _mm_storeu_ps(c, dst1);
        _mm_storeu_ps(c + 4, dst2);
        _mm_storeu_ps(c + 8, dst3);
        _mm_storeu_ps(c + 12, dst4);
        _mm_storeu_ps(c + 16, dst5);
        _mm_storeu_ps(c + 20, dst6);
        _mm_storeu_ps(c + 24, dst7);
        _mm_storeu_ps(c + 28, dst8);
        c += C8Steps;
      } else {
        switch (cc) {
          case 1:  // write1
            c = dst + 1;
            _mm_store_ss(dst, dst1);
            if (r > 1) {
              dst += stride;
              _mm_store_ss(dst, dst3);
            }
            if (r > 2) {
              dst += stride;
              _mm_store_ss(dst, dst5);
            }
            if (r > 3) {
              dst += stride;
              _mm_store_ss(dst, dst7);
              dst += stride;
              dst += 1;
            }
            break;
          case 2:  // write2
            c = dst + 2;
            _mm_store_ss(dst, dst1);
            dst1 = _mm_shuffle_ps(dst1, dst1, _MM_SHUFFLE(0, 3, 2, 1));
            _mm_store_ss(dst + 1, dst1);
            if (r > 1) {
              dst += stride;
              _mm_store_ss(dst, dst3);
              dst3 = _mm_shuffle_ps(dst3, dst3, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 1, dst3);
            }
            if (r > 2) {
              dst += stride;
              _mm_store_ss(dst, dst5);
              dst5 = _mm_shuffle_ps(dst5, dst5, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 1, dst5);
            }
            if (r > 3) {
              dst += stride;
              _mm_store_ss(dst, dst7);
              dst7 = _mm_shuffle_ps(dst7, dst7, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 1, dst7);
              dst += stride;
              dst += 2;
            }
            break;
          case 3:  // write3
            c = dst + 3;
            _mm_store_ss(dst, dst1);
            dst1 = _mm_shuffle_ps(dst1, dst1, _MM_SHUFFLE(0, 3, 2, 1));
            _mm_store_ss(dst + 1, dst1);
            dst1 = _mm_shuffle_ps(dst1, dst1, _MM_SHUFFLE(0, 3, 2, 1));
            _mm_store_ss(dst + 2, dst1);
            if (r > 1) {
              dst += stride;
              _mm_store_ss(dst, dst3);
              dst3 = _mm_shuffle_ps(dst3, dst3, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 1, dst3);
              dst3 = _mm_shuffle_ps(dst3, dst3, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 2, dst3);
            }
            if (r > 2) {
              dst += stride;
              _mm_store_ss(dst, dst5);
              dst5 = _mm_shuffle_ps(dst5, dst5, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 1, dst5);
              dst5 = _mm_shuffle_ps(dst5, dst5, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 2, dst5);
            }
            if (r > 3) {
              dst += stride;
              _mm_store_ss(dst, dst7);
              dst7 = _mm_shuffle_ps(dst7, dst7, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 1, dst7);
              dst7 = _mm_shuffle_ps(dst7, dst7, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 2, dst7);
              dst += stride;
              dst += 3;
            }
            break;
          case 4:  // write4
            c = dst + 4;
            _mm_storeu_ps(dst, dst1);
            if (r > 1) {
              dst += stride;
              _mm_storeu_ps(dst, dst3);
            }
            if (r > 2) {
              dst += stride;
              _mm_storeu_ps(dst, dst5);
            }
            if (r > 3) {
              dst += stride;
              _mm_storeu_ps(dst, dst7);
              dst += stride;
              dst += 4;
            }
            break;
          case 5:  // write5
            c = dst + 5;
            _mm_storeu_ps(dst, dst1);
            _mm_store_ss(dst + 4, dst2);
            if (r > 1) {
              dst += stride;
              _mm_storeu_ps(dst, dst3);
              _mm_store_ss(dst + 4, dst4);
            }
            if (r > 2) {
              dst += stride;
              _mm_storeu_ps(dst, dst5);
              _mm_store_ss(dst + 4, dst6);
            }
            if (r > 3) {
              dst += stride;
              _mm_storeu_ps(dst, dst7);
              _mm_store_ss(dst + 4, dst8);
              dst += stride;
              dst += 5;
            }
            break;
          case 6:  // write6
            c = dst + 6;
            _mm_storeu_ps(dst, dst1);
            _mm_store_ss(dst + 4, dst2);
            dst2 = _mm_shuffle_ps(dst2, dst2, _MM_SHUFFLE(0, 3, 2, 1));
            _mm_store_ss(dst + 5, dst2);
            if (r > 1) {
              dst += stride;
              _mm_storeu_ps(dst, dst3);
              _mm_store_ss(dst + 4, dst4);
              dst4 = _mm_shuffle_ps(dst4, dst4, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 5, dst4);
            }
            if (r > 2) {
              dst += stride;
              _mm_storeu_ps(dst, dst5);
              _mm_store_ss(dst + 4, dst6);
              dst6 = _mm_shuffle_ps(dst6, dst6, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 5, dst6);
            }
            if (r > 3) {
              dst += stride;
              _mm_storeu_ps(dst, dst7);
              _mm_store_ss(dst + 4, dst8);
              dst8 = _mm_shuffle_ps(dst8, dst8, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 5, dst8);
              dst += stride;
              dst += 6;
            }
            break;
          case 7:  // write7
            c = dst + 7;
            _mm_storeu_ps(dst, dst1);
            _mm_store_ss(dst + 4, dst2);
            dst2 = _mm_shuffle_ps(dst2, dst2, _MM_SHUFFLE(0, 3, 2, 1));
            _mm_store_ss(dst + 5, dst2);
            dst2 = _mm_shuffle_ps(dst2, dst2, _MM_SHUFFLE(0, 3, 2, 1));
            _mm_store_ss(dst + 6, dst2);
            if (r > 1) {
              dst += stride;
              _mm_storeu_ps(dst, dst3);
              _mm_store_ss(dst + 4, dst4);
              dst4 = _mm_shuffle_ps(dst4, dst4, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 5, dst4);
              dst4 = _mm_shuffle_ps(dst4, dst4, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 6, dst4);
            }
            if (r > 2) {
              dst += stride;
              _mm_storeu_ps(dst, dst5);
              _mm_store_ss(dst + 4, dst6);
              dst6 = _mm_shuffle_ps(dst6, dst6, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 5, dst6);
              dst6 = _mm_shuffle_ps(dst6, dst6, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 6, dst6);
            }
            if (r > 3) {
              dst += stride;
              _mm_storeu_ps(dst, dst7);
              _mm_store_ss(dst + 4, dst8);
              dst8 = _mm_shuffle_ps(dst8, dst8, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 5, dst8);
              dst8 = _mm_shuffle_ps(dst8, dst8, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 6, dst8);
              dst += stride;
              dst += 7;
            }
            break;
          default:  // write8
            c = dst + C8NUM;
            _mm_storeu_ps(dst, dst1);
            _mm_storeu_ps(dst + 4, dst2);
            if (r > 1) {
              dst += stride;
              _mm_storeu_ps(dst, dst3);
              _mm_storeu_ps(dst + 4, dst4);
            }
            if (r > 2) {
              dst += stride;
              _mm_storeu_ps(dst, dst5);
              _mm_storeu_ps(dst + 4, dst6);
            }
            if (r > 3) {
              dst += stride;
              _mm_storeu_ps(dst, dst7);
              _mm_storeu_ps(dst + 4, dst8);
              dst += stride;
              dst += C8NUM;
            }
            break;
        }
      }
      if (cc <= C8NUM) {  // write end
        break;
      }
    }  // col end
    a += C4NUM * depth;
    switch (write_mode) {
      case 0:  // C8DstStep
        c += 32;
        break;
      case 2:
        c = dst + WinoSteps2;
        break;
      default:
        c = dst - col;
        break;
    }
    if (r <= C4NUM) {
      break;
    }
  }
}

void MatmulFloatSse64(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                      int col, int stride, size_t writeNhwc, size_t WriteWino) {
  size_t DstWinoSteps = stride * C8NUM;
  size_t WriteWinoSteps = stride * col;
  for (int col_tmp = col; col_tmp > 0; col_tmp -= C8NUM) {
    const float *srca_d = a;
    float *dst = c;
    for (int r = row; r > 0; r -= C4NUM) {
      const float *srcb_d = b;
      __m128 dst1 = _mm_setzero_ps();
      __m128 dst2 = _mm_setzero_ps();
      __m128 dst3 = _mm_setzero_ps();
      __m128 dst4 = _mm_setzero_ps();
      __m128 dst5 = _mm_setzero_ps();
      __m128 dst6 = _mm_setzero_ps();
      __m128 dst7 = _mm_setzero_ps();
      __m128 dst8 = _mm_setzero_ps();
      for (int d = 0; d < depth; d++) {
        __m128 b1 = _mm_loadu_ps(srcb_d);
        __m128 b2 = _mm_loadu_ps(srcb_d + 4);
        __m128 a1 = _mm_load_ps1(srca_d);
        __m128 a2 = _mm_load_ps1(srca_d + 1);
        __m128 tmp1 = _mm_mul_ps(b1, a1);
        __m128 tmp2 = _mm_mul_ps(b2, a1);
        __m128 tmp3 = _mm_mul_ps(b1, a2);
        __m128 tmp4 = _mm_mul_ps(b2, a2);
        a1 = _mm_load_ps1(srca_d + 2);
        dst1 = _mm_add_ps(dst1, tmp1);
        dst2 = _mm_add_ps(dst2, tmp2);
        a2 = _mm_load_ps1(srca_d + 3);
        dst3 = _mm_add_ps(dst3, tmp3);
        dst4 = _mm_add_ps(dst4, tmp4);
        tmp1 = _mm_mul_ps(b1, a1);
        tmp2 = _mm_mul_ps(b2, a1);
        tmp3 = _mm_mul_ps(b1, a2);
        tmp4 = _mm_mul_ps(b2, a2);
        dst5 = _mm_add_ps(dst5, tmp1);
        dst6 = _mm_add_ps(dst6, tmp2);
        dst7 = _mm_add_ps(dst7, tmp3);
        dst8 = _mm_add_ps(dst8, tmp4);
        srcb_d += C8NUM;
        srca_d += C4NUM;
      }
      if (bias != NULL) {
        __m128 bias1 = _mm_loadu_ps(bias);
        __m128 bias2 = _mm_loadu_ps(bias + C4NUM);
        dst1 = _mm_add_ps(dst1, bias1);
        dst2 = _mm_add_ps(dst2, bias2);
        dst3 = _mm_add_ps(dst3, bias1);
        dst4 = _mm_add_ps(dst4, bias2);
        dst5 = _mm_add_ps(dst5, bias1);
        dst6 = _mm_add_ps(dst6, bias2);
        dst7 = _mm_add_ps(dst7, bias1);
        dst8 = _mm_add_ps(dst8, bias2);
      }
      if (act_type == 3) {
        __m128 relu6 = _mm_set_ps(6.0, 6.0, 6.0, 6.0);
        dst1 = _mm_min_ps(dst1, relu6);
        dst2 = _mm_min_ps(dst2, relu6);
        dst3 = _mm_min_ps(dst3, relu6);
        dst4 = _mm_min_ps(dst4, relu6);
        dst5 = _mm_min_ps(dst5, relu6);
        dst6 = _mm_min_ps(dst6, relu6);
        dst7 = _mm_min_ps(dst7, relu6);
        dst8 = _mm_min_ps(dst8, relu6);
      }
      if (act_type == 1 || act_type == 3) {
        __m128 zero = _mm_setzero_ps();
        dst1 = _mm_max_ps(dst1, zero);
        dst2 = _mm_max_ps(dst2, zero);
        dst3 = _mm_max_ps(dst3, zero);
        dst4 = _mm_max_ps(dst4, zero);
        dst5 = _mm_max_ps(dst5, zero);
        dst6 = _mm_max_ps(dst6, zero);
        dst7 = _mm_max_ps(dst7, zero);
        dst8 = _mm_max_ps(dst8, zero);
      }
      if (WriteWino != 0) {  // WriteWino
        _mm_storeu_ps(dst, dst1);
        _mm_storeu_ps(dst + 4, dst2);
        dst += WriteWinoSteps;
        _mm_storeu_ps(dst, dst3);
        _mm_storeu_ps(dst + 4, dst4);
        dst += WriteWinoSteps;
        _mm_storeu_ps(dst, dst5);
        _mm_storeu_ps(dst + 4, dst6);
        dst += WriteWinoSteps;
        _mm_storeu_ps(dst, dst7);
        _mm_storeu_ps(dst + 4, dst8);
        dst += WriteWinoSteps;
      } else if (writeNhwc == 0) {  // WriteC8
        _mm_storeu_ps(dst, dst1);
        _mm_storeu_ps(dst + 4, dst2);
        _mm_storeu_ps(dst + 8, dst3);
        _mm_storeu_ps(dst + 12, dst4);
        _mm_storeu_ps(dst + 16, dst5);
        _mm_storeu_ps(dst + 20, dst6);
        _mm_storeu_ps(dst + 24, dst7);
        _mm_storeu_ps(dst + 28, dst8);
        dst += 32;
        c = dst;
      } else {
        switch (col) {
          case 1:  // write1
            _mm_store_ss(dst, dst1);
            if (r > 1) {
              dst += stride;
              _mm_store_ss(dst, dst3);
            }
            if (r > 2) {
              dst += stride;
              _mm_store_ss(dst, dst5);
            }
            if (r > 3) {
              dst += stride;
              _mm_store_ss(dst, dst7);
              dst += stride;
            }
          case 2:  // write2
            _mm_store_ss(dst, dst1);
            dst1 = _mm_shuffle_ps(dst1, dst1, _MM_SHUFFLE(0, 3, 2, 1));
            _mm_store_ss(dst, dst1);
            if (r > 1) {
              dst += stride;
              _mm_store_ss(dst, dst3);
              dst3 = _mm_shuffle_ps(dst3, dst3, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst, dst3);
            }
            if (r > 2) {
              dst += stride;
              _mm_store_ss(dst, dst5);
              dst5 = _mm_shuffle_ps(dst5, dst5, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst, dst5);
            }
            if (r > 3) {
              dst += stride;
              _mm_store_ss(dst, dst7);
              dst7 = _mm_shuffle_ps(dst7, dst7, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst, dst7);
            }
          case 3:  // write3
            _mm_store_ss(dst, dst1);
            dst1 = _mm_shuffle_ps(dst1, dst1, _MM_SHUFFLE(0, 3, 2, 1));
            _mm_store_ss(dst + 1, dst1);
            dst1 = _mm_shuffle_ps(dst1 + 2, dst1, _MM_SHUFFLE(0, 3, 2, 1));
            _mm_store_ss(dst, dst1);
            if (r > 1) {
              dst += stride;
              _mm_store_ss(dst, dst3);
              dst3 = _mm_shuffle_ps(dst3, dst3, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 1, dst3);
              dst3 = _mm_shuffle_ps(dst3, dst3, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 2, dst3);
            }
            if (r > 2) {
              dst += stride;
              _mm_store_ss(dst, dst5);
              dst5 = _mm_shuffle_ps(dst5, dst5, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 1, dst5);
              dst5 = _mm_shuffle_ps(dst5, dst5, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 2, dst5);
            }
            if (r > 3) {
              dst += stride;
              _mm_store_ss(dst, dst7);
              dst7 = _mm_shuffle_ps(dst7, dst7, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 1, dst7);
              dst7 = _mm_shuffle_ps(dst7, dst7, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 2, dst7);
              dst += stride;
            }
          case 4:  // write4
            _mm_storeu_ps(dst, dst1);
            if (r > 1) {
              dst += stride;
              _mm_storeu_ps(dst, dst3);
            }
            if (r > 2) {
              dst += stride;
              _mm_storeu_ps(dst, dst5);
            }
            if (r > 3) {
              dst += stride;
              _mm_storeu_ps(dst, dst7);
              dst += stride;
            }
          case 5:  // // write5
            _mm_storeu_ps(dst, dst1);
            _mm_store_ss(dst + 4, dst2);
            if (r > 1) {
              dst += stride;
              _mm_storeu_ps(dst, dst3);
              _mm_store_ss(dst + 4, dst4);
            }
            if (r > 2) {
              dst += stride;
              _mm_storeu_ps(dst, dst5);
              _mm_store_ss(dst + 4, dst6);
            }
            if (r > 3) {
              dst += stride;
              _mm_storeu_ps(dst, dst7);
              _mm_store_ss(dst + 4, dst8);
              dst += stride;
            }
          case 6:  // write6
            _mm_storeu_ps(dst, dst1);
            _mm_store_ss(dst + 4, dst2);
            dst2 = _mm_shuffle_ps(dst2, dst2, _MM_SHUFFLE(0, 3, 2, 1));
            _mm_store_ss(dst + 5, dst2);
            if (r > 1) {
              dst += stride;
              _mm_storeu_ps(dst, dst3);
              _mm_store_ss(dst + 4, dst4);
              dst4 = _mm_shuffle_ps(dst4, dst4, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 5, dst4);
            }
            if (r > 2) {
              dst += stride;
              _mm_storeu_ps(dst, dst5);
              _mm_store_ss(dst + 4, dst6);
              dst6 = _mm_shuffle_ps(dst6, dst6, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 5, dst6);
            }
            if (r > 3) {
              dst += stride;
              _mm_storeu_ps(dst, dst7);
              _mm_store_ss(dst + 4, dst8);
              dst8 = _mm_shuffle_ps(dst8, dst8, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 5, dst8);
              dst += stride;
            }
          case 7:  // write7
            _mm_storeu_ps(dst, dst1);
            _mm_store_ss(dst + 4, dst2);
            dst2 = _mm_shuffle_ps(dst2, dst2, _MM_SHUFFLE(0, 3, 2, 1));
            _mm_store_ss(dst + 5, dst2);
            dst2 = _mm_shuffle_ps(dst2, dst2, _MM_SHUFFLE(0, 3, 2, 1));
            _mm_store_ss(dst + 6, dst2);
            if (r > 1) {
              dst += stride;
              _mm_storeu_ps(dst, dst3);
              _mm_store_ss(dst + 4, dst4);
              dst4 = _mm_shuffle_ps(dst4, dst4, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 5, dst4);
              dst4 = _mm_shuffle_ps(dst4, dst4, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 6, dst4);
            }
            if (r > 2) {
              dst += stride;
              _mm_storeu_ps(dst, dst5);
              _mm_store_ss(dst + 4, dst6);
              dst6 = _mm_shuffle_ps(dst6, dst6, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 5, dst6);
              dst6 = _mm_shuffle_ps(dst6, dst6, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 6, dst6);
            }
            if (r > 3) {
              dst += stride;
              _mm_storeu_ps(dst, dst7);
              _mm_store_ss(dst + 4, dst8);
              dst8 = _mm_shuffle_ps(dst8, dst8, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 5, dst8);
              dst8 = _mm_shuffle_ps(dst8, dst8, _MM_SHUFFLE(0, 3, 2, 1));
              _mm_store_ss(dst + 6, dst8);
              dst += stride;
            }
          default:  // write8
            _mm_storeu_ps(dst, dst1);
            _mm_storeu_ps(dst + 4, dst2);
            if (r > 1) {
              dst += stride;
              _mm_storeu_ps(dst, dst3);
              _mm_storeu_ps(dst + 4, dst4);
            }
            if (r > 2) {
              dst += stride;
              _mm_storeu_ps(dst, dst5);
              _mm_storeu_ps(dst + 4, dst6);
            }
            if (r > 3) {
              dst += stride;
              _mm_storeu_ps(dst, dst7);
              _mm_storeu_ps(dst + 4, dst8);
              dst += stride;
            }
        }
      }
      if (r <= C4NUM) {  // WriteEnd
        break;
      }
    }
    b += depth * C8NUM;
    bias += (bias != NULL) ? C8NUM : 0;
    if (WriteWino != 0) {
      c += DstWinoSteps;
    } else if (writeNhwc != 0) {
      c += C8NUM;
    }
    if (col_tmp <= C8NUM) {
      break;
    }
  }
}
#endif
