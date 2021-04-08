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
#include "nnacl/op_base.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/intrinsics/sse/sse_common.h"
#include "nnacl/base/minimal_filtering_generator.h"

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
  int C8Steps = row * C8NUM, WinoSteps1 = stride * col, WinoSteps2 = stride * C8NUM;
  for (int r = row; r > 0; r -= C4NUM) {
    const float *srcb_d = b, *bias_d = bias;
    float *dst = NULL;
    for (int cc = col; cc > 0; cc -= C8NUM) {
      if (write_mode != 0) {  // writec8
        dst = c;
      }
      const float *srca_d = a;
      __m128 dst1 = _mm_setzero_ps(), dst2 = _mm_setzero_ps(), dst3 = _mm_setzero_ps(), dst4 = _mm_setzero_ps();
      __m128 dst5 = _mm_setzero_ps(), dst6 = _mm_setzero_ps(), dst7 = _mm_setzero_ps(), dst8 = _mm_setzero_ps();
      for (int d = depth; d > 0; --d) {
        __m128 b1 = _mm_loadu_ps(srcb_d), b2 = _mm_loadu_ps(srcb_d + 4);
        __m128 a1 = _mm_load_ps1(srca_d), a2 = _mm_load_ps1(srca_d + 1);
        __m128 tmp1 = _mm_mul_ps(b1, a1), tmp2 = _mm_mul_ps(b2, a1);
        __m128 tmp3 = _mm_mul_ps(b1, a2), tmp4 = _mm_mul_ps(b2, a2);
        a1 = _mm_load_ps1(srca_d + 2);
        dst1 = _mm_add_ps(dst1, tmp1), dst2 = _mm_add_ps(dst2, tmp2);
        a2 = _mm_load_ps1(srca_d + 3);
        dst3 = _mm_add_ps(dst3, tmp3), dst4 = _mm_add_ps(dst4, tmp4);
        tmp1 = _mm_mul_ps(b1, a1), tmp2 = _mm_mul_ps(b2, a1);
        tmp3 = _mm_mul_ps(b1, a2), tmp4 = _mm_mul_ps(b2, a2);
        dst5 = _mm_add_ps(dst5, tmp1), dst6 = _mm_add_ps(dst6, tmp2);
        dst7 = _mm_add_ps(dst7, tmp3), dst8 = _mm_add_ps(dst8, tmp4);
        srcb_d += C8NUM, srca_d += C4NUM;
      }

      if (bias != NULL) {
        DoBiasBlock8(bias_d, &dst1, &dst2, &dst3, &dst4, &dst5, &dst6, &dst7, &dst8);
        bias_d += C8NUM;
      }

      ActBlock8(&dst1, &dst2, &dst3, &dst4, &dst5, &dst6, &dst7, &dst8, act_type);

      if (write_mode == OutType_TileC8) {  // WriteWino
        c = dst + WinoSteps2;
        _mm_storeu_ps(dst, dst1), _mm_storeu_ps(dst + 4, dst2);
        dst += WinoSteps1;
        _mm_storeu_ps(dst, dst3), _mm_storeu_ps(dst + 4, dst4);
        dst += WinoSteps1;
        _mm_storeu_ps(dst, dst5), _mm_storeu_ps(dst + 4, dst6);
        dst += WinoSteps1;
        _mm_storeu_ps(dst, dst7), _mm_storeu_ps(dst + 4, dst8);
      } else if (write_mode == OutType_C8) {  // WriteC8
        _mm_storeu_ps(c, dst1), _mm_storeu_ps(c + 4, dst2);
        _mm_storeu_ps(c + 8, dst3), _mm_storeu_ps(c + 12, dst4);
        _mm_storeu_ps(c + 16, dst5), _mm_storeu_ps(c + 20, dst6);
        _mm_storeu_ps(c + 24, dst7), _mm_storeu_ps(c + 28, dst8);
        c += C8Steps;
      } else {
        switch (cc) {
          case 1:  // write1
            c = dst + 1;
            WriteCol1(&dst, &dst1, &dst2, &dst3, &dst4, &dst5, &dst6, &dst7, &dst8, stride, 1, r);
            break;
          case 2:  // write2
            c = dst + 2;
            WriteCol2Opt(&dst, &dst1, &dst2, &dst3, &dst4, &dst5, &dst6, &dst7, &dst8, stride, r);
            break;
          case 3:  // write3
            c = dst + 3;
            _mm_store_ss(dst, dst1);
            dst1 = _mm_shuffle_ps(dst1, dst1, _MM_SHUFFLE(0, 3, 2, 1));
            _mm_store_ss(dst + 1, dst1);
            dst1 = _mm_shuffle_ps(dst1, dst1, _MM_SHUFFLE(0, 3, 2, 1));
            _mm_store_ss(dst + 2, dst1);
            WriteCol3(&dst, &dst1, &dst2, &dst3, &dst4, &dst5, &dst6, &dst7, &dst8, stride, 3, r);
            break;
          case 4:  // write4
            c = dst + 4;
            WriteCol4(&dst, &dst1, &dst2, &dst3, &dst4, &dst5, &dst6, &dst7, &dst8, stride, 4, r);
            break;
          case 5:  // write5
            c = dst + 5;
            WriteCol5(&dst, &dst1, &dst2, &dst3, &dst4, &dst5, &dst6, &dst7, &dst8, stride, 5, r);
            break;
          case 6:  // write6
            c = dst + 6;
            WriteCol6(&dst, &dst1, &dst2, &dst3, &dst4, &dst5, &dst6, &dst7, &dst8, stride, 6, r);
            break;
          case 7:  // write7
            c = dst + 7;
            WriteCol7(&dst, &dst1, &dst2, &dst3, &dst4, &dst5, &dst6, &dst7, &dst8, stride, 7, r);
            break;
          default:  // write8
            c = dst + C8NUM;
            WriteCol8(&dst, &dst1, &dst2, &dst3, &dst4, &dst5, &dst6, &dst7, &dst8, stride, 8, r);
            break;
        }
      }
      if (cc <= C8NUM) break;  // write end
    }
    a += C4NUM * depth;
    if (write_mode == OutType_C8) c += 32;
    if (write_mode == OutType_TileC8) c = dst + WinoSteps2;
    if (write_mode == OutType_Nhwc) c = dst - col;
    if (r <= C4NUM) break;
  }
}

void MatmulFloatSse64(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int row,
                      int col, int stride, size_t writeNhwc, size_t WriteWino) {
  size_t DstWinoSteps = stride * C8NUM, WriteWinoSteps = stride * col;
  for (int col_tmp = col; col_tmp > 0; col_tmp -= C8NUM) {
    const float *srca_d = a;
    float *dst = c;
    for (int r = row; r > 0; r -= C4NUM) {
      const float *srcb_d = b;
      __m128 dst1 = _mm_setzero_ps(), dst2 = _mm_setzero_ps();
      __m128 dst3 = _mm_setzero_ps(), dst4 = _mm_setzero_ps();
      __m128 dst5 = _mm_setzero_ps(), dst6 = _mm_setzero_ps();
      __m128 dst7 = _mm_setzero_ps(), dst8 = _mm_setzero_ps();
      for (int d = 0; d < depth; d++) {
        __m128 b1 = _mm_loadu_ps(srcb_d), b2 = _mm_loadu_ps(srcb_d + 4);
        __m128 a1 = _mm_load_ps1(srca_d), a2 = _mm_load_ps1(srca_d + 1);
        __m128 tmp1 = _mm_mul_ps(b1, a1), tmp2 = _mm_mul_ps(b2, a1);
        __m128 tmp3 = _mm_mul_ps(b1, a2), tmp4 = _mm_mul_ps(b2, a2);
        a1 = _mm_load_ps1(srca_d + 2);
        dst1 = _mm_add_ps(dst1, tmp1), dst2 = _mm_add_ps(dst2, tmp2);
        a2 = _mm_load_ps1(srca_d + 3);
        dst3 = _mm_add_ps(dst3, tmp3), dst4 = _mm_add_ps(dst4, tmp4);
        tmp1 = _mm_mul_ps(b1, a1), tmp2 = _mm_mul_ps(b2, a1);
        tmp3 = _mm_mul_ps(b1, a2), tmp4 = _mm_mul_ps(b2, a2);
        dst5 = _mm_add_ps(dst5, tmp1), dst6 = _mm_add_ps(dst6, tmp2);
        dst7 = _mm_add_ps(dst7, tmp3), dst8 = _mm_add_ps(dst8, tmp4);
        srcb_d += C8NUM, srca_d += C4NUM;
      }

      if (bias != NULL) {
        DoBiasBlock8(bias, &dst1, &dst2, &dst3, &dst4, &dst5, &dst6, &dst7, &dst8);
      }

      ActBlock8(&dst1, &dst2, &dst3, &dst4, &dst5, &dst6, &dst7, &dst8, act_type);

      if (WriteWino != 0) {  // WriteWino
        _mm_storeu_ps(dst, dst1), _mm_storeu_ps(dst + 4, dst2);
        dst += WriteWinoSteps;
        _mm_storeu_ps(dst, dst3), _mm_storeu_ps(dst + 4, dst4);
        dst += WriteWinoSteps;
        _mm_storeu_ps(dst, dst5), _mm_storeu_ps(dst + 4, dst6);
        dst += WriteWinoSteps;
        _mm_storeu_ps(dst, dst7), _mm_storeu_ps(dst + 4, dst8);
        dst += WriteWinoSteps;
      } else if (writeNhwc == 0) {  // WriteC8
        _mm_storeu_ps(dst, dst1), _mm_storeu_ps(dst + 4, dst2);
        _mm_storeu_ps(dst + 8, dst3), _mm_storeu_ps(dst + 12, dst4);
        _mm_storeu_ps(dst + 16, dst5), _mm_storeu_ps(dst + 20, dst6);
        _mm_storeu_ps(dst + 24, dst7), _mm_storeu_ps(dst + 28, dst8);
        dst += 32;
        c = dst;
      } else {
        switch (col) {
          case 1:  // write1
            WriteCol1(&dst, &dst1, &dst2, &dst3, &dst4, &dst5, &dst6, &dst7, &dst8, stride, 0, r);
          case 2:  // write2
            WriteCol2(&dst, &dst1, &dst2, &dst3, &dst4, &dst5, &dst6, &dst7, &dst8, stride, r);
          case 3:  // write3
            WriteCol3(&dst, &dst1, &dst2, &dst3, &dst4, &dst5, &dst6, &dst7, &dst8, stride, 0, r);
          case 4:  // write4
            WriteCol4(&dst, &dst1, &dst2, &dst3, &dst4, &dst5, &dst6, &dst7, &dst8, stride, 0, r);
          case 5:  // // write
            WriteCol5(&dst, &dst1, &dst2, &dst3, &dst4, &dst5, &dst6, &dst7, &dst8, stride, 0, r);
          case 6:  // write6
            WriteCol6(&dst, &dst1, &dst2, &dst3, &dst4, &dst5, &dst6, &dst7, &dst8, stride, 0, r);
          case 7:  // write7
            WriteCol7(&dst, &dst1, &dst2, &dst3, &dst4, &dst5, &dst6, &dst7, &dst8, stride, 0, r);
          default:  // write8
            WriteCol8(&dst, &dst1, &dst2, &dst3, &dst4, &dst5, &dst6, &dst7, &dst8, stride, 0, r);
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
