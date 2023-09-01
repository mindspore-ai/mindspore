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

#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/fp32/matmul_avx512_fp32.h"
#include "nnacl/matmul_fp32_simd.h"

#ifndef ENABLE_ARM
void MatVecMulFp32(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int col) {
  for (int ci = 0; ci < col; ci++) {
    float value = 0;
    for (int di = 0; di < depth; di++) {
      value += a[di] * b[ci * depth + di];
    }
    if (bias != NULL) value += bias[ci];
    if (act_type == ActType_Relu6) value = MSMIN(6.0f, value);
    if (act_type == ActType_Relu || act_type == ActType_Relu6) value = MSMAX(0.0f, value);
    c[ci] = value;
  }
}
#endif

void MatVecMulFp32Block8(const float *a, const float *b, float *c, const float *bias, int act_type, int depth,
                         int col) {
  int col8 = col / C8NUM * C8NUM;
  int ci = 0;
  for (; ci < col8; ci += C8NUM, c += C8NUM) {
#ifdef ENABLE_NEON
    float32x4_t value0 = vdupq_n_f32(0.0f);
    float32x4_t value1 = vdupq_n_f32(0.0f);
    for (int di = 0; di < depth; ++di, b += C8NUM) {
      value0 += vdupq_n_f32(a[di]) * vld1q_f32(b);
      value1 += vdupq_n_f32(a[di]) * vld1q_f32(b + C4NUM);
    }
    if (bias != NULL) {
      value0 += vld1q_f32(bias + ci);
      value1 += vld1q_f32(bias + ci + C4NUM);
    }
    if (act_type == ActType_Relu || act_type == ActType_Relu6) {
      value0 = vmaxq_f32(value0, vdupq_n_f32(0.0f));
      value1 = vmaxq_f32(value1, vdupq_n_f32(0.0f));
    }
    if (act_type == ActType_Relu6) {
      value0 = vminq_f32(value0, vdupq_n_f32(6.0f));
      value1 = vminq_f32(value1, vdupq_n_f32(6.0f));
    }
    vst1q_f32(c, value0);
    vst1q_f32(c + 4, value1);
#else
    float value[C8NUM] = {0};
    for (int di = 0; di < depth; ++di, b += C8NUM) {
      for (int j = 0; j < C8NUM; ++j) {
        value[j] += a[di] * b[j];
      }
    }
    for (int j = 0; j < C8NUM; ++j) {
      ADD_BIAS(value[j], bias, ci + j);
      DO_RELU(value[j], act_type);
      DO_RELU6(value[j], act_type);
    }
    memcpy(c, value, C8NUM * sizeof(float));
#endif
  }
  int res = col - col8;
  float value[C8NUM] = {0};
  for (int di = 0; di < depth; ++di, b += C8NUM) {
    for (int j = 0; j < res; ++j) {
      value[j] += a[di] * b[j];
    }
  }
  for (int j = 0; j < res; ++j) {
    ADD_BIAS(value[j], bias, ci + j);
    DO_RELU(value[j], act_type);
    DO_RELU6(value[j], act_type);
  }
  memcpy(c, value, res * sizeof(float));
}

#ifdef ENABLE_ARM32
void MatVecMulFp32Block4(const float *a, const float *b, float *c, const float *bias, int act_type, int depth,
                         int col) {
  int col4 = col / C4NUM * C4NUM;
  int ci = 0;
  for (; ci < col4; ci += C4NUM, c += C4NUM) {
#ifdef ENABLE_NEON
    float32x4_t value = vdupq_n_f32(0.0f);
    for (int di = 0; di < depth; ++di, b += C4NUM) {
      value += vdupq_n_f32(a[di]) * vld1q_f32(b);
    }
    if (bias != NULL) {
      value += vld1q_f32(&(bias[ci]));
    }
    if (act_type == ActType_Relu || act_type == ActType_Relu6) {
      value = vmaxq_f32(value, vdupq_n_f32(0.0f));
    }
    if (act_type == ActType_Relu6) {
      value = vminq_f32(value, vdupq_n_f32(6.0f));
    }
    vst1q_f32(c, value);
#else
    float value[C4NUM] = {0};
    for (int di = 0; di < depth; ++di, b += C4NUM) {
      for (int j = 0; j < C4NUM; ++j) {
        value[j] += a[di] * b[j];
      }
    }
    for (int j = 0; j < C4NUM; ++j) {
      ADD_BIAS(value[j], bias, ci + j);
      DO_RELU(value[j], act_type);
      DO_RELU6(value[j], act_type);
    }
    memcpy(c, value, C4NUM * sizeof(float));
#endif
  }
  int res = col - col4;
  float value[C4NUM] = {0};
  for (int di = 0; di < depth; ++di, b += C4NUM) {
    for (int j = 0; j < res; ++j) {
      value[j] += a[di] * b[j];
    }
  }
  for (int j = 0; j < res; ++j) {
    ADD_BIAS(value[j], bias, ci + j);
    DO_RELU(value[j], act_type);
    DO_RELU6(value[j], act_type);
  }
  memcpy(c, value, res * sizeof(float));
}
#endif

#ifdef ENABLE_ARM64
// 4x8
void MatVecMulFp32Neon64(const float *a, const float *b, float *c, const float *bias, int act_type, int depth, int col,
                         int align_col) {
  int ci = 0;
  for (; ci < align_col - C8NUM + 1; ci += C8NUM) {
    float32x4_t acc_0;
    float32x4_t acc_1;
    if (bias != NULL) {
      acc_0 = vld1q_f32(bias + ci);
      acc_1 = vld1q_f32(bias + ci + C4NUM);
    } else {
      acc_0 = vdupq_n_f32(0.0f);
      acc_1 = vdupq_n_f32(0.0f);
    }
    const float *bv_base = b + ci * depth;
    int di = 0;
    for (; di < depth - C4NUM + 1; di += C4NUM) {
      float32x4_t av = vld1q_f32(a + di);
      float32x4_t bv_00 = vld1q_f32(bv_base);
      float32x4_t bv_10 = vld1q_f32(bv_base + C4NUM);
      bv_base += C8NUM;
      float32x4_t bv_01 = vld1q_f32(bv_base);
      float32x4_t bv_11 = vld1q_f32(bv_base + C4NUM);
      bv_base += C8NUM;
      float32x4_t bv_02 = vld1q_f32(bv_base);
      float32x4_t bv_12 = vld1q_f32(bv_base + C4NUM);
      bv_base += C8NUM;
      float32x4_t bv_03 = vld1q_f32(bv_base);
      float32x4_t bv_13 = vld1q_f32(bv_base + C4NUM);
      bv_base += C8NUM;
      acc_0 = vmlaq_n_f32(acc_0, bv_00, av[0]);
      acc_1 = vmlaq_n_f32(acc_1, bv_10, av[0]);
      acc_0 = vmlaq_n_f32(acc_0, bv_01, av[1]);
      acc_1 = vmlaq_n_f32(acc_1, bv_11, av[1]);
      acc_0 = vmlaq_n_f32(acc_0, bv_02, av[2]);
      acc_1 = vmlaq_n_f32(acc_1, bv_12, av[2]);
      acc_0 = vmlaq_n_f32(acc_0, bv_03, av[3]);
      acc_1 = vmlaq_n_f32(acc_1, bv_13, av[3]);
    }
    if (di < depth) {
      for (; di < depth; ++di) {
        float ai = a[di];
        float32x4_t bv0 = vld1q_f32(bv_base);
        float32x4_t bv1 = vld1q_f32(bv_base + C4NUM);
        acc_0 = vmlaq_n_f32(acc_0, bv0, ai);
        acc_1 = vmlaq_n_f32(acc_1, bv1, ai);
        bv_base += C8NUM;
      }
    }  // only save actual col num data
    if (ci + C4NUM - 1 >= col) {
      int c_remain = col - ci;
      for (int i = 0; i < c_remain; ++i) {
        if (act_type == ActType_Relu) {
          c[i] = MSMAX(acc_0[i], 0.0f);
        } else if (act_type == ActType_Relu6) {
          c[i] = MSMIN(MSMAX(acc_0[i], 0.0f), 6.0f);
        } else {
          c[i] = acc_0[i];
        }
      }
      return;
    }
    if (act_type == ActType_Relu) {
      acc_0 = vmaxq_f32(acc_0, vdupq_n_f32(0.0f));
    } else if (act_type == ActType_Relu6) {
      acc_0 = vminq_f32(vmaxq_f32(acc_0, vdupq_n_f32(0.0f)), vdupq_n_f32(6.0f));
    }
    vst1q_f32(c, acc_0);
    if (ci + C8NUM - 1 >= col) {
      int c_remain = col - ci - C4NUM;
      for (int i = 0; i < c_remain; ++i) {
        if (act_type == ActType_Relu) {
          c[C4NUM + i] = MSMAX(acc_1[i], 0.0f);
        } else if (act_type == ActType_Relu6) {
          c[C4NUM + i] = MSMIN(MSMAX(acc_1[i], 0.0f), 6.0f);
        } else {
          c[C4NUM + i] = acc_1[i];
        }
      }
      return;
    }
    if (act_type == ActType_Relu) {
      acc_1 = vmaxq_f32(acc_1, vdupq_n_f32(0.0f));
    } else if (act_type == ActType_Relu6) {
      acc_1 = vminq_f32(vmaxq_f32(acc_1, vdupq_n_f32(0.0f)), vdupq_n_f32(6.0f));
    }
    vst1q_f32(c + C4NUM, acc_1);
    c += C8NUM;
  }
}
#endif

void MatMul12x8(const float *a, const float *b, float *dst, const float *bias, ActType act_type, int deep, int row,
                int col, int stride, int out_type) {
  if (out_type == OutType_Nhwc) {
    for (int r = 0; r < row; r++) {
      for (int c = 0; c < col; c++) {
        int r12div = r / 12, r12mod = r % 12;
        int c8div = c / 8, c8mod = c % 8;
        size_t ci = r * stride + c;
        float value = 0;
        for (int d = 0; d < deep; d++) {
          size_t ai = r12div * deep * 12 + d * 12 + r12mod;
          size_t bi = c8div * deep * 8 + d * 8 + c8mod;
          value = value + a[ai] * b[bi];
        }
        ADD_BIAS(value, bias, c)
        DO_RELU(value, act_type)
        DO_RELU6(value, act_type)
        dst[ci] = value;
      }
    }
  } else if (out_type == OutType_C8) {
    int col_8 = UP_ROUND(col, C8NUM);
    int row_12 = UP_ROUND(row, C12NUM);
    for (int r = 0; r < row_12; r++) {
      for (int c = 0; c < col_8; c++) {
        int r12div = r / C12NUM, r12mod = r % C12NUM;
        int c8div = c / C8NUM, c8mod = c % C8NUM;
        size_t ci = (c8div * C8NUM * row_12 + r * C8NUM + c8mod);
        float value = 0;
        for (int d = 0; d < deep; d++) {
          size_t ai = r12div * deep * C12NUM + d * C12NUM + r12mod;
          size_t bi = c8div * deep * C8NUM + d * C8NUM + c8mod;
          value = value + a[ai] * b[bi];
        }
        ADD_BIAS(value, bias, c)
        DO_RELU(value, act_type)
        DO_RELU6(value, act_type)
        dst[ci] = value;
      }
    }
  } else if (out_type == OutType_TileC8) {
    for (int i = 0; i < row; ++i) {
      int src_r_offset = i;
      int dst_r_offset = i * col * stride;
      for (int j = 0; j < col; ++j) {
        int c8div = j / 8, c8mod = j % 8;
        size_t ci = dst_r_offset + c8div * 8 * stride + c8mod;
        float value = 0;
        for (int d = 0; d < deep; ++d) {
          size_t ai = src_r_offset + d * C12NUM;
          size_t bi = c8div * deep * 8 + d * 8 + c8mod;
          value = value + a[ai] * b[bi];
        }
        ADD_BIAS(value, bias, j)
        DO_RELU(value, act_type)
        DO_RELU6(value, act_type)
        dst[ci] = value;
      }
    }
  }
}

void MatMulOpt(const float *a, const float *b, float *c, const float *bias, ActType act_type, int deep, int row,
               int col, size_t stride, int out_type) {
#ifdef ENABLE_ARM64
  if (out_type == OutType_C8) {
    MatmulFloatNeon64(a, b, c, bias, (int)act_type, deep, row, col, stride, 0, 0);
  } else if (out_type == OutType_Nhwc && deep > C512NUM) {
    BigMatmulFloatNeon64Opt(a, b, c, bias, (int)act_type, deep, row, col, stride);
  } else {
    MatmulFloatNeon64Opt(a, b, c, bias, (int)act_type, deep, row, col, stride, (int)(out_type));
  }
#elif ENABLE_ARM32
  if (out_type == OutType_C8) {
    MatmulFloatNeon32(a, b, c, bias, (int)act_type, deep, row, col, stride, 0, 0);
  } else if (out_type == OutType_Nhwc) {
    MatmulFloatNeon32Opt12x4(a, b, c, bias, (int)act_type, deep, row, col, stride, 1);
  } else {
    MatmulFloatNeon32Opt(a, b, c, bias, (int)act_type, deep, row, col, stride, (int)(out_type));
  }
#elif ENABLE_AVX
  MatmulFloatAvxOpt(a, b, c, bias, (size_t)act_type, deep, row, col, stride, (size_t)(out_type));
#elif ENABLE_SSE
  MatmulFloatSse64Opt(a, b, c, bias, (int)act_type, deep, row, col, stride, (int)(out_type));
#else
  MatMul12x8(a, b, c, bias, act_type, deep, row, col, stride, out_type);
#endif
}

#define ActCompute(bit_num, down_threshold, up_threshold) \
  if (act_type != 0) {                                    \
    dst = MS_MAX##bit_num##_F32(dst, down_threshold);     \
    if (act_type == 3) {                                  \
      dst = MS_MIN##bit_num##_F32(dst, up_threshold);     \
    }                                                     \
  }

// act_type must be 0, 1, 2. 0: no_act, 1: relu, 3: relu6.
void GemmIsNotPack(const float *a, const float *b, float *c, const float *bias, int row, int deep, int act_type) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(GemmIsNotPack, index, a, b, c, bias, row, deep, act_type);

  for (; index < row; ++index) {
    float dst = a[index] * b[0] + bias[0];
    ActCompute(32, 0, C6NUM);
    c[index] = dst;
  }
}

// act_type must be 0, 1, 2. 0: no_act, 1: relu, 3: relu6.
void Row1Deep1GemmIsNotPack(const float *a, const float *b, float *c, const float *bias, int col, int deep,
                            int act_type) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(Row1Deep1GemmIsNotPack, index, a, b, c, bias, col, act_type);
  for (; index < col; ++index) {
    float dst = a[0] * b[index] + bias[index];
    ActCompute(32, 0, C6NUM);
    c[index] = dst;
  }
}

// act_type must be 0, 1, 2. 0: no_act, 1: relu, 3: relu6.
void Row1Deep1NoBiasGemmIsNotPack(const float *a, const float *b, float *c, const float *bias, int col, int deep,
                                  int act_type) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(Row1Deep1NoBiasGemmIsNotPack, index, a, b, c, bias, col, act_type);
  for (; index < col; ++index) {
    float dst = a[0] * b[index];
    ActCompute(32, 0, C6NUM);
    c[index] = dst;
  }
}

// act_type must be 0, 1, 2. 0: no_act, 1: relu, 3: relu6.
void GemmIsNotPackOptimize(const float *a, const float *b, float *c, const float *bias, int m, int k, int act_type) {
  // gemm dot is [m, k] * [k, 1] ==>> [m, 1]
  int m_index = 0;

  SIMD_RUN_AVX512(GemmIsNotPackOptimize, m_index, a, b, c, bias, m, k, act_type);

#ifdef ENABLE_AVX
  // block 4
  MS_FLOAT32X4 down_threshold128 = MS_MOVQ_F32(0);
  MS_FLOAT32X4 up_threshold128 = MS_MOVQ_F32(C6NUM);
  for (; m_index <= m - C4NUM; m_index += C4NUM) {
    int k_index = 0;
    MS_FLOAT32X4 dst = MS_MOV128_F32(bias[0]);
    MS_SET_ZERO256X4_F32(dst_)
    for (; k_index <= k - C8NUM; k_index += C8NUM) {
      MS_FLOAT32X8 weight = MS_LD256_F32(b + k_index);
      MS_LOAD256X4_F32(src, a + m_index * k + k_index, k);
      MS_FMADD256X4_F32(src, weight, dst_);
    }
    MS_F32X8_GETI(dst, 0) += MS_REDUCE_ADD256_F32(dst_1);
    MS_F32X8_GETI(dst, 1) += MS_REDUCE_ADD256_F32(dst_2);
    MS_F32X8_GETI(dst, C2NUM) += MS_REDUCE_ADD256_F32(dst_3);
    MS_F32X8_GETI(dst, C3NUM) += MS_REDUCE_ADD256_F32(dst_4);
    for (; k_index < k; ++k_index) {
      MS_F32X8_GETI(dst, 0) += b[k_index] * a[m_index * k + k_index];
      MS_F32X8_GETI(dst, 1) += b[k_index] * a[m_index * k + k_index + k];
      MS_F32X8_GETI(dst, C2NUM) += b[k_index] * a[m_index * k + k_index + C2NUM * k];
      MS_F32X8_GETI(dst, C3NUM) += b[k_index] * a[m_index * k + k_index + C3NUM * k];
    }
    ActCompute(128, down_threshold128, up_threshold128);
    MS_ST128_F32(c + m_index, dst);
  }
#endif

  // block 1
  for (; m_index < m; m_index++) {
    float dst = bias[0];
    int k_index = 0;

    SIMD_RUN_AVX512(GemmIsNotPackOptimizeCore, k_index, a + m_index * k, b, k, &dst);
    SIMD_RUN_AVX(GemmIsNotPackOptimizeCore, k_index, a + m_index * k, b, k, &dst);

    for (; k_index < k; k_index++) {
      dst += b[k_index] * a[m_index * k + k_index];
    }
    ActCompute(32, 0, C6NUM);
    c[m_index] = dst;
  }
}

void MatVecMulNoPackFp32(const float *a, const float *b, float *c, const float *bias, int act_type, int64_t depth,
                         int64_t cur_col, int64_t col) {
  int inc_flag = 0;
  int64_t k = 0;
  for (; k <= depth - C1500NUM; k += C1500NUM) {
    inc_flag = (k == 0) + (k + C1500NUM == depth ? C2NUM : 0);
    int64_t oc_index = 0;
    SIMD_RUN_NO_SCALAR(MatVecMulNoPackCore, oc_index, a, b, c, bias, act_type, C1500NUM, cur_col, col, inc_flag);
    for (; oc_index < cur_col; ++oc_index) {
      float dst = (inc_flag & 1) == 0 ? c[oc_index] : (bias == NULL ? 0 : bias[oc_index]);
      for (int64_t k_index = 0; k_index < k; ++k_index) {
        dst += a[k_index] * b[oc_index + k_index * col];
      }
      if ((inc_flag & 0x2) != 0) {
        ActCompute(32, 0, C6NUM);
      }
      c[oc_index] = dst;
    }
    a += C1500NUM;
    b += C1500NUM * col;
  }
  if (k == depth) {
    return;
  }
  inc_flag = (k == 0) + C2NUM;
  int64_t oc_index = 0;
  SIMD_RUN_NO_SCALAR(MatVecMulNoPackCore, oc_index, a, b, c, bias, act_type, depth - k, cur_col, col, inc_flag);
  for (; oc_index < cur_col; ++oc_index) {
    float dst = (inc_flag & 1) == 0 ? c[oc_index] : (bias == NULL ? 0 : bias[oc_index]);
    for (int64_t k_index = 0; k_index < depth; ++k_index) {
      dst += a[k_index] * b[oc_index + k_index * col];
    }
    ActCompute(32, 0, C6NUM);
    c[oc_index] = dst;
  }
}

#ifdef ENABLE_ARM64
// act_type must be 0, 1, 2. 0: no_act, 1: relu, 3: relu6.
void MatMul4x1Kernel(const float *input, const float *weight, float *output, const float *bias, size_t deep,
                     size_t act_type) {
  // 1: LoopD16, 2: LoopD12, 3: LoopD8, 4: LoopD4, 5: LoopD1, 6: LoopDEnd, 7: LoopDTail, 8: LoopDTailCompute
  // 9: WriteBack
  asm volatile(
    "mov x8, %[input]\n"
    "mov x9, %[weight]\n"
    "mov x10, %[deep]\n"
    "add x5, %[input], %[deep], LSL #2\n"
    "add x6, %[input], %[deep], LSL #3\n"
    "add x7, x5, %[deep], LSL #3\n"
    "dup v0.2d, xzr\n"
    "dup v1.2d, xzr\n"
    "dup v2.2d, xzr\n"
    "dup v3.2d, xzr\n"
    "subs x10, x10, #16\n"
    "blt 2f\n"
    "1:\n"  // LoopD16
    "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x8], #64\n"
    "ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x5], #64\n"
    "ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x6], #64\n"
    "ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x7], #64\n"
    "ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x9], #64\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v1.4s, v16.4s, v28.4s\n"
    "fmla v2.4s, v20.4s, v28.4s\n"
    "fmla v3.4s, v24.4s, v28.4s\n"
    "fmla v0.4s, v5.4s, v29.4s\n"
    "fmla v1.4s, v17.4s, v29.4s\n"
    "fmla v2.4s, v21.4s, v29.4s\n"
    "fmla v3.4s, v25.4s, v29.4s\n"
    "fmla v0.4s, v6.4s, v30.4s\n"
    "fmla v1.4s, v18.4s, v30.4s\n"
    "fmla v2.4s, v22.4s, v30.4s\n"
    "fmla v3.4s, v26.4s, v30.4s\n"
    "fmla v0.4s, v7.4s, v31.4s\n"
    "fmla v1.4s, v19.4s, v31.4s\n"
    "fmla v2.4s, v23.4s, v31.4s\n"
    "fmla v3.4s, v27.4s, v31.4s\n"
    "subs x10, x10, #16\n"
    "bge 1b\n"
    "2:\n"  // LoopD12
    "adds x10, x10, #16\n"
    "cbz x10, 6f\n"
    "cmp x10, #12\n"
    "blt 3f\n"
    "ld1 {v4.4s, v5.4s, v6.4s}, [x8], #48\n"
    "ld1 {v16.4s, v17.4s, v18.4s}, [x5], #48\n"
    "ld1 {v20.4s, v21.4s, v22.4s}, [x6], #48\n"
    "ld1 {v24.4s, v25.4s, v26.4s}, [x7], #48\n"
    "ld1 {v28.4s, v29.4s, v30.4s}, [x9], #48\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v1.4s, v16.4s, v28.4s\n"
    "fmla v2.4s, v20.4s, v28.4s\n"
    "fmla v3.4s, v24.4s, v28.4s\n"
    "fmla v0.4s, v5.4s, v29.4s\n"
    "fmla v1.4s, v17.4s, v29.4s\n"
    "fmla v2.4s, v21.4s, v29.4s\n"
    "fmla v3.4s, v25.4s, v29.4s\n"
    "fmla v0.4s, v6.4s, v30.4s\n"
    "fmla v1.4s, v18.4s, v30.4s\n"
    "fmla v2.4s, v22.4s, v30.4s\n"
    "fmla v3.4s, v26.4s, v30.4s\n"
    "sub x10, x10, #12\n"
    "b 7f\n"
    "3:\n"  // LoopD8
    "cmp x10, #8\n"
    "blt 4f\n"
    "ld1 {v4.4s, v5.4s}, [x8], #32\n"
    "ld1 {v16.4s, v17.4s}, [x5], #32\n"
    "ld1 {v20.4s, v21.4s}, [x6], #32\n"
    "ld1 {v24.4s, v25.4s}, [x7], #32\n"
    "ld1 {v28.4s, v29.4s}, [x9], #32\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v1.4s, v16.4s, v28.4s\n"
    "fmla v2.4s, v20.4s, v28.4s\n"
    "fmla v3.4s, v24.4s, v28.4s\n"
    "fmla v0.4s, v5.4s, v29.4s\n"
    "fmla v1.4s, v17.4s, v29.4s\n"
    "fmla v2.4s, v21.4s, v29.4s\n"
    "fmla v3.4s, v25.4s, v29.4s\n"
    "sub x10, x10, #8\n"
    "b 7f\n"
    "4:\n"  // LoopD4
    "cmp x10, #4\n"
    "blt 7f\n"
    "ld1 {v4.4s}, [x8], #16\n"
    "ld1 {v16.4s}, [x5], #16\n"
    "ld1 {v20.4s}, [x6], #16\n"
    "ld1 {v24.4s}, [x7], #16\n"
    "ld1 {v28.4s}, [x9], #16\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v1.4s, v16.4s, v28.4s\n"
    "fmla v2.4s, v20.4s, v28.4s\n"
    "fmla v3.4s, v24.4s, v28.4s\n"
    "sub x10, x10, #4\n"
    "7:\n"
    "cbz x10, 6f\n"
    "dup v4.2d, xzr\n"
    "dup v16.2d, xzr\n"
    "dup v20.2d, xzr\n"
    "dup v24.2d, xzr\n"
    "dup v28.2d, xzr\n"
    "subs x10, x10, #2\n"
    "blt 5f\n"
    "ld1 {v4.d}[0], [x8], #8\n"  // LoopD2
    "ld1 {v16.d}[0], [x5], #8\n"
    "ld1 {v20.d}[0], [x6], #8\n"
    "ld1 {v24.d}[0], [x7], #8\n"
    "ld1 {v28.d}[0], [x9], #8\n"
    "cbz x10, 8f\n"
    "5:\n"  // LoopD1
    "ld1 {v4.s}[2], [x8]\n"
    "ld1 {v16.s}[2], [x5]\n"
    "ld1 {v20.s}[2], [x6]\n"
    "ld1 {v24.s}[2], [x7]\n"
    "ld1 {v28.s}[2], [x9]\n"
    "8:\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v1.4s, v16.4s, v28.4s\n"
    "fmla v2.4s, v20.4s, v28.4s\n"
    "fmla v3.4s, v24.4s, v28.4s\n"
    "6:\n"
    "faddp v4.4s, v0.4s, v1.4s\n"
    "faddp v5.4s, v2.4s, v3.4s\n"
    "faddp v0.4s, v4.4s, v5.4s\n"
    "cbz %[bias], 9f\n"
    "ld1r {v1.4s}, [%[bias]]\n"
    "fadd v0.4s, v0.4s, v1.4s\n"
    "9:\n"
    "cbz %[act], 10f\n"
    "dup v1.2d, xzr\n"
    "fmax v0.4s, v0.4s, v1.4s\n"
    "cmp %[act], #3\n"
    "bne 10f\n"
    "movi v1.4s, #6\n"
    "scvtf v1.4s, v1.4s\n"
    "fmin v0.4s, v0.4s, v1.4s\n"
    "10:\n"
    "st1 {v0.4s}, [%[output]]\n"

    :
    : [ input ] "r"(input), [ weight ] "r"(weight), [ output ] "r"(output), [ bias ] "r"(bias), [ deep ] "r"(deep),
      [ act ] "r"(act_type)
    : "cc", "x5", "x6", "x7", "x8", "x9", "x10", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18",
      "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

void MatMul2x1Kernel(const float *input, const float *weight, float *output, const float *bias, size_t deep,
                     size_t act_type) {
  // 1: LoopD16, 2: LoopD12, 3: LoopD8, 4: LoopD4, 5: LoopD1, 6: LoopDEnd, 7: LoopDTail, 8: LoopDTailCompute
  // 9: WriteBack
  asm volatile(
    "mov x8, %[input]\n"
    "mov x9, %[weight]\n"
    "mov x10, %[deep]\n"
    "add x5, %[input], %[deep], LSL #2\n"
    "dup v0.2d, xzr\n"
    "dup v1.2d, xzr\n"
    "subs x10, x10, #16\n"
    "blt 2f\n"
    "1:\n"  // LoopD16
    "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x8], #64\n"
    "ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x5], #64\n"
    "ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x9], #64\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v1.4s, v16.4s, v28.4s\n"
    "fmla v0.4s, v5.4s, v29.4s\n"
    "fmla v1.4s, v17.4s, v29.4s\n"
    "fmla v0.4s, v6.4s, v30.4s\n"
    "fmla v1.4s, v18.4s, v30.4s\n"
    "fmla v0.4s, v7.4s, v31.4s\n"
    "fmla v1.4s, v19.4s, v31.4s\n"
    "subs x10, x10, #16\n"
    "bge 1b\n"
    "2:\n"  // LoopD12
    "adds x10, x10, #16\n"
    "cbz x10, 6f\n"
    "cmp x10, #12\n"
    "blt 3f\n"
    "ld1 {v4.4s, v5.4s, v6.4s}, [x8], #48\n"
    "ld1 {v16.4s, v17.4s, v18.4s}, [x5], #48\n"
    "ld1 {v28.4s, v29.4s, v30.4s}, [x9], #48\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v1.4s, v16.4s, v28.4s\n"
    "fmla v0.4s, v5.4s, v29.4s\n"
    "fmla v1.4s, v17.4s, v29.4s\n"
    "fmla v0.4s, v6.4s, v30.4s\n"
    "fmla v1.4s, v18.4s, v30.4s\n"
    "sub x10, x10, #12\n"
    "b 7f\n"
    "3:\n"  // LoopD8
    "cmp x10, #8\n"
    "blt 4f\n"
    "ld1 {v4.4s, v5.4s}, [x8], #32\n"
    "ld1 {v16.4s, v17.4s}, [x5], #32\n"
    "ld1 {v28.4s, v29.4s}, [x9], #32\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v1.4s, v16.4s, v28.4s\n"
    "fmla v0.4s, v5.4s, v29.4s\n"
    "fmla v1.4s, v17.4s, v29.4s\n"
    "sub x10, x10, #8\n"
    "b 7f\n"
    "4:\n"  // LoopD4
    "cmp x10, #4\n"
    "blt 7f\n"
    "ld1 {v4.4s}, [x8], #16\n"
    "ld1 {v16.4s}, [x5], #16\n"
    "ld1 {v28.4s}, [x9], #16\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v1.4s, v16.4s, v28.4s\n"
    "sub x10, x10, #4\n"
    "7:\n"
    "cbz x10, 6f\n"
    "dup v4.2d, xzr\n"
    "dup v16.2d, xzr\n"
    "subs x10, x10, #2\n"
    "blt 5f\n"
    "ld1 {v4.d}[0], [x8], #8\n"  // LoopD2
    "ld1 {v16.d}[0], [x5], #8\n"
    "ld1 {v28.d}[0], [x9], #8\n"
    "cbz x10, 8f\n"
    "5:\n"  // LoopD1
    "ld1 {v4.s}[2], [x8]\n"
    "ld1 {v16.s}[2], [x5]\n"
    "ld1 {v28.s}[2], [x9]\n"
    "8:\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v1.4s, v16.4s, v28.4s\n"
    "6:\n"
    "faddp v4.4s, v0.4s, v1.4s\n"
    "faddp v0.4s, v4.4s, v4.4s\n"
    "cbz %[bias], 9f\n"
    "ld1r {v1.4s}, [%[bias]]\n"
    "fadd v0.2s, v0.2s, v1.2s\n"
    "9:\n"
    "cbz %[act], 10f\n"
    "fmov d1, xzr\n"
    "fmax v0.2s, v0.2s, v1.2s\n"
    "cmp %[act], #3\n"
    "bne 10f\n"
    "movi v1.2s, #6\n"
    "scvtf v1.2s, v1.2s\n"
    "fmin v0.2s, v0.2s, v1.2s\n"
    "10:\n"
    "st1 {v0.2s}, [%[output]]\n"

    :
    : [ input ] "r"(input), [ weight ] "r"(weight), [ output ] "r"(output), [ bias ] "r"(bias), [ deep ] "r"(deep),
      [ act ] "r"(act_type)
    : "cc", "x5", "x8", "x9", "x10", "v0", "v1", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v28", "v29",
      "v30", "v31", "memory");
}

void MatMul1x1Kernel(const float *input, const float *weight, float *output, const float *bias, size_t deep,
                     size_t act_type) {
  // 1: LoopD16, 2: LoopD12, 3: LoopD8, 4: LoopD4, 5: LoopD1, 6: LoopDEnd, 7: LoopDTail, 8: LoopDTailCompute
  // 9: WriteBack
  asm volatile(
    "mov x8, %[input]\n"
    "mov x9, %[weight]\n"
    "mov x10, %[deep]\n"
    "dup v0.2d, xzr\n"
    "subs x10, x10, #16\n"
    "blt 2f\n"
    "1:\n"  // LoopD16
    "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x8], #64\n"
    "ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x9], #64\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v0.4s, v5.4s, v29.4s\n"
    "fmla v0.4s, v6.4s, v30.4s\n"
    "fmla v0.4s, v7.4s, v31.4s\n"
    "subs x10, x10, #16\n"
    "bge 1b\n"
    "2:\n"  // LoopD12
    "adds x10, x10, #16\n"
    "cbz x10, 6f\n"
    "cmp x10, #12\n"
    "blt 3f\n"
    "ld1 {v4.4s, v5.4s, v6.4s}, [x8], #48\n"
    "ld1 {v28.4s, v29.4s, v30.4s}, [x9], #48\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v0.4s, v5.4s, v29.4s\n"
    "fmla v0.4s, v6.4s, v30.4s\n"
    "sub x10, x10, #12\n"
    "b 7f\n"
    "3:\n"  // LoopD8
    "cmp x10, #8\n"
    "blt 4f\n"
    "ld1 {v4.4s, v5.4s}, [x8], #32\n"
    "ld1 {v28.4s, v29.4s}, [x9], #32\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "fmla v0.4s, v5.4s, v29.4s\n"
    "sub x10, x10, #8\n"
    "b 7f\n"
    "4:\n"  // LoopD4
    "cmp x10, #4\n"
    "blt 7f\n"
    "ld1 {v4.4s}, [x8], #16\n"
    "ld1 {v28.4s}, [x9], #16\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "sub x10, x10, #4\n"
    "7:\n"
    "cbz x10, 6f\n"
    "dup v4.2d, xzr\n"
    "subs x10, x10, #2\n"
    "blt 5f\n"
    "ld1 {v4.d}[0], [x8], #8\n"  // LoopD2
    "ld1 {v28.d}[0], [x9], #8\n"
    "cbz x10, 8f\n"
    "5:\n"  // LoopD1
    "ld1 {v4.s}[3], [x8]\n"
    "ld1 {v28.s}[3], [x9]\n"
    "8:\n"
    "fmla v0.4s, v4.4s, v28.4s\n"
    "6:\n"
    "faddp v4.4s, v0.4s, v0.4s\n"
    "faddp v0.4s, v4.4s, v4.4s\n"
    "cbz %[bias], 9f\n"
    "ld1 {v1.s}[0], [%[bias]]\n"
    "fadd s0, s0, s1\n"
    "9:\n"
    "cbz %[act], 10f\n"
    "fmov s1, wzr\n"
    "fmax s0, s0, s1\n"
    "cmp %[act], #3\n"
    "bne 10f\n"
    "mov x10, #6\n"
    "scvtf s1, x10\n"
    "fmin s0, s0, s1\n"
    "10:\n"
    "str s0, [%[output]]\n"

    :
    : [ input ] "r"(input), [ weight ] "r"(weight), [ output ] "r"(output), [ bias ] "r"(bias), [ deep ] "r"(deep),
      [ act ] "r"(act_type)
    : "cc", "x8", "x9", "x10", "v0", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v28", "v29", "v30", "v31");
}

void GemmIsNotPackByRow(const float *a, const float *b, float *c, const float *bias, int start_row, int end_row,
                        int deep, int act_type) {
  const float *input = a + start_row * deep;
  float *output = c + start_row;
  const int step = C4NUM * deep;
  for (; start_row <= end_row - C4NUM; start_row += C4NUM) {
    MatMul4x1Kernel(input, b, output, bias, deep, act_type);
    input += step;
    output += C4NUM;
  }
  for (; start_row <= end_row - C2NUM; start_row += C2NUM) {
    MatMul2x1Kernel(input, b, output, bias, deep, act_type);
    input += C2NUM * deep;
    output += C2NUM;
  }
  if (start_row == end_row - 1) {
    MatMul1x1Kernel(input, b, output, bias, deep, act_type);
  }
}
#endif
