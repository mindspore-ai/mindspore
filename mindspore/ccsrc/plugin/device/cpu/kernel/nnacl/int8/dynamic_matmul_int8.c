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

#include "nnacl/int8/dynamic_matmul_int8.h"
#include "nnacl/int8/fixed_point.h"

void DynamicMatmul4x4x16AIWI(const int8_t *a, const int8_t *b, float *out, size_t deep4, float *multi_scales,
                             float *bias, size_t row, size_t col, size_t stride, const int32_t *a_sums,
                             const int32_t *b_sums, int64_t a_zp, int64_t b_zp_sum) {
  /* *
   * row4x4-major * row4x16-major => (int8)row-major
   * support activation per-layer symmetric && weight per-layer/per-channel symmetric
   * */
  for (int r = 0; r < row; r++) {
    int64_t s2 = a_sums[r];
    for (int c = 0; c < col; c++) {
      int r4div = r / C4NUM, r4mod = r % C4NUM;
      int c16div = c / C16NUM, c16mod = c % C16NUM;
      int32_t s1 = 0;
      for (int d = 0; d < deep4; d++) {
        int d4div = d / C4NUM, d4mod = d % C4NUM;
        size_t ai = r4div * deep4 * C4NUM + d4div * C4NUM * C4NUM + r4mod * C4NUM + d4mod;
        size_t bi = c16div * deep4 * C16NUM + d4div * C4NUM * C16NUM + c16mod * C4NUM + d4mod;
        s1 += a[ai] * b[bi];
      }
      int64_t s3 = b_sums[c] * a_zp;
      int64_t s4 = a_zp * b_zp_sum;
      size_t ci = r * stride / sizeof(float) + c;
      out[ci] = multi_scales[c] * (s1 - s2 - s3 + s4);
      if (bias != NULL) {
        out[ci] += bias[c];
      }
    }
  }
  return;
}

void DynamicMatmul4x16x4AIWI(const int8_t *a, const int8_t *b, const float *bias, float *dst, int row, int col,
                             int deep, int deep16, size_t stride, int input_zp, float input_scale,
                             const float *filter_scale, const int filter_zp, bool filter_per_channel) {
  /* *
   * row4x16-major * row16x4-major => (int8)row-major
   * support activation per-layer symmetric && weight per-layer/per-channel symmetric
   * */
  for (int r = 0; r < row; r++) {
    for (int c = 0; c < col; c++) {
      int r4div = r / C4NUM, r4mod = r % C4NUM;
      int c4div = c / C4NUM, c4mod = c % C4NUM;
      int32_t value = 0;
      int32_t s0 = 0;
      int32_t s1 = 0;
      int32_t s2 = 0;
      int32_t s3 = 0;
      for (int d = 0; d < deep; d++) {
        int d16div = d / C16NUM, d16mod = d % C16NUM;
        size_t ai = r4div * deep16 * C4NUM + d16div * C4NUM * C16NUM + r4mod * C16NUM + d16mod;
        size_t bi = c4div * deep16 * C4NUM + d16div * C4NUM * C16NUM + c4mod * C16NUM + d16mod;
        s0 += a[ai] * b[bi];
        s1 += filter_zp * a[ai];
        s2 += input_zp * b[bi];
        s3 += input_zp * filter_zp;
      }
      value = s0 - s1 - s2 + s3;
      int filter_quant_index = filter_per_channel ? c : 0;
      float multi_scale = input_scale * filter_scale[filter_quant_index];
      size_t ci = r * stride + c;
      dst[ci] = multi_scale * value;
      if (bias != NULL) {
        dst[ci] += bias[c];
      }
    }
  }
  return;
}

#ifdef ENABLE_ARM64
void PackInput4x4Asm(const int8_t *src_ic, int8_t *pack_ic, size_t ic_4div, size_t input_channel) {
  size_t src_stride = input_channel;
  size_t ic_4res = input_channel - ic_4div;
  asm volatile(
    "dup v2.4s, wzr \n"

    "mov x10, %[src_ic] \n"
    "mov x11, %[pack_ic] \n"

    "mov x15, #0 \n"
    "1: \n"
    "cmp x15, %[ic_4div] \n"
    "add x15, x15, #4\n"
    "mov x12, x10 \n"
    "add x10, x10, #4\n"
    "blt 2f \n"
    "cmp %[ic_4res], #0\n"
    "beq 6f \n"
    "cmp %[ic_4res], #1\n"
    "beq 3f \n"
    "cmp %[ic_4res], #2\n"
    "beq 4f \n"
    "cmp %[ic_4res], #3\n"
    "beq 5f \n"

    "2: \n"
    "ld1 {v0.s}[0], [x12], %[src_stride]\n"
    "ld1 {v0.s}[1], [x12], %[src_stride]\n"
    "ld1 {v0.s}[2], [x12], %[src_stride]\n"
    "ld1 {v0.s}[3], [x12], %[src_stride]\n"

    "st1 {v0.16b}, [x11], #16\n"

    "b 1b \n"

    "3: \n" /* ic res 1 */
    "dup v0.4s, wzr \n"

    "ld1 {v0.b}[0],  [x12], %[src_stride]\n"
    "ld1 {v0.b}[4],  [x12], %[src_stride]\n"
    "ld1 {v0.b}[8],  [x12], %[src_stride]\n"
    "ld1 {v0.b}[12], [x12], %[src_stride]\n"

    "st1 {v0.16b}, [x11], #16\n"

    "b 6f \n"

    "4: \n" /* ic res 2 */
    "dup v0.4s, wzr \n"

    "ld1 {v0.h}[0], [x12], %[src_stride]\n"
    "ld1 {v0.h}[2], [x12], %[src_stride]\n"
    "ld1 {v0.h}[4], [x12], %[src_stride]\n"
    "ld1 {v0.h}[6], [x12], %[src_stride]\n"

    "st1 {v0.16b}, [x11], #16\n"

    "b 6f \n"

    "5: \n" /* ic res 3 */
    "dup v0.4s, wzr \n"
    "add x13, x12, #2 \n"

    "ld1 {v0.h}[0], [x12], %[src_stride]\n"
    "ld1 {v0.b}[2], [x13], %[src_stride]\n"
    "ld1 {v0.h}[2], [x12], %[src_stride]\n"
    "ld1 {v0.b}[6], [x13], %[src_stride]\n"
    "ld1 {v0.h}[4], [x12], %[src_stride]\n"
    "ld1 {v0.b}[10], [x13], %[src_stride]\n"
    "ld1 {v0.h}[6], [x12], %[src_stride]\n"
    "ld1 {v0.b}[14], [x13], %[src_stride]\n"

    "st1 {v0.16b}, [x11], #16\n"

    "b 6f \n"

    "6: \n"

    :
    : [ src_ic ] "r"(src_ic), [ pack_ic ] "r"(pack_ic), [ src_stride ] "r"(src_stride), [ ic_4div ] "r"(ic_4div),
      [ ic_4res ] "r"(ic_4res)
    : "x10", "x11", "x12", "x13", "x14", "x15", "v0", "v1", "v2", "v3");
}
#endif

void PackInput4x4(const int8_t *src_input, int8_t *packed_input, size_t input_channel, size_t plane_size) {
  int ic4 = UP_ROUND(input_channel, C4NUM);
  size_t hw_4div = plane_size / C4NUM * C4NUM;
  size_t ic_4div = input_channel / C4NUM * C4NUM;

  const int8_t *src_r = src_input;
  int8_t *pack_r = packed_input;
  /* per layer */
  for (int hwi = 0; hwi < hw_4div; hwi += C4NUM) {
    const int8_t *src_ic = src_r;
    int8_t *pack_ic = pack_r;
#ifdef ENABLE_ARM64
    PackInput4x4Asm(src_ic, pack_ic, ic_4div, input_channel);
#else
    for (int ici = 0; ici < ic_4div; ici += C4NUM) {
      for (size_t i = 0; i < C4NUM; i++) {
        pack_ic[0 + i * C4NUM] = src_ic[0 + i * input_channel];
        pack_ic[1 + i * C4NUM] = src_ic[1 + i * input_channel];
        pack_ic[2 + i * C4NUM] = src_ic[2 + i * input_channel];
        pack_ic[3 + i * C4NUM] = src_ic[3 + i * input_channel];
      }
      src_ic += C4NUM;
      pack_ic += C4NUM * C4NUM;
    }
    for (int ici = ic_4div; ici < input_channel; ici += 1) {
      for (int i = 0; i < C4NUM; i++) {
        pack_ic[i * C4NUM] = src_ic[i * input_channel];
      }
      src_ic += 1;
      pack_ic += 1;
    }

    for (int ici = input_channel; ici < ic4; ici += 1) {
      for (int i = 0; i < C4NUM; i++) {
        pack_ic[i * C4NUM] = 0;
      }
      pack_ic += 1;
    }
#endif
    src_r += input_channel * C4NUM;
    pack_r += ic4 * C4NUM;
  }

  if (hw_4div != plane_size) {
    memset(pack_r, 0, C4NUM * ic4);
    for (int hwi = hw_4div; hwi < plane_size; hwi += 1) {
      const int8_t *src_ic = src_r;
      int8_t *pack_ic = pack_r;
      for (int ici = 0; ici < ic_4div; ici += C4NUM) {
        pack_ic[0] = src_ic[0];
        pack_ic[1] = src_ic[1];
        pack_ic[2] = src_ic[2];
        pack_ic[3] = src_ic[3];
        src_ic += C4NUM;
        pack_ic += C4NUM * C4NUM;
      }
      src_r += input_channel;
      pack_r += C4NUM;
    }
  }
  return;
}

// For matmul input a transpose case
void PackInput2Col4x4(const int8_t *src_input, int8_t *packed_input, int row, int col, int row_stride) {
  const int row_tile = C4NUM;
  int row_align = UP_ROUND(row, row_tile);
  int row_div = row / row_tile * row_tile;
  const int row_res = row - row_div;

  const int col_tile = C4NUM;
  int col_div = col / col_tile * col_tile;
  const int col_res = col - col_div;

  const int8_t *src_ic = NULL;
  int8_t *packed_ic = NULL;
  for (int c = 0; c < col_div; c += C4NUM) {
    int r = 0;
    src_ic = src_input + c;
    packed_ic = packed_input + c * row_align;
#ifdef ENABLE_ARM64
    size_t row_stride_int64 = row_stride;
    asm volatile(
      "mov w10, %w[row]\n"
      "mov x11, %[src_ic]\n"
      "mov x12, %[packed_ic]\n"
      "cmp w10, wzr\n"
      "beq 1f\n"
      "2:\n"
      "subs w10, w10, #4\n"
      "ld1 {v0.s}[0], [x11], %[row_stride]\n"
      "ld1 {v1.s}[0], [x11], %[row_stride]\n"
      "ld1 {v0.s}[1], [x11], %[row_stride]\n"
      "ld1 {v1.s}[1], [x11], %[row_stride]\n"
      "zip1 v2.8b, v0.8b, v1.8b\n"
      "zip2 v3.8b, v0.8b, v1.8b\n"
      "zip1 v4.4h, v2.4h, v3.4h\n"
      "zip2 v5.4h, v2.4h, v3.4h\n"
      "st1 {v4.4h, v5.4h}, [x12], #16\n"

      "bgt 2b\n"
      "1:\n"

      :
      : [ src_ic ] "r"(src_ic), [ packed_ic ] "r"(packed_ic), [ row ] "r"(row_div), [ row_stride ] "r"(row_stride_int64)
      : "memory", "w10", "x11", "x12", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12");
    packed_ic += C4NUM * row_div;
    src_ic += row_div * row_stride;
#else
    for (; r < row_div; r += C4NUM) {
      for (int i = 0; i < row_tile; i++) {
        packed_ic[0 * row_tile + i] = src_ic[i * row_stride + 0];
        packed_ic[1 * row_tile + i] = src_ic[i * row_stride + 1];
        packed_ic[2 * row_tile + i] = src_ic[i * row_stride + 2];
        packed_ic[3 * row_tile + i] = src_ic[i * row_stride + 3];
      }
      packed_ic += C16NUM;
      src_ic += row_tile * row_stride;
    }
#endif
    for (r = 0; r < row_res; ++r) {
      for (int i = 0; i < C4NUM; ++i) {
        packed_ic[i * row_tile + r] = src_ic[r * row_stride + i];
      }
    }
  }
  if (col_res == 0) {
    return;
  }
  src_ic = src_input + col_div;
  packed_ic = packed_input + row_align * col_div;
  for (int r = 0; r < row_div; r += row_tile) {
    for (int i = 0; i < col_res; ++i) {
      packed_ic[i * row_tile + 0] = src_ic[0 * row_stride + i];
      packed_ic[i * row_tile + 1] = src_ic[1 * row_stride + i];
      packed_ic[i * row_tile + 2] = src_ic[2 * row_stride + i];
      packed_ic[i * row_tile + 3] = src_ic[3 * row_stride + i];
    }
    src_ic += row_tile * row_stride;
    packed_ic += row_tile * col_tile;
  }

  for (int r = 0; r < row_res; ++r) {
    for (int c = 0; c < col_res; ++c) {
      packed_ic[c * row_tile + r] = src_ic[r * row_stride + c];
    }
  }
}

void CalcWeightSums(const int8_t *weight, int row, int col, int32_t *dst, DataOrder order) {
  if (order == RowMajor) {
    for (int c = 0; c < col; ++c) {
      int sum = 0;
      for (int r = 0; r < row; ++r) {
        sum += weight[r * col + c];
      }
      dst[c] = sum;
    }
  } else {
    for (int c = 0; c < col; ++c) {
      int sum = 0;
      for (int r = 0; r < row; ++r) {
        sum += weight[c * row + r];
      }
      dst[c] = sum;
    }
  }
  return;
}

void CalcPartWeightSums(const int8_t *weight, int row, int stride, int cur_col, int32_t *dst, DataOrder order) {
  if (order == RowMajor) {
    for (int c = 0; c < cur_col; ++c) {
      int sum = 0;
      for (int r = 0; r < row; ++r) {
        sum += weight[r * stride + c];
      }
      dst[c] = sum;
    }
  } else {
    for (int c = 0; c < cur_col; ++c) {
      int sum = 0;
      for (int r = 0; r < row; ++r) {
        sum += weight[c * row + r];
      }
      dst[c] = sum;
    }
  }
  return;
}
