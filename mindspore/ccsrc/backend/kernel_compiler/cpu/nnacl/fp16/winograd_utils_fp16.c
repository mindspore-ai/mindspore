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

#include "nnacl/fp16/winograd_utils_fp16.h"
#include "nnacl/fp16/matrix_fp16.h"

#define MIN_UNIT_FP16 2
#define MAX_UNIT_FP16 4

static InputTransFp16Func InputTransFp16FuncList[] = {
  NULL, NULL, NULL, NULL, InputTransform4x4UnitFp16, NULL, InputTransform6x6UnitFp16, NULL, InputTransform8x8UnitFp16};

static OutputTransFp16Func OutputTransFp16FuncList4[] = {NULL, NULL, OutputTransform4x2UnitFp16,
                                                         OutputTransform4x3UnitFp16};

static OutputTransFp16Func OutputTransFp16FuncReluList4[] = {NULL, NULL, OutputTransform4x2ReluUnitFp16,
                                                             OutputTransform4x3ReluUnitFp16};
static OutputTransFp16Func OutputTransFp16FuncRelu6List4[] = {NULL, NULL, OutputTransform4x2Relu6UnitFp16,
                                                              OutputTransform4x3Relu6UnitFp16};

static OutputTransFp16Func OutputTransFp16FuncList6[] = {NULL,
                                                         NULL,
                                                         OutputTransform6x2UnitFp16,
                                                         OutputTransform6x3UnitFp16,
                                                         OutputTransform6x4UnitFp16,
                                                         OutputTransform6x5UnitFp16};

static OutputTransFp16Func OutputTransFp16FuncReluList6[] = {NULL,
                                                             NULL,
                                                             OutputTransform6x2ReluUnitFp16,
                                                             OutputTransform6x3ReluUnitFp16,
                                                             OutputTransform6x4ReluUnitFp16,
                                                             OutputTransform6x5ReluUnitFp16};

static OutputTransFp16Func OutputTransFp16FuncRelu6List6[] = {NULL,
                                                              NULL,
                                                              OutputTransform6x2Relu6UnitFp16,
                                                              OutputTransform6x3Relu6UnitFp16,
                                                              OutputTransform6x4Relu6UnitFp16,
                                                              OutputTransform6x5Relu6UnitFp16};

static OutputTransFp16Func OutputTransFp16FuncList8[] = {NULL,
                                                         NULL,
                                                         OutputTransform8x2UnitFp16,
                                                         OutputTransform8x3UnitFp16,
                                                         OutputTransform8x4UnitFp16,
                                                         OutputTransform8x5UnitFp16,
                                                         OutputTransform8x6UnitFp16,
                                                         OutputTransform8x7UnitFp16};

static OutputTransFp16Func OutputTransFp16FuncReluList8[] = {NULL,
                                                             NULL,
                                                             OutputTransform8x2ReluUnitFp16,
                                                             OutputTransform8x3ReluUnitFp16,
                                                             OutputTransform8x4ReluUnitFp16,
                                                             OutputTransform8x5ReluUnitFp16,
                                                             OutputTransform8x6ReluUnitFp16,
                                                             OutputTransform8x7ReluUnitFp16};

static OutputTransFp16Func OutputTransFp16FuncRelu6List8[] = {NULL,
                                                              NULL,
                                                              OutputTransform8x2Relu6UnitFp16,
                                                              OutputTransform8x3Relu6UnitFp16,
                                                              OutputTransform8x4Relu6UnitFp16,
                                                              OutputTransform8x5Relu6UnitFp16,
                                                              OutputTransform8x6Relu6UnitFp16,
                                                              OutputTransform8x7Relu6UnitFp16};

InputTransFp16Func GetInputTransFp16Func(int input_unit) { return InputTransFp16FuncList[input_unit]; }

void InputTransform4x4UnitFp16(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step, int real_c) {
  int j = 0;
  if (real_c == 8) {
    float16x8_t src[16];
    float16x8_t t[16];
    float16x8_t m[16];
    Load16DataFp16;
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      t[l] = vsubq_f16(src[offset], src[2 + offset]);
      t[4 + l] = vaddq_f16(src[1 + offset], src[2 + offset]);
      t[8 + l] = vsubq_f16(src[2 + offset], src[1 + offset]);
      t[12 + l] = vsubq_f16(src[3 + offset], src[1 + offset]);
    }
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      m[l] = vsubq_f16(t[offset], t[2 + offset]);
      m[4 + l] = vaddq_f16(t[1 + offset], t[2 + offset]);
      m[8 + l] = vsubq_f16(t[2 + offset], t[1 + offset]);
      m[12 + l] = vsubq_f16(t[3 + offset], t[1 + offset]);
    }
    for (int i = 0; i < 16; i++) {
      int dst_offset = i * dst_step;
      vst1q_f16(dst_data + dst_offset, m[i]);
    }
    real_c -= 8;
  } else if (real_c < 8 && real_c >= 4) {
    float16x4_t src[16];
    float16x4_t t[16];
    float16x4_t m[16];
    Load16DataC4Fp16;
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      t[l] = vsub_f16(src[offset], src[2 + offset]);
      t[4 + l] = vadd_f16(src[1 + offset], src[2 + offset]);
      t[8 + l] = vsub_f16(src[2 + offset], src[1 + offset]);
      t[12 + l] = vsub_f16(src[3 + offset], src[1 + offset]);
    }
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      m[l] = vsub_f16(t[offset], t[2 + offset]);
      m[4 + l] = vadd_f16(t[1 + offset], t[2 + offset]);
      m[8 + l] = vsub_f16(t[2 + offset], t[1 + offset]);
      m[12 + l] = vsub_f16(t[3 + offset], t[1 + offset]);
    }
    for (int i = 0; i < 16; i++) {
      int dst_offset = i * dst_step;
      vst1_f16(dst_data + dst_offset, m[i]);
    }
    j = 4;
  }
  for (; j < real_c; ++j) {
    float16_t src[16];
    float16_t t[16];
    float16_t m[16];
    for (int k = 0; k < 16; ++k) {
      src[k] = src_data[j + k * src_step];
    }
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      t[l] = src[offset] - src[2 + offset];
      t[4 + l] = src[1 + offset] + src[2 + offset];
      t[8 + l] = src[2 + offset] - src[1 + offset];
      t[12 + l] = src[3 + offset] - src[1 + offset];
    }
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      m[l] = t[offset] - t[2 + offset];
      m[4 + l] = t[1 + offset] + t[2 + offset];
      m[8 + l] = t[2 + offset] - t[1 + offset];
      m[12 + l] = t[3 + offset] - t[1 + offset];
    }
    for (int i = 0; i < 16; i++) {
      int dst_offset = i * dst_step;
      dst_data[j + dst_offset] = m[i];
    }
  }
}

void InputTransform6x6UnitFp16(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step, int real_c) {
  int j = 0;
  if (real_c == 8) {
    float16x8_t src[36];
    float16x8_t t[36];
    float16x8_t m[36];
    Load36DataFp16;
    for (int l = 0; l < 6; ++l) {
      int offset = l * 6;
      float16x8_t tmp1 = vsubq_f16(src[3 + offset], src[1 + offset]);
      float16x8_t tmp2 = vsubq_f16(src[4 + offset], src[2 + offset]);
      t[l] = vaddq_f16(vsubq_f16(vmulq_n_f16(src[offset], 4), vmulq_n_f16(src[2 + offset], 5)), src[4 + offset]);
      t[6 + l] = vaddq_f16(vmulq_n_f16(vaddq_f16(src[1 + offset], src[2 + offset]), -4),
                           vaddq_f16(src[3 + offset], src[4 + offset]));
      t[12 + l] = vaddq_f16(vmulq_n_f16(vsubq_f16(src[1 + offset], src[2 + offset]), 4),
                            vsubq_f16(src[4 + offset], src[3 + offset]));
      t[18 + l] = vaddq_f16(vmulq_n_f16(tmp1, 2), tmp2);
      t[24 + l] = vaddq_f16(vmulq_n_f16(tmp1, -2), tmp2);
      t[30 + l] =
        vaddq_f16(vsubq_f16(vmulq_n_f16(src[1 + offset], 4), vmulq_n_f16(src[3 + offset], 5)), src[5 + offset]);
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 6;
      float16x8_t tmp1 = vsubq_f16(t[3 + offset], t[1 + offset]);
      float16x8_t tmp2 = vsubq_f16(t[4 + offset], t[2 + offset]);
      m[l] = vaddq_f16(vsubq_f16(vmulq_n_f16(t[offset], 4), vmulq_n_f16(t[2 + offset], 5)), t[4 + offset]);
      m[6 + l] =
        vaddq_f16(vmulq_n_f16(vaddq_f16(t[1 + offset], t[2 + offset]), -4), vaddq_f16(t[3 + offset], t[4 + offset]));
      m[12 + l] =
        vaddq_f16(vmulq_n_f16(vsubq_f16(t[1 + offset], t[2 + offset]), 4), vsubq_f16(t[4 + offset], t[3 + offset]));
      m[18 + l] = vaddq_f16(vmulq_n_f16(tmp1, 2), tmp2);
      m[24 + l] = vaddq_f16(vmulq_n_f16(tmp1, -2), tmp2);
      m[30 + l] = vaddq_f16(vsubq_f16(vmulq_n_f16(t[1 + offset], 4), vmulq_n_f16(t[3 + offset], 5)), t[5 + offset]);
    }
    for (int i = 0; i < 36; i++) {
      int dst_offset = i * dst_step;
      vst1q_f16(dst_data + dst_offset, m[i]);
    }
    real_c -= 8;
  } else if (real_c < 8 && real_c >= 4) {
    float16x4_t src[36];
    float16x4_t t[36];
    float16x4_t m[36];
    Load36DataC4Fp16;
    for (int l = 0; l < 6; ++l) {
      int offset = l * 6;
      float16x4_t tmp1 = vsub_f16(src[3 + offset], src[1 + offset]);
      float16x4_t tmp2 = vsub_f16(src[4 + offset], src[2 + offset]);
      t[l] = vadd_f16(vsub_f16(vmul_n_f16(src[offset], 4), vmul_n_f16(src[2 + offset], 5)), src[4 + offset]);
      t[6 + l] = vadd_f16(vmul_n_f16(vadd_f16(src[1 + offset], src[2 + offset]), -4),
                          vadd_f16(src[3 + offset], src[4 + offset]));
      t[12 + l] =
        vadd_f16(vmul_n_f16(vsub_f16(src[1 + offset], src[2 + offset]), 4), vsub_f16(src[4 + offset], src[3 + offset]));
      t[18 + l] = vadd_f16(vmul_n_f16(tmp1, 2), tmp2);
      t[24 + l] = vadd_f16(vmul_n_f16(tmp1, -2), tmp2);
      t[30 + l] = vadd_f16(vsub_f16(vmul_n_f16(src[1 + offset], 4), vmul_n_f16(src[3 + offset], 5)), src[5 + offset]);
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 6;
      float16x4_t tmp1 = vsub_f16(t[3 + offset], t[1 + offset]);
      float16x4_t tmp2 = vsub_f16(t[4 + offset], t[2 + offset]);
      m[l] = vadd_f16(vsub_f16(vmul_n_f16(t[offset], 4), vmul_n_f16(t[2 + offset], 5)), t[4 + offset]);
      m[6 + l] =
        vadd_f16(vmul_n_f16(vadd_f16(t[1 + offset], t[2 + offset]), -4), vadd_f16(t[3 + offset], t[4 + offset]));
      m[12 + l] =
        vadd_f16(vmul_n_f16(vsub_f16(t[1 + offset], t[2 + offset]), 4), vsub_f16(t[4 + offset], t[3 + offset]));
      m[18 + l] = vadd_f16(vmul_n_f16(tmp1, 2), tmp2);
      m[24 + l] = vadd_f16(vmul_n_f16(tmp1, -2), tmp2);
      m[30 + l] = vadd_f16(vsub_f16(vmul_n_f16(t[1 + offset], 4), vmul_n_f16(t[3 + offset], 5)), t[5 + offset]);
    }
    for (int i = 0; i < 36; i++) {
      int dst_offset = i * dst_step;
      vst1_f16(dst_data + dst_offset, m[i]);
    }
    j = 4;
  }
  for (; j < real_c; ++j) {
    float16_t src[36];
    float16_t t[36];
    float16_t m[36];
    for (int k = 0; k < 36; ++k) {
      src[k] = src_data[j + k * src_step];
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 6;
      float16_t tmp1 = src[3 + offset] - src[1 + offset];
      float16_t tmp2 = src[4 + offset] - src[2 + offset];
      t[l] = src[offset] * 4 - src[2 + offset] * 5 + src[4 + offset];
      t[6 + l] = (src[1 + offset] + src[2 + offset]) * -4 + (src[3 + offset] + src[4 + offset]);
      t[12 + l] = (src[1 + offset] - src[2 + offset]) * 4 + (src[4 + offset] - src[3 + offset]);
      t[18 + l] = tmp1 * 2 + tmp2;
      t[24 + l] = tmp1 * -2 + tmp2;
      t[30 + l] = src[1 + offset] * 4 - src[3 + offset] * 5 + src[5 + offset];
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 6;
      float16_t tmp1 = t[3 + offset] - t[1 + offset];
      float16_t tmp2 = t[4 + offset] - t[2 + offset];
      m[l] = t[offset] * 4 - t[2 + offset] * 5 + t[4 + offset];
      m[6 + l] = (t[1 + offset] + t[2 + offset]) * -4 + (t[3 + offset] + t[4 + offset]);
      m[12 + l] = (t[1 + offset] - t[2 + offset]) * 4 + (t[4 + offset] - t[3 + offset]);
      m[18 + l] = tmp1 * 2 + tmp2;
      m[24 + l] = tmp1 * -2 + tmp2;
      m[30 + l] = t[1 + offset] * 4 - t[3 + offset] * 5 + t[5 + offset];
    }
    for (int i = 0; i < 36; i++) {
      int dst_offset = i * dst_step;
      dst_data[j + dst_offset] = m[i];
    }
  }
}

void InputTransform8x8UnitFp16(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step, int real_c) {
  int j = 0;
  if (real_c == 8) {
    float16x8_t src[64];
    float16x8_t t[64];
    float16x8_t m[64];
    Load64DataFp16;
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = vsubq_f16(vaddq_f16(vsubq_f16(vmulq_n_f16(src[offset], 0.5625), vmulq_n_f16(src[2 + offset], 3.0625)),
                                 vmulq_n_f16(src[4 + offset], 3.5)),
                       src[6 + offset]);
      float16x8_t tmp1 = vaddq_f16(vmulq_n_f16(src[1 + offset], 1.125), vmulq_n_f16(src[5 + offset], 0.5));
      float16x8_t tmp2 = vsubq_f16(vmulq_n_f16(src[2 + offset], 2.25), vmulq_n_f16(src[4 + offset], 3.25));
      t[8 + l] = vaddq_f16(vsubq_f16(vaddq_f16(tmp1, tmp2), vmulq_n_f16(src[3 + offset], 1.625)), src[6 + offset]);
      t[16 + l] = vaddq_f16(vaddq_f16(vsubq_f16(tmp2, tmp1), vmulq_n_f16(src[3 + offset], 1.625)), src[6 + offset]);
      tmp1 = vaddq_f16(vmulq_n_f16(src[1 + offset], 0.5625), src[5 + offset]);
      tmp2 = vsubq_f16(vmulq_n_f16(src[2 + offset], 0.5625), vmulq_n_f16(src[4 + offset], 2.5));
      t[24 + l] = vaddq_f16(vsubq_f16(vaddq_f16(tmp1, tmp2), vmulq_n_f16(src[3 + offset], 2.5)), src[6 + offset]);
      t[32 + l] = vaddq_f16(vaddq_f16(vsubq_f16(tmp2, tmp1), vmulq_n_f16(src[3 + offset], 2.5)), src[6 + offset]);
      tmp1 = vaddq_f16(vmulq_n_f16(src[1 + offset], 0.375), vmulq_n_f16(src[5 + offset], 1.5));
      tmp2 = vsubq_f16(vmulq_n_f16(src[2 + offset], 0.25), vmulq_n_f16(src[4 + offset], 1.25));
      t[40 + l] = vaddq_f16(vsubq_f16(vaddq_f16(tmp1, tmp2), vmulq_n_f16(src[3 + offset], 1.875)), src[6 + offset]);
      t[48 + l] = vaddq_f16(vaddq_f16(vsubq_f16(tmp2, tmp1), vmulq_n_f16(src[3 + offset], 1.875)), src[6 + offset]);
      t[56 + l] =
        vaddq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src[1 + offset], -0.5625), vmulq_n_f16(src[3 + offset], 3.0625)),
                            vmulq_n_f16(src[5 + offset], 3.5)),
                  src[7 + offset]);
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      m[l] = vsubq_f16(vaddq_f16(vsubq_f16(vmulq_n_f16(t[offset], 0.5625), vmulq_n_f16(t[2 + offset], 3.0625)),
                                 vmulq_n_f16(t[4 + offset], 3.5)),
                       t[6 + offset]);
      float16x8_t tmp1 = vaddq_f16(vmulq_n_f16(t[1 + offset], 1.125), vmulq_n_f16(t[5 + offset], 0.5));
      float16x8_t tmp2 = vsubq_f16(vmulq_n_f16(t[2 + offset], 2.25), vmulq_n_f16(t[4 + offset], 3.25));
      m[8 + l] = vaddq_f16(vsubq_f16(vaddq_f16(tmp1, tmp2), vmulq_n_f16(t[3 + offset], 1.625)), t[6 + offset]);
      m[16 + l] = vaddq_f16(vaddq_f16(vsubq_f16(tmp2, tmp1), vmulq_n_f16(t[3 + offset], 1.625)), t[6 + offset]);
      tmp1 = vaddq_f16(vmulq_n_f16(t[1 + offset], 0.5625), t[5 + offset]);
      tmp2 = vsubq_f16(vmulq_n_f16(t[2 + offset], 0.5625), vmulq_n_f16(t[4 + offset], 2.5));
      m[24 + l] = vaddq_f16(vsubq_f16(vaddq_f16(tmp1, tmp2), vmulq_n_f16(t[3 + offset], 2.5)), t[6 + offset]);
      m[32 + l] = vaddq_f16(vaddq_f16(vsubq_f16(tmp2, tmp1), vmulq_n_f16(t[3 + offset], 2.5)), t[6 + offset]);
      tmp1 = vaddq_f16(vmulq_n_f16(t[1 + offset], 0.375), vmulq_n_f16(t[5 + offset], 1.5));
      tmp2 = vsubq_f16(vmulq_n_f16(t[2 + offset], 0.25), vmulq_n_f16(t[4 + offset], 1.25));
      m[40 + l] = vaddq_f16(vsubq_f16(vaddq_f16(tmp1, tmp2), vmulq_n_f16(t[3 + offset], 1.875)), t[6 + offset]);
      m[48 + l] = vaddq_f16(vaddq_f16(vsubq_f16(tmp2, tmp1), vmulq_n_f16(t[3 + offset], 1.875)), t[6 + offset]);
      m[56 + l] =
        vaddq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t[1 + offset], -0.5625), vmulq_n_f16(t[3 + offset], 3.0625)),
                            vmulq_n_f16(t[5 + offset], 3.5)),
                  t[7 + offset]);
    }
    for (int i = 0; i < 64; i++) {
      int dst_offset = i * dst_step;
      vst1q_f16(dst_data + dst_offset, m[i]);
    }
    real_c -= 8;
  } else if (real_c < 8 && real_c >= 4) {
    float16x4_t src[64];
    float16x4_t t[64];
    float16x4_t m[64];
    Load64DataC4Fp16;
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = vsub_f16(vadd_f16(vsub_f16(vmul_n_f16(src[offset], 0.5625), vmul_n_f16(src[2 + offset], 3.0625)),
                               vmul_n_f16(src[4 + offset], 3.5)),
                      src[6 + offset]);
      float16x4_t tmp1 = vadd_f16(vmul_n_f16(src[1 + offset], 1.125), vmul_n_f16(src[5 + offset], 0.5));
      float16x4_t tmp2 = vsub_f16(vmul_n_f16(src[2 + offset], 2.25), vmul_n_f16(src[4 + offset], 3.25));
      t[8 + l] = vadd_f16(vsub_f16(vadd_f16(tmp1, tmp2), vmul_n_f16(src[3 + offset], 1.625)), src[6 + offset]);
      t[16 + l] = vadd_f16(vadd_f16(vsub_f16(tmp2, tmp1), vmul_n_f16(src[3 + offset], 1.625)), src[6 + offset]);
      tmp1 = vadd_f16(vmul_n_f16(src[1 + offset], 0.5625), src[5 + offset]);
      tmp2 = vsub_f16(vmul_n_f16(src[2 + offset], 0.5625), vmul_n_f16(src[4 + offset], 2.5));
      t[24 + l] = vadd_f16(vsub_f16(vadd_f16(tmp1, tmp2), vmul_n_f16(src[3 + offset], 2.5)), src[6 + offset]);
      t[32 + l] = vadd_f16(vadd_f16(vsub_f16(tmp2, tmp1), vmul_n_f16(src[3 + offset], 2.5)), src[6 + offset]);
      tmp1 = vadd_f16(vmul_n_f16(src[1 + offset], 0.375), vmul_n_f16(src[5 + offset], 1.5));
      tmp2 = vsub_f16(vmul_n_f16(src[2 + offset], 0.25), vmul_n_f16(src[4 + offset], 1.25));
      t[40 + l] = vadd_f16(vsub_f16(vadd_f16(tmp1, tmp2), vmul_n_f16(src[3 + offset], 1.875)), src[6 + offset]);
      t[48 + l] = vadd_f16(vadd_f16(vsub_f16(tmp2, tmp1), vmul_n_f16(src[3 + offset], 1.875)), src[6 + offset]);
      t[56 + l] = vadd_f16(vsub_f16(vadd_f16(vmul_n_f16(src[1 + offset], -0.5625), vmul_n_f16(src[3 + offset], 3.0625)),
                                    vmul_n_f16(src[5 + offset], 3.5)),
                           src[7 + offset]);
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      m[l] = vsub_f16(vadd_f16(vsub_f16(vmul_n_f16(t[offset], 0.5625), vmul_n_f16(t[2 + offset], 3.0625)),
                               vmul_n_f16(t[4 + offset], 3.5)),
                      t[6 + offset]);
      float16x4_t tmp1 = vadd_f16(vmul_n_f16(t[1 + offset], 1.125), vmul_n_f16(t[5 + offset], 0.5));
      float16x4_t tmp2 = vsub_f16(vmul_n_f16(t[2 + offset], 2.25), vmul_n_f16(t[4 + offset], 3.25));
      m[8 + l] = vadd_f16(vsub_f16(vadd_f16(tmp1, tmp2), vmul_n_f16(t[3 + offset], 1.625)), t[6 + offset]);
      m[16 + l] = vadd_f16(vadd_f16(vsub_f16(tmp2, tmp1), vmul_n_f16(t[3 + offset], 1.625)), t[6 + offset]);
      tmp1 = vadd_f16(vmul_n_f16(t[1 + offset], 0.5625), t[5 + offset]);
      tmp2 = vsub_f16(vmul_n_f16(t[2 + offset], 0.5625), vmul_n_f16(t[4 + offset], 2.5));
      m[24 + l] = vadd_f16(vsub_f16(vadd_f16(tmp1, tmp2), vmul_n_f16(t[3 + offset], 2.5)), t[6 + offset]);
      m[32 + l] = vadd_f16(vadd_f16(vsub_f16(tmp2, tmp1), vmul_n_f16(t[3 + offset], 2.5)), t[6 + offset]);
      tmp1 = vadd_f16(vmul_n_f16(t[1 + offset], 0.375), vmul_n_f16(t[5 + offset], 1.5));
      tmp2 = vsub_f16(vmul_n_f16(t[2 + offset], 0.25), vmul_n_f16(t[4 + offset], 1.25));
      m[40 + l] = vadd_f16(vsub_f16(vadd_f16(tmp1, tmp2), vmul_n_f16(t[3 + offset], 1.875)), t[6 + offset]);
      m[48 + l] = vadd_f16(vadd_f16(vsub_f16(tmp2, tmp1), vmul_n_f16(t[3 + offset], 1.875)), t[6 + offset]);
      m[56 + l] = vadd_f16(vsub_f16(vadd_f16(vmul_n_f16(t[1 + offset], -0.5625), vmul_n_f16(t[3 + offset], 3.0625)),
                                    vmul_n_f16(t[5 + offset], 3.5)),
                           t[7 + offset]);
    }
    for (int i = 0; i < 64; i++) {
      int dst_offset = i * dst_step;
      vst1_f16(dst_data + dst_offset, m[i]);
    }
    j = 4;
  }
  for (; j < real_c; ++j) {
    float16_t src[64];
    float16_t t[64];
    float16_t m[64];
    for (int k = 0; k < 64; ++k) {
      src[k] = src_data[j + k * src_step];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      t[l] = src[offset] * 0.5625f - src[2 + offset] * 3.0625f + src[4 + offset] * 3.5f - src[6 + offset];
      float16_t tmp1 = src[1 + offset] * 1.125f + src[5 + offset] * 0.5f;
      float16_t tmp2 = src[2 + offset] * 2.25f - src[4 + offset] * 3.25f;
      t[8 + l] = tmp1 + tmp2 - src[3 + offset] * 1.625f + src[6 + offset];
      t[16 + l] = tmp2 - tmp1 + src[3 + offset] * 1.625f + src[6 + offset];
      tmp1 = src[1 + offset] * 0.5625f + src[5 + offset];
      tmp2 = src[2 + offset] * 0.5625f - src[4 + offset] * 2.5f;
      t[24 + l] = tmp1 + tmp2 - src[3 + offset] * 2.5f + src[6 + offset];
      t[32 + l] = tmp2 - tmp1 + src[3 + offset] * 2.5f + src[6 + offset];
      tmp1 = src[1 + offset] * 0.375f + src[5 + offset] * 1.5f;
      tmp2 = src[2 + offset] * 0.25f - src[4 + offset] * 1.25f;
      t[40 + l] = tmp1 + tmp2 - src[3 + offset] * 1.875f + src[6 + offset];
      t[48 + l] = tmp2 - tmp1 + src[3 + offset] * 1.875f + src[6 + offset];
      t[56 + l] = src[1 + offset] * -0.5625 + src[3 + offset] * 3.0625f - src[5 + offset] * 3.5f + src[7 + offset];
    }
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      m[l] = t[offset] * 0.5625f - t[2 + offset] * 3.0625f + t[4 + offset] * 3.5f - t[6 + offset];
      float16_t tmp1 = t[1 + offset] * 1.125f + t[5 + offset] * 0.5f;
      float16_t tmp2 = t[2 + offset] * 2.25f - t[4 + offset] * 3.25f;
      m[8 + l] = tmp1 + tmp2 - t[3 + offset] * 1.625f + t[6 + offset];
      m[16 + l] = tmp2 - tmp1 + t[3 + offset] * 1.625f + t[6 + offset];
      tmp1 = t[1 + offset] * 0.5625f + t[5 + offset];
      tmp2 = t[2 + offset] * 0.5625f - t[4 + offset] * 2.5f;
      m[24 + l] = tmp1 + tmp2 - t[3 + offset] * 2.5f + t[6 + offset];
      m[32 + l] = tmp2 - tmp1 + t[3 + offset] * 2.5f + t[6 + offset];
      tmp1 = t[1 + offset] * 0.375f + t[5 + offset] * 1.5f;
      tmp2 = t[2 + offset] * 0.25f - t[4 + offset] * 1.25f;
      m[40 + l] = tmp1 + tmp2 - t[3 + offset] * 1.875f + t[6 + offset];
      m[48 + l] = tmp2 - tmp1 + t[3 + offset] * 1.875f + t[6 + offset];
      m[56 + l] = t[1 + offset] * -0.5625 + t[3 + offset] * 3.0625f - t[5 + offset] * 3.5f + t[7 + offset];
    }
    for (int i = 0; i < 64; i++) {
      int dst_offset = i * dst_step;
      dst_data[j + dst_offset] = m[i];
    }
  }
}

OutputTransFp16Func GetOutputTransFp16Func(int input_unit, int output_unit, ActType act_type) {
  if (input_unit == 4 && output_unit < 4) {
    if (act_type == ActType_Relu) {
      return OutputTransFp16FuncReluList4[output_unit];
    } else if (act_type == ActType_Relu6) {
      return OutputTransFp16FuncRelu6List4[output_unit];
    } else {
      return OutputTransFp16FuncList4[output_unit];
    }
  } else if (input_unit == 6 && output_unit < 6) {
    if (act_type == ActType_Relu) {
      return OutputTransFp16FuncReluList6[output_unit];
    } else if (act_type == ActType_Relu6) {
      return OutputTransFp16FuncRelu6List6[output_unit];
    } else {
      return OutputTransFp16FuncList6[output_unit];
    }
  } else if (input_unit == 8 && output_unit < 8) {
    if (act_type == ActType_Relu) {
      return OutputTransFp16FuncReluList8[output_unit];
    } else if (act_type == ActType_Relu6) {
      return OutputTransFp16FuncRelu6List8[output_unit];
    } else {
      return OutputTransFp16FuncList8[output_unit];
    }
  } else {
    return NULL;
  }
}

void OutputTransform4x2UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  int z = 0;
  if (r_c == 8) {
    float16x8_t src[16];
    float16x8_t t[8];
    float16x8_t m[4];
    Load16DataFp16;
    float16x8_t bias_ptr = vld1q_f16(bias_data);
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      t[l] = vaddq_f16(vaddq_f16(src[offset], src[1 + offset]), src[2 + offset]);
      t[l + 4] = vaddq_f16(vsubq_f16(src[1 + offset], src[2 + offset]), src[3 + offset]);
    }
    for (int l = 0; l < 2; ++l) {
      int offset = l * 4;
      m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], t[1 + offset]), t[2 + offset]), bias_ptr);
      m[l + 2] = vaddq_f16(vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]), t[3 + offset]), bias_ptr);
    }
    if (r_h == 2 && r_w == 2) {
      Store4DataFp16;
    } else {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          vst1q_f16(dst_data + dst_k_offset + k * out_c, m[k + m_k_offset]);
        }
      }
    }
    r_c -= 8;
  } else if (r_c < 8 && r_c >= 4) {
    float16x4_t src[16];
    float16x4_t t[8];
    float16x4_t m[4];
    Load16DataC4Fp16;
    float16x4_t bias_ptr = vld1_f16(bias_data);
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      t[l] = vadd_f16(vadd_f16(src[offset], src[1 + offset]), src[2 + offset]);
      t[l + 4] = vadd_f16(vsub_f16(src[1 + offset], src[2 + offset]), src[3 + offset]);
    }
    for (int l = 0; l < 2; ++l) {
      int offset = l * 4;
      m[l] = vadd_f16(vadd_f16(vadd_f16(t[offset], t[1 + offset]), t[2 + offset]), bias_ptr);
      m[l + 2] = vadd_f16(vadd_f16(vsub_f16(t[1 + offset], t[2 + offset]), t[3 + offset]), bias_ptr);
    }
    if (r_h == 2 && r_w == 2) {
      Store4DataC4Fp16;
    } else {
      for (int i = 0; i < r_c; i++) {
        for (int j = 0; j < r_h; j++) {
          int dst_k_offset = j * dst_step * out_c;
          int m_k_offset = j * 2;
          for (int k = 0; k < r_w; k++) {
            dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
          }
        }
      }
    }
    z = 4;
  }
  for (; z < r_c; ++z) {
    float16_t src[16];
    float16_t t[8];
    float16_t m[4];
    for (int k = 0; k < 16; ++k) {
      src[k] = src_data[z + k * src_step];
    }
    float16_t bias_ptr = bias_data[z];
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset];
      t[l + 4] = src[1 + offset] - src[2 + offset] + src[3 + offset];
    }
    for (int l = 0; l < 2; ++l) {
      int offset = l * 4;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + bias_ptr;
      m[l + 2] = t[1 + offset] - t[2 + offset] + t[3 + offset] + bias_ptr;
    }
    for (int j = 0; j < r_h; j++) {
      int dst_k_offset = j * dst_step * out_c;
      int m_k_offset = j * 2;
      for (int k = 0; k < r_w; k++) {
        dst_data[z + dst_k_offset + k * out_c] = m[k + m_k_offset];
      }
    }
  }
}

void OutputTransform4x2ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  int z = 0;
  if (r_c == 8) {
    float16x8_t src[16];
    float16x8_t t[8];
    float16x8_t m[4];
    float16x8_t zero = vdupq_n_f16(0);
    Load16DataFp16;
    float16x8_t bias_ptr = vld1q_f16(bias_data);
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      t[l] = vaddq_f16(vaddq_f16(src[offset], src[1 + offset]), src[2 + offset]);
      t[l + 4] = vaddq_f16(vsubq_f16(src[1 + offset], src[2 + offset]), src[3 + offset]);
    }
    for (int l = 0; l < 2; ++l) {
      int offset = l * 4;
      m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], t[1 + offset]), t[2 + offset]), bias_ptr);
      m[l + 2] = vaddq_f16(vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]), t[3 + offset]), bias_ptr);
      m[l] = vmaxq_f16(zero, m[l]);
      m[l + 2] = vmaxq_f16(zero, m[l + 2]);
    }
    if (r_h == 2 && r_w == 2) {
      Store4DataFp16;
    } else {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          vst1q_f16(dst_data + dst_k_offset + k * out_c, m[k + m_k_offset]);
        }
      }
    }
    r_c -= 8;
  } else if (r_c < 8 && r_c >= 4) {
    float16x4_t src[16];
    float16x4_t t[8];
    float16x4_t m[4];
    float16x4_t zero = vdup_n_f16(0);
    Load16DataC4Fp16;
    float16x4_t bias_ptr = vld1_f16(bias_data);
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      t[l] = vadd_f16(vadd_f16(src[offset], src[1 + offset]), src[2 + offset]);
      t[l + 4] = vadd_f16(vsub_f16(src[1 + offset], src[2 + offset]), src[3 + offset]);
    }
    for (int l = 0; l < 2; ++l) {
      int offset = l * 4;
      m[l] = vadd_f16(vadd_f16(vadd_f16(t[offset], t[1 + offset]), t[2 + offset]), bias_ptr);
      m[l + 2] = vadd_f16(vadd_f16(vsub_f16(t[1 + offset], t[2 + offset]), t[3 + offset]), bias_ptr);
      m[l] = vmax_f16(zero, m[l]);
      m[l + 2] = vmax_f16(zero, m[l + 2]);
    }
    if (r_h == 2 && r_w == 2) {
      Store4DataC4Fp16;
    } else {
      for (int i = 0; i < r_c; i++) {
        for (int j = 0; j < r_h; j++) {
          int dst_k_offset = j * dst_step * out_c;
          int m_k_offset = j * 2;
          for (int k = 0; k < r_w; k++) {
            dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
          }
        }
      }
    }
    z = 4;
  }
  for (; z < r_c; ++z) {
    float16_t src[16];
    float16_t t[8];
    float16_t m[4];
    for (int k = 0; k < 16; ++k) {
      src[k] = src_data[z + k * src_step];
    }
    float16_t bias_ptr = bias_data[z];
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset];
      t[l + 4] = src[1 + offset] - src[2 + offset] + src[3 + offset];
    }
    for (int l = 0; l < 2; ++l) {
      int offset = l * 4;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + bias_ptr;
      m[l + 2] = t[1 + offset] - t[2 + offset] + t[3 + offset] + bias_ptr;
      m[l] = m[l] > 0 ? m[l] : 0;
      m[l + 2] = m[l + 2] > 0 ? m[l + 2] : 0;
    }
    for (int j = 0; j < r_h; j++) {
      int dst_k_offset = j * dst_step * out_c;
      int m_k_offset = j * 2;
      for (int k = 0; k < r_w; k++) {
        dst_data[z + dst_k_offset + k * out_c] = m[k + m_k_offset];
      }
    }
  }
}

void OutputTransform4x2Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  int z = 0;
  if (r_c == 8) {
    float16x8_t src[16];
    float16x8_t t[8];
    float16x8_t m[4];
    float16x8_t zero = vdupq_n_f16(0);
    float16x8_t six = vdupq_n_f16(6);
    Load16DataFp16;
    float16x8_t bias_ptr = vld1q_f16(bias_data);
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      t[l] = vaddq_f16(vaddq_f16(src[offset], src[1 + offset]), src[2 + offset]);
      t[l + 4] = vaddq_f16(vsubq_f16(src[1 + offset], src[2 + offset]), src[3 + offset]);
    }
    for (int l = 0; l < 2; ++l) {
      int offset = l * 4;
      m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], t[1 + offset]), t[2 + offset]), bias_ptr);
      m[l + 2] = vaddq_f16(vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]), t[3 + offset]), bias_ptr);
      m[l] = vmaxq_f16(zero, m[l]);
      m[l] = vminq_f16(six, m[l]);
      m[l + 2] = vmaxq_f16(zero, m[l + 2]);
      m[l + 2] = vminq_f16(six, m[l + 2]);
    }
    if (r_h == 2 && r_w == 2) {
      Store4DataFp16;
    } else {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          vst1q_f16(dst_data + dst_k_offset + k * out_c, m[k + m_k_offset]);
        }
      }
    }
    r_c -= 8;
  } else if (r_c < 8 && r_c >= 4) {
    float16x4_t src[16];
    float16x4_t t[8];
    float16x4_t m[4];
    float16x4_t zero = vdup_n_f16(0);
    float16x4_t six = vdup_n_f16(6);
    Load16DataC4Fp16;
    float16x4_t bias_ptr = vld1_f16(bias_data);
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      t[l] = vadd_f16(vadd_f16(src[offset], src[1 + offset]), src[2 + offset]);
      t[l + 4] = vadd_f16(vsub_f16(src[1 + offset], src[2 + offset]), src[3 + offset]);
    }
    for (int l = 0; l < 2; ++l) {
      int offset = l * 4;
      m[l] = vadd_f16(vadd_f16(vadd_f16(t[offset], t[1 + offset]), t[2 + offset]), bias_ptr);
      m[l + 2] = vadd_f16(vadd_f16(vsub_f16(t[1 + offset], t[2 + offset]), t[3 + offset]), bias_ptr);
      m[l] = vmax_f16(zero, m[l]);
      m[l] = vmin_f16(six, m[l]);
      m[l + 2] = vmax_f16(zero, m[l + 2]);
      m[l + 2] = vmin_f16(six, m[l + 2]);
    }
    if (r_h == 2 && r_w == 2) {
      Store4DataC4Fp16;
    } else {
      for (int i = 0; i < r_c; i++) {
        for (int j = 0; j < r_h; j++) {
          int dst_k_offset = j * dst_step * out_c;
          int m_k_offset = j * 2;
          for (int k = 0; k < r_w; k++) {
            dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
          }
        }
      }
    }
    z = 4;
  }
  for (; z < r_c; ++z) {
    float16_t src[16];
    float16_t t[8];
    float16_t m[4];
    for (int k = 0; k < 16; ++k) {
      src[k] = src_data[z + k * src_step];
    }
    float16_t bias_ptr = bias_data[z];
    for (int l = 0; l < 4; ++l) {
      int offset = l * 4;
      t[l] = src[offset] + src[1 + offset] + src[2 + offset];
      t[l + 4] = src[1 + offset] - src[2 + offset] + src[3 + offset];
    }
    for (int l = 0; l < 2; ++l) {
      int offset = l * 4;
      m[l] = t[offset] + t[1 + offset] + t[2 + offset] + bias_ptr;
      m[l + 2] = t[1 + offset] - t[2 + offset] + t[3 + offset] + bias_ptr;
      m[l] = m[l] > 0 ? m[l] : 0;
      m[l] = m[l] < 6 ? m[l] : 6;
      m[l + 2] = m[l + 2] > 0 ? m[l + 2] : 0;
      m[l + 2] = m[l + 2] < 6 ? m[l + 2] : 6;
    }
    for (int j = 0; j < r_h; j++) {
      int dst_k_offset = j * dst_step * out_c;
      int m_k_offset = j * 2;
      for (int k = 0; k < r_w; k++) {
        dst_data[z + dst_k_offset + k * out_c] = m[k + m_k_offset];
      }
    }
  }
}

void OutputTransform4x3UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[16];
  float16x8_t t[12];
  float16x8_t m[9];
  Load16DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    float16x8_t tmp = vaddq_f16(src[1 + offset], src[2 + offset]);
    t[l] = vaddq_f16(src[offset], tmp);
    t[l + 4] = vsubq_f16(src[1 + offset], src[2 + offset]);
    t[l + 8] = vaddq_f16(tmp, src[3 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 4;
    float16x8_t tmp = vaddq_f16(t[1 + offset], t[2 + offset]);
    m[l] = vaddq_f16(vaddq_f16(t[offset], tmp), bias_ptr);
    m[l + 3] = vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]), bias_ptr);
    m[l + 6] = vaddq_f16(vaddq_f16(tmp, t[3 + offset]), bias_ptr);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    Store9DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform4x3ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[16];
  float16x8_t t[12];
  float16x8_t m[9];
  float16x8_t zero = vdupq_n_f16(0);
  Load16DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    float16x8_t tmp = vaddq_f16(src[1 + offset], src[2 + offset]);
    t[l] = vaddq_f16(src[offset], tmp);
    t[l + 4] = vsubq_f16(src[1 + offset], src[2 + offset]);
    t[l + 8] = vaddq_f16(tmp, src[3 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 4;
    float16x8_t tmp = vaddq_f16(t[1 + offset], t[2 + offset]);
    m[l] = vaddq_f16(vaddq_f16(t[offset], tmp), bias_ptr);
    m[l + 3] = vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]), bias_ptr);
    m[l + 6] = vaddq_f16(vaddq_f16(tmp, t[3 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l + 3] = vmaxq_f16(zero, m[l + 3]);
    m[l + 6] = vmaxq_f16(zero, m[l + 6]);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    Store9DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform4x3Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[16];
  float16x8_t t[12];
  float16x8_t m[9];
  float16x8_t zero = vdupq_n_f16(0);
  float16x8_t six = vdupq_n_f16(6);
  Load16DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 4; ++l) {
    int offset = l * 4;
    float16x8_t tmp = vaddq_f16(src[1 + offset], src[2 + offset]);
    t[l] = vaddq_f16(src[offset], tmp);
    t[l + 4] = vsubq_f16(src[1 + offset], src[2 + offset]);
    t[l + 8] = vaddq_f16(tmp, src[3 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 4;
    float16x8_t tmp = vaddq_f16(t[1 + offset], t[2 + offset]);
    m[l] = vaddq_f16(vaddq_f16(t[offset], tmp), bias_ptr);
    m[l + 3] = vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]), bias_ptr);
    m[l + 6] = vaddq_f16(vaddq_f16(tmp, t[3 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l] = vminq_f16(six, m[l]);
    m[l + 3] = vmaxq_f16(zero, m[l + 3]);
    m[l + 3] = vminq_f16(six, m[l + 3]);
    m[l + 6] = vmaxq_f16(zero, m[l + 6]);
    m[l + 6] = vminq_f16(six, m[l + 6]);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    Store9DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x2UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[12];
  float16x8_t m[4];
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src[offset], src[1 + offset]), src[2 + offset]), src[3 + offset]),
                     src[4 + offset]);
    t[l + 6] = vaddq_f16(vaddq_f16(vsubq_f16(src[1 + offset], src[2 + offset]),
                                   vmulq_n_f16(vsubq_f16(src[3 + offset], src[4 + offset]), 2)),
                         src[5 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 6;
    m[l] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], t[1 + offset]), t[2 + offset]), t[3 + offset]), t[4 + offset]),
      bias_ptr);
    m[l + 2] = vaddq_f16(vaddq_f16(vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]),
                                             vmulq_n_f16(vsubq_f16(t[3 + offset], t[4 + offset]), 2)),
                                   t[5 + offset]),
                         bias_ptr);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    Store4DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x2ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[12];
  float16x8_t m[4];
  float16x8_t zero = vdupq_n_f16(0);
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src[offset], src[1 + offset]), src[2 + offset]), src[3 + offset]),
                     src[4 + offset]);
    t[l + 6] = vaddq_f16(vaddq_f16(vsubq_f16(src[1 + offset], src[2 + offset]),
                                   vmulq_n_f16(vsubq_f16(src[3 + offset], src[4 + offset]), 2)),
                         src[5 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 6;
    m[l] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], t[1 + offset]), t[2 + offset]), t[3 + offset]), t[4 + offset]),
      bias_ptr);
    m[l + 2] = vaddq_f16(vaddq_f16(vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]),
                                             vmulq_n_f16(vsubq_f16(t[3 + offset], t[4 + offset]), 2)),
                                   t[5 + offset]),
                         bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l + 2] = vmaxq_f16(zero, m[l + 2]);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    Store4DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x2Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[12];
  float16x8_t m[4];
  float16x8_t zero = vdupq_n_f16(0);
  float16x8_t six = vdupq_n_f16(6);
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src[offset], src[1 + offset]), src[2 + offset]), src[3 + offset]),
                     src[4 + offset]);
    t[l + 6] = vaddq_f16(vaddq_f16(vsubq_f16(src[1 + offset], src[2 + offset]),
                                   vmulq_n_f16(vsubq_f16(src[3 + offset], src[4 + offset]), 2)),
                         src[5 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 6;
    m[l] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], t[1 + offset]), t[2 + offset]), t[3 + offset]), t[4 + offset]),
      bias_ptr);
    m[l + 2] = vaddq_f16(vaddq_f16(vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]),
                                             vmulq_n_f16(vsubq_f16(t[3 + offset], t[4 + offset]), 2)),
                                   t[5 + offset]),
                         bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l] = vminq_f16(six, m[l]);
    m[l + 2] = vmaxq_f16(zero, m[l + 2]);
    m[l + 2] = vminq_f16(six, m[l + 2]);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    Store4DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x3UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[18];
  float16x8_t m[9];
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    t[l] = vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2);
    t[l + 6] = vaddq_f16(vsubq_f16(src[1 + offset], src[2 + offset]),
                         vmulq_n_f16(vsubq_f16(src[3 + offset], src[4 + offset]), 2));
    t[l + 12] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), src[5 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 3] = vaddq_f16(
      vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]), vmulq_n_f16(vsubq_f16(t[3 + offset], t[4 + offset]), 2)),
      bias_ptr);
    m[l + 6] = vaddq_f16(vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), t[5 + offset]), bias_ptr);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    Store9DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x3ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[18];
  float16x8_t m[9];
  float16x8_t zero = vdupq_n_f16(0);
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    t[l] = vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2);
    t[l + 6] = vaddq_f16(vsubq_f16(src[1 + offset], src[2 + offset]),
                         vmulq_n_f16(vsubq_f16(src[3 + offset], src[4 + offset]), 2));
    t[l + 12] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), src[5 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 3] = vaddq_f16(
      vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]), vmulq_n_f16(vsubq_f16(t[3 + offset], t[4 + offset]), 2)),
      bias_ptr);
    m[l + 6] = vaddq_f16(vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), t[5 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l + 3] = vmaxq_f16(zero, m[l + 3]);
    m[l + 6] = vmaxq_f16(zero, m[l + 6]);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    Store9DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x3Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[18];
  float16x8_t m[9];
  float16x8_t zero = vdupq_n_f16(0);
  float16x8_t six = vdupq_n_f16(6);
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    t[l] = vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2);
    t[l + 6] = vaddq_f16(vsubq_f16(src[1 + offset], src[2 + offset]),
                         vmulq_n_f16(vsubq_f16(src[3 + offset], src[4 + offset]), 2));
    t[l + 12] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), src[5 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 3] = vaddq_f16(
      vaddq_f16(vsubq_f16(t[1 + offset], t[2 + offset]), vmulq_n_f16(vsubq_f16(t[3 + offset], t[4 + offset]), 2)),
      bias_ptr);
    m[l + 6] = vaddq_f16(vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), t[5 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l] = vminq_f16(six, m[l]);
    m[l + 3] = vmaxq_f16(zero, m[l + 3]);
    m[l + 3] = vminq_f16(six, m[l + 3]);
    m[l + 6] = vmaxq_f16(zero, m[l + 6]);
    m[l + 6] = vminq_f16(six, m[l + 6]);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    Store9DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x4UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[24];
  float16x8_t m[16];
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[3 + offset], src[4 + offset]);
    t[l] = vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2);
    t[l + 6] = vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2));
    t[l + 12] = vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4));
    t[l + 18] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8)), src[5 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[3 + offset], t[4 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 4] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2)), bias_ptr);
    m[l + 8] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), bias_ptr);
    m[l + 12] = vaddq_f16(vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8)), t[5 + offset]), bias_ptr);
  }
  if (r_c == C8NUM && r_h == 4 && r_w == 4) {
    Store16DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 4;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x4ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[24];
  float16x8_t m[16];
  float16x8_t zero = vdupq_n_f16(0);
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[3 + offset], src[4 + offset]);
    t[l] = vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2);
    t[l + 6] = vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2));
    t[l + 12] = vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4));
    t[l + 18] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8)), src[5 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[3 + offset], t[4 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 4] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2)), bias_ptr);
    m[l + 8] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), bias_ptr);
    m[l + 12] = vaddq_f16(vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8)), t[5 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l + 4] = vmaxq_f16(zero, m[l + 4]);
    m[l + 8] = vmaxq_f16(zero, m[l + 8]);
    m[l + 12] = vmaxq_f16(zero, m[l + 12]);
  }
  if (r_c == C8NUM && r_h == 4 && r_w == 4) {
    Store16DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 4;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x4Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[24];
  float16x8_t m[16];
  float16x8_t zero = vdupq_n_f16(0);
  float16x8_t six = vdupq_n_f16(6);
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[3 + offset], src[4 + offset]);
    t[l] = vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2);
    t[l + 6] = vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2));
    t[l + 12] = vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4));
    t[l + 18] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8)), src[5 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[3 + offset], t[4 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 4] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2)), bias_ptr);
    m[l + 8] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), bias_ptr);
    m[l + 12] = vaddq_f16(vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8)), t[5 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l] = vminq_f16(six, m[l]);
    m[l + 4] = vmaxq_f16(zero, m[l + 4]);
    m[l + 4] = vminq_f16(six, m[l + 4]);
    m[l + 8] = vmaxq_f16(zero, m[l + 8]);
    m[l + 8] = vminq_f16(six, m[l + 8]);
    m[l + 12] = vmaxq_f16(zero, m[l + 12]);
    m[l + 12] = vminq_f16(six, m[l + 12]);
  }
  if (r_c == C8NUM && r_h == 4 && r_w == 4) {
    Store16DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 4;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x5UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[30];
  float16x8_t m[25];
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[3 + offset], src[4 + offset]);
    t[l] = vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2);
    t[l + 6] = vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2));
    t[l + 12] = vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4));
    t[l + 18] = vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8));
    t[l + 24] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 16)), src[5 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[3 + offset], t[4 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 5] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2)), bias_ptr);
    m[l + 10] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), bias_ptr);
    m[l + 15] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8)), bias_ptr);
    m[l + 20] = vaddq_f16(vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 16)), t[5 + offset]), bias_ptr);
  }
  if (r_c == C8NUM && r_h == 5 && r_w == 5) {
    Store25DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 5;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x5ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[30];
  float16x8_t m[25];
  float16x8_t zero = vdupq_n_f16(0);
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[3 + offset], src[4 + offset]);
    t[l] = vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2);
    t[l + 6] = vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2));
    t[l + 12] = vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4));
    t[l + 18] = vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8));
    t[l + 24] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 16)), src[5 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[3 + offset], t[4 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 5] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2)), bias_ptr);
    m[l + 10] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), bias_ptr);
    m[l + 15] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8)), bias_ptr);
    m[l + 20] = vaddq_f16(vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 16)), t[5 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l + 5] = vmaxq_f16(zero, m[l + 5]);
    m[l + 10] = vmaxq_f16(zero, m[l + 10]);
    m[l + 15] = vmaxq_f16(zero, m[l + 15]);
    m[l + 20] = vmaxq_f16(zero, m[l + 20]);
  }
  if (r_c == C8NUM && r_h == 5 && r_w == 5) {
    Store25DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 5;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform6x5Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[36];
  float16x8_t t[30];
  float16x8_t m[25];
  float16x8_t zero = vdupq_n_f16(0);
  float16x8_t six = vdupq_n_f16(6);
  Load36DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 6; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[3 + offset], src[4 + offset]);
    t[l] = vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2);
    t[l + 6] = vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2));
    t[l + 12] = vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4));
    t[l + 18] = vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8));
    t[l + 24] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 16)), src[5 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 6;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[3 + offset], t[4 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), bias_ptr);
    m[l + 5] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 2)), bias_ptr);
    m[l + 10] = vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 4)), bias_ptr);
    m[l + 15] = vaddq_f16(vaddq_f16(tmp3, vmulq_n_f16(tmp4, 8)), bias_ptr);
    m[l + 20] = vaddq_f16(vaddq_f16(vaddq_f16(tmp1, vmulq_n_f16(tmp2, 16)), t[5 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l] = vminq_f16(six, m[l]);
    m[l + 5] = vmaxq_f16(zero, m[l + 5]);
    m[l + 5] = vminq_f16(six, m[l + 5]);
    m[l + 10] = vmaxq_f16(zero, m[l + 10]);
    m[l + 10] = vminq_f16(six, m[l + 10]);
    m[l + 15] = vmaxq_f16(zero, m[l + 15]);
    m[l + 15] = vminq_f16(six, m[l + 15]);
    m[l + 20] = vmaxq_f16(zero, m[l + 20]);
    m[l + 20] = vminq_f16(six, m[l + 20]);
  }
  if (r_c == C8NUM && r_h == 5 && r_w == 5) {
    Store25DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 5;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x2UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[16];
  float16x8_t m[4];
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), src[7 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 2] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), t[7 + offset]), bias_ptr);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    Store4DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x2ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[16];
  float16x8_t m[4];
  float16x8_t zero = vdupq_n_f16(0);
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), src[7 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 2] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), t[7 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l + 2] = vmaxq_f16(zero, m[l + 2]);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    Store4DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x2Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[16];
  float16x8_t m[4];
  float16x8_t zero = vdupq_n_f16(0);
  float16x8_t six = vdupq_n_f16(6);
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), src[7 + offset]);
  }
  for (int l = 0; l < 2; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 2] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), t[7 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l] = vminq_f16(six, m[l]);
    m[l + 2] = vmaxq_f16(zero, m[l + 2]);
    m[l + 2] = vminq_f16(six, m[l + 2]);
  }
  if (r_c == C8NUM && r_h == 2 && r_w == 2) {
    Store4DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 2;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x3UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[24];
  float16x8_t m[9];
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), src[7 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 3] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 6] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), t[7 + offset]), bias_ptr);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    Store9DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x3ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[24];
  float16x8_t m[9];
  float16x8_t zero = vdupq_n_f16(0);
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), src[7 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 3] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 6] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), t[7 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l + 3] = vmaxq_f16(zero, m[l + 3]);
    m[l + 6] = vmaxq_f16(zero, m[l + 6]);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    Store9DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x3Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[24];
  float16x8_t m[9];
  float16x8_t zero = vdupq_n_f16(0);
  float16x8_t six = vdupq_n_f16(6);
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), src[7 + offset]);
  }
  for (int l = 0; l < 3; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 3] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 6] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), t[7 + offset]), bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l] = vminq_f16(six, m[l]);
    m[l + 3] = vmaxq_f16(zero, m[l + 3]);
    m[l + 3] = vminq_f16(six, m[l + 3]);
    m[l + 6] = vmaxq_f16(zero, m[l + 6]);
    m[l + 6] = vminq_f16(six, m[l + 6]);
  }
  if (r_c == C8NUM && r_h == 3 && r_w == 3) {
    Store9DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 3;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x4UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[32];
  float16x8_t m[16];
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
    t[l + 24] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), src[7 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 4] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 8] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
    m[l + 12] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), t[7 + offset]),
      bias_ptr);
  }
  if (r_c == C8NUM && r_h == 4 && r_w == 4) {
    Store16DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 4;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x4ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[32];
  float16x8_t m[16];
  float16x8_t zero = vdupq_n_f16(0);
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
    t[l + 24] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), src[7 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 4] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 8] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
    m[l + 12] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), t[7 + offset]),
      bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l + 4] = vmaxq_f16(zero, m[l + 4]);
    m[l + 8] = vmaxq_f16(zero, m[l + 8]);
    m[l + 12] = vmaxq_f16(zero, m[l + 12]);
  }
  if (r_c == C8NUM && r_h == 4 && r_w == 4) {
    Store16DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 4;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x4Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[32];
  float16x8_t m[16];
  float16x8_t zero = vdupq_n_f16(0);
  float16x8_t six = vdupq_n_f16(6);
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
    t[l + 24] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), src[7 + offset]);
  }
  for (int l = 0; l < 4; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 4] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 8] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
    m[l + 12] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), t[7 + offset]),
      bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l] = vminq_f16(six, m[l]);
    m[l + 4] = vmaxq_f16(zero, m[l + 4]);
    m[l + 4] = vminq_f16(six, m[l + 4]);
    m[l + 8] = vmaxq_f16(zero, m[l + 8]);
    m[l + 8] = vminq_f16(six, m[l + 8]);
    m[l + 12] = vmaxq_f16(zero, m[l + 12]);
    m[l + 12] = vminq_f16(six, m[l + 12]);
  }
  if (r_c == C8NUM && r_h == 4 && r_w == 4) {
    Store16DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 4;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x5UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[40];
  float16x8_t m[25];
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
    t[l + 24] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375));
    t[l + 32] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), src[7 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 5] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 10] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
    m[l + 15] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), bias_ptr);
    m[l + 20] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), t[7 + offset]),
      bias_ptr);
  }
  if (r_c == C8NUM && r_h == 5 && r_w == 5) {
    Store25DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 5;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x5ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[40];
  float16x8_t m[25];
  float16x8_t zero = vdupq_n_f16(0);
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
    t[l + 24] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375));
    t[l + 32] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), src[7 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 5] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 10] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
    m[l + 15] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), bias_ptr);
    m[l + 20] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), t[7 + offset]),
      bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l + 5] = vmaxq_f16(zero, m[l + 5]);
    m[l + 10] = vmaxq_f16(zero, m[l + 10]);
    m[l + 15] = vmaxq_f16(zero, m[l + 15]);
    m[l + 20] = vmaxq_f16(zero, m[l + 20]);
  }
  if (r_c == C8NUM && r_h == 5 && r_w == 5) {
    Store25DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 5;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x5Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[40];
  float16x8_t m[25];
  float16x8_t zero = vdupq_n_f16(0);
  float16x8_t six = vdupq_n_f16(6);
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
    t[l + 24] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375));
    t[l + 32] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), src[7 + offset]);
  }
  for (int l = 0; l < 5; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 5] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 10] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
    m[l + 15] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), bias_ptr);
    m[l + 20] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), t[7 + offset]),
      bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l] = vminq_f16(six, m[l]);
    m[l + 5] = vmaxq_f16(zero, m[l + 5]);
    m[l + 5] = vminq_f16(six, m[l + 5]);
    m[l + 10] = vmaxq_f16(zero, m[l + 10]);
    m[l + 10] = vminq_f16(six, m[l + 10]);
    m[l + 15] = vmaxq_f16(zero, m[l + 15]);
    m[l + 15] = vminq_f16(six, m[l + 15]);
    m[l + 20] = vmaxq_f16(zero, m[l + 20]);
    m[l + 20] = vminq_f16(six, m[l + 20]);
  }
  if (r_c == C8NUM && r_h == 5 && r_w == 5) {
    Store25DataFp16;
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 5;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  int z = 0;
  if (r_c == 8) {
    float16x8_t src[64];
    float16x8_t t[48];
    float16x8_t m[36];
    Load64DataFp16;
    float16x8_t bias_ptr = vld1q_f16(bias_data);
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
      float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
      float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
      float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
      float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
      float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
      t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
      t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
      t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
      t[l + 24] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375));
      t[l + 32] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625));
      t[l + 40] =
        vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375)), src[7 + offset]);
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 8;
      float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
      float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
      float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
      float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
      float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
      float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
      m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
      m[l + 6] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
      m[l + 12] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
      m[l + 18] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), bias_ptr);
      m[l + 24] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), bias_ptr);
      m[l + 30] = vaddq_f16(
        vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375)), t[7 + offset]),
        bias_ptr);
    }
    if (r_h == 6 && r_w == 6) {
      for (int i = 0; i < 6; i++) {
        int dst_k_offset = i * dst_step * out_c;
        int m_k_offset = i * 6;
        vst1q_f16(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
        vst1q_f16(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
        vst1q_f16(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
        vst1q_f16(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
        vst1q_f16(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
        vst1q_f16(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
      }
    } else {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 6;
        for (int k = 0; k < r_w; k++) {
          vst1q_f16(dst_data + dst_k_offset + k * out_c, m[k + m_k_offset]);
        }
      }
    }
    r_c -= 8;
  } else if (r_c < 8 && r_c >= 4) {
    float16x4_t src[64];
    float16x4_t t[48];
    float16x4_t m[36];
    Load64DataC4Fp16;
    float16x4_t bias_ptr = vld1_f16(bias_data);
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      float16x4_t tmp1 = vadd_f16(src[1 + offset], src[2 + offset]);
      float16x4_t tmp2 = vadd_f16(src[3 + offset], src[4 + offset]);
      float16x4_t tmp3 = vadd_f16(src[5 + offset], src[6 + offset]);
      float16x4_t tmp4 = vsub_f16(src[1 + offset], src[2 + offset]);
      float16x4_t tmp5 = vsub_f16(src[3 + offset], src[4 + offset]);
      float16x4_t tmp6 = vsub_f16(src[5 + offset], src[6 + offset]);
      t[l] = vadd_f16(vadd_f16(vadd_f16(src[offset], tmp1), tmp2), tmp3);
      t[l + 8] = vadd_f16(vadd_f16(vmul_n_f16(tmp4, 0.5), tmp5), vmul_n_f16(tmp6, 1.5));
      t[l + 16] = vadd_f16(vadd_f16(vmul_n_f16(tmp1, 0.25), tmp2), vmul_n_f16(tmp3, 2.25));
      t[l + 24] = vadd_f16(vadd_f16(vmul_n_f16(tmp4, 0.125), tmp5), vmul_n_f16(tmp6, 3.375));
      t[l + 32] = vadd_f16(vadd_f16(vmul_n_f16(tmp1, 0.0625), tmp2), vmul_n_f16(tmp3, 5.0625));
      t[l + 40] =
        vadd_f16(vadd_f16(vadd_f16(vmul_n_f16(tmp4, 0.03125), tmp5), vmul_n_f16(tmp6, 7.59375)), src[7 + offset]);
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 8;
      float16x4_t tmp1 = vadd_f16(t[1 + offset], t[2 + offset]);
      float16x4_t tmp2 = vadd_f16(t[3 + offset], t[4 + offset]);
      float16x4_t tmp3 = vadd_f16(t[5 + offset], t[6 + offset]);
      float16x4_t tmp4 = vsub_f16(t[1 + offset], t[2 + offset]);
      float16x4_t tmp5 = vsub_f16(t[3 + offset], t[4 + offset]);
      float16x4_t tmp6 = vsub_f16(t[5 + offset], t[6 + offset]);
      m[l] = vadd_f16(vadd_f16(vadd_f16(vadd_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
      m[l + 6] = vadd_f16(vadd_f16(vadd_f16(vmul_n_f16(tmp4, 0.5), tmp5), vmul_n_f16(tmp6, 1.5)), bias_ptr);
      m[l + 12] = vadd_f16(vadd_f16(vadd_f16(vmul_n_f16(tmp1, 0.25), tmp2), vmul_n_f16(tmp3, 2.25)), bias_ptr);
      m[l + 18] = vadd_f16(vadd_f16(vadd_f16(vmul_n_f16(tmp4, 0.125), tmp5), vmul_n_f16(tmp6, 3.375)), bias_ptr);
      m[l + 24] = vadd_f16(vadd_f16(vadd_f16(vmul_n_f16(tmp1, 0.0625), tmp2), vmul_n_f16(tmp3, 5.0625)), bias_ptr);
      m[l + 30] = vadd_f16(
        vadd_f16(vadd_f16(vadd_f16(vmul_n_f16(tmp4, 0.03125), tmp5), vmul_n_f16(tmp6, 7.59375)), t[7 + offset]),
        bias_ptr);
    }
    if (r_h == 6 && r_w == 6) {
      for (int i = 0; i < 6; i++) {
        int dst_k_offset = i * dst_step * out_c;
        int m_k_offset = i * 6;
        vst1_f16(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
        vst1_f16(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
        vst1_f16(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
        vst1_f16(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
        vst1_f16(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
        vst1_f16(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
      }
    } else {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 6;
        for (int k = 0; k < r_w; k++) {
          vst1_f16(dst_data + dst_k_offset + k * out_c, m[k + m_k_offset]);
        }
      }
    }
    z = 4;
  }
  for (; z < r_c; ++z) {
    float16_t src[64];
    float16_t t[48];
    float16_t m[36];
    for (int k = 0; k < 16; ++k) {
      src[k] = src_data[z + k * src_step];
    }
    float16_t bias_ptr = bias_data[z];
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      float16_t tmp1 = src[1 + offset] + src[2 + offset];
      float16_t tmp2 = src[3 + offset] + src[4 + offset];
      float16_t tmp3 = src[5 + offset] + src[6 + offset];
      float16_t tmp4 = src[1 + offset] - src[2 + offset];
      float16_t tmp5 = src[3 + offset] - src[4 + offset];
      float16_t tmp6 = src[5 + offset] - src[6 + offset];
      t[l] = src[offset] + tmp1 + tmp2 + tmp3;
      t[l + 8] = tmp4 * 0.5f + tmp5 + tmp6 * 1.5f;
      t[l + 16] = tmp1 * 0.25f + tmp2 + tmp3 * 2.25f;
      t[l + 24] = tmp4 * 0.125f + tmp5 + tmp6 * 3.375f;
      t[l + 32] = tmp1 * 0.0625f + tmp2 + tmp3 * 5.0625f;
      t[l + 40] = tmp4 * 0.03125f + tmp5 + tmp6 * 7.59375f + src[7 + offset];
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 8;
      float16_t tmp1 = t[1 + offset] + t[2 + offset];
      float16_t tmp2 = t[3 + offset] + t[4 + offset];
      float16_t tmp3 = t[5 + offset] + t[6 + offset];
      float16_t tmp4 = t[1 + offset] - t[2 + offset];
      float16_t tmp5 = t[3 + offset] - t[4 + offset];
      float16_t tmp6 = t[5 + offset] - t[6 + offset];
      m[l] = t[offset] + tmp1 + tmp2 + tmp3 + bias_ptr;
      m[l + 6] = tmp4 * 0.5f + tmp5 + tmp6 * 1.5f + bias_ptr;
      m[l + 12] = tmp1 * 0.25f + tmp2 + tmp3 * 2.25f + bias_ptr;
      m[l + 18] = tmp4 * 0.125f + tmp5 + tmp6 * 3.375f + bias_ptr;
      m[l + 24] = tmp1 * 0.0625f + tmp2 + tmp3 * 5.0625f + bias_ptr;
      m[l + 30] = tmp4 * 0.03125f + tmp5 + tmp6 * 7.59375f + t[7 + offset] + bias_ptr;
    }
    for (int j = 0; j < r_h; j++) {
      int dst_k_offset = j * dst_step * out_c;
      int m_k_offset = j * 6;
      for (int k = 0; k < r_w; k++) {
        dst_data[z + dst_k_offset + k * out_c] = m[k + m_k_offset];
      }
    }
  }
}

void OutputTransform8x6ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  int z = 0;
  if (r_c == 8) {
    float16x8_t src[64];
    float16x8_t t[48];
    float16x8_t m[36];
    float16x8_t zero = vdupq_n_f16(0);
    Load64DataFp16;
    float16x8_t bias_ptr = vld1q_f16(bias_data);
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
      float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
      float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
      float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
      float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
      float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
      t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
      t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
      t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
      t[l + 24] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375));
      t[l + 32] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625));
      t[l + 40] =
        vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375)), src[7 + offset]);
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 8;
      float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
      float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
      float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
      float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
      float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
      float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
      m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
      m[l + 6] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
      m[l + 12] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
      m[l + 18] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), bias_ptr);
      m[l + 24] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), bias_ptr);
      m[l + 30] = vaddq_f16(
        vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375)), t[7 + offset]),
        bias_ptr);
      m[l] = vmaxq_f16(zero, m[l]);
      m[l + 6] = vmaxq_f16(zero, m[l + 6]);
      m[l + 12] = vmaxq_f16(zero, m[l + 12]);
      m[l + 18] = vmaxq_f16(zero, m[l + 18]);
      m[l + 24] = vmaxq_f16(zero, m[l + 24]);
      m[l + 30] = vmaxq_f16(zero, m[l + 30]);
    }
    if (r_h == 6 && r_w == 6) {
      for (int i = 0; i < 6; i++) {
        int dst_k_offset = i * dst_step * out_c;
        int m_k_offset = i * 6;
        vst1q_f16(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
        vst1q_f16(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
        vst1q_f16(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
        vst1q_f16(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
        vst1q_f16(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
        vst1q_f16(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
      }
    } else {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 6;
        for (int k = 0; k < r_w; k++) {
          vst1q_f16(dst_data + dst_k_offset + k * out_c, m[k + m_k_offset]);
        }
      }
    }
    r_c -= 8;
  } else if (r_c < 8 && r_c >= 4) {
    float16x4_t src[64];
    float16x4_t t[48];
    float16x4_t m[36];
    float16x4_t zero = vdup_n_f16(0);
    Load64DataC4Fp16;
    float16x4_t bias_ptr = vld1_f16(bias_data);
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      float16x4_t tmp1 = vadd_f16(src[1 + offset], src[2 + offset]);
      float16x4_t tmp2 = vadd_f16(src[3 + offset], src[4 + offset]);
      float16x4_t tmp3 = vadd_f16(src[5 + offset], src[6 + offset]);
      float16x4_t tmp4 = vsub_f16(src[1 + offset], src[2 + offset]);
      float16x4_t tmp5 = vsub_f16(src[3 + offset], src[4 + offset]);
      float16x4_t tmp6 = vsub_f16(src[5 + offset], src[6 + offset]);
      t[l] = vadd_f16(vadd_f16(vadd_f16(src[offset], tmp1), tmp2), tmp3);
      t[l + 8] = vadd_f16(vadd_f16(vmul_n_f16(tmp4, 0.5), tmp5), vmul_n_f16(tmp6, 1.5));
      t[l + 16] = vadd_f16(vadd_f16(vmul_n_f16(tmp1, 0.25), tmp2), vmul_n_f16(tmp3, 2.25));
      t[l + 24] = vadd_f16(vadd_f16(vmul_n_f16(tmp4, 0.125), tmp5), vmul_n_f16(tmp6, 3.375));
      t[l + 32] = vadd_f16(vadd_f16(vmul_n_f16(tmp1, 0.0625), tmp2), vmul_n_f16(tmp3, 5.0625));
      t[l + 40] =
        vadd_f16(vadd_f16(vadd_f16(vmul_n_f16(tmp4, 0.03125), tmp5), vmul_n_f16(tmp6, 7.59375)), src[7 + offset]);
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 8;
      float16x4_t tmp1 = vadd_f16(t[1 + offset], t[2 + offset]);
      float16x4_t tmp2 = vadd_f16(t[3 + offset], t[4 + offset]);
      float16x4_t tmp3 = vadd_f16(t[5 + offset], t[6 + offset]);
      float16x4_t tmp4 = vsub_f16(t[1 + offset], t[2 + offset]);
      float16x4_t tmp5 = vsub_f16(t[3 + offset], t[4 + offset]);
      float16x4_t tmp6 = vsub_f16(t[5 + offset], t[6 + offset]);
      m[l] = vadd_f16(vadd_f16(vadd_f16(vadd_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
      m[l + 6] = vadd_f16(vadd_f16(vadd_f16(vmul_n_f16(tmp4, 0.5), tmp5), vmul_n_f16(tmp6, 1.5)), bias_ptr);
      m[l + 12] = vadd_f16(vadd_f16(vadd_f16(vmul_n_f16(tmp1, 0.25), tmp2), vmul_n_f16(tmp3, 2.25)), bias_ptr);
      m[l + 18] = vadd_f16(vadd_f16(vadd_f16(vmul_n_f16(tmp4, 0.125), tmp5), vmul_n_f16(tmp6, 3.375)), bias_ptr);
      m[l + 24] = vadd_f16(vadd_f16(vadd_f16(vmul_n_f16(tmp1, 0.0625), tmp2), vmul_n_f16(tmp3, 5.0625)), bias_ptr);
      m[l + 30] = vadd_f16(
        vadd_f16(vadd_f16(vadd_f16(vmul_n_f16(tmp4, 0.03125), tmp5), vmul_n_f16(tmp6, 7.59375)), t[7 + offset]),
        bias_ptr);
      m[l] = vmax_f16(zero, m[l]);
      m[l + 6] = vmax_f16(zero, m[l + 6]);
      m[l + 12] = vmax_f16(zero, m[l + 12]);
      m[l + 18] = vmax_f16(zero, m[l + 18]);
      m[l + 24] = vmax_f16(zero, m[l + 24]);
      m[l + 30] = vmax_f16(zero, m[l + 30]);
    }
    if (r_h == 6 && r_w == 6) {
      for (int i = 0; i < 6; i++) {
        int dst_k_offset = i * dst_step * out_c;
        int m_k_offset = i * 6;
        vst1_f16(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
        vst1_f16(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
        vst1_f16(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
        vst1_f16(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
        vst1_f16(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
        vst1_f16(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
      }
    } else {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 6;
        for (int k = 0; k < r_w; k++) {
          vst1_f16(dst_data + dst_k_offset + k * out_c, m[k + m_k_offset]);
        }
      }
    }
    z = 4;
  }
  for (; z < r_c; ++z) {
    float16_t src[64];
    float16_t t[48];
    float16_t m[36];
    for (int k = 0; k < 16; ++k) {
      src[k] = src_data[z + k * src_step];
    }
    float16_t bias_ptr = bias_data[z];
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      float16_t tmp1 = src[1 + offset] + src[2 + offset];
      float16_t tmp2 = src[3 + offset] + src[4 + offset];
      float16_t tmp3 = src[5 + offset] + src[6 + offset];
      float16_t tmp4 = src[1 + offset] - src[2 + offset];
      float16_t tmp5 = src[3 + offset] - src[4 + offset];
      float16_t tmp6 = src[5 + offset] - src[6 + offset];
      t[l] = src[offset] + tmp1 + tmp2 + tmp3;
      t[l + 8] = tmp4 * 0.5f + tmp5 + tmp6 * 1.5f;
      t[l + 16] = tmp1 * 0.25f + tmp2 + tmp3 * 2.25f;
      t[l + 24] = tmp4 * 0.125f + tmp5 + tmp6 * 3.375f;
      t[l + 32] = tmp1 * 0.0625f + tmp2 + tmp3 * 5.0625f;
      t[l + 40] = tmp4 * 0.03125f + tmp5 + tmp6 * 7.59375f + src[7 + offset];
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 8;
      float16_t tmp1 = t[1 + offset] + t[2 + offset];
      float16_t tmp2 = t[3 + offset] + t[4 + offset];
      float16_t tmp3 = t[5 + offset] + t[6 + offset];
      float16_t tmp4 = t[1 + offset] - t[2 + offset];
      float16_t tmp5 = t[3 + offset] - t[4 + offset];
      float16_t tmp6 = t[5 + offset] - t[6 + offset];
      m[l] = t[offset] + tmp1 + tmp2 + tmp3 + bias_ptr;
      m[l + 6] = tmp4 * 0.5f + tmp5 + tmp6 * 1.5f + bias_ptr;
      m[l + 12] = tmp1 * 0.25f + tmp2 + tmp3 * 2.25f + bias_ptr;
      m[l + 18] = tmp4 * 0.125f + tmp5 + tmp6 * 3.375f + bias_ptr;
      m[l + 24] = tmp1 * 0.0625f + tmp2 + tmp3 * 5.0625f + bias_ptr;
      m[l + 30] = tmp4 * 0.03125f + tmp5 + tmp6 * 7.59375f + t[7 + offset] + bias_ptr;
      m[l] = m[l] > 0 ? m[l] : 0;
      m[l + 6] = m[l + 6] > 0 ? m[l + 6] : 0;
      m[l + 12] = m[l + 12] > 0 ? m[l + 12] : 0;
      m[l + 18] = m[l + 18] > 0 ? m[l + 18] : 0;
      m[l + 24] = m[l + 24] > 0 ? m[l + 24] : 0;
      m[l + 30] = m[l + 30] > 0 ? m[l + 30] : 0;
    }
    for (int j = 0; j < r_h; j++) {
      int dst_k_offset = j * dst_step * out_c;
      int m_k_offset = j * 6;
      for (int k = 0; k < r_w; k++) {
        dst_data[z + dst_k_offset + k * out_c] = m[k + m_k_offset];
      }
    }
  }
}

void OutputTransform8x6Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  int z = 0;
  if (r_c == 8) {
    float16x8_t src[64];
    float16x8_t t[48];
    float16x8_t m[36];
    float16x8_t zero = vdupq_n_f16(0);
    float16x8_t six = vdupq_n_f16(6);
    Load64DataFp16;
    float16x8_t bias_ptr = vld1q_f16(bias_data);
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
      float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
      float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
      float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
      float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
      float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
      t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
      t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
      t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
      t[l + 24] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375));
      t[l + 32] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625));
      t[l + 40] =
        vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375)), src[7 + offset]);
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 8;
      float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
      float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
      float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
      float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
      float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
      float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
      m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
      m[l + 6] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
      m[l + 12] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
      m[l + 18] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), bias_ptr);
      m[l + 24] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), bias_ptr);
      m[l + 30] = vaddq_f16(
        vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375)), t[7 + offset]),
        bias_ptr);
      m[l] = vmaxq_f16(zero, m[l]);
      m[l] = vminq_f16(six, m[l]);
      m[l + 6] = vmaxq_f16(zero, m[l + 6]);
      m[l + 6] = vminq_f16(six, m[l + 6]);
      m[l + 12] = vmaxq_f16(zero, m[l + 12]);
      m[l + 12] = vminq_f16(six, m[l + 12]);
      m[l + 18] = vmaxq_f16(zero, m[l + 18]);
      m[l + 18] = vminq_f16(six, m[l + 18]);
      m[l + 24] = vmaxq_f16(zero, m[l + 24]);
      m[l + 24] = vminq_f16(six, m[l + 24]);
      m[l + 30] = vmaxq_f16(zero, m[l + 30]);
      m[l + 30] = vminq_f16(six, m[l + 30]);
    }
    if (r_h == 6 && r_w == 6) {
      for (int i = 0; i < 6; i++) {
        int dst_k_offset = i * dst_step * out_c;
        int m_k_offset = i * 6;
        vst1q_f16(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
        vst1q_f16(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
        vst1q_f16(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
        vst1q_f16(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
        vst1q_f16(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
        vst1q_f16(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
      }
    } else {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 6;
        for (int k = 0; k < r_w; k++) {
          vst1q_f16(dst_data + dst_k_offset + k * out_c, m[k + m_k_offset]);
        }
      }
    }
    r_c -= 8;
  } else if (r_c < 8 && r_c >= 4) {
    float16x4_t src[64];
    float16x4_t t[48];
    float16x4_t m[36];
    float16x4_t zero = vdup_n_f16(0);
    float16x4_t six = vdup_n_f16(6);
    Load64DataC4Fp16;
    float16x4_t bias_ptr = vld1_f16(bias_data);
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      float16x4_t tmp1 = vadd_f16(src[1 + offset], src[2 + offset]);
      float16x4_t tmp2 = vadd_f16(src[3 + offset], src[4 + offset]);
      float16x4_t tmp3 = vadd_f16(src[5 + offset], src[6 + offset]);
      float16x4_t tmp4 = vsub_f16(src[1 + offset], src[2 + offset]);
      float16x4_t tmp5 = vsub_f16(src[3 + offset], src[4 + offset]);
      float16x4_t tmp6 = vsub_f16(src[5 + offset], src[6 + offset]);
      t[l] = vadd_f16(vadd_f16(vadd_f16(src[offset], tmp1), tmp2), tmp3);
      t[l + 8] = vadd_f16(vadd_f16(vmul_n_f16(tmp4, 0.5), tmp5), vmul_n_f16(tmp6, 1.5));
      t[l + 16] = vadd_f16(vadd_f16(vmul_n_f16(tmp1, 0.25), tmp2), vmul_n_f16(tmp3, 2.25));
      t[l + 24] = vadd_f16(vadd_f16(vmul_n_f16(tmp4, 0.125), tmp5), vmul_n_f16(tmp6, 3.375));
      t[l + 32] = vadd_f16(vadd_f16(vmul_n_f16(tmp1, 0.0625), tmp2), vmul_n_f16(tmp3, 5.0625));
      t[l + 40] =
        vadd_f16(vadd_f16(vadd_f16(vmul_n_f16(tmp4, 0.03125), tmp5), vmul_n_f16(tmp6, 7.59375)), src[7 + offset]);
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 8;
      float16x4_t tmp1 = vadd_f16(t[1 + offset], t[2 + offset]);
      float16x4_t tmp2 = vadd_f16(t[3 + offset], t[4 + offset]);
      float16x4_t tmp3 = vadd_f16(t[5 + offset], t[6 + offset]);
      float16x4_t tmp4 = vsub_f16(t[1 + offset], t[2 + offset]);
      float16x4_t tmp5 = vsub_f16(t[3 + offset], t[4 + offset]);
      float16x4_t tmp6 = vsub_f16(t[5 + offset], t[6 + offset]);
      m[l] = vadd_f16(vadd_f16(vadd_f16(vadd_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
      m[l + 6] = vadd_f16(vadd_f16(vadd_f16(vmul_n_f16(tmp4, 0.5), tmp5), vmul_n_f16(tmp6, 1.5)), bias_ptr);
      m[l + 12] = vadd_f16(vadd_f16(vadd_f16(vmul_n_f16(tmp1, 0.25), tmp2), vmul_n_f16(tmp3, 2.25)), bias_ptr);
      m[l + 18] = vadd_f16(vadd_f16(vadd_f16(vmul_n_f16(tmp4, 0.125), tmp5), vmul_n_f16(tmp6, 3.375)), bias_ptr);
      m[l + 24] = vadd_f16(vadd_f16(vadd_f16(vmul_n_f16(tmp1, 0.0625), tmp2), vmul_n_f16(tmp3, 5.0625)), bias_ptr);
      m[l + 30] = vadd_f16(
        vadd_f16(vadd_f16(vadd_f16(vmul_n_f16(tmp4, 0.03125), tmp5), vmul_n_f16(tmp6, 7.59375)), t[7 + offset]),
        bias_ptr);
      m[l] = vmax_f16(zero, m[l]);
      m[l] = vmin_f16(six, m[l]);
      m[l + 6] = vmax_f16(zero, m[l + 6]);
      m[l + 6] = vmin_f16(six, m[l + 6]);
      m[l + 12] = vmax_f16(zero, m[l + 12]);
      m[l + 12] = vmin_f16(six, m[l + 12]);
      m[l + 18] = vmax_f16(zero, m[l + 18]);
      m[l + 18] = vmin_f16(six, m[l + 18]);
      m[l + 24] = vmax_f16(zero, m[l + 24]);
      m[l + 24] = vmin_f16(six, m[l + 24]);
      m[l + 30] = vmax_f16(zero, m[l + 30]);
      m[l + 30] = vmin_f16(six, m[l + 30]);
    }
    if (r_h == 6 && r_w == 6) {
      for (int i = 0; i < 6; i++) {
        int dst_k_offset = i * dst_step * out_c;
        int m_k_offset = i * 6;
        vst1_f16(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
        vst1_f16(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
        vst1_f16(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
        vst1_f16(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
        vst1_f16(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
        vst1_f16(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
      }
    } else {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 6;
        for (int k = 0; k < r_w; k++) {
          vst1_f16(dst_data + dst_k_offset + k * out_c, m[k + m_k_offset]);
        }
      }
    }
    z = 4;
  }
  for (; z < r_c; ++z) {
    float16_t src[64];
    float16_t t[48];
    float16_t m[36];
    for (int k = 0; k < 16; ++k) {
      src[k] = src_data[z + k * src_step];
    }
    float16_t bias_ptr = bias_data[z];
    for (int l = 0; l < 8; ++l) {
      int offset = l * 8;
      float16_t tmp1 = src[1 + offset] + src[2 + offset];
      float16_t tmp2 = src[3 + offset] + src[4 + offset];
      float16_t tmp3 = src[5 + offset] + src[6 + offset];
      float16_t tmp4 = src[1 + offset] - src[2 + offset];
      float16_t tmp5 = src[3 + offset] - src[4 + offset];
      float16_t tmp6 = src[5 + offset] - src[6 + offset];
      t[l] = src[offset] + tmp1 + tmp2 + tmp3;
      t[l + 8] = tmp4 * 0.5f + tmp5 + tmp6 * 1.5f;
      t[l + 16] = tmp1 * 0.25f + tmp2 + tmp3 * 2.25f;
      t[l + 24] = tmp4 * 0.125f + tmp5 + tmp6 * 3.375f;
      t[l + 32] = tmp1 * 0.0625f + tmp2 + tmp3 * 5.0625f;
      t[l + 40] = tmp4 * 0.03125f + tmp5 + tmp6 * 7.59375f + src[7 + offset];
    }
    for (int l = 0; l < 6; ++l) {
      int offset = l * 8;
      float16_t tmp1 = t[1 + offset] + t[2 + offset];
      float16_t tmp2 = t[3 + offset] + t[4 + offset];
      float16_t tmp3 = t[5 + offset] + t[6 + offset];
      float16_t tmp4 = t[1 + offset] - t[2 + offset];
      float16_t tmp5 = t[3 + offset] - t[4 + offset];
      float16_t tmp6 = t[5 + offset] - t[6 + offset];
      m[l] = t[offset] + tmp1 + tmp2 + tmp3 + bias_ptr;
      m[l + 6] = tmp4 * 0.5f + tmp5 + tmp6 * 1.5f + bias_ptr;
      m[l + 12] = tmp1 * 0.25f + tmp2 + tmp3 * 2.25f + bias_ptr;
      m[l + 18] = tmp4 * 0.125f + tmp5 + tmp6 * 3.375f + bias_ptr;
      m[l + 24] = tmp1 * 0.0625f + tmp2 + tmp3 * 5.0625f + bias_ptr;
      m[l + 30] = tmp4 * 0.03125f + tmp5 + tmp6 * 7.59375f + t[7 + offset] + bias_ptr;
      m[l] = m[l] > 0 ? m[l] : 0;
      m[l] = m[l] > 0 ? m[l] : 0;
      m[l + 6] = m[l + 6] > 0 ? m[l + 6] : 0;
      m[l + 6] = m[l + 6] < 6 ? m[l + 6] : 6;
      m[l + 12] = m[l + 12] > 0 ? m[l + 12] : 0;
      m[l + 12] = m[l + 12] < 6 ? m[l + 12] : 6;
      m[l + 18] = m[l + 18] > 0 ? m[l + 18] : 0;
      m[l + 18] = m[l + 18] < 6 ? m[l + 18] : 6;
      m[l + 24] = m[l + 24] > 0 ? m[l + 24] : 0;
      m[l + 24] = m[l + 24] < 6 ? m[l + 24] : 6;
      m[l + 30] = m[l + 30] > 0 ? m[l + 30] : 0;
      m[l + 30] = m[l + 30] < 6 ? m[l + 30] : 6;
    }
    for (int j = 0; j < r_h; j++) {
      int dst_k_offset = j * dst_step * out_c;
      int m_k_offset = j * 6;
      for (int k = 0; k < r_w; k++) {
        dst_data[z + dst_k_offset + k * out_c] = m[k + m_k_offset];
      }
    }
  }
}

void OutputTransform8x7UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[56];
  float16x8_t m[49];
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
    t[l + 24] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375));
    t[l + 32] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625));
    t[l + 40] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375));
    t[l + 48] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.015625), tmp2), vmulq_n_f16(tmp3, 11.390625)), src[7 + offset]);
  }
  for (int l = 0; l < 7; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 7] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 14] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
    m[l + 21] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), bias_ptr);
    m[l + 28] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), bias_ptr);
    m[l + 35] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375)), bias_ptr);
    m[l + 42] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.015625), tmp2), vmulq_n_f16(tmp3, 11.390625)), t[7 + offset]),
      bias_ptr);
  }
  if (r_c == C8NUM && r_h == 7 && r_w == 7) {
    for (int i = 0; i < 7; i++) {
      int dst_k_offset = i * dst_step * out_c;
      int m_k_offset = i * 7;
      vst1q_f16(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
      vst1q_f16(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
      vst1q_f16(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
      vst1q_f16(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
      vst1q_f16(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
      vst1q_f16(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
      vst1q_f16(dst_data + dst_k_offset + 6 * out_c, m[m_k_offset + 6]);
    }
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 7;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x7ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[56];
  float16x8_t m[49];
  float16x8_t zero = vdupq_n_f16(0);
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
    t[l + 24] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375));
    t[l + 32] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625));
    t[l + 40] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375));
    t[l + 48] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.015625), tmp2), vmulq_n_f16(tmp3, 11.390625)), src[7 + offset]);
  }
  for (int l = 0; l < 7; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 7] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 14] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
    m[l + 21] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), bias_ptr);
    m[l + 28] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), bias_ptr);
    m[l + 35] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375)), bias_ptr);
    m[l + 42] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.015625), tmp2), vmulq_n_f16(tmp3, 11.390625)), t[7 + offset]),
      bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l + 7] = vmaxq_f16(zero, m[l + 7]);
    m[l + 14] = vmaxq_f16(zero, m[l + 14]);
    m[l + 21] = vmaxq_f16(zero, m[l + 21]);
    m[l + 28] = vmaxq_f16(zero, m[l + 28]);
    m[l + 35] = vmaxq_f16(zero, m[l + 35]);
    m[l + 42] = vmaxq_f16(zero, m[l + 42]);
  }
  if (r_c == C8NUM && r_h == 7 && r_w == 7) {
    for (int i = 0; i < 7; i++) {
      int dst_k_offset = i * dst_step * out_c;
      int m_k_offset = i * 7;
      vst1q_f16(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
      vst1q_f16(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
      vst1q_f16(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
      vst1q_f16(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
      vst1q_f16(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
      vst1q_f16(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
      vst1q_f16(dst_data + dst_k_offset + 6 * out_c, m[m_k_offset + 6]);
    }
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 7;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

void OutputTransform8x7Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c) {
  float16x8_t src[64];
  float16x8_t t[56];
  float16x8_t m[49];
  float16x8_t zero = vdupq_n_f16(0);
  float16x8_t six = vdupq_n_f16(6);
  Load64DataFp16;
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  for (int l = 0; l < 8; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(src[5 + offset], src[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(src[1 + offset], src[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(src[3 + offset], src[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(src[5 + offset], src[6 + offset]);
    t[l] = vaddq_f16(vaddq_f16(vaddq_f16(src[offset], tmp1), tmp2), tmp3);
    t[l + 8] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5));
    t[l + 16] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25));
    t[l + 24] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375));
    t[l + 32] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625));
    t[l + 40] = vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375));
    t[l + 48] =
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.015625), tmp2), vmulq_n_f16(tmp3, 11.390625)), src[7 + offset]);
  }
  for (int l = 0; l < 7; ++l) {
    int offset = l * 8;
    float16x8_t tmp1 = vaddq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp2 = vaddq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp3 = vaddq_f16(t[5 + offset], t[6 + offset]);
    float16x8_t tmp4 = vsubq_f16(t[1 + offset], t[2 + offset]);
    float16x8_t tmp5 = vsubq_f16(t[3 + offset], t[4 + offset]);
    float16x8_t tmp6 = vsubq_f16(t[5 + offset], t[6 + offset]);
    m[l] = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t[offset], tmp1), tmp2), tmp3), bias_ptr);
    m[l + 7] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.5), tmp5), vmulq_n_f16(tmp6, 1.5)), bias_ptr);
    m[l + 14] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.25), tmp2), vmulq_n_f16(tmp3, 2.25)), bias_ptr);
    m[l + 21] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.125), tmp5), vmulq_n_f16(tmp6, 3.375)), bias_ptr);
    m[l + 28] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.0625), tmp2), vmulq_n_f16(tmp3, 5.0625)), bias_ptr);
    m[l + 35] = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp4, 0.03125), tmp5), vmulq_n_f16(tmp6, 7.59375)), bias_ptr);
    m[l + 42] = vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(tmp1, 0.015625), tmp2), vmulq_n_f16(tmp3, 11.390625)), t[7 + offset]),
      bias_ptr);
    m[l] = vmaxq_f16(zero, m[l]);
    m[l] = vminq_f16(six, m[l]);
    m[l + 7] = vmaxq_f16(zero, m[l + 7]);
    m[l + 7] = vminq_f16(six, m[l + 7]);
    m[l + 14] = vmaxq_f16(zero, m[l + 14]);
    m[l + 14] = vminq_f16(six, m[l + 14]);
    m[l + 21] = vmaxq_f16(zero, m[l + 21]);
    m[l + 21] = vminq_f16(six, m[l + 21]);
    m[l + 28] = vmaxq_f16(zero, m[l + 28]);
    m[l + 28] = vminq_f16(six, m[l + 28]);
    m[l + 35] = vmaxq_f16(zero, m[l + 35]);
    m[l + 35] = vminq_f16(six, m[l + 35]);
    m[l + 42] = vmaxq_f16(zero, m[l + 42]);
    m[l + 42] = vminq_f16(six, m[l + 42]);
  }
  if (r_c == C8NUM && r_h == 7 && r_w == 7) {
    for (int i = 0; i < 7; i++) {
      int dst_k_offset = i * dst_step * out_c;
      int m_k_offset = i * 7;
      vst1q_f16(dst_data + dst_k_offset + 0 * out_c, m[m_k_offset]);
      vst1q_f16(dst_data + dst_k_offset + 1 * out_c, m[m_k_offset + 1]);
      vst1q_f16(dst_data + dst_k_offset + 2 * out_c, m[m_k_offset + 2]);
      vst1q_f16(dst_data + dst_k_offset + 3 * out_c, m[m_k_offset + 3]);
      vst1q_f16(dst_data + dst_k_offset + 4 * out_c, m[m_k_offset + 4]);
      vst1q_f16(dst_data + dst_k_offset + 5 * out_c, m[m_k_offset + 5]);
      vst1q_f16(dst_data + dst_k_offset + 6 * out_c, m[m_k_offset + 6]);
    }
  } else {
    for (int i = 0; i < r_c; i++) {
      for (int j = 0; j < r_h; j++) {
        int dst_k_offset = j * dst_step * out_c;
        int m_k_offset = j * 7;
        for (int k = 0; k < r_w; k++) {
          dst_data[i + dst_k_offset + k * out_c] = m[k + m_k_offset][i];
        }
      }
    }
  }
}

int SelectOutputUnitFp16(const ConvParameter *conv_param) {
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int in_c = conv_param->input_channel_;
  int out_w = conv_param->output_w_;
  int out_h = conv_param->output_h_;
  int out_c = conv_param->output_channel_;
  int unit2 = UP_DIV(out_w * out_h, C16NUM * conv_param->op_parameter_.thread_num_);
  int max_out_unit = (int)(sqrtf((float)unit2));
  max_out_unit = max_out_unit < MAX_UNIT_FP16 ? max_out_unit : MAX_UNIT_FP16;
  max_out_unit = max_out_unit > MIN_UNIT_FP16 ? max_out_unit : MIN_UNIT_FP16;

  int unit = 0;
  float max_rate = 0.0f;
  float common_cost = (float)out_h * out_w * in_c * out_c * kernel_h * kernel_w;

  for (int i = MIN_UNIT_FP16; i <= max_out_unit; ++i) {
    int input_unit = i + kernel_w - 1;
    if (!GetOutputTransFp16Func(input_unit, i, ActType_No)) {
      continue;
    }
    float penalty = ((float)input_unit * input_unit) / ((float)kernel_h * kernel_w) * 0.12f;
    float wino_cost = ((2 + out_c) * (float)input_unit * input_unit * in_c + ((float)input_unit + i) * i * out_c) *
                      UP_DIV(out_w, i) * UP_DIV(out_h, i);
    float reduce_rate = common_cost / wino_cost - penalty;
    if (reduce_rate > max_rate) {
      max_rate = reduce_rate;
      unit = i;
    }
  }
  if (max_rate < 1.0f) {
    return 1;
  }
  // If output_unit is 1, then it is conventional convolution
  return unit;
}

void CheckIfUseWinogradFp16(bool *use_winograd, int *output_unit, const ConvParameter *conv_param) {
  if (conv_param->kernel_w_ == conv_param->kernel_h_ && conv_param->dilation_h_ == 1 && conv_param->dilation_w_ == 1 &&
      conv_param->stride_h_ == 1 && conv_param->stride_w_ == 1) {
    *output_unit = SelectOutputUnitFp16(conv_param);
    if (*output_unit > 1) {
      *use_winograd = true;
    }
  } else {
    *use_winograd = false;
  }
}
