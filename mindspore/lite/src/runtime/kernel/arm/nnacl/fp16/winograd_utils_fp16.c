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

#define MIN_UNIT 2
#define MAX_UNIT 8

static OutputTransformUnitFp16Func outputTransformUnitFp16[] = {
  NULL,  // 0
  NULL,  // 1
  OutputTransform8x2UnitFp16,
  OutputTransform8x3UnitFp16,
  OutputTransform8x4UnitFp16,
  OutputTransform8x5UnitFp16,
  OutputTransform8x6UnitFp16,
  OutputTransform8x7UnitFp16,
};

void InputTransform4x4UnitFp16(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step) {
#ifdef ENABLE_ARM
  float16x8_t src_data_00 = vld1q_f16(src_data + 0 * src_step);
  float16x8_t src_data_01 = vld1q_f16(src_data + 1 * src_step);
  float16x8_t src_data_02 = vld1q_f16(src_data + 2 * src_step);
  float16x8_t src_data_03 = vld1q_f16(src_data + 3 * src_step);
  float16x8_t src_data_10 = vld1q_f16(src_data + 4 * src_step);
  float16x8_t src_data_11 = vld1q_f16(src_data + 5 * src_step);
  float16x8_t src_data_12 = vld1q_f16(src_data + 6 * src_step);
  float16x8_t src_data_13 = vld1q_f16(src_data + 7 * src_step);
  float16x8_t src_data_20 = vld1q_f16(src_data + 8 * src_step);
  float16x8_t src_data_21 = vld1q_f16(src_data + 9 * src_step);
  float16x8_t src_data_22 = vld1q_f16(src_data + 10 * src_step);
  float16x8_t src_data_23 = vld1q_f16(src_data + 11 * src_step);
  float16x8_t src_data_30 = vld1q_f16(src_data + 12 * src_step);
  float16x8_t src_data_31 = vld1q_f16(src_data + 13 * src_step);
  float16x8_t src_data_32 = vld1q_f16(src_data + 14 * src_step);
  float16x8_t src_data_33 = vld1q_f16(src_data + 15 * src_step);

  float16x8_t t00 = vsubq_f16(src_data_00, vmulq_n_f16(src_data_20, 4));
  float16x8_t t01 = vsubq_f16(src_data_01, vmulq_n_f16(src_data_21, 4));
  float16x8_t t02 = vsubq_f16(src_data_02, vmulq_n_f16(src_data_22, 4));
  float16x8_t t03 = vsubq_f16(src_data_03, vmulq_n_f16(src_data_23, 4));

  float16x8_t t10 = vaddq_f16(src_data_10, vmulq_n_f16(src_data_20, 2));
  float16x8_t t11 = vaddq_f16(src_data_11, vmulq_n_f16(src_data_21, 2));
  float16x8_t t12 = vaddq_f16(src_data_12, vmulq_n_f16(src_data_22, 2));
  float16x8_t t13 = vaddq_f16(src_data_13, vmulq_n_f16(src_data_23, 2));

  float16x8_t t20 = vsubq_f16(vmulq_n_f16(src_data_20, 2), src_data_10);
  float16x8_t t21 = vsubq_f16(vmulq_n_f16(src_data_21, 2), src_data_11);
  float16x8_t t22 = vsubq_f16(vmulq_n_f16(src_data_22, 2), src_data_12);
  float16x8_t t23 = vsubq_f16(vmulq_n_f16(src_data_23, 2), src_data_13);

  float16x8_t t30 = vsubq_f16(src_data_30, vmulq_n_f16(src_data_10, 0.25));
  float16x8_t t31 = vsubq_f16(src_data_31, vmulq_n_f16(src_data_11, 0.25));
  float16x8_t t32 = vsubq_f16(src_data_32, vmulq_n_f16(src_data_12, 0.25));
  float16x8_t t33 = vsubq_f16(src_data_33, vmulq_n_f16(src_data_13, 0.25));

  float16x8_t m00 = vsubq_f16(t00, vmulq_n_f16(t02, 4));
  float16x8_t m01 = vaddq_f16(t01, vmulq_n_f16(t02, 2));
  float16x8_t m02 = vsubq_f16(vmulq_n_f16(t02, 2), t01);
  float16x8_t m03 = vsubq_f16(t03, vmulq_n_f16(t01, 0.25));

  float16x8_t m10 = vsubq_f16(t10, vmulq_n_f16(t12, 4));
  float16x8_t m11 = vaddq_f16(t11, vmulq_n_f16(t12, 2));
  float16x8_t m12 = vsubq_f16(vmulq_n_f16(t12, 2), t11);
  float16x8_t m13 = vsubq_f16(t13, vmulq_n_f16(t11, 0.25));

  float16x8_t m20 = vsubq_f16(t20, vmulq_n_f16(t22, 4));
  float16x8_t m21 = vaddq_f16(t21, vmulq_n_f16(t22, 2));
  float16x8_t m22 = vsubq_f16(vmulq_n_f16(t22, 2), t21);
  float16x8_t m23 = vsubq_f16(t23, vmulq_n_f16(t21, 0.25));

  float16x8_t m30 = vsubq_f16(t30, vmulq_n_f16(t32, 4));
  float16x8_t m31 = vaddq_f16(t31, vmulq_n_f16(t32, 2));
  float16x8_t m32 = vsubq_f16(vmulq_n_f16(t32, 2), t31);
  float16x8_t m33 = vsubq_f16(t33, vmulq_n_f16(t31, 0.25));

  vst1q_f16(dst_data + 0 * dst_step, m00);
  vst1q_f16(dst_data + 1 * dst_step, m01);
  vst1q_f16(dst_data + 2 * dst_step, m02);
  vst1q_f16(dst_data + 3 * dst_step, m03);
  vst1q_f16(dst_data + 4 * dst_step, m10);
  vst1q_f16(dst_data + 5 * dst_step, m11);
  vst1q_f16(dst_data + 6 * dst_step, m12);
  vst1q_f16(dst_data + 7 * dst_step, m13);
  vst1q_f16(dst_data + 8 * dst_step, m20);
  vst1q_f16(dst_data + 9 * dst_step, m21);
  vst1q_f16(dst_data + 10 * dst_step, m22);
  vst1q_f16(dst_data + 11 * dst_step, m23);
  vst1q_f16(dst_data + 12 * dst_step, m30);
  vst1q_f16(dst_data + 13 * dst_step, m31);
  vst1q_f16(dst_data + 14 * dst_step, m32);
  vst1q_f16(dst_data + 15 * dst_step, m33);
#else
  for (int i = 0; i < C8NUM; i++) {
    float16_t src_data_00 = src_data[i];
    float16_t src_data_01 = src_data[i + src_step];
    float16_t src_data_02 = src_data[i + 2 * src_step];
    float16_t src_data_03 = src_data[i + 3 * src_step];
    float16_t src_data_10 = src_data[i + 4 * src_step];
    float16_t src_data_11 = src_data[i + 5 * src_step];
    float16_t src_data_12 = src_data[i + 6 * src_step];
    float16_t src_data_13 = src_data[i + 7 * src_step];
    float16_t src_data_20 = src_data[i + 8 * src_step];
    float16_t src_data_21 = src_data[i + 9 * src_step];
    float16_t src_data_22 = src_data[i + 10 * src_step];
    float16_t src_data_23 = src_data[i + 11 * src_step];
    float16_t src_data_30 = src_data[i + 12 * src_step];
    float16_t src_data_31 = src_data[i + 13 * src_step];
    float16_t src_data_32 = src_data[i + 14 * src_step];
    float16_t src_data_33 = src_data[i + 15 * src_step];

    float16_t t00 = src_data_00 - 4 * src_data_20;
    float16_t t01 = src_data_01 - 4 * src_data_21;
    float16_t t02 = src_data_02 - 4 * src_data_22;
    float16_t t03 = src_data_03 - 4 * src_data_23;

    float16_t t10 = src_data_10 + 2 * src_data_20;
    float16_t t11 = src_data_11 + 2 * src_data_21;
    float16_t t12 = src_data_12 + 2 * src_data_22;
    float16_t t13 = src_data_13 + 2 * src_data_23;

    const float16_t t20 = 2 * src_data_20 - src_data_10;
    const float16_t t21 = 2 * src_data_21 - src_data_11;
    const float16_t t22 = 2 * src_data_22 - src_data_12;
    const float16_t t23 = 2 * src_data_23 - src_data_13;

    float16_t t30 = src_data_30 - 0.25f * src_data_10;
    float16_t t31 = src_data_31 - 0.25f * src_data_11;
    float16_t t32 = src_data_32 - 0.25f * src_data_12;
    float16_t t33 = src_data_33 - 0.25f * src_data_13;

    float16_t m00 = t00 - 4 * t02;
    float16_t m01 = t01 + 2 * t02;
    const float16_t m02 = 2 * t02 - t01;
    float16_t m03 = t03 - 0.25f * t01;

    float16_t m10 = t10 - 4 * t12;
    float16_t m11 = t11 + 2 * t12;
    const float16_t m12 = 2 * t12 - t11;
    float16_t m13 = t13 - 0.25f * t11;

    float16_t m20 = t20 - 4 * t22;
    float16_t m21 = t21 + 2 * t22;
    const float16_t m22 = 2 * t22 - t21;
    float16_t m23 = t23 - 0.25f * t21;

    float16_t m30 = t30 - 4 * t32;
    float16_t m31 = t31 + 2 * t32;
    float16_t m32 = 2 * t32 - t31;
    float16_t m33 = t33 - 0.25f * t31;

    (dst_data + i)[0] = m00;
    (dst_data + i + dst_step)[0] = m01;
    (dst_data + i + 2 * dst_step)[0] = m02;
    (dst_data + i + 3 * dst_step)[0] = m03;

    (dst_data + i + 4 * dst_step)[0] = m10;
    (dst_data + i + 5 * dst_step)[0] = m11;
    (dst_data + i + 6 * dst_step)[0] = m12;
    (dst_data + i + 7 * dst_step)[0] = m13;

    (dst_data + i + 8 * dst_step)[0] = m20;
    (dst_data + i + 9 * dst_step)[0] = m21;
    (dst_data + i + 10 * dst_step)[0] = m22;
    (dst_data + i + 11 * dst_step)[0] = m23;

    (dst_data + i + 12 * dst_step)[0] = m30;
    (dst_data + i + 13 * dst_step)[0] = m31;
    (dst_data + i + 14 * dst_step)[0] = m32;
    (dst_data + i + 15 * dst_step)[0] = m33;
  }
#endif
}

void InputTransform8x8UnitFp16(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step) {
#ifdef ENABLE_ARM
  float16x8_t src_data_00 = vld1q_f16(src_data + 0 * src_step);
  float16x8_t src_data_01 = vld1q_f16(src_data + 1 * src_step);
  float16x8_t src_data_02 = vld1q_f16(src_data + 2 * src_step);
  float16x8_t src_data_03 = vld1q_f16(src_data + 3 * src_step);
  float16x8_t src_data_04 = vld1q_f16(src_data + 4 * src_step);
  float16x8_t src_data_05 = vld1q_f16(src_data + 5 * src_step);
  float16x8_t src_data_06 = vld1q_f16(src_data + 6 * src_step);
  float16x8_t src_data_07 = vld1q_f16(src_data + 7 * src_step);
  float16x8_t src_data_10 = vld1q_f16(src_data + 8 * src_step);
  float16x8_t src_data_11 = vld1q_f16(src_data + 9 * src_step);
  float16x8_t src_data_12 = vld1q_f16(src_data + 10 * src_step);
  float16x8_t src_data_13 = vld1q_f16(src_data + 11 * src_step);
  float16x8_t src_data_14 = vld1q_f16(src_data + 12 * src_step);
  float16x8_t src_data_15 = vld1q_f16(src_data + 13 * src_step);
  float16x8_t src_data_16 = vld1q_f16(src_data + 14 * src_step);
  float16x8_t src_data_17 = vld1q_f16(src_data + 15 * src_step);
  float16x8_t src_data_20 = vld1q_f16(src_data + 16 * src_step);
  float16x8_t src_data_21 = vld1q_f16(src_data + 17 * src_step);
  float16x8_t src_data_22 = vld1q_f16(src_data + 18 * src_step);
  float16x8_t src_data_23 = vld1q_f16(src_data + 19 * src_step);
  float16x8_t src_data_24 = vld1q_f16(src_data + 20 * src_step);
  float16x8_t src_data_25 = vld1q_f16(src_data + 21 * src_step);
  float16x8_t src_data_26 = vld1q_f16(src_data + 22 * src_step);
  float16x8_t src_data_27 = vld1q_f16(src_data + 23 * src_step);
  float16x8_t src_data_30 = vld1q_f16(src_data + 24 * src_step);
  float16x8_t src_data_31 = vld1q_f16(src_data + 25 * src_step);
  float16x8_t src_data_32 = vld1q_f16(src_data + 26 * src_step);
  float16x8_t src_data_33 = vld1q_f16(src_data + 27 * src_step);
  float16x8_t src_data_34 = vld1q_f16(src_data + 28 * src_step);
  float16x8_t src_data_35 = vld1q_f16(src_data + 29 * src_step);
  float16x8_t src_data_36 = vld1q_f16(src_data + 30 * src_step);
  float16x8_t src_data_37 = vld1q_f16(src_data + 31 * src_step);
  float16x8_t src_data_40 = vld1q_f16(src_data + 32 * src_step);
  float16x8_t src_data_41 = vld1q_f16(src_data + 33 * src_step);
  float16x8_t src_data_42 = vld1q_f16(src_data + 34 * src_step);
  float16x8_t src_data_43 = vld1q_f16(src_data + 35 * src_step);
  float16x8_t src_data_44 = vld1q_f16(src_data + 36 * src_step);
  float16x8_t src_data_45 = vld1q_f16(src_data + 37 * src_step);
  float16x8_t src_data_46 = vld1q_f16(src_data + 38 * src_step);
  float16x8_t src_data_47 = vld1q_f16(src_data + 39 * src_step);
  float16x8_t src_data_50 = vld1q_f16(src_data + 40 * src_step);
  float16x8_t src_data_51 = vld1q_f16(src_data + 41 * src_step);
  float16x8_t src_data_52 = vld1q_f16(src_data + 42 * src_step);
  float16x8_t src_data_53 = vld1q_f16(src_data + 43 * src_step);
  float16x8_t src_data_54 = vld1q_f16(src_data + 44 * src_step);
  float16x8_t src_data_55 = vld1q_f16(src_data + 45 * src_step);
  float16x8_t src_data_56 = vld1q_f16(src_data + 46 * src_step);
  float16x8_t src_data_57 = vld1q_f16(src_data + 47 * src_step);
  float16x8_t src_data_60 = vld1q_f16(src_data + 48 * src_step);
  float16x8_t src_data_61 = vld1q_f16(src_data + 49 * src_step);
  float16x8_t src_data_62 = vld1q_f16(src_data + 50 * src_step);
  float16x8_t src_data_63 = vld1q_f16(src_data + 51 * src_step);
  float16x8_t src_data_64 = vld1q_f16(src_data + 52 * src_step);
  float16x8_t src_data_65 = vld1q_f16(src_data + 53 * src_step);
  float16x8_t src_data_66 = vld1q_f16(src_data + 54 * src_step);
  float16x8_t src_data_67 = vld1q_f16(src_data + 55 * src_step);
  float16x8_t src_data_70 = vld1q_f16(src_data + 56 * src_step);
  float16x8_t src_data_71 = vld1q_f16(src_data + 57 * src_step);
  float16x8_t src_data_72 = vld1q_f16(src_data + 58 * src_step);
  float16x8_t src_data_73 = vld1q_f16(src_data + 59 * src_step);
  float16x8_t src_data_74 = vld1q_f16(src_data + 60 * src_step);
  float16x8_t src_data_75 = vld1q_f16(src_data + 61 * src_step);
  float16x8_t src_data_76 = vld1q_f16(src_data + 62 * src_step);
  float16x8_t src_data_77 = vld1q_f16(src_data + 63 * src_step);

  float16x8_t t00 = vsubq_f16(vaddq_f16(vsubq_f16(src_data_00, vmulq_n_f16(src_data_20, 5.44444444444444444444444445)),
                                        vmulq_n_f16(src_data_40, 6.222222222222)),
                              vmulq_n_f16(src_data_60, 1.7777777777777));
  float16x8_t t01 = vsubq_f16(vaddq_f16(vsubq_f16(src_data_01, vmulq_n_f16(src_data_21, 5.44444444444444444444444445)),
                                        vmulq_n_f16(src_data_41, 6.222222222222)),
                              vmulq_n_f16(src_data_61, 1.7777777777777));
  float16x8_t t02 = vsubq_f16(vaddq_f16(vsubq_f16(src_data_02, vmulq_n_f16(src_data_22, 5.44444444444444444444444445)),
                                        vmulq_n_f16(src_data_42, 6.222222222222)),
                              vmulq_n_f16(src_data_62, 1.7777777777777));
  float16x8_t t03 = vsubq_f16(vaddq_f16(vsubq_f16(src_data_03, vmulq_n_f16(src_data_23, 5.44444444444444444444444445)),
                                        vmulq_n_f16(src_data_43, 6.222222222222)),
                              vmulq_n_f16(src_data_63, 1.7777777777777));
  float16x8_t t04 = vsubq_f16(vaddq_f16(vsubq_f16(src_data_04, vmulq_n_f16(src_data_24, 5.44444444444444444444444445)),
                                        vmulq_n_f16(src_data_44, 6.222222222222)),
                              vmulq_n_f16(src_data_64, 1.7777777777777));
  float16x8_t t05 = vsubq_f16(vaddq_f16(vsubq_f16(src_data_05, vmulq_n_f16(src_data_25, 5.44444444444444444444444445)),
                                        vmulq_n_f16(src_data_45, 6.222222222222)),
                              vmulq_n_f16(src_data_65, 1.7777777777777));
  float16x8_t t06 = vsubq_f16(vaddq_f16(vsubq_f16(src_data_06, vmulq_n_f16(src_data_26, 5.44444444444444444444444445)),
                                        vmulq_n_f16(src_data_46, 6.222222222222)),
                              vmulq_n_f16(src_data_66, 1.7777777777777));
  float16x8_t t07 = vsubq_f16(vaddq_f16(vsubq_f16(src_data_07, vmulq_n_f16(src_data_27, 5.44444444444444444444444445)),
                                        vmulq_n_f16(src_data_47, 6.222222222222)),
                              vmulq_n_f16(src_data_67, 1.7777777777777));

  float16x8_t t10 =
    vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_10, 1.5), vmulq_n_f16(src_data_20, 3)),
                                            vmulq_n_f16(src_data_30, 2.166666666666666667)),
                                  vmulq_n_f16(src_data_40, 4.333333333333)),
                        vmulq_n_f16(src_data_50, 0.66666666666)),
              vmulq_n_f16(src_data_60, 1.333333333333));
  float16x8_t t11 =
    vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_11, 1.5), vmulq_n_f16(src_data_21, 3)),
                                            vmulq_n_f16(src_data_31, 2.166666666666666667)),
                                  vmulq_n_f16(src_data_41, 4.333333333333)),
                        vmulq_n_f16(src_data_51, 0.66666666666)),
              vmulq_n_f16(src_data_61, 1.333333333333));
  float16x8_t t12 =
    vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_12, 1.5), vmulq_n_f16(src_data_22, 3)),
                                            vmulq_n_f16(src_data_32, 2.166666666666666667)),
                                  vmulq_n_f16(src_data_42, 4.333333333333)),
                        vmulq_n_f16(src_data_52, 0.66666666666)),
              vmulq_n_f16(src_data_62, 1.333333333333));
  float16x8_t t13 =
    vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_13, 1.5), vmulq_n_f16(src_data_23, 3)),
                                            vmulq_n_f16(src_data_33, 2.166666666666666667)),
                                  vmulq_n_f16(src_data_43, 4.333333333333)),
                        vmulq_n_f16(src_data_53, 0.66666666666)),
              vmulq_n_f16(src_data_63, 1.333333333333));
  float16x8_t t14 =
    vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_14, 1.5), vmulq_n_f16(src_data_24, 3)),
                                            vmulq_n_f16(src_data_34, 2.166666666666666667)),
                                  vmulq_n_f16(src_data_44, 4.333333333333)),
                        vmulq_n_f16(src_data_54, 0.66666666666)),
              vmulq_n_f16(src_data_64, 1.333333333333));
  float16x8_t t15 =
    vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_15, 1.5), vmulq_n_f16(src_data_25, 3)),
                                            vmulq_n_f16(src_data_35, 2.166666666666666667)),
                                  vmulq_n_f16(src_data_45, 4.333333333333)),
                        vmulq_n_f16(src_data_55, 0.66666666666)),
              vmulq_n_f16(src_data_65, 1.333333333333));
  float16x8_t t16 =
    vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_16, 1.5), vmulq_n_f16(src_data_26, 3)),
                                            vmulq_n_f16(src_data_36, 2.166666666666666667)),
                                  vmulq_n_f16(src_data_46, 4.333333333333)),
                        vmulq_n_f16(src_data_56, 0.66666666666)),
              vmulq_n_f16(src_data_66, 1.333333333333));
  float16x8_t t17 =
    vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_17, 1.5), vmulq_n_f16(src_data_27, 3)),
                                            vmulq_n_f16(src_data_37, 2.166666666666666667)),
                                  vmulq_n_f16(src_data_47, 4.333333333333)),
                        vmulq_n_f16(src_data_57, 0.66666666666)),
              vmulq_n_f16(src_data_67, 1.333333333333));

  float16x8_t t20 =
    vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(src_data_10, -1.5), vmulq_n_f16(src_data_20, 3)),
                                            vmulq_n_f16(src_data_30, 2.166666666666666667)),
                                  vmulq_n_f16(src_data_40, 4.333333333333)),
                        vmulq_n_f16(src_data_50, 0.66666666666)),
              vmulq_n_f16(src_data_60, 1.333333333333));
  float16x8_t t21 =
    vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(src_data_11, -1.5), vmulq_n_f16(src_data_21, 3)),
                                            vmulq_n_f16(src_data_31, 2.166666666666666667)),
                                  vmulq_n_f16(src_data_41, 4.333333333333)),
                        vmulq_n_f16(src_data_51, 0.66666666666)),
              vmulq_n_f16(src_data_61, 1.333333333333));
  float16x8_t t22 =
    vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(src_data_12, -1.5), vmulq_n_f16(src_data_22, 3)),
                                            vmulq_n_f16(src_data_32, 2.166666666666666667)),
                                  vmulq_n_f16(src_data_42, 4.333333333333)),
                        vmulq_n_f16(src_data_52, 0.66666666666)),
              vmulq_n_f16(src_data_62, 1.333333333333));
  float16x8_t t23 =
    vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(src_data_13, -1.5), vmulq_n_f16(src_data_23, 3)),
                                            vmulq_n_f16(src_data_33, 2.166666666666666667)),
                                  vmulq_n_f16(src_data_43, 4.333333333333)),
                        vmulq_n_f16(src_data_53, 0.66666666666)),
              vmulq_n_f16(src_data_63, 1.333333333333));
  float16x8_t t24 =
    vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(src_data_14, -1.5), vmulq_n_f16(src_data_24, 3)),
                                            vmulq_n_f16(src_data_34, 2.166666666666666667)),
                                  vmulq_n_f16(src_data_44, 4.333333333333)),
                        vmulq_n_f16(src_data_54, 0.66666666666)),
              vmulq_n_f16(src_data_64, 1.333333333333));
  float16x8_t t25 =
    vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(src_data_15, -1.5), vmulq_n_f16(src_data_25, 3)),
                                            vmulq_n_f16(src_data_35, 2.166666666666666667)),
                                  vmulq_n_f16(src_data_45, 4.333333333333)),
                        vmulq_n_f16(src_data_55, 0.66666666666)),
              vmulq_n_f16(src_data_65, 1.333333333333));
  float16x8_t t26 =
    vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(src_data_16, -1.5), vmulq_n_f16(src_data_26, 3)),
                                            vmulq_n_f16(src_data_36, 2.166666666666666667)),
                                  vmulq_n_f16(src_data_46, 4.333333333333)),
                        vmulq_n_f16(src_data_56, 0.66666666666)),
              vmulq_n_f16(src_data_66, 1.333333333333));
  float16x8_t t27 =
    vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(src_data_17, -1.5), vmulq_n_f16(src_data_27, 3)),
                                            vmulq_n_f16(src_data_37, 2.166666666666666667)),
                                  vmulq_n_f16(src_data_47, 4.333333333333)),
                        vmulq_n_f16(src_data_57, 0.66666666666)),
              vmulq_n_f16(src_data_67, 1.333333333333));

  float16x8_t t30 = vsubq_f16(vaddq_f16(vmulq_n_f16(vaddq_f16(src_data_30, src_data_40), 1.3333333333333),
                                        vmulq_n_f16(vaddq_f16(src_data_10, src_data_20), -0.3)),
                              vmulq_n_f16(vaddq_f16(src_data_50, src_data_60), 0.53333333333));
  float16x8_t t31 = vsubq_f16(vaddq_f16(vmulq_n_f16(vaddq_f16(src_data_31, src_data_41), 1.3333333333333),
                                        vmulq_n_f16(vaddq_f16(src_data_11, src_data_21), -0.3)),
                              vmulq_n_f16(vaddq_f16(src_data_51, src_data_61), 0.53333333333));
  float16x8_t t32 = vsubq_f16(vaddq_f16(vmulq_n_f16(vaddq_f16(src_data_32, src_data_42), 1.3333333333333),
                                        vmulq_n_f16(vaddq_f16(src_data_12, src_data_22), -0.3)),
                              vmulq_n_f16(vaddq_f16(src_data_52, src_data_62), 0.53333333333));
  float16x8_t t33 = vsubq_f16(vaddq_f16(vmulq_n_f16(vaddq_f16(src_data_33, src_data_43), 1.3333333333333),
                                        vmulq_n_f16(vaddq_f16(src_data_13, src_data_23), -0.3)),
                              vmulq_n_f16(vaddq_f16(src_data_53, src_data_63), 0.53333333333));
  float16x8_t t34 = vsubq_f16(vaddq_f16(vmulq_n_f16(vaddq_f16(src_data_34, src_data_44), 1.3333333333333),
                                        vmulq_n_f16(vaddq_f16(src_data_14, src_data_24), -0.3)),
                              vmulq_n_f16(vaddq_f16(src_data_54, src_data_64), 0.53333333333));
  float16x8_t t35 = vsubq_f16(vaddq_f16(vmulq_n_f16(vaddq_f16(src_data_35, src_data_45), 1.3333333333333),
                                        vmulq_n_f16(vaddq_f16(src_data_15, src_data_25), -0.3)),
                              vmulq_n_f16(vaddq_f16(src_data_55, src_data_65), 0.53333333333));
  float16x8_t t36 = vsubq_f16(vaddq_f16(vmulq_n_f16(vaddq_f16(src_data_36, src_data_46), 1.3333333333333),
                                        vmulq_n_f16(vaddq_f16(src_data_16, src_data_26), -0.3)),
                              vmulq_n_f16(vaddq_f16(src_data_56, src_data_66), 0.53333333333));
  float16x8_t t37 = vsubq_f16(vaddq_f16(vmulq_n_f16(vaddq_f16(src_data_37, src_data_47), 1.3333333333333),
                                        vmulq_n_f16(vaddq_f16(src_data_17, src_data_27), -0.3)),
                              vmulq_n_f16(vaddq_f16(src_data_57, src_data_67), 0.53333333333));

  float16x8_t t40 = vaddq_f16(vaddq_f16(vmulq_n_f16(vsubq_f16(src_data_40, src_data_30), 1.3333333333333),
                                        vmulq_n_f16(vsubq_f16(src_data_10, src_data_20), 0.3)),
                              vmulq_n_f16(vsubq_f16(src_data_50, src_data_60), 0.53333333333));
  float16x8_t t41 = vaddq_f16(vaddq_f16(vmulq_n_f16(vsubq_f16(src_data_41, src_data_31), 1.3333333333333),
                                        vmulq_n_f16(vsubq_f16(src_data_11, src_data_21), 0.3)),
                              vmulq_n_f16(vsubq_f16(src_data_51, src_data_61), 0.53333333333));
  float16x8_t t42 = vaddq_f16(vaddq_f16(vmulq_n_f16(vsubq_f16(src_data_42, src_data_32), 1.3333333333333),
                                        vmulq_n_f16(vsubq_f16(src_data_12, src_data_22), 0.3)),
                              vmulq_n_f16(vsubq_f16(src_data_52, src_data_62), 0.53333333333));
  float16x8_t t43 = vaddq_f16(vaddq_f16(vmulq_n_f16(vsubq_f16(src_data_43, src_data_33), 1.3333333333333),
                                        vmulq_n_f16(vsubq_f16(src_data_13, src_data_23), 0.3)),
                              vmulq_n_f16(vsubq_f16(src_data_53, src_data_63), 0.53333333333));
  float16x8_t t44 = vaddq_f16(vaddq_f16(vmulq_n_f16(vsubq_f16(src_data_44, src_data_34), 1.3333333333333),
                                        vmulq_n_f16(vsubq_f16(src_data_14, src_data_24), 0.3)),
                              vmulq_n_f16(vsubq_f16(src_data_54, src_data_64), 0.53333333333));
  float16x8_t t45 = vaddq_f16(vaddq_f16(vmulq_n_f16(vsubq_f16(src_data_45, src_data_35), 1.3333333333333),
                                        vmulq_n_f16(vsubq_f16(src_data_15, src_data_25), 0.3)),
                              vmulq_n_f16(vsubq_f16(src_data_55, src_data_65), 0.53333333333));
  float16x8_t t46 = vaddq_f16(vaddq_f16(vmulq_n_f16(vsubq_f16(src_data_46, src_data_36), 1.3333333333333),
                                        vmulq_n_f16(vsubq_f16(src_data_16, src_data_26), 0.3)),
                              vmulq_n_f16(vsubq_f16(src_data_56, src_data_66), 0.53333333333));
  float16x8_t t47 = vaddq_f16(vaddq_f16(vmulq_n_f16(vsubq_f16(src_data_47, src_data_37), 1.3333333333333),
                                        vmulq_n_f16(vsubq_f16(src_data_17, src_data_27), 0.3)),
                              vmulq_n_f16(vsubq_f16(src_data_57, src_data_67), 0.53333333333));

  float16x8_t t50 = vaddq_f16(
    vaddq_f16(
      vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_10, 0.03333333), vmulq_n_f16(src_data_20, 0.022222222)),
                          vmulq_n_f16(src_data_30, 0.1666666666)),
                vmulq_n_f16(src_data_40, 0.11111111111)),
      vmulq_n_f16(src_data_50, 0.133333333)),
    vmulq_n_f16(src_data_60, 0.088888888));
  float16x8_t t51 = vaddq_f16(
    vaddq_f16(
      vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_11, 0.03333333), vmulq_n_f16(src_data_21, 0.022222222)),
                          vmulq_n_f16(src_data_31, 0.1666666666)),
                vmulq_n_f16(src_data_41, 0.11111111111)),
      vmulq_n_f16(src_data_51, 0.133333333)),
    vmulq_n_f16(src_data_61, 0.088888888));
  float16x8_t t52 = vaddq_f16(
    vaddq_f16(
      vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_12, 0.03333333), vmulq_n_f16(src_data_22, 0.022222222)),
                          vmulq_n_f16(src_data_32, 0.1666666666)),
                vmulq_n_f16(src_data_42, 0.11111111111)),
      vmulq_n_f16(src_data_52, 0.133333333)),
    vmulq_n_f16(src_data_62, 0.088888888));
  float16x8_t t53 = vaddq_f16(
    vaddq_f16(
      vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_13, 0.03333333), vmulq_n_f16(src_data_23, 0.022222222)),
                          vmulq_n_f16(src_data_33, 0.1666666666)),
                vmulq_n_f16(src_data_43, 0.11111111111)),
      vmulq_n_f16(src_data_53, 0.133333333)),
    vmulq_n_f16(src_data_63, 0.088888888));
  float16x8_t t54 = vaddq_f16(
    vaddq_f16(
      vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_14, 0.03333333), vmulq_n_f16(src_data_24, 0.022222222)),
                          vmulq_n_f16(src_data_34, 0.1666666666)),
                vmulq_n_f16(src_data_44, 0.11111111111)),
      vmulq_n_f16(src_data_54, 0.133333333)),
    vmulq_n_f16(src_data_64, 0.088888888));
  float16x8_t t55 = vaddq_f16(
    vaddq_f16(
      vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_15, 0.03333333), vmulq_n_f16(src_data_25, 0.022222222)),
                          vmulq_n_f16(src_data_35, 0.1666666666)),
                vmulq_n_f16(src_data_45, 0.11111111111)),
      vmulq_n_f16(src_data_55, 0.133333333)),
    vmulq_n_f16(src_data_65, 0.088888888));
  float16x8_t t56 = vaddq_f16(
    vaddq_f16(
      vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_16, 0.03333333), vmulq_n_f16(src_data_26, 0.022222222)),
                          vmulq_n_f16(src_data_36, 0.1666666666)),
                vmulq_n_f16(src_data_46, 0.11111111111)),
      vmulq_n_f16(src_data_56, 0.133333333)),
    vmulq_n_f16(src_data_66, 0.088888888));
  float16x8_t t57 = vaddq_f16(
    vaddq_f16(
      vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_17, 0.03333333), vmulq_n_f16(src_data_27, 0.022222222)),
                          vmulq_n_f16(src_data_37, 0.1666666666)),
                vmulq_n_f16(src_data_47, 0.11111111111)),
      vmulq_n_f16(src_data_57, 0.133333333)),
    vmulq_n_f16(src_data_67, 0.088888888));

  float16x8_t t60 = vaddq_f16(
    vaddq_f16(
      vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(src_data_10, -0.03333333), vmulq_n_f16(src_data_20, 0.022222222)),
                          vmulq_n_f16(src_data_30, 0.1666666666)),
                vmulq_n_f16(src_data_40, 0.11111111111)),
      vmulq_n_f16(src_data_50, -0.133333333)),
    vmulq_n_f16(src_data_60, 0.088888888));
  float16x8_t t61 = vaddq_f16(
    vaddq_f16(
      vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(src_data_11, -0.03333333), vmulq_n_f16(src_data_21, 0.022222222)),
                          vmulq_n_f16(src_data_31, 0.1666666666)),
                vmulq_n_f16(src_data_41, 0.11111111111)),
      vmulq_n_f16(src_data_51, -0.133333333)),
    vmulq_n_f16(src_data_61, 0.088888888));
  float16x8_t t62 = vaddq_f16(
    vaddq_f16(
      vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(src_data_12, -0.03333333), vmulq_n_f16(src_data_22, 0.022222222)),
                          vmulq_n_f16(src_data_32, 0.1666666666)),
                vmulq_n_f16(src_data_42, 0.11111111111)),
      vmulq_n_f16(src_data_52, -0.133333333)),
    vmulq_n_f16(src_data_62, 0.088888888));
  float16x8_t t63 = vaddq_f16(
    vaddq_f16(
      vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(src_data_13, -0.03333333), vmulq_n_f16(src_data_23, 0.022222222)),
                          vmulq_n_f16(src_data_33, 0.1666666666)),
                vmulq_n_f16(src_data_43, 0.11111111111)),
      vmulq_n_f16(src_data_53, -0.133333333)),
    vmulq_n_f16(src_data_63, 0.088888888));
  float16x8_t t64 = vaddq_f16(
    vaddq_f16(
      vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(src_data_14, -0.03333333), vmulq_n_f16(src_data_24, 0.022222222)),
                          vmulq_n_f16(src_data_34, 0.1666666666)),
                vmulq_n_f16(src_data_44, 0.11111111111)),
      vmulq_n_f16(src_data_54, -0.133333333)),
    vmulq_n_f16(src_data_64, 0.088888888));
  float16x8_t t65 = vaddq_f16(
    vaddq_f16(
      vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(src_data_15, -0.03333333), vmulq_n_f16(src_data_25, 0.022222222)),
                          vmulq_n_f16(src_data_35, 0.1666666666)),
                vmulq_n_f16(src_data_45, 0.11111111111)),
      vmulq_n_f16(src_data_55, -0.133333333)),
    vmulq_n_f16(src_data_65, 0.088888888));
  float16x8_t t66 = vaddq_f16(
    vaddq_f16(
      vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(src_data_16, -0.03333333), vmulq_n_f16(src_data_26, 0.022222222)),
                          vmulq_n_f16(src_data_36, 0.1666666666)),
                vmulq_n_f16(src_data_46, 0.11111111111)),
      vmulq_n_f16(src_data_56, -0.133333333)),
    vmulq_n_f16(src_data_66, 0.088888888));
  float16x8_t t67 = vaddq_f16(
    vaddq_f16(
      vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(src_data_17, -0.03333333), vmulq_n_f16(src_data_27, 0.022222222)),
                          vmulq_n_f16(src_data_37, 0.1666666666)),
                vmulq_n_f16(src_data_47, 0.11111111111)),
      vmulq_n_f16(src_data_57, -0.133333333)),
    vmulq_n_f16(src_data_67, 0.088888888));

  float16x8_t t70 = vaddq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_30, 3.0625), vmulq_n_f16(src_data_10, -0.5625)),
                                        vmulq_n_f16(src_data_50, 3.5)),
                              src_data_70);
  float16x8_t t71 = vaddq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_31, 3.0625), vmulq_n_f16(src_data_11, -0.5625)),
                                        vmulq_n_f16(src_data_51, 3.5)),
                              src_data_71);
  float16x8_t t72 = vaddq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_32, 3.0625), vmulq_n_f16(src_data_12, -0.5625)),
                                        vmulq_n_f16(src_data_52, 3.5)),
                              src_data_72);
  float16x8_t t73 = vaddq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_33, 3.0625), vmulq_n_f16(src_data_13, -0.5625)),
                                        vmulq_n_f16(src_data_53, 3.5)),
                              src_data_73);
  float16x8_t t74 = vaddq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_34, 3.0625), vmulq_n_f16(src_data_14, -0.5625)),
                                        vmulq_n_f16(src_data_54, 3.5)),
                              src_data_74);
  float16x8_t t75 = vaddq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_35, 3.0625), vmulq_n_f16(src_data_15, -0.5625)),
                                        vmulq_n_f16(src_data_55, 3.5)),
                              src_data_75);
  float16x8_t t76 = vaddq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_36, 3.0625), vmulq_n_f16(src_data_16, -0.5625)),
                                        vmulq_n_f16(src_data_56, 3.5)),
                              src_data_76);
  float16x8_t t77 = vaddq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(src_data_37, 3.0625), vmulq_n_f16(src_data_17, -0.5625)),
                                        vmulq_n_f16(src_data_57, 3.5)),
                              src_data_77);

  float16x8_t m00 =
    vsubq_f16(vaddq_f16(vsubq_f16(t00, vmulq_n_f16(t02, 5.444444444444444)), vmulq_n_f16(t04, 6.22222222222)),
              vmulq_n_f16(t06, 1.77777777777777777778));
  float16x8_t m01 = vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t01, 1.5), vmulq_n_f16(t02, 3)),
                                                            vmulq_n_f16(t03, 2.16666666666666667)),
                                                  vmulq_n_f16(t04, 4.3333333333)),
                                        vmulq_n_f16(t05, 0.66666666667)),
                              vmulq_n_f16(t06, 1.333333333333));
  float16x8_t m02 = vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(t01, -1.5), vmulq_n_f16(t02, 3)),
                                                            vmulq_n_f16(t03, 2.16666666666666667)),
                                                  vmulq_n_f16(t04, 4.3333333333)),
                                        vmulq_n_f16(t05, 0.66666666667)),
                              vmulq_n_f16(t06, 1.333333333333));
  float16x8_t m03 =
    vaddq_f16(vaddq_f16(vmulq_n_f16(vaddq_f16(t01, t02), -0.3), vmulq_n_f16(vaddq_f16(t03, t04), 1.33333333333)),
              vmulq_n_f16(vaddq_f16(t05, t06), -0.533333333333));
  float16x8_t m04 =
    vaddq_f16(vaddq_f16(vmulq_n_f16(vsubq_f16(t01, t02), 0.3), vmulq_n_f16(vsubq_f16(t04, t03), 1.33333333333)),
              vmulq_n_f16(vsubq_f16(t05, t06), 0.533333333333));
  float16x8_t m05 =
    vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t01, 0.03333333), vmulq_n_f16(t02, 0.0222222)),
                                            vmulq_n_f16(t03, 0.16666666666666667)),
                                  vmulq_n_f16(t04, 0.11111111111)),
                        vmulq_n_f16(t05, 0.1333333333)),
              vmulq_n_f16(t06, 0.08888888888));
  float16x8_t m06 =
    vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(t01, -0.03333333), vmulq_n_f16(t02, 0.0222222)),
                                            vmulq_n_f16(t03, 0.16666666666666667)),
                                  vmulq_n_f16(t04, 0.11111111111)),
                        vmulq_n_f16(t05, 0.1333333333)),
              vmulq_n_f16(t06, 0.08888888888));
  float16x8_t m07 =
    vaddq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t01, -0.5625), vmulq_n_f16(t03, 3.0625)), vmulq_n_f16(t05, 3.5)), t07);

  float16x8_t m10 =
    vsubq_f16(vaddq_f16(vsubq_f16(t10, vmulq_n_f16(t12, 5.444444444444444)), vmulq_n_f16(t14, 6.22222222222)),
              vmulq_n_f16(t16, 1.77777777777777777778));
  float16x8_t m11 = vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t11, 1.5), vmulq_n_f16(t12, 3)),
                                                            vmulq_n_f16(t13, 2.16666666666666667)),
                                                  vmulq_n_f16(t14, 4.3333333333)),
                                        vmulq_n_f16(t15, 0.66666666667)),
                              vmulq_n_f16(t16, 1.333333333333));
  float16x8_t m12 = vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(t11, -1.5), vmulq_n_f16(t12, 3)),
                                                            vmulq_n_f16(t13, 2.16666666666666667)),
                                                  vmulq_n_f16(t14, 4.3333333333)),
                                        vmulq_n_f16(t15, 0.66666666667)),
                              vmulq_n_f16(t16, 1.333333333333));
  float16x8_t m13 =
    vaddq_f16(vaddq_f16(vmulq_n_f16(vaddq_f16(t11, t12), -0.3), vmulq_n_f16(vaddq_f16(t13, t14), 1.33333333333)),
              vmulq_n_f16(vaddq_f16(t15, t16), -0.533333333333));
  float16x8_t m14 =
    vaddq_f16(vaddq_f16(vmulq_n_f16(vsubq_f16(t11, t12), 0.3), vmulq_n_f16(vsubq_f16(t14, t13), 1.33333333333)),
              vmulq_n_f16(vsubq_f16(t15, t16), 0.533333333333));
  float16x8_t m15 =
    vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t11, 0.03333333), vmulq_n_f16(t12, 0.0222222)),
                                            vmulq_n_f16(t13, 0.16666666666666667)),
                                  vmulq_n_f16(t14, 0.11111111111)),
                        vmulq_n_f16(t15, 0.1333333333)),
              vmulq_n_f16(t16, 0.08888888888));
  float16x8_t m16 =
    vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(t11, -0.03333333), vmulq_n_f16(t12, 0.0222222)),
                                            vmulq_n_f16(t13, 0.16666666666666667)),
                                  vmulq_n_f16(t14, 0.11111111111)),
                        vmulq_n_f16(t15, 0.1333333333)),
              vmulq_n_f16(t16, 0.08888888888));
  float16x8_t m17 =
    vaddq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t11, -0.5625), vmulq_n_f16(t13, 3.0625)), vmulq_n_f16(t15, 3.5)), t17);

  float16x8_t m20 =
    vsubq_f16(vaddq_f16(vsubq_f16(t20, vmulq_n_f16(t22, 5.444444444444444)), vmulq_n_f16(t24, 6.22222222222)),
              vmulq_n_f16(t26, 1.77777777777777777778));
  float16x8_t m21 = vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t21, 1.5), vmulq_n_f16(t22, 3)),
                                                            vmulq_n_f16(t23, 2.16666666666666667)),
                                                  vmulq_n_f16(t24, 4.3333333333)),
                                        vmulq_n_f16(t25, 0.66666666667)),
                              vmulq_n_f16(t26, 1.333333333333));
  float16x8_t m22 = vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(t21, -1.5), vmulq_n_f16(t22, 3)),
                                                            vmulq_n_f16(t23, 2.16666666666666667)),
                                                  vmulq_n_f16(t24, 4.3333333333)),
                                        vmulq_n_f16(t25, 0.66666666667)),
                              vmulq_n_f16(t26, 1.333333333333));
  float16x8_t m23 =
    vaddq_f16(vaddq_f16(vmulq_n_f16(vaddq_f16(t21, t22), -0.3), vmulq_n_f16(vaddq_f16(t23, t24), 1.33333333333)),
              vmulq_n_f16(vaddq_f16(t25, t26), -0.533333333333));
  float16x8_t m24 =
    vaddq_f16(vaddq_f16(vmulq_n_f16(vsubq_f16(t21, t22), 0.3), vmulq_n_f16(vsubq_f16(t24, t23), 1.33333333333)),
              vmulq_n_f16(vsubq_f16(t25, t26), 0.533333333333));
  float16x8_t m25 =
    vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t21, 0.03333333), vmulq_n_f16(t22, 0.0222222)),
                                            vmulq_n_f16(t23, 0.16666666666666667)),
                                  vmulq_n_f16(t24, 0.11111111111)),
                        vmulq_n_f16(t25, 0.1333333333)),
              vmulq_n_f16(t26, 0.08888888888));
  float16x8_t m26 =
    vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(t21, -0.03333333), vmulq_n_f16(t22, 0.0222222)),
                                            vmulq_n_f16(t23, 0.16666666666666667)),
                                  vmulq_n_f16(t24, 0.11111111111)),
                        vmulq_n_f16(t25, 0.1333333333)),
              vmulq_n_f16(t26, 0.08888888888));
  float16x8_t m27 =
    vaddq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t21, -0.5625), vmulq_n_f16(t23, 3.0625)), vmulq_n_f16(t25, 3.5)), t27);

  float16x8_t m30 =
    vsubq_f16(vaddq_f16(vsubq_f16(t30, vmulq_n_f16(t32, 5.444444444444444)), vmulq_n_f16(t34, 6.22222222222)),
              vmulq_n_f16(t36, 1.77777777777777777778));
  float16x8_t m31 = vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t31, 1.5), vmulq_n_f16(t32, 3)),
                                                            vmulq_n_f16(t33, 2.16666666666666667)),
                                                  vmulq_n_f16(t34, 4.3333333333)),
                                        vmulq_n_f16(t35, 0.66666666667)),
                              vmulq_n_f16(t36, 1.333333333333));
  float16x8_t m32 = vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(t31, -1.5), vmulq_n_f16(t32, 3)),
                                                            vmulq_n_f16(t33, 2.16666666666666667)),
                                                  vmulq_n_f16(t34, 4.3333333333)),
                                        vmulq_n_f16(t35, 0.66666666667)),
                              vmulq_n_f16(t36, 1.333333333333));
  float16x8_t m33 =
    vaddq_f16(vaddq_f16(vmulq_n_f16(vaddq_f16(t31, t32), -0.3), vmulq_n_f16(vaddq_f16(t33, t34), 1.33333333333)),
              vmulq_n_f16(vaddq_f16(t35, t36), -0.533333333333));
  float16x8_t m34 =
    vaddq_f16(vaddq_f16(vmulq_n_f16(vsubq_f16(t31, t32), 0.3), vmulq_n_f16(vsubq_f16(t34, t33), 1.33333333333)),
              vmulq_n_f16(vsubq_f16(t35, t36), 0.533333333333));
  float16x8_t m35 =
    vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t31, 0.03333333), vmulq_n_f16(t32, 0.0222222)),
                                            vmulq_n_f16(t33, 0.16666666666666667)),
                                  vmulq_n_f16(t34, 0.11111111111)),
                        vmulq_n_f16(t35, 0.1333333333)),
              vmulq_n_f16(t36, 0.08888888888));
  float16x8_t m36 =
    vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(t31, -0.03333333), vmulq_n_f16(t32, 0.0222222)),
                                            vmulq_n_f16(t33, 0.16666666666666667)),
                                  vmulq_n_f16(t34, 0.11111111111)),
                        vmulq_n_f16(t35, 0.1333333333)),
              vmulq_n_f16(t36, 0.08888888888));
  float16x8_t m37 =
    vaddq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t31, -0.5625), vmulq_n_f16(t33, 3.0625)), vmulq_n_f16(t35, 3.5)), t37);

  float16x8_t m40 =
    vsubq_f16(vaddq_f16(vsubq_f16(t40, vmulq_n_f16(t42, 5.444444444444444)), vmulq_n_f16(t44, 6.22222222222)),
              vmulq_n_f16(t46, 1.77777777777777777778));
  float16x8_t m41 = vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t41, 1.5), vmulq_n_f16(t42, 3)),
                                                            vmulq_n_f16(t43, 2.16666666666666667)),
                                                  vmulq_n_f16(t44, 4.3333333333)),
                                        vmulq_n_f16(t45, 0.66666666667)),
                              vmulq_n_f16(t46, 1.333333333333));
  float16x8_t m42 = vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(t41, -1.5), vmulq_n_f16(t42, 3)),
                                                            vmulq_n_f16(t43, 2.16666666666666667)),
                                                  vmulq_n_f16(t44, 4.3333333333)),
                                        vmulq_n_f16(t45, 0.66666666667)),
                              vmulq_n_f16(t46, 1.333333333333));
  float16x8_t m43 =
    vaddq_f16(vaddq_f16(vmulq_n_f16(vaddq_f16(t41, t42), -0.3), vmulq_n_f16(vaddq_f16(t43, t44), 1.33333333333)),
              vmulq_n_f16(vaddq_f16(t45, t46), -0.533333333333));
  float16x8_t m44 =
    vaddq_f16(vaddq_f16(vmulq_n_f16(vsubq_f16(t41, t42), 0.3), vmulq_n_f16(vsubq_f16(t44, t43), 1.33333333333)),
              vmulq_n_f16(vsubq_f16(t45, t46), 0.533333333333));
  float16x8_t m45 =
    vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t41, 0.03333333), vmulq_n_f16(t42, 0.0222222)),
                                            vmulq_n_f16(t43, 0.16666666666666667)),
                                  vmulq_n_f16(t44, 0.11111111111)),
                        vmulq_n_f16(t45, 0.1333333333)),
              vmulq_n_f16(t46, 0.08888888888));
  float16x8_t m46 =
    vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(t41, -0.03333333), vmulq_n_f16(t42, 0.0222222)),
                                            vmulq_n_f16(t43, 0.16666666666666667)),
                                  vmulq_n_f16(t44, 0.11111111111)),
                        vmulq_n_f16(t45, 0.1333333333)),
              vmulq_n_f16(t46, 0.08888888888));
  float16x8_t m47 =
    vaddq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t41, -0.5625), vmulq_n_f16(t43, 3.0625)), vmulq_n_f16(t45, 3.5)), t47);

  float16x8_t m50 =
    vsubq_f16(vaddq_f16(vsubq_f16(t50, vmulq_n_f16(t52, 5.444444444444444)), vmulq_n_f16(t54, 6.22222222222)),
              vmulq_n_f16(t56, 1.77777777777777777778));
  float16x8_t m51 = vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t51, 1.5), vmulq_n_f16(t52, 3)),
                                                            vmulq_n_f16(t53, 2.16666666666666667)),
                                                  vmulq_n_f16(t54, 4.3333333333)),
                                        vmulq_n_f16(t55, 0.66666666667)),
                              vmulq_n_f16(t56, 1.333333333333));
  float16x8_t m52 = vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(t51, -1.5), vmulq_n_f16(t52, 3)),
                                                            vmulq_n_f16(t53, 2.16666666666666667)),
                                                  vmulq_n_f16(t54, 4.3333333333)),
                                        vmulq_n_f16(t55, 0.66666666667)),
                              vmulq_n_f16(t56, 1.333333333333));
  float16x8_t m53 =
    vaddq_f16(vaddq_f16(vmulq_n_f16(vaddq_f16(t51, t52), -0.3), vmulq_n_f16(vaddq_f16(t53, t54), 1.33333333333)),
              vmulq_n_f16(vaddq_f16(t55, t56), -0.533333333333));
  float16x8_t m54 =
    vaddq_f16(vaddq_f16(vmulq_n_f16(vsubq_f16(t51, t52), 0.3), vmulq_n_f16(vsubq_f16(t54, t53), 1.33333333333)),
              vmulq_n_f16(vsubq_f16(t55, t56), 0.533333333333));
  float16x8_t m55 =
    vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t51, 0.03333333), vmulq_n_f16(t52, 0.0222222)),
                                            vmulq_n_f16(t53, 0.16666666666666667)),
                                  vmulq_n_f16(t54, 0.11111111111)),
                        vmulq_n_f16(t55, 0.1333333333)),
              vmulq_n_f16(t56, 0.08888888888));
  float16x8_t m56 =
    vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(t51, -0.03333333), vmulq_n_f16(t52, 0.0222222)),
                                            vmulq_n_f16(t53, 0.16666666666666667)),
                                  vmulq_n_f16(t54, 0.11111111111)),
                        vmulq_n_f16(t55, 0.1333333333)),
              vmulq_n_f16(t56, 0.08888888888));
  float16x8_t m57 =
    vaddq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t51, -0.5625), vmulq_n_f16(t53, 3.0625)), vmulq_n_f16(t55, 3.5)), t57);

  float16x8_t m60 =
    vsubq_f16(vaddq_f16(vsubq_f16(t60, vmulq_n_f16(t62, 5.444444444444444)), vmulq_n_f16(t64, 6.22222222222)),
              vmulq_n_f16(t66, 1.77777777777777777778));
  float16x8_t m61 = vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t61, 1.5), vmulq_n_f16(t62, 3)),
                                                            vmulq_n_f16(t63, 2.16666666666666667)),
                                                  vmulq_n_f16(t64, 4.3333333333)),
                                        vmulq_n_f16(t65, 0.66666666667)),
                              vmulq_n_f16(t66, 1.333333333333));
  float16x8_t m62 = vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(t61, -1.5), vmulq_n_f16(t62, 3)),
                                                            vmulq_n_f16(t63, 2.16666666666666667)),
                                                  vmulq_n_f16(t64, 4.3333333333)),
                                        vmulq_n_f16(t65, 0.66666666667)),
                              vmulq_n_f16(t66, 1.333333333333));
  float16x8_t m63 =
    vaddq_f16(vaddq_f16(vmulq_n_f16(vaddq_f16(t61, t62), -0.3), vmulq_n_f16(vaddq_f16(t63, t64), 1.33333333333)),
              vmulq_n_f16(vaddq_f16(t65, t66), -0.533333333333));
  float16x8_t m64 =
    vaddq_f16(vaddq_f16(vmulq_n_f16(vsubq_f16(t61, t62), 0.3), vmulq_n_f16(vsubq_f16(t64, t63), 1.33333333333)),
              vmulq_n_f16(vsubq_f16(t65, t66), 0.533333333333));
  float16x8_t m65 =
    vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t61, 0.03333333), vmulq_n_f16(t62, 0.0222222)),
                                            vmulq_n_f16(t63, 0.16666666666666667)),
                                  vmulq_n_f16(t64, 0.11111111111)),
                        vmulq_n_f16(t65, 0.1333333333)),
              vmulq_n_f16(t66, 0.08888888888));
  float16x8_t m66 =
    vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(t61, -0.03333333), vmulq_n_f16(t62, 0.0222222)),
                                            vmulq_n_f16(t63, 0.16666666666666667)),
                                  vmulq_n_f16(t64, 0.11111111111)),
                        vmulq_n_f16(t65, 0.1333333333)),
              vmulq_n_f16(t66, 0.08888888888));
  float16x8_t m67 =
    vaddq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t61, -0.5625), vmulq_n_f16(t63, 3.0625)), vmulq_n_f16(t65, 3.5)), t67);

  float16x8_t m70 =
    vsubq_f16(vaddq_f16(vsubq_f16(t70, vmulq_n_f16(t72, 5.444444444444444)), vmulq_n_f16(t74, 6.22222222222)),
              vmulq_n_f16(t76, 1.77777777777777777778));
  float16x8_t m71 = vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t71, 1.5), vmulq_n_f16(t72, 3)),
                                                            vmulq_n_f16(t73, 2.16666666666666667)),
                                                  vmulq_n_f16(t74, 4.3333333333)),
                                        vmulq_n_f16(t75, 0.66666666667)),
                              vmulq_n_f16(t76, 1.333333333333));
  float16x8_t m72 = vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(t71, -1.5), vmulq_n_f16(t72, 3)),
                                                            vmulq_n_f16(t73, 2.16666666666666667)),
                                                  vmulq_n_f16(t74, 4.3333333333)),
                                        vmulq_n_f16(t75, 0.66666666667)),
                              vmulq_n_f16(t76, 1.333333333333));
  float16x8_t m73 =
    vaddq_f16(vaddq_f16(vmulq_n_f16(vaddq_f16(t71, t72), -0.3), vmulq_n_f16(vaddq_f16(t73, t74), 1.33333333333)),
              vmulq_n_f16(vaddq_f16(t75, t76), -0.533333333333));
  float16x8_t m74 =
    vaddq_f16(vaddq_f16(vmulq_n_f16(vsubq_f16(t71, t72), 0.3), vmulq_n_f16(vsubq_f16(t74, t73), 1.33333333333)),
              vmulq_n_f16(vsubq_f16(t75, t76), 0.533333333333));
  float16x8_t m75 =
    vaddq_f16(vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t71, 0.03333333), vmulq_n_f16(t72, 0.0222222)),
                                            vmulq_n_f16(t73, 0.16666666666666667)),
                                  vmulq_n_f16(t74, 0.11111111111)),
                        vmulq_n_f16(t75, 0.1333333333)),
              vmulq_n_f16(t76, 0.08888888888));
  float16x8_t m76 =
    vaddq_f16(vsubq_f16(vsubq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(t71, -0.03333333), vmulq_n_f16(t72, 0.0222222)),
                                            vmulq_n_f16(t73, 0.16666666666666667)),
                                  vmulq_n_f16(t74, 0.11111111111)),
                        vmulq_n_f16(t75, 0.1333333333)),
              vmulq_n_f16(t76, 0.08888888888));
  float16x8_t m77 =
    vaddq_f16(vsubq_f16(vaddq_f16(vmulq_n_f16(t71, -0.5625), vmulq_n_f16(t73, 3.0625)), vmulq_n_f16(t75, 3.5)), t77);

  vst1q_f16(dst_data + 0 * dst_step, m00);
  vst1q_f16(dst_data + 1 * dst_step, m01);
  vst1q_f16(dst_data + 2 * dst_step, m02);
  vst1q_f16(dst_data + 3 * dst_step, m03);
  vst1q_f16(dst_data + 4 * dst_step, m04);
  vst1q_f16(dst_data + 5 * dst_step, m05);
  vst1q_f16(dst_data + 6 * dst_step, m06);
  vst1q_f16(dst_data + 7 * dst_step, m07);
  vst1q_f16(dst_data + 8 * dst_step, m10);
  vst1q_f16(dst_data + 9 * dst_step, m11);
  vst1q_f16(dst_data + 10 * dst_step, m12);
  vst1q_f16(dst_data + 11 * dst_step, m13);
  vst1q_f16(dst_data + 12 * dst_step, m14);
  vst1q_f16(dst_data + 13 * dst_step, m15);
  vst1q_f16(dst_data + 14 * dst_step, m16);
  vst1q_f16(dst_data + 15 * dst_step, m17);
  vst1q_f16(dst_data + 16 * dst_step, m20);
  vst1q_f16(dst_data + 17 * dst_step, m21);
  vst1q_f16(dst_data + 18 * dst_step, m22);
  vst1q_f16(dst_data + 19 * dst_step, m23);
  vst1q_f16(dst_data + 20 * dst_step, m24);
  vst1q_f16(dst_data + 21 * dst_step, m25);
  vst1q_f16(dst_data + 22 * dst_step, m26);
  vst1q_f16(dst_data + 23 * dst_step, m27);
  vst1q_f16(dst_data + 24 * dst_step, m30);
  vst1q_f16(dst_data + 25 * dst_step, m31);
  vst1q_f16(dst_data + 26 * dst_step, m32);
  vst1q_f16(dst_data + 27 * dst_step, m33);
  vst1q_f16(dst_data + 28 * dst_step, m34);
  vst1q_f16(dst_data + 29 * dst_step, m35);
  vst1q_f16(dst_data + 30 * dst_step, m36);
  vst1q_f16(dst_data + 31 * dst_step, m37);
  vst1q_f16(dst_data + 32 * dst_step, m40);
  vst1q_f16(dst_data + 33 * dst_step, m41);
  vst1q_f16(dst_data + 34 * dst_step, m42);
  vst1q_f16(dst_data + 35 * dst_step, m43);
  vst1q_f16(dst_data + 36 * dst_step, m44);
  vst1q_f16(dst_data + 37 * dst_step, m45);
  vst1q_f16(dst_data + 38 * dst_step, m46);
  vst1q_f16(dst_data + 39 * dst_step, m47);
  vst1q_f16(dst_data + 40 * dst_step, m50);
  vst1q_f16(dst_data + 41 * dst_step, m51);
  vst1q_f16(dst_data + 42 * dst_step, m52);
  vst1q_f16(dst_data + 43 * dst_step, m53);
  vst1q_f16(dst_data + 44 * dst_step, m54);
  vst1q_f16(dst_data + 45 * dst_step, m55);
  vst1q_f16(dst_data + 46 * dst_step, m56);
  vst1q_f16(dst_data + 47 * dst_step, m57);
  vst1q_f16(dst_data + 48 * dst_step, m60);
  vst1q_f16(dst_data + 49 * dst_step, m61);
  vst1q_f16(dst_data + 50 * dst_step, m62);
  vst1q_f16(dst_data + 51 * dst_step, m63);
  vst1q_f16(dst_data + 52 * dst_step, m64);
  vst1q_f16(dst_data + 53 * dst_step, m65);
  vst1q_f16(dst_data + 54 * dst_step, m66);
  vst1q_f16(dst_data + 55 * dst_step, m67);
  vst1q_f16(dst_data + 56 * dst_step, m70);
  vst1q_f16(dst_data + 57 * dst_step, m71);
  vst1q_f16(dst_data + 58 * dst_step, m72);
  vst1q_f16(dst_data + 59 * dst_step, m73);
  vst1q_f16(dst_data + 60 * dst_step, m74);
  vst1q_f16(dst_data + 61 * dst_step, m75);
  vst1q_f16(dst_data + 62 * dst_step, m76);
  vst1q_f16(dst_data + 63 * dst_step, m77);
#else
  for (int i = 0; i < C8NUM; i++) {
    float16_t src_data_00 = src_data[i];
    float16_t src_data_01 = src_data[i + src_step];
    float16_t src_data_02 = src_data[i + 2 * src_step];
    float16_t src_data_03 = src_data[i + 3 * src_step];
    float16_t src_data_04 = src_data[i + 4 * src_step];
    float16_t src_data_05 = src_data[i + 5 * src_step];
    float16_t src_data_06 = src_data[i + 6 * src_step];
    float16_t src_data_07 = src_data[i + 7 * src_step];
    float16_t src_data_10 = src_data[i + 8 * src_step];
    float16_t src_data_11 = src_data[i + 9 * src_step];
    float16_t src_data_12 = src_data[i + 10 * src_step];
    float16_t src_data_13 = src_data[i + 11 * src_step];
    float16_t src_data_14 = src_data[i + 12 * src_step];
    float16_t src_data_15 = src_data[i + 13 * src_step];
    float16_t src_data_16 = src_data[i + 14 * src_step];
    float16_t src_data_17 = src_data[i + 15 * src_step];
    float16_t src_data_20 = src_data[i + 16 * src_step];
    float16_t src_data_21 = src_data[i + 17 * src_step];
    float16_t src_data_22 = src_data[i + 18 * src_step];
    float16_t src_data_23 = src_data[i + 19 * src_step];
    float16_t src_data_24 = src_data[i + 20 * src_step];
    float16_t src_data_25 = src_data[i + 21 * src_step];
    float16_t src_data_26 = src_data[i + 22 * src_step];
    float16_t src_data_27 = src_data[i + 23 * src_step];
    float16_t src_data_30 = src_data[i + 24 * src_step];
    float16_t src_data_31 = src_data[i + 25 * src_step];
    float16_t src_data_32 = src_data[i + 26 * src_step];
    float16_t src_data_33 = src_data[i + 27 * src_step];
    float16_t src_data_34 = src_data[i + 28 * src_step];
    float16_t src_data_35 = src_data[i + 29 * src_step];
    float16_t src_data_36 = src_data[i + 30 * src_step];
    float16_t src_data_37 = src_data[i + 31 * src_step];
    float16_t src_data_40 = src_data[i + 32 * src_step];
    float16_t src_data_41 = src_data[i + 33 * src_step];
    float16_t src_data_42 = src_data[i + 34 * src_step];
    float16_t src_data_43 = src_data[i + 35 * src_step];
    float16_t src_data_44 = src_data[i + 36 * src_step];
    float16_t src_data_45 = src_data[i + 37 * src_step];
    float16_t src_data_46 = src_data[i + 38 * src_step];
    float16_t src_data_47 = src_data[i + 39 * src_step];
    float16_t src_data_50 = src_data[i + 40 * src_step];
    float16_t src_data_51 = src_data[i + 41 * src_step];
    float16_t src_data_52 = src_data[i + 42 * src_step];
    float16_t src_data_53 = src_data[i + 43 * src_step];
    float16_t src_data_54 = src_data[i + 44 * src_step];
    float16_t src_data_55 = src_data[i + 45 * src_step];
    float16_t src_data_56 = src_data[i + 46 * src_step];
    float16_t src_data_57 = src_data[i + 47 * src_step];
    float16_t src_data_60 = src_data[i + 48 * src_step];
    float16_t src_data_61 = src_data[i + 49 * src_step];
    float16_t src_data_62 = src_data[i + 50 * src_step];
    float16_t src_data_63 = src_data[i + 51 * src_step];
    float16_t src_data_64 = src_data[i + 52 * src_step];
    float16_t src_data_65 = src_data[i + 53 * src_step];
    float16_t src_data_66 = src_data[i + 54 * src_step];
    float16_t src_data_67 = src_data[i + 55 * src_step];
    float16_t src_data_70 = src_data[i + 56 * src_step];
    float16_t src_data_71 = src_data[i + 57 * src_step];
    float16_t src_data_72 = src_data[i + 58 * src_step];
    float16_t src_data_73 = src_data[i + 59 * src_step];
    float16_t src_data_74 = src_data[i + 60 * src_step];
    float16_t src_data_75 = src_data[i + 61 * src_step];
    float16_t src_data_76 = src_data[i + 62 * src_step];
    float16_t src_data_77 = src_data[i + 63 * src_step];

    float16_t t00 = src_data_00 - 5.444444444444444445125f * src_data_20 + 6.222222222222222222223f * src_data_40 -
                    1.77777777777777778f * src_data_60;
    float16_t t01 = src_data_01 - 5.444444444444444445125f * src_data_21 + 6.222222222222222222223f * src_data_41 -
                    1.77777777777777778f * src_data_61;
    float16_t t02 = src_data_02 - 5.444444444444444445125f * src_data_22 + 6.222222222222222222223f * src_data_42 -
                    1.77777777777777778f * src_data_62;
    float16_t t03 = src_data_03 - 5.444444444444444445125f * src_data_23 + 6.222222222222222222223f * src_data_43 -
                    1.77777777777777778f * src_data_63;
    float16_t t04 = src_data_04 - 5.444444444444444445125f * src_data_24 + 6.222222222222222222223f * src_data_44 -
                    1.77777777777777778f * src_data_64;
    float16_t t05 = src_data_05 - 5.444444444444444445125f * src_data_25 + 6.222222222222222222223f * src_data_45 -
                    1.77777777777777778f * src_data_65;
    float16_t t06 = src_data_06 - 5.444444444444444445125f * src_data_26 + 6.222222222222222222223f * src_data_46 -
                    1.77777777777777778f * src_data_66;
    float16_t t07 = src_data_07 - 5.444444444444444445125f * src_data_27 + 6.222222222222222222223f * src_data_47 -
                    1.77777777777777778f * src_data_67;

    const float16_t t10 = 1.5f * src_data_10 + 3.0f * src_data_20 - 2.1666666666666667f * src_data_30 -
                          4.333333333333333333f * src_data_40 + 0.66666666666666667f * src_data_50 +
                          1.333333333333333f * src_data_60;
    const float16_t t11 = 1.5f * src_data_11 + 3.0f * src_data_21 - 2.1666666666666667f * src_data_31 -
                          4.333333333333333333f * src_data_41 + 0.66666666666666667f * src_data_51 +
                          1.333333333333333f * src_data_61;
    const float16_t t12 = 1.5f * src_data_12 + 3.0f * src_data_22 - 2.1666666666666667f * src_data_32 -
                          4.333333333333333333f * src_data_42 + 0.66666666666666667f * src_data_52 +
                          1.333333333333333f * src_data_62;
    const float16_t t13 = 1.5f * src_data_13 + 3.0f * src_data_23 - 2.1666666666666667f * src_data_33 -
                          4.333333333333333333f * src_data_43 + 0.66666666666666667f * src_data_53 +
                          1.333333333333333f * src_data_63;
    const float16_t t14 = 1.5f * src_data_14 + 3.0f * src_data_24 - 2.1666666666666667f * src_data_34 -
                          4.333333333333333333f * src_data_44 + 0.66666666666666667f * src_data_54 +
                          1.333333333333333f * src_data_64;
    const float16_t t15 = 1.5f * src_data_15 + 3.0f * src_data_25 - 2.1666666666666667f * src_data_35 -
                          4.333333333333333333f * src_data_45 + 0.66666666666666667f * src_data_55 +
                          1.333333333333333f * src_data_65;
    const float16_t t16 = 1.5f * src_data_16 + 3.0f * src_data_26 - 2.1666666666666667f * src_data_36 -
                          4.333333333333333333f * src_data_46 + 0.66666666666666667f * src_data_56 +
                          1.333333333333333f * src_data_66;
    const float16_t t17 = 1.5f * src_data_17 + 3.0f * src_data_27 - 2.1666666666666667f * src_data_37 -
                          4.333333333333333333f * src_data_47 + 0.66666666666666667f * src_data_57 +
                          1.333333333333333f * src_data_67;

    const float16_t t20 = -1.5f * src_data_10 + 3.0f * src_data_20 + 2.1666666666666667f * src_data_30 -
                          4.333333333333333333f * src_data_40 - 0.66666666666666667f * src_data_50 +
                          1.333333333333333f * src_data_60;
    const float16_t t21 = -1.5f * src_data_11 + 3.0f * src_data_21 + 2.1666666666666667f * src_data_31 -
                          4.333333333333333333f * src_data_41 - 0.66666666666666667f * src_data_51 +
                          1.333333333333333f * src_data_61;
    const float16_t t22 = -1.5f * src_data_12 + 3.0f * src_data_22 + 2.1666666666666667f * src_data_32 -
                          4.333333333333333333f * src_data_42 - 0.66666666666666667f * src_data_52 +
                          1.333333333333333f * src_data_62;
    const float16_t t23 = -1.5f * src_data_13 + 3.0f * src_data_23 + 2.1666666666666667f * src_data_33 -
                          4.333333333333333333f * src_data_43 - 0.66666666666666667f * src_data_53 +
                          1.333333333333333f * src_data_63;
    const float16_t t24 = -1.5f * src_data_14 + 3.0f * src_data_24 + 2.1666666666666667f * src_data_34 -
                          4.333333333333333333f * src_data_44 - 0.66666666666666667f * src_data_54 +
                          1.333333333333333f * src_data_64;
    const float16_t t25 = -1.5f * src_data_15 + 3.0f * src_data_25 + 2.1666666666666667f * src_data_35 -
                          4.333333333333333333f * src_data_45 - 0.66666666666666667f * src_data_55 +
                          1.333333333333333f * src_data_65;
    const float16_t t26 = -1.5f * src_data_16 + 3.0f * src_data_26 + 2.1666666666666667f * src_data_36 -
                          4.333333333333333333f * src_data_46 - 0.66666666666666667f * src_data_56 +
                          1.333333333333333f * src_data_66;
    const float16_t t27 = -1.5f * src_data_17 + 3.0f * src_data_27 + 2.1666666666666667f * src_data_37 -
                          4.333333333333333333f * src_data_47 - 0.66666666666666667f * src_data_57 +
                          1.333333333333333f * src_data_67;

    const float16_t t30 = -0.3f * (src_data_10 + src_data_20) + 1.33333333333333f * (src_data_30 + src_data_40) -
                          0.53333333333f * (src_data_50 + src_data_60);
    const float16_t t31 = -0.3f * (src_data_11 + src_data_21) + 1.33333333333333f * (src_data_31 + src_data_41) -
                          0.53333333333f * (src_data_51 + src_data_61);
    const float16_t t32 = -0.3f * (src_data_12 + src_data_22) + 1.33333333333333f * (src_data_32 + src_data_42) -
                          0.53333333333f * (src_data_52 + src_data_62);
    const float16_t t33 = -0.3f * (src_data_13 + src_data_23) + 1.33333333333333f * (src_data_33 + src_data_43) -
                          0.53333333333f * (src_data_53 + src_data_63);
    const float16_t t34 = -0.3f * (src_data_14 + src_data_24) + 1.33333333333333f * (src_data_34 + src_data_44) -
                          0.53333333333f * (src_data_54 + src_data_64);
    const float16_t t35 = -0.3f * (src_data_15 + src_data_25) + 1.33333333333333f * (src_data_35 + src_data_45) -
                          0.53333333333f * (src_data_55 + src_data_65);
    const const float16_t t36 = -0.3f * (src_data_16 + src_data_26) + 1.33333333333333f * (src_data_36 + src_data_46) -
                                0.53333333333f * (src_data_56 + src_data_66);
    const const float16_t t37 = -0.3f * (src_data_17 + src_data_27) + 1.33333333333333f * (src_data_37 + src_data_47) -
                                0.53333333333f * (src_data_57 + src_data_67);

    const float16_t t40 = 0.3f * (src_data_10 - src_data_20) + 1.33333333333333f * (src_data_40 - src_data_30) +
                          0.53333333333f * (src_data_50 - src_data_60);
    const float16_t t41 = 0.3f * (src_data_11 - src_data_21) + 1.33333333333333f * (src_data_41 - src_data_31) +
                          0.53333333333f * (src_data_51 - src_data_61);
    const float16_t t42 = 0.3f * (src_data_12 - src_data_22) + 1.33333333333333f * (src_data_42 - src_data_32) +
                          0.53333333333f * (src_data_52 - src_data_62);
    const float16_t t43 = 0.3f * (src_data_13 - src_data_23) + 1.33333333333333f * (src_data_43 - src_data_33) +
                          0.53333333333f * (src_data_53 - src_data_63);
    const float16_t t44 = 0.3f * (src_data_14 - src_data_24) + 1.33333333333333f * (src_data_44 - src_data_34) +
                          0.53333333333f * (src_data_54 - src_data_64);
    const float16_t t45 = 0.3f * (src_data_15 - src_data_25) + 1.33333333333333f * (src_data_45 - src_data_35) +
                          0.53333333333f * (src_data_55 - src_data_65);
    const float16_t t46 = 0.3f * (src_data_16 - src_data_26) + 1.33333333333333f * (src_data_46 - src_data_36) +
                          0.53333333333f * (src_data_56 - src_data_66);
    const float16_t t47 = 0.3f * (src_data_17 - src_data_27) + 1.33333333333333f * (src_data_47 - src_data_37) +
                          0.53333333333f * (src_data_57 - src_data_67);

    const float16_t t50 = 0.0333333333f * src_data_10 + 0.02222222f * src_data_20 - 0.1666666666f * src_data_30 -
                          0.1111111111f * src_data_40 + 0.1333333f * src_data_50 + 0.0888888f * src_data_60;
    const float16_t t51 = 0.0333333333f * src_data_11 + 0.02222222f * src_data_21 - 0.1666666666f * src_data_31 -
                          0.1111111111f * src_data_41 + 0.1333333f * src_data_51 + 0.0888888f * src_data_61;
    const float16_t t52 = 0.0333333333f * src_data_12 + 0.02222222f * src_data_22 - 0.1666666666f * src_data_32 -
                          0.1111111111f * src_data_42 + 0.1333333f * src_data_52 + 0.0888888f * src_data_62;
    const float16_t t53 = 0.0333333333f * src_data_13 + 0.02222222f * src_data_23 - 0.1666666666f * src_data_33 -
                          0.1111111111f * src_data_43 + 0.1333333f * src_data_53 + 0.0888888f * src_data_63;
    const float16_t t54 = 0.0333333333f * src_data_14 + 0.02222222f * src_data_24 - 0.1666666666f * src_data_34 -
                          0.1111111111f * src_data_44 + 0.1333333f * src_data_54 + 0.0888888f * src_data_64;
    const float16_t t55 = 0.0333333333f * src_data_15 + 0.02222222f * src_data_25 - 0.1666666666f * src_data_35 -
                          0.1111111111f * src_data_45 + 0.1333333f * src_data_55 + 0.0888888f * src_data_65;
    const float16_t t56 = 0.0333333333f * src_data_16 + 0.02222222f * src_data_26 - 0.1666666666f * src_data_36 -
                          0.1111111111f * src_data_46 + 0.1333333f * src_data_56 + 0.0888888f * src_data_66;
    const float16_t t57 = 0.0333333333f * src_data_17 + 0.02222222f * src_data_27 - 0.1666666666f * src_data_37 -
                          0.1111111111f * src_data_47 + 0.1333333f * src_data_57 + 0.0888888f * src_data_67;

    const float16_t t60 = -0.0333333333f * src_data_10 + 0.02222222f * src_data_20 + 0.1666666666f * src_data_30 -
                          0.1111111111f * src_data_40 - 0.1333333f * src_data_50 + 0.0888888f * src_data_60;
    const float16_t t61 = -0.0333333333f * src_data_11 + 0.02222222f * src_data_21 + 0.1666666666f * src_data_31 -
                          0.1111111111f * src_data_41 - 0.1333333f * src_data_51 + 0.0888888f * src_data_61;
    const float16_t t62 = -0.0333333333f * src_data_12 + 0.02222222f * src_data_22 + 0.1666666666f * src_data_32 -
                          0.1111111111f * src_data_42 - 0.1333333f * src_data_52 + 0.0888888f * src_data_62;
    const float16_t t63 = -0.0333333333f * src_data_13 + 0.02222222f * src_data_23 + 0.1666666666f * src_data_33 -
                          0.1111111111f * src_data_43 - 0.1333333f * src_data_53 + 0.0888888f * src_data_63;
    const float16_t t64 = -0.0333333333f * src_data_14 + 0.02222222f * src_data_24 + 0.1666666666f * src_data_34 -
                          0.1111111111f * src_data_44 - 0.1333333f * src_data_54 + 0.0888888f * src_data_64;
    const float16_t t65 = -0.0333333333f * src_data_15 + 0.02222222f * src_data_25 + 0.1666666666f * src_data_35 -
                          0.1111111111f * src_data_45 - 0.1333333f * src_data_55 + 0.0888888f * src_data_65;
    const float16_t t66 = -0.0333333333f * src_data_16 + 0.02222222f * src_data_26 + 0.1666666666f * src_data_36 -
                          0.1111111111f * src_data_46 - 0.1333333f * src_data_56 + 0.0888888f * src_data_66;
    const float16_t t67 = -0.0333333333f * src_data_17 + 0.02222222f * src_data_27 + 0.1666666666f * src_data_37 -
                          0.1111111111f * src_data_47 - 0.1333333f * src_data_57 + 0.0888888f * src_data_67;

    const float16_t t70 = -0.5625f * src_data_10 + 3.0625f * src_data_30 - 3.5f * src_data_50 + src_data_70;
    const float16_t t71 = -0.5625f * src_data_11 + 3.0625f * src_data_31 - 3.5f * src_data_51 + src_data_71;
    const float16_t t72 = -0.5625f * src_data_12 + 3.0625f * src_data_32 - 3.5f * src_data_52 + src_data_72;
    const float16_t t73 = -0.5625f * src_data_13 + 3.0625f * src_data_33 - 3.5f * src_data_53 + src_data_73;
    const float16_t t74 = -0.5625f * src_data_14 + 3.0625f * src_data_34 - 3.5f * src_data_54 + src_data_74;
    const float16_t t75 = -0.5625f * src_data_15 + 3.0625f * src_data_35 - 3.5f * src_data_55 + src_data_75;
    const float16_t t76 = -0.5625f * src_data_16 + 3.0625f * src_data_36 - 3.5f * src_data_56 + src_data_76;
    const float16_t t77 = -0.5625f * src_data_17 + 3.0625f * src_data_37 - 3.5f * src_data_57 + src_data_77;

    const float16_t m00 =
      t00 - 5.444444444444444445125f * t02 + 6.222222222222222222223f * t04 - 1.77777777777777778f * t06;
    const float16_t m01 = 1.5f * t01 + 3.0f * t02 - 2.1666666666666667f * t03 - 4.333333333333333333f * t04 +
                          0.66666666666666667f * t05 + 1.333333333333333f * t06;
    const float16_t m02 = -1.5f * t01 + 3.0f * t02 + 2.1666666666666667f * t03 - 4.333333333333333333f * t04 -
                          0.66666666666666667f * t05 + 1.333333333333333f * t06;
    const float16_t m03 = -0.3f * (t01 + t02) + 1.33333333333333f * (t03 + t04) - 0.53333333333f * (t05 + t06);
    const float16_t m04 = 0.3f * (t01 - t02) + 1.33333333333333f * (t04 - t03) + 0.53333333333f * (t05 - t06);
    const float16_t m05 = 0.0333333333f * t01 + 0.02222222f * t02 - 0.1666666666f * t03 - 0.1111111111f * t04 +
                          0.1333333f * t05 + 0.0888888f * t06;
    const float16_t m06 = -0.0333333333f * t01 + 0.02222222f * t02 + 0.1666666666f * t03 - 0.1111111111f * t04 -
                          0.1333333f * t05 + 0.0888888f * t06;
    const float16_t m07 = -0.5625f * t01 + 3.0625f * t03 - 3.5f * t05 + t07;

    float16_t m10 = t10 - 5.444444444444444445125f * t12 + 6.222222222222222222223f * t14 - 1.77777777777777778f * t16;
    const float16_t m11 = 1.5f * t11 + 3.0f * t12 - 2.1666666666666667f * t13 - 4.333333333333333333f * t14 +
                          0.66666666666666667f * t15 + 1.333333333333333f * t16;
    const float16_t m12 = -1.5f * t11 + 3.0f * t12 + 2.1666666666666667f * t13 - 4.333333333333333333f * t14 -
                          0.66666666666666667f * t15 + 1.333333333333333f * t16;
    const float16_t m13 = -0.3f * (t11 + t12) + 1.33333333333333f * (t13 + t14) - 0.53333333333f * (t15 + t16);
    const float16_t m14 = 0.3f * (t11 - t12) + 1.33333333333333f * (t14 - t13) + 0.53333333333f * (t15 - t16);
    const float16_t m15 = 0.0333333333f * t11 + 0.02222222f * t12 - 0.1666666666f * t13 - 0.1111111111f * t14 +
                          0.1333333f * t15 + 0.0888888f * t16;
    const float16_t m16 = -0.0333333333f * t11 + 0.02222222f * t12 + 0.1666666666f * t13 - 0.1111111111f * t14 -
                          0.1333333f * t15 + 0.0888888f * t16;
    const float16_t m17 = -0.5625f * t11 + 3.0625f * t13 - 3.5f * t15 + t17;

    const float16_t m20 =
      t20 - 5.444444444444444445125f * t22 + 6.222222222222222222223f * t24 - 1.77777777777777778f * t26;
    const float16_t m21 = 1.5f * t21 + 3.0f * t22 - 2.1666666666666667f * t23 - 4.333333333333333333f * t24 +
                          0.66666666666666667f * t25 + 1.333333333333333f * t26;
    const float16_t m22 = -1.5f * t21 + 3.0f * t22 + 2.1666666666666667f * t23 - 4.333333333333333333f * t24 -
                          0.66666666666666667f * t25 + 1.333333333333333f * t26;
    const float16_t m23 = -0.3f * (t21 + t22) + 1.33333333333333f * (t23 + t24) - 0.53333333333f * (t25 + t26);
    const float16_t m24 = 0.3f * (t21 - t22) + 1.33333333333333f * (t24 - t23) + 0.53333333333f * (t25 - t26);
    const float16_t m25 = 0.0333333333f * t21 + 0.02222222f * t22 - 0.1666666666f * t23 - 0.1111111111f * t24 +
                          0.1333333f * t25 + 0.0888888f * t26;
    const float16_t m26 = -0.0333333333f * t21 + 0.02222222f * t22 + 0.1666666666f * t23 - 0.1111111111f * t24 -
                          0.1333333f * t25 + 0.0888888f * t26;
    const float16_t m27 = -0.5625f * t21 + 3.0625f * t23 - 3.5f * t25 + t27;

    float16_t m30 = t30 - 5.444444444444444445125f * t32 + 6.222222222222222222223f * t34 - 1.77777777777777778f * t36;
    const float16_t m31 = 1.5f * t31 + 3.0f * t32 - 2.1666666666666667f * t33 - 4.333333333333333333f * t34 +
                          0.66666666666666667f * t35 + 1.333333333333333f * t36;
    const float16_t m32 = -1.5f * t31 + 3.0f * t32 + 2.1666666666666667f * t33 - 4.333333333333333333f * t34 -
                          0.66666666666666667f * t35 + 1.333333333333333f * t36;
    const float16_t m33 = -0.3f * (t31 + t32) + 1.33333333333333f * (t33 + t34) - 0.53333333333f * (t35 + t36);
    const float16_t m34 = 0.3f * (t31 - t32) + 1.33333333333333f * (t34 - t33) + 0.53333333333f * (t35 - t36);
    const float16_t m35 = 0.0333333333f * t31 + 0.02222222f * t32 - 0.1666666666f * t33 - 0.1111111111f * t34 +
                          0.1333333f * t35 + 0.0888888f * t36;
    const float16_t m36 = -0.0333333333f * t31 + 0.02222222f * t32 + 0.1666666666f * t33 - 0.1111111111f * t34 -
                          0.1333333f * t35 + 0.0888888f * t36;
    const float16_t m37 = -0.5625f * t31 + 3.0625f * t33 - 3.5f * t35 + t37;

    const float16_t m40 =
      t40 - 5.444444444444444445125f * t42 + 6.222222222222222222223f * t44 - 1.77777777777777778f * t46;
    const float16_t m41 = 1.5f * t41 + 3.0f * t42 - 2.1666666666666667f * t43 - 4.333333333333333333f * t44 +
                          0.66666666666666667f * t45 + 1.333333333333333f * t46;
    const float16_t m42 = -1.5f * t41 + 3.0f * t42 + 2.1666666666666667f * t43 - 4.333333333333333333f * t44 -
                          0.66666666666666667f * t45 + 1.333333333333333f * t46;
    const float16_t m43 = -0.3f * (t41 + t42) + 1.33333333333333f * (t43 + t44) - 0.53333333333f * (t45 + t46);
    const float16_t m44 = 0.3f * (t41 - t42) + 1.33333333333333f * (t44 - t43) + 0.53333333333f * (t45 - t46);
    const float16_t m45 = 0.0333333333f * t41 + 0.02222222f * t42 - 0.1666666666f * t43 - 0.1111111111f * t44 +
                          0.1333333f * t45 + 0.0888888f * t46;
    const float16_t m46 = -0.0333333333f * t41 + 0.02222222f * t42 + 0.1666666666f * t43 - 0.1111111111f * t44 -
                          0.1333333f * t45 + 0.0888888f * t46;
    const float16_t m47 = -0.5625f * t41 + 3.0625f * t43 - 3.5f * t45 + t47;

    float16_t m50 = t50 - 5.444444444444444445125f * t52 + 6.222222222222222222223f * t54 - 1.77777777777777778f * t56;
    const float16_t m51 = 1.5f * t51 + 3.0f * t52 - 2.1666666666666667f * t53 - 4.333333333333333333f * t54 +
                          0.66666666666666667f * t55 + 1.333333333333333f * t56;
    const float16_t m52 = -1.5f * t51 + 3.0f * t52 + 2.1666666666666667f * t53 - 4.333333333333333333f * t54 -
                          0.66666666666666667f * t55 + 1.333333333333333f * t56;
    const float16_t m53 = -0.3f * (t51 + t52) + 1.33333333333333f * (t53 + t54) - 0.53333333333f * (t55 + t56);
    const float16_t m54 = 0.3f * (t51 - t52) + 1.33333333333333f * (t54 - t53) + 0.53333333333f * (t55 - t56);
    const float16_t m55 = 0.0333333333f * t51 + 0.02222222f * t52 - 0.1666666666f * t53 - 0.1111111111f * t54 +
                          0.1333333f * t55 + 0.0888888f * t56;
    const float16_t m56 = -0.0333333333f * t51 + 0.02222222f * t52 + 0.1666666666f * t53 - 0.1111111111f * t54 -
                          0.1333333f * t55 + 0.0888888f * t56;
    const float16_t m57 = -0.5625f * t51 + 3.0625f * t53 - 3.5f * t55 + t57;

    float16_t m60 = t60 - 5.444444444444444445125f * t62 + 6.222222222222222222223f * t64 - 1.77777777777777778f * t66;
    const float16_t m61 = 1.5f * t61 + 3.0f * t62 - 2.1666666666666667f * t63 - 4.333333333333333333f * t64 +
                          0.66666666666666667f * t65 + 1.333333333333333f * t66;
    const float16_t m62 = -1.5f * t61 + 3.0f * t62 + 2.1666666666666667f * t63 - 4.333333333333333333f * t64 -
                          0.66666666666666667f * t65 + 1.333333333333333f * t66;
    const float16_t m63 = -0.3f * (t61 + t62) + 1.33333333333333f * (t63 + t64) - 0.53333333333f * (t65 + t66);
    const float16_t m64 = 0.3f * (t61 - t62) + 1.33333333333333f * (t64 - t63) + 0.53333333333f * (t65 - t66);
    const float16_t m65 = 0.0333333333f * t61 + 0.02222222f * t62 - 0.1666666666f * t63 - 0.1111111111f * t64 +
                          0.1333333f * t65 + 0.0888888f * t66;
    const float16_t m66 = -0.0333333333f * t61 + 0.02222222f * t62 + 0.1666666666f * t63 - 0.1111111111f * t64 -
                          0.1333333f * t65 + 0.0888888f * t66;
    const float16_t m67 = -0.5625f * t61 + 3.0625f * t63 - 3.5f * t65 + t67;

    float16_t m70 = t70 - 5.444444444444444445125f * t72 + 6.222222222222222222223f * t74 - 1.77777777777777778f * t76;
    const float16_t m71 = 1.5f * t71 + 3.0f * t72 - 2.1666666666666667f * t73 - 4.333333333333333333f * t74 +
                          0.66666666666666667f * t75 + 1.333333333333333f * t76;
    const float16_t m72 = -1.5f * t71 + 3.0f * t72 + 2.1666666666666667f * t73 - 4.333333333333333333f * t74 -
                          0.66666666666666667f * t75 + 1.333333333333333f * t76;
    const float16_t m73 = -0.3f * (t71 + t72) + 1.33333333333333f * (t73 + t74) - 0.53333333333f * (t75 + t76);
    const float16_t m74 = 0.3f * (t71 - t72) + 1.33333333333333f * (t74 - t73) + 0.53333333333f * (t75 - t76);
    const float16_t m75 = 0.0333333333f * t71 + 0.02222222f * t72 - 0.1666666666f * t73 - 0.1111111111f * t74 +
                          0.1333333f * t75 + 0.0888888f * t76;
    const float16_t m76 = -0.0333333333f * t71 + 0.02222222f * t72 + 0.1666666666f * t73 - 0.1111111111f * t74 -
                          0.1333333f * t75 + 0.0888888f * t76;
    const float16_t m77 = -0.5625f * t71 + 3.0625f * t73 - 3.5f * t75 + t77;

    (dst_data + i)[0] = m00;
    (dst_data + i + dst_step)[0] = m01;
    (dst_data + i + 2 * dst_step)[0] = m02;
    (dst_data + i + 3 * dst_step)[0] = m03;
    (dst_data + i + 4 * dst_step)[0] = m04;
    (dst_data + i + 5 * dst_step)[0] = m05;
    (dst_data + i + 6 * dst_step)[0] = m06;
    (dst_data + i + 7 * dst_step)[0] = m07;

    (dst_data + i + 8 * dst_step)[0] = m10;
    (dst_data + i + 9 * dst_step)[0] = m11;
    (dst_data + i + 10 * dst_step)[0] = m12;
    (dst_data + i + 11 * dst_step)[0] = m13;
    (dst_data + i + 12 * dst_step)[0] = m14;
    (dst_data + i + 13 * dst_step)[0] = m15;
    (dst_data + i + 14 * dst_step)[0] = m16;
    (dst_data + i + 15 * dst_step)[0] = m17;

    (dst_data + i + 16 * dst_step)[0] = m20;
    (dst_data + i + 17 * dst_step)[0] = m21;
    (dst_data + i + 18 * dst_step)[0] = m22;
    (dst_data + i + 19 * dst_step)[0] = m23;
    (dst_data + i + 20 * dst_step)[0] = m24;
    (dst_data + i + 21 * dst_step)[0] = m25;
    (dst_data + i + 22 * dst_step)[0] = m26;
    (dst_data + i + 23 * dst_step)[0] = m27;

    (dst_data + i + 24 * dst_step)[0] = m30;
    (dst_data + i + 25 * dst_step)[0] = m31;
    (dst_data + i + 26 * dst_step)[0] = m32;
    (dst_data + i + 27 * dst_step)[0] = m33;
    (dst_data + i + 28 * dst_step)[0] = m34;
    (dst_data + i + 29 * dst_step)[0] = m35;
    (dst_data + i + 30 * dst_step)[0] = m36;
    (dst_data + i + 31 * dst_step)[0] = m37;

    (dst_data + i + 32 * dst_step)[0] = m40;
    (dst_data + i + 33 * dst_step)[0] = m41;
    (dst_data + i + 34 * dst_step)[0] = m42;
    (dst_data + i + 35 * dst_step)[0] = m43;
    (dst_data + i + 36 * dst_step)[0] = m44;
    (dst_data + i + 37 * dst_step)[0] = m45;
    (dst_data + i + 38 * dst_step)[0] = m46;
    (dst_data + i + 39 * dst_step)[0] = m47;

    (dst_data + i + 40 * dst_step)[0] = m50;
    (dst_data + i + 41 * dst_step)[0] = m51;
    (dst_data + i + 42 * dst_step)[0] = m52;
    (dst_data + i + 43 * dst_step)[0] = m53;
    (dst_data + i + 44 * dst_step)[0] = m54;
    (dst_data + i + 45 * dst_step)[0] = m55;
    (dst_data + i + 46 * dst_step)[0] = m56;
    (dst_data + i + 47 * dst_step)[0] = m57;

    (dst_data + i + 48 * dst_step)[0] = m60;
    (dst_data + i + 49 * dst_step)[0] = m61;
    (dst_data + i + 50 * dst_step)[0] = m62;
    (dst_data + i + 51 * dst_step)[0] = m63;
    (dst_data + i + 52 * dst_step)[0] = m64;
    (dst_data + i + 53 * dst_step)[0] = m65;
    (dst_data + i + 54 * dst_step)[0] = m66;
    (dst_data + i + 55 * dst_step)[0] = m67;

    (dst_data + i + 56 * dst_step)[0] = m70;
    (dst_data + i + 57 * dst_step)[0] = m71;
    (dst_data + i + 58 * dst_step)[0] = m72;
    (dst_data + i + 59 * dst_step)[0] = m73;
    (dst_data + i + 60 * dst_step)[0] = m74;
    (dst_data + i + 61 * dst_step)[0] = m75;
    (dst_data + i + 62 * dst_step)[0] = m76;
    (dst_data + i + 63 * dst_step)[0] = m77;
  }
#endif
}

void OutputTransform4x2UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step) {
#ifdef ENABLE_ARM
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  float16x8_t src_data_00 = vld1q_f16(src_data + 0 * src_step);
  float16x8_t src_data_01 = vld1q_f16(src_data + 1 * src_step);
  float16x8_t src_data_02 = vld1q_f16(src_data + 2 * src_step);
  float16x8_t src_data_03 = vld1q_f16(src_data + 3 * src_step);
  float16x8_t src_data_10 = vld1q_f16(src_data + 4 * src_step);
  float16x8_t src_data_11 = vld1q_f16(src_data + 5 * src_step);
  float16x8_t src_data_12 = vld1q_f16(src_data + 6 * src_step);
  float16x8_t src_data_13 = vld1q_f16(src_data + 7 * src_step);
  float16x8_t src_data_20 = vld1q_f16(src_data + 8 * src_step);
  float16x8_t src_data_21 = vld1q_f16(src_data + 9 * src_step);
  float16x8_t src_data_22 = vld1q_f16(src_data + 10 * src_step);
  float16x8_t src_data_23 = vld1q_f16(src_data + 11 * src_step);
  float16x8_t src_data_30 = vld1q_f16(src_data + 12 * src_step);
  float16x8_t src_data_31 = vld1q_f16(src_data + 13 * src_step);
  float16x8_t src_data_32 = vld1q_f16(src_data + 14 * src_step);
  float16x8_t src_data_33 = vld1q_f16(src_data + 15 * src_step);

  float16x8_t t00 = vaddq_f16(src_data_00, vaddq_f16(src_data_10, src_data_20));
  float16x8_t t01 = vaddq_f16(src_data_01, vaddq_f16(src_data_11, src_data_21));
  float16x8_t t02 = vaddq_f16(src_data_02, vaddq_f16(src_data_12, src_data_22));
  float16x8_t t03 = vaddq_f16(src_data_03, vaddq_f16(src_data_13, src_data_23));

  float16x8_t t10 = vsubq_f16(src_data_30, vmulq_n_f16(vsubq_f16(src_data_10, src_data_20), 0.5));
  float16x8_t t11 = vsubq_f16(src_data_31, vmulq_n_f16(vsubq_f16(src_data_11, src_data_21), 0.5));
  float16x8_t t12 = vsubq_f16(src_data_32, vmulq_n_f16(vsubq_f16(src_data_12, src_data_22), 0.5));
  float16x8_t t13 = vsubq_f16(src_data_33, vmulq_n_f16(vsubq_f16(src_data_13, src_data_23), 0.5));

  float16x8_t m00 = vaddq_f16(vaddq_f16(t00, vaddq_f16(t01, t02)), bias_ptr);
  float16x8_t m01 = vaddq_f16(vaddq_f16(t03, vmulq_n_f16(vsubq_f16(t01, t02), 0.5)), bias_ptr);
  float16x8_t m10 = vaddq_f16(vaddq_f16(t10, vaddq_f16(t11, t12)), bias_ptr);
  float16x8_t m11 = vaddq_f16(vaddq_f16(t13, vmulq_n_f16(vsubq_f16(t11, t12), 0.5)), bias_ptr);

  vst1q_f16(dst_data, m00);
  vst1q_f16(dst_data + C8NUM, m01);
  vst1q_f16(dst_data + dst_step * C8NUM, m10);
  vst1q_f16(dst_data + dst_step * C8NUM + C8NUM, m11);
#else
  for (int i = 0; i < C8NUM; i++) {
    float16_t src_data_00 = src_data[i];
    float16_t src_data_01 = src_data[i + src_step];
    float16_t src_data_02 = src_data[i + 2 * src_step];
    float16_t src_data_03 = src_data[i + 3 * src_step];
    float16_t src_data_10 = src_data[i + 4 * src_step];
    float16_t src_data_11 = src_data[i + 5 * src_step];
    float16_t src_data_12 = src_data[i + 6 * src_step];
    float16_t src_data_13 = src_data[i + 7 * src_step];
    float16_t src_data_20 = src_data[i + 8 * src_step];
    float16_t src_data_21 = src_data[i + 9 * src_step];
    float16_t src_data_22 = src_data[i + 10 * src_step];
    float16_t src_data_23 = src_data[i + 11 * src_step];
    float16_t src_data_30 = src_data[i + 12 * src_step];
    float16_t src_data_31 = src_data[i + 13 * src_step];
    float16_t src_data_32 = src_data[i + 14 * src_step];
    float16_t src_data_33 = src_data[i + 15 * src_step];

    float16_t t00 = src_data_00 + src_data_10 + src_data_20;
    float16_t t01 = src_data_01 + src_data_11 + src_data_21;
    float16_t t02 = src_data_02 + src_data_12 + src_data_22;
    float16_t t03 = src_data_03 + src_data_13 + src_data_23;

    const float16_t t10 = 0.5f * (src_data_10 - src_data_20) + src_data_30;
    const float16_t t11 = 0.5f * (src_data_11 - src_data_21) + src_data_31;
    const float16_t t12 = 0.5f * (src_data_12 - src_data_22) + src_data_32;
    const float16_t t13 = 0.5f * (src_data_13 - src_data_23) + src_data_33;

    float16_t m00 = t00 + t01 + t02 + bias_data[i];
    const float16_t m01 = 0.5f * (t01 - t02) + t03 + bias_data[i];
    float16_t m10 = t10 + t11 + t12 + bias_data[i];
    const float16_t m11 = 0.5f * (t11 - t12) + t13 + bias_data[i];

    (dst_data + i)[0] = m00;
    (dst_data + i + C8NUM)[0] = m01;
    (dst_data + i + dst_step * C8NUM)[0] = m10;
    (dst_data + i + dst_step * C8NUM + C8NUM)[0] = m11;
  }
#endif
}

void OutputTransform4x3UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step) {
#ifdef ENABLE_ARM
  float16x8_t bias_ptr = vld1q_f16(bias_data);
  float16x8_t src_data_00 = vld1q_f16(src_data + 0 * src_step);
  float16x8_t src_data_01 = vld1q_f16(src_data + 1 * src_step);
  float16x8_t src_data_02 = vld1q_f16(src_data + 2 * src_step);
  float16x8_t src_data_03 = vld1q_f16(src_data + 3 * src_step);
  float16x8_t src_data_10 = vld1q_f16(src_data + 4 * src_step);
  float16x8_t src_data_11 = vld1q_f16(src_data + 5 * src_step);
  float16x8_t src_data_12 = vld1q_f16(src_data + 6 * src_step);
  float16x8_t src_data_13 = vld1q_f16(src_data + 7 * src_step);
  float16x8_t src_data_20 = vld1q_f16(src_data + 8 * src_step);
  float16x8_t src_data_21 = vld1q_f16(src_data + 9 * src_step);
  float16x8_t src_data_22 = vld1q_f16(src_data + 10 * src_step);
  float16x8_t src_data_23 = vld1q_f16(src_data + 11 * src_step);
  float16x8_t src_data_30 = vld1q_f16(src_data + 12 * src_step);
  float16x8_t src_data_31 = vld1q_f16(src_data + 13 * src_step);
  float16x8_t src_data_32 = vld1q_f16(src_data + 14 * src_step);
  float16x8_t src_data_33 = vld1q_f16(src_data + 15 * src_step);

  float16x8_t t00 = vaddq_f16(src_data_00, vaddq_f16(src_data_10, src_data_20));
  float16x8_t t01 = vaddq_f16(src_data_01, vaddq_f16(src_data_11, src_data_21));
  float16x8_t t02 = vaddq_f16(src_data_02, vaddq_f16(src_data_12, src_data_22));
  float16x8_t t03 = vaddq_f16(src_data_03, vaddq_f16(src_data_13, src_data_23));

  float16x8_t t10 = vmulq_n_f16(vsubq_f16(src_data_10, src_data_20), 0.5);
  float16x8_t t11 = vmulq_n_f16(vsubq_f16(src_data_11, src_data_21), 0.5);
  float16x8_t t12 = vmulq_n_f16(vsubq_f16(src_data_12, src_data_22), 0.5);
  float16x8_t t13 = vmulq_n_f16(vsubq_f16(src_data_13, src_data_23), 0.5);

  float16x8_t t20 = vaddq_f16(src_data_30, vmulq_n_f16(vaddq_f16(src_data_10, src_data_20), 0.25));
  float16x8_t t21 = vaddq_f16(src_data_31, vmulq_n_f16(vaddq_f16(src_data_11, src_data_21), 0.25));
  float16x8_t t22 = vaddq_f16(src_data_32, vmulq_n_f16(vaddq_f16(src_data_12, src_data_22), 0.25));
  float16x8_t t23 = vaddq_f16(src_data_33, vmulq_n_f16(vaddq_f16(src_data_13, src_data_23), 0.25));

  float16x8_t m00 = vaddq_f16(vaddq_f16(t00, vaddq_f16(t01, t02)), bias_ptr);
  float16x8_t m01 = vaddq_f16(vmulq_n_f16(vsubq_f16(t01, t02), 0.5), bias_ptr);
  float16x8_t m02 = vaddq_f16(vaddq_f16(t03, vmulq_n_f16(vaddq_f16(t01, t02), 0.25)), bias_ptr);
  float16x8_t m10 = vaddq_f16(vaddq_f16(t10, vaddq_f16(t11, t12)), bias_ptr);
  float16x8_t m11 = vaddq_f16(vmulq_n_f16(vsubq_f16(t11, t12), 0.5), bias_ptr);
  float16x8_t m12 = vaddq_f16(vaddq_f16(t13, vmulq_n_f16(vaddq_f16(t11, t12), 0.25)), bias_ptr);
  float16x8_t m20 = vaddq_f16(vaddq_f16(t20, vaddq_f16(t21, t22)), bias_ptr);
  float16x8_t m21 = vaddq_f16(vmulq_n_f16(vsubq_f16(t21, t22), 0.5), bias_ptr);
  float16x8_t m22 = vaddq_f16(vaddq_f16(t23, vmulq_n_f16(vaddq_f16(t21, t22), 0.25)), bias_ptr);

  vst1q_f16(dst_data, m00);
  vst1q_f16(dst_data + C8NUM, m01);
  vst1q_f16(dst_data + 2 * C8NUM, m02);
  vst1q_f16(dst_data + dst_step * C8NUM, m10);
  vst1q_f16(dst_data + dst_step * C8NUM + C8NUM, m11);
  vst1q_f16(dst_data + dst_step * C8NUM + 2 * C8NUM, m12);
  vst1q_f16(dst_data + 2 * dst_step * C8NUM, m20);
  vst1q_f16(dst_data + 2 * dst_step * C8NUM + C8NUM, m21);
  vst1q_f16(dst_data + 2 * dst_step * C8NUM + 2 * C8NUM, m22);
#else
  for (int i = 0; i < C8NUM; i++) {
    float16_t src_data_00 = src_data[i];
    float16_t src_data_01 = src_data[i + src_step];
    float16_t src_data_02 = src_data[i + 2 * src_step];
    float16_t src_data_03 = src_data[i + 3 * src_step];
    float16_t src_data_10 = src_data[i + 4 * src_step];
    float16_t src_data_11 = src_data[i + 5 * src_step];
    float16_t src_data_12 = src_data[i + 6 * src_step];
    float16_t src_data_13 = src_data[i + 7 * src_step];
    float16_t src_data_20 = src_data[i + 8 * src_step];
    float16_t src_data_21 = src_data[i + 9 * src_step];
    float16_t src_data_22 = src_data[i + 10 * src_step];
    float16_t src_data_23 = src_data[i + 11 * src_step];
    float16_t src_data_30 = src_data[i + 12 * src_step];
    float16_t src_data_31 = src_data[i + 13 * src_step];
    float16_t src_data_32 = src_data[i + 14 * src_step];
    float16_t src_data_33 = src_data[i + 15 * src_step];

    float16_t t00 = src_data_00 + src_data_10 + src_data_20;
    float16_t t01 = src_data_01 + src_data_11 + src_data_21;
    float16_t t02 = src_data_02 + src_data_12 + src_data_22;
    float16_t t03 = src_data_03 + src_data_13 + src_data_23;

    const float16_t t10 = 0.5f * (src_data_10 - src_data_20);
    const float16_t t11 = 0.5f * (src_data_11 - src_data_21);
    const float16_t t12 = 0.5f * (src_data_12 - src_data_22);
    const const float16_t t13 = 0.5f * (src_data_13 - src_data_23);

    const float16_t t20 = 0.25f * (src_data_10 + src_data_20) + src_data_30;
    const float16_t t21 = 0.25f * (src_data_11 + src_data_21) + src_data_31;
    const float16_t t22 = 0.25f * (src_data_12 + src_data_22) + src_data_32;
    const float16_t t23 = 0.25f * (src_data_13 + src_data_23) + src_data_33;

    float16_t m00 = t00 + t01 + t02 + bias_data[i];
    const float16_t m01 = 0.5f * (t01 - t02) + bias_data[i];
    const float16_t m02 = 0.25f * (t01 + t02) + t03 + bias_data[i];

    float16_t m10 = t10 + t11 + t12 + bias_data[i];
    const float16_t m11 = 0.5f * (t11 - t12) + bias_data[i];
    const float16_t m12 = 0.25f * (t11 + t12) + t13 + bias_data[i];

    float16_t m20 = t20 + t21 + t22 + bias_data[i];
    const float16_t m21 = 0.5f * (t21 - t22) + bias_data[i];
    const float16_t m22 = 0.25f * (t21 + t22) + t23 + bias_data[i];

    (dst_data + i)[0] = m00;
    (dst_data + i + C8NUM)[0] = m01;
    (dst_data + i + 2 * C8NUM)[0] = m02;

    (dst_data + i + dst_step * C8NUM)[0] = m10;
    (dst_data + i + dst_step * C8NUM + C8NUM)[0] = m11;
    (dst_data + i + dst_step * C8NUM + 2 * C8NUM)[0] = m12;

    (dst_data + i + 2 * dst_step * C8NUM)[0] = m20;
    (dst_data + i + 2 * dst_step * C8NUM + C8NUM)[0] = m21;
    (dst_data + i + 2 * dst_step * C8NUM + 2 * C8NUM)[0] = m22;
  }
#endif
}

void OutputTransform8x2UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step) {
#ifdef ENABLE_ARM
  float16x8_t src_data_00 = vld1q_f16(src_data + 0 * src_step);
  float16x8_t src_data_01 = vld1q_f16(src_data + 1 * src_step);
  float16x8_t src_data_02 = vld1q_f16(src_data + 2 * src_step);
  float16x8_t src_data_03 = vld1q_f16(src_data + 3 * src_step);
  float16x8_t src_data_04 = vld1q_f16(src_data + 4 * src_step);
  float16x8_t src_data_05 = vld1q_f16(src_data + 5 * src_step);
  float16x8_t src_data_06 = vld1q_f16(src_data + 6 * src_step);
  float16x8_t src_data_07 = vld1q_f16(src_data + 7 * src_step);
  float16x8_t src_data_10 = vld1q_f16(src_data + 8 * src_step);
  float16x8_t src_data_11 = vld1q_f16(src_data + 9 * src_step);
  float16x8_t src_data_12 = vld1q_f16(src_data + 10 * src_step);
  float16x8_t src_data_13 = vld1q_f16(src_data + 11 * src_step);
  float16x8_t src_data_14 = vld1q_f16(src_data + 12 * src_step);
  float16x8_t src_data_15 = vld1q_f16(src_data + 13 * src_step);
  float16x8_t src_data_16 = vld1q_f16(src_data + 14 * src_step);
  float16x8_t src_data_17 = vld1q_f16(src_data + 15 * src_step);
  float16x8_t src_data_20 = vld1q_f16(src_data + 16 * src_step);
  float16x8_t src_data_21 = vld1q_f16(src_data + 17 * src_step);
  float16x8_t src_data_22 = vld1q_f16(src_data + 18 * src_step);
  float16x8_t src_data_23 = vld1q_f16(src_data + 19 * src_step);
  float16x8_t src_data_24 = vld1q_f16(src_data + 20 * src_step);
  float16x8_t src_data_25 = vld1q_f16(src_data + 21 * src_step);
  float16x8_t src_data_26 = vld1q_f16(src_data + 22 * src_step);
  float16x8_t src_data_27 = vld1q_f16(src_data + 23 * src_step);
  float16x8_t src_data_30 = vld1q_f16(src_data + 24 * src_step);
  float16x8_t src_data_31 = vld1q_f16(src_data + 25 * src_step);
  float16x8_t src_data_32 = vld1q_f16(src_data + 26 * src_step);
  float16x8_t src_data_33 = vld1q_f16(src_data + 27 * src_step);
  float16x8_t src_data_34 = vld1q_f16(src_data + 28 * src_step);
  float16x8_t src_data_35 = vld1q_f16(src_data + 29 * src_step);
  float16x8_t src_data_36 = vld1q_f16(src_data + 30 * src_step);
  float16x8_t src_data_37 = vld1q_f16(src_data + 31 * src_step);
  float16x8_t src_data_40 = vld1q_f16(src_data + 32 * src_step);
  float16x8_t src_data_41 = vld1q_f16(src_data + 33 * src_step);
  float16x8_t src_data_42 = vld1q_f16(src_data + 34 * src_step);
  float16x8_t src_data_43 = vld1q_f16(src_data + 35 * src_step);
  float16x8_t src_data_44 = vld1q_f16(src_data + 36 * src_step);
  float16x8_t src_data_45 = vld1q_f16(src_data + 37 * src_step);
  float16x8_t src_data_46 = vld1q_f16(src_data + 38 * src_step);
  float16x8_t src_data_47 = vld1q_f16(src_data + 39 * src_step);
  float16x8_t src_data_50 = vld1q_f16(src_data + 40 * src_step);
  float16x8_t src_data_51 = vld1q_f16(src_data + 41 * src_step);
  float16x8_t src_data_52 = vld1q_f16(src_data + 42 * src_step);
  float16x8_t src_data_53 = vld1q_f16(src_data + 43 * src_step);
  float16x8_t src_data_54 = vld1q_f16(src_data + 44 * src_step);
  float16x8_t src_data_55 = vld1q_f16(src_data + 45 * src_step);
  float16x8_t src_data_56 = vld1q_f16(src_data + 46 * src_step);
  float16x8_t src_data_57 = vld1q_f16(src_data + 47 * src_step);
  float16x8_t src_data_60 = vld1q_f16(src_data + 48 * src_step);
  float16x8_t src_data_61 = vld1q_f16(src_data + 49 * src_step);
  float16x8_t src_data_62 = vld1q_f16(src_data + 50 * src_step);
  float16x8_t src_data_63 = vld1q_f16(src_data + 51 * src_step);
  float16x8_t src_data_64 = vld1q_f16(src_data + 52 * src_step);
  float16x8_t src_data_65 = vld1q_f16(src_data + 53 * src_step);
  float16x8_t src_data_66 = vld1q_f16(src_data + 54 * src_step);
  float16x8_t src_data_67 = vld1q_f16(src_data + 55 * src_step);
  float16x8_t src_data_70 = vld1q_f16(src_data + 56 * src_step);
  float16x8_t src_data_71 = vld1q_f16(src_data + 57 * src_step);
  float16x8_t src_data_72 = vld1q_f16(src_data + 58 * src_step);
  float16x8_t src_data_73 = vld1q_f16(src_data + 59 * src_step);
  float16x8_t src_data_74 = vld1q_f16(src_data + 60 * src_step);
  float16x8_t src_data_75 = vld1q_f16(src_data + 61 * src_step);
  float16x8_t src_data_76 = vld1q_f16(src_data + 62 * src_step);
  float16x8_t src_data_77 = vld1q_f16(src_data + 63 * src_step);

  float16x8_t d01 = vsubq_f16(src_data_10, src_data_20);
  float16x8_t d02 = vsubq_f16(src_data_11, src_data_21);
  float16x8_t d03 = vsubq_f16(src_data_12, src_data_22);
  float16x8_t d04 = vsubq_f16(src_data_13, src_data_23);
  float16x8_t d05 = vsubq_f16(src_data_14, src_data_24);
  float16x8_t d06 = vsubq_f16(src_data_15, src_data_25);
  float16x8_t d07 = vsubq_f16(src_data_16, src_data_26);
  float16x8_t d08 = vsubq_f16(src_data_17, src_data_27);

  float16x8_t d11 = vsubq_f16(src_data_30, src_data_40);
  float16x8_t d12 = vsubq_f16(src_data_31, src_data_41);
  float16x8_t d13 = vsubq_f16(src_data_32, src_data_42);
  float16x8_t d14 = vsubq_f16(src_data_33, src_data_43);
  float16x8_t d15 = vsubq_f16(src_data_34, src_data_44);
  float16x8_t d16 = vsubq_f16(src_data_35, src_data_45);
  float16x8_t d17 = vsubq_f16(src_data_36, src_data_46);
  float16x8_t d18 = vsubq_f16(src_data_37, src_data_47);

  float16x8_t d21 = vsubq_f16(src_data_50, src_data_60);
  float16x8_t d22 = vsubq_f16(src_data_51, src_data_61);
  float16x8_t d23 = vsubq_f16(src_data_52, src_data_62);
  float16x8_t d24 = vsubq_f16(src_data_53, src_data_63);
  float16x8_t d25 = vsubq_f16(src_data_54, src_data_64);
  float16x8_t d26 = vsubq_f16(src_data_55, src_data_65);
  float16x8_t d27 = vsubq_f16(src_data_56, src_data_66);
  float16x8_t d28 = vsubq_f16(src_data_57, src_data_67);

  float16x8_t t00 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_00, src_data_10), src_data_20), src_data_30), src_data_40),
      src_data_50),
    src_data_60);
  float16x8_t t01 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_01, src_data_11), src_data_21), src_data_31), src_data_41),
      src_data_51),
    src_data_61);
  float16x8_t t02 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_02, src_data_12), src_data_22), src_data_32), src_data_42),
      src_data_52),
    src_data_62);
  float16x8_t t03 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_03, src_data_13), src_data_23), src_data_33), src_data_43),
      src_data_53),
    src_data_63);
  float16x8_t t04 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_04, src_data_14), src_data_24), src_data_34), src_data_44),
      src_data_54),
    src_data_64);
  float16x8_t t05 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_05, src_data_15), src_data_25), src_data_35), src_data_45),
      src_data_55),
    src_data_65);
  float16x8_t t06 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_06, src_data_16), src_data_26), src_data_36), src_data_46),
      src_data_56),
    src_data_66);
  float16x8_t t07 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_07, src_data_17), src_data_27), src_data_37), src_data_47),
      src_data_57),
    src_data_67);

  float16x8_t t10 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d01, 0.5), d11), vmulq_n_f16(d21, 1.5)), src_data_70);
  float16x8_t t11 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d02, 0.5), d12), vmulq_n_f16(d22, 1.5)), src_data_71);
  float16x8_t t12 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d03, 0.5), d13), vmulq_n_f16(d23, 1.5)), src_data_72);
  float16x8_t t13 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d04, 0.5), d14), vmulq_n_f16(d24, 1.5)), src_data_73);
  float16x8_t t14 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d05, 0.5), d15), vmulq_n_f16(d25, 1.5)), src_data_74);
  float16x8_t t15 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d06, 0.5), d16), vmulq_n_f16(d26, 1.5)), src_data_75);
  float16x8_t t16 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d07, 0.5), d17), vmulq_n_f16(d27, 1.5)), src_data_76);
  float16x8_t t17 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d08, 0.5), d18), vmulq_n_f16(d28, 1.5)), src_data_77);

  float16x8_t s11 = vsubq_f16(t01, t02);
  float16x8_t s12 = vsubq_f16(t11, t12);

  float16x8_t s21 = vsubq_f16(t03, t04);
  float16x8_t s22 = vsubq_f16(t13, t14);

  float16x8_t s31 = vsubq_f16(t05, t06);
  float16x8_t s32 = vsubq_f16(t15, t16);

  float16x8_t m00 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t00, t01), t02), t03), t04), t05), t06);
  float16x8_t m01 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s11, 0.5), s21), vmulq_n_f16(s31, 1.5)), t07);

  float16x8_t m10 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t10, t11), t12), t13), t14), t15), t16);
  float16x8_t m11 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s12, 0.5), s22), vmulq_n_f16(s32, 1.5)), t17);

  float16x8_t bias_ptr = vld1q_f16(bias_data);
  vst1q_f16(dst_data, vaddq_f16(m00, bias_ptr));
  vst1q_f16(dst_data + C8NUM, vaddq_f16(m01, bias_ptr));

  vst1q_f16(dst_data + dst_step * C8NUM, vaddq_f16(m10, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM + C8NUM, vaddq_f16(m11, bias_ptr));
#else
  for (int i = 0; i < C8NUM; i++) {
    float16_t src_data_00 = src_data[i];
    float16_t src_data_01 = src_data[i + src_step];
    float16_t src_data_02 = src_data[i + 2 * src_step];
    float16_t src_data_03 = src_data[i + 3 * src_step];
    float16_t src_data_04 = src_data[i + 4 * src_step];
    float16_t src_data_05 = src_data[i + 5 * src_step];
    float16_t src_data_06 = src_data[i + 6 * src_step];
    float16_t src_data_07 = src_data[i + 7 * src_step];
    float16_t src_data_10 = src_data[i + 8 * src_step];
    float16_t src_data_11 = src_data[i + 9 * src_step];
    float16_t src_data_12 = src_data[i + 10 * src_step];
    float16_t src_data_13 = src_data[i + 11 * src_step];
    float16_t src_data_14 = src_data[i + 12 * src_step];
    float16_t src_data_15 = src_data[i + 13 * src_step];
    float16_t src_data_16 = src_data[i + 14 * src_step];
    float16_t src_data_17 = src_data[i + 15 * src_step];
    float16_t src_data_20 = src_data[i + 16 * src_step];
    float16_t src_data_21 = src_data[i + 17 * src_step];
    float16_t src_data_22 = src_data[i + 18 * src_step];
    float16_t src_data_23 = src_data[i + 19 * src_step];
    float16_t src_data_24 = src_data[i + 20 * src_step];
    float16_t src_data_25 = src_data[i + 21 * src_step];
    float16_t src_data_26 = src_data[i + 22 * src_step];
    float16_t src_data_27 = src_data[i + 23 * src_step];
    float16_t src_data_30 = src_data[i + 24 * src_step];
    float16_t src_data_31 = src_data[i + 25 * src_step];
    float16_t src_data_32 = src_data[i + 26 * src_step];
    float16_t src_data_33 = src_data[i + 27 * src_step];
    float16_t src_data_34 = src_data[i + 28 * src_step];
    float16_t src_data_35 = src_data[i + 29 * src_step];
    float16_t src_data_36 = src_data[i + 30 * src_step];
    float16_t src_data_37 = src_data[i + 31 * src_step];
    float16_t src_data_40 = src_data[i + 32 * src_step];
    float16_t src_data_41 = src_data[i + 33 * src_step];
    float16_t src_data_42 = src_data[i + 34 * src_step];
    float16_t src_data_43 = src_data[i + 35 * src_step];
    float16_t src_data_44 = src_data[i + 36 * src_step];
    float16_t src_data_45 = src_data[i + 37 * src_step];
    float16_t src_data_46 = src_data[i + 38 * src_step];
    float16_t src_data_47 = src_data[i + 39 * src_step];
    float16_t src_data_50 = src_data[i + 40 * src_step];
    float16_t src_data_51 = src_data[i + 41 * src_step];
    float16_t src_data_52 = src_data[i + 42 * src_step];
    float16_t src_data_53 = src_data[i + 43 * src_step];
    float16_t src_data_54 = src_data[i + 44 * src_step];
    float16_t src_data_55 = src_data[i + 45 * src_step];
    float16_t src_data_56 = src_data[i + 46 * src_step];
    float16_t src_data_57 = src_data[i + 47 * src_step];
    float16_t src_data_60 = src_data[i + 48 * src_step];
    float16_t src_data_61 = src_data[i + 49 * src_step];
    float16_t src_data_62 = src_data[i + 50 * src_step];
    float16_t src_data_63 = src_data[i + 51 * src_step];
    float16_t src_data_64 = src_data[i + 52 * src_step];
    float16_t src_data_65 = src_data[i + 53 * src_step];
    float16_t src_data_66 = src_data[i + 54 * src_step];
    float16_t src_data_67 = src_data[i + 55 * src_step];
    float16_t src_data_70 = src_data[i + 56 * src_step];
    float16_t src_data_71 = src_data[i + 57 * src_step];
    float16_t src_data_72 = src_data[i + 58 * src_step];
    float16_t src_data_73 = src_data[i + 59 * src_step];
    float16_t src_data_74 = src_data[i + 60 * src_step];
    float16_t src_data_75 = src_data[i + 61 * src_step];
    float16_t src_data_76 = src_data[i + 62 * src_step];
    float16_t src_data_77 = src_data[i + 63 * src_step];

    float16_t d01 = src_data_10 - src_data_20;
    float16_t d02 = src_data_11 - src_data_21;
    float16_t d03 = src_data_12 - src_data_22;
    float16_t d04 = src_data_13 - src_data_23;
    float16_t d05 = src_data_14 - src_data_24;
    float16_t d06 = src_data_15 - src_data_25;
    float16_t d07 = src_data_16 - src_data_26;
    float16_t d08 = src_data_17 - src_data_27;

    float16_t d11 = src_data_30 - src_data_40;
    float16_t d12 = src_data_31 - src_data_41;
    float16_t d13 = src_data_32 - src_data_42;
    float16_t d14 = src_data_33 - src_data_43;
    float16_t d15 = src_data_34 - src_data_44;
    float16_t d16 = src_data_35 - src_data_45;
    float16_t d17 = src_data_36 - src_data_46;
    float16_t d18 = src_data_37 - src_data_47;

    float16_t d21 = src_data_50 - src_data_60;
    float16_t d22 = src_data_51 - src_data_61;
    float16_t d23 = src_data_52 - src_data_62;
    float16_t d24 = src_data_53 - src_data_63;
    float16_t d25 = src_data_54 - src_data_64;
    float16_t d26 = src_data_55 - src_data_65;
    float16_t d27 = src_data_56 - src_data_66;
    float16_t d28 = src_data_57 - src_data_67;

    float16_t t00 = src_data_00 + src_data_10 + src_data_20 + src_data_30 + src_data_40 + src_data_50 + src_data_60;
    float16_t t01 = src_data_01 + src_data_11 + src_data_21 + src_data_31 + src_data_41 + src_data_51 + src_data_61;
    float16_t t02 = src_data_02 + src_data_12 + src_data_22 + src_data_32 + src_data_42 + src_data_52 + src_data_62;
    float16_t t03 = src_data_03 + src_data_13 + src_data_23 + src_data_33 + src_data_43 + src_data_53 + src_data_63;
    float16_t t04 = src_data_04 + src_data_14 + src_data_24 + src_data_34 + src_data_44 + src_data_54 + src_data_64;
    float16_t t05 = src_data_05 + src_data_15 + src_data_25 + src_data_35 + src_data_45 + src_data_55 + src_data_65;
    float16_t t06 = src_data_06 + src_data_16 + src_data_26 + src_data_36 + src_data_46 + src_data_56 + src_data_66;
    float16_t t07 = src_data_07 + src_data_17 + src_data_27 + src_data_37 + src_data_47 + src_data_57 + src_data_67;

    const float16_t t10 = 0.5f * d01 + d11 + 1.5f * d21 + src_data_70;
    const float16_t t11 = 0.5f * d02 + d12 + 1.5f * d22 + src_data_71;
    const float16_t t12 = 0.5f * d03 + d13 + 1.5f * d23 + src_data_72;
    const float16_t t13 = 0.5f * d04 + d14 + 1.5f * d24 + src_data_73;
    const float16_t t14 = 0.5f * d05 + d15 + 1.5f * d25 + src_data_74;
    const float16_t t15 = 0.5f * d06 + d16 + 1.5f * d26 + src_data_75;
    const float16_t t16 = 0.5f * d07 + d17 + 1.5f * d27 + src_data_76;
    const float16_t t17 = 0.5f * d08 + d18 + 1.5f * d28 + src_data_77;

    float16_t s11 = t01 - t02;
    float16_t s12 = t11 - t12;
    float16_t s21 = t03 - t04;
    float16_t s22 = t13 - t14;
    float16_t s31 = t05 - t06;
    float16_t s32 = t15 - t16;

    float16_t m00 = t00 + t01 + t02 + t03 + t04 + t05 + t06;
    const float16_t m01 = 0.5f * s11 + s21 + 1.5f * s31 + t07;
    float16_t m10 = t10 + t11 + t12 + t13 + t14 + t15 + t16;
    const float16_t m11 = 0.5f * s12 + s22 + 1.5f * s32 + t17;

    (dst_data + i)[0] = m00 + bias_data[i];
    (dst_data + i + C8NUM)[0] = m01 + bias_data[i];
    (dst_data + i + dst_step * C8NUM)[0] = m10 + bias_data[i];
    (dst_data + i + dst_step * C8NUM + C8NUM)[0] = m11 + bias_data[i];
  }
#endif
}

void OutputTransform8x3UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step) {
#ifdef ENABLE_ARM
  float16x8_t src_data_00 = vld1q_f16(src_data + 0 * src_step);
  float16x8_t src_data_01 = vld1q_f16(src_data + 1 * src_step);
  float16x8_t src_data_02 = vld1q_f16(src_data + 2 * src_step);
  float16x8_t src_data_03 = vld1q_f16(src_data + 3 * src_step);
  float16x8_t src_data_04 = vld1q_f16(src_data + 4 * src_step);
  float16x8_t src_data_05 = vld1q_f16(src_data + 5 * src_step);
  float16x8_t src_data_06 = vld1q_f16(src_data + 6 * src_step);
  float16x8_t src_data_07 = vld1q_f16(src_data + 7 * src_step);
  float16x8_t src_data_10 = vld1q_f16(src_data + 8 * src_step);
  float16x8_t src_data_11 = vld1q_f16(src_data + 9 * src_step);
  float16x8_t src_data_12 = vld1q_f16(src_data + 10 * src_step);
  float16x8_t src_data_13 = vld1q_f16(src_data + 11 * src_step);
  float16x8_t src_data_14 = vld1q_f16(src_data + 12 * src_step);
  float16x8_t src_data_15 = vld1q_f16(src_data + 13 * src_step);
  float16x8_t src_data_16 = vld1q_f16(src_data + 14 * src_step);
  float16x8_t src_data_17 = vld1q_f16(src_data + 15 * src_step);
  float16x8_t src_data_20 = vld1q_f16(src_data + 16 * src_step);
  float16x8_t src_data_21 = vld1q_f16(src_data + 17 * src_step);
  float16x8_t src_data_22 = vld1q_f16(src_data + 18 * src_step);
  float16x8_t src_data_23 = vld1q_f16(src_data + 19 * src_step);
  float16x8_t src_data_24 = vld1q_f16(src_data + 20 * src_step);
  float16x8_t src_data_25 = vld1q_f16(src_data + 21 * src_step);
  float16x8_t src_data_26 = vld1q_f16(src_data + 22 * src_step);
  float16x8_t src_data_27 = vld1q_f16(src_data + 23 * src_step);
  float16x8_t src_data_30 = vld1q_f16(src_data + 24 * src_step);
  float16x8_t src_data_31 = vld1q_f16(src_data + 25 * src_step);
  float16x8_t src_data_32 = vld1q_f16(src_data + 26 * src_step);
  float16x8_t src_data_33 = vld1q_f16(src_data + 27 * src_step);
  float16x8_t src_data_34 = vld1q_f16(src_data + 28 * src_step);
  float16x8_t src_data_35 = vld1q_f16(src_data + 29 * src_step);
  float16x8_t src_data_36 = vld1q_f16(src_data + 30 * src_step);
  float16x8_t src_data_37 = vld1q_f16(src_data + 31 * src_step);
  float16x8_t src_data_40 = vld1q_f16(src_data + 32 * src_step);
  float16x8_t src_data_41 = vld1q_f16(src_data + 33 * src_step);
  float16x8_t src_data_42 = vld1q_f16(src_data + 34 * src_step);
  float16x8_t src_data_43 = vld1q_f16(src_data + 35 * src_step);
  float16x8_t src_data_44 = vld1q_f16(src_data + 36 * src_step);
  float16x8_t src_data_45 = vld1q_f16(src_data + 37 * src_step);
  float16x8_t src_data_46 = vld1q_f16(src_data + 38 * src_step);
  float16x8_t src_data_47 = vld1q_f16(src_data + 39 * src_step);
  float16x8_t src_data_50 = vld1q_f16(src_data + 40 * src_step);
  float16x8_t src_data_51 = vld1q_f16(src_data + 41 * src_step);
  float16x8_t src_data_52 = vld1q_f16(src_data + 42 * src_step);
  float16x8_t src_data_53 = vld1q_f16(src_data + 43 * src_step);
  float16x8_t src_data_54 = vld1q_f16(src_data + 44 * src_step);
  float16x8_t src_data_55 = vld1q_f16(src_data + 45 * src_step);
  float16x8_t src_data_56 = vld1q_f16(src_data + 46 * src_step);
  float16x8_t src_data_57 = vld1q_f16(src_data + 47 * src_step);
  float16x8_t src_data_60 = vld1q_f16(src_data + 48 * src_step);
  float16x8_t src_data_61 = vld1q_f16(src_data + 49 * src_step);
  float16x8_t src_data_62 = vld1q_f16(src_data + 50 * src_step);
  float16x8_t src_data_63 = vld1q_f16(src_data + 51 * src_step);
  float16x8_t src_data_64 = vld1q_f16(src_data + 52 * src_step);
  float16x8_t src_data_65 = vld1q_f16(src_data + 53 * src_step);
  float16x8_t src_data_66 = vld1q_f16(src_data + 54 * src_step);
  float16x8_t src_data_67 = vld1q_f16(src_data + 55 * src_step);
  float16x8_t src_data_70 = vld1q_f16(src_data + 56 * src_step);
  float16x8_t src_data_71 = vld1q_f16(src_data + 57 * src_step);
  float16x8_t src_data_72 = vld1q_f16(src_data + 58 * src_step);
  float16x8_t src_data_73 = vld1q_f16(src_data + 59 * src_step);
  float16x8_t src_data_74 = vld1q_f16(src_data + 60 * src_step);
  float16x8_t src_data_75 = vld1q_f16(src_data + 61 * src_step);
  float16x8_t src_data_76 = vld1q_f16(src_data + 62 * src_step);
  float16x8_t src_data_77 = vld1q_f16(src_data + 63 * src_step);

  float16x8_t d01 = vsubq_f16(src_data_10, src_data_20);
  float16x8_t d02 = vsubq_f16(src_data_11, src_data_21);
  float16x8_t d03 = vsubq_f16(src_data_12, src_data_22);
  float16x8_t d04 = vsubq_f16(src_data_13, src_data_23);
  float16x8_t d05 = vsubq_f16(src_data_14, src_data_24);
  float16x8_t d06 = vsubq_f16(src_data_15, src_data_25);
  float16x8_t d07 = vsubq_f16(src_data_16, src_data_26);
  float16x8_t d08 = vsubq_f16(src_data_17, src_data_27);

  float16x8_t d11 = vsubq_f16(src_data_30, src_data_40);
  float16x8_t d12 = vsubq_f16(src_data_31, src_data_41);
  float16x8_t d13 = vsubq_f16(src_data_32, src_data_42);
  float16x8_t d14 = vsubq_f16(src_data_33, src_data_43);
  float16x8_t d15 = vsubq_f16(src_data_34, src_data_44);
  float16x8_t d16 = vsubq_f16(src_data_35, src_data_45);
  float16x8_t d17 = vsubq_f16(src_data_36, src_data_46);
  float16x8_t d18 = vsubq_f16(src_data_37, src_data_47);

  float16x8_t d21 = vsubq_f16(src_data_50, src_data_60);
  float16x8_t d22 = vsubq_f16(src_data_51, src_data_61);
  float16x8_t d23 = vsubq_f16(src_data_52, src_data_62);
  float16x8_t d24 = vsubq_f16(src_data_53, src_data_63);
  float16x8_t d25 = vsubq_f16(src_data_54, src_data_64);
  float16x8_t d26 = vsubq_f16(src_data_55, src_data_65);
  float16x8_t d27 = vsubq_f16(src_data_56, src_data_66);
  float16x8_t d28 = vsubq_f16(src_data_57, src_data_67);

  float16x8_t d31 = vaddq_f16(src_data_10, src_data_20);
  float16x8_t d32 = vaddq_f16(src_data_11, src_data_21);
  float16x8_t d33 = vaddq_f16(src_data_12, src_data_22);
  float16x8_t d34 = vaddq_f16(src_data_13, src_data_23);
  float16x8_t d35 = vaddq_f16(src_data_14, src_data_24);
  float16x8_t d36 = vaddq_f16(src_data_15, src_data_25);
  float16x8_t d37 = vaddq_f16(src_data_16, src_data_26);
  float16x8_t d38 = vaddq_f16(src_data_17, src_data_27);

  float16x8_t d41 = vaddq_f16(src_data_30, src_data_40);
  float16x8_t d42 = vaddq_f16(src_data_31, src_data_41);
  float16x8_t d43 = vaddq_f16(src_data_32, src_data_42);
  float16x8_t d44 = vaddq_f16(src_data_33, src_data_43);
  float16x8_t d45 = vaddq_f16(src_data_34, src_data_44);
  float16x8_t d46 = vaddq_f16(src_data_35, src_data_45);
  float16x8_t d47 = vaddq_f16(src_data_36, src_data_46);
  float16x8_t d48 = vaddq_f16(src_data_37, src_data_47);

  float16x8_t d51 = vaddq_f16(src_data_50, src_data_60);
  float16x8_t d52 = vaddq_f16(src_data_51, src_data_61);
  float16x8_t d53 = vaddq_f16(src_data_52, src_data_62);
  float16x8_t d54 = vaddq_f16(src_data_53, src_data_63);
  float16x8_t d55 = vaddq_f16(src_data_54, src_data_64);
  float16x8_t d56 = vaddq_f16(src_data_55, src_data_65);
  float16x8_t d57 = vaddq_f16(src_data_56, src_data_66);
  float16x8_t d58 = vaddq_f16(src_data_57, src_data_67);

  float16x8_t t00 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_00, src_data_10), src_data_20), src_data_30), src_data_40),
      src_data_50),
    src_data_60);
  float16x8_t t01 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_01, src_data_11), src_data_21), src_data_31), src_data_41),
      src_data_51),
    src_data_61);
  float16x8_t t02 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_02, src_data_12), src_data_22), src_data_32), src_data_42),
      src_data_52),
    src_data_62);
  float16x8_t t03 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_03, src_data_13), src_data_23), src_data_33), src_data_43),
      src_data_53),
    src_data_63);
  float16x8_t t04 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_04, src_data_14), src_data_24), src_data_34), src_data_44),
      src_data_54),
    src_data_64);
  float16x8_t t05 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_05, src_data_15), src_data_25), src_data_35), src_data_45),
      src_data_55),
    src_data_65);
  float16x8_t t06 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_06, src_data_16), src_data_26), src_data_36), src_data_46),
      src_data_56),
    src_data_66);
  float16x8_t t07 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_07, src_data_17), src_data_27), src_data_37), src_data_47),
      src_data_57),
    src_data_67);

  float16x8_t t10 = vaddq_f16(vaddq_f16(vmulq_n_f16(d01, 0.5), d11), vmulq_n_f16(d21, 1.5));
  float16x8_t t11 = vaddq_f16(vaddq_f16(vmulq_n_f16(d02, 0.5), d12), vmulq_n_f16(d22, 1.5));
  float16x8_t t12 = vaddq_f16(vaddq_f16(vmulq_n_f16(d03, 0.5), d13), vmulq_n_f16(d23, 1.5));
  float16x8_t t13 = vaddq_f16(vaddq_f16(vmulq_n_f16(d04, 0.5), d14), vmulq_n_f16(d24, 1.5));
  float16x8_t t14 = vaddq_f16(vaddq_f16(vmulq_n_f16(d05, 0.5), d15), vmulq_n_f16(d25, 1.5));
  float16x8_t t15 = vaddq_f16(vaddq_f16(vmulq_n_f16(d06, 0.5), d16), vmulq_n_f16(d26, 1.5));
  float16x8_t t16 = vaddq_f16(vaddq_f16(vmulq_n_f16(d07, 0.5), d17), vmulq_n_f16(d27, 1.5));
  float16x8_t t17 = vaddq_f16(vaddq_f16(vmulq_n_f16(d08, 0.5), d18), vmulq_n_f16(d28, 1.5));

  float16x8_t t20 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d31, 0.25), d41), vmulq_n_f16(d51, 2.25)), src_data_70);
  float16x8_t t21 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d32, 0.25), d42), vmulq_n_f16(d52, 2.25)), src_data_71);
  float16x8_t t22 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d33, 0.25), d43), vmulq_n_f16(d53, 2.25)), src_data_72);
  float16x8_t t23 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d34, 0.25), d44), vmulq_n_f16(d54, 2.25)), src_data_73);
  float16x8_t t24 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d35, 0.25), d45), vmulq_n_f16(d55, 2.25)), src_data_74);
  float16x8_t t25 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d36, 0.25), d46), vmulq_n_f16(d56, 2.25)), src_data_75);
  float16x8_t t26 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d37, 0.25), d47), vmulq_n_f16(d57, 2.25)), src_data_76);
  float16x8_t t27 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d38, 0.25), d48), vmulq_n_f16(d58, 2.25)), src_data_77);

  float16x8_t s11 = vsubq_f16(t01, t02);
  float16x8_t s12 = vsubq_f16(t11, t12);
  float16x8_t s13 = vsubq_f16(t21, t22);

  float16x8_t s21 = vsubq_f16(t03, t04);
  float16x8_t s22 = vsubq_f16(t13, t14);
  float16x8_t s23 = vsubq_f16(t23, t24);

  float16x8_t s31 = vsubq_f16(t05, t06);
  float16x8_t s32 = vsubq_f16(t15, t16);
  float16x8_t s33 = vsubq_f16(t25, t26);

  float16x8_t s41 = vaddq_f16(t01, t02);
  float16x8_t s42 = vaddq_f16(t11, t12);
  float16x8_t s43 = vaddq_f16(t21, t22);

  float16x8_t s51 = vaddq_f16(t03, t04);
  float16x8_t s52 = vaddq_f16(t13, t14);
  float16x8_t s53 = vaddq_f16(t23, t24);

  float16x8_t s61 = vaddq_f16(t05, t06);
  float16x8_t s62 = vaddq_f16(t15, t16);
  float16x8_t s63 = vaddq_f16(t25, t26);

  float16x8_t m00 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t00, t01), t02), t03), t04), t05), t06);
  float16x8_t m01 = vaddq_f16(vaddq_f16(vmulq_n_f16(s11, 0.5), s21), vmulq_n_f16(s31, 1.5));
  float16x8_t m02 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s41, 0.25), s51), vmulq_n_f16(s61, 2.25)), t07);

  float16x8_t m10 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t10, t11), t12), t13), t14), t15), t16);
  float16x8_t m11 = vaddq_f16(vaddq_f16(vmulq_n_f16(s12, 0.5), s22), vmulq_n_f16(s32, 1.5));
  float16x8_t m12 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s42, 0.25), s52), vmulq_n_f16(s62, 2.25)), t17);

  float16x8_t m20 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t20, t21), t22), t23), t24), t25), t26);
  float16x8_t m21 = vaddq_f16(vaddq_f16(vmulq_n_f16(s13, 0.5), s23), vmulq_n_f16(s33, 1.5));
  float16x8_t m22 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s43, 0.25), s53), vmulq_n_f16(s63, 2.25)), t27);

  float16x8_t bias_ptr = vld1q_f16(bias_data);
  vst1q_f16(dst_data, vaddq_f16(m00, bias_ptr));
  vst1q_f16(dst_data + C8NUM, vaddq_f16(m01, bias_ptr));
  vst1q_f16(dst_data + 2 * C8NUM, vaddq_f16(m02, bias_ptr));

  vst1q_f16(dst_data + dst_step * C8NUM, vaddq_f16(m10, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM + C8NUM, vaddq_f16(m11, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM + 2 * C8NUM, vaddq_f16(m12, bias_ptr));

  vst1q_f16(dst_data + 2 * dst_step * C8NUM, vaddq_f16(m20, bias_ptr));
  vst1q_f16(dst_data + 2 * dst_step * C8NUM + C8NUM, vaddq_f16(m21, bias_ptr));
  vst1q_f16(dst_data + 2 * dst_step * C8NUM + 2 * C8NUM, vaddq_f16(m22, bias_ptr));
#else
  for (int i = 0; i < C8NUM; i++) {
    float16_t src_data_00 = src_data[i];
    float16_t src_data_01 = src_data[i + src_step];
    float16_t src_data_02 = src_data[i + 2 * src_step];
    float16_t src_data_03 = src_data[i + 3 * src_step];
    float16_t src_data_04 = src_data[i + 4 * src_step];
    float16_t src_data_05 = src_data[i + 5 * src_step];
    float16_t src_data_06 = src_data[i + 6 * src_step];
    float16_t src_data_07 = src_data[i + 7 * src_step];
    float16_t src_data_10 = src_data[i + 8 * src_step];
    float16_t src_data_11 = src_data[i + 9 * src_step];
    float16_t src_data_12 = src_data[i + 10 * src_step];
    float16_t src_data_13 = src_data[i + 11 * src_step];
    float16_t src_data_14 = src_data[i + 12 * src_step];
    float16_t src_data_15 = src_data[i + 13 * src_step];
    float16_t src_data_16 = src_data[i + 14 * src_step];
    float16_t src_data_17 = src_data[i + 15 * src_step];
    float16_t src_data_20 = src_data[i + 16 * src_step];
    float16_t src_data_21 = src_data[i + 17 * src_step];
    float16_t src_data_22 = src_data[i + 18 * src_step];
    float16_t src_data_23 = src_data[i + 19 * src_step];
    float16_t src_data_24 = src_data[i + 20 * src_step];
    float16_t src_data_25 = src_data[i + 21 * src_step];
    float16_t src_data_26 = src_data[i + 22 * src_step];
    float16_t src_data_27 = src_data[i + 23 * src_step];
    float16_t src_data_30 = src_data[i + 24 * src_step];
    float16_t src_data_31 = src_data[i + 25 * src_step];
    float16_t src_data_32 = src_data[i + 26 * src_step];
    float16_t src_data_33 = src_data[i + 27 * src_step];
    float16_t src_data_34 = src_data[i + 28 * src_step];
    float16_t src_data_35 = src_data[i + 29 * src_step];
    float16_t src_data_36 = src_data[i + 30 * src_step];
    float16_t src_data_37 = src_data[i + 31 * src_step];
    float16_t src_data_40 = src_data[i + 32 * src_step];
    float16_t src_data_41 = src_data[i + 33 * src_step];
    float16_t src_data_42 = src_data[i + 34 * src_step];
    float16_t src_data_43 = src_data[i + 35 * src_step];
    float16_t src_data_44 = src_data[i + 36 * src_step];
    float16_t src_data_45 = src_data[i + 37 * src_step];
    float16_t src_data_46 = src_data[i + 38 * src_step];
    float16_t src_data_47 = src_data[i + 39 * src_step];
    float16_t src_data_50 = src_data[i + 40 * src_step];
    float16_t src_data_51 = src_data[i + 41 * src_step];
    float16_t src_data_52 = src_data[i + 42 * src_step];
    float16_t src_data_53 = src_data[i + 43 * src_step];
    float16_t src_data_54 = src_data[i + 44 * src_step];
    float16_t src_data_55 = src_data[i + 45 * src_step];
    float16_t src_data_56 = src_data[i + 46 * src_step];
    float16_t src_data_57 = src_data[i + 47 * src_step];
    float16_t src_data_60 = src_data[i + 48 * src_step];
    float16_t src_data_61 = src_data[i + 49 * src_step];
    float16_t src_data_62 = src_data[i + 50 * src_step];
    float16_t src_data_63 = src_data[i + 51 * src_step];
    float16_t src_data_64 = src_data[i + 52 * src_step];
    float16_t src_data_65 = src_data[i + 53 * src_step];
    float16_t src_data_66 = src_data[i + 54 * src_step];
    float16_t src_data_67 = src_data[i + 55 * src_step];
    float16_t src_data_70 = src_data[i + 56 * src_step];
    float16_t src_data_71 = src_data[i + 57 * src_step];
    float16_t src_data_72 = src_data[i + 58 * src_step];
    float16_t src_data_73 = src_data[i + 59 * src_step];
    float16_t src_data_74 = src_data[i + 60 * src_step];
    float16_t src_data_75 = src_data[i + 61 * src_step];
    float16_t src_data_76 = src_data[i + 62 * src_step];
    float16_t src_data_77 = src_data[i + 63 * src_step];

    float16_t d01 = src_data_10 - src_data_20;
    float16_t d02 = src_data_11 - src_data_21;
    float16_t d03 = src_data_12 - src_data_22;
    float16_t d04 = src_data_13 - src_data_23;
    float16_t d05 = src_data_14 - src_data_24;
    float16_t d06 = src_data_15 - src_data_25;
    float16_t d07 = src_data_16 - src_data_26;
    float16_t d08 = src_data_17 - src_data_27;

    float16_t d11 = src_data_30 - src_data_40;
    float16_t d12 = src_data_31 - src_data_41;
    float16_t d13 = src_data_32 - src_data_42;
    float16_t d14 = src_data_33 - src_data_43;
    float16_t d15 = src_data_34 - src_data_44;
    float16_t d16 = src_data_35 - src_data_45;
    float16_t d17 = src_data_36 - src_data_46;
    float16_t d18 = src_data_37 - src_data_47;

    float16_t d21 = src_data_50 - src_data_60;
    float16_t d22 = src_data_51 - src_data_61;
    float16_t d23 = src_data_52 - src_data_62;
    float16_t d24 = src_data_53 - src_data_63;
    float16_t d25 = src_data_54 - src_data_64;
    float16_t d26 = src_data_55 - src_data_65;
    float16_t d27 = src_data_56 - src_data_66;
    float16_t d28 = src_data_57 - src_data_67;

    float16_t d31 = src_data_10 + src_data_20;
    float16_t d32 = src_data_11 + src_data_21;
    float16_t d33 = src_data_12 + src_data_22;
    float16_t d34 = src_data_13 + src_data_23;
    float16_t d35 = src_data_14 + src_data_24;
    float16_t d36 = src_data_15 + src_data_25;
    float16_t d37 = src_data_16 + src_data_26;
    float16_t d38 = src_data_17 + src_data_27;

    float16_t d41 = src_data_30 + src_data_40;
    float16_t d42 = src_data_31 + src_data_41;
    float16_t d43 = src_data_32 + src_data_42;
    float16_t d44 = src_data_33 + src_data_43;
    float16_t d45 = src_data_34 + src_data_44;
    float16_t d46 = src_data_35 + src_data_45;
    float16_t d47 = src_data_36 + src_data_46;
    float16_t d48 = src_data_37 + src_data_47;

    float16_t d51 = src_data_50 + src_data_60;
    float16_t d52 = src_data_51 + src_data_61;
    float16_t d53 = src_data_52 + src_data_62;
    float16_t d54 = src_data_53 + src_data_63;
    float16_t d55 = src_data_54 + src_data_64;
    float16_t d56 = src_data_55 + src_data_65;
    float16_t d57 = src_data_56 + src_data_66;
    float16_t d58 = src_data_57 + src_data_67;

    float16_t t00 = src_data_00 + src_data_10 + src_data_20 + src_data_30 + src_data_40 + src_data_50 + src_data_60;
    float16_t t01 = src_data_01 + src_data_11 + src_data_21 + src_data_31 + src_data_41 + src_data_51 + src_data_61;
    float16_t t02 = src_data_02 + src_data_12 + src_data_22 + src_data_32 + src_data_42 + src_data_52 + src_data_62;
    float16_t t03 = src_data_03 + src_data_13 + src_data_23 + src_data_33 + src_data_43 + src_data_53 + src_data_63;
    float16_t t04 = src_data_04 + src_data_14 + src_data_24 + src_data_34 + src_data_44 + src_data_54 + src_data_64;
    float16_t t05 = src_data_05 + src_data_15 + src_data_25 + src_data_35 + src_data_45 + src_data_55 + src_data_65;
    float16_t t06 = src_data_06 + src_data_16 + src_data_26 + src_data_36 + src_data_46 + src_data_56 + src_data_66;
    float16_t t07 = src_data_07 + src_data_17 + src_data_27 + src_data_37 + src_data_47 + src_data_57 + src_data_67;

    const float16_t t10 = 0.5f * d01 + d11 + 1.5f * d21;
    const float16_t t11 = 0.5f * d02 + d12 + 1.5f * d22;
    const float16_t t12 = 0.5f * d03 + d13 + 1.5f * d23;
    const float16_t t13 = 0.5f * d04 + d14 + 1.5f * d24;
    const float16_t t14 = 0.5f * d05 + d15 + 1.5f * d25;
    const float16_t t15 = 0.5f * d06 + d16 + 1.5f * d26;
    const float16_t t16 = 0.5f * d07 + d17 + 1.5f * d27;
    const float16_t t17 = 0.5f * d08 + d18 + 1.5f * d28;

    const float16_t t20 = 0.25f * d31 + d41 + 2.25f * d51 + src_data_70;
    const float16_t t21 = 0.25f * d32 + d42 + 2.25f * d52 + src_data_71;
    const float16_t t22 = 0.25f * d33 + d43 + 2.25f * d53 + src_data_72;
    const float16_t t23 = 0.25f * d34 + d44 + 2.25f * d54 + src_data_73;
    const float16_t t24 = 0.25f * d35 + d45 + 2.25f * d55 + src_data_74;
    const float16_t t25 = 0.25f * d36 + d46 + 2.25f * d56 + src_data_75;
    const float16_t t26 = 0.25f * d37 + d47 + 2.25f * d57 + src_data_76;
    const float16_t t27 = 0.25f * d38 + d48 + 2.25f * d58 + src_data_77;

    float16_t s11 = t01 - t02;
    float16_t s12 = t11 - t12;
    float16_t s13 = t21 - t22;

    float16_t s21 = t03 - t04;
    float16_t s22 = t13 - t14;
    float16_t s23 = t23 - t24;

    float16_t s31 = t05 - t06;
    float16_t s32 = t15 - t16;
    float16_t s33 = t25 - t26;

    float16_t s41 = t01 + t02;
    float16_t s42 = t11 + t12;
    float16_t s43 = t21 + t22;

    float16_t s51 = t03 + t04;
    float16_t s52 = t13 + t14;
    float16_t s53 = t23 + t24;

    float16_t s61 = t05 + t06;
    float16_t s62 = t15 + t16;
    float16_t s63 = t25 + t26;

    float16_t m00 = t00 + t01 + t02 + t03 + t04 + t05 + t06;
    const float16_t m01 = 0.5f * s11 + s21 + 1.5f * s31;
    const float16_t m02 = 0.25f * s41 + s51 + 2.25f * s61 + t07;

    float16_t m10 = t10 + t11 + t12 + t13 + t14 + t15 + t16;
    const float16_t m11 = 0.5f * s12 + s22 + 1.5f * s32;
    const float16_t m12 = 0.25f * s42 + s52 + 2.25f * s62 + t17;

    float16_t m20 = t20 + t21 + t22 + t23 + t24 + t25 + t26;
    const float16_t m21 = 0.5f * s13 + s23 + 1.5f * s33;
    const float16_t m22 = 0.25f * s43 + s53 + 2.25f * s63 + t27;

    (dst_data + i)[0] = m00 + bias_data[i];
    (dst_data + i + C8NUM)[0] = m01 + bias_data[i];
    (dst_data + i + 2 * C8NUM)[0] = m02 + bias_data[i];

    (dst_data + i + dst_step * C8NUM)[0] = m10 + bias_data[i];
    (dst_data + i + dst_step * C8NUM + C8NUM)[0] = m11 + bias_data[i];
    (dst_data + i + dst_step * C8NUM + 2 * C8NUM)[0] = m12 + bias_data[i];

    (dst_data + i + 2 * dst_step * C8NUM)[0] = m20 + bias_data[i];
    (dst_data + i + 2 * dst_step * C8NUM + C8NUM)[0] = m21 + bias_data[i];
    (dst_data + i + 2 * dst_step * C8NUM + 2 * C8NUM)[0] = m22 + bias_data[i];
  }
#endif
}

void OutputTransform8x4UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step) {
#ifdef ENABLE_ARM
  float16x8_t src_data_00 = vld1q_f16(src_data + 0 * src_step);
  float16x8_t src_data_01 = vld1q_f16(src_data + 1 * src_step);
  float16x8_t src_data_02 = vld1q_f16(src_data + 2 * src_step);
  float16x8_t src_data_03 = vld1q_f16(src_data + 3 * src_step);
  float16x8_t src_data_04 = vld1q_f16(src_data + 4 * src_step);
  float16x8_t src_data_05 = vld1q_f16(src_data + 5 * src_step);
  float16x8_t src_data_06 = vld1q_f16(src_data + 6 * src_step);
  float16x8_t src_data_07 = vld1q_f16(src_data + 7 * src_step);
  float16x8_t src_data_10 = vld1q_f16(src_data + 8 * src_step);
  float16x8_t src_data_11 = vld1q_f16(src_data + 9 * src_step);
  float16x8_t src_data_12 = vld1q_f16(src_data + 10 * src_step);
  float16x8_t src_data_13 = vld1q_f16(src_data + 11 * src_step);
  float16x8_t src_data_14 = vld1q_f16(src_data + 12 * src_step);
  float16x8_t src_data_15 = vld1q_f16(src_data + 13 * src_step);
  float16x8_t src_data_16 = vld1q_f16(src_data + 14 * src_step);
  float16x8_t src_data_17 = vld1q_f16(src_data + 15 * src_step);
  float16x8_t src_data_20 = vld1q_f16(src_data + 16 * src_step);
  float16x8_t src_data_21 = vld1q_f16(src_data + 17 * src_step);
  float16x8_t src_data_22 = vld1q_f16(src_data + 18 * src_step);
  float16x8_t src_data_23 = vld1q_f16(src_data + 19 * src_step);
  float16x8_t src_data_24 = vld1q_f16(src_data + 20 * src_step);
  float16x8_t src_data_25 = vld1q_f16(src_data + 21 * src_step);
  float16x8_t src_data_26 = vld1q_f16(src_data + 22 * src_step);
  float16x8_t src_data_27 = vld1q_f16(src_data + 23 * src_step);
  float16x8_t src_data_30 = vld1q_f16(src_data + 24 * src_step);
  float16x8_t src_data_31 = vld1q_f16(src_data + 25 * src_step);
  float16x8_t src_data_32 = vld1q_f16(src_data + 26 * src_step);
  float16x8_t src_data_33 = vld1q_f16(src_data + 27 * src_step);
  float16x8_t src_data_34 = vld1q_f16(src_data + 28 * src_step);
  float16x8_t src_data_35 = vld1q_f16(src_data + 29 * src_step);
  float16x8_t src_data_36 = vld1q_f16(src_data + 30 * src_step);
  float16x8_t src_data_37 = vld1q_f16(src_data + 31 * src_step);
  float16x8_t src_data_40 = vld1q_f16(src_data + 32 * src_step);
  float16x8_t src_data_41 = vld1q_f16(src_data + 33 * src_step);
  float16x8_t src_data_42 = vld1q_f16(src_data + 34 * src_step);
  float16x8_t src_data_43 = vld1q_f16(src_data + 35 * src_step);
  float16x8_t src_data_44 = vld1q_f16(src_data + 36 * src_step);
  float16x8_t src_data_45 = vld1q_f16(src_data + 37 * src_step);
  float16x8_t src_data_46 = vld1q_f16(src_data + 38 * src_step);
  float16x8_t src_data_47 = vld1q_f16(src_data + 39 * src_step);
  float16x8_t src_data_50 = vld1q_f16(src_data + 40 * src_step);
  float16x8_t src_data_51 = vld1q_f16(src_data + 41 * src_step);
  float16x8_t src_data_52 = vld1q_f16(src_data + 42 * src_step);
  float16x8_t src_data_53 = vld1q_f16(src_data + 43 * src_step);
  float16x8_t src_data_54 = vld1q_f16(src_data + 44 * src_step);
  float16x8_t src_data_55 = vld1q_f16(src_data + 45 * src_step);
  float16x8_t src_data_56 = vld1q_f16(src_data + 46 * src_step);
  float16x8_t src_data_57 = vld1q_f16(src_data + 47 * src_step);
  float16x8_t src_data_60 = vld1q_f16(src_data + 48 * src_step);
  float16x8_t src_data_61 = vld1q_f16(src_data + 49 * src_step);
  float16x8_t src_data_62 = vld1q_f16(src_data + 50 * src_step);
  float16x8_t src_data_63 = vld1q_f16(src_data + 51 * src_step);
  float16x8_t src_data_64 = vld1q_f16(src_data + 52 * src_step);
  float16x8_t src_data_65 = vld1q_f16(src_data + 53 * src_step);
  float16x8_t src_data_66 = vld1q_f16(src_data + 54 * src_step);
  float16x8_t src_data_67 = vld1q_f16(src_data + 55 * src_step);
  float16x8_t src_data_70 = vld1q_f16(src_data + 56 * src_step);
  float16x8_t src_data_71 = vld1q_f16(src_data + 57 * src_step);
  float16x8_t src_data_72 = vld1q_f16(src_data + 58 * src_step);
  float16x8_t src_data_73 = vld1q_f16(src_data + 59 * src_step);
  float16x8_t src_data_74 = vld1q_f16(src_data + 60 * src_step);
  float16x8_t src_data_75 = vld1q_f16(src_data + 61 * src_step);
  float16x8_t src_data_76 = vld1q_f16(src_data + 62 * src_step);
  float16x8_t src_data_77 = vld1q_f16(src_data + 63 * src_step);

  float16x8_t d01 = vsubq_f16(src_data_10, src_data_20);
  float16x8_t d02 = vsubq_f16(src_data_11, src_data_21);
  float16x8_t d03 = vsubq_f16(src_data_12, src_data_22);
  float16x8_t d04 = vsubq_f16(src_data_13, src_data_23);
  float16x8_t d05 = vsubq_f16(src_data_14, src_data_24);
  float16x8_t d06 = vsubq_f16(src_data_15, src_data_25);
  float16x8_t d07 = vsubq_f16(src_data_16, src_data_26);
  float16x8_t d08 = vsubq_f16(src_data_17, src_data_27);

  float16x8_t d11 = vsubq_f16(src_data_30, src_data_40);
  float16x8_t d12 = vsubq_f16(src_data_31, src_data_41);
  float16x8_t d13 = vsubq_f16(src_data_32, src_data_42);
  float16x8_t d14 = vsubq_f16(src_data_33, src_data_43);
  float16x8_t d15 = vsubq_f16(src_data_34, src_data_44);
  float16x8_t d16 = vsubq_f16(src_data_35, src_data_45);
  float16x8_t d17 = vsubq_f16(src_data_36, src_data_46);
  float16x8_t d18 = vsubq_f16(src_data_37, src_data_47);

  float16x8_t d21 = vsubq_f16(src_data_50, src_data_60);
  float16x8_t d22 = vsubq_f16(src_data_51, src_data_61);
  float16x8_t d23 = vsubq_f16(src_data_52, src_data_62);
  float16x8_t d24 = vsubq_f16(src_data_53, src_data_63);
  float16x8_t d25 = vsubq_f16(src_data_54, src_data_64);
  float16x8_t d26 = vsubq_f16(src_data_55, src_data_65);
  float16x8_t d27 = vsubq_f16(src_data_56, src_data_66);
  float16x8_t d28 = vsubq_f16(src_data_57, src_data_67);

  float16x8_t d31 = vaddq_f16(src_data_10, src_data_20);
  float16x8_t d32 = vaddq_f16(src_data_11, src_data_21);
  float16x8_t d33 = vaddq_f16(src_data_12, src_data_22);
  float16x8_t d34 = vaddq_f16(src_data_13, src_data_23);
  float16x8_t d35 = vaddq_f16(src_data_14, src_data_24);
  float16x8_t d36 = vaddq_f16(src_data_15, src_data_25);
  float16x8_t d37 = vaddq_f16(src_data_16, src_data_26);
  float16x8_t d38 = vaddq_f16(src_data_17, src_data_27);

  float16x8_t d41 = vaddq_f16(src_data_30, src_data_40);
  float16x8_t d42 = vaddq_f16(src_data_31, src_data_41);
  float16x8_t d43 = vaddq_f16(src_data_32, src_data_42);
  float16x8_t d44 = vaddq_f16(src_data_33, src_data_43);
  float16x8_t d45 = vaddq_f16(src_data_34, src_data_44);
  float16x8_t d46 = vaddq_f16(src_data_35, src_data_45);
  float16x8_t d47 = vaddq_f16(src_data_36, src_data_46);
  float16x8_t d48 = vaddq_f16(src_data_37, src_data_47);

  float16x8_t d51 = vaddq_f16(src_data_50, src_data_60);
  float16x8_t d52 = vaddq_f16(src_data_51, src_data_61);
  float16x8_t d53 = vaddq_f16(src_data_52, src_data_62);
  float16x8_t d54 = vaddq_f16(src_data_53, src_data_63);
  float16x8_t d55 = vaddq_f16(src_data_54, src_data_64);
  float16x8_t d56 = vaddq_f16(src_data_55, src_data_65);
  float16x8_t d57 = vaddq_f16(src_data_56, src_data_66);
  float16x8_t d58 = vaddq_f16(src_data_57, src_data_67);

  float16x8_t t00 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_00, src_data_10), src_data_20), src_data_30), src_data_40),
      src_data_50),
    src_data_60);
  float16x8_t t01 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_01, src_data_11), src_data_21), src_data_31), src_data_41),
      src_data_51),
    src_data_61);
  float16x8_t t02 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_02, src_data_12), src_data_22), src_data_32), src_data_42),
      src_data_52),
    src_data_62);
  float16x8_t t03 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_03, src_data_13), src_data_23), src_data_33), src_data_43),
      src_data_53),
    src_data_63);
  float16x8_t t04 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_04, src_data_14), src_data_24), src_data_34), src_data_44),
      src_data_54),
    src_data_64);
  float16x8_t t05 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_05, src_data_15), src_data_25), src_data_35), src_data_45),
      src_data_55),
    src_data_65);
  float16x8_t t06 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_06, src_data_16), src_data_26), src_data_36), src_data_46),
      src_data_56),
    src_data_66);
  float16x8_t t07 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_07, src_data_17), src_data_27), src_data_37), src_data_47),
      src_data_57),
    src_data_67);

  float16x8_t t10 = vaddq_f16(vaddq_f16(vmulq_n_f16(d01, 0.5), d11), vmulq_n_f16(d21, 1.5));
  float16x8_t t11 = vaddq_f16(vaddq_f16(vmulq_n_f16(d02, 0.5), d12), vmulq_n_f16(d22, 1.5));
  float16x8_t t12 = vaddq_f16(vaddq_f16(vmulq_n_f16(d03, 0.5), d13), vmulq_n_f16(d23, 1.5));
  float16x8_t t13 = vaddq_f16(vaddq_f16(vmulq_n_f16(d04, 0.5), d14), vmulq_n_f16(d24, 1.5));
  float16x8_t t14 = vaddq_f16(vaddq_f16(vmulq_n_f16(d05, 0.5), d15), vmulq_n_f16(d25, 1.5));
  float16x8_t t15 = vaddq_f16(vaddq_f16(vmulq_n_f16(d06, 0.5), d16), vmulq_n_f16(d26, 1.5));
  float16x8_t t16 = vaddq_f16(vaddq_f16(vmulq_n_f16(d07, 0.5), d17), vmulq_n_f16(d27, 1.5));
  float16x8_t t17 = vaddq_f16(vaddq_f16(vmulq_n_f16(d08, 0.5), d18), vmulq_n_f16(d28, 1.5));

  float16x8_t t20 = vaddq_f16(vaddq_f16(vmulq_n_f16(d31, 0.25), d41), vmulq_n_f16(d51, 2.25));
  float16x8_t t21 = vaddq_f16(vaddq_f16(vmulq_n_f16(d32, 0.25), d42), vmulq_n_f16(d52, 2.25));
  float16x8_t t22 = vaddq_f16(vaddq_f16(vmulq_n_f16(d33, 0.25), d43), vmulq_n_f16(d53, 2.25));
  float16x8_t t23 = vaddq_f16(vaddq_f16(vmulq_n_f16(d34, 0.25), d44), vmulq_n_f16(d54, 2.25));
  float16x8_t t24 = vaddq_f16(vaddq_f16(vmulq_n_f16(d35, 0.25), d45), vmulq_n_f16(d55, 2.25));
  float16x8_t t25 = vaddq_f16(vaddq_f16(vmulq_n_f16(d36, 0.25), d46), vmulq_n_f16(d56, 2.25));
  float16x8_t t26 = vaddq_f16(vaddq_f16(vmulq_n_f16(d37, 0.25), d47), vmulq_n_f16(d57, 2.25));
  float16x8_t t27 = vaddq_f16(vaddq_f16(vmulq_n_f16(d38, 0.25), d48), vmulq_n_f16(d58, 2.25));

  float16x8_t t30 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d01, 0.125), d11), vmulq_n_f16(d21, 3.375)), src_data_70);
  float16x8_t t31 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d02, 0.125), d12), vmulq_n_f16(d22, 3.375)), src_data_71);
  float16x8_t t32 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d03, 0.125), d13), vmulq_n_f16(d23, 3.375)), src_data_72);
  float16x8_t t33 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d04, 0.125), d14), vmulq_n_f16(d24, 3.375)), src_data_73);
  float16x8_t t34 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d05, 0.125), d15), vmulq_n_f16(d25, 3.375)), src_data_74);
  float16x8_t t35 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d06, 0.125), d16), vmulq_n_f16(d26, 3.375)), src_data_75);
  float16x8_t t36 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d07, 0.125), d17), vmulq_n_f16(d27, 3.375)), src_data_76);
  float16x8_t t37 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d08, 0.125), d18), vmulq_n_f16(d28, 3.375)), src_data_77);

  float16x8_t s11 = vsubq_f16(t01, t02);
  float16x8_t s12 = vsubq_f16(t11, t12);
  float16x8_t s13 = vsubq_f16(t21, t22);
  float16x8_t s14 = vsubq_f16(t31, t32);

  float16x8_t s21 = vsubq_f16(t03, t04);
  float16x8_t s22 = vsubq_f16(t13, t14);
  float16x8_t s23 = vsubq_f16(t23, t24);
  float16x8_t s24 = vsubq_f16(t33, t34);

  float16x8_t s31 = vsubq_f16(t05, t06);
  float16x8_t s32 = vsubq_f16(t15, t16);
  float16x8_t s33 = vsubq_f16(t25, t26);
  float16x8_t s34 = vsubq_f16(t35, t36);

  float16x8_t s41 = vaddq_f16(t01, t02);
  float16x8_t s42 = vaddq_f16(t11, t12);
  float16x8_t s43 = vaddq_f16(t21, t22);
  float16x8_t s44 = vaddq_f16(t31, t32);

  float16x8_t s51 = vaddq_f16(t03, t04);
  float16x8_t s52 = vaddq_f16(t13, t14);
  float16x8_t s53 = vaddq_f16(t23, t24);
  float16x8_t s54 = vaddq_f16(t33, t34);

  float16x8_t s61 = vaddq_f16(t05, t06);
  float16x8_t s62 = vaddq_f16(t15, t16);
  float16x8_t s63 = vaddq_f16(t25, t26);
  float16x8_t s64 = vaddq_f16(t35, t36);

  float16x8_t m00 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t00, t01), t02), t03), t04), t05), t06);
  float16x8_t m01 = vaddq_f16(vaddq_f16(vmulq_n_f16(s11, 0.5), s21), vmulq_n_f16(s31, 1.5));
  float16x8_t m02 = vaddq_f16(vaddq_f16(vmulq_n_f16(s41, 0.25), s51), vmulq_n_f16(s61, 2.25));
  float16x8_t m03 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s11, 0.125), s21), vmulq_n_f16(s31, 3.375)), t07);

  float16x8_t m10 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t10, t11), t12), t13), t14), t15), t16);
  float16x8_t m11 = vaddq_f16(vaddq_f16(vmulq_n_f16(s12, 0.5), s22), vmulq_n_f16(s32, 1.5));
  float16x8_t m12 = vaddq_f16(vaddq_f16(vmulq_n_f16(s42, 0.25), s52), vmulq_n_f16(s62, 2.25));
  float16x8_t m13 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s12, 0.125), s22), vmulq_n_f16(s32, 3.375)), t17);

  float16x8_t m20 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t20, t21), t22), t23), t24), t25), t26);
  float16x8_t m21 = vaddq_f16(vaddq_f16(vmulq_n_f16(s13, 0.5), s23), vmulq_n_f16(s33, 1.5));
  float16x8_t m22 = vaddq_f16(vaddq_f16(vmulq_n_f16(s43, 0.25), s53), vmulq_n_f16(s63, 2.25));
  float16x8_t m23 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s13, 0.125), s23), vmulq_n_f16(s33, 3.375)), t27);

  float16x8_t m30 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t30, t31), t32), t33), t34), t35), t36);
  float16x8_t m31 = vaddq_f16(vaddq_f16(vmulq_n_f16(s14, 0.5), s24), vmulq_n_f16(s34, 1.5));
  float16x8_t m32 = vaddq_f16(vaddq_f16(vmulq_n_f16(s44, 0.25), s54), vmulq_n_f16(s64, 2.25));
  float16x8_t m33 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s14, 0.125), s24), vmulq_n_f16(s34, 3.375)), t37);

  float16x8_t bias_ptr = vld1q_f16(bias_data);
  vst1q_f16(dst_data, vaddq_f16(m00, bias_ptr));
  vst1q_f16(dst_data + C8NUM, vaddq_f16(m01, bias_ptr));
  vst1q_f16(dst_data + 2 * C8NUM, vaddq_f16(m02, bias_ptr));
  vst1q_f16(dst_data + 3 * C8NUM, vaddq_f16(m03, bias_ptr));

  vst1q_f16(dst_data + dst_step * C8NUM, vaddq_f16(m10, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM + C8NUM, vaddq_f16(m11, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM + 2 * C8NUM, vaddq_f16(m12, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM + 3 * C8NUM, vaddq_f16(m13, bias_ptr));

  vst1q_f16(dst_data + 2 * dst_step * C8NUM, vaddq_f16(m20, bias_ptr));
  vst1q_f16(dst_data + 2 * dst_step * C8NUM + C8NUM, vaddq_f16(m21, bias_ptr));
  vst1q_f16(dst_data + 2 * dst_step * C8NUM + 2 * C8NUM, vaddq_f16(m22, bias_ptr));
  vst1q_f16(dst_data + 2 * dst_step * C8NUM + 3 * C8NUM, vaddq_f16(m23, bias_ptr));

  vst1q_f16(dst_data + 3 * dst_step * C8NUM, vaddq_f16(m30, bias_ptr));
  vst1q_f16(dst_data + 3 * dst_step * C8NUM + C8NUM, vaddq_f16(m31, bias_ptr));
  vst1q_f16(dst_data + 3 * dst_step * C8NUM + 2 * C8NUM, vaddq_f16(m32, bias_ptr));
  vst1q_f16(dst_data + 3 * dst_step * C8NUM + 3 * C8NUM, vaddq_f16(m33, bias_ptr));
#else
  for (int i = 0; i < C8NUM; i++) {
    float16_t src_data_00 = src_data[i];
    float16_t src_data_01 = src_data[i + src_step];
    float16_t src_data_02 = src_data[i + 2 * src_step];
    float16_t src_data_03 = src_data[i + 3 * src_step];
    float16_t src_data_04 = src_data[i + 4 * src_step];
    float16_t src_data_05 = src_data[i + 5 * src_step];
    float16_t src_data_06 = src_data[i + 6 * src_step];
    float16_t src_data_07 = src_data[i + 7 * src_step];
    float16_t src_data_10 = src_data[i + 8 * src_step];
    float16_t src_data_11 = src_data[i + 9 * src_step];
    float16_t src_data_12 = src_data[i + 10 * src_step];
    float16_t src_data_13 = src_data[i + 11 * src_step];
    float16_t src_data_14 = src_data[i + 12 * src_step];
    float16_t src_data_15 = src_data[i + 13 * src_step];
    float16_t src_data_16 = src_data[i + 14 * src_step];
    float16_t src_data_17 = src_data[i + 15 * src_step];
    float16_t src_data_20 = src_data[i + 16 * src_step];
    float16_t src_data_21 = src_data[i + 17 * src_step];
    float16_t src_data_22 = src_data[i + 18 * src_step];
    float16_t src_data_23 = src_data[i + 19 * src_step];
    float16_t src_data_24 = src_data[i + 20 * src_step];
    float16_t src_data_25 = src_data[i + 21 * src_step];
    float16_t src_data_26 = src_data[i + 22 * src_step];
    float16_t src_data_27 = src_data[i + 23 * src_step];
    float16_t src_data_30 = src_data[i + 24 * src_step];
    float16_t src_data_31 = src_data[i + 25 * src_step];
    float16_t src_data_32 = src_data[i + 26 * src_step];
    float16_t src_data_33 = src_data[i + 27 * src_step];
    float16_t src_data_34 = src_data[i + 28 * src_step];
    float16_t src_data_35 = src_data[i + 29 * src_step];
    float16_t src_data_36 = src_data[i + 30 * src_step];
    float16_t src_data_37 = src_data[i + 31 * src_step];
    float16_t src_data_40 = src_data[i + 32 * src_step];
    float16_t src_data_41 = src_data[i + 33 * src_step];
    float16_t src_data_42 = src_data[i + 34 * src_step];
    float16_t src_data_43 = src_data[i + 35 * src_step];
    float16_t src_data_44 = src_data[i + 36 * src_step];
    float16_t src_data_45 = src_data[i + 37 * src_step];
    float16_t src_data_46 = src_data[i + 38 * src_step];
    float16_t src_data_47 = src_data[i + 39 * src_step];
    float16_t src_data_50 = src_data[i + 40 * src_step];
    float16_t src_data_51 = src_data[i + 41 * src_step];
    float16_t src_data_52 = src_data[i + 42 * src_step];
    float16_t src_data_53 = src_data[i + 43 * src_step];
    float16_t src_data_54 = src_data[i + 44 * src_step];
    float16_t src_data_55 = src_data[i + 45 * src_step];
    float16_t src_data_56 = src_data[i + 46 * src_step];
    float16_t src_data_57 = src_data[i + 47 * src_step];
    float16_t src_data_60 = src_data[i + 48 * src_step];
    float16_t src_data_61 = src_data[i + 49 * src_step];
    float16_t src_data_62 = src_data[i + 50 * src_step];
    float16_t src_data_63 = src_data[i + 51 * src_step];
    float16_t src_data_64 = src_data[i + 52 * src_step];
    float16_t src_data_65 = src_data[i + 53 * src_step];
    float16_t src_data_66 = src_data[i + 54 * src_step];
    float16_t src_data_67 = src_data[i + 55 * src_step];
    float16_t src_data_70 = src_data[i + 56 * src_step];
    float16_t src_data_71 = src_data[i + 57 * src_step];
    float16_t src_data_72 = src_data[i + 58 * src_step];
    float16_t src_data_73 = src_data[i + 59 * src_step];
    float16_t src_data_74 = src_data[i + 60 * src_step];
    float16_t src_data_75 = src_data[i + 61 * src_step];
    float16_t src_data_76 = src_data[i + 62 * src_step];
    float16_t src_data_77 = src_data[i + 63 * src_step];

    float16_t d01 = src_data_10 - src_data_20;
    float16_t d02 = src_data_11 - src_data_21;
    float16_t d03 = src_data_12 - src_data_22;
    float16_t d04 = src_data_13 - src_data_23;
    float16_t d05 = src_data_14 - src_data_24;
    float16_t d06 = src_data_15 - src_data_25;
    float16_t d07 = src_data_16 - src_data_26;
    float16_t d08 = src_data_17 - src_data_27;

    float16_t d11 = src_data_30 - src_data_40;
    float16_t d12 = src_data_31 - src_data_41;
    float16_t d13 = src_data_32 - src_data_42;
    float16_t d14 = src_data_33 - src_data_43;
    float16_t d15 = src_data_34 - src_data_44;
    float16_t d16 = src_data_35 - src_data_45;
    float16_t d17 = src_data_36 - src_data_46;
    float16_t d18 = src_data_37 - src_data_47;

    float16_t d21 = src_data_50 - src_data_60;
    float16_t d22 = src_data_51 - src_data_61;
    float16_t d23 = src_data_52 - src_data_62;
    float16_t d24 = src_data_53 - src_data_63;
    float16_t d25 = src_data_54 - src_data_64;
    float16_t d26 = src_data_55 - src_data_65;
    float16_t d27 = src_data_56 - src_data_66;
    float16_t d28 = src_data_57 - src_data_67;

    float16_t d31 = src_data_10 + src_data_20;
    float16_t d32 = src_data_11 + src_data_21;
    float16_t d33 = src_data_12 + src_data_22;
    float16_t d34 = src_data_13 + src_data_23;
    float16_t d35 = src_data_14 + src_data_24;
    float16_t d36 = src_data_15 + src_data_25;
    float16_t d37 = src_data_16 + src_data_26;
    float16_t d38 = src_data_17 + src_data_27;

    float16_t d41 = src_data_30 + src_data_40;
    float16_t d42 = src_data_31 + src_data_41;
    float16_t d43 = src_data_32 + src_data_42;
    float16_t d44 = src_data_33 + src_data_43;
    float16_t d45 = src_data_34 + src_data_44;
    float16_t d46 = src_data_35 + src_data_45;
    float16_t d47 = src_data_36 + src_data_46;
    float16_t d48 = src_data_37 + src_data_47;

    float16_t d51 = src_data_50 + src_data_60;
    float16_t d52 = src_data_51 + src_data_61;
    float16_t d53 = src_data_52 + src_data_62;
    float16_t d54 = src_data_53 + src_data_63;
    float16_t d55 = src_data_54 + src_data_64;
    float16_t d56 = src_data_55 + src_data_65;
    float16_t d57 = src_data_56 + src_data_66;
    float16_t d58 = src_data_57 + src_data_67;

    float16_t t00 = src_data_00 + src_data_10 + src_data_20 + src_data_30 + src_data_40 + src_data_50 + src_data_60;
    float16_t t01 = src_data_01 + src_data_11 + src_data_21 + src_data_31 + src_data_41 + src_data_51 + src_data_61;
    float16_t t02 = src_data_02 + src_data_12 + src_data_22 + src_data_32 + src_data_42 + src_data_52 + src_data_62;
    float16_t t03 = src_data_03 + src_data_13 + src_data_23 + src_data_33 + src_data_43 + src_data_53 + src_data_63;
    float16_t t04 = src_data_04 + src_data_14 + src_data_24 + src_data_34 + src_data_44 + src_data_54 + src_data_64;
    float16_t t05 = src_data_05 + src_data_15 + src_data_25 + src_data_35 + src_data_45 + src_data_55 + src_data_65;
    float16_t t06 = src_data_06 + src_data_16 + src_data_26 + src_data_36 + src_data_46 + src_data_56 + src_data_66;
    float16_t t07 = src_data_07 + src_data_17 + src_data_27 + src_data_37 + src_data_47 + src_data_57 + src_data_67;

    const float16_t t10 = 0.5f * d01 + d11 + 1.5f * d21;
    const float16_t t11 = 0.5f * d02 + d12 + 1.5f * d22;
    const float16_t t12 = 0.5f * d03 + d13 + 1.5f * d23;
    const float16_t t13 = 0.5f * d04 + d14 + 1.5f * d24;
    const float16_t t14 = 0.5f * d05 + d15 + 1.5f * d25;
    const float16_t t15 = 0.5f * d06 + d16 + 1.5f * d26;
    const float16_t t16 = 0.5f * d07 + d17 + 1.5f * d27;
    const float16_t t17 = 0.5f * d08 + d18 + 1.5f * d28;

    const float16_t t20 = 0.25f * d31 + d41 + 2.25f * d51;
    const float16_t t21 = 0.25f * d32 + d42 + 2.25f * d52;
    const float16_t t22 = 0.25f * d33 + d43 + 2.25f * d53;
    const float16_t t23 = 0.25f * d34 + d44 + 2.25f * d54;
    const float16_t t24 = 0.25f * d35 + d45 + 2.25f * d55;
    const float16_t t25 = 0.25f * d36 + d46 + 2.25f * d56;
    const float16_t t26 = 0.25f * d37 + d47 + 2.25f * d57;
    const const float16_t t27 = 0.25f * d38 + d48 + 2.25f * d58;

    const float16_t t30 = 0.125f * d01 + d11 + 3.375f * d21 + src_data_70;
    const float16_t t31 = 0.125f * d02 + d12 + 3.375f * d22 + src_data_71;
    const float16_t t32 = 0.125f * d03 + d13 + 3.375f * d23 + src_data_72;
    const float16_t t33 = 0.125f * d04 + d14 + 3.375f * d24 + src_data_73;
    const float16_t t34 = 0.125f * d05 + d15 + 3.375f * d25 + src_data_74;
    const float16_t t35 = 0.125f * d06 + d16 + 3.375f * d26 + src_data_75;
    const float16_t t36 = 0.125f * d07 + d17 + 3.375f * d27 + src_data_76;
    const float16_t t37 = 0.125f * d08 + d18 + 3.375f * d28 + src_data_77;

    float16_t s11 = t01 - t02;
    float16_t s12 = t11 - t12;
    float16_t s13 = t21 - t22;
    float16_t s14 = t31 - t32;

    float16_t s21 = t03 - t04;
    float16_t s22 = t13 - t14;
    float16_t s23 = t23 - t24;
    float16_t s24 = t33 - t34;

    float16_t s31 = t05 - t06;
    float16_t s32 = t15 - t16;
    float16_t s33 = t25 - t26;
    float16_t s34 = t35 - t36;

    float16_t s41 = t01 + t02;
    float16_t s42 = t11 + t12;
    float16_t s43 = t21 + t22;
    float16_t s44 = t31 + t32;

    float16_t s51 = t03 + t04;
    float16_t s52 = t13 + t14;
    float16_t s53 = t23 + t24;
    float16_t s54 = t33 + t34;

    float16_t s61 = t05 + t06;
    float16_t s62 = t15 + t16;
    float16_t s63 = t25 + t26;
    float16_t s64 = t35 + t36;

    float16_t m00 = t00 + t01 + t02 + t03 + t04 + t05 + t06;
    const float16_t m01 = 0.5f * s11 + s21 + 1.5f * s31;
    const float16_t m02 = 0.25f * s41 + s51 + 2.25f * s61;
    const float16_t m03 = 0.125f * s11 + s21 + 3.375f * s31 + t07;

    float16_t m10 = t10 + t11 + t12 + t13 + t14 + t15 + t16;
    const float16_t m11 = 0.5f * s12 + s22 + 1.5f * s32;
    const float16_t m12 = 0.25f * s42 + s52 + 2.25f * s62;
    const float16_t m13 = 0.125f * s12 + s22 + 3.375f * s32 + t17;

    float16_t m20 = t20 + t21 + t22 + t23 + t24 + t25 + t26;
    const float16_t m21 = 0.5f * s13 + s23 + 1.5f * s33;
    const float16_t m22 = 0.25f * s43 + s53 + 2.25f * s63;
    const float16_t m23 = 0.125f * s13 + s23 + 3.375f * s33 + t27;

    float16_t m30 = t30 + t31 + t32 + t33 + t34 + t35 + t36;
    const float16_t m31 = 0.5f * s14 + s24 + 1.5f * s34;
    const float16_t m32 = 0.25f * s44 + s54 + 2.25f * s64;
    const float16_t m33 = 0.125f * s14 + s24 + 3.375f * s34 + t37;

    (dst_data + i)[0] = m00 + bias_data[i];
    (dst_data + i + C8NUM)[0] = m01 + bias_data[i];
    (dst_data + i + 2 * C8NUM)[0] = m02 + bias_data[i];
    (dst_data + i + 3 * C8NUM)[0] = m03 + bias_data[i];

    (dst_data + i + dst_step * C8NUM)[0] = m10 + bias_data[i];
    (dst_data + i + dst_step * C8NUM + C8NUM)[0] = m11 + bias_data[i];
    (dst_data + i + dst_step * C8NUM + 2 * C8NUM)[0] = m12 + bias_data[i];
    (dst_data + i + dst_step * C8NUM + 3 * C8NUM)[0] = m13 + bias_data[i];

    (dst_data + i + 2 * dst_step * C8NUM)[0] = m20 + bias_data[i];
    (dst_data + i + 2 * dst_step * C8NUM + C8NUM)[0] = m21 + bias_data[i];
    (dst_data + i + 2 * dst_step * C8NUM + 2 * C8NUM)[0] = m22 + bias_data[i];
    (dst_data + i + 2 * dst_step * C8NUM + 3 * C8NUM)[0] = m23 + bias_data[i];

    (dst_data + i + 3 * dst_step * C8NUM)[0] = m30 + bias_data[i];
    (dst_data + i + 3 * dst_step * C8NUM + C8NUM)[0] = m31 + bias_data[i];
    (dst_data + i + 3 * dst_step * C8NUM + 2 * C8NUM)[0] = m32 + bias_data[i];
    (dst_data + i + 3 * dst_step * C8NUM + 3 * C8NUM)[0] = m33 + bias_data[i];
  }
#endif
}

void OutputTransform8x5UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step) {
#ifdef ENABLE_ARM
  float16x8_t src_data_00 = vld1q_f16(src_data + 0 * src_step);
  float16x8_t src_data_01 = vld1q_f16(src_data + 1 * src_step);
  float16x8_t src_data_02 = vld1q_f16(src_data + 2 * src_step);
  float16x8_t src_data_03 = vld1q_f16(src_data + 3 * src_step);
  float16x8_t src_data_04 = vld1q_f16(src_data + 4 * src_step);
  float16x8_t src_data_05 = vld1q_f16(src_data + 5 * src_step);
  float16x8_t src_data_06 = vld1q_f16(src_data + 6 * src_step);
  float16x8_t src_data_07 = vld1q_f16(src_data + 7 * src_step);
  float16x8_t src_data_10 = vld1q_f16(src_data + 8 * src_step);
  float16x8_t src_data_11 = vld1q_f16(src_data + 9 * src_step);
  float16x8_t src_data_12 = vld1q_f16(src_data + 10 * src_step);
  float16x8_t src_data_13 = vld1q_f16(src_data + 11 * src_step);
  float16x8_t src_data_14 = vld1q_f16(src_data + 12 * src_step);
  float16x8_t src_data_15 = vld1q_f16(src_data + 13 * src_step);
  float16x8_t src_data_16 = vld1q_f16(src_data + 14 * src_step);
  float16x8_t src_data_17 = vld1q_f16(src_data + 15 * src_step);
  float16x8_t src_data_20 = vld1q_f16(src_data + 16 * src_step);
  float16x8_t src_data_21 = vld1q_f16(src_data + 17 * src_step);
  float16x8_t src_data_22 = vld1q_f16(src_data + 18 * src_step);
  float16x8_t src_data_23 = vld1q_f16(src_data + 19 * src_step);
  float16x8_t src_data_24 = vld1q_f16(src_data + 20 * src_step);
  float16x8_t src_data_25 = vld1q_f16(src_data + 21 * src_step);
  float16x8_t src_data_26 = vld1q_f16(src_data + 22 * src_step);
  float16x8_t src_data_27 = vld1q_f16(src_data + 23 * src_step);
  float16x8_t src_data_30 = vld1q_f16(src_data + 24 * src_step);
  float16x8_t src_data_31 = vld1q_f16(src_data + 25 * src_step);
  float16x8_t src_data_32 = vld1q_f16(src_data + 26 * src_step);
  float16x8_t src_data_33 = vld1q_f16(src_data + 27 * src_step);
  float16x8_t src_data_34 = vld1q_f16(src_data + 28 * src_step);
  float16x8_t src_data_35 = vld1q_f16(src_data + 29 * src_step);
  float16x8_t src_data_36 = vld1q_f16(src_data + 30 * src_step);
  float16x8_t src_data_37 = vld1q_f16(src_data + 31 * src_step);
  float16x8_t src_data_40 = vld1q_f16(src_data + 32 * src_step);
  float16x8_t src_data_41 = vld1q_f16(src_data + 33 * src_step);
  float16x8_t src_data_42 = vld1q_f16(src_data + 34 * src_step);
  float16x8_t src_data_43 = vld1q_f16(src_data + 35 * src_step);
  float16x8_t src_data_44 = vld1q_f16(src_data + 36 * src_step);
  float16x8_t src_data_45 = vld1q_f16(src_data + 37 * src_step);
  float16x8_t src_data_46 = vld1q_f16(src_data + 38 * src_step);
  float16x8_t src_data_47 = vld1q_f16(src_data + 39 * src_step);
  float16x8_t src_data_50 = vld1q_f16(src_data + 40 * src_step);
  float16x8_t src_data_51 = vld1q_f16(src_data + 41 * src_step);
  float16x8_t src_data_52 = vld1q_f16(src_data + 42 * src_step);
  float16x8_t src_data_53 = vld1q_f16(src_data + 43 * src_step);
  float16x8_t src_data_54 = vld1q_f16(src_data + 44 * src_step);
  float16x8_t src_data_55 = vld1q_f16(src_data + 45 * src_step);
  float16x8_t src_data_56 = vld1q_f16(src_data + 46 * src_step);
  float16x8_t src_data_57 = vld1q_f16(src_data + 47 * src_step);
  float16x8_t src_data_60 = vld1q_f16(src_data + 48 * src_step);
  float16x8_t src_data_61 = vld1q_f16(src_data + 49 * src_step);
  float16x8_t src_data_62 = vld1q_f16(src_data + 50 * src_step);
  float16x8_t src_data_63 = vld1q_f16(src_data + 51 * src_step);
  float16x8_t src_data_64 = vld1q_f16(src_data + 52 * src_step);
  float16x8_t src_data_65 = vld1q_f16(src_data + 53 * src_step);
  float16x8_t src_data_66 = vld1q_f16(src_data + 54 * src_step);
  float16x8_t src_data_67 = vld1q_f16(src_data + 55 * src_step);
  float16x8_t src_data_70 = vld1q_f16(src_data + 56 * src_step);
  float16x8_t src_data_71 = vld1q_f16(src_data + 57 * src_step);
  float16x8_t src_data_72 = vld1q_f16(src_data + 58 * src_step);
  float16x8_t src_data_73 = vld1q_f16(src_data + 59 * src_step);
  float16x8_t src_data_74 = vld1q_f16(src_data + 60 * src_step);
  float16x8_t src_data_75 = vld1q_f16(src_data + 61 * src_step);
  float16x8_t src_data_76 = vld1q_f16(src_data + 62 * src_step);
  float16x8_t src_data_77 = vld1q_f16(src_data + 63 * src_step);

  float16x8_t d01 = vsubq_f16(src_data_10, src_data_20);
  float16x8_t d02 = vsubq_f16(src_data_11, src_data_21);
  float16x8_t d03 = vsubq_f16(src_data_12, src_data_22);
  float16x8_t d04 = vsubq_f16(src_data_13, src_data_23);
  float16x8_t d05 = vsubq_f16(src_data_14, src_data_24);
  float16x8_t d06 = vsubq_f16(src_data_15, src_data_25);
  float16x8_t d07 = vsubq_f16(src_data_16, src_data_26);
  float16x8_t d08 = vsubq_f16(src_data_17, src_data_27);

  float16x8_t d11 = vsubq_f16(src_data_30, src_data_40);
  float16x8_t d12 = vsubq_f16(src_data_31, src_data_41);
  float16x8_t d13 = vsubq_f16(src_data_32, src_data_42);
  float16x8_t d14 = vsubq_f16(src_data_33, src_data_43);
  float16x8_t d15 = vsubq_f16(src_data_34, src_data_44);
  float16x8_t d16 = vsubq_f16(src_data_35, src_data_45);
  float16x8_t d17 = vsubq_f16(src_data_36, src_data_46);
  float16x8_t d18 = vsubq_f16(src_data_37, src_data_47);

  float16x8_t d21 = vsubq_f16(src_data_50, src_data_60);
  float16x8_t d22 = vsubq_f16(src_data_51, src_data_61);
  float16x8_t d23 = vsubq_f16(src_data_52, src_data_62);
  float16x8_t d24 = vsubq_f16(src_data_53, src_data_63);
  float16x8_t d25 = vsubq_f16(src_data_54, src_data_64);
  float16x8_t d26 = vsubq_f16(src_data_55, src_data_65);
  float16x8_t d27 = vsubq_f16(src_data_56, src_data_66);
  float16x8_t d28 = vsubq_f16(src_data_57, src_data_67);

  float16x8_t d31 = vaddq_f16(src_data_10, src_data_20);
  float16x8_t d32 = vaddq_f16(src_data_11, src_data_21);
  float16x8_t d33 = vaddq_f16(src_data_12, src_data_22);
  float16x8_t d34 = vaddq_f16(src_data_13, src_data_23);
  float16x8_t d35 = vaddq_f16(src_data_14, src_data_24);
  float16x8_t d36 = vaddq_f16(src_data_15, src_data_25);
  float16x8_t d37 = vaddq_f16(src_data_16, src_data_26);
  float16x8_t d38 = vaddq_f16(src_data_17, src_data_27);

  float16x8_t d41 = vaddq_f16(src_data_30, src_data_40);
  float16x8_t d42 = vaddq_f16(src_data_31, src_data_41);
  float16x8_t d43 = vaddq_f16(src_data_32, src_data_42);
  float16x8_t d44 = vaddq_f16(src_data_33, src_data_43);
  float16x8_t d45 = vaddq_f16(src_data_34, src_data_44);
  float16x8_t d46 = vaddq_f16(src_data_35, src_data_45);
  float16x8_t d47 = vaddq_f16(src_data_36, src_data_46);
  float16x8_t d48 = vaddq_f16(src_data_37, src_data_47);

  float16x8_t d51 = vaddq_f16(src_data_50, src_data_60);
  float16x8_t d52 = vaddq_f16(src_data_51, src_data_61);
  float16x8_t d53 = vaddq_f16(src_data_52, src_data_62);
  float16x8_t d54 = vaddq_f16(src_data_53, src_data_63);
  float16x8_t d55 = vaddq_f16(src_data_54, src_data_64);
  float16x8_t d56 = vaddq_f16(src_data_55, src_data_65);
  float16x8_t d57 = vaddq_f16(src_data_56, src_data_66);
  float16x8_t d58 = vaddq_f16(src_data_57, src_data_67);

  float16x8_t t00 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_00, src_data_10), src_data_20), src_data_30), src_data_40),
      src_data_50),
    src_data_60);
  float16x8_t t01 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_01, src_data_11), src_data_21), src_data_31), src_data_41),
      src_data_51),
    src_data_61);
  float16x8_t t02 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_02, src_data_12), src_data_22), src_data_32), src_data_42),
      src_data_52),
    src_data_62);
  float16x8_t t03 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_03, src_data_13), src_data_23), src_data_33), src_data_43),
      src_data_53),
    src_data_63);
  float16x8_t t04 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_04, src_data_14), src_data_24), src_data_34), src_data_44),
      src_data_54),
    src_data_64);
  float16x8_t t05 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_05, src_data_15), src_data_25), src_data_35), src_data_45),
      src_data_55),
    src_data_65);
  float16x8_t t06 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_06, src_data_16), src_data_26), src_data_36), src_data_46),
      src_data_56),
    src_data_66);
  float16x8_t t07 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_07, src_data_17), src_data_27), src_data_37), src_data_47),
      src_data_57),
    src_data_67);

  float16x8_t t10 = vaddq_f16(vaddq_f16(vmulq_n_f16(d01, 0.5), d11), vmulq_n_f16(d21, 1.5));
  float16x8_t t11 = vaddq_f16(vaddq_f16(vmulq_n_f16(d02, 0.5), d12), vmulq_n_f16(d22, 1.5));
  float16x8_t t12 = vaddq_f16(vaddq_f16(vmulq_n_f16(d03, 0.5), d13), vmulq_n_f16(d23, 1.5));
  float16x8_t t13 = vaddq_f16(vaddq_f16(vmulq_n_f16(d04, 0.5), d14), vmulq_n_f16(d24, 1.5));
  float16x8_t t14 = vaddq_f16(vaddq_f16(vmulq_n_f16(d05, 0.5), d15), vmulq_n_f16(d25, 1.5));
  float16x8_t t15 = vaddq_f16(vaddq_f16(vmulq_n_f16(d06, 0.5), d16), vmulq_n_f16(d26, 1.5));
  float16x8_t t16 = vaddq_f16(vaddq_f16(vmulq_n_f16(d07, 0.5), d17), vmulq_n_f16(d27, 1.5));
  float16x8_t t17 = vaddq_f16(vaddq_f16(vmulq_n_f16(d08, 0.5), d18), vmulq_n_f16(d28, 1.5));

  float16x8_t t20 = vaddq_f16(vaddq_f16(vmulq_n_f16(d31, 0.25), d41), vmulq_n_f16(d51, 2.25));
  float16x8_t t21 = vaddq_f16(vaddq_f16(vmulq_n_f16(d32, 0.25), d42), vmulq_n_f16(d52, 2.25));
  float16x8_t t22 = vaddq_f16(vaddq_f16(vmulq_n_f16(d33, 0.25), d43), vmulq_n_f16(d53, 2.25));
  float16x8_t t23 = vaddq_f16(vaddq_f16(vmulq_n_f16(d34, 0.25), d44), vmulq_n_f16(d54, 2.25));
  float16x8_t t24 = vaddq_f16(vaddq_f16(vmulq_n_f16(d35, 0.25), d45), vmulq_n_f16(d55, 2.25));
  float16x8_t t25 = vaddq_f16(vaddq_f16(vmulq_n_f16(d36, 0.25), d46), vmulq_n_f16(d56, 2.25));
  float16x8_t t26 = vaddq_f16(vaddq_f16(vmulq_n_f16(d37, 0.25), d47), vmulq_n_f16(d57, 2.25));
  float16x8_t t27 = vaddq_f16(vaddq_f16(vmulq_n_f16(d38, 0.25), d48), vmulq_n_f16(d58, 2.25));

  float16x8_t t30 = vaddq_f16(vaddq_f16(vmulq_n_f16(d01, 0.125), d11), vmulq_n_f16(d21, 3.375));
  float16x8_t t31 = vaddq_f16(vaddq_f16(vmulq_n_f16(d02, 0.125), d12), vmulq_n_f16(d22, 3.375));
  float16x8_t t32 = vaddq_f16(vaddq_f16(vmulq_n_f16(d03, 0.125), d13), vmulq_n_f16(d23, 3.375));
  float16x8_t t33 = vaddq_f16(vaddq_f16(vmulq_n_f16(d04, 0.125), d14), vmulq_n_f16(d24, 3.375));
  float16x8_t t34 = vaddq_f16(vaddq_f16(vmulq_n_f16(d05, 0.125), d15), vmulq_n_f16(d25, 3.375));
  float16x8_t t35 = vaddq_f16(vaddq_f16(vmulq_n_f16(d06, 0.125), d16), vmulq_n_f16(d26, 3.375));
  float16x8_t t36 = vaddq_f16(vaddq_f16(vmulq_n_f16(d07, 0.125), d17), vmulq_n_f16(d27, 3.375));
  float16x8_t t37 = vaddq_f16(vaddq_f16(vmulq_n_f16(d08, 0.125), d18), vmulq_n_f16(d28, 3.375));

  float16x8_t t40 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d31, 0.0625), d41), vmulq_n_f16(d51, 5.0625)), src_data_70);
  float16x8_t t41 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d32, 0.0625), d42), vmulq_n_f16(d52, 5.0625)), src_data_71);
  float16x8_t t42 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d33, 0.0625), d43), vmulq_n_f16(d53, 5.0625)), src_data_72);
  float16x8_t t43 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d34, 0.0625), d44), vmulq_n_f16(d54, 5.0625)), src_data_73);
  float16x8_t t44 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d35, 0.0625), d45), vmulq_n_f16(d55, 5.0625)), src_data_74);
  float16x8_t t45 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d36, 0.0625), d46), vmulq_n_f16(d56, 5.0625)), src_data_75);
  float16x8_t t46 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d37, 0.0625), d47), vmulq_n_f16(d57, 5.0625)), src_data_76);
  float16x8_t t47 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d38, 0.0625), d48), vmulq_n_f16(d58, 5.0625)), src_data_77);

  float16x8_t s11 = vsubq_f16(t01, t02);
  float16x8_t s12 = vsubq_f16(t11, t12);
  float16x8_t s13 = vsubq_f16(t21, t22);
  float16x8_t s14 = vsubq_f16(t31, t32);
  float16x8_t s15 = vsubq_f16(t41, t42);

  float16x8_t s21 = vsubq_f16(t03, t04);
  float16x8_t s22 = vsubq_f16(t13, t14);
  float16x8_t s23 = vsubq_f16(t23, t24);
  float16x8_t s24 = vsubq_f16(t33, t34);
  float16x8_t s25 = vsubq_f16(t43, t44);

  float16x8_t s31 = vsubq_f16(t05, t06);
  float16x8_t s32 = vsubq_f16(t15, t16);
  float16x8_t s33 = vsubq_f16(t25, t26);
  float16x8_t s34 = vsubq_f16(t35, t36);
  float16x8_t s35 = vsubq_f16(t45, t46);

  float16x8_t s41 = vaddq_f16(t01, t02);
  float16x8_t s42 = vaddq_f16(t11, t12);
  float16x8_t s43 = vaddq_f16(t21, t22);
  float16x8_t s44 = vaddq_f16(t31, t32);
  float16x8_t s45 = vaddq_f16(t41, t42);

  float16x8_t s51 = vaddq_f16(t03, t04);
  float16x8_t s52 = vaddq_f16(t13, t14);
  float16x8_t s53 = vaddq_f16(t23, t24);
  float16x8_t s54 = vaddq_f16(t33, t34);
  float16x8_t s55 = vaddq_f16(t43, t44);

  float16x8_t s61 = vaddq_f16(t05, t06);
  float16x8_t s62 = vaddq_f16(t15, t16);
  float16x8_t s63 = vaddq_f16(t25, t26);
  float16x8_t s64 = vaddq_f16(t35, t36);
  float16x8_t s65 = vaddq_f16(t45, t46);

  float16x8_t m00 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t00, t01), t02), t03), t04), t05), t06);
  float16x8_t m01 = vaddq_f16(vaddq_f16(vmulq_n_f16(s11, 0.5), s21), vmulq_n_f16(s31, 1.5));
  float16x8_t m02 = vaddq_f16(vaddq_f16(vmulq_n_f16(s41, 0.25), s51), vmulq_n_f16(s61, 2.25));
  float16x8_t m03 = vaddq_f16(vaddq_f16(vmulq_n_f16(s11, 0.125), s21), vmulq_n_f16(s31, 3.375));
  float16x8_t m04 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s41, 0.0625), s51), vmulq_n_f16(s61, 5.0625)), t07);

  float16x8_t m10 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t10, t11), t12), t13), t14), t15), t16);
  float16x8_t m11 = vaddq_f16(vaddq_f16(vmulq_n_f16(s12, 0.5), s22), vmulq_n_f16(s32, 1.5));
  float16x8_t m12 = vaddq_f16(vaddq_f16(vmulq_n_f16(s42, 0.25), s52), vmulq_n_f16(s62, 2.25));
  float16x8_t m13 = vaddq_f16(vaddq_f16(vmulq_n_f16(s12, 0.125), s22), vmulq_n_f16(s32, 3.375));
  float16x8_t m14 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s42, 0.0625), s52), vmulq_n_f16(s62, 5.0625)), t17);

  float16x8_t m20 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t20, t21), t22), t23), t24), t25), t26);
  float16x8_t m21 = vaddq_f16(vaddq_f16(vmulq_n_f16(s13, 0.5), s23), vmulq_n_f16(s33, 1.5));
  float16x8_t m22 = vaddq_f16(vaddq_f16(vmulq_n_f16(s43, 0.25), s53), vmulq_n_f16(s63, 2.25));
  float16x8_t m23 = vaddq_f16(vaddq_f16(vmulq_n_f16(s13, 0.125), s23), vmulq_n_f16(s33, 3.375));
  float16x8_t m24 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s43, 0.0625), s53), vmulq_n_f16(s63, 5.0625)), t27);

  float16x8_t m30 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t30, t31), t32), t33), t34), t35), t36);
  float16x8_t m31 = vaddq_f16(vaddq_f16(vmulq_n_f16(s14, 0.5), s24), vmulq_n_f16(s34, 1.5));
  float16x8_t m32 = vaddq_f16(vaddq_f16(vmulq_n_f16(s44, 0.25), s54), vmulq_n_f16(s64, 2.25));
  float16x8_t m33 = vaddq_f16(vaddq_f16(vmulq_n_f16(s14, 0.125), s24), vmulq_n_f16(s34, 3.375));
  float16x8_t m34 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s44, 0.0625), s54), vmulq_n_f16(s64, 5.0625)), t37);

  float16x8_t m40 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t40, t41), t42), t43), t44), t45), t46);
  float16x8_t m41 = vaddq_f16(vaddq_f16(vmulq_n_f16(s15, 0.5), s25), vmulq_n_f16(s35, 1.5));
  float16x8_t m42 = vaddq_f16(vaddq_f16(vmulq_n_f16(s45, 0.25), s55), vmulq_n_f16(s65, 2.25));
  float16x8_t m43 = vaddq_f16(vaddq_f16(vmulq_n_f16(s15, 0.125), s25), vmulq_n_f16(s35, 3.375));
  float16x8_t m44 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s45, 0.0625), s55), vmulq_n_f16(s65, 5.0625)), t47);

  float16x8_t bias_ptr = vld1q_f16(bias_data);
  vst1q_f16(dst_data, vaddq_f16(m00, bias_ptr));
  vst1q_f16(dst_data + C8NUM, vaddq_f16(m01, bias_ptr));
  vst1q_f16(dst_data + 2 * C8NUM, vaddq_f16(m02, bias_ptr));
  vst1q_f16(dst_data + 3 * C8NUM, vaddq_f16(m03, bias_ptr));
  vst1q_f16(dst_data + 4 * C8NUM, vaddq_f16(m04, bias_ptr));

  vst1q_f16(dst_data + dst_step * C8NUM, vaddq_f16(m10, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM + C8NUM, vaddq_f16(m11, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM + 2 * C8NUM, vaddq_f16(m12, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM + 3 * C8NUM, vaddq_f16(m13, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM + 4 * C8NUM, vaddq_f16(m14, bias_ptr));

  vst1q_f16(dst_data + 2 * dst_step * C8NUM, vaddq_f16(m20, bias_ptr));
  vst1q_f16(dst_data + 2 * dst_step * C8NUM + C8NUM, vaddq_f16(m21, bias_ptr));
  vst1q_f16(dst_data + 2 * dst_step * C8NUM + 2 * C8NUM, vaddq_f16(m22, bias_ptr));
  vst1q_f16(dst_data + 2 * dst_step * C8NUM + 3 * C8NUM, vaddq_f16(m23, bias_ptr));
  vst1q_f16(dst_data + 2 * dst_step * C8NUM + 4 * C8NUM, vaddq_f16(m24, bias_ptr));

  vst1q_f16(dst_data + 3 * dst_step * C8NUM, vaddq_f16(m30, bias_ptr));
  vst1q_f16(dst_data + 3 * dst_step * C8NUM + C8NUM, vaddq_f16(m31, bias_ptr));
  vst1q_f16(dst_data + 3 * dst_step * C8NUM + 2 * C8NUM, vaddq_f16(m32, bias_ptr));
  vst1q_f16(dst_data + 3 * dst_step * C8NUM + 3 * C8NUM, vaddq_f16(m33, bias_ptr));
  vst1q_f16(dst_data + 3 * dst_step * C8NUM + 4 * C8NUM, vaddq_f16(m34, bias_ptr));

  vst1q_f16(dst_data + 4 * dst_step * C8NUM, vaddq_f16(m40, bias_ptr));
  vst1q_f16(dst_data + 4 * dst_step * C8NUM + C8NUM, vaddq_f16(m41, bias_ptr));
  vst1q_f16(dst_data + 4 * dst_step * C8NUM + 2 * C8NUM, vaddq_f16(m42, bias_ptr));
  vst1q_f16(dst_data + 4 * dst_step * C8NUM + 3 * C8NUM, vaddq_f16(m43, bias_ptr));
  vst1q_f16(dst_data + 4 * dst_step * C8NUM + 4 * C8NUM, vaddq_f16(m44, bias_ptr));
#else
  for (int i = 0; i < C8NUM; i++) {
    float16_t src_data_00 = src_data[i];
    float16_t src_data_01 = src_data[i + src_step];
    float16_t src_data_02 = src_data[i + 2 * src_step];
    float16_t src_data_03 = src_data[i + 3 * src_step];
    float16_t src_data_04 = src_data[i + 4 * src_step];
    float16_t src_data_05 = src_data[i + 5 * src_step];
    float16_t src_data_06 = src_data[i + 6 * src_step];
    float16_t src_data_07 = src_data[i + 7 * src_step];
    float16_t src_data_10 = src_data[i + 8 * src_step];
    float16_t src_data_11 = src_data[i + 9 * src_step];
    float16_t src_data_12 = src_data[i + 10 * src_step];
    float16_t src_data_13 = src_data[i + 11 * src_step];
    float16_t src_data_14 = src_data[i + 12 * src_step];
    float16_t src_data_15 = src_data[i + 13 * src_step];
    float16_t src_data_16 = src_data[i + 14 * src_step];
    float16_t src_data_17 = src_data[i + 15 * src_step];
    float16_t src_data_20 = src_data[i + 16 * src_step];
    float16_t src_data_21 = src_data[i + 17 * src_step];
    float16_t src_data_22 = src_data[i + 18 * src_step];
    float16_t src_data_23 = src_data[i + 19 * src_step];
    float16_t src_data_24 = src_data[i + 20 * src_step];
    float16_t src_data_25 = src_data[i + 21 * src_step];
    float16_t src_data_26 = src_data[i + 22 * src_step];
    float16_t src_data_27 = src_data[i + 23 * src_step];
    float16_t src_data_30 = src_data[i + 24 * src_step];
    float16_t src_data_31 = src_data[i + 25 * src_step];
    float16_t src_data_32 = src_data[i + 26 * src_step];
    float16_t src_data_33 = src_data[i + 27 * src_step];
    float16_t src_data_34 = src_data[i + 28 * src_step];
    float16_t src_data_35 = src_data[i + 29 * src_step];
    float16_t src_data_36 = src_data[i + 30 * src_step];
    float16_t src_data_37 = src_data[i + 31 * src_step];
    float16_t src_data_40 = src_data[i + 32 * src_step];
    float16_t src_data_41 = src_data[i + 33 * src_step];
    float16_t src_data_42 = src_data[i + 34 * src_step];
    float16_t src_data_43 = src_data[i + 35 * src_step];
    float16_t src_data_44 = src_data[i + 36 * src_step];
    float16_t src_data_45 = src_data[i + 37 * src_step];
    float16_t src_data_46 = src_data[i + 38 * src_step];
    float16_t src_data_47 = src_data[i + 39 * src_step];
    float16_t src_data_50 = src_data[i + 40 * src_step];
    float16_t src_data_51 = src_data[i + 41 * src_step];
    float16_t src_data_52 = src_data[i + 42 * src_step];
    float16_t src_data_53 = src_data[i + 43 * src_step];
    float16_t src_data_54 = src_data[i + 44 * src_step];
    float16_t src_data_55 = src_data[i + 45 * src_step];
    float16_t src_data_56 = src_data[i + 46 * src_step];
    float16_t src_data_57 = src_data[i + 47 * src_step];
    float16_t src_data_60 = src_data[i + 48 * src_step];
    float16_t src_data_61 = src_data[i + 49 * src_step];
    float16_t src_data_62 = src_data[i + 50 * src_step];
    float16_t src_data_63 = src_data[i + 51 * src_step];
    float16_t src_data_64 = src_data[i + 52 * src_step];
    float16_t src_data_65 = src_data[i + 53 * src_step];
    float16_t src_data_66 = src_data[i + 54 * src_step];
    float16_t src_data_67 = src_data[i + 55 * src_step];
    float16_t src_data_70 = src_data[i + 56 * src_step];
    float16_t src_data_71 = src_data[i + 57 * src_step];
    float16_t src_data_72 = src_data[i + 58 * src_step];
    float16_t src_data_73 = src_data[i + 59 * src_step];
    float16_t src_data_74 = src_data[i + 60 * src_step];
    float16_t src_data_75 = src_data[i + 61 * src_step];
    float16_t src_data_76 = src_data[i + 62 * src_step];
    float16_t src_data_77 = src_data[i + 63 * src_step];

    float16_t d01 = src_data_10 - src_data_20;
    float16_t d02 = src_data_11 - src_data_21;
    float16_t d03 = src_data_12 - src_data_22;
    float16_t d04 = src_data_13 - src_data_23;
    float16_t d05 = src_data_14 - src_data_24;
    float16_t d06 = src_data_15 - src_data_25;
    float16_t d07 = src_data_16 - src_data_26;
    float16_t d08 = src_data_17 - src_data_27;

    float16_t d11 = src_data_30 - src_data_40;
    float16_t d12 = src_data_31 - src_data_41;
    float16_t d13 = src_data_32 - src_data_42;
    float16_t d14 = src_data_33 - src_data_43;
    float16_t d15 = src_data_34 - src_data_44;
    float16_t d16 = src_data_35 - src_data_45;
    float16_t d17 = src_data_36 - src_data_46;
    float16_t d18 = src_data_37 - src_data_47;

    float16_t d21 = src_data_50 - src_data_60;
    float16_t d22 = src_data_51 - src_data_61;
    float16_t d23 = src_data_52 - src_data_62;
    float16_t d24 = src_data_53 - src_data_63;
    float16_t d25 = src_data_54 - src_data_64;
    float16_t d26 = src_data_55 - src_data_65;
    float16_t d27 = src_data_56 - src_data_66;
    float16_t d28 = src_data_57 - src_data_67;

    float16_t d31 = src_data_10 + src_data_20;
    float16_t d32 = src_data_11 + src_data_21;
    float16_t d33 = src_data_12 + src_data_22;
    float16_t d34 = src_data_13 + src_data_23;
    float16_t d35 = src_data_14 + src_data_24;
    float16_t d36 = src_data_15 + src_data_25;
    float16_t d37 = src_data_16 + src_data_26;
    float16_t d38 = src_data_17 + src_data_27;

    float16_t d41 = src_data_30 + src_data_40;
    float16_t d42 = src_data_31 + src_data_41;
    float16_t d43 = src_data_32 + src_data_42;
    float16_t d44 = src_data_33 + src_data_43;
    float16_t d45 = src_data_34 + src_data_44;
    float16_t d46 = src_data_35 + src_data_45;
    float16_t d47 = src_data_36 + src_data_46;
    float16_t d48 = src_data_37 + src_data_47;

    float16_t d51 = src_data_50 + src_data_60;
    float16_t d52 = src_data_51 + src_data_61;
    float16_t d53 = src_data_52 + src_data_62;
    float16_t d54 = src_data_53 + src_data_63;
    float16_t d55 = src_data_54 + src_data_64;
    float16_t d56 = src_data_55 + src_data_65;
    float16_t d57 = src_data_56 + src_data_66;
    float16_t d58 = src_data_57 + src_data_67;

    float16_t t00 = src_data_00 + src_data_10 + src_data_20 + src_data_30 + src_data_40 + src_data_50 + src_data_60;
    float16_t t01 = src_data_01 + src_data_11 + src_data_21 + src_data_31 + src_data_41 + src_data_51 + src_data_61;
    float16_t t02 = src_data_02 + src_data_12 + src_data_22 + src_data_32 + src_data_42 + src_data_52 + src_data_62;
    float16_t t03 = src_data_03 + src_data_13 + src_data_23 + src_data_33 + src_data_43 + src_data_53 + src_data_63;
    float16_t t04 = src_data_04 + src_data_14 + src_data_24 + src_data_34 + src_data_44 + src_data_54 + src_data_64;
    float16_t t05 = src_data_05 + src_data_15 + src_data_25 + src_data_35 + src_data_45 + src_data_55 + src_data_65;
    float16_t t06 = src_data_06 + src_data_16 + src_data_26 + src_data_36 + src_data_46 + src_data_56 + src_data_66;
    float16_t t07 = src_data_07 + src_data_17 + src_data_27 + src_data_37 + src_data_47 + src_data_57 + src_data_67;

    const float16_t t10 = 0.5f * d01 + d11 + 1.5f * d21;
    const float16_t t11 = 0.5f * d02 + d12 + 1.5f * d22;
    const float16_t t12 = 0.5f * d03 + d13 + 1.5f * d23;
    const float16_t t13 = 0.5f * d04 + d14 + 1.5f * d24;
    const float16_t t14 = 0.5f * d05 + d15 + 1.5f * d25;
    const float16_t t15 = 0.5f * d06 + d16 + 1.5f * d26;
    const float16_t t16 = 0.5f * d07 + d17 + 1.5f * d27;
    const float16_t t17 = 0.5f * d08 + d18 + 1.5f * d28;

    const float16_t t20 = 0.25f * d31 + d41 + 2.25f * d51;
    const float16_t t21 = 0.25f * d32 + d42 + 2.25f * d52;
    const float16_t t22 = 0.25f * d33 + d43 + 2.25f * d53;
    const float16_t t23 = 0.25f * d34 + d44 + 2.25f * d54;
    const float16_t t24 = 0.25f * d35 + d45 + 2.25f * d55;
    const float16_t t25 = 0.25f * d36 + d46 + 2.25f * d56;
    const float16_t t26 = 0.25f * d37 + d47 + 2.25f * d57;
    const float16_t t27 = 0.25f * d38 + d48 + 2.25f * d58;

    const float16_t t30 = 0.125f * d01 + d11 + 3.375f * d21;
    const float16_t t31 = 0.125f * d02 + d12 + 3.375f * d22;
    const float16_t t32 = 0.125f * d03 + d13 + 3.375f * d23;
    const float16_t t33 = 0.125f * d04 + d14 + 3.375f * d24;
    const float16_t t34 = 0.125f * d05 + d15 + 3.375f * d25;
    const float16_t t35 = 0.125f * d06 + d16 + 3.375f * d26;
    const float16_t t36 = 0.125f * d07 + d17 + 3.375f * d27;
    const float16_t t37 = 0.125f * d08 + d18 + 3.375f * d28;

    const float16_t t40 = 0.0625f * d31 + d41 + 5.0625f * d51 + src_data_70;
    const float16_t t41 = 0.0625f * d32 + d42 + 5.0625f * d52 + src_data_71;
    const float16_t t42 = 0.0625f * d33 + d43 + 5.0625f * d53 + src_data_72;
    const float16_t t43 = 0.0625f * d34 + d44 + 5.0625f * d54 + src_data_73;
    const float16_t t44 = 0.0625f * d35 + d45 + 5.0625f * d55 + src_data_74;
    const float16_t t45 = 0.0625f * d36 + d46 + 5.0625f * d56 + src_data_75;
    const float16_t t46 = 0.0625f * d37 + d47 + 5.0625f * d57 + src_data_76;
    const float16_t t47 = 0.0625f * d38 + d48 + 5.0625f * d58 + src_data_77;

    float16_t s11 = t01 - t02;
    float16_t s12 = t11 - t12;
    float16_t s13 = t21 - t22;
    float16_t s14 = t31 - t32;
    float16_t s15 = t41 - t42;

    float16_t s21 = t03 - t04;
    float16_t s22 = t13 - t14;
    float16_t s23 = t23 - t24;
    float16_t s24 = t33 - t34;
    float16_t s25 = t43 - t44;

    float16_t s31 = t05 - t06;
    float16_t s32 = t15 - t16;
    float16_t s33 = t25 - t26;
    float16_t s34 = t35 - t36;
    float16_t s35 = t45 - t46;

    float16_t s41 = t01 + t02;
    float16_t s42 = t11 + t12;
    float16_t s43 = t21 + t22;
    float16_t s44 = t31 + t32;
    float16_t s45 = t41 + t42;

    float16_t s51 = t03 + t04;
    float16_t s52 = t13 + t14;
    float16_t s53 = t23 + t24;
    float16_t s54 = t33 + t34;
    float16_t s55 = t43 + t44;

    float16_t s61 = t05 + t06;
    float16_t s62 = t15 + t16;
    float16_t s63 = t25 + t26;
    float16_t s64 = t35 + t36;
    float16_t s65 = t45 + t46;

    float16_t m00 = t00 + t01 + t02 + t03 + t04 + t05 + t06;
    const float16_t m01 = 0.5f * s11 + s21 + 1.5f * s31;
    const float16_t m02 = 0.25f * s41 + s51 + 2.25f * s61;
    const float16_t m03 = 0.125f * s11 + s21 + 3.375f * s31;
    const float16_t m04 = 0.0625f * s41 + s51 + 5.0625f * s61 + t07;

    float16_t m10 = t10 + t11 + t12 + t13 + t14 + t15 + t16;
    const float16_t m11 = 0.5f * s12 + s22 + 1.5f * s32;
    const float16_t m12 = 0.25f * s42 + s52 + 2.25f * s62;
    const float16_t m13 = 0.125f * s12 + s22 + 3.375f * s32;
    const float16_t m14 = 0.0625f * s42 + s52 + 5.0625f * s62 + t17;

    float16_t m20 = t20 + t21 + t22 + t23 + t24 + t25 + t26;
    const float16_t m21 = 0.5f * s13 + s23 + 1.5f * s33;
    const float16_t m22 = 0.25f * s43 + s53 + 2.25f * s63;
    const float16_t m23 = 0.125f * s13 + s23 + 3.375f * s33;
    const float16_t m24 = 0.0625f * s43 + s53 + 5.0625f * s63 + t27;

    float16_t m30 = t30 + t31 + t32 + t33 + t34 + t35 + t36;
    const float16_t m31 = 0.5f * s14 + s24 + 1.5f * s34;
    const float16_t m32 = 0.25f * s44 + s54 + 2.25f * s64;
    const float16_t m33 = 0.125f * s14 + s24 + 3.375f * s34;
    const float16_t m34 = 0.0625f * s44 + s54 + 5.0625f * s64 + t37;

    float16_t m40 = t40 + t41 + t42 + t43 + t44 + t45 + t46;
    const float16_t m41 = 0.5f * s15 + s25 + 1.5f * s35;
    const float16_t m42 = 0.25f * s45 + s55 + 2.25f * s65;
    const float16_t m43 = 0.125f * s15 + s25 + 3.375f * s35;
    const float16_t m44 = 0.0625f * s45 + s55 + 5.0625f * s65 + t47;

    (dst_data + i)[0] = m00 + bias_data[i];
    (dst_data + i + C8NUM)[0] = m01 + bias_data[i];
    (dst_data + i + 2 * C8NUM)[0] = m02 + bias_data[i];
    (dst_data + i + 3 * C8NUM)[0] = m03 + bias_data[i];
    (dst_data + i + 4 * C8NUM)[0] = m04 + bias_data[i];

    (dst_data + i + dst_step * C8NUM)[0] = m10 + bias_data[i];
    (dst_data + i + dst_step * C8NUM + C8NUM)[0] = m11 + bias_data[i];
    (dst_data + i + dst_step * C8NUM + 2 * C8NUM)[0] = m12 + bias_data[i];
    (dst_data + i + dst_step * C8NUM + 3 * C8NUM)[0] = m13 + bias_data[i];
    (dst_data + i + dst_step * C8NUM + 4 * C8NUM)[0] = m14 + bias_data[i];

    (dst_data + i + 2 * dst_step * C8NUM)[0] = m20 + bias_data[i];
    (dst_data + i + 2 * dst_step * C8NUM + C8NUM)[0] = m21 + bias_data[i];
    (dst_data + i + 2 * dst_step * C8NUM + 2 * C8NUM)[0] = m22 + bias_data[i];
    (dst_data + i + 2 * dst_step * C8NUM + 3 * C8NUM)[0] = m23 + bias_data[i];
    (dst_data + i + 2 * dst_step * C8NUM + 4 * C8NUM)[0] = m24 + bias_data[i];

    (dst_data + i + 3 * dst_step * C8NUM)[0] = m30 + bias_data[i];
    (dst_data + i + 3 * dst_step * C8NUM + C8NUM)[0] = m31 + bias_data[i];
    (dst_data + i + 3 * dst_step * C8NUM + 2 * C8NUM)[0] = m32 + bias_data[i];
    (dst_data + i + 3 * dst_step * C8NUM + 3 * C8NUM)[0] = m33 + bias_data[i];
    (dst_data + i + 3 * dst_step * C8NUM + 4 * C8NUM)[0] = m34 + bias_data[i];

    (dst_data + i + 4 * dst_step * C8NUM)[0] = m40 + bias_data[i];
    (dst_data + i + 4 * dst_step * C8NUM + C8NUM)[0] = m41 + bias_data[i];
    (dst_data + i + 4 * dst_step * C8NUM + 2 * C8NUM)[0] = m42 + bias_data[i];
    (dst_data + i + 4 * dst_step * C8NUM + 3 * C8NUM)[0] = m43 + bias_data[i];
    (dst_data + i + 4 * dst_step * C8NUM + 4 * C8NUM)[0] = m44 + bias_data[i];
  }
#endif
}

void OutputTransform8x6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step) {
#ifdef ENABLE_ARM
  float16x8_t src_data_00 = vld1q_f16(src_data + 0 * src_step);
  float16x8_t src_data_01 = vld1q_f16(src_data + 1 * src_step);
  float16x8_t src_data_02 = vld1q_f16(src_data + 2 * src_step);
  float16x8_t src_data_03 = vld1q_f16(src_data + 3 * src_step);
  float16x8_t src_data_04 = vld1q_f16(src_data + 4 * src_step);
  float16x8_t src_data_05 = vld1q_f16(src_data + 5 * src_step);
  float16x8_t src_data_06 = vld1q_f16(src_data + 6 * src_step);
  float16x8_t src_data_07 = vld1q_f16(src_data + 7 * src_step);
  float16x8_t src_data_10 = vld1q_f16(src_data + 8 * src_step);
  float16x8_t src_data_11 = vld1q_f16(src_data + 9 * src_step);
  float16x8_t src_data_12 = vld1q_f16(src_data + 10 * src_step);
  float16x8_t src_data_13 = vld1q_f16(src_data + 11 * src_step);
  float16x8_t src_data_14 = vld1q_f16(src_data + 12 * src_step);
  float16x8_t src_data_15 = vld1q_f16(src_data + 13 * src_step);
  float16x8_t src_data_16 = vld1q_f16(src_data + 14 * src_step);
  float16x8_t src_data_17 = vld1q_f16(src_data + 15 * src_step);
  float16x8_t src_data_20 = vld1q_f16(src_data + 16 * src_step);
  float16x8_t src_data_21 = vld1q_f16(src_data + 17 * src_step);
  float16x8_t src_data_22 = vld1q_f16(src_data + 18 * src_step);
  float16x8_t src_data_23 = vld1q_f16(src_data + 19 * src_step);
  float16x8_t src_data_24 = vld1q_f16(src_data + 20 * src_step);
  float16x8_t src_data_25 = vld1q_f16(src_data + 21 * src_step);
  float16x8_t src_data_26 = vld1q_f16(src_data + 22 * src_step);
  float16x8_t src_data_27 = vld1q_f16(src_data + 23 * src_step);
  float16x8_t src_data_30 = vld1q_f16(src_data + 24 * src_step);
  float16x8_t src_data_31 = vld1q_f16(src_data + 25 * src_step);
  float16x8_t src_data_32 = vld1q_f16(src_data + 26 * src_step);
  float16x8_t src_data_33 = vld1q_f16(src_data + 27 * src_step);
  float16x8_t src_data_34 = vld1q_f16(src_data + 28 * src_step);
  float16x8_t src_data_35 = vld1q_f16(src_data + 29 * src_step);
  float16x8_t src_data_36 = vld1q_f16(src_data + 30 * src_step);
  float16x8_t src_data_37 = vld1q_f16(src_data + 31 * src_step);
  float16x8_t src_data_40 = vld1q_f16(src_data + 32 * src_step);
  float16x8_t src_data_41 = vld1q_f16(src_data + 33 * src_step);
  float16x8_t src_data_42 = vld1q_f16(src_data + 34 * src_step);
  float16x8_t src_data_43 = vld1q_f16(src_data + 35 * src_step);
  float16x8_t src_data_44 = vld1q_f16(src_data + 36 * src_step);
  float16x8_t src_data_45 = vld1q_f16(src_data + 37 * src_step);
  float16x8_t src_data_46 = vld1q_f16(src_data + 38 * src_step);
  float16x8_t src_data_47 = vld1q_f16(src_data + 39 * src_step);
  float16x8_t src_data_50 = vld1q_f16(src_data + 40 * src_step);
  float16x8_t src_data_51 = vld1q_f16(src_data + 41 * src_step);
  float16x8_t src_data_52 = vld1q_f16(src_data + 42 * src_step);
  float16x8_t src_data_53 = vld1q_f16(src_data + 43 * src_step);
  float16x8_t src_data_54 = vld1q_f16(src_data + 44 * src_step);
  float16x8_t src_data_55 = vld1q_f16(src_data + 45 * src_step);
  float16x8_t src_data_56 = vld1q_f16(src_data + 46 * src_step);
  float16x8_t src_data_57 = vld1q_f16(src_data + 47 * src_step);
  float16x8_t src_data_60 = vld1q_f16(src_data + 48 * src_step);
  float16x8_t src_data_61 = vld1q_f16(src_data + 49 * src_step);
  float16x8_t src_data_62 = vld1q_f16(src_data + 50 * src_step);
  float16x8_t src_data_63 = vld1q_f16(src_data + 51 * src_step);
  float16x8_t src_data_64 = vld1q_f16(src_data + 52 * src_step);
  float16x8_t src_data_65 = vld1q_f16(src_data + 53 * src_step);
  float16x8_t src_data_66 = vld1q_f16(src_data + 54 * src_step);
  float16x8_t src_data_67 = vld1q_f16(src_data + 55 * src_step);
  float16x8_t src_data_70 = vld1q_f16(src_data + 56 * src_step);
  float16x8_t src_data_71 = vld1q_f16(src_data + 57 * src_step);
  float16x8_t src_data_72 = vld1q_f16(src_data + 58 * src_step);
  float16x8_t src_data_73 = vld1q_f16(src_data + 59 * src_step);
  float16x8_t src_data_74 = vld1q_f16(src_data + 60 * src_step);
  float16x8_t src_data_75 = vld1q_f16(src_data + 61 * src_step);
  float16x8_t src_data_76 = vld1q_f16(src_data + 62 * src_step);
  float16x8_t src_data_77 = vld1q_f16(src_data + 63 * src_step);

  float16x8_t d01 = vsubq_f16(src_data_10, src_data_20);
  float16x8_t d02 = vsubq_f16(src_data_11, src_data_21);
  float16x8_t d03 = vsubq_f16(src_data_12, src_data_22);
  float16x8_t d04 = vsubq_f16(src_data_13, src_data_23);
  float16x8_t d05 = vsubq_f16(src_data_14, src_data_24);
  float16x8_t d06 = vsubq_f16(src_data_15, src_data_25);
  float16x8_t d07 = vsubq_f16(src_data_16, src_data_26);
  float16x8_t d08 = vsubq_f16(src_data_17, src_data_27);

  float16x8_t d11 = vsubq_f16(src_data_30, src_data_40);
  float16x8_t d12 = vsubq_f16(src_data_31, src_data_41);
  float16x8_t d13 = vsubq_f16(src_data_32, src_data_42);
  float16x8_t d14 = vsubq_f16(src_data_33, src_data_43);
  float16x8_t d15 = vsubq_f16(src_data_34, src_data_44);
  float16x8_t d16 = vsubq_f16(src_data_35, src_data_45);
  float16x8_t d17 = vsubq_f16(src_data_36, src_data_46);
  float16x8_t d18 = vsubq_f16(src_data_37, src_data_47);

  float16x8_t d21 = vsubq_f16(src_data_50, src_data_60);
  float16x8_t d22 = vsubq_f16(src_data_51, src_data_61);
  float16x8_t d23 = vsubq_f16(src_data_52, src_data_62);
  float16x8_t d24 = vsubq_f16(src_data_53, src_data_63);
  float16x8_t d25 = vsubq_f16(src_data_54, src_data_64);
  float16x8_t d26 = vsubq_f16(src_data_55, src_data_65);
  float16x8_t d27 = vsubq_f16(src_data_56, src_data_66);
  float16x8_t d28 = vsubq_f16(src_data_57, src_data_67);

  float16x8_t d31 = vaddq_f16(src_data_10, src_data_20);
  float16x8_t d32 = vaddq_f16(src_data_11, src_data_21);
  float16x8_t d33 = vaddq_f16(src_data_12, src_data_22);
  float16x8_t d34 = vaddq_f16(src_data_13, src_data_23);
  float16x8_t d35 = vaddq_f16(src_data_14, src_data_24);
  float16x8_t d36 = vaddq_f16(src_data_15, src_data_25);
  float16x8_t d37 = vaddq_f16(src_data_16, src_data_26);
  float16x8_t d38 = vaddq_f16(src_data_17, src_data_27);

  float16x8_t d41 = vaddq_f16(src_data_30, src_data_40);
  float16x8_t d42 = vaddq_f16(src_data_31, src_data_41);
  float16x8_t d43 = vaddq_f16(src_data_32, src_data_42);
  float16x8_t d44 = vaddq_f16(src_data_33, src_data_43);
  float16x8_t d45 = vaddq_f16(src_data_34, src_data_44);
  float16x8_t d46 = vaddq_f16(src_data_35, src_data_45);
  float16x8_t d47 = vaddq_f16(src_data_36, src_data_46);
  float16x8_t d48 = vaddq_f16(src_data_37, src_data_47);

  float16x8_t d51 = vaddq_f16(src_data_50, src_data_60);
  float16x8_t d52 = vaddq_f16(src_data_51, src_data_61);
  float16x8_t d53 = vaddq_f16(src_data_52, src_data_62);
  float16x8_t d54 = vaddq_f16(src_data_53, src_data_63);
  float16x8_t d55 = vaddq_f16(src_data_54, src_data_64);
  float16x8_t d56 = vaddq_f16(src_data_55, src_data_65);
  float16x8_t d57 = vaddq_f16(src_data_56, src_data_66);
  float16x8_t d58 = vaddq_f16(src_data_57, src_data_67);

  float16x8_t t00 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_00, src_data_10), src_data_20), src_data_30), src_data_40),
      src_data_50),
    src_data_60);
  float16x8_t t01 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_01, src_data_11), src_data_21), src_data_31), src_data_41),
      src_data_51),
    src_data_61);
  float16x8_t t02 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_02, src_data_12), src_data_22), src_data_32), src_data_42),
      src_data_52),
    src_data_62);
  float16x8_t t03 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_03, src_data_13), src_data_23), src_data_33), src_data_43),
      src_data_53),
    src_data_63);
  float16x8_t t04 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_04, src_data_14), src_data_24), src_data_34), src_data_44),
      src_data_54),
    src_data_64);
  float16x8_t t05 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_05, src_data_15), src_data_25), src_data_35), src_data_45),
      src_data_55),
    src_data_65);
  float16x8_t t06 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_06, src_data_16), src_data_26), src_data_36), src_data_46),
      src_data_56),
    src_data_66);
  float16x8_t t07 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_07, src_data_17), src_data_27), src_data_37), src_data_47),
      src_data_57),
    src_data_67);

  float16x8_t t10 = vaddq_f16(vaddq_f16(vmulq_n_f16(d01, 0.5), d11), vmulq_n_f16(d21, 1.5));
  float16x8_t t11 = vaddq_f16(vaddq_f16(vmulq_n_f16(d02, 0.5), d12), vmulq_n_f16(d22, 1.5));
  float16x8_t t12 = vaddq_f16(vaddq_f16(vmulq_n_f16(d03, 0.5), d13), vmulq_n_f16(d23, 1.5));
  float16x8_t t13 = vaddq_f16(vaddq_f16(vmulq_n_f16(d04, 0.5), d14), vmulq_n_f16(d24, 1.5));
  float16x8_t t14 = vaddq_f16(vaddq_f16(vmulq_n_f16(d05, 0.5), d15), vmulq_n_f16(d25, 1.5));
  float16x8_t t15 = vaddq_f16(vaddq_f16(vmulq_n_f16(d06, 0.5), d16), vmulq_n_f16(d26, 1.5));
  float16x8_t t16 = vaddq_f16(vaddq_f16(vmulq_n_f16(d07, 0.5), d17), vmulq_n_f16(d27, 1.5));
  float16x8_t t17 = vaddq_f16(vaddq_f16(vmulq_n_f16(d08, 0.5), d18), vmulq_n_f16(d28, 1.5));

  float16x8_t t20 = vaddq_f16(vaddq_f16(vmulq_n_f16(d31, 0.25), d41), vmulq_n_f16(d51, 2.25));
  float16x8_t t21 = vaddq_f16(vaddq_f16(vmulq_n_f16(d32, 0.25), d42), vmulq_n_f16(d52, 2.25));
  float16x8_t t22 = vaddq_f16(vaddq_f16(vmulq_n_f16(d33, 0.25), d43), vmulq_n_f16(d53, 2.25));
  float16x8_t t23 = vaddq_f16(vaddq_f16(vmulq_n_f16(d34, 0.25), d44), vmulq_n_f16(d54, 2.25));
  float16x8_t t24 = vaddq_f16(vaddq_f16(vmulq_n_f16(d35, 0.25), d45), vmulq_n_f16(d55, 2.25));
  float16x8_t t25 = vaddq_f16(vaddq_f16(vmulq_n_f16(d36, 0.25), d46), vmulq_n_f16(d56, 2.25));
  float16x8_t t26 = vaddq_f16(vaddq_f16(vmulq_n_f16(d37, 0.25), d47), vmulq_n_f16(d57, 2.25));
  float16x8_t t27 = vaddq_f16(vaddq_f16(vmulq_n_f16(d38, 0.25), d48), vmulq_n_f16(d58, 2.25));

  float16x8_t t30 = vaddq_f16(vaddq_f16(vmulq_n_f16(d01, 0.125), d11), vmulq_n_f16(d21, 3.375));
  float16x8_t t31 = vaddq_f16(vaddq_f16(vmulq_n_f16(d02, 0.125), d12), vmulq_n_f16(d22, 3.375));
  float16x8_t t32 = vaddq_f16(vaddq_f16(vmulq_n_f16(d03, 0.125), d13), vmulq_n_f16(d23, 3.375));
  float16x8_t t33 = vaddq_f16(vaddq_f16(vmulq_n_f16(d04, 0.125), d14), vmulq_n_f16(d24, 3.375));
  float16x8_t t34 = vaddq_f16(vaddq_f16(vmulq_n_f16(d05, 0.125), d15), vmulq_n_f16(d25, 3.375));
  float16x8_t t35 = vaddq_f16(vaddq_f16(vmulq_n_f16(d06, 0.125), d16), vmulq_n_f16(d26, 3.375));
  float16x8_t t36 = vaddq_f16(vaddq_f16(vmulq_n_f16(d07, 0.125), d17), vmulq_n_f16(d27, 3.375));
  float16x8_t t37 = vaddq_f16(vaddq_f16(vmulq_n_f16(d08, 0.125), d18), vmulq_n_f16(d28, 3.375));

  float16x8_t t40 = vaddq_f16(vaddq_f16(vmulq_n_f16(d31, 0.0625), d41), vmulq_n_f16(d51, 5.0625));
  float16x8_t t41 = vaddq_f16(vaddq_f16(vmulq_n_f16(d32, 0.0625), d42), vmulq_n_f16(d52, 5.0625));
  float16x8_t t42 = vaddq_f16(vaddq_f16(vmulq_n_f16(d33, 0.0625), d43), vmulq_n_f16(d53, 5.0625));
  float16x8_t t43 = vaddq_f16(vaddq_f16(vmulq_n_f16(d34, 0.0625), d44), vmulq_n_f16(d54, 5.0625));
  float16x8_t t44 = vaddq_f16(vaddq_f16(vmulq_n_f16(d35, 0.0625), d45), vmulq_n_f16(d55, 5.0625));
  float16x8_t t45 = vaddq_f16(vaddq_f16(vmulq_n_f16(d36, 0.0625), d46), vmulq_n_f16(d56, 5.0625));
  float16x8_t t46 = vaddq_f16(vaddq_f16(vmulq_n_f16(d37, 0.0625), d47), vmulq_n_f16(d57, 5.0625));
  float16x8_t t47 = vaddq_f16(vaddq_f16(vmulq_n_f16(d38, 0.0625), d48), vmulq_n_f16(d58, 5.0625));

  float16x8_t t50 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d01, 0.03125), d11), vmulq_n_f16(d21, 7.59375)), src_data_70);
  float16x8_t t51 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d02, 0.03125), d12), vmulq_n_f16(d22, 7.59375)), src_data_71);
  float16x8_t t52 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d03, 0.03125), d13), vmulq_n_f16(d23, 7.59375)), src_data_72);
  float16x8_t t53 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d04, 0.03125), d14), vmulq_n_f16(d24, 7.59375)), src_data_73);
  float16x8_t t54 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d05, 0.03125), d15), vmulq_n_f16(d25, 7.59375)), src_data_74);
  float16x8_t t55 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d06, 0.03125), d16), vmulq_n_f16(d26, 7.59375)), src_data_75);
  float16x8_t t56 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d07, 0.03125), d17), vmulq_n_f16(d27, 7.59375)), src_data_76);
  float16x8_t t57 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d08, 0.03125), d18), vmulq_n_f16(d28, 7.59375)), src_data_77);

  float16x8_t s11 = vsubq_f16(t01, t02);
  float16x8_t s12 = vsubq_f16(t11, t12);
  float16x8_t s13 = vsubq_f16(t21, t22);
  float16x8_t s14 = vsubq_f16(t31, t32);
  float16x8_t s15 = vsubq_f16(t41, t42);
  float16x8_t s16 = vsubq_f16(t51, t52);

  float16x8_t s21 = vsubq_f16(t03, t04);
  float16x8_t s22 = vsubq_f16(t13, t14);
  float16x8_t s23 = vsubq_f16(t23, t24);
  float16x8_t s24 = vsubq_f16(t33, t34);
  float16x8_t s25 = vsubq_f16(t43, t44);
  float16x8_t s26 = vsubq_f16(t53, t54);

  float16x8_t s31 = vsubq_f16(t05, t06);
  float16x8_t s32 = vsubq_f16(t15, t16);
  float16x8_t s33 = vsubq_f16(t25, t26);
  float16x8_t s34 = vsubq_f16(t35, t36);
  float16x8_t s35 = vsubq_f16(t45, t46);
  float16x8_t s36 = vsubq_f16(t55, t56);

  float16x8_t s41 = vaddq_f16(t01, t02);
  float16x8_t s42 = vaddq_f16(t11, t12);
  float16x8_t s43 = vaddq_f16(t21, t22);
  float16x8_t s44 = vaddq_f16(t31, t32);
  float16x8_t s45 = vaddq_f16(t41, t42);
  float16x8_t s46 = vaddq_f16(t51, t52);

  float16x8_t s51 = vaddq_f16(t03, t04);
  float16x8_t s52 = vaddq_f16(t13, t14);
  float16x8_t s53 = vaddq_f16(t23, t24);
  float16x8_t s54 = vaddq_f16(t33, t34);
  float16x8_t s55 = vaddq_f16(t43, t44);
  float16x8_t s56 = vaddq_f16(t53, t54);

  float16x8_t s61 = vaddq_f16(t05, t06);
  float16x8_t s62 = vaddq_f16(t15, t16);
  float16x8_t s63 = vaddq_f16(t25, t26);
  float16x8_t s64 = vaddq_f16(t35, t36);
  float16x8_t s65 = vaddq_f16(t45, t46);
  float16x8_t s66 = vaddq_f16(t55, t56);

  float16x8_t m00 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t00, t01), t02), t03), t04), t05), t06);
  float16x8_t m01 = vaddq_f16(vaddq_f16(vmulq_n_f16(s11, 0.5), s21), vmulq_n_f16(s31, 1.5));
  float16x8_t m02 = vaddq_f16(vaddq_f16(vmulq_n_f16(s41, 0.25), s51), vmulq_n_f16(s61, 2.25));
  float16x8_t m03 = vaddq_f16(vaddq_f16(vmulq_n_f16(s11, 0.125), s21), vmulq_n_f16(s31, 3.375));
  float16x8_t m04 = vaddq_f16(vaddq_f16(vmulq_n_f16(s41, 0.0625), s51), vmulq_n_f16(s61, 5.0625));
  float16x8_t m05 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s11, 0.03125), s21), vmulq_n_f16(s31, 7.59375)), t07);

  float16x8_t m10 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t10, t11), t12), t13), t14), t15), t16);
  float16x8_t m11 = vaddq_f16(vaddq_f16(vmulq_n_f16(s12, 0.5), s22), vmulq_n_f16(s32, 1.5));
  float16x8_t m12 = vaddq_f16(vaddq_f16(vmulq_n_f16(s42, 0.25), s52), vmulq_n_f16(s62, 2.25));
  float16x8_t m13 = vaddq_f16(vaddq_f16(vmulq_n_f16(s12, 0.125), s22), vmulq_n_f16(s32, 3.375));
  float16x8_t m14 = vaddq_f16(vaddq_f16(vmulq_n_f16(s42, 0.0625), s52), vmulq_n_f16(s62, 5.0625));
  float16x8_t m15 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s12, 0.03125), s22), vmulq_n_f16(s32, 7.59375)), t17);

  float16x8_t m20 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t20, t21), t22), t23), t24), t25), t26);
  float16x8_t m21 = vaddq_f16(vaddq_f16(vmulq_n_f16(s13, 0.5), s23), vmulq_n_f16(s33, 1.5));
  float16x8_t m22 = vaddq_f16(vaddq_f16(vmulq_n_f16(s43, 0.25), s53), vmulq_n_f16(s63, 2.25));
  float16x8_t m23 = vaddq_f16(vaddq_f16(vmulq_n_f16(s13, 0.125), s23), vmulq_n_f16(s33, 3.375));
  float16x8_t m24 = vaddq_f16(vaddq_f16(vmulq_n_f16(s43, 0.0625), s53), vmulq_n_f16(s63, 5.0625));
  float16x8_t m25 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s13, 0.03125), s23), vmulq_n_f16(s33, 7.59375)), t27);

  float16x8_t m30 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t30, t31), t32), t33), t34), t35), t36);
  float16x8_t m31 = vaddq_f16(vaddq_f16(vmulq_n_f16(s14, 0.5), s24), vmulq_n_f16(s34, 1.5));
  float16x8_t m32 = vaddq_f16(vaddq_f16(vmulq_n_f16(s44, 0.25), s54), vmulq_n_f16(s64, 2.25));
  float16x8_t m33 = vaddq_f16(vaddq_f16(vmulq_n_f16(s14, 0.125), s24), vmulq_n_f16(s34, 3.375));
  float16x8_t m34 = vaddq_f16(vaddq_f16(vmulq_n_f16(s44, 0.0625), s54), vmulq_n_f16(s64, 5.0625));
  float16x8_t m35 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s14, 0.03125), s24), vmulq_n_f16(s34, 7.59375)), t37);

  float16x8_t m40 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t40, t41), t42), t43), t44), t45), t46);
  float16x8_t m41 = vaddq_f16(vaddq_f16(vmulq_n_f16(s15, 0.5), s25), vmulq_n_f16(s35, 1.5));
  float16x8_t m42 = vaddq_f16(vaddq_f16(vmulq_n_f16(s45, 0.25), s55), vmulq_n_f16(s65, 2.25));
  float16x8_t m43 = vaddq_f16(vaddq_f16(vmulq_n_f16(s15, 0.125), s25), vmulq_n_f16(s35, 3.375));
  float16x8_t m44 = vaddq_f16(vaddq_f16(vmulq_n_f16(s45, 0.0625), s55), vmulq_n_f16(s65, 5.0625));
  float16x8_t m45 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s15, 0.03125), s25), vmulq_n_f16(s35, 7.59375)), t47);

  float16x8_t m50 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t50, t51), t52), t53), t54), t55), t56);
  float16x8_t m51 = vaddq_f16(vaddq_f16(vmulq_n_f16(s16, 0.5), s26), vmulq_n_f16(s36, 1.5));
  float16x8_t m52 = vaddq_f16(vaddq_f16(vmulq_n_f16(s46, 0.25), s56), vmulq_n_f16(s66, 2.25));
  float16x8_t m53 = vaddq_f16(vaddq_f16(vmulq_n_f16(s16, 0.125), s26), vmulq_n_f16(s36, 3.375));
  float16x8_t m54 = vaddq_f16(vaddq_f16(vmulq_n_f16(s46, 0.0625), s56), vmulq_n_f16(s66, 5.0625));
  float16x8_t m55 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s16, 0.03125), s26), vmulq_n_f16(s36, 7.59375)), t57);

  float16x8_t bias_ptr = vld1q_f16(bias_data);
  vst1q_f16(dst_data, vaddq_f16(m00, bias_ptr));
  vst1q_f16(dst_data + C8NUM, vaddq_f16(m01, bias_ptr));
  vst1q_f16(dst_data + 2 * C8NUM, vaddq_f16(m02, bias_ptr));
  vst1q_f16(dst_data + 3 * C8NUM, vaddq_f16(m03, bias_ptr));
  vst1q_f16(dst_data + 4 * C8NUM, vaddq_f16(m04, bias_ptr));
  vst1q_f16(dst_data + 5 * C8NUM, vaddq_f16(m05, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM, vaddq_f16(m10, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM + C8NUM, vaddq_f16(m11, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM + 2 * C8NUM, vaddq_f16(m12, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM + 3 * C8NUM, vaddq_f16(m13, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM + 4 * C8NUM, vaddq_f16(m14, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM + 5 * C8NUM, vaddq_f16(m15, bias_ptr));
  vst1q_f16(dst_data + 2 * dst_step * C8NUM, vaddq_f16(m20, bias_ptr));
  vst1q_f16(dst_data + 2 * dst_step * C8NUM + C8NUM, vaddq_f16(m21, bias_ptr));
  vst1q_f16(dst_data + 2 * dst_step * C8NUM + 2 * C8NUM, vaddq_f16(m22, bias_ptr));
  vst1q_f16(dst_data + 2 * dst_step * C8NUM + 3 * C8NUM, vaddq_f16(m23, bias_ptr));
  vst1q_f16(dst_data + 2 * dst_step * C8NUM + 4 * C8NUM, vaddq_f16(m24, bias_ptr));
  vst1q_f16(dst_data + 2 * dst_step * C8NUM + 5 * C8NUM, vaddq_f16(m25, bias_ptr));
  vst1q_f16(dst_data + 3 * dst_step * C8NUM, vaddq_f16(m30, bias_ptr));
  vst1q_f16(dst_data + 3 * dst_step * C8NUM + C8NUM, vaddq_f16(m31, bias_ptr));
  vst1q_f16(dst_data + 3 * dst_step * C8NUM + 2 * C8NUM, vaddq_f16(m32, bias_ptr));
  vst1q_f16(dst_data + 3 * dst_step * C8NUM + 3 * C8NUM, vaddq_f16(m33, bias_ptr));
  vst1q_f16(dst_data + 3 * dst_step * C8NUM + 4 * C8NUM, vaddq_f16(m34, bias_ptr));
  vst1q_f16(dst_data + 3 * dst_step * C8NUM + 5 * C8NUM, vaddq_f16(m35, bias_ptr));
  vst1q_f16(dst_data + 4 * dst_step * C8NUM, vaddq_f16(m40, bias_ptr));
  vst1q_f16(dst_data + 4 * dst_step * C8NUM + C8NUM, vaddq_f16(m41, bias_ptr));
  vst1q_f16(dst_data + 4 * dst_step * C8NUM + 2 * C8NUM, vaddq_f16(m42, bias_ptr));
  vst1q_f16(dst_data + 4 * dst_step * C8NUM + 3 * C8NUM, vaddq_f16(m43, bias_ptr));
  vst1q_f16(dst_data + 4 * dst_step * C8NUM + 4 * C8NUM, vaddq_f16(m44, bias_ptr));
  vst1q_f16(dst_data + 4 * dst_step * C8NUM + 5 * C8NUM, vaddq_f16(m45, bias_ptr));
  vst1q_f16(dst_data + 5 * dst_step * C8NUM, vaddq_f16(m50, bias_ptr));
  vst1q_f16(dst_data + 5 * dst_step * C8NUM + C8NUM, vaddq_f16(m51, bias_ptr));
  vst1q_f16(dst_data + 5 * dst_step * C8NUM + 2 * C8NUM, vaddq_f16(m52, bias_ptr));
  vst1q_f16(dst_data + 5 * dst_step * C8NUM + 3 * C8NUM, vaddq_f16(m53, bias_ptr));
  vst1q_f16(dst_data + 5 * dst_step * C8NUM + 4 * C8NUM, vaddq_f16(m54, bias_ptr));
  vst1q_f16(dst_data + 5 * dst_step * C8NUM + 5 * C8NUM, vaddq_f16(m55, bias_ptr));
#else
  for (int i = 0; i < C8NUM; i++) {
    float16_t src_data_00 = src_data[i];
    float16_t src_data_01 = src_data[i + src_step];
    float16_t src_data_02 = src_data[i + 2 * src_step];
    float16_t src_data_03 = src_data[i + 3 * src_step];
    float16_t src_data_04 = src_data[i + 4 * src_step];
    float16_t src_data_05 = src_data[i + 5 * src_step];
    float16_t src_data_06 = src_data[i + 6 * src_step];
    float16_t src_data_07 = src_data[i + 7 * src_step];
    float16_t src_data_10 = src_data[i + 8 * src_step];
    float16_t src_data_11 = src_data[i + 9 * src_step];
    float16_t src_data_12 = src_data[i + 10 * src_step];
    float16_t src_data_13 = src_data[i + 11 * src_step];
    float16_t src_data_14 = src_data[i + 12 * src_step];
    float16_t src_data_15 = src_data[i + 13 * src_step];
    float16_t src_data_16 = src_data[i + 14 * src_step];
    float16_t src_data_17 = src_data[i + 15 * src_step];
    float16_t src_data_20 = src_data[i + 16 * src_step];
    float16_t src_data_21 = src_data[i + 17 * src_step];
    float16_t src_data_22 = src_data[i + 18 * src_step];
    float16_t src_data_23 = src_data[i + 19 * src_step];
    float16_t src_data_24 = src_data[i + 20 * src_step];
    float16_t src_data_25 = src_data[i + 21 * src_step];
    float16_t src_data_26 = src_data[i + 22 * src_step];
    float16_t src_data_27 = src_data[i + 23 * src_step];
    float16_t src_data_30 = src_data[i + 24 * src_step];
    float16_t src_data_31 = src_data[i + 25 * src_step];
    float16_t src_data_32 = src_data[i + 26 * src_step];
    float16_t src_data_33 = src_data[i + 27 * src_step];
    float16_t src_data_34 = src_data[i + 28 * src_step];
    float16_t src_data_35 = src_data[i + 29 * src_step];
    float16_t src_data_36 = src_data[i + 30 * src_step];
    float16_t src_data_37 = src_data[i + 31 * src_step];
    float16_t src_data_40 = src_data[i + 32 * src_step];
    float16_t src_data_41 = src_data[i + 33 * src_step];
    float16_t src_data_42 = src_data[i + 34 * src_step];
    float16_t src_data_43 = src_data[i + 35 * src_step];
    float16_t src_data_44 = src_data[i + 36 * src_step];
    float16_t src_data_45 = src_data[i + 37 * src_step];
    float16_t src_data_46 = src_data[i + 38 * src_step];
    float16_t src_data_47 = src_data[i + 39 * src_step];
    float16_t src_data_50 = src_data[i + 40 * src_step];
    float16_t src_data_51 = src_data[i + 41 * src_step];
    float16_t src_data_52 = src_data[i + 42 * src_step];
    float16_t src_data_53 = src_data[i + 43 * src_step];
    float16_t src_data_54 = src_data[i + 44 * src_step];
    float16_t src_data_55 = src_data[i + 45 * src_step];
    float16_t src_data_56 = src_data[i + 46 * src_step];
    float16_t src_data_57 = src_data[i + 47 * src_step];
    float16_t src_data_60 = src_data[i + 48 * src_step];
    float16_t src_data_61 = src_data[i + 49 * src_step];
    float16_t src_data_62 = src_data[i + 50 * src_step];
    float16_t src_data_63 = src_data[i + 51 * src_step];
    float16_t src_data_64 = src_data[i + 52 * src_step];
    float16_t src_data_65 = src_data[i + 53 * src_step];
    float16_t src_data_66 = src_data[i + 54 * src_step];
    float16_t src_data_67 = src_data[i + 55 * src_step];
    float16_t src_data_70 = src_data[i + 56 * src_step];
    float16_t src_data_71 = src_data[i + 57 * src_step];
    float16_t src_data_72 = src_data[i + 58 * src_step];
    float16_t src_data_73 = src_data[i + 59 * src_step];
    float16_t src_data_74 = src_data[i + 60 * src_step];
    float16_t src_data_75 = src_data[i + 61 * src_step];
    float16_t src_data_76 = src_data[i + 62 * src_step];
    float16_t src_data_77 = src_data[i + 63 * src_step];

    float16_t d01 = src_data_10 - src_data_20;
    float16_t d02 = src_data_11 - src_data_21;
    float16_t d03 = src_data_12 - src_data_22;
    float16_t d04 = src_data_13 - src_data_23;
    float16_t d05 = src_data_14 - src_data_24;
    float16_t d06 = src_data_15 - src_data_25;
    float16_t d07 = src_data_16 - src_data_26;
    float16_t d08 = src_data_17 - src_data_27;

    float16_t d11 = src_data_30 - src_data_40;
    float16_t d12 = src_data_31 - src_data_41;
    float16_t d13 = src_data_32 - src_data_42;
    float16_t d14 = src_data_33 - src_data_43;
    float16_t d15 = src_data_34 - src_data_44;
    float16_t d16 = src_data_35 - src_data_45;
    float16_t d17 = src_data_36 - src_data_46;
    float16_t d18 = src_data_37 - src_data_47;

    float16_t d21 = src_data_50 - src_data_60;
    float16_t d22 = src_data_51 - src_data_61;
    float16_t d23 = src_data_52 - src_data_62;
    float16_t d24 = src_data_53 - src_data_63;
    float16_t d25 = src_data_54 - src_data_64;
    float16_t d26 = src_data_55 - src_data_65;
    float16_t d27 = src_data_56 - src_data_66;
    float16_t d28 = src_data_57 - src_data_67;

    float16_t d31 = src_data_10 + src_data_20;
    float16_t d32 = src_data_11 + src_data_21;
    float16_t d33 = src_data_12 + src_data_22;
    float16_t d34 = src_data_13 + src_data_23;
    float16_t d35 = src_data_14 + src_data_24;
    float16_t d36 = src_data_15 + src_data_25;
    float16_t d37 = src_data_16 + src_data_26;
    float16_t d38 = src_data_17 + src_data_27;

    float16_t d41 = src_data_30 + src_data_40;
    float16_t d42 = src_data_31 + src_data_41;
    float16_t d43 = src_data_32 + src_data_42;
    float16_t d44 = src_data_33 + src_data_43;
    float16_t d45 = src_data_34 + src_data_44;
    float16_t d46 = src_data_35 + src_data_45;
    float16_t d47 = src_data_36 + src_data_46;
    float16_t d48 = src_data_37 + src_data_47;

    float16_t d51 = src_data_50 + src_data_60;
    float16_t d52 = src_data_51 + src_data_61;
    float16_t d53 = src_data_52 + src_data_62;
    float16_t d54 = src_data_53 + src_data_63;
    float16_t d55 = src_data_54 + src_data_64;
    float16_t d56 = src_data_55 + src_data_65;
    float16_t d57 = src_data_56 + src_data_66;
    float16_t d58 = src_data_57 + src_data_67;

    float16_t t00 = src_data_00 + src_data_10 + src_data_20 + src_data_30 + src_data_40 + src_data_50 + src_data_60;
    float16_t t01 = src_data_01 + src_data_11 + src_data_21 + src_data_31 + src_data_41 + src_data_51 + src_data_61;
    float16_t t02 = src_data_02 + src_data_12 + src_data_22 + src_data_32 + src_data_42 + src_data_52 + src_data_62;
    float16_t t03 = src_data_03 + src_data_13 + src_data_23 + src_data_33 + src_data_43 + src_data_53 + src_data_63;
    float16_t t04 = src_data_04 + src_data_14 + src_data_24 + src_data_34 + src_data_44 + src_data_54 + src_data_64;
    float16_t t05 = src_data_05 + src_data_15 + src_data_25 + src_data_35 + src_data_45 + src_data_55 + src_data_65;
    float16_t t06 = src_data_06 + src_data_16 + src_data_26 + src_data_36 + src_data_46 + src_data_56 + src_data_66;
    float16_t t07 = src_data_07 + src_data_17 + src_data_27 + src_data_37 + src_data_47 + src_data_57 + src_data_67;

    const float16_t t10 = 0.5f * d01 + d11 + 1.5f * d21;
    const float16_t t11 = 0.5f * d02 + d12 + 1.5f * d22;
    const float16_t t12 = 0.5f * d03 + d13 + 1.5f * d23;
    const float16_t t13 = 0.5f * d04 + d14 + 1.5f * d24;
    const float16_t t14 = 0.5f * d05 + d15 + 1.5f * d25;
    const float16_t t15 = 0.5f * d06 + d16 + 1.5f * d26;
    const float16_t t16 = 0.5f * d07 + d17 + 1.5f * d27;
    const float16_t t17 = 0.5f * d08 + d18 + 1.5f * d28;

    const float16_t t20 = 0.25f * d31 + d41 + 2.25f * d51;
    const float16_t t21 = 0.25f * d32 + d42 + 2.25f * d52;
    const float16_t t22 = 0.25f * d33 + d43 + 2.25f * d53;
    const float16_t t23 = 0.25f * d34 + d44 + 2.25f * d54;
    const float16_t t24 = 0.25f * d35 + d45 + 2.25f * d55;
    const float16_t t25 = 0.25f * d36 + d46 + 2.25f * d56;
    const float16_t t26 = 0.25f * d37 + d47 + 2.25f * d57;
    const float16_t t27 = 0.25f * d38 + d48 + 2.25f * d58;

    const float16_t t30 = 0.125f * d01 + d11 + 3.375f * d21;
    const float16_t t31 = 0.125f * d02 + d12 + 3.375f * d22;
    const float16_t t32 = 0.125f * d03 + d13 + 3.375f * d23;
    const float16_t t33 = 0.125f * d04 + d14 + 3.375f * d24;
    const float16_t t34 = 0.125f * d05 + d15 + 3.375f * d25;
    const float16_t t35 = 0.125f * d06 + d16 + 3.375f * d26;
    const float16_t t36 = 0.125f * d07 + d17 + 3.375f * d27;
    const float16_t t37 = 0.125f * d08 + d18 + 3.375f * d28;

    const float16_t t40 = 0.0625f * d31 + d41 + 5.0625f * d51;
    const float16_t t41 = 0.0625f * d32 + d42 + 5.0625f * d52;
    const float16_t t42 = 0.0625f * d33 + d43 + 5.0625f * d53;
    const float16_t t43 = 0.0625f * d34 + d44 + 5.0625f * d54;
    const float16_t t44 = 0.0625f * d35 + d45 + 5.0625f * d55;
    const float16_t t45 = 0.0625f * d36 + d46 + 5.0625f * d56;
    const float16_t t46 = 0.0625f * d37 + d47 + 5.0625f * d57;
    const float16_t t47 = 0.0625f * d38 + d48 + 5.0625f * d58;

    const float16_t t50 = 0.03125f * d01 + d11 + 7.59375f * d21 + src_data_70;
    const float16_t t51 = 0.03125f * d02 + d12 + 7.59375f * d22 + src_data_71;
    const float16_t t52 = 0.03125f * d03 + d13 + 7.59375f * d23 + src_data_72;
    const float16_t t53 = 0.03125f * d04 + d14 + 7.59375f * d24 + src_data_73;
    const float16_t t54 = 0.03125f * d05 + d15 + 7.59375f * d25 + src_data_74;
    const const float16_t t55 = 0.03125f * d06 + d16 + 7.59375f * d26 + src_data_75;
    const float16_t t56 = 0.03125f * d07 + d17 + 7.59375f * d27 + src_data_76;
    const float16_t t57 = 0.03125f * d08 + d18 + 7.59375f * d28 + src_data_77;

    float16_t s11 = t01 - t02;
    float16_t s12 = t11 - t12;
    float16_t s13 = t21 - t22;
    float16_t s14 = t31 - t32;
    float16_t s15 = t41 - t42;
    float16_t s16 = t51 - t52;

    float16_t s21 = t03 - t04;
    float16_t s22 = t13 - t14;
    float16_t s23 = t23 - t24;
    float16_t s24 = t33 - t34;
    float16_t s25 = t43 - t44;
    float16_t s26 = t53 - t54;

    float16_t s31 = t05 - t06;
    float16_t s32 = t15 - t16;
    float16_t s33 = t25 - t26;
    float16_t s34 = t35 - t36;
    float16_t s35 = t45 - t46;
    float16_t s36 = t55 - t56;

    float16_t s41 = t01 + t02;
    float16_t s42 = t11 + t12;
    float16_t s43 = t21 + t22;
    float16_t s44 = t31 + t32;
    float16_t s45 = t41 + t42;
    float16_t s46 = t51 + t52;

    float16_t s51 = t03 + t04;
    float16_t s52 = t13 + t14;
    float16_t s53 = t23 + t24;
    float16_t s54 = t33 + t34;
    float16_t s55 = t43 + t44;
    float16_t s56 = t53 + t54;

    float16_t s61 = t05 + t06;
    float16_t s62 = t15 + t16;
    float16_t s63 = t25 + t26;
    float16_t s64 = t35 + t36;
    float16_t s65 = t45 + t46;
    float16_t s66 = t55 + t56;

    float16_t m00 = t00 + t01 + t02 + t03 + t04 + t05 + t06;
    const float16_t m01 = 0.5f * s11 + s21 + 1.5f * s31;
    const float16_t m02 = 0.25f * s41 + s51 + 2.25f * s61;
    const float16_t m03 = 0.125f * s11 + s21 + 3.375f * s31;
    const float16_t m04 = 0.0625f * s41 + s51 + 5.0625f * s61;
    const float16_t m05 = 0.03125f * s11 + s21 + 7.59375f * s31 + t07;

    float16_t m10 = t10 + t11 + t12 + t13 + t14 + t15 + t16;
    const float16_t m11 = 0.5f * s12 + s22 + 1.5f * s32;
    const float16_t m12 = 0.25f * s42 + s52 + 2.25f * s62;
    const float16_t m13 = 0.125f * s12 + s22 + 3.375f * s32;
    const float16_t m14 = 0.0625f * s42 + s52 + 5.0625f * s62;
    const float16_t m15 = 0.03125f * s12 + s22 + 7.59375f * s32 + t17;

    float16_t m20 = t20 + t21 + t22 + t23 + t24 + t25 + t26;
    const float16_t m21 = 0.5f * s13 + s23 + 1.5f * s33;
    const float16_t m22 = 0.25f * s43 + s53 + 2.25f * s63;
    const float16_t m23 = 0.125f * s13 + s23 + 3.375f * s33;
    const float16_t m24 = 0.0625f * s43 + s53 + 5.0625f * s63;
    const float16_t m25 = 0.03125f * s13 + s23 + 7.59375f * s33 + t27;

    float16_t m30 = t30 + t31 + t32 + t33 + t34 + t35 + t36;
    const float16_t m31 = 0.5f * s14 + s24 + 1.5f * s34;
    const float16_t m32 = 0.25f * s44 + s54 + 2.25f * s64;
    const float16_t m33 = 0.125f * s14 + s24 + 3.375f * s34;
    const float16_t m34 = 0.0625f * s44 + s54 + 5.0625f * s64;
    const float16_t m35 = 0.03125f * s14 + s24 + 7.59375f * s34 + t37;

    float16_t m40 = t40 + t41 + t42 + t43 + t44 + t45 + t46;
    const float16_t m41 = 0.5f * s15 + s25 + 1.5f * s35;
    const float16_t m42 = 0.25f * s45 + s55 + 2.25f * s65;
    const float16_t m43 = 0.125f * s15 + s25 + 3.375f * s35;
    const float16_t m44 = 0.0625f * s45 + s55 + 5.0625f * s65;
    const float16_t m45 = 0.03125f * s15 + s25 + 7.59375f * s35 + t47;

    float16_t m50 = t50 + t51 + t52 + t53 + t54 + t55 + t56;
    const float16_t m51 = 0.5f * s16 + s26 + 1.5f * s36;
    const float16_t m52 = 0.25f * s46 + s56 + 2.25f * s66;
    const float16_t m53 = 0.125f * s16 + s26 + 3.375f * s36;
    const float16_t m54 = 0.0625f * s46 + s56 + 5.0625f * s66;
    const float16_t m55 = 0.03125f * s16 + s26 + 7.59375f * s36 + t57;

    (dst_data + i)[0] = m00 + bias_data[i];
    (dst_data + i + C8NUM)[0] = m01 + bias_data[i];
    (dst_data + i + 2 * C8NUM)[0] = m02 + bias_data[i];
    (dst_data + i + 3 * C8NUM)[0] = m03 + bias_data[i];
    (dst_data + i + 4 * C8NUM)[0] = m04 + bias_data[i];
    (dst_data + i + 5 * C8NUM)[0] = m05 + bias_data[i];

    (dst_data + i + dst_step * C8NUM)[0] = m10 + bias_data[i];
    (dst_data + i + dst_step * C8NUM + C8NUM)[0] = m11 + bias_data[i];
    (dst_data + i + dst_step * C8NUM + 2 * C8NUM)[0] = m12 + bias_data[i];
    (dst_data + i + dst_step * C8NUM + 3 * C8NUM)[0] = m13 + bias_data[i];
    (dst_data + i + dst_step * C8NUM + 4 * C8NUM)[0] = m14 + bias_data[i];
    (dst_data + i + dst_step * C8NUM + 5 * C8NUM)[0] = m15 + bias_data[i];

    (dst_data + i + 2 * dst_step * C8NUM)[0] = m20 + bias_data[i];
    (dst_data + i + 2 * dst_step * C8NUM + C8NUM)[0] = m21 + bias_data[i];
    (dst_data + i + 2 * dst_step * C8NUM + 2 * C8NUM)[0] = m22 + bias_data[i];
    (dst_data + i + 2 * dst_step * C8NUM + 3 * C8NUM)[0] = m23 + bias_data[i];
    (dst_data + i + 2 * dst_step * C8NUM + 4 * C8NUM)[0] = m24 + bias_data[i];
    (dst_data + i + 2 * dst_step * C8NUM + 5 * C8NUM)[0] = m25 + bias_data[i];

    (dst_data + i + 3 * dst_step * C8NUM)[0] = m30 + bias_data[i];
    (dst_data + i + 3 * dst_step * C8NUM + C8NUM)[0] = m31 + bias_data[i];
    (dst_data + i + 3 * dst_step * C8NUM + 2 * C8NUM)[0] = m32 + bias_data[i];
    (dst_data + i + 3 * dst_step * C8NUM + 3 * C8NUM)[0] = m33 + bias_data[i];
    (dst_data + i + 3 * dst_step * C8NUM + 4 * C8NUM)[0] = m34 + bias_data[i];
    (dst_data + i + 3 * dst_step * C8NUM + 5 * C8NUM)[0] = m35 + bias_data[i];

    (dst_data + i + 4 * dst_step * C8NUM)[0] = m40 + bias_data[i];
    (dst_data + i + 4 * dst_step * C8NUM + C8NUM)[0] = m41 + bias_data[i];
    (dst_data + i + 4 * dst_step * C8NUM + 2 * C8NUM)[0] = m42 + bias_data[i];
    (dst_data + i + 4 * dst_step * C8NUM + 3 * C8NUM)[0] = m43 + bias_data[i];
    (dst_data + i + 4 * dst_step * C8NUM + 4 * C8NUM)[0] = m44 + bias_data[i];
    (dst_data + i + 4 * dst_step * C8NUM + 5 * C8NUM)[0] = m45 + bias_data[i];

    (dst_data + i + 5 * dst_step * C8NUM)[0] = m50 + bias_data[i];
    (dst_data + i + 5 * dst_step * C8NUM + C8NUM)[0] = m51 + bias_data[i];
    (dst_data + i + 5 * dst_step * C8NUM + 2 * C8NUM)[0] = m52 + bias_data[i];
    (dst_data + i + 5 * dst_step * C8NUM + 3 * C8NUM)[0] = m53 + bias_data[i];
    (dst_data + i + 5 * dst_step * C8NUM + 4 * C8NUM)[0] = m54 + bias_data[i];
    (dst_data + i + 5 * dst_step * C8NUM + 5 * C8NUM)[0] = m55 + bias_data[i];
  }
#endif
}

void OutputTransform8x7UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step) {
#ifdef ENABLE_ARM
  float16x8_t src_data_00 = vld1q_f16(src_data + 0 * src_step);
  float16x8_t src_data_01 = vld1q_f16(src_data + 1 * src_step);
  float16x8_t src_data_02 = vld1q_f16(src_data + 2 * src_step);
  float16x8_t src_data_03 = vld1q_f16(src_data + 3 * src_step);
  float16x8_t src_data_04 = vld1q_f16(src_data + 4 * src_step);
  float16x8_t src_data_05 = vld1q_f16(src_data + 5 * src_step);
  float16x8_t src_data_06 = vld1q_f16(src_data + 6 * src_step);
  float16x8_t src_data_07 = vld1q_f16(src_data + 7 * src_step);
  float16x8_t src_data_10 = vld1q_f16(src_data + 8 * src_step);
  float16x8_t src_data_11 = vld1q_f16(src_data + 9 * src_step);
  float16x8_t src_data_12 = vld1q_f16(src_data + 10 * src_step);
  float16x8_t src_data_13 = vld1q_f16(src_data + 11 * src_step);
  float16x8_t src_data_14 = vld1q_f16(src_data + 12 * src_step);
  float16x8_t src_data_15 = vld1q_f16(src_data + 13 * src_step);
  float16x8_t src_data_16 = vld1q_f16(src_data + 14 * src_step);
  float16x8_t src_data_17 = vld1q_f16(src_data + 15 * src_step);
  float16x8_t src_data_20 = vld1q_f16(src_data + 16 * src_step);
  float16x8_t src_data_21 = vld1q_f16(src_data + 17 * src_step);
  float16x8_t src_data_22 = vld1q_f16(src_data + 18 * src_step);
  float16x8_t src_data_23 = vld1q_f16(src_data + 19 * src_step);
  float16x8_t src_data_24 = vld1q_f16(src_data + 20 * src_step);
  float16x8_t src_data_25 = vld1q_f16(src_data + 21 * src_step);
  float16x8_t src_data_26 = vld1q_f16(src_data + 22 * src_step);
  float16x8_t src_data_27 = vld1q_f16(src_data + 23 * src_step);
  float16x8_t src_data_30 = vld1q_f16(src_data + 24 * src_step);
  float16x8_t src_data_31 = vld1q_f16(src_data + 25 * src_step);
  float16x8_t src_data_32 = vld1q_f16(src_data + 26 * src_step);
  float16x8_t src_data_33 = vld1q_f16(src_data + 27 * src_step);
  float16x8_t src_data_34 = vld1q_f16(src_data + 28 * src_step);
  float16x8_t src_data_35 = vld1q_f16(src_data + 29 * src_step);
  float16x8_t src_data_36 = vld1q_f16(src_data + 30 * src_step);
  float16x8_t src_data_37 = vld1q_f16(src_data + 31 * src_step);
  float16x8_t src_data_40 = vld1q_f16(src_data + 32 * src_step);
  float16x8_t src_data_41 = vld1q_f16(src_data + 33 * src_step);
  float16x8_t src_data_42 = vld1q_f16(src_data + 34 * src_step);
  float16x8_t src_data_43 = vld1q_f16(src_data + 35 * src_step);
  float16x8_t src_data_44 = vld1q_f16(src_data + 36 * src_step);
  float16x8_t src_data_45 = vld1q_f16(src_data + 37 * src_step);
  float16x8_t src_data_46 = vld1q_f16(src_data + 38 * src_step);
  float16x8_t src_data_47 = vld1q_f16(src_data + 39 * src_step);
  float16x8_t src_data_50 = vld1q_f16(src_data + 40 * src_step);
  float16x8_t src_data_51 = vld1q_f16(src_data + 41 * src_step);
  float16x8_t src_data_52 = vld1q_f16(src_data + 42 * src_step);
  float16x8_t src_data_53 = vld1q_f16(src_data + 43 * src_step);
  float16x8_t src_data_54 = vld1q_f16(src_data + 44 * src_step);
  float16x8_t src_data_55 = vld1q_f16(src_data + 45 * src_step);
  float16x8_t src_data_56 = vld1q_f16(src_data + 46 * src_step);
  float16x8_t src_data_57 = vld1q_f16(src_data + 47 * src_step);
  float16x8_t src_data_60 = vld1q_f16(src_data + 48 * src_step);
  float16x8_t src_data_61 = vld1q_f16(src_data + 49 * src_step);
  float16x8_t src_data_62 = vld1q_f16(src_data + 50 * src_step);
  float16x8_t src_data_63 = vld1q_f16(src_data + 51 * src_step);
  float16x8_t src_data_64 = vld1q_f16(src_data + 52 * src_step);
  float16x8_t src_data_65 = vld1q_f16(src_data + 53 * src_step);
  float16x8_t src_data_66 = vld1q_f16(src_data + 54 * src_step);
  float16x8_t src_data_67 = vld1q_f16(src_data + 55 * src_step);
  float16x8_t src_data_70 = vld1q_f16(src_data + 56 * src_step);
  float16x8_t src_data_71 = vld1q_f16(src_data + 57 * src_step);
  float16x8_t src_data_72 = vld1q_f16(src_data + 58 * src_step);
  float16x8_t src_data_73 = vld1q_f16(src_data + 59 * src_step);
  float16x8_t src_data_74 = vld1q_f16(src_data + 60 * src_step);
  float16x8_t src_data_75 = vld1q_f16(src_data + 61 * src_step);
  float16x8_t src_data_76 = vld1q_f16(src_data + 62 * src_step);
  float16x8_t src_data_77 = vld1q_f16(src_data + 63 * src_step);

  float16x8_t d01 = vsubq_f16(src_data_10, src_data_20);
  float16x8_t d02 = vsubq_f16(src_data_11, src_data_21);
  float16x8_t d03 = vsubq_f16(src_data_12, src_data_22);
  float16x8_t d04 = vsubq_f16(src_data_13, src_data_23);
  float16x8_t d05 = vsubq_f16(src_data_14, src_data_24);
  float16x8_t d06 = vsubq_f16(src_data_15, src_data_25);
  float16x8_t d07 = vsubq_f16(src_data_16, src_data_26);
  float16x8_t d08 = vsubq_f16(src_data_17, src_data_27);

  float16x8_t d11 = vsubq_f16(src_data_30, src_data_40);
  float16x8_t d12 = vsubq_f16(src_data_31, src_data_41);
  float16x8_t d13 = vsubq_f16(src_data_32, src_data_42);
  float16x8_t d14 = vsubq_f16(src_data_33, src_data_43);
  float16x8_t d15 = vsubq_f16(src_data_34, src_data_44);
  float16x8_t d16 = vsubq_f16(src_data_35, src_data_45);
  float16x8_t d17 = vsubq_f16(src_data_36, src_data_46);
  float16x8_t d18 = vsubq_f16(src_data_37, src_data_47);

  float16x8_t d21 = vsubq_f16(src_data_50, src_data_60);
  float16x8_t d22 = vsubq_f16(src_data_51, src_data_61);
  float16x8_t d23 = vsubq_f16(src_data_52, src_data_62);
  float16x8_t d24 = vsubq_f16(src_data_53, src_data_63);
  float16x8_t d25 = vsubq_f16(src_data_54, src_data_64);
  float16x8_t d26 = vsubq_f16(src_data_55, src_data_65);
  float16x8_t d27 = vsubq_f16(src_data_56, src_data_66);
  float16x8_t d28 = vsubq_f16(src_data_57, src_data_67);

  float16x8_t d31 = vaddq_f16(src_data_10, src_data_20);
  float16x8_t d32 = vaddq_f16(src_data_11, src_data_21);
  float16x8_t d33 = vaddq_f16(src_data_12, src_data_22);
  float16x8_t d34 = vaddq_f16(src_data_13, src_data_23);
  float16x8_t d35 = vaddq_f16(src_data_14, src_data_24);
  float16x8_t d36 = vaddq_f16(src_data_15, src_data_25);
  float16x8_t d37 = vaddq_f16(src_data_16, src_data_26);
  float16x8_t d38 = vaddq_f16(src_data_17, src_data_27);

  float16x8_t d41 = vaddq_f16(src_data_30, src_data_40);
  float16x8_t d42 = vaddq_f16(src_data_31, src_data_41);
  float16x8_t d43 = vaddq_f16(src_data_32, src_data_42);
  float16x8_t d44 = vaddq_f16(src_data_33, src_data_43);
  float16x8_t d45 = vaddq_f16(src_data_34, src_data_44);
  float16x8_t d46 = vaddq_f16(src_data_35, src_data_45);
  float16x8_t d47 = vaddq_f16(src_data_36, src_data_46);
  float16x8_t d48 = vaddq_f16(src_data_37, src_data_47);

  float16x8_t d51 = vaddq_f16(src_data_50, src_data_60);
  float16x8_t d52 = vaddq_f16(src_data_51, src_data_61);
  float16x8_t d53 = vaddq_f16(src_data_52, src_data_62);
  float16x8_t d54 = vaddq_f16(src_data_53, src_data_63);
  float16x8_t d55 = vaddq_f16(src_data_54, src_data_64);
  float16x8_t d56 = vaddq_f16(src_data_55, src_data_65);
  float16x8_t d57 = vaddq_f16(src_data_56, src_data_66);
  float16x8_t d58 = vaddq_f16(src_data_57, src_data_67);

  float16x8_t t00 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_00, src_data_10), src_data_20), src_data_30), src_data_40),
      src_data_50),
    src_data_60);
  float16x8_t t01 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_01, src_data_11), src_data_21), src_data_31), src_data_41),
      src_data_51),
    src_data_61);
  float16x8_t t02 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_02, src_data_12), src_data_22), src_data_32), src_data_42),
      src_data_52),
    src_data_62);
  float16x8_t t03 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_03, src_data_13), src_data_23), src_data_33), src_data_43),
      src_data_53),
    src_data_63);
  float16x8_t t04 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_04, src_data_14), src_data_24), src_data_34), src_data_44),
      src_data_54),
    src_data_64);
  float16x8_t t05 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_05, src_data_15), src_data_25), src_data_35), src_data_45),
      src_data_55),
    src_data_65);
  float16x8_t t06 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_06, src_data_16), src_data_26), src_data_36), src_data_46),
      src_data_56),
    src_data_66);
  float16x8_t t07 = vaddq_f16(
    vaddq_f16(
      vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(src_data_07, src_data_17), src_data_27), src_data_37), src_data_47),
      src_data_57),
    src_data_67);

  float16x8_t t10 = vaddq_f16(vaddq_f16(vmulq_n_f16(d01, 0.5), d11), vmulq_n_f16(d21, 1.5));
  float16x8_t t11 = vaddq_f16(vaddq_f16(vmulq_n_f16(d02, 0.5), d12), vmulq_n_f16(d22, 1.5));
  float16x8_t t12 = vaddq_f16(vaddq_f16(vmulq_n_f16(d03, 0.5), d13), vmulq_n_f16(d23, 1.5));
  float16x8_t t13 = vaddq_f16(vaddq_f16(vmulq_n_f16(d04, 0.5), d14), vmulq_n_f16(d24, 1.5));
  float16x8_t t14 = vaddq_f16(vaddq_f16(vmulq_n_f16(d05, 0.5), d15), vmulq_n_f16(d25, 1.5));
  float16x8_t t15 = vaddq_f16(vaddq_f16(vmulq_n_f16(d06, 0.5), d16), vmulq_n_f16(d26, 1.5));
  float16x8_t t16 = vaddq_f16(vaddq_f16(vmulq_n_f16(d07, 0.5), d17), vmulq_n_f16(d27, 1.5));
  float16x8_t t17 = vaddq_f16(vaddq_f16(vmulq_n_f16(d08, 0.5), d18), vmulq_n_f16(d28, 1.5));

  float16x8_t t20 = vaddq_f16(vaddq_f16(vmulq_n_f16(d31, 0.25), d41), vmulq_n_f16(d51, 2.25));
  float16x8_t t21 = vaddq_f16(vaddq_f16(vmulq_n_f16(d32, 0.25), d42), vmulq_n_f16(d52, 2.25));
  float16x8_t t22 = vaddq_f16(vaddq_f16(vmulq_n_f16(d33, 0.25), d43), vmulq_n_f16(d53, 2.25));
  float16x8_t t23 = vaddq_f16(vaddq_f16(vmulq_n_f16(d34, 0.25), d44), vmulq_n_f16(d54, 2.25));
  float16x8_t t24 = vaddq_f16(vaddq_f16(vmulq_n_f16(d35, 0.25), d45), vmulq_n_f16(d55, 2.25));
  float16x8_t t25 = vaddq_f16(vaddq_f16(vmulq_n_f16(d36, 0.25), d46), vmulq_n_f16(d56, 2.25));
  float16x8_t t26 = vaddq_f16(vaddq_f16(vmulq_n_f16(d37, 0.25), d47), vmulq_n_f16(d57, 2.25));
  float16x8_t t27 = vaddq_f16(vaddq_f16(vmulq_n_f16(d38, 0.25), d48), vmulq_n_f16(d58, 2.25));

  float16x8_t t30 = vaddq_f16(vaddq_f16(vmulq_n_f16(d01, 0.125), d11), vmulq_n_f16(d21, 3.375));
  float16x8_t t31 = vaddq_f16(vaddq_f16(vmulq_n_f16(d02, 0.125), d12), vmulq_n_f16(d22, 3.375));
  float16x8_t t32 = vaddq_f16(vaddq_f16(vmulq_n_f16(d03, 0.125), d13), vmulq_n_f16(d23, 3.375));
  float16x8_t t33 = vaddq_f16(vaddq_f16(vmulq_n_f16(d04, 0.125), d14), vmulq_n_f16(d24, 3.375));
  float16x8_t t34 = vaddq_f16(vaddq_f16(vmulq_n_f16(d05, 0.125), d15), vmulq_n_f16(d25, 3.375));
  float16x8_t t35 = vaddq_f16(vaddq_f16(vmulq_n_f16(d06, 0.125), d16), vmulq_n_f16(d26, 3.375));
  float16x8_t t36 = vaddq_f16(vaddq_f16(vmulq_n_f16(d07, 0.125), d17), vmulq_n_f16(d27, 3.375));
  float16x8_t t37 = vaddq_f16(vaddq_f16(vmulq_n_f16(d08, 0.125), d18), vmulq_n_f16(d28, 3.375));

  float16x8_t t40 = vaddq_f16(vaddq_f16(vmulq_n_f16(d31, 0.0625), d41), vmulq_n_f16(d51, 5.0625));
  float16x8_t t41 = vaddq_f16(vaddq_f16(vmulq_n_f16(d32, 0.0625), d42), vmulq_n_f16(d52, 5.0625));
  float16x8_t t42 = vaddq_f16(vaddq_f16(vmulq_n_f16(d33, 0.0625), d43), vmulq_n_f16(d53, 5.0625));
  float16x8_t t43 = vaddq_f16(vaddq_f16(vmulq_n_f16(d34, 0.0625), d44), vmulq_n_f16(d54, 5.0625));
  float16x8_t t44 = vaddq_f16(vaddq_f16(vmulq_n_f16(d35, 0.0625), d45), vmulq_n_f16(d55, 5.0625));
  float16x8_t t45 = vaddq_f16(vaddq_f16(vmulq_n_f16(d36, 0.0625), d46), vmulq_n_f16(d56, 5.0625));
  float16x8_t t46 = vaddq_f16(vaddq_f16(vmulq_n_f16(d37, 0.0625), d47), vmulq_n_f16(d57, 5.0625));
  float16x8_t t47 = vaddq_f16(vaddq_f16(vmulq_n_f16(d38, 0.0625), d48), vmulq_n_f16(d58, 5.0625));

  float16x8_t t50 = vaddq_f16(vaddq_f16(vmulq_n_f16(d01, 0.03125), d11), vmulq_n_f16(d21, 7.59375));
  float16x8_t t51 = vaddq_f16(vaddq_f16(vmulq_n_f16(d02, 0.03125), d12), vmulq_n_f16(d22, 7.59375));
  float16x8_t t52 = vaddq_f16(vaddq_f16(vmulq_n_f16(d03, 0.03125), d13), vmulq_n_f16(d23, 7.59375));
  float16x8_t t53 = vaddq_f16(vaddq_f16(vmulq_n_f16(d04, 0.03125), d14), vmulq_n_f16(d24, 7.59375));
  float16x8_t t54 = vaddq_f16(vaddq_f16(vmulq_n_f16(d05, 0.03125), d15), vmulq_n_f16(d25, 7.59375));
  float16x8_t t55 = vaddq_f16(vaddq_f16(vmulq_n_f16(d06, 0.03125), d16), vmulq_n_f16(d26, 7.59375));
  float16x8_t t56 = vaddq_f16(vaddq_f16(vmulq_n_f16(d07, 0.03125), d17), vmulq_n_f16(d27, 7.59375));
  float16x8_t t57 = vaddq_f16(vaddq_f16(vmulq_n_f16(d08, 0.03125), d18), vmulq_n_f16(d28, 7.59375));

  float16x8_t t60 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d31, 0.015625), d41), vmulq_n_f16(d51, 11.390625)), src_data_70);
  float16x8_t t61 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d32, 0.015625), d42), vmulq_n_f16(d52, 11.390625)), src_data_71);
  float16x8_t t62 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d33, 0.015625), d43), vmulq_n_f16(d53, 11.390625)), src_data_72);
  float16x8_t t63 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d34, 0.015625), d44), vmulq_n_f16(d54, 11.390625)), src_data_73);
  float16x8_t t64 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d35, 0.015625), d45), vmulq_n_f16(d55, 11.390625)), src_data_74);
  float16x8_t t65 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d36, 0.015625), d46), vmulq_n_f16(d56, 11.390625)), src_data_75);
  float16x8_t t66 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d37, 0.015625), d47), vmulq_n_f16(d57, 11.390625)), src_data_76);
  float16x8_t t67 =
    vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(d38, 0.015625), d48), vmulq_n_f16(d58, 11.390625)), src_data_77);

  float16x8_t s11 = vsubq_f16(t01, t02);
  float16x8_t s12 = vsubq_f16(t11, t12);
  float16x8_t s13 = vsubq_f16(t21, t22);
  float16x8_t s14 = vsubq_f16(t31, t32);
  float16x8_t s15 = vsubq_f16(t41, t42);
  float16x8_t s16 = vsubq_f16(t51, t52);
  float16x8_t s17 = vsubq_f16(t61, t62);

  float16x8_t s21 = vsubq_f16(t03, t04);
  float16x8_t s22 = vsubq_f16(t13, t14);
  float16x8_t s23 = vsubq_f16(t23, t24);
  float16x8_t s24 = vsubq_f16(t33, t34);
  float16x8_t s25 = vsubq_f16(t43, t44);
  float16x8_t s26 = vsubq_f16(t53, t54);
  float16x8_t s27 = vsubq_f16(t63, t64);

  float16x8_t s31 = vsubq_f16(t05, t06);
  float16x8_t s32 = vsubq_f16(t15, t16);
  float16x8_t s33 = vsubq_f16(t25, t26);
  float16x8_t s34 = vsubq_f16(t35, t36);
  float16x8_t s35 = vsubq_f16(t45, t46);
  float16x8_t s36 = vsubq_f16(t55, t56);
  float16x8_t s37 = vsubq_f16(t65, t66);

  float16x8_t s41 = vaddq_f16(t01, t02);
  float16x8_t s42 = vaddq_f16(t11, t12);
  float16x8_t s43 = vaddq_f16(t21, t22);
  float16x8_t s44 = vaddq_f16(t31, t32);
  float16x8_t s45 = vaddq_f16(t41, t42);
  float16x8_t s46 = vaddq_f16(t51, t52);
  float16x8_t s47 = vaddq_f16(t61, t62);

  float16x8_t s51 = vaddq_f16(t03, t04);
  float16x8_t s52 = vaddq_f16(t13, t14);
  float16x8_t s53 = vaddq_f16(t23, t24);
  float16x8_t s54 = vaddq_f16(t33, t34);
  float16x8_t s55 = vaddq_f16(t43, t44);
  float16x8_t s56 = vaddq_f16(t53, t54);
  float16x8_t s57 = vaddq_f16(t63, t64);

  float16x8_t s61 = vaddq_f16(t05, t06);
  float16x8_t s62 = vaddq_f16(t15, t16);
  float16x8_t s63 = vaddq_f16(t25, t26);
  float16x8_t s64 = vaddq_f16(t35, t36);
  float16x8_t s65 = vaddq_f16(t45, t46);
  float16x8_t s66 = vaddq_f16(t55, t56);
  float16x8_t s67 = vaddq_f16(t65, t66);

  float16x8_t m00 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t00, t01), t02), t03), t04), t05), t06);
  float16x8_t m01 = vaddq_f16(vaddq_f16(vmulq_n_f16(s11, 0.5), s21), vmulq_n_f16(s31, 1.5));
  float16x8_t m02 = vaddq_f16(vaddq_f16(vmulq_n_f16(s41, 0.25), s51), vmulq_n_f16(s61, 2.25));
  float16x8_t m03 = vaddq_f16(vaddq_f16(vmulq_n_f16(s11, 0.125), s21), vmulq_n_f16(s31, 3.375));
  float16x8_t m04 = vaddq_f16(vaddq_f16(vmulq_n_f16(s41, 0.0625), s51), vmulq_n_f16(s61, 5.0625));
  float16x8_t m05 = vaddq_f16(vaddq_f16(vmulq_n_f16(s11, 0.03125), s21), vmulq_n_f16(s31, 7.59375));
  float16x8_t m06 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s41, 0.015625), s51), vmulq_n_f16(s61, 11.390625)), t07);

  float16x8_t m10 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t10, t11), t12), t13), t14), t15), t16);
  float16x8_t m11 = vaddq_f16(vaddq_f16(vmulq_n_f16(s12, 0.5), s22), vmulq_n_f16(s32, 1.5));
  float16x8_t m12 = vaddq_f16(vaddq_f16(vmulq_n_f16(s42, 0.25), s52), vmulq_n_f16(s62, 2.25));
  float16x8_t m13 = vaddq_f16(vaddq_f16(vmulq_n_f16(s12, 0.125), s22), vmulq_n_f16(s32, 3.375));
  float16x8_t m14 = vaddq_f16(vaddq_f16(vmulq_n_f16(s42, 0.0625), s52), vmulq_n_f16(s62, 5.0625));
  float16x8_t m15 = vaddq_f16(vaddq_f16(vmulq_n_f16(s12, 0.03125), s22), vmulq_n_f16(s32, 7.59375));
  float16x8_t m16 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s42, 0.015625), s52), vmulq_n_f16(s62, 11.390625)), t17);

  float16x8_t m20 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t20, t21), t22), t23), t24), t25), t26);
  float16x8_t m21 = vaddq_f16(vaddq_f16(vmulq_n_f16(s13, 0.5), s23), vmulq_n_f16(s33, 1.5));
  float16x8_t m22 = vaddq_f16(vaddq_f16(vmulq_n_f16(s43, 0.25), s53), vmulq_n_f16(s63, 2.25));
  float16x8_t m23 = vaddq_f16(vaddq_f16(vmulq_n_f16(s13, 0.125), s23), vmulq_n_f16(s33, 3.375));
  float16x8_t m24 = vaddq_f16(vaddq_f16(vmulq_n_f16(s43, 0.0625), s53), vmulq_n_f16(s63, 5.0625));
  float16x8_t m25 = vaddq_f16(vaddq_f16(vmulq_n_f16(s13, 0.03125), s23), vmulq_n_f16(s33, 7.59375));
  float16x8_t m26 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s43, 0.015625), s53), vmulq_n_f16(s63, 11.390625)), t27);

  float16x8_t m30 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t30, t31), t32), t33), t34), t35), t36);
  float16x8_t m31 = vaddq_f16(vaddq_f16(vmulq_n_f16(s14, 0.5), s24), vmulq_n_f16(s34, 1.5));
  float16x8_t m32 = vaddq_f16(vaddq_f16(vmulq_n_f16(s44, 0.25), s54), vmulq_n_f16(s64, 2.25));
  float16x8_t m33 = vaddq_f16(vaddq_f16(vmulq_n_f16(s14, 0.125), s24), vmulq_n_f16(s34, 3.375));
  float16x8_t m34 = vaddq_f16(vaddq_f16(vmulq_n_f16(s44, 0.0625), s54), vmulq_n_f16(s64, 5.0625));
  float16x8_t m35 = vaddq_f16(vaddq_f16(vmulq_n_f16(s14, 0.03125), s24), vmulq_n_f16(s34, 7.59375));
  float16x8_t m36 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s44, 0.015625), s54), vmulq_n_f16(s64, 11.390625)), t37);

  float16x8_t m40 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t40, t41), t42), t43), t44), t45), t46);
  float16x8_t m41 = vaddq_f16(vaddq_f16(vmulq_n_f16(s15, 0.5), s25), vmulq_n_f16(s35, 1.5));
  float16x8_t m42 = vaddq_f16(vaddq_f16(vmulq_n_f16(s45, 0.25), s55), vmulq_n_f16(s65, 2.25));
  float16x8_t m43 = vaddq_f16(vaddq_f16(vmulq_n_f16(s15, 0.125), s25), vmulq_n_f16(s35, 3.375));
  float16x8_t m44 = vaddq_f16(vaddq_f16(vmulq_n_f16(s45, 0.0625), s55), vmulq_n_f16(s65, 5.0625));
  float16x8_t m45 = vaddq_f16(vaddq_f16(vmulq_n_f16(s15, 0.03125), s25), vmulq_n_f16(s35, 7.59375));
  float16x8_t m46 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s45, 0.015625), s55), vmulq_n_f16(s65, 11.390625)), t47);

  float16x8_t m50 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t50, t51), t52), t53), t54), t55), t56);
  float16x8_t m51 = vaddq_f16(vaddq_f16(vmulq_n_f16(s16, 0.5), s26), vmulq_n_f16(s36, 1.5));
  float16x8_t m52 = vaddq_f16(vaddq_f16(vmulq_n_f16(s46, 0.25), s56), vmulq_n_f16(s66, 2.25));
  float16x8_t m53 = vaddq_f16(vaddq_f16(vmulq_n_f16(s16, 0.125), s26), vmulq_n_f16(s36, 3.375));
  float16x8_t m54 = vaddq_f16(vaddq_f16(vmulq_n_f16(s46, 0.0625), s56), vmulq_n_f16(s66, 5.0625));
  float16x8_t m55 = vaddq_f16(vaddq_f16(vmulq_n_f16(s16, 0.03125), s26), vmulq_n_f16(s36, 7.59375));
  float16x8_t m56 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s46, 0.015625), s56), vmulq_n_f16(s66, 11.390625)), t57);

  float16x8_t m60 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t60, t61), t62), t63), t64), t65), t66);
  float16x8_t m61 = vaddq_f16(vaddq_f16(vmulq_n_f16(s17, 0.5), s27), vmulq_n_f16(s37, 1.5));
  float16x8_t m62 = vaddq_f16(vaddq_f16(vmulq_n_f16(s47, 0.25), s57), vmulq_n_f16(s67, 2.25));
  float16x8_t m63 = vaddq_f16(vaddq_f16(vmulq_n_f16(s17, 0.125), s27), vmulq_n_f16(s37, 3.375));
  float16x8_t m64 = vaddq_f16(vaddq_f16(vmulq_n_f16(s47, 0.0625), s57), vmulq_n_f16(s67, 5.0625));
  float16x8_t m65 = vaddq_f16(vaddq_f16(vmulq_n_f16(s17, 0.03125), s27), vmulq_n_f16(s37, 7.59375));
  float16x8_t m66 = vaddq_f16(vaddq_f16(vaddq_f16(vmulq_n_f16(s47, 0.015625), s57), vmulq_n_f16(s67, 11.390625)), t67);

  float16x8_t bias_ptr = vld1q_f16(bias_data);
  vst1q_f16(dst_data, vaddq_f16(m00, bias_ptr));
  vst1q_f16(dst_data + C8NUM, vaddq_f16(m01, bias_ptr));
  vst1q_f16(dst_data + 2 * C8NUM, vaddq_f16(m02, bias_ptr));
  vst1q_f16(dst_data + 3 * C8NUM, vaddq_f16(m03, bias_ptr));
  vst1q_f16(dst_data + 4 * C8NUM, vaddq_f16(m04, bias_ptr));
  vst1q_f16(dst_data + 5 * C8NUM, vaddq_f16(m05, bias_ptr));
  vst1q_f16(dst_data + 6 * C8NUM, vaddq_f16(m06, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM, vaddq_f16(m10, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM + C8NUM, vaddq_f16(m11, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM + 2 * C8NUM, vaddq_f16(m12, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM + 3 * C8NUM, vaddq_f16(m13, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM + 4 * C8NUM, vaddq_f16(m14, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM + 5 * C8NUM, vaddq_f16(m15, bias_ptr));
  vst1q_f16(dst_data + dst_step * C8NUM + 6 * C8NUM, vaddq_f16(m16, bias_ptr));
  vst1q_f16(dst_data + 2 * dst_step * C8NUM, vaddq_f16(m20, bias_ptr));
  vst1q_f16(dst_data + 2 * dst_step * C8NUM + C8NUM, vaddq_f16(m21, bias_ptr));
  vst1q_f16(dst_data + 2 * dst_step * C8NUM + 2 * C8NUM, vaddq_f16(m22, bias_ptr));
  vst1q_f16(dst_data + 2 * dst_step * C8NUM + 3 * C8NUM, vaddq_f16(m23, bias_ptr));
  vst1q_f16(dst_data + 2 * dst_step * C8NUM + 4 * C8NUM, vaddq_f16(m24, bias_ptr));
  vst1q_f16(dst_data + 2 * dst_step * C8NUM + 5 * C8NUM, vaddq_f16(m25, bias_ptr));
  vst1q_f16(dst_data + 2 * dst_step * C8NUM + 6 * C8NUM, vaddq_f16(m26, bias_ptr));
  vst1q_f16(dst_data + 3 * dst_step * C8NUM, vaddq_f16(m30, bias_ptr));
  vst1q_f16(dst_data + 3 * dst_step * C8NUM + C8NUM, vaddq_f16(m31, bias_ptr));
  vst1q_f16(dst_data + 3 * dst_step * C8NUM + 2 * C8NUM, vaddq_f16(m32, bias_ptr));
  vst1q_f16(dst_data + 3 * dst_step * C8NUM + 3 * C8NUM, vaddq_f16(m33, bias_ptr));
  vst1q_f16(dst_data + 3 * dst_step * C8NUM + 4 * C8NUM, vaddq_f16(m34, bias_ptr));
  vst1q_f16(dst_data + 3 * dst_step * C8NUM + 5 * C8NUM, vaddq_f16(m35, bias_ptr));
  vst1q_f16(dst_data + 3 * dst_step * C8NUM + 6 * C8NUM, vaddq_f16(m36, bias_ptr));
  vst1q_f16(dst_data + 4 * dst_step * C8NUM, vaddq_f16(m40, bias_ptr));
  vst1q_f16(dst_data + 4 * dst_step * C8NUM + C8NUM, vaddq_f16(m41, bias_ptr));
  vst1q_f16(dst_data + 4 * dst_step * C8NUM + 2 * C8NUM, vaddq_f16(m42, bias_ptr));
  vst1q_f16(dst_data + 4 * dst_step * C8NUM + 3 * C8NUM, vaddq_f16(m43, bias_ptr));
  vst1q_f16(dst_data + 4 * dst_step * C8NUM + 4 * C8NUM, vaddq_f16(m44, bias_ptr));
  vst1q_f16(dst_data + 4 * dst_step * C8NUM + 5 * C8NUM, vaddq_f16(m45, bias_ptr));
  vst1q_f16(dst_data + 4 * dst_step * C8NUM + 6 * C8NUM, vaddq_f16(m46, bias_ptr));
  vst1q_f16(dst_data + 5 * dst_step * C8NUM, vaddq_f16(m50, bias_ptr));
  vst1q_f16(dst_data + 5 * dst_step * C8NUM + C8NUM, vaddq_f16(m51, bias_ptr));
  vst1q_f16(dst_data + 5 * dst_step * C8NUM + 2 * C8NUM, vaddq_f16(m52, bias_ptr));
  vst1q_f16(dst_data + 5 * dst_step * C8NUM + 3 * C8NUM, vaddq_f16(m53, bias_ptr));
  vst1q_f16(dst_data + 5 * dst_step * C8NUM + 4 * C8NUM, vaddq_f16(m54, bias_ptr));
  vst1q_f16(dst_data + 5 * dst_step * C8NUM + 5 * C8NUM, vaddq_f16(m55, bias_ptr));
  vst1q_f16(dst_data + 5 * dst_step * C8NUM + 6 * C8NUM, vaddq_f16(m56, bias_ptr));
  vst1q_f16(dst_data + 6 * dst_step * C8NUM, vaddq_f16(m60, bias_ptr));
  vst1q_f16(dst_data + 6 * dst_step * C8NUM + C8NUM, vaddq_f16(m61, bias_ptr));
  vst1q_f16(dst_data + 6 * dst_step * C8NUM + 2 * C8NUM, vaddq_f16(m62, bias_ptr));
  vst1q_f16(dst_data + 6 * dst_step * C8NUM + 3 * C8NUM, vaddq_f16(m63, bias_ptr));
  vst1q_f16(dst_data + 6 * dst_step * C8NUM + 4 * C8NUM, vaddq_f16(m64, bias_ptr));
  vst1q_f16(dst_data + 6 * dst_step * C8NUM + 5 * C8NUM, vaddq_f16(m65, bias_ptr));
  vst1q_f16(dst_data + 6 * dst_step * C8NUM + 6 * C8NUM, vaddq_f16(m66, bias_ptr));
#else
  for (int i = 0; i < C8NUM; i++) {
    float16_t src_data_00 = src_data[i];
    float16_t src_data_01 = src_data[i + src_step];
    float16_t src_data_02 = src_data[i + 2 * src_step];
    float16_t src_data_03 = src_data[i + 3 * src_step];
    float16_t src_data_04 = src_data[i + 4 * src_step];
    float16_t src_data_05 = src_data[i + 5 * src_step];
    float16_t src_data_06 = src_data[i + 6 * src_step];
    float16_t src_data_07 = src_data[i + 7 * src_step];
    float16_t src_data_10 = src_data[i + 8 * src_step];
    float16_t src_data_11 = src_data[i + 9 * src_step];
    float16_t src_data_12 = src_data[i + 10 * src_step];
    float16_t src_data_13 = src_data[i + 11 * src_step];
    float16_t src_data_14 = src_data[i + 12 * src_step];
    float16_t src_data_15 = src_data[i + 13 * src_step];
    float16_t src_data_16 = src_data[i + 14 * src_step];
    float16_t src_data_17 = src_data[i + 15 * src_step];
    float16_t src_data_20 = src_data[i + 16 * src_step];
    float16_t src_data_21 = src_data[i + 17 * src_step];
    float16_t src_data_22 = src_data[i + 18 * src_step];
    float16_t src_data_23 = src_data[i + 19 * src_step];
    float16_t src_data_24 = src_data[i + 20 * src_step];
    float16_t src_data_25 = src_data[i + 21 * src_step];
    float16_t src_data_26 = src_data[i + 22 * src_step];
    float16_t src_data_27 = src_data[i + 23 * src_step];
    float16_t src_data_30 = src_data[i + 24 * src_step];
    float16_t src_data_31 = src_data[i + 25 * src_step];
    float16_t src_data_32 = src_data[i + 26 * src_step];
    float16_t src_data_33 = src_data[i + 27 * src_step];
    float16_t src_data_34 = src_data[i + 28 * src_step];
    float16_t src_data_35 = src_data[i + 29 * src_step];
    float16_t src_data_36 = src_data[i + 30 * src_step];
    float16_t src_data_37 = src_data[i + 31 * src_step];
    float16_t src_data_40 = src_data[i + 32 * src_step];
    float16_t src_data_41 = src_data[i + 33 * src_step];
    float16_t src_data_42 = src_data[i + 34 * src_step];
    float16_t src_data_43 = src_data[i + 35 * src_step];
    float16_t src_data_44 = src_data[i + 36 * src_step];
    float16_t src_data_45 = src_data[i + 37 * src_step];
    float16_t src_data_46 = src_data[i + 38 * src_step];
    float16_t src_data_47 = src_data[i + 39 * src_step];
    float16_t src_data_50 = src_data[i + 40 * src_step];
    float16_t src_data_51 = src_data[i + 41 * src_step];
    float16_t src_data_52 = src_data[i + 42 * src_step];
    float16_t src_data_53 = src_data[i + 43 * src_step];
    float16_t src_data_54 = src_data[i + 44 * src_step];
    float16_t src_data_55 = src_data[i + 45 * src_step];
    float16_t src_data_56 = src_data[i + 46 * src_step];
    float16_t src_data_57 = src_data[i + 47 * src_step];
    float16_t src_data_60 = src_data[i + 48 * src_step];
    float16_t src_data_61 = src_data[i + 49 * src_step];
    float16_t src_data_62 = src_data[i + 50 * src_step];
    float16_t src_data_63 = src_data[i + 51 * src_step];
    float16_t src_data_64 = src_data[i + 52 * src_step];
    float16_t src_data_65 = src_data[i + 53 * src_step];
    float16_t src_data_66 = src_data[i + 54 * src_step];
    float16_t src_data_67 = src_data[i + 55 * src_step];
    float16_t src_data_70 = src_data[i + 56 * src_step];
    float16_t src_data_71 = src_data[i + 57 * src_step];
    float16_t src_data_72 = src_data[i + 58 * src_step];
    float16_t src_data_73 = src_data[i + 59 * src_step];
    float16_t src_data_74 = src_data[i + 60 * src_step];
    float16_t src_data_75 = src_data[i + 61 * src_step];
    float16_t src_data_76 = src_data[i + 62 * src_step];
    float16_t src_data_77 = src_data[i + 63 * src_step];

    float16_t d01 = src_data_10 - src_data_20;
    float16_t d02 = src_data_11 - src_data_21;
    float16_t d03 = src_data_12 - src_data_22;
    float16_t d04 = src_data_13 - src_data_23;
    float16_t d05 = src_data_14 - src_data_24;
    float16_t d06 = src_data_15 - src_data_25;
    float16_t d07 = src_data_16 - src_data_26;
    float16_t d08 = src_data_17 - src_data_27;

    float16_t d11 = src_data_30 - src_data_40;
    float16_t d12 = src_data_31 - src_data_41;
    float16_t d13 = src_data_32 - src_data_42;
    float16_t d14 = src_data_33 - src_data_43;
    float16_t d15 = src_data_34 - src_data_44;
    float16_t d16 = src_data_35 - src_data_45;
    float16_t d17 = src_data_36 - src_data_46;
    float16_t d18 = src_data_37 - src_data_47;

    float16_t d21 = src_data_50 - src_data_60;
    float16_t d22 = src_data_51 - src_data_61;
    float16_t d23 = src_data_52 - src_data_62;
    float16_t d24 = src_data_53 - src_data_63;
    float16_t d25 = src_data_54 - src_data_64;
    float16_t d26 = src_data_55 - src_data_65;
    float16_t d27 = src_data_56 - src_data_66;
    float16_t d28 = src_data_57 - src_data_67;

    float16_t d31 = src_data_10 + src_data_20;
    float16_t d32 = src_data_11 + src_data_21;
    float16_t d33 = src_data_12 + src_data_22;
    float16_t d34 = src_data_13 + src_data_23;
    float16_t d35 = src_data_14 + src_data_24;
    float16_t d36 = src_data_15 + src_data_25;
    float16_t d37 = src_data_16 + src_data_26;
    float16_t d38 = src_data_17 + src_data_27;

    float16_t d41 = src_data_30 + src_data_40;
    float16_t d42 = src_data_31 + src_data_41;
    float16_t d43 = src_data_32 + src_data_42;
    float16_t d44 = src_data_33 + src_data_43;
    float16_t d45 = src_data_34 + src_data_44;
    float16_t d46 = src_data_35 + src_data_45;
    float16_t d47 = src_data_36 + src_data_46;
    float16_t d48 = src_data_37 + src_data_47;

    float16_t d51 = src_data_50 + src_data_60;
    float16_t d52 = src_data_51 + src_data_61;
    float16_t d53 = src_data_52 + src_data_62;
    float16_t d54 = src_data_53 + src_data_63;
    float16_t d55 = src_data_54 + src_data_64;
    float16_t d56 = src_data_55 + src_data_65;
    float16_t d57 = src_data_56 + src_data_66;
    float16_t d58 = src_data_57 + src_data_67;

    float16_t t00 = src_data_00 + src_data_10 + src_data_20 + src_data_30 + src_data_40 + src_data_50 + src_data_60;
    float16_t t01 = src_data_01 + src_data_11 + src_data_21 + src_data_31 + src_data_41 + src_data_51 + src_data_61;
    float16_t t02 = src_data_02 + src_data_12 + src_data_22 + src_data_32 + src_data_42 + src_data_52 + src_data_62;
    float16_t t03 = src_data_03 + src_data_13 + src_data_23 + src_data_33 + src_data_43 + src_data_53 + src_data_63;
    float16_t t04 = src_data_04 + src_data_14 + src_data_24 + src_data_34 + src_data_44 + src_data_54 + src_data_64;
    float16_t t05 = src_data_05 + src_data_15 + src_data_25 + src_data_35 + src_data_45 + src_data_55 + src_data_65;
    float16_t t06 = src_data_06 + src_data_16 + src_data_26 + src_data_36 + src_data_46 + src_data_56 + src_data_66;
    float16_t t07 = src_data_07 + src_data_17 + src_data_27 + src_data_37 + src_data_47 + src_data_57 + src_data_67;

    const float16_t t10 = 0.5f * d01 + d11 + 1.5f * d21;
    const float16_t t11 = 0.5f * d02 + d12 + 1.5f * d22;
    const float16_t t12 = 0.5f * d03 + d13 + 1.5f * d23;
    const float16_t t13 = 0.5f * d04 + d14 + 1.5f * d24;
    const float16_t t14 = 0.5f * d05 + d15 + 1.5f * d25;
    const float16_t t15 = 0.5f * d06 + d16 + 1.5f * d26;
    const float16_t t16 = 0.5f * d07 + d17 + 1.5f * d27;
    const float16_t t17 = 0.5f * d08 + d18 + 1.5f * d28;

    const float16_t t20 = 0.25f * d31 + d41 + 2.25f * d51;
    const float16_t t21 = 0.25f * d32 + d42 + 2.25f * d52;
    const float16_t t22 = 0.25f * d33 + d43 + 2.25f * d53;
    const float16_t t23 = 0.25f * d34 + d44 + 2.25f * d54;
    const float16_t t24 = 0.25f * d35 + d45 + 2.25f * d55;
    const float16_t t25 = 0.25f * d36 + d46 + 2.25f * d56;
    const float16_t t26 = 0.25f * d37 + d47 + 2.25f * d57;
    const float16_t t27 = 0.25f * d38 + d48 + 2.25f * d58;

    const float16_t t30 = 0.125f * d01 + d11 + 3.375f * d21;
    const float16_t t31 = 0.125f * d02 + d12 + 3.375f * d22;
    const float16_t t32 = 0.125f * d03 + d13 + 3.375f * d23;
    const float16_t t33 = 0.125f * d04 + d14 + 3.375f * d24;
    const float16_t t34 = 0.125f * d05 + d15 + 3.375f * d25;
    const float16_t t35 = 0.125f * d06 + d16 + 3.375f * d26;
    const float16_t t36 = 0.125f * d07 + d17 + 3.375f * d27;
    const float16_t t37 = 0.125f * d08 + d18 + 3.375f * d28;

    const float16_t t40 = 0.0625f * d31 + d41 + 5.0625f * d51;
    const float16_t t41 = 0.0625f * d32 + d42 + 5.0625f * d52;
    const float16_t t42 = 0.0625f * d33 + d43 + 5.0625f * d53;
    const float16_t t43 = 0.0625f * d34 + d44 + 5.0625f * d54;
    const float16_t t44 = 0.0625f * d35 + d45 + 5.0625f * d55;
    const float16_t t45 = 0.0625f * d36 + d46 + 5.0625f * d56;
    const float16_t t46 = 0.0625f * d37 + d47 + 5.0625f * d57;
    const float16_t t47 = 0.0625f * d38 + d48 + 5.0625f * d58;

    const float16_t t50 = 0.03125f * d01 + d11 + 7.59375f * d21;
    const float16_t t51 = 0.03125f * d02 + d12 + 7.59375f * d22;
    const float16_t t52 = 0.03125f * d03 + d13 + 7.59375f * d23;
    const float16_t t53 = 0.03125f * d04 + d14 + 7.59375f * d24;
    const float16_t t54 = 0.03125f * d05 + d15 + 7.59375f * d25;
    const float16_t t55 = 0.03125f * d06 + d16 + 7.59375f * d26;
    const float16_t t56 = 0.03125f * d07 + d17 + 7.59375f * d27;
    const float16_t t57 = 0.03125f * d08 + d18 + 7.59375f * d28;

    const float16_t t60 = 0.015625f * d31 + d41 + 11.390625f * d51 + src_data_70;
    const float16_t t61 = 0.015625f * d32 + d42 + 11.390625f * d52 + src_data_71;
    const float16_t t62 = 0.015625f * d33 + d43 + 11.390625f * d53 + src_data_72;
    const float16_t t63 = 0.015625f * d34 + d44 + 11.390625f * d54 + src_data_73;
    const float16_t t64 = 0.015625f * d35 + d45 + 11.390625f * d55 + src_data_74;
    const float16_t t65 = 0.015625f * d36 + d46 + 11.390625f * d56 + src_data_75;
    const float16_t t66 = 0.015625f * d37 + d47 + 11.390625f * d57 + src_data_76;
    const float16_t t67 = 0.015625f * d38 + d48 + 11.390625f * d58 + src_data_77;

    float16_t s11 = t01 - t02;
    float16_t s12 = t11 - t12;
    float16_t s13 = t21 - t22;
    float16_t s14 = t31 - t32;
    float16_t s15 = t41 - t42;
    float16_t s16 = t51 - t52;
    float16_t s17 = t61 - t62;

    float16_t s21 = t03 - t04;
    float16_t s22 = t13 - t14;
    float16_t s23 = t23 - t24;
    float16_t s24 = t33 - t34;
    float16_t s25 = t43 - t44;
    float16_t s26 = t53 - t54;
    float16_t s27 = t63 - t64;

    float16_t s31 = t05 - t06;
    float16_t s32 = t15 - t16;
    float16_t s33 = t25 - t26;
    float16_t s34 = t35 - t36;
    float16_t s35 = t45 - t46;
    float16_t s36 = t55 - t56;
    float16_t s37 = t56 - t66;

    float16_t s41 = t01 + t02;
    float16_t s42 = t11 + t12;
    float16_t s43 = t21 + t22;
    float16_t s44 = t31 + t32;
    float16_t s45 = t41 + t42;
    float16_t s46 = t51 + t52;
    float16_t s47 = t61 + t62;

    float16_t s51 = t03 + t04;
    float16_t s52 = t13 + t14;
    float16_t s53 = t23 + t24;
    float16_t s54 = t33 + t34;
    float16_t s55 = t43 + t44;
    float16_t s56 = t53 + t54;
    float16_t s57 = t63 + t64;

    float16_t s61 = t05 + t06;
    float16_t s62 = t15 + t16;
    float16_t s63 = t25 + t26;
    float16_t s64 = t35 + t36;
    float16_t s65 = t45 + t46;
    float16_t s66 = t55 + t56;
    float16_t s67 = t65 + t66;

    float16_t m00 = t00 + t01 + t02 + t03 + t04 + t05 + t06;
    const float16_t m01 = 0.5f * s11 + s21 + 1.5f * s31;
    const float16_t m02 = 0.25f * s41 + s51 + 2.25f * s61;
    const float16_t m03 = 0.125f * s11 + s21 + 3.375f * s31;
    const float16_t m04 = 0.0625f * s41 + s51 + 5.0625f * s61;
    const float16_t m05 = 0.03125f * s11 + s21 + 7.59375f * s31;
    const float16_t m06 = 0.015625f * s41 + s51 + 11.390625f * s61 + t07;

    float16_t m10 = t10 + t11 + t12 + t13 + t14 + t15 + t16;
    const float16_t m11 = 0.5f * s12 + s22 + 1.5f * s32;
    const float16_t m12 = 0.25f * s42 + s52 + 2.25f * s62;
    const float16_t m13 = 0.125f * s12 + s22 + 3.375f * s32;
    const float16_t m14 = 0.0625f * s42 + s52 + 5.0625f * s62;
    const float16_t m15 = 0.03125f * s12 + s22 + 7.59375f * s32;
    const float16_t m16 = 0.015625f * s42 + s52 + 11.390625f * s62 + t17;

    float16_t m20 = t20 + t21 + t22 + t23 + t24 + t25 + t26;
    const float16_t m21 = 0.5f * s13 + s23 + 1.5f * s33;
    const float16_t m22 = 0.25f * s43 + s53 + 2.25f * s63;
    const float16_t m23 = 0.125f * s13 + s23 + 3.375f * s33;
    const float16_t m24 = 0.0625f * s43 + s53 + 5.0625f * s63;
    const float16_t m25 = 0.03125f * s13 + s23 + 7.59375f * s33;
    const float16_t m26 = 0.015625f * s43 + s53 + 11.390625f * s63 + t27;

    float16_t m30 = t30 + t31 + t32 + t33 + t34 + t35 + t36;
    const float16_t m31 = 0.5f * s14 + s24 + 1.5f * s34;
    const float16_t m32 = 0.25f * s44 + s54 + 2.25f * s64;
    const float16_t m33 = 0.125f * s14 + s24 + 3.375f * s34;
    const float16_t m34 = 0.0625f * s44 + s54 + 5.0625f * s64;
    const float16_t m35 = 0.03125f * s14 + s24 + 7.59375f * s34;
    const float16_t m36 = 0.015625f * s44 + s54 + 11.390625f * s64 + t37;

    float16_t m40 = t40 + t41 + t42 + t43 + t44 + t45 + t46;
    const float16_t m41 = 0.5f * s15 + s25 + 1.5f * s35;
    const float16_t m42 = 0.25f * s45 + s55 + 2.25f * s65;
    const float16_t m43 = 0.125f * s15 + s25 + 3.375f * s35;
    const float16_t m44 = 0.0625f * s45 + s55 + 5.0625f * s65;
    const float16_t m45 = 0.03125f * s15 + s25 + 7.59375f * s35;
    const float16_t m46 = 0.015625f * s45 + s55 + 11.390625f * s65 + t47;

    float16_t m50 = t50 + t51 + t52 + t53 + t54 + t55 + t56;
    const float16_t m51 = 0.5f * s16 + s26 + 1.5f * s36;
    const float16_t m52 = 0.25f * s46 + s56 + 2.25f * s66;
    const float16_t m53 = 0.125f * s16 + s26 + 3.375f * s36;
    const float16_t m54 = 0.0625f * s46 + s56 + 5.0625f * s66;
    const float16_t m55 = 0.03125f * s16 + s26 + 7.59375f * s36;
    const float16_t m56 = 0.015625f * s46 + s56 + 11.390625f * s66 + t57;

    float16_t m60 = t60 + t61 + t62 + t63 + t64 + t65 + t66;
    const float16_t m61 = 0.5f * s17 + s27 + 1.5f * s37;
    const float16_t m62 = 0.25f * s47 + s57 + 2.25f * s67;
    const float16_t m63 = 0.125f * s17 + s27 + 3.375f * s37;
    const float16_t m64 = 0.0625f * s47 + s57 + 5.0625f * s67;
    const float16_t m65 = 0.03125f * s17 + s27 + 7.59375f * s37;
    const float16_t m66 = 0.015625f * s47 + s57 + 11.390625f * s67 + t67;

    (dst_data + i)[0] = m00 + bias_data[i];
    (dst_data + i + C8NUM)[0] = m01 + bias_data[i];
    (dst_data + i + 2 * C8NUM)[0] = m02 + bias_data[i];
    (dst_data + i + 3 * C8NUM)[0] = m03 + bias_data[i];
    (dst_data + i + 4 * C8NUM)[0] = m04 + bias_data[i];
    (dst_data + i + 5 * C8NUM)[0] = m05 + bias_data[i];
    (dst_data + i + 6 * C8NUM)[0] = m06 + bias_data[i];

    (dst_data + i + dst_step * C8NUM)[0] = m10 + bias_data[i];
    (dst_data + i + dst_step * C8NUM + C8NUM)[0] = m11 + bias_data[i];
    (dst_data + i + dst_step * C8NUM + 2 * C8NUM)[0] = m12 + bias_data[i];
    (dst_data + i + dst_step * C8NUM + 3 * C8NUM)[0] = m13 + bias_data[i];
    (dst_data + i + dst_step * C8NUM + 4 * C8NUM)[0] = m14 + bias_data[i];
    (dst_data + i + dst_step * C8NUM + 5 * C8NUM)[0] = m15 + bias_data[i];
    (dst_data + i + dst_step * C8NUM + 6 * C8NUM)[0] = m16 + bias_data[i];

    (dst_data + i + 2 * dst_step * C8NUM)[0] = m20 + bias_data[i];
    (dst_data + i + 2 * dst_step * C8NUM + C8NUM)[0] = m21 + bias_data[i];
    (dst_data + i + 2 * dst_step * C8NUM + 2 * C8NUM)[0] = m22 + bias_data[i];
    (dst_data + i + 2 * dst_step * C8NUM + 3 * C8NUM)[0] = m23 + bias_data[i];
    (dst_data + i + 2 * dst_step * C8NUM + 4 * C8NUM)[0] = m24 + bias_data[i];
    (dst_data + i + 2 * dst_step * C8NUM + 5 * C8NUM)[0] = m25 + bias_data[i];
    (dst_data + i + 2 * dst_step * C8NUM + 6 * C8NUM)[0] = m26 + bias_data[i];

    (dst_data + i + 3 * dst_step * C8NUM)[0] = m30 + bias_data[i];
    (dst_data + i + 3 * dst_step * C8NUM + C8NUM)[0] = m31 + bias_data[i];
    (dst_data + i + 3 * dst_step * C8NUM + 2 * C8NUM)[0] = m32 + bias_data[i];
    (dst_data + i + 3 * dst_step * C8NUM + 3 * C8NUM)[0] = m33 + bias_data[i];
    (dst_data + i + 3 * dst_step * C8NUM + 4 * C8NUM)[0] = m34 + bias_data[i];
    (dst_data + i + 3 * dst_step * C8NUM + 5 * C8NUM)[0] = m35 + bias_data[i];
    (dst_data + i + 3 * dst_step * C8NUM + 6 * C8NUM)[0] = m36 + bias_data[i];

    (dst_data + i + 4 * dst_step * C8NUM)[0] = m40 + bias_data[i];
    (dst_data + i + 4 * dst_step * C8NUM + C8NUM)[0] = m41 + bias_data[i];
    (dst_data + i + 4 * dst_step * C8NUM + 2 * C8NUM)[0] = m42 + bias_data[i];
    (dst_data + i + 4 * dst_step * C8NUM + 3 * C8NUM)[0] = m43 + bias_data[i];
    (dst_data + i + 4 * dst_step * C8NUM + 4 * C8NUM)[0] = m44 + bias_data[i];
    (dst_data + i + 4 * dst_step * C8NUM + 5 * C8NUM)[0] = m45 + bias_data[i];
    (dst_data + i + 4 * dst_step * C8NUM + 6 * C8NUM)[0] = m46 + bias_data[i];

    (dst_data + i + 5 * dst_step * C8NUM)[0] = m50 + bias_data[i];
    (dst_data + i + 5 * dst_step * C8NUM + C8NUM)[0] = m51 + bias_data[i];
    (dst_data + i + 5 * dst_step * C8NUM + 2 * C8NUM)[0] = m52 + bias_data[i];
    (dst_data + i + 5 * dst_step * C8NUM + 3 * C8NUM)[0] = m53 + bias_data[i];
    (dst_data + i + 5 * dst_step * C8NUM + 4 * C8NUM)[0] = m54 + bias_data[i];
    (dst_data + i + 5 * dst_step * C8NUM + 5 * C8NUM)[0] = m55 + bias_data[i];
    (dst_data + i + 5 * dst_step * C8NUM + 6 * C8NUM)[0] = m56 + bias_data[i];

    (dst_data + i + 6 * dst_step * C8NUM)[0] = m60 + bias_data[i];
    (dst_data + i + 6 * dst_step * C8NUM + C8NUM)[0] = m61 + bias_data[i];
    (dst_data + i + 6 * dst_step * C8NUM + 2 * C8NUM)[0] = m62 + bias_data[i];
    (dst_data + i + 6 * dst_step * C8NUM + 3 * C8NUM)[0] = m63 + bias_data[i];
    (dst_data + i + 6 * dst_step * C8NUM + 4 * C8NUM)[0] = m64 + bias_data[i];
    (dst_data + i + 6 * dst_step * C8NUM + 5 * C8NUM)[0] = m65 + bias_data[i];
    (dst_data + i + 6 * dst_step * C8NUM + 6 * C8NUM)[0] = m66 + bias_data[i];
  }
#endif
}

InputTransformUnitFp16Func GetInputTransFuncFp16(int input_unit) {
  if (input_unit == 4) {
    return InputTransform4x4UnitFp16;
  } else if (input_unit == 8) {
    return InputTransform8x8UnitFp16;
  } else {
    printf("Only support 4 or 8 for input unit.");
    return NULL;
  }
}

OutputTransformUnitFp16Func GetOutputTransFuncFp16(int input_unit, int output_unit) {
  if (input_unit == 4 && output_unit == 2) {
    return OutputTransform4x2UnitFp16;
  } else if (input_unit == 4 && output_unit == 3) {
    return OutputTransform4x3UnitFp16;
  } else if (input_unit == 8) {
    return outputTransformUnitFp16[output_unit];
  } else {
    printf(".");
    return NULL;
  }
}
