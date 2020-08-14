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

#include "nnacl/winograd_utils.h"
#include <stdio.h>

#define MIN_UNIT 2
#define MAX_UNIT 8

static OutputTransformUnitFunc outputTransformUnit[] = {
  NULL,  // 0
  NULL,  // 1
  OutputTransform8x2Unit,
  OutputTransform8x3Unit,
  OutputTransform8x4Unit,
  OutputTransform8x5Unit,
  OutputTransform8x6Unit,
  OutputTransform8x7Unit,
};

void InputTransform4x4Unit(const float *src_data, float *dst_data, int src_step, int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src_data_00 = vld1q_f32(src_data + 0 * src_step);
  float32x4_t src_data_01 = vld1q_f32(src_data + 1 * src_step);
  float32x4_t src_data_02 = vld1q_f32(src_data + 2 * src_step);
  float32x4_t src_data_03 = vld1q_f32(src_data + 3 * src_step);
  float32x4_t src_data_10 = vld1q_f32(src_data + 4 * src_step);
  float32x4_t src_data_11 = vld1q_f32(src_data + 5 * src_step);
  float32x4_t src_data_12 = vld1q_f32(src_data + 6 * src_step);
  float32x4_t src_data_13 = vld1q_f32(src_data + 7 * src_step);
  float32x4_t src_data_20 = vld1q_f32(src_data + 8 * src_step);
  float32x4_t src_data_21 = vld1q_f32(src_data + 9 * src_step);
  float32x4_t src_data_22 = vld1q_f32(src_data + 10 * src_step);
  float32x4_t src_data_23 = vld1q_f32(src_data + 11 * src_step);
  float32x4_t src_data_30 = vld1q_f32(src_data + 12 * src_step);
  float32x4_t src_data_31 = vld1q_f32(src_data + 13 * src_step);
  float32x4_t src_data_32 = vld1q_f32(src_data + 14 * src_step);
  float32x4_t src_data_33 = vld1q_f32(src_data + 15 * src_step);

  float32x4_t t00 = vsubq_f32(src_data_00, vmulq_n_f32(src_data_20, 4));
  float32x4_t t01 = vsubq_f32(src_data_01, vmulq_n_f32(src_data_21, 4));
  float32x4_t t02 = vsubq_f32(src_data_02, vmulq_n_f32(src_data_22, 4));
  float32x4_t t03 = vsubq_f32(src_data_03, vmulq_n_f32(src_data_23, 4));

  float32x4_t t10 = vaddq_f32(src_data_10, vmulq_n_f32(src_data_20, 2));
  float32x4_t t11 = vaddq_f32(src_data_11, vmulq_n_f32(src_data_21, 2));
  float32x4_t t12 = vaddq_f32(src_data_12, vmulq_n_f32(src_data_22, 2));
  float32x4_t t13 = vaddq_f32(src_data_13, vmulq_n_f32(src_data_23, 2));

  float32x4_t t20 = vsubq_f32(vmulq_n_f32(src_data_20, 2), src_data_10);
  float32x4_t t21 = vsubq_f32(vmulq_n_f32(src_data_21, 2), src_data_11);
  float32x4_t t22 = vsubq_f32(vmulq_n_f32(src_data_22, 2), src_data_12);
  float32x4_t t23 = vsubq_f32(vmulq_n_f32(src_data_23, 2), src_data_13);

  float32x4_t t30 = vsubq_f32(src_data_30, vmulq_n_f32(src_data_10, 0.25));
  float32x4_t t31 = vsubq_f32(src_data_31, vmulq_n_f32(src_data_11, 0.25));
  float32x4_t t32 = vsubq_f32(src_data_32, vmulq_n_f32(src_data_12, 0.25));
  float32x4_t t33 = vsubq_f32(src_data_33, vmulq_n_f32(src_data_13, 0.25));

  float32x4_t m00 = vsubq_f32(t00, vmulq_n_f32(t02, 4));
  float32x4_t m01 = vaddq_f32(t01, vmulq_n_f32(t02, 2));
  float32x4_t m02 = vsubq_f32(vmulq_n_f32(t02, 2), t01);
  float32x4_t m03 = vsubq_f32(t03, vmulq_n_f32(t01, 0.25));

  float32x4_t m10 = vsubq_f32(t10, vmulq_n_f32(t12, 4));
  float32x4_t m11 = vaddq_f32(t11, vmulq_n_f32(t12, 2));
  float32x4_t m12 = vsubq_f32(vmulq_n_f32(t12, 2), t11);
  float32x4_t m13 = vsubq_f32(t13, vmulq_n_f32(t11, 0.25));

  float32x4_t m20 = vsubq_f32(t20, vmulq_n_f32(t22, 4));
  float32x4_t m21 = vaddq_f32(t21, vmulq_n_f32(t22, 2));
  float32x4_t m22 = vsubq_f32(vmulq_n_f32(t22, 2), t21);
  float32x4_t m23 = vsubq_f32(t23, vmulq_n_f32(t21, 0.25));

  float32x4_t m30 = vsubq_f32(t30, vmulq_n_f32(t32, 4));
  float32x4_t m31 = vaddq_f32(t31, vmulq_n_f32(t32, 2));
  float32x4_t m32 = vsubq_f32(vmulq_n_f32(t32, 2), t31);
  float32x4_t m33 = vsubq_f32(t33, vmulq_n_f32(t31, 0.25));

  vst1q_f32(dst_data + 0 * dst_step, m00);
  vst1q_f32(dst_data + 1 * dst_step, m01);
  vst1q_f32(dst_data + 2 * dst_step, m02);
  vst1q_f32(dst_data + 3 * dst_step, m03);
  vst1q_f32(dst_data + 4 * dst_step, m10);
  vst1q_f32(dst_data + 5 * dst_step, m11);
  vst1q_f32(dst_data + 6 * dst_step, m12);
  vst1q_f32(dst_data + 7 * dst_step, m13);
  vst1q_f32(dst_data + 8 * dst_step, m20);
  vst1q_f32(dst_data + 9 * dst_step, m21);
  vst1q_f32(dst_data + 10 * dst_step, m22);
  vst1q_f32(dst_data + 11 * dst_step, m23);
  vst1q_f32(dst_data + 12 * dst_step, m30);
  vst1q_f32(dst_data + 13 * dst_step, m31);
  vst1q_f32(dst_data + 14 * dst_step, m32);
  vst1q_f32(dst_data + 15 * dst_step, m33);
#else
  for (int i = 0; i < C4NUM; i++) {
    float src_data_00 = src_data[i];
    float src_data_01 = src_data[i + src_step];
    float src_data_02 = src_data[i + 2 * src_step];
    float src_data_03 = src_data[i + 3 * src_step];
    float src_data_10 = src_data[i + 4 * src_step];
    float src_data_11 = src_data[i + 5 * src_step];
    float src_data_12 = src_data[i + 6 * src_step];
    float src_data_13 = src_data[i + 7 * src_step];
    float src_data_20 = src_data[i + 8 * src_step];
    float src_data_21 = src_data[i + 9 * src_step];
    float src_data_22 = src_data[i + 10 * src_step];
    float src_data_23 = src_data[i + 11 * src_step];
    float src_data_30 = src_data[i + 12 * src_step];
    float src_data_31 = src_data[i + 13 * src_step];
    float src_data_32 = src_data[i + 14 * src_step];
    float src_data_33 = src_data[i + 15 * src_step];

    float t00 = src_data_00 - 4 * src_data_20;
    float t01 = src_data_01 - 4 * src_data_21;
    float t02 = src_data_02 - 4 * src_data_22;
    float t03 = src_data_03 - 4 * src_data_23;

    float t10 = src_data_10 + 2 * src_data_20;
    float t11 = src_data_11 + 2 * src_data_21;
    float t12 = src_data_12 + 2 * src_data_22;
    float t13 = src_data_13 + 2 * src_data_23;

    const float t20 = 2 * src_data_20 - src_data_10;
    const float t21 = 2 * src_data_21 - src_data_11;
    const float t22 = 2 * src_data_22 - src_data_12;
    const float t23 = 2 * src_data_23 - src_data_13;

    float t30 = src_data_30 - 0.25f * src_data_10;
    float t31 = src_data_31 - 0.25f * src_data_11;
    float t32 = src_data_32 - 0.25f * src_data_12;
    float t33 = src_data_33 - 0.25f * src_data_13;

    float m00 = t00 - 4 * t02;
    float m01 = t01 + 2 * t02;
    const float m02 = 2 * t02 - t01;
    float m03 = t03 - 0.25f * t01;

    float m10 = t10 - 4 * t12;
    float m11 = t11 + 2 * t12;
    const float m12 = 2 * t12 - t11;
    float m13 = t13 - 0.25f * t11;

    float m20 = t20 - 4 * t22;
    float m21 = t21 + 2 * t22;
    const float m22 = 2 * t22 - t21;
    float m23 = t23 - 0.25f * t21;

    float m30 = t30 - 4 * t32;
    float m31 = t31 + 2 * t32;
    float m32 = 2 * t32 - t31;
    float m33 = t33 - 0.25f * t31;

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

void InputTransform8x8Unit(const float *src_data, float *dst_data, int src_step, int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src_data_00 = vld1q_f32(src_data + 0 * src_step);
  float32x4_t src_data_01 = vld1q_f32(src_data + 1 * src_step);
  float32x4_t src_data_02 = vld1q_f32(src_data + 2 * src_step);
  float32x4_t src_data_03 = vld1q_f32(src_data + 3 * src_step);
  float32x4_t src_data_04 = vld1q_f32(src_data + 4 * src_step);
  float32x4_t src_data_05 = vld1q_f32(src_data + 5 * src_step);
  float32x4_t src_data_06 = vld1q_f32(src_data + 6 * src_step);
  float32x4_t src_data_07 = vld1q_f32(src_data + 7 * src_step);
  float32x4_t src_data_10 = vld1q_f32(src_data + 8 * src_step);
  float32x4_t src_data_11 = vld1q_f32(src_data + 9 * src_step);
  float32x4_t src_data_12 = vld1q_f32(src_data + 10 * src_step);
  float32x4_t src_data_13 = vld1q_f32(src_data + 11 * src_step);
  float32x4_t src_data_14 = vld1q_f32(src_data + 12 * src_step);
  float32x4_t src_data_15 = vld1q_f32(src_data + 13 * src_step);
  float32x4_t src_data_16 = vld1q_f32(src_data + 14 * src_step);
  float32x4_t src_data_17 = vld1q_f32(src_data + 15 * src_step);
  float32x4_t src_data_20 = vld1q_f32(src_data + 16 * src_step);
  float32x4_t src_data_21 = vld1q_f32(src_data + 17 * src_step);
  float32x4_t src_data_22 = vld1q_f32(src_data + 18 * src_step);
  float32x4_t src_data_23 = vld1q_f32(src_data + 19 * src_step);
  float32x4_t src_data_24 = vld1q_f32(src_data + 20 * src_step);
  float32x4_t src_data_25 = vld1q_f32(src_data + 21 * src_step);
  float32x4_t src_data_26 = vld1q_f32(src_data + 22 * src_step);
  float32x4_t src_data_27 = vld1q_f32(src_data + 23 * src_step);
  float32x4_t src_data_30 = vld1q_f32(src_data + 24 * src_step);
  float32x4_t src_data_31 = vld1q_f32(src_data + 25 * src_step);
  float32x4_t src_data_32 = vld1q_f32(src_data + 26 * src_step);
  float32x4_t src_data_33 = vld1q_f32(src_data + 27 * src_step);
  float32x4_t src_data_34 = vld1q_f32(src_data + 28 * src_step);
  float32x4_t src_data_35 = vld1q_f32(src_data + 29 * src_step);
  float32x4_t src_data_36 = vld1q_f32(src_data + 30 * src_step);
  float32x4_t src_data_37 = vld1q_f32(src_data + 31 * src_step);
  float32x4_t src_data_40 = vld1q_f32(src_data + 32 * src_step);
  float32x4_t src_data_41 = vld1q_f32(src_data + 33 * src_step);
  float32x4_t src_data_42 = vld1q_f32(src_data + 34 * src_step);
  float32x4_t src_data_43 = vld1q_f32(src_data + 35 * src_step);
  float32x4_t src_data_44 = vld1q_f32(src_data + 36 * src_step);
  float32x4_t src_data_45 = vld1q_f32(src_data + 37 * src_step);
  float32x4_t src_data_46 = vld1q_f32(src_data + 38 * src_step);
  float32x4_t src_data_47 = vld1q_f32(src_data + 39 * src_step);
  float32x4_t src_data_50 = vld1q_f32(src_data + 40 * src_step);
  float32x4_t src_data_51 = vld1q_f32(src_data + 41 * src_step);
  float32x4_t src_data_52 = vld1q_f32(src_data + 42 * src_step);
  float32x4_t src_data_53 = vld1q_f32(src_data + 43 * src_step);
  float32x4_t src_data_54 = vld1q_f32(src_data + 44 * src_step);
  float32x4_t src_data_55 = vld1q_f32(src_data + 45 * src_step);
  float32x4_t src_data_56 = vld1q_f32(src_data + 46 * src_step);
  float32x4_t src_data_57 = vld1q_f32(src_data + 47 * src_step);
  float32x4_t src_data_60 = vld1q_f32(src_data + 48 * src_step);
  float32x4_t src_data_61 = vld1q_f32(src_data + 49 * src_step);
  float32x4_t src_data_62 = vld1q_f32(src_data + 50 * src_step);
  float32x4_t src_data_63 = vld1q_f32(src_data + 51 * src_step);
  float32x4_t src_data_64 = vld1q_f32(src_data + 52 * src_step);
  float32x4_t src_data_65 = vld1q_f32(src_data + 53 * src_step);
  float32x4_t src_data_66 = vld1q_f32(src_data + 54 * src_step);
  float32x4_t src_data_67 = vld1q_f32(src_data + 55 * src_step);
  float32x4_t src_data_70 = vld1q_f32(src_data + 56 * src_step);
  float32x4_t src_data_71 = vld1q_f32(src_data + 57 * src_step);
  float32x4_t src_data_72 = vld1q_f32(src_data + 58 * src_step);
  float32x4_t src_data_73 = vld1q_f32(src_data + 59 * src_step);
  float32x4_t src_data_74 = vld1q_f32(src_data + 60 * src_step);
  float32x4_t src_data_75 = vld1q_f32(src_data + 61 * src_step);
  float32x4_t src_data_76 = vld1q_f32(src_data + 62 * src_step);
  float32x4_t src_data_77 = vld1q_f32(src_data + 63 * src_step);

  float32x4_t t00 = vsubq_f32(vaddq_f32(vsubq_f32(src_data_00, vmulq_n_f32(src_data_20, 5.44444444444444444444444445)),
                                        vmulq_n_f32(src_data_40, 6.222222222222)),
                              vmulq_n_f32(src_data_60, 1.7777777777777));
  float32x4_t t01 = vsubq_f32(vaddq_f32(vsubq_f32(src_data_01, vmulq_n_f32(src_data_21, 5.44444444444444444444444445)),
                                        vmulq_n_f32(src_data_41, 6.222222222222)),
                              vmulq_n_f32(src_data_61, 1.7777777777777));
  float32x4_t t02 = vsubq_f32(vaddq_f32(vsubq_f32(src_data_02, vmulq_n_f32(src_data_22, 5.44444444444444444444444445)),
                                        vmulq_n_f32(src_data_42, 6.222222222222)),
                              vmulq_n_f32(src_data_62, 1.7777777777777));
  float32x4_t t03 = vsubq_f32(vaddq_f32(vsubq_f32(src_data_03, vmulq_n_f32(src_data_23, 5.44444444444444444444444445)),
                                        vmulq_n_f32(src_data_43, 6.222222222222)),
                              vmulq_n_f32(src_data_63, 1.7777777777777));
  float32x4_t t04 = vsubq_f32(vaddq_f32(vsubq_f32(src_data_04, vmulq_n_f32(src_data_24, 5.44444444444444444444444445)),
                                        vmulq_n_f32(src_data_44, 6.222222222222)),
                              vmulq_n_f32(src_data_64, 1.7777777777777));
  float32x4_t t05 = vsubq_f32(vaddq_f32(vsubq_f32(src_data_05, vmulq_n_f32(src_data_25, 5.44444444444444444444444445)),
                                        vmulq_n_f32(src_data_45, 6.222222222222)),
                              vmulq_n_f32(src_data_65, 1.7777777777777));
  float32x4_t t06 = vsubq_f32(vaddq_f32(vsubq_f32(src_data_06, vmulq_n_f32(src_data_26, 5.44444444444444444444444445)),
                                        vmulq_n_f32(src_data_46, 6.222222222222)),
                              vmulq_n_f32(src_data_66, 1.7777777777777));
  float32x4_t t07 = vsubq_f32(vaddq_f32(vsubq_f32(src_data_07, vmulq_n_f32(src_data_27, 5.44444444444444444444444445)),
                                        vmulq_n_f32(src_data_47, 6.222222222222)),
                              vmulq_n_f32(src_data_67, 1.7777777777777));

  float32x4_t t10 =
    vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_10, 1.5), vmulq_n_f32(src_data_20, 3)),
                                            vmulq_n_f32(src_data_30, 2.166666666666666667)),
                                  vmulq_n_f32(src_data_40, 4.333333333333)),
                        vmulq_n_f32(src_data_50, 0.66666666666)),
              vmulq_n_f32(src_data_60, 1.333333333333));
  float32x4_t t11 =
    vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_11, 1.5), vmulq_n_f32(src_data_21, 3)),
                                            vmulq_n_f32(src_data_31, 2.166666666666666667)),
                                  vmulq_n_f32(src_data_41, 4.333333333333)),
                        vmulq_n_f32(src_data_51, 0.66666666666)),
              vmulq_n_f32(src_data_61, 1.333333333333));
  float32x4_t t12 =
    vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_12, 1.5), vmulq_n_f32(src_data_22, 3)),
                                            vmulq_n_f32(src_data_32, 2.166666666666666667)),
                                  vmulq_n_f32(src_data_42, 4.333333333333)),
                        vmulq_n_f32(src_data_52, 0.66666666666)),
              vmulq_n_f32(src_data_62, 1.333333333333));
  float32x4_t t13 =
    vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_13, 1.5), vmulq_n_f32(src_data_23, 3)),
                                            vmulq_n_f32(src_data_33, 2.166666666666666667)),
                                  vmulq_n_f32(src_data_43, 4.333333333333)),
                        vmulq_n_f32(src_data_53, 0.66666666666)),
              vmulq_n_f32(src_data_63, 1.333333333333));
  float32x4_t t14 =
    vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_14, 1.5), vmulq_n_f32(src_data_24, 3)),
                                            vmulq_n_f32(src_data_34, 2.166666666666666667)),
                                  vmulq_n_f32(src_data_44, 4.333333333333)),
                        vmulq_n_f32(src_data_54, 0.66666666666)),
              vmulq_n_f32(src_data_64, 1.333333333333));
  float32x4_t t15 =
    vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_15, 1.5), vmulq_n_f32(src_data_25, 3)),
                                            vmulq_n_f32(src_data_35, 2.166666666666666667)),
                                  vmulq_n_f32(src_data_45, 4.333333333333)),
                        vmulq_n_f32(src_data_55, 0.66666666666)),
              vmulq_n_f32(src_data_65, 1.333333333333));
  float32x4_t t16 =
    vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_16, 1.5), vmulq_n_f32(src_data_26, 3)),
                                            vmulq_n_f32(src_data_36, 2.166666666666666667)),
                                  vmulq_n_f32(src_data_46, 4.333333333333)),
                        vmulq_n_f32(src_data_56, 0.66666666666)),
              vmulq_n_f32(src_data_66, 1.333333333333));
  float32x4_t t17 =
    vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_17, 1.5), vmulq_n_f32(src_data_27, 3)),
                                            vmulq_n_f32(src_data_37, 2.166666666666666667)),
                                  vmulq_n_f32(src_data_47, 4.333333333333)),
                        vmulq_n_f32(src_data_57, 0.66666666666)),
              vmulq_n_f32(src_data_67, 1.333333333333));

  float32x4_t t20 =
    vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(src_data_10, -1.5), vmulq_n_f32(src_data_20, 3)),
                                            vmulq_n_f32(src_data_30, 2.166666666666666667)),
                                  vmulq_n_f32(src_data_40, 4.333333333333)),
                        vmulq_n_f32(src_data_50, 0.66666666666)),
              vmulq_n_f32(src_data_60, 1.333333333333));
  float32x4_t t21 =
    vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(src_data_11, -1.5), vmulq_n_f32(src_data_21, 3)),
                                            vmulq_n_f32(src_data_31, 2.166666666666666667)),
                                  vmulq_n_f32(src_data_41, 4.333333333333)),
                        vmulq_n_f32(src_data_51, 0.66666666666)),
              vmulq_n_f32(src_data_61, 1.333333333333));
  float32x4_t t22 =
    vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(src_data_12, -1.5), vmulq_n_f32(src_data_22, 3)),
                                            vmulq_n_f32(src_data_32, 2.166666666666666667)),
                                  vmulq_n_f32(src_data_42, 4.333333333333)),
                        vmulq_n_f32(src_data_52, 0.66666666666)),
              vmulq_n_f32(src_data_62, 1.333333333333));
  float32x4_t t23 =
    vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(src_data_13, -1.5), vmulq_n_f32(src_data_23, 3)),
                                            vmulq_n_f32(src_data_33, 2.166666666666666667)),
                                  vmulq_n_f32(src_data_43, 4.333333333333)),
                        vmulq_n_f32(src_data_53, 0.66666666666)),
              vmulq_n_f32(src_data_63, 1.333333333333));
  float32x4_t t24 =
    vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(src_data_14, -1.5), vmulq_n_f32(src_data_24, 3)),
                                            vmulq_n_f32(src_data_34, 2.166666666666666667)),
                                  vmulq_n_f32(src_data_44, 4.333333333333)),
                        vmulq_n_f32(src_data_54, 0.66666666666)),
              vmulq_n_f32(src_data_64, 1.333333333333));
  float32x4_t t25 =
    vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(src_data_15, -1.5), vmulq_n_f32(src_data_25, 3)),
                                            vmulq_n_f32(src_data_35, 2.166666666666666667)),
                                  vmulq_n_f32(src_data_45, 4.333333333333)),
                        vmulq_n_f32(src_data_55, 0.66666666666)),
              vmulq_n_f32(src_data_65, 1.333333333333));
  float32x4_t t26 =
    vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(src_data_16, -1.5), vmulq_n_f32(src_data_26, 3)),
                                            vmulq_n_f32(src_data_36, 2.166666666666666667)),
                                  vmulq_n_f32(src_data_46, 4.333333333333)),
                        vmulq_n_f32(src_data_56, 0.66666666666)),
              vmulq_n_f32(src_data_66, 1.333333333333));
  float32x4_t t27 =
    vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(src_data_17, -1.5), vmulq_n_f32(src_data_27, 3)),
                                            vmulq_n_f32(src_data_37, 2.166666666666666667)),
                                  vmulq_n_f32(src_data_47, 4.333333333333)),
                        vmulq_n_f32(src_data_57, 0.66666666666)),
              vmulq_n_f32(src_data_67, 1.333333333333));

  float32x4_t t30 = vsubq_f32(vaddq_f32(vmulq_n_f32(vaddq_f32(src_data_30, src_data_40), 1.3333333333333),
                                        vmulq_n_f32(vaddq_f32(src_data_10, src_data_20), -0.3)),
                              vmulq_n_f32(vaddq_f32(src_data_50, src_data_60), 0.53333333333));
  float32x4_t t31 = vsubq_f32(vaddq_f32(vmulq_n_f32(vaddq_f32(src_data_31, src_data_41), 1.3333333333333),
                                        vmulq_n_f32(vaddq_f32(src_data_11, src_data_21), -0.3)),
                              vmulq_n_f32(vaddq_f32(src_data_51, src_data_61), 0.53333333333));
  float32x4_t t32 = vsubq_f32(vaddq_f32(vmulq_n_f32(vaddq_f32(src_data_32, src_data_42), 1.3333333333333),
                                        vmulq_n_f32(vaddq_f32(src_data_12, src_data_22), -0.3)),
                              vmulq_n_f32(vaddq_f32(src_data_52, src_data_62), 0.53333333333));
  float32x4_t t33 = vsubq_f32(vaddq_f32(vmulq_n_f32(vaddq_f32(src_data_33, src_data_43), 1.3333333333333),
                                        vmulq_n_f32(vaddq_f32(src_data_13, src_data_23), -0.3)),
                              vmulq_n_f32(vaddq_f32(src_data_53, src_data_63), 0.53333333333));
  float32x4_t t34 = vsubq_f32(vaddq_f32(vmulq_n_f32(vaddq_f32(src_data_34, src_data_44), 1.3333333333333),
                                        vmulq_n_f32(vaddq_f32(src_data_14, src_data_24), -0.3)),
                              vmulq_n_f32(vaddq_f32(src_data_54, src_data_64), 0.53333333333));
  float32x4_t t35 = vsubq_f32(vaddq_f32(vmulq_n_f32(vaddq_f32(src_data_35, src_data_45), 1.3333333333333),
                                        vmulq_n_f32(vaddq_f32(src_data_15, src_data_25), -0.3)),
                              vmulq_n_f32(vaddq_f32(src_data_55, src_data_65), 0.53333333333));
  float32x4_t t36 = vsubq_f32(vaddq_f32(vmulq_n_f32(vaddq_f32(src_data_36, src_data_46), 1.3333333333333),
                                        vmulq_n_f32(vaddq_f32(src_data_16, src_data_26), -0.3)),
                              vmulq_n_f32(vaddq_f32(src_data_56, src_data_66), 0.53333333333));
  float32x4_t t37 = vsubq_f32(vaddq_f32(vmulq_n_f32(vaddq_f32(src_data_37, src_data_47), 1.3333333333333),
                                        vmulq_n_f32(vaddq_f32(src_data_17, src_data_27), -0.3)),
                              vmulq_n_f32(vaddq_f32(src_data_57, src_data_67), 0.53333333333));

  float32x4_t t40 = vaddq_f32(vaddq_f32(vmulq_n_f32(vsubq_f32(src_data_40, src_data_30), 1.3333333333333),
                                        vmulq_n_f32(vsubq_f32(src_data_10, src_data_20), 0.3)),
                              vmulq_n_f32(vsubq_f32(src_data_50, src_data_60), 0.53333333333));
  float32x4_t t41 = vaddq_f32(vaddq_f32(vmulq_n_f32(vsubq_f32(src_data_41, src_data_31), 1.3333333333333),
                                        vmulq_n_f32(vsubq_f32(src_data_11, src_data_21), 0.3)),
                              vmulq_n_f32(vsubq_f32(src_data_51, src_data_61), 0.53333333333));
  float32x4_t t42 = vaddq_f32(vaddq_f32(vmulq_n_f32(vsubq_f32(src_data_42, src_data_32), 1.3333333333333),
                                        vmulq_n_f32(vsubq_f32(src_data_12, src_data_22), 0.3)),
                              vmulq_n_f32(vsubq_f32(src_data_52, src_data_62), 0.53333333333));
  float32x4_t t43 = vaddq_f32(vaddq_f32(vmulq_n_f32(vsubq_f32(src_data_43, src_data_33), 1.3333333333333),
                                        vmulq_n_f32(vsubq_f32(src_data_13, src_data_23), 0.3)),
                              vmulq_n_f32(vsubq_f32(src_data_53, src_data_63), 0.53333333333));
  float32x4_t t44 = vaddq_f32(vaddq_f32(vmulq_n_f32(vsubq_f32(src_data_44, src_data_34), 1.3333333333333),
                                        vmulq_n_f32(vsubq_f32(src_data_14, src_data_24), 0.3)),
                              vmulq_n_f32(vsubq_f32(src_data_54, src_data_64), 0.53333333333));
  float32x4_t t45 = vaddq_f32(vaddq_f32(vmulq_n_f32(vsubq_f32(src_data_45, src_data_35), 1.3333333333333),
                                        vmulq_n_f32(vsubq_f32(src_data_15, src_data_25), 0.3)),
                              vmulq_n_f32(vsubq_f32(src_data_55, src_data_65), 0.53333333333));
  float32x4_t t46 = vaddq_f32(vaddq_f32(vmulq_n_f32(vsubq_f32(src_data_46, src_data_36), 1.3333333333333),
                                        vmulq_n_f32(vsubq_f32(src_data_16, src_data_26), 0.3)),
                              vmulq_n_f32(vsubq_f32(src_data_56, src_data_66), 0.53333333333));
  float32x4_t t47 = vaddq_f32(vaddq_f32(vmulq_n_f32(vsubq_f32(src_data_47, src_data_37), 1.3333333333333),
                                        vmulq_n_f32(vsubq_f32(src_data_17, src_data_27), 0.3)),
                              vmulq_n_f32(vsubq_f32(src_data_57, src_data_67), 0.53333333333));

  float32x4_t t50 = vaddq_f32(
    vaddq_f32(
      vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_10, 0.03333333), vmulq_n_f32(src_data_20, 0.022222222)),
                          vmulq_n_f32(src_data_30, 0.1666666666)),
                vmulq_n_f32(src_data_40, 0.11111111111)),
      vmulq_n_f32(src_data_50, 0.133333333)),
    vmulq_n_f32(src_data_60, 0.088888888));
  float32x4_t t51 = vaddq_f32(
    vaddq_f32(
      vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_11, 0.03333333), vmulq_n_f32(src_data_21, 0.022222222)),
                          vmulq_n_f32(src_data_31, 0.1666666666)),
                vmulq_n_f32(src_data_41, 0.11111111111)),
      vmulq_n_f32(src_data_51, 0.133333333)),
    vmulq_n_f32(src_data_61, 0.088888888));
  float32x4_t t52 = vaddq_f32(
    vaddq_f32(
      vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_12, 0.03333333), vmulq_n_f32(src_data_22, 0.022222222)),
                          vmulq_n_f32(src_data_32, 0.1666666666)),
                vmulq_n_f32(src_data_42, 0.11111111111)),
      vmulq_n_f32(src_data_52, 0.133333333)),
    vmulq_n_f32(src_data_62, 0.088888888));
  float32x4_t t53 = vaddq_f32(
    vaddq_f32(
      vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_13, 0.03333333), vmulq_n_f32(src_data_23, 0.022222222)),
                          vmulq_n_f32(src_data_33, 0.1666666666)),
                vmulq_n_f32(src_data_43, 0.11111111111)),
      vmulq_n_f32(src_data_53, 0.133333333)),
    vmulq_n_f32(src_data_63, 0.088888888));
  float32x4_t t54 = vaddq_f32(
    vaddq_f32(
      vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_14, 0.03333333), vmulq_n_f32(src_data_24, 0.022222222)),
                          vmulq_n_f32(src_data_34, 0.1666666666)),
                vmulq_n_f32(src_data_44, 0.11111111111)),
      vmulq_n_f32(src_data_54, 0.133333333)),
    vmulq_n_f32(src_data_64, 0.088888888));
  float32x4_t t55 = vaddq_f32(
    vaddq_f32(
      vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_15, 0.03333333), vmulq_n_f32(src_data_25, 0.022222222)),
                          vmulq_n_f32(src_data_35, 0.1666666666)),
                vmulq_n_f32(src_data_45, 0.11111111111)),
      vmulq_n_f32(src_data_55, 0.133333333)),
    vmulq_n_f32(src_data_65, 0.088888888));
  float32x4_t t56 = vaddq_f32(
    vaddq_f32(
      vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_16, 0.03333333), vmulq_n_f32(src_data_26, 0.022222222)),
                          vmulq_n_f32(src_data_36, 0.1666666666)),
                vmulq_n_f32(src_data_46, 0.11111111111)),
      vmulq_n_f32(src_data_56, 0.133333333)),
    vmulq_n_f32(src_data_66, 0.088888888));
  float32x4_t t57 = vaddq_f32(
    vaddq_f32(
      vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_17, 0.03333333), vmulq_n_f32(src_data_27, 0.022222222)),
                          vmulq_n_f32(src_data_37, 0.1666666666)),
                vmulq_n_f32(src_data_47, 0.11111111111)),
      vmulq_n_f32(src_data_57, 0.133333333)),
    vmulq_n_f32(src_data_67, 0.088888888));

  float32x4_t t60 = vaddq_f32(
    vaddq_f32(
      vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(src_data_10, -0.03333333), vmulq_n_f32(src_data_20, 0.022222222)),
                          vmulq_n_f32(src_data_30, 0.1666666666)),
                vmulq_n_f32(src_data_40, 0.11111111111)),
      vmulq_n_f32(src_data_50, -0.133333333)),
    vmulq_n_f32(src_data_60, 0.088888888));
  float32x4_t t61 = vaddq_f32(
    vaddq_f32(
      vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(src_data_11, -0.03333333), vmulq_n_f32(src_data_21, 0.022222222)),
                          vmulq_n_f32(src_data_31, 0.1666666666)),
                vmulq_n_f32(src_data_41, 0.11111111111)),
      vmulq_n_f32(src_data_51, -0.133333333)),
    vmulq_n_f32(src_data_61, 0.088888888));
  float32x4_t t62 = vaddq_f32(
    vaddq_f32(
      vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(src_data_12, -0.03333333), vmulq_n_f32(src_data_22, 0.022222222)),
                          vmulq_n_f32(src_data_32, 0.1666666666)),
                vmulq_n_f32(src_data_42, 0.11111111111)),
      vmulq_n_f32(src_data_52, -0.133333333)),
    vmulq_n_f32(src_data_62, 0.088888888));
  float32x4_t t63 = vaddq_f32(
    vaddq_f32(
      vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(src_data_13, -0.03333333), vmulq_n_f32(src_data_23, 0.022222222)),
                          vmulq_n_f32(src_data_33, 0.1666666666)),
                vmulq_n_f32(src_data_43, 0.11111111111)),
      vmulq_n_f32(src_data_53, -0.133333333)),
    vmulq_n_f32(src_data_63, 0.088888888));
  float32x4_t t64 = vaddq_f32(
    vaddq_f32(
      vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(src_data_14, -0.03333333), vmulq_n_f32(src_data_24, 0.022222222)),
                          vmulq_n_f32(src_data_34, 0.1666666666)),
                vmulq_n_f32(src_data_44, 0.11111111111)),
      vmulq_n_f32(src_data_54, -0.133333333)),
    vmulq_n_f32(src_data_64, 0.088888888));
  float32x4_t t65 = vaddq_f32(
    vaddq_f32(
      vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(src_data_15, -0.03333333), vmulq_n_f32(src_data_25, 0.022222222)),
                          vmulq_n_f32(src_data_35, 0.1666666666)),
                vmulq_n_f32(src_data_45, 0.11111111111)),
      vmulq_n_f32(src_data_55, -0.133333333)),
    vmulq_n_f32(src_data_65, 0.088888888));
  float32x4_t t66 = vaddq_f32(
    vaddq_f32(
      vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(src_data_16, -0.03333333), vmulq_n_f32(src_data_26, 0.022222222)),
                          vmulq_n_f32(src_data_36, 0.1666666666)),
                vmulq_n_f32(src_data_46, 0.11111111111)),
      vmulq_n_f32(src_data_56, -0.133333333)),
    vmulq_n_f32(src_data_66, 0.088888888));
  float32x4_t t67 = vaddq_f32(
    vaddq_f32(
      vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(src_data_17, -0.03333333), vmulq_n_f32(src_data_27, 0.022222222)),
                          vmulq_n_f32(src_data_37, 0.1666666666)),
                vmulq_n_f32(src_data_47, 0.11111111111)),
      vmulq_n_f32(src_data_57, -0.133333333)),
    vmulq_n_f32(src_data_67, 0.088888888));

  float32x4_t t70 = vaddq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_30, 3.0625), vmulq_n_f32(src_data_10, -0.5625)),
                                        vmulq_n_f32(src_data_50, 3.5)),
                              src_data_70);
  float32x4_t t71 = vaddq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_31, 3.0625), vmulq_n_f32(src_data_11, -0.5625)),
                                        vmulq_n_f32(src_data_51, 3.5)),
                              src_data_71);
  float32x4_t t72 = vaddq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_32, 3.0625), vmulq_n_f32(src_data_12, -0.5625)),
                                        vmulq_n_f32(src_data_52, 3.5)),
                              src_data_72);
  float32x4_t t73 = vaddq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_33, 3.0625), vmulq_n_f32(src_data_13, -0.5625)),
                                        vmulq_n_f32(src_data_53, 3.5)),
                              src_data_73);
  float32x4_t t74 = vaddq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_34, 3.0625), vmulq_n_f32(src_data_14, -0.5625)),
                                        vmulq_n_f32(src_data_54, 3.5)),
                              src_data_74);
  float32x4_t t75 = vaddq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_35, 3.0625), vmulq_n_f32(src_data_15, -0.5625)),
                                        vmulq_n_f32(src_data_55, 3.5)),
                              src_data_75);
  float32x4_t t76 = vaddq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_36, 3.0625), vmulq_n_f32(src_data_16, -0.5625)),
                                        vmulq_n_f32(src_data_56, 3.5)),
                              src_data_76);
  float32x4_t t77 = vaddq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(src_data_37, 3.0625), vmulq_n_f32(src_data_17, -0.5625)),
                                        vmulq_n_f32(src_data_57, 3.5)),
                              src_data_77);

  float32x4_t m00 =
    vsubq_f32(vaddq_f32(vsubq_f32(t00, vmulq_n_f32(t02, 5.444444444444444)), vmulq_n_f32(t04, 6.22222222222)),
              vmulq_n_f32(t06, 1.77777777777777777778));
  float32x4_t m01 = vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t01, 1.5), vmulq_n_f32(t02, 3)),
                                                            vmulq_n_f32(t03, 2.16666666666666667)),
                                                  vmulq_n_f32(t04, 4.3333333333)),
                                        vmulq_n_f32(t05, 0.66666666667)),
                              vmulq_n_f32(t06, 1.333333333333));
  float32x4_t m02 = vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(t01, -1.5), vmulq_n_f32(t02, 3)),
                                                            vmulq_n_f32(t03, 2.16666666666666667)),
                                                  vmulq_n_f32(t04, 4.3333333333)),
                                        vmulq_n_f32(t05, 0.66666666667)),
                              vmulq_n_f32(t06, 1.333333333333));
  float32x4_t m03 =
    vaddq_f32(vaddq_f32(vmulq_n_f32(vaddq_f32(t01, t02), -0.3), vmulq_n_f32(vaddq_f32(t03, t04), 1.33333333333)),
              vmulq_n_f32(vaddq_f32(t05, t06), -0.533333333333));
  float32x4_t m04 =
    vaddq_f32(vaddq_f32(vmulq_n_f32(vsubq_f32(t01, t02), 0.3), vmulq_n_f32(vsubq_f32(t04, t03), 1.33333333333)),
              vmulq_n_f32(vsubq_f32(t05, t06), 0.533333333333));
  float32x4_t m05 =
    vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t01, 0.03333333), vmulq_n_f32(t02, 0.0222222)),
                                            vmulq_n_f32(t03, 0.16666666666666667)),
                                  vmulq_n_f32(t04, 0.11111111111)),
                        vmulq_n_f32(t05, 0.1333333333)),
              vmulq_n_f32(t06, 0.08888888888));
  float32x4_t m06 =
    vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(t01, -0.03333333), vmulq_n_f32(t02, 0.0222222)),
                                            vmulq_n_f32(t03, 0.16666666666666667)),
                                  vmulq_n_f32(t04, 0.11111111111)),
                        vmulq_n_f32(t05, 0.1333333333)),
              vmulq_n_f32(t06, 0.08888888888));
  float32x4_t m07 =
    vaddq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t01, -0.5625), vmulq_n_f32(t03, 3.0625)), vmulq_n_f32(t05, 3.5)), t07);

  float32x4_t m10 =
    vsubq_f32(vaddq_f32(vsubq_f32(t10, vmulq_n_f32(t12, 5.444444444444444)), vmulq_n_f32(t14, 6.22222222222)),
              vmulq_n_f32(t16, 1.77777777777777777778));
  float32x4_t m11 = vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t11, 1.5), vmulq_n_f32(t12, 3)),
                                                            vmulq_n_f32(t13, 2.16666666666666667)),
                                                  vmulq_n_f32(t14, 4.3333333333)),
                                        vmulq_n_f32(t15, 0.66666666667)),
                              vmulq_n_f32(t16, 1.333333333333));
  float32x4_t m12 = vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(t11, -1.5), vmulq_n_f32(t12, 3)),
                                                            vmulq_n_f32(t13, 2.16666666666666667)),
                                                  vmulq_n_f32(t14, 4.3333333333)),
                                        vmulq_n_f32(t15, 0.66666666667)),
                              vmulq_n_f32(t16, 1.333333333333));
  float32x4_t m13 =
    vaddq_f32(vaddq_f32(vmulq_n_f32(vaddq_f32(t11, t12), -0.3), vmulq_n_f32(vaddq_f32(t13, t14), 1.33333333333)),
              vmulq_n_f32(vaddq_f32(t15, t16), -0.533333333333));
  float32x4_t m14 =
    vaddq_f32(vaddq_f32(vmulq_n_f32(vsubq_f32(t11, t12), 0.3), vmulq_n_f32(vsubq_f32(t14, t13), 1.33333333333)),
              vmulq_n_f32(vsubq_f32(t15, t16), 0.533333333333));
  float32x4_t m15 =
    vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t11, 0.03333333), vmulq_n_f32(t12, 0.0222222)),
                                            vmulq_n_f32(t13, 0.16666666666666667)),
                                  vmulq_n_f32(t14, 0.11111111111)),
                        vmulq_n_f32(t15, 0.1333333333)),
              vmulq_n_f32(t16, 0.08888888888));
  float32x4_t m16 =
    vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(t11, -0.03333333), vmulq_n_f32(t12, 0.0222222)),
                                            vmulq_n_f32(t13, 0.16666666666666667)),
                                  vmulq_n_f32(t14, 0.11111111111)),
                        vmulq_n_f32(t15, 0.1333333333)),
              vmulq_n_f32(t16, 0.08888888888));
  float32x4_t m17 =
    vaddq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t11, -0.5625), vmulq_n_f32(t13, 3.0625)), vmulq_n_f32(t15, 3.5)), t17);

  float32x4_t m20 =
    vsubq_f32(vaddq_f32(vsubq_f32(t20, vmulq_n_f32(t22, 5.444444444444444)), vmulq_n_f32(t24, 6.22222222222)),
              vmulq_n_f32(t26, 1.77777777777777777778));
  float32x4_t m21 = vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t21, 1.5), vmulq_n_f32(t22, 3)),
                                                            vmulq_n_f32(t23, 2.16666666666666667)),
                                                  vmulq_n_f32(t24, 4.3333333333)),
                                        vmulq_n_f32(t25, 0.66666666667)),
                              vmulq_n_f32(t26, 1.333333333333));
  float32x4_t m22 = vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(t21, -1.5), vmulq_n_f32(t22, 3)),
                                                            vmulq_n_f32(t23, 2.16666666666666667)),
                                                  vmulq_n_f32(t24, 4.3333333333)),
                                        vmulq_n_f32(t25, 0.66666666667)),
                              vmulq_n_f32(t26, 1.333333333333));
  float32x4_t m23 =
    vaddq_f32(vaddq_f32(vmulq_n_f32(vaddq_f32(t21, t22), -0.3), vmulq_n_f32(vaddq_f32(t23, t24), 1.33333333333)),
              vmulq_n_f32(vaddq_f32(t25, t26), -0.533333333333));
  float32x4_t m24 =
    vaddq_f32(vaddq_f32(vmulq_n_f32(vsubq_f32(t21, t22), 0.3), vmulq_n_f32(vsubq_f32(t24, t23), 1.33333333333)),
              vmulq_n_f32(vsubq_f32(t25, t26), 0.533333333333));
  float32x4_t m25 =
    vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t21, 0.03333333), vmulq_n_f32(t22, 0.0222222)),
                                            vmulq_n_f32(t23, 0.16666666666666667)),
                                  vmulq_n_f32(t24, 0.11111111111)),
                        vmulq_n_f32(t25, 0.1333333333)),
              vmulq_n_f32(t26, 0.08888888888));
  float32x4_t m26 =
    vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(t21, -0.03333333), vmulq_n_f32(t22, 0.0222222)),
                                            vmulq_n_f32(t23, 0.16666666666666667)),
                                  vmulq_n_f32(t24, 0.11111111111)),
                        vmulq_n_f32(t25, 0.1333333333)),
              vmulq_n_f32(t26, 0.08888888888));
  float32x4_t m27 =
    vaddq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t21, -0.5625), vmulq_n_f32(t23, 3.0625)), vmulq_n_f32(t25, 3.5)), t27);

  float32x4_t m30 =
    vsubq_f32(vaddq_f32(vsubq_f32(t30, vmulq_n_f32(t32, 5.444444444444444)), vmulq_n_f32(t34, 6.22222222222)),
              vmulq_n_f32(t36, 1.77777777777777777778));
  float32x4_t m31 = vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t31, 1.5), vmulq_n_f32(t32, 3)),
                                                            vmulq_n_f32(t33, 2.16666666666666667)),
                                                  vmulq_n_f32(t34, 4.3333333333)),
                                        vmulq_n_f32(t35, 0.66666666667)),
                              vmulq_n_f32(t36, 1.333333333333));
  float32x4_t m32 = vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(t31, -1.5), vmulq_n_f32(t32, 3)),
                                                            vmulq_n_f32(t33, 2.16666666666666667)),
                                                  vmulq_n_f32(t34, 4.3333333333)),
                                        vmulq_n_f32(t35, 0.66666666667)),
                              vmulq_n_f32(t36, 1.333333333333));
  float32x4_t m33 =
    vaddq_f32(vaddq_f32(vmulq_n_f32(vaddq_f32(t31, t32), -0.3), vmulq_n_f32(vaddq_f32(t33, t34), 1.33333333333)),
              vmulq_n_f32(vaddq_f32(t35, t36), -0.533333333333));
  float32x4_t m34 =
    vaddq_f32(vaddq_f32(vmulq_n_f32(vsubq_f32(t31, t32), 0.3), vmulq_n_f32(vsubq_f32(t34, t33), 1.33333333333)),
              vmulq_n_f32(vsubq_f32(t35, t36), 0.533333333333));
  float32x4_t m35 =
    vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t31, 0.03333333), vmulq_n_f32(t32, 0.0222222)),
                                            vmulq_n_f32(t33, 0.16666666666666667)),
                                  vmulq_n_f32(t34, 0.11111111111)),
                        vmulq_n_f32(t35, 0.1333333333)),
              vmulq_n_f32(t36, 0.08888888888));
  float32x4_t m36 =
    vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(t31, -0.03333333), vmulq_n_f32(t32, 0.0222222)),
                                            vmulq_n_f32(t33, 0.16666666666666667)),
                                  vmulq_n_f32(t34, 0.11111111111)),
                        vmulq_n_f32(t35, 0.1333333333)),
              vmulq_n_f32(t36, 0.08888888888));
  float32x4_t m37 =
    vaddq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t31, -0.5625), vmulq_n_f32(t33, 3.0625)), vmulq_n_f32(t35, 3.5)), t37);

  float32x4_t m40 =
    vsubq_f32(vaddq_f32(vsubq_f32(t40, vmulq_n_f32(t42, 5.444444444444444)), vmulq_n_f32(t44, 6.22222222222)),
              vmulq_n_f32(t46, 1.77777777777777777778));
  float32x4_t m41 = vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t41, 1.5), vmulq_n_f32(t42, 3)),
                                                            vmulq_n_f32(t43, 2.16666666666666667)),
                                                  vmulq_n_f32(t44, 4.3333333333)),
                                        vmulq_n_f32(t45, 0.66666666667)),
                              vmulq_n_f32(t46, 1.333333333333));
  float32x4_t m42 = vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(t41, -1.5), vmulq_n_f32(t42, 3)),
                                                            vmulq_n_f32(t43, 2.16666666666666667)),
                                                  vmulq_n_f32(t44, 4.3333333333)),
                                        vmulq_n_f32(t45, 0.66666666667)),
                              vmulq_n_f32(t46, 1.333333333333));
  float32x4_t m43 =
    vaddq_f32(vaddq_f32(vmulq_n_f32(vaddq_f32(t41, t42), -0.3), vmulq_n_f32(vaddq_f32(t43, t44), 1.33333333333)),
              vmulq_n_f32(vaddq_f32(t45, t46), -0.533333333333));
  float32x4_t m44 =
    vaddq_f32(vaddq_f32(vmulq_n_f32(vsubq_f32(t41, t42), 0.3), vmulq_n_f32(vsubq_f32(t44, t43), 1.33333333333)),
              vmulq_n_f32(vsubq_f32(t45, t46), 0.533333333333));
  float32x4_t m45 =
    vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t41, 0.03333333), vmulq_n_f32(t42, 0.0222222)),
                                            vmulq_n_f32(t43, 0.16666666666666667)),
                                  vmulq_n_f32(t44, 0.11111111111)),
                        vmulq_n_f32(t45, 0.1333333333)),
              vmulq_n_f32(t46, 0.08888888888));
  float32x4_t m46 =
    vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(t41, -0.03333333), vmulq_n_f32(t42, 0.0222222)),
                                            vmulq_n_f32(t43, 0.16666666666666667)),
                                  vmulq_n_f32(t44, 0.11111111111)),
                        vmulq_n_f32(t45, 0.1333333333)),
              vmulq_n_f32(t46, 0.08888888888));
  float32x4_t m47 =
    vaddq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t41, -0.5625), vmulq_n_f32(t43, 3.0625)), vmulq_n_f32(t45, 3.5)), t47);

  float32x4_t m50 =
    vsubq_f32(vaddq_f32(vsubq_f32(t50, vmulq_n_f32(t52, 5.444444444444444)), vmulq_n_f32(t54, 6.22222222222)),
              vmulq_n_f32(t56, 1.77777777777777777778));
  float32x4_t m51 = vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t51, 1.5), vmulq_n_f32(t52, 3)),
                                                            vmulq_n_f32(t53, 2.16666666666666667)),
                                                  vmulq_n_f32(t54, 4.3333333333)),
                                        vmulq_n_f32(t55, 0.66666666667)),
                              vmulq_n_f32(t56, 1.333333333333));
  float32x4_t m52 = vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(t51, -1.5), vmulq_n_f32(t52, 3)),
                                                            vmulq_n_f32(t53, 2.16666666666666667)),
                                                  vmulq_n_f32(t54, 4.3333333333)),
                                        vmulq_n_f32(t55, 0.66666666667)),
                              vmulq_n_f32(t56, 1.333333333333));
  float32x4_t m53 =
    vaddq_f32(vaddq_f32(vmulq_n_f32(vaddq_f32(t51, t52), -0.3), vmulq_n_f32(vaddq_f32(t53, t54), 1.33333333333)),
              vmulq_n_f32(vaddq_f32(t55, t56), -0.533333333333));
  float32x4_t m54 =
    vaddq_f32(vaddq_f32(vmulq_n_f32(vsubq_f32(t51, t52), 0.3), vmulq_n_f32(vsubq_f32(t54, t53), 1.33333333333)),
              vmulq_n_f32(vsubq_f32(t55, t56), 0.533333333333));
  float32x4_t m55 =
    vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t51, 0.03333333), vmulq_n_f32(t52, 0.0222222)),
                                            vmulq_n_f32(t53, 0.16666666666666667)),
                                  vmulq_n_f32(t54, 0.11111111111)),
                        vmulq_n_f32(t55, 0.1333333333)),
              vmulq_n_f32(t56, 0.08888888888));
  float32x4_t m56 =
    vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(t51, -0.03333333), vmulq_n_f32(t52, 0.0222222)),
                                            vmulq_n_f32(t53, 0.16666666666666667)),
                                  vmulq_n_f32(t54, 0.11111111111)),
                        vmulq_n_f32(t55, 0.1333333333)),
              vmulq_n_f32(t56, 0.08888888888));
  float32x4_t m57 =
    vaddq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t51, -0.5625), vmulq_n_f32(t53, 3.0625)), vmulq_n_f32(t55, 3.5)), t57);

  float32x4_t m60 =
    vsubq_f32(vaddq_f32(vsubq_f32(t60, vmulq_n_f32(t62, 5.444444444444444)), vmulq_n_f32(t64, 6.22222222222)),
              vmulq_n_f32(t66, 1.77777777777777777778));
  float32x4_t m61 = vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t61, 1.5), vmulq_n_f32(t62, 3)),
                                                            vmulq_n_f32(t63, 2.16666666666666667)),
                                                  vmulq_n_f32(t64, 4.3333333333)),
                                        vmulq_n_f32(t65, 0.66666666667)),
                              vmulq_n_f32(t66, 1.333333333333));
  float32x4_t m62 = vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(t61, -1.5), vmulq_n_f32(t62, 3)),
                                                            vmulq_n_f32(t63, 2.16666666666666667)),
                                                  vmulq_n_f32(t64, 4.3333333333)),
                                        vmulq_n_f32(t65, 0.66666666667)),
                              vmulq_n_f32(t66, 1.333333333333));
  float32x4_t m63 =
    vaddq_f32(vaddq_f32(vmulq_n_f32(vaddq_f32(t61, t62), -0.3), vmulq_n_f32(vaddq_f32(t63, t64), 1.33333333333)),
              vmulq_n_f32(vaddq_f32(t65, t66), -0.533333333333));
  float32x4_t m64 =
    vaddq_f32(vaddq_f32(vmulq_n_f32(vsubq_f32(t61, t62), 0.3), vmulq_n_f32(vsubq_f32(t64, t63), 1.33333333333)),
              vmulq_n_f32(vsubq_f32(t65, t66), 0.533333333333));
  float32x4_t m65 =
    vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t61, 0.03333333), vmulq_n_f32(t62, 0.0222222)),
                                            vmulq_n_f32(t63, 0.16666666666666667)),
                                  vmulq_n_f32(t64, 0.11111111111)),
                        vmulq_n_f32(t65, 0.1333333333)),
              vmulq_n_f32(t66, 0.08888888888));
  float32x4_t m66 =
    vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(t61, -0.03333333), vmulq_n_f32(t62, 0.0222222)),
                                            vmulq_n_f32(t63, 0.16666666666666667)),
                                  vmulq_n_f32(t64, 0.11111111111)),
                        vmulq_n_f32(t65, 0.1333333333)),
              vmulq_n_f32(t66, 0.08888888888));
  float32x4_t m67 =
    vaddq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t61, -0.5625), vmulq_n_f32(t63, 3.0625)), vmulq_n_f32(t65, 3.5)), t67);

  float32x4_t m70 =
    vsubq_f32(vaddq_f32(vsubq_f32(t70, vmulq_n_f32(t72, 5.444444444444444)), vmulq_n_f32(t74, 6.22222222222)),
              vmulq_n_f32(t76, 1.77777777777777777778));
  float32x4_t m71 = vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t71, 1.5), vmulq_n_f32(t72, 3)),
                                                            vmulq_n_f32(t73, 2.16666666666666667)),
                                                  vmulq_n_f32(t74, 4.3333333333)),
                                        vmulq_n_f32(t75, 0.66666666667)),
                              vmulq_n_f32(t76, 1.333333333333));
  float32x4_t m72 = vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(t71, -1.5), vmulq_n_f32(t72, 3)),
                                                            vmulq_n_f32(t73, 2.16666666666666667)),
                                                  vmulq_n_f32(t74, 4.3333333333)),
                                        vmulq_n_f32(t75, 0.66666666667)),
                              vmulq_n_f32(t76, 1.333333333333));
  float32x4_t m73 =
    vaddq_f32(vaddq_f32(vmulq_n_f32(vaddq_f32(t71, t72), -0.3), vmulq_n_f32(vaddq_f32(t73, t74), 1.33333333333)),
              vmulq_n_f32(vaddq_f32(t75, t76), -0.533333333333));
  float32x4_t m74 =
    vaddq_f32(vaddq_f32(vmulq_n_f32(vsubq_f32(t71, t72), 0.3), vmulq_n_f32(vsubq_f32(t74, t73), 1.33333333333)),
              vmulq_n_f32(vsubq_f32(t75, t76), 0.533333333333));
  float32x4_t m75 =
    vaddq_f32(vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t71, 0.03333333), vmulq_n_f32(t72, 0.0222222)),
                                            vmulq_n_f32(t73, 0.16666666666666667)),
                                  vmulq_n_f32(t74, 0.11111111111)),
                        vmulq_n_f32(t75, 0.1333333333)),
              vmulq_n_f32(t76, 0.08888888888));
  float32x4_t m76 =
    vaddq_f32(vsubq_f32(vsubq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(t71, -0.03333333), vmulq_n_f32(t72, 0.0222222)),
                                            vmulq_n_f32(t73, 0.16666666666666667)),
                                  vmulq_n_f32(t74, 0.11111111111)),
                        vmulq_n_f32(t75, 0.1333333333)),
              vmulq_n_f32(t76, 0.08888888888));
  float32x4_t m77 =
    vaddq_f32(vsubq_f32(vaddq_f32(vmulq_n_f32(t71, -0.5625), vmulq_n_f32(t73, 3.0625)), vmulq_n_f32(t75, 3.5)), t77);

  vst1q_f32(dst_data + 0 * dst_step, m00);
  vst1q_f32(dst_data + 1 * dst_step, m01);
  vst1q_f32(dst_data + 2 * dst_step, m02);
  vst1q_f32(dst_data + 3 * dst_step, m03);
  vst1q_f32(dst_data + 4 * dst_step, m04);
  vst1q_f32(dst_data + 5 * dst_step, m05);
  vst1q_f32(dst_data + 6 * dst_step, m06);
  vst1q_f32(dst_data + 7 * dst_step, m07);
  vst1q_f32(dst_data + 8 * dst_step, m10);
  vst1q_f32(dst_data + 9 * dst_step, m11);
  vst1q_f32(dst_data + 10 * dst_step, m12);
  vst1q_f32(dst_data + 11 * dst_step, m13);
  vst1q_f32(dst_data + 12 * dst_step, m14);
  vst1q_f32(dst_data + 13 * dst_step, m15);
  vst1q_f32(dst_data + 14 * dst_step, m16);
  vst1q_f32(dst_data + 15 * dst_step, m17);
  vst1q_f32(dst_data + 16 * dst_step, m20);
  vst1q_f32(dst_data + 17 * dst_step, m21);
  vst1q_f32(dst_data + 18 * dst_step, m22);
  vst1q_f32(dst_data + 19 * dst_step, m23);
  vst1q_f32(dst_data + 20 * dst_step, m24);
  vst1q_f32(dst_data + 21 * dst_step, m25);
  vst1q_f32(dst_data + 22 * dst_step, m26);
  vst1q_f32(dst_data + 23 * dst_step, m27);
  vst1q_f32(dst_data + 24 * dst_step, m30);
  vst1q_f32(dst_data + 25 * dst_step, m31);
  vst1q_f32(dst_data + 26 * dst_step, m32);
  vst1q_f32(dst_data + 27 * dst_step, m33);
  vst1q_f32(dst_data + 28 * dst_step, m34);
  vst1q_f32(dst_data + 29 * dst_step, m35);
  vst1q_f32(dst_data + 30 * dst_step, m36);
  vst1q_f32(dst_data + 31 * dst_step, m37);
  vst1q_f32(dst_data + 32 * dst_step, m40);
  vst1q_f32(dst_data + 33 * dst_step, m41);
  vst1q_f32(dst_data + 34 * dst_step, m42);
  vst1q_f32(dst_data + 35 * dst_step, m43);
  vst1q_f32(dst_data + 36 * dst_step, m44);
  vst1q_f32(dst_data + 37 * dst_step, m45);
  vst1q_f32(dst_data + 38 * dst_step, m46);
  vst1q_f32(dst_data + 39 * dst_step, m47);
  vst1q_f32(dst_data + 40 * dst_step, m50);
  vst1q_f32(dst_data + 41 * dst_step, m51);
  vst1q_f32(dst_data + 42 * dst_step, m52);
  vst1q_f32(dst_data + 43 * dst_step, m53);
  vst1q_f32(dst_data + 44 * dst_step, m54);
  vst1q_f32(dst_data + 45 * dst_step, m55);
  vst1q_f32(dst_data + 46 * dst_step, m56);
  vst1q_f32(dst_data + 47 * dst_step, m57);
  vst1q_f32(dst_data + 48 * dst_step, m60);
  vst1q_f32(dst_data + 49 * dst_step, m61);
  vst1q_f32(dst_data + 50 * dst_step, m62);
  vst1q_f32(dst_data + 51 * dst_step, m63);
  vst1q_f32(dst_data + 52 * dst_step, m64);
  vst1q_f32(dst_data + 53 * dst_step, m65);
  vst1q_f32(dst_data + 54 * dst_step, m66);
  vst1q_f32(dst_data + 55 * dst_step, m67);
  vst1q_f32(dst_data + 56 * dst_step, m70);
  vst1q_f32(dst_data + 57 * dst_step, m71);
  vst1q_f32(dst_data + 58 * dst_step, m72);
  vst1q_f32(dst_data + 59 * dst_step, m73);
  vst1q_f32(dst_data + 60 * dst_step, m74);
  vst1q_f32(dst_data + 61 * dst_step, m75);
  vst1q_f32(dst_data + 62 * dst_step, m76);
  vst1q_f32(dst_data + 63 * dst_step, m77);
#else
  for (int i = 0; i < C4NUM; i++) {
    float src_data_00 = src_data[i];
    float src_data_01 = src_data[i + src_step];
    float src_data_02 = src_data[i + 2 * src_step];
    float src_data_03 = src_data[i + 3 * src_step];
    float src_data_04 = src_data[i + 4 * src_step];
    float src_data_05 = src_data[i + 5 * src_step];
    float src_data_06 = src_data[i + 6 * src_step];
    float src_data_07 = src_data[i + 7 * src_step];
    float src_data_10 = src_data[i + 8 * src_step];
    float src_data_11 = src_data[i + 9 * src_step];
    float src_data_12 = src_data[i + 10 * src_step];
    float src_data_13 = src_data[i + 11 * src_step];
    float src_data_14 = src_data[i + 12 * src_step];
    float src_data_15 = src_data[i + 13 * src_step];
    float src_data_16 = src_data[i + 14 * src_step];
    float src_data_17 = src_data[i + 15 * src_step];
    float src_data_20 = src_data[i + 16 * src_step];
    float src_data_21 = src_data[i + 17 * src_step];
    float src_data_22 = src_data[i + 18 * src_step];
    float src_data_23 = src_data[i + 19 * src_step];
    float src_data_24 = src_data[i + 20 * src_step];
    float src_data_25 = src_data[i + 21 * src_step];
    float src_data_26 = src_data[i + 22 * src_step];
    float src_data_27 = src_data[i + 23 * src_step];
    float src_data_30 = src_data[i + 24 * src_step];
    float src_data_31 = src_data[i + 25 * src_step];
    float src_data_32 = src_data[i + 26 * src_step];
    float src_data_33 = src_data[i + 27 * src_step];
    float src_data_34 = src_data[i + 28 * src_step];
    float src_data_35 = src_data[i + 29 * src_step];
    float src_data_36 = src_data[i + 30 * src_step];
    float src_data_37 = src_data[i + 31 * src_step];
    float src_data_40 = src_data[i + 32 * src_step];
    float src_data_41 = src_data[i + 33 * src_step];
    float src_data_42 = src_data[i + 34 * src_step];
    float src_data_43 = src_data[i + 35 * src_step];
    float src_data_44 = src_data[i + 36 * src_step];
    float src_data_45 = src_data[i + 37 * src_step];
    float src_data_46 = src_data[i + 38 * src_step];
    float src_data_47 = src_data[i + 39 * src_step];
    float src_data_50 = src_data[i + 40 * src_step];
    float src_data_51 = src_data[i + 41 * src_step];
    float src_data_52 = src_data[i + 42 * src_step];
    float src_data_53 = src_data[i + 43 * src_step];
    float src_data_54 = src_data[i + 44 * src_step];
    float src_data_55 = src_data[i + 45 * src_step];
    float src_data_56 = src_data[i + 46 * src_step];
    float src_data_57 = src_data[i + 47 * src_step];
    float src_data_60 = src_data[i + 48 * src_step];
    float src_data_61 = src_data[i + 49 * src_step];
    float src_data_62 = src_data[i + 50 * src_step];
    float src_data_63 = src_data[i + 51 * src_step];
    float src_data_64 = src_data[i + 52 * src_step];
    float src_data_65 = src_data[i + 53 * src_step];
    float src_data_66 = src_data[i + 54 * src_step];
    float src_data_67 = src_data[i + 55 * src_step];
    float src_data_70 = src_data[i + 56 * src_step];
    float src_data_71 = src_data[i + 57 * src_step];
    float src_data_72 = src_data[i + 58 * src_step];
    float src_data_73 = src_data[i + 59 * src_step];
    float src_data_74 = src_data[i + 60 * src_step];
    float src_data_75 = src_data[i + 61 * src_step];
    float src_data_76 = src_data[i + 62 * src_step];
    float src_data_77 = src_data[i + 63 * src_step];

    float t00 = src_data_00 - 5.444444444444444445125f * src_data_20 + 6.222222222222222222223f * src_data_40 -
                1.77777777777777778f * src_data_60;
    float t01 = src_data_01 - 5.444444444444444445125f * src_data_21 + 6.222222222222222222223f * src_data_41 -
                1.77777777777777778f * src_data_61;
    float t02 = src_data_02 - 5.444444444444444445125f * src_data_22 + 6.222222222222222222223f * src_data_42 -
                1.77777777777777778f * src_data_62;
    float t03 = src_data_03 - 5.444444444444444445125f * src_data_23 + 6.222222222222222222223f * src_data_43 -
                1.77777777777777778f * src_data_63;
    float t04 = src_data_04 - 5.444444444444444445125f * src_data_24 + 6.222222222222222222223f * src_data_44 -
                1.77777777777777778f * src_data_64;
    float t05 = src_data_05 - 5.444444444444444445125f * src_data_25 + 6.222222222222222222223f * src_data_45 -
                1.77777777777777778f * src_data_65;
    float t06 = src_data_06 - 5.444444444444444445125f * src_data_26 + 6.222222222222222222223f * src_data_46 -
                1.77777777777777778f * src_data_66;
    float t07 = src_data_07 - 5.444444444444444445125f * src_data_27 + 6.222222222222222222223f * src_data_47 -
                1.77777777777777778f * src_data_67;

    const float t10 = 1.5f * src_data_10 + 3.0f * src_data_20 - 2.1666666666666667f * src_data_30 -
                4.333333333333333333f * src_data_40 + 0.66666666666666667f * src_data_50 +
                1.333333333333333f * src_data_60;
    const float t11 = 1.5f * src_data_11 + 3.0f * src_data_21 - 2.1666666666666667f * src_data_31 -
                4.333333333333333333f * src_data_41 + 0.66666666666666667f * src_data_51 +
                1.333333333333333f * src_data_61;
    const float t12 = 1.5f * src_data_12 + 3.0f * src_data_22 - 2.1666666666666667f * src_data_32 -
                4.333333333333333333f * src_data_42 + 0.66666666666666667f * src_data_52 +
                1.333333333333333f * src_data_62;
    const float t13 = 1.5f * src_data_13 + 3.0f * src_data_23 - 2.1666666666666667f * src_data_33 -
                4.333333333333333333f * src_data_43 + 0.66666666666666667f * src_data_53 +
                1.333333333333333f * src_data_63;
    const float t14 = 1.5f * src_data_14 + 3.0f * src_data_24 - 2.1666666666666667f * src_data_34 -
                4.333333333333333333f * src_data_44 + 0.66666666666666667f * src_data_54 +
                1.333333333333333f * src_data_64;
    const float t15 = 1.5f * src_data_15 + 3.0f * src_data_25 - 2.1666666666666667f * src_data_35 -
                4.333333333333333333f * src_data_45 + 0.66666666666666667f * src_data_55 +
                1.333333333333333f * src_data_65;
    const float t16 = 1.5f * src_data_16 + 3.0f * src_data_26 - 2.1666666666666667f * src_data_36 -
                4.333333333333333333f * src_data_46 + 0.66666666666666667f * src_data_56 +
                1.333333333333333f * src_data_66;
    const float t17 = 1.5f * src_data_17 + 3.0f * src_data_27 - 2.1666666666666667f * src_data_37 -
                4.333333333333333333f * src_data_47 + 0.66666666666666667f * src_data_57 +
                1.333333333333333f * src_data_67;

    const float t20 = -1.5f * src_data_10 + 3.0f * src_data_20 + 2.1666666666666667f * src_data_30 -
                4.333333333333333333f * src_data_40 - 0.66666666666666667f * src_data_50 +
                1.333333333333333f * src_data_60;
    const float t21 = -1.5f * src_data_11 + 3.0f * src_data_21 + 2.1666666666666667f * src_data_31 -
                4.333333333333333333f * src_data_41 - 0.66666666666666667f * src_data_51 +
                1.333333333333333f * src_data_61;
    const float t22 = -1.5f * src_data_12 + 3.0f * src_data_22 + 2.1666666666666667f * src_data_32 -
                4.333333333333333333f * src_data_42 - 0.66666666666666667f * src_data_52 +
                1.333333333333333f * src_data_62;
    const float t23 = -1.5f * src_data_13 + 3.0f * src_data_23 + 2.1666666666666667f * src_data_33 -
                4.333333333333333333f * src_data_43 - 0.66666666666666667f * src_data_53 +
                1.333333333333333f * src_data_63;
    const float t24 = -1.5f * src_data_14 + 3.0f * src_data_24 + 2.1666666666666667f * src_data_34 -
                4.333333333333333333f * src_data_44 - 0.66666666666666667f * src_data_54 +
                1.333333333333333f * src_data_64;
    const float t25 = -1.5f * src_data_15 + 3.0f * src_data_25 + 2.1666666666666667f * src_data_35 -
                4.333333333333333333f * src_data_45 - 0.66666666666666667f * src_data_55 +
                1.333333333333333f * src_data_65;
    const float t26 = -1.5f * src_data_16 + 3.0f * src_data_26 + 2.1666666666666667f * src_data_36 -
                4.333333333333333333f * src_data_46 - 0.66666666666666667f * src_data_56 +
                1.333333333333333f * src_data_66;
    const float t27 = -1.5f * src_data_17 + 3.0f * src_data_27 + 2.1666666666666667f * src_data_37 -
                4.333333333333333333f * src_data_47 - 0.66666666666666667f * src_data_57 +
                1.333333333333333f * src_data_67;

    const float t30 = -0.3f * (src_data_10 + src_data_20) + 1.33333333333333f * (src_data_30 + src_data_40) -
                0.53333333333f * (src_data_50 + src_data_60);
    const float t31 = -0.3f * (src_data_11 + src_data_21) + 1.33333333333333f * (src_data_31 + src_data_41) -
                0.53333333333f * (src_data_51 + src_data_61);
    const float t32 = -0.3f * (src_data_12 + src_data_22) + 1.33333333333333f * (src_data_32 + src_data_42) -
                0.53333333333f * (src_data_52 + src_data_62);
    const float t33 = -0.3f * (src_data_13 + src_data_23) + 1.33333333333333f * (src_data_33 + src_data_43) -
                0.53333333333f * (src_data_53 + src_data_63);
    const float t34 = -0.3f * (src_data_14 + src_data_24) + 1.33333333333333f * (src_data_34 + src_data_44) -
                0.53333333333f * (src_data_54 + src_data_64);
    const float t35 = -0.3f * (src_data_15 + src_data_25) + 1.33333333333333f * (src_data_35 + src_data_45) -
                0.53333333333f * (src_data_55 + src_data_65);
    const const float t36 = -0.3f * (src_data_16 + src_data_26) + 1.33333333333333f * (src_data_36 + src_data_46) -
                0.53333333333f * (src_data_56 + src_data_66);
    const const float t37 = -0.3f * (src_data_17 + src_data_27) + 1.33333333333333f * (src_data_37 + src_data_47) -
                0.53333333333f * (src_data_57 + src_data_67);

    const float t40 = 0.3f * (src_data_10 - src_data_20) + 1.33333333333333f * (src_data_40 - src_data_30) +
                0.53333333333f * (src_data_50 - src_data_60);
    const float t41 = 0.3f * (src_data_11 - src_data_21) + 1.33333333333333f * (src_data_41 - src_data_31) +
                0.53333333333f * (src_data_51 - src_data_61);
    const float t42 = 0.3f * (src_data_12 - src_data_22) + 1.33333333333333f * (src_data_42 - src_data_32) +
                0.53333333333f * (src_data_52 - src_data_62);
    const float t43 = 0.3f * (src_data_13 - src_data_23) + 1.33333333333333f * (src_data_43 - src_data_33) +
                0.53333333333f * (src_data_53 - src_data_63);
    const float t44 = 0.3f * (src_data_14 - src_data_24) + 1.33333333333333f * (src_data_44 - src_data_34) +
                0.53333333333f * (src_data_54 - src_data_64);
    const float t45 = 0.3f * (src_data_15 - src_data_25) + 1.33333333333333f * (src_data_45 - src_data_35) +
                0.53333333333f * (src_data_55 - src_data_65);
    const float t46 = 0.3f * (src_data_16 - src_data_26) + 1.33333333333333f * (src_data_46 - src_data_36) +
                0.53333333333f * (src_data_56 - src_data_66);
    const float t47 = 0.3f * (src_data_17 - src_data_27) + 1.33333333333333f * (src_data_47 - src_data_37) +
                0.53333333333f * (src_data_57 - src_data_67);

    const float t50 = 0.0333333333f * src_data_10 + 0.02222222f * src_data_20 - 0.1666666666f * src_data_30 -
                0.1111111111f * src_data_40 + 0.1333333f * src_data_50 + 0.0888888f * src_data_60;
    const float t51 = 0.0333333333f * src_data_11 + 0.02222222f * src_data_21 - 0.1666666666f * src_data_31 -
                0.1111111111f * src_data_41 + 0.1333333f * src_data_51 + 0.0888888f * src_data_61;
    const float t52 = 0.0333333333f * src_data_12 + 0.02222222f * src_data_22 - 0.1666666666f * src_data_32 -
                0.1111111111f * src_data_42 + 0.1333333f * src_data_52 + 0.0888888f * src_data_62;
    const float t53 = 0.0333333333f * src_data_13 + 0.02222222f * src_data_23 - 0.1666666666f * src_data_33 -
                0.1111111111f * src_data_43 + 0.1333333f * src_data_53 + 0.0888888f * src_data_63;
    const float t54 = 0.0333333333f * src_data_14 + 0.02222222f * src_data_24 - 0.1666666666f * src_data_34 -
                0.1111111111f * src_data_44 + 0.1333333f * src_data_54 + 0.0888888f * src_data_64;
    const float t55 = 0.0333333333f * src_data_15 + 0.02222222f * src_data_25 - 0.1666666666f * src_data_35 -
                0.1111111111f * src_data_45 + 0.1333333f * src_data_55 + 0.0888888f * src_data_65;
    const float t56 = 0.0333333333f * src_data_16 + 0.02222222f * src_data_26 - 0.1666666666f * src_data_36 -
                0.1111111111f * src_data_46 + 0.1333333f * src_data_56 + 0.0888888f * src_data_66;
    const float t57 = 0.0333333333f * src_data_17 + 0.02222222f * src_data_27 - 0.1666666666f * src_data_37 -
                0.1111111111f * src_data_47 + 0.1333333f * src_data_57 + 0.0888888f * src_data_67;

    const float t60 = -0.0333333333f * src_data_10 + 0.02222222f * src_data_20 + 0.1666666666f * src_data_30 -
                0.1111111111f * src_data_40 - 0.1333333f * src_data_50 + 0.0888888f * src_data_60;
    const float t61 = -0.0333333333f * src_data_11 + 0.02222222f * src_data_21 + 0.1666666666f * src_data_31 -
                0.1111111111f * src_data_41 - 0.1333333f * src_data_51 + 0.0888888f * src_data_61;
    const float t62 = -0.0333333333f * src_data_12 + 0.02222222f * src_data_22 + 0.1666666666f * src_data_32 -
                0.1111111111f * src_data_42 - 0.1333333f * src_data_52 + 0.0888888f * src_data_62;
    const float t63 = -0.0333333333f * src_data_13 + 0.02222222f * src_data_23 + 0.1666666666f * src_data_33 -
                0.1111111111f * src_data_43 - 0.1333333f * src_data_53 + 0.0888888f * src_data_63;
    const float t64 = -0.0333333333f * src_data_14 + 0.02222222f * src_data_24 + 0.1666666666f * src_data_34 -
                0.1111111111f * src_data_44 - 0.1333333f * src_data_54 + 0.0888888f * src_data_64;
    const float t65 = -0.0333333333f * src_data_15 + 0.02222222f * src_data_25 + 0.1666666666f * src_data_35 -
                0.1111111111f * src_data_45 - 0.1333333f * src_data_55 + 0.0888888f * src_data_65;
    const float t66 = -0.0333333333f * src_data_16 + 0.02222222f * src_data_26 + 0.1666666666f * src_data_36 -
                0.1111111111f * src_data_46 - 0.1333333f * src_data_56 + 0.0888888f * src_data_66;
    const float t67 = -0.0333333333f * src_data_17 + 0.02222222f * src_data_27 + 0.1666666666f * src_data_37 -
                0.1111111111f * src_data_47 - 0.1333333f * src_data_57 + 0.0888888f * src_data_67;

    const float t70 = -0.5625f * src_data_10 + 3.0625f * src_data_30 - 3.5f * src_data_50 + src_data_70;
    const float t71 = -0.5625f * src_data_11 + 3.0625f * src_data_31 - 3.5f * src_data_51 + src_data_71;
    const float t72 = -0.5625f * src_data_12 + 3.0625f * src_data_32 - 3.5f * src_data_52 + src_data_72;
    const float t73 = -0.5625f * src_data_13 + 3.0625f * src_data_33 - 3.5f * src_data_53 + src_data_73;
    const float t74 = -0.5625f * src_data_14 + 3.0625f * src_data_34 - 3.5f * src_data_54 + src_data_74;
    const float t75 = -0.5625f * src_data_15 + 3.0625f * src_data_35 - 3.5f * src_data_55 + src_data_75;
    const float t76 = -0.5625f * src_data_16 + 3.0625f * src_data_36 - 3.5f * src_data_56 + src_data_76;
    const float t77 = -0.5625f * src_data_17 + 3.0625f * src_data_37 - 3.5f * src_data_57 + src_data_77;

    const float m00 = t00 - 5.444444444444444445125f * t02 + 6.222222222222222222223f * t04 -
                      1.77777777777777778f * t06;
    const float m01 = 1.5f * t01 + 3.0f * t02 - 2.1666666666666667f * t03 - 4.333333333333333333f * t04 +
                0.66666666666666667f * t05 + 1.333333333333333f * t06;
    const float m02 = -1.5f * t01 + 3.0f * t02 + 2.1666666666666667f * t03 - 4.333333333333333333f * t04 -
                0.66666666666666667f * t05 + 1.333333333333333f * t06;
    const float m03 = -0.3f * (t01 + t02) + 1.33333333333333f * (t03 + t04) - 0.53333333333f * (t05 + t06);
    const float m04 = 0.3f * (t01 - t02) + 1.33333333333333f * (t04 - t03) + 0.53333333333f * (t05 - t06);
    const float m05 = 0.0333333333f * t01 + 0.02222222f * t02 - 0.1666666666f * t03 - 0.1111111111f * t04 +
                      0.1333333f * t05 + 0.0888888f * t06;
    const float m06 = -0.0333333333f * t01 + 0.02222222f * t02 + 0.1666666666f * t03 - 0.1111111111f * t04 -
                0.1333333f * t05 + 0.0888888f * t06;
    const float m07 = -0.5625f * t01 + 3.0625f * t03 - 3.5f * t05 + t07;

    float m10 = t10 - 5.444444444444444445125f * t12 + 6.222222222222222222223f * t14 - 1.77777777777777778f * t16;
    const float m11 = 1.5f * t11 + 3.0f * t12 - 2.1666666666666667f * t13 - 4.333333333333333333f * t14 +
                0.66666666666666667f * t15 + 1.333333333333333f * t16;
    const float m12 = -1.5f * t11 + 3.0f * t12 + 2.1666666666666667f * t13 - 4.333333333333333333f * t14 -
                0.66666666666666667f * t15 + 1.333333333333333f * t16;
    const float m13 = -0.3f * (t11 + t12) + 1.33333333333333f * (t13 + t14) - 0.53333333333f * (t15 + t16);
    const float m14 = 0.3f * (t11 - t12) + 1.33333333333333f * (t14 - t13) + 0.53333333333f * (t15 - t16);
    const float m15 = 0.0333333333f * t11 + 0.02222222f * t12 - 0.1666666666f * t13 - 0.1111111111f * t14 +
                      0.1333333f * t15 + 0.0888888f * t16;
    const float m16 = -0.0333333333f * t11 + 0.02222222f * t12 + 0.1666666666f * t13 - 0.1111111111f * t14 -
                0.1333333f * t15 + 0.0888888f * t16;
    const float m17 = -0.5625f * t11 + 3.0625f * t13 - 3.5f * t15 + t17;

    const float m20 = t20 - 5.444444444444444445125f * t22 + 6.222222222222222222223f * t24 -
                      1.77777777777777778f * t26;
    const float m21 = 1.5f * t21 + 3.0f * t22 - 2.1666666666666667f * t23 - 4.333333333333333333f * t24 +
                0.66666666666666667f * t25 + 1.333333333333333f * t26;
    const float m22 = -1.5f * t21 + 3.0f * t22 + 2.1666666666666667f * t23 - 4.333333333333333333f * t24 -
                0.66666666666666667f * t25 + 1.333333333333333f * t26;
    const float m23 = -0.3f * (t21 + t22) + 1.33333333333333f * (t23 + t24) - 0.53333333333f * (t25 + t26);
    const float m24 = 0.3f * (t21 - t22) + 1.33333333333333f * (t24 - t23) + 0.53333333333f * (t25 - t26);
    const float m25 = 0.0333333333f * t21 + 0.02222222f * t22 - 0.1666666666f * t23 - 0.1111111111f * t24 +
                      0.1333333f * t25 + 0.0888888f * t26;
    const float m26 = -0.0333333333f * t21 + 0.02222222f * t22 + 0.1666666666f * t23 - 0.1111111111f * t24 -
                0.1333333f * t25 + 0.0888888f * t26;
    const float m27 = -0.5625f * t21 + 3.0625f * t23 - 3.5f * t25 + t27;

    float m30 = t30 - 5.444444444444444445125f * t32 + 6.222222222222222222223f * t34 - 1.77777777777777778f * t36;
    const float m31 = 1.5f * t31 + 3.0f * t32 - 2.1666666666666667f * t33 - 4.333333333333333333f * t34 +
                0.66666666666666667f * t35 + 1.333333333333333f * t36;
    const float m32 = -1.5f * t31 + 3.0f * t32 + 2.1666666666666667f * t33 - 4.333333333333333333f * t34 -
                0.66666666666666667f * t35 + 1.333333333333333f * t36;
    const float m33 = -0.3f * (t31 + t32) + 1.33333333333333f * (t33 + t34) - 0.53333333333f * (t35 + t36);
    const float m34 = 0.3f * (t31 - t32) + 1.33333333333333f * (t34 - t33) + 0.53333333333f * (t35 - t36);
    const float m35 = 0.0333333333f * t31 + 0.02222222f * t32 - 0.1666666666f * t33 - 0.1111111111f * t34 +
                      0.1333333f * t35 + 0.0888888f * t36;
    const float m36 = -0.0333333333f * t31 + 0.02222222f * t32 + 0.1666666666f * t33 - 0.1111111111f * t34 -
                0.1333333f * t35 + 0.0888888f * t36;
    const float m37 = -0.5625f * t31 + 3.0625f * t33 - 3.5f * t35 + t37;

    const float m40 = t40 - 5.444444444444444445125f * t42 + 6.222222222222222222223f * t44 -
                      1.77777777777777778f * t46;
    const float m41 = 1.5f * t41 + 3.0f * t42 - 2.1666666666666667f * t43 - 4.333333333333333333f * t44 +
                0.66666666666666667f * t45 + 1.333333333333333f * t46;
    const float m42 = -1.5f * t41 + 3.0f * t42 + 2.1666666666666667f * t43 - 4.333333333333333333f * t44 -
                0.66666666666666667f * t45 + 1.333333333333333f * t46;
    const float m43 = -0.3f * (t41 + t42) + 1.33333333333333f * (t43 + t44) - 0.53333333333f * (t45 + t46);
    const float m44 = 0.3f * (t41 - t42) + 1.33333333333333f * (t44 - t43) + 0.53333333333f * (t45 - t46);
    const float m45 = 0.0333333333f * t41 + 0.02222222f * t42 - 0.1666666666f * t43 - 0.1111111111f * t44 +
                      0.1333333f * t45 + 0.0888888f * t46;
    const float m46 = -0.0333333333f * t41 + 0.02222222f * t42 + 0.1666666666f * t43 - 0.1111111111f * t44 -
                0.1333333f * t45 + 0.0888888f * t46;
    const float m47 = -0.5625f * t41 + 3.0625f * t43 - 3.5f * t45 + t47;

    float m50 = t50 - 5.444444444444444445125f * t52 + 6.222222222222222222223f * t54 - 1.77777777777777778f * t56;
    const float m51 = 1.5f * t51 + 3.0f * t52 - 2.1666666666666667f * t53 - 4.333333333333333333f * t54 +
                0.66666666666666667f * t55 + 1.333333333333333f * t56;
    const float m52 = -1.5f * t51 + 3.0f * t52 + 2.1666666666666667f * t53 - 4.333333333333333333f * t54 -
                0.66666666666666667f * t55 + 1.333333333333333f * t56;
    const float m53 = -0.3f * (t51 + t52) + 1.33333333333333f * (t53 + t54) - 0.53333333333f * (t55 + t56);
    const float m54 = 0.3f * (t51 - t52) + 1.33333333333333f * (t54 - t53) + 0.53333333333f * (t55 - t56);
    const float m55 = 0.0333333333f * t51 + 0.02222222f * t52 - 0.1666666666f * t53 - 0.1111111111f * t54 +
                      0.1333333f * t55 + 0.0888888f * t56;
    const float m56 = -0.0333333333f * t51 + 0.02222222f * t52 + 0.1666666666f * t53 - 0.1111111111f * t54 -
                0.1333333f * t55 + 0.0888888f * t56;
    const float m57 = -0.5625f * t51 + 3.0625f * t53 - 3.5f * t55 + t57;

    float m60 = t60 - 5.444444444444444445125f * t62 + 6.222222222222222222223f * t64 - 1.77777777777777778f * t66;
    const float m61 = 1.5f * t61 + 3.0f * t62 - 2.1666666666666667f * t63 - 4.333333333333333333f * t64 +
                0.66666666666666667f * t65 + 1.333333333333333f * t66;
    const float m62 = -1.5f * t61 + 3.0f * t62 + 2.1666666666666667f * t63 - 4.333333333333333333f * t64 -
                0.66666666666666667f * t65 + 1.333333333333333f * t66;
    const float m63 = -0.3f * (t61 + t62) + 1.33333333333333f * (t63 + t64) - 0.53333333333f * (t65 + t66);
    const float m64 = 0.3f * (t61 - t62) + 1.33333333333333f * (t64 - t63) + 0.53333333333f * (t65 - t66);
    const float m65 = 0.0333333333f * t61 + 0.02222222f * t62 - 0.1666666666f * t63 - 0.1111111111f * t64 +
                      0.1333333f * t65 + 0.0888888f * t66;
    const float m66 = -0.0333333333f * t61 + 0.02222222f * t62 + 0.1666666666f * t63 - 0.1111111111f * t64 -
                0.1333333f * t65 + 0.0888888f * t66;
    const float m67 = -0.5625f * t61 + 3.0625f * t63 - 3.5f * t65 + t67;

    float m70 = t70 - 5.444444444444444445125f * t72 + 6.222222222222222222223f * t74 - 1.77777777777777778f * t76;
    const float m71 = 1.5f * t71 + 3.0f * t72 - 2.1666666666666667f * t73 - 4.333333333333333333f * t74 +
                0.66666666666666667f * t75 + 1.333333333333333f * t76;
    const float m72 = -1.5f * t71 + 3.0f * t72 + 2.1666666666666667f * t73 - 4.333333333333333333f * t74 -
                0.66666666666666667f * t75 + 1.333333333333333f * t76;
    const float m73 = -0.3f * (t71 + t72) + 1.33333333333333f * (t73 + t74) - 0.53333333333f * (t75 + t76);
    const float m74 = 0.3f * (t71 - t72) + 1.33333333333333f * (t74 - t73) + 0.53333333333f * (t75 - t76);
    const float m75 = 0.0333333333f * t71 + 0.02222222f * t72 - 0.1666666666f * t73 - 0.1111111111f * t74 +
                      0.1333333f * t75 + 0.0888888f * t76;
    const float m76 = -0.0333333333f * t71 + 0.02222222f * t72 + 0.1666666666f * t73 - 0.1111111111f * t74 -
                0.1333333f * t75 + 0.0888888f * t76;
    const float m77 = -0.5625f * t71 + 3.0625f * t73 - 3.5f * t75 + t77;

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

void OutputTransform4x2Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                            int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t bias_ptr = vld1q_f32(bias_data);
  float32x4_t src_data_00 = vld1q_f32(src_data + 0 * src_step);
  float32x4_t src_data_01 = vld1q_f32(src_data + 1 * src_step);
  float32x4_t src_data_02 = vld1q_f32(src_data + 2 * src_step);
  float32x4_t src_data_03 = vld1q_f32(src_data + 3 * src_step);
  float32x4_t src_data_10 = vld1q_f32(src_data + 4 * src_step);
  float32x4_t src_data_11 = vld1q_f32(src_data + 5 * src_step);
  float32x4_t src_data_12 = vld1q_f32(src_data + 6 * src_step);
  float32x4_t src_data_13 = vld1q_f32(src_data + 7 * src_step);
  float32x4_t src_data_20 = vld1q_f32(src_data + 8 * src_step);
  float32x4_t src_data_21 = vld1q_f32(src_data + 9 * src_step);
  float32x4_t src_data_22 = vld1q_f32(src_data + 10 * src_step);
  float32x4_t src_data_23 = vld1q_f32(src_data + 11 * src_step);
  float32x4_t src_data_30 = vld1q_f32(src_data + 12 * src_step);
  float32x4_t src_data_31 = vld1q_f32(src_data + 13 * src_step);
  float32x4_t src_data_32 = vld1q_f32(src_data + 14 * src_step);
  float32x4_t src_data_33 = vld1q_f32(src_data + 15 * src_step);

  float32x4_t t00 = vaddq_f32(src_data_00, vaddq_f32(src_data_10, src_data_20));
  float32x4_t t01 = vaddq_f32(src_data_01, vaddq_f32(src_data_11, src_data_21));
  float32x4_t t02 = vaddq_f32(src_data_02, vaddq_f32(src_data_12, src_data_22));
  float32x4_t t03 = vaddq_f32(src_data_03, vaddq_f32(src_data_13, src_data_23));

  float32x4_t t10 = vsubq_f32(src_data_30, vmulq_n_f32(vsubq_f32(src_data_10, src_data_20), 0.5));
  float32x4_t t11 = vsubq_f32(src_data_31, vmulq_n_f32(vsubq_f32(src_data_11, src_data_21), 0.5));
  float32x4_t t12 = vsubq_f32(src_data_32, vmulq_n_f32(vsubq_f32(src_data_12, src_data_22), 0.5));
  float32x4_t t13 = vsubq_f32(src_data_33, vmulq_n_f32(vsubq_f32(src_data_13, src_data_23), 0.5));

  float32x4_t m00 = vaddq_f32(vaddq_f32(t00, vaddq_f32(t01, t02)), bias_ptr);
  float32x4_t m01 = vaddq_f32(vaddq_f32(t03, vmulq_n_f32(vsubq_f32(t01, t02), 0.5)), bias_ptr);
  float32x4_t m10 = vaddq_f32(vaddq_f32(t10, vaddq_f32(t11, t12)), bias_ptr);
  float32x4_t m11 = vaddq_f32(vaddq_f32(t13, vmulq_n_f32(vsubq_f32(t11, t12), 0.5)), bias_ptr);

  vst1q_f32(dst_data, m00);
  vst1q_f32(dst_data + C4NUM, m01);
  vst1q_f32(dst_data + dst_step * C4NUM, m10);
  vst1q_f32(dst_data + dst_step * C4NUM + C4NUM, m11);
#else
  for (int i = 0; i < C4NUM; i++) {
    float src_data_00 = src_data[i];
    float src_data_01 = src_data[i + src_step];
    float src_data_02 = src_data[i + 2 * src_step];
    float src_data_03 = src_data[i + 3 * src_step];
    float src_data_10 = src_data[i + 4 * src_step];
    float src_data_11 = src_data[i + 5 * src_step];
    float src_data_12 = src_data[i + 6 * src_step];
    float src_data_13 = src_data[i + 7 * src_step];
    float src_data_20 = src_data[i + 8 * src_step];
    float src_data_21 = src_data[i + 9 * src_step];
    float src_data_22 = src_data[i + 10 * src_step];
    float src_data_23 = src_data[i + 11 * src_step];
    float src_data_30 = src_data[i + 12 * src_step];
    float src_data_31 = src_data[i + 13 * src_step];
    float src_data_32 = src_data[i + 14 * src_step];
    float src_data_33 = src_data[i + 15 * src_step];

    float t00 = src_data_00 + src_data_10 + src_data_20;
    float t01 = src_data_01 + src_data_11 + src_data_21;
    float t02 = src_data_02 + src_data_12 + src_data_22;
    float t03 = src_data_03 + src_data_13 + src_data_23;

    const float t10 = 0.5f * (src_data_10 - src_data_20) + src_data_30;
    const float t11 = 0.5f * (src_data_11 - src_data_21) + src_data_31;
    const float t12 = 0.5f * (src_data_12 - src_data_22) + src_data_32;
    const float t13 = 0.5f * (src_data_13 - src_data_23) + src_data_33;

    float m00 = t00 + t01 + t02 + bias_data[i];
    const float m01 = 0.5f * (t01 - t02) + t03 + bias_data[i];
    float m10 = t10 + t11 + t12 + bias_data[i];
    const float m11 = 0.5f * (t11 - t12) + t13 + bias_data[i];

    (dst_data + i)[0] = m00;
    (dst_data + i + C4NUM)[0] = m01;
    (dst_data + i + dst_step * C4NUM)[0] = m10;
    (dst_data + i + dst_step * C4NUM + C4NUM)[0] = m11;
  }
#endif
}

void OutputTransform4x3Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                            int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t bias_ptr = vld1q_f32(bias_data);
  float32x4_t src_data_00 = vld1q_f32(src_data + 0 * src_step);
  float32x4_t src_data_01 = vld1q_f32(src_data + 1 * src_step);
  float32x4_t src_data_02 = vld1q_f32(src_data + 2 * src_step);
  float32x4_t src_data_03 = vld1q_f32(src_data + 3 * src_step);
  float32x4_t src_data_10 = vld1q_f32(src_data + 4 * src_step);
  float32x4_t src_data_11 = vld1q_f32(src_data + 5 * src_step);
  float32x4_t src_data_12 = vld1q_f32(src_data + 6 * src_step);
  float32x4_t src_data_13 = vld1q_f32(src_data + 7 * src_step);
  float32x4_t src_data_20 = vld1q_f32(src_data + 8 * src_step);
  float32x4_t src_data_21 = vld1q_f32(src_data + 9 * src_step);
  float32x4_t src_data_22 = vld1q_f32(src_data + 10 * src_step);
  float32x4_t src_data_23 = vld1q_f32(src_data + 11 * src_step);
  float32x4_t src_data_30 = vld1q_f32(src_data + 12 * src_step);
  float32x4_t src_data_31 = vld1q_f32(src_data + 13 * src_step);
  float32x4_t src_data_32 = vld1q_f32(src_data + 14 * src_step);
  float32x4_t src_data_33 = vld1q_f32(src_data + 15 * src_step);

  float32x4_t t00 = vaddq_f32(src_data_00, vaddq_f32(src_data_10, src_data_20));
  float32x4_t t01 = vaddq_f32(src_data_01, vaddq_f32(src_data_11, src_data_21));
  float32x4_t t02 = vaddq_f32(src_data_02, vaddq_f32(src_data_12, src_data_22));
  float32x4_t t03 = vaddq_f32(src_data_03, vaddq_f32(src_data_13, src_data_23));

  float32x4_t t10 = vmulq_n_f32(vsubq_f32(src_data_10, src_data_20), 0.5);
  float32x4_t t11 = vmulq_n_f32(vsubq_f32(src_data_11, src_data_21), 0.5);
  float32x4_t t12 = vmulq_n_f32(vsubq_f32(src_data_12, src_data_22), 0.5);
  float32x4_t t13 = vmulq_n_f32(vsubq_f32(src_data_13, src_data_23), 0.5);

  float32x4_t t20 = vaddq_f32(src_data_30, vmulq_n_f32(vaddq_f32(src_data_10, src_data_20), 0.25));
  float32x4_t t21 = vaddq_f32(src_data_31, vmulq_n_f32(vaddq_f32(src_data_11, src_data_21), 0.25));
  float32x4_t t22 = vaddq_f32(src_data_32, vmulq_n_f32(vaddq_f32(src_data_12, src_data_22), 0.25));
  float32x4_t t23 = vaddq_f32(src_data_33, vmulq_n_f32(vaddq_f32(src_data_13, src_data_23), 0.25));

  float32x4_t m00 = vaddq_f32(vaddq_f32(t00, vaddq_f32(t01, t02)), bias_ptr);
  float32x4_t m01 = vaddq_f32(vmulq_n_f32(vsubq_f32(t01, t02), 0.5), bias_ptr);
  float32x4_t m02 = vaddq_f32(vaddq_f32(t03, vmulq_n_f32(vaddq_f32(t01, t02), 0.25)), bias_ptr);
  float32x4_t m10 = vaddq_f32(vaddq_f32(t10, vaddq_f32(t11, t12)), bias_ptr);
  float32x4_t m11 = vaddq_f32(vmulq_n_f32(vsubq_f32(t11, t12), 0.5), bias_ptr);
  float32x4_t m12 = vaddq_f32(vaddq_f32(t13, vmulq_n_f32(vaddq_f32(t11, t12), 0.25)), bias_ptr);
  float32x4_t m20 = vaddq_f32(vaddq_f32(t20, vaddq_f32(t21, t22)), bias_ptr);
  float32x4_t m21 = vaddq_f32(vmulq_n_f32(vsubq_f32(t21, t22), 0.5), bias_ptr);
  float32x4_t m22 = vaddq_f32(vaddq_f32(t23, vmulq_n_f32(vaddq_f32(t21, t22), 0.25)), bias_ptr);

  vst1q_f32(dst_data, m00);
  vst1q_f32(dst_data + C4NUM, m01);
  vst1q_f32(dst_data + 2 * C4NUM, m02);
  vst1q_f32(dst_data + dst_step * C4NUM, m10);
  vst1q_f32(dst_data + dst_step * C4NUM + C4NUM, m11);
  vst1q_f32(dst_data + dst_step * C4NUM + 2 * C4NUM, m12);
  vst1q_f32(dst_data + 2 * dst_step * C4NUM, m20);
  vst1q_f32(dst_data + 2 * dst_step * C4NUM + C4NUM, m21);
  vst1q_f32(dst_data + 2 * dst_step * C4NUM + 2 * C4NUM, m22);
#else
  for (int i = 0; i < C4NUM; i++) {
    float src_data_00 = src_data[i];
    float src_data_01 = src_data[i + src_step];
    float src_data_02 = src_data[i + 2 * src_step];
    float src_data_03 = src_data[i + 3 * src_step];
    float src_data_10 = src_data[i + 4 * src_step];
    float src_data_11 = src_data[i + 5 * src_step];
    float src_data_12 = src_data[i + 6 * src_step];
    float src_data_13 = src_data[i + 7 * src_step];
    float src_data_20 = src_data[i + 8 * src_step];
    float src_data_21 = src_data[i + 9 * src_step];
    float src_data_22 = src_data[i + 10 * src_step];
    float src_data_23 = src_data[i + 11 * src_step];
    float src_data_30 = src_data[i + 12 * src_step];
    float src_data_31 = src_data[i + 13 * src_step];
    float src_data_32 = src_data[i + 14 * src_step];
    float src_data_33 = src_data[i + 15 * src_step];

    float t00 = src_data_00 + src_data_10 + src_data_20;
    float t01 = src_data_01 + src_data_11 + src_data_21;
    float t02 = src_data_02 + src_data_12 + src_data_22;
    float t03 = src_data_03 + src_data_13 + src_data_23;

    const float t10 = 0.5f * (src_data_10 - src_data_20);
    const float t11 = 0.5f * (src_data_11 - src_data_21);
    const float t12 = 0.5f * (src_data_12 - src_data_22);
    const const float t13 = 0.5f * (src_data_13 - src_data_23);

    const float t20 = 0.25f * (src_data_10 + src_data_20) + src_data_30;
    const float t21 = 0.25f * (src_data_11 + src_data_21) + src_data_31;
    const float t22 = 0.25f * (src_data_12 + src_data_22) + src_data_32;
    const float t23 = 0.25f * (src_data_13 + src_data_23) + src_data_33;

    float m00 = t00 + t01 + t02 + bias_data[i];
    const float m01 = 0.5f * (t01 - t02) + bias_data[i];
    const float m02 = 0.25f * (t01 + t02) + t03 + bias_data[i];

    float m10 = t10 + t11 + t12 + bias_data[i];
    const float m11 = 0.5f * (t11 - t12) + bias_data[i];
    const float m12 = 0.25f * (t11 + t12) + t13 + bias_data[i];

    float m20 = t20 + t21 + t22 + bias_data[i];
    const float m21 = 0.5f * (t21 - t22) + bias_data[i];
    const float m22 = 0.25f * (t21 + t22) + t23 + bias_data[i];

    (dst_data + i)[0] = m00;
    (dst_data + i + C4NUM)[0] = m01;
    (dst_data + i + 2 * C4NUM)[0] = m02;

    (dst_data + i + dst_step * C4NUM)[0] = m10;
    (dst_data + i + dst_step * C4NUM + C4NUM)[0] = m11;
    (dst_data + i + dst_step * C4NUM + 2 * C4NUM)[0] = m12;

    (dst_data + i + 2 * dst_step * C4NUM)[0] = m20;
    (dst_data + i + 2 * dst_step * C4NUM + C4NUM)[0] = m21;
    (dst_data + i + 2 * dst_step * C4NUM + 2 * C4NUM)[0] = m22;
  }
#endif
}

void OutputTransform8x2Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                            int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src_data_00 = vld1q_f32(src_data + 0 * src_step);
  float32x4_t src_data_01 = vld1q_f32(src_data + 1 * src_step);
  float32x4_t src_data_02 = vld1q_f32(src_data + 2 * src_step);
  float32x4_t src_data_03 = vld1q_f32(src_data + 3 * src_step);
  float32x4_t src_data_04 = vld1q_f32(src_data + 4 * src_step);
  float32x4_t src_data_05 = vld1q_f32(src_data + 5 * src_step);
  float32x4_t src_data_06 = vld1q_f32(src_data + 6 * src_step);
  float32x4_t src_data_07 = vld1q_f32(src_data + 7 * src_step);
  float32x4_t src_data_10 = vld1q_f32(src_data + 8 * src_step);
  float32x4_t src_data_11 = vld1q_f32(src_data + 9 * src_step);
  float32x4_t src_data_12 = vld1q_f32(src_data + 10 * src_step);
  float32x4_t src_data_13 = vld1q_f32(src_data + 11 * src_step);
  float32x4_t src_data_14 = vld1q_f32(src_data + 12 * src_step);
  float32x4_t src_data_15 = vld1q_f32(src_data + 13 * src_step);
  float32x4_t src_data_16 = vld1q_f32(src_data + 14 * src_step);
  float32x4_t src_data_17 = vld1q_f32(src_data + 15 * src_step);
  float32x4_t src_data_20 = vld1q_f32(src_data + 16 * src_step);
  float32x4_t src_data_21 = vld1q_f32(src_data + 17 * src_step);
  float32x4_t src_data_22 = vld1q_f32(src_data + 18 * src_step);
  float32x4_t src_data_23 = vld1q_f32(src_data + 19 * src_step);
  float32x4_t src_data_24 = vld1q_f32(src_data + 20 * src_step);
  float32x4_t src_data_25 = vld1q_f32(src_data + 21 * src_step);
  float32x4_t src_data_26 = vld1q_f32(src_data + 22 * src_step);
  float32x4_t src_data_27 = vld1q_f32(src_data + 23 * src_step);
  float32x4_t src_data_30 = vld1q_f32(src_data + 24 * src_step);
  float32x4_t src_data_31 = vld1q_f32(src_data + 25 * src_step);
  float32x4_t src_data_32 = vld1q_f32(src_data + 26 * src_step);
  float32x4_t src_data_33 = vld1q_f32(src_data + 27 * src_step);
  float32x4_t src_data_34 = vld1q_f32(src_data + 28 * src_step);
  float32x4_t src_data_35 = vld1q_f32(src_data + 29 * src_step);
  float32x4_t src_data_36 = vld1q_f32(src_data + 30 * src_step);
  float32x4_t src_data_37 = vld1q_f32(src_data + 31 * src_step);
  float32x4_t src_data_40 = vld1q_f32(src_data + 32 * src_step);
  float32x4_t src_data_41 = vld1q_f32(src_data + 33 * src_step);
  float32x4_t src_data_42 = vld1q_f32(src_data + 34 * src_step);
  float32x4_t src_data_43 = vld1q_f32(src_data + 35 * src_step);
  float32x4_t src_data_44 = vld1q_f32(src_data + 36 * src_step);
  float32x4_t src_data_45 = vld1q_f32(src_data + 37 * src_step);
  float32x4_t src_data_46 = vld1q_f32(src_data + 38 * src_step);
  float32x4_t src_data_47 = vld1q_f32(src_data + 39 * src_step);
  float32x4_t src_data_50 = vld1q_f32(src_data + 40 * src_step);
  float32x4_t src_data_51 = vld1q_f32(src_data + 41 * src_step);
  float32x4_t src_data_52 = vld1q_f32(src_data + 42 * src_step);
  float32x4_t src_data_53 = vld1q_f32(src_data + 43 * src_step);
  float32x4_t src_data_54 = vld1q_f32(src_data + 44 * src_step);
  float32x4_t src_data_55 = vld1q_f32(src_data + 45 * src_step);
  float32x4_t src_data_56 = vld1q_f32(src_data + 46 * src_step);
  float32x4_t src_data_57 = vld1q_f32(src_data + 47 * src_step);
  float32x4_t src_data_60 = vld1q_f32(src_data + 48 * src_step);
  float32x4_t src_data_61 = vld1q_f32(src_data + 49 * src_step);
  float32x4_t src_data_62 = vld1q_f32(src_data + 50 * src_step);
  float32x4_t src_data_63 = vld1q_f32(src_data + 51 * src_step);
  float32x4_t src_data_64 = vld1q_f32(src_data + 52 * src_step);
  float32x4_t src_data_65 = vld1q_f32(src_data + 53 * src_step);
  float32x4_t src_data_66 = vld1q_f32(src_data + 54 * src_step);
  float32x4_t src_data_67 = vld1q_f32(src_data + 55 * src_step);
  float32x4_t src_data_70 = vld1q_f32(src_data + 56 * src_step);
  float32x4_t src_data_71 = vld1q_f32(src_data + 57 * src_step);
  float32x4_t src_data_72 = vld1q_f32(src_data + 58 * src_step);
  float32x4_t src_data_73 = vld1q_f32(src_data + 59 * src_step);
  float32x4_t src_data_74 = vld1q_f32(src_data + 60 * src_step);
  float32x4_t src_data_75 = vld1q_f32(src_data + 61 * src_step);
  float32x4_t src_data_76 = vld1q_f32(src_data + 62 * src_step);
  float32x4_t src_data_77 = vld1q_f32(src_data + 63 * src_step);

  float32x4_t d01 = vsubq_f32(src_data_10, src_data_20);
  float32x4_t d02 = vsubq_f32(src_data_11, src_data_21);
  float32x4_t d03 = vsubq_f32(src_data_12, src_data_22);
  float32x4_t d04 = vsubq_f32(src_data_13, src_data_23);
  float32x4_t d05 = vsubq_f32(src_data_14, src_data_24);
  float32x4_t d06 = vsubq_f32(src_data_15, src_data_25);
  float32x4_t d07 = vsubq_f32(src_data_16, src_data_26);
  float32x4_t d08 = vsubq_f32(src_data_17, src_data_27);

  float32x4_t d11 = vsubq_f32(src_data_30, src_data_40);
  float32x4_t d12 = vsubq_f32(src_data_31, src_data_41);
  float32x4_t d13 = vsubq_f32(src_data_32, src_data_42);
  float32x4_t d14 = vsubq_f32(src_data_33, src_data_43);
  float32x4_t d15 = vsubq_f32(src_data_34, src_data_44);
  float32x4_t d16 = vsubq_f32(src_data_35, src_data_45);
  float32x4_t d17 = vsubq_f32(src_data_36, src_data_46);
  float32x4_t d18 = vsubq_f32(src_data_37, src_data_47);

  float32x4_t d21 = vsubq_f32(src_data_50, src_data_60);
  float32x4_t d22 = vsubq_f32(src_data_51, src_data_61);
  float32x4_t d23 = vsubq_f32(src_data_52, src_data_62);
  float32x4_t d24 = vsubq_f32(src_data_53, src_data_63);
  float32x4_t d25 = vsubq_f32(src_data_54, src_data_64);
  float32x4_t d26 = vsubq_f32(src_data_55, src_data_65);
  float32x4_t d27 = vsubq_f32(src_data_56, src_data_66);
  float32x4_t d28 = vsubq_f32(src_data_57, src_data_67);

  float32x4_t t00 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_00, src_data_10), src_data_20), src_data_30), src_data_40),
      src_data_50),
    src_data_60);
  float32x4_t t01 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_01, src_data_11), src_data_21), src_data_31), src_data_41),
      src_data_51),
    src_data_61);
  float32x4_t t02 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_02, src_data_12), src_data_22), src_data_32), src_data_42),
      src_data_52),
    src_data_62);
  float32x4_t t03 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_03, src_data_13), src_data_23), src_data_33), src_data_43),
      src_data_53),
    src_data_63);
  float32x4_t t04 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_04, src_data_14), src_data_24), src_data_34), src_data_44),
      src_data_54),
    src_data_64);
  float32x4_t t05 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_05, src_data_15), src_data_25), src_data_35), src_data_45),
      src_data_55),
    src_data_65);
  float32x4_t t06 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_06, src_data_16), src_data_26), src_data_36), src_data_46),
      src_data_56),
    src_data_66);
  float32x4_t t07 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_07, src_data_17), src_data_27), src_data_37), src_data_47),
      src_data_57),
    src_data_67);

  float32x4_t t10 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d01, 0.5), d11), vmulq_n_f32(d21, 1.5)), src_data_70);
  float32x4_t t11 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d02, 0.5), d12), vmulq_n_f32(d22, 1.5)), src_data_71);
  float32x4_t t12 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d03, 0.5), d13), vmulq_n_f32(d23, 1.5)), src_data_72);
  float32x4_t t13 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d04, 0.5), d14), vmulq_n_f32(d24, 1.5)), src_data_73);
  float32x4_t t14 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d05, 0.5), d15), vmulq_n_f32(d25, 1.5)), src_data_74);
  float32x4_t t15 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d06, 0.5), d16), vmulq_n_f32(d26, 1.5)), src_data_75);
  float32x4_t t16 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d07, 0.5), d17), vmulq_n_f32(d27, 1.5)), src_data_76);
  float32x4_t t17 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d08, 0.5), d18), vmulq_n_f32(d28, 1.5)), src_data_77);

  float32x4_t s11 = vsubq_f32(t01, t02);
  float32x4_t s12 = vsubq_f32(t11, t12);

  float32x4_t s21 = vsubq_f32(t03, t04);
  float32x4_t s22 = vsubq_f32(t13, t14);

  float32x4_t s31 = vsubq_f32(t05, t06);
  float32x4_t s32 = vsubq_f32(t15, t16);

  float32x4_t m00 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t00, t01), t02), t03), t04), t05), t06);
  float32x4_t m01 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s11, 0.5), s21), vmulq_n_f32(s31, 1.5)), t07);

  float32x4_t m10 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t10, t11), t12), t13), t14), t15), t16);
  float32x4_t m11 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s12, 0.5), s22), vmulq_n_f32(s32, 1.5)), t17);

  float32x4_t bias_ptr = vld1q_f32(bias_data);
  vst1q_f32(dst_data, vaddq_f32(m00, bias_ptr));
  vst1q_f32(dst_data + C4NUM, vaddq_f32(m01, bias_ptr));

  vst1q_f32(dst_data + dst_step * C4NUM, vaddq_f32(m10, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM + C4NUM, vaddq_f32(m11, bias_ptr));
#else
  for (int i = 0; i < C4NUM; i++) {
    float src_data_00 = src_data[i];
    float src_data_01 = src_data[i + src_step];
    float src_data_02 = src_data[i + 2 * src_step];
    float src_data_03 = src_data[i + 3 * src_step];
    float src_data_04 = src_data[i + 4 * src_step];
    float src_data_05 = src_data[i + 5 * src_step];
    float src_data_06 = src_data[i + 6 * src_step];
    float src_data_07 = src_data[i + 7 * src_step];
    float src_data_10 = src_data[i + 8 * src_step];
    float src_data_11 = src_data[i + 9 * src_step];
    float src_data_12 = src_data[i + 10 * src_step];
    float src_data_13 = src_data[i + 11 * src_step];
    float src_data_14 = src_data[i + 12 * src_step];
    float src_data_15 = src_data[i + 13 * src_step];
    float src_data_16 = src_data[i + 14 * src_step];
    float src_data_17 = src_data[i + 15 * src_step];
    float src_data_20 = src_data[i + 16 * src_step];
    float src_data_21 = src_data[i + 17 * src_step];
    float src_data_22 = src_data[i + 18 * src_step];
    float src_data_23 = src_data[i + 19 * src_step];
    float src_data_24 = src_data[i + 20 * src_step];
    float src_data_25 = src_data[i + 21 * src_step];
    float src_data_26 = src_data[i + 22 * src_step];
    float src_data_27 = src_data[i + 23 * src_step];
    float src_data_30 = src_data[i + 24 * src_step];
    float src_data_31 = src_data[i + 25 * src_step];
    float src_data_32 = src_data[i + 26 * src_step];
    float src_data_33 = src_data[i + 27 * src_step];
    float src_data_34 = src_data[i + 28 * src_step];
    float src_data_35 = src_data[i + 29 * src_step];
    float src_data_36 = src_data[i + 30 * src_step];
    float src_data_37 = src_data[i + 31 * src_step];
    float src_data_40 = src_data[i + 32 * src_step];
    float src_data_41 = src_data[i + 33 * src_step];
    float src_data_42 = src_data[i + 34 * src_step];
    float src_data_43 = src_data[i + 35 * src_step];
    float src_data_44 = src_data[i + 36 * src_step];
    float src_data_45 = src_data[i + 37 * src_step];
    float src_data_46 = src_data[i + 38 * src_step];
    float src_data_47 = src_data[i + 39 * src_step];
    float src_data_50 = src_data[i + 40 * src_step];
    float src_data_51 = src_data[i + 41 * src_step];
    float src_data_52 = src_data[i + 42 * src_step];
    float src_data_53 = src_data[i + 43 * src_step];
    float src_data_54 = src_data[i + 44 * src_step];
    float src_data_55 = src_data[i + 45 * src_step];
    float src_data_56 = src_data[i + 46 * src_step];
    float src_data_57 = src_data[i + 47 * src_step];
    float src_data_60 = src_data[i + 48 * src_step];
    float src_data_61 = src_data[i + 49 * src_step];
    float src_data_62 = src_data[i + 50 * src_step];
    float src_data_63 = src_data[i + 51 * src_step];
    float src_data_64 = src_data[i + 52 * src_step];
    float src_data_65 = src_data[i + 53 * src_step];
    float src_data_66 = src_data[i + 54 * src_step];
    float src_data_67 = src_data[i + 55 * src_step];
    float src_data_70 = src_data[i + 56 * src_step];
    float src_data_71 = src_data[i + 57 * src_step];
    float src_data_72 = src_data[i + 58 * src_step];
    float src_data_73 = src_data[i + 59 * src_step];
    float src_data_74 = src_data[i + 60 * src_step];
    float src_data_75 = src_data[i + 61 * src_step];
    float src_data_76 = src_data[i + 62 * src_step];
    float src_data_77 = src_data[i + 63 * src_step];

    float d01 = src_data_10 - src_data_20;
    float d02 = src_data_11 - src_data_21;
    float d03 = src_data_12 - src_data_22;
    float d04 = src_data_13 - src_data_23;
    float d05 = src_data_14 - src_data_24;
    float d06 = src_data_15 - src_data_25;
    float d07 = src_data_16 - src_data_26;
    float d08 = src_data_17 - src_data_27;

    float d11 = src_data_30 - src_data_40;
    float d12 = src_data_31 - src_data_41;
    float d13 = src_data_32 - src_data_42;
    float d14 = src_data_33 - src_data_43;
    float d15 = src_data_34 - src_data_44;
    float d16 = src_data_35 - src_data_45;
    float d17 = src_data_36 - src_data_46;
    float d18 = src_data_37 - src_data_47;

    float d21 = src_data_50 - src_data_60;
    float d22 = src_data_51 - src_data_61;
    float d23 = src_data_52 - src_data_62;
    float d24 = src_data_53 - src_data_63;
    float d25 = src_data_54 - src_data_64;
    float d26 = src_data_55 - src_data_65;
    float d27 = src_data_56 - src_data_66;
    float d28 = src_data_57 - src_data_67;

    float t00 = src_data_00 + src_data_10 + src_data_20 + src_data_30 + src_data_40 + src_data_50 + src_data_60;
    float t01 = src_data_01 + src_data_11 + src_data_21 + src_data_31 + src_data_41 + src_data_51 + src_data_61;
    float t02 = src_data_02 + src_data_12 + src_data_22 + src_data_32 + src_data_42 + src_data_52 + src_data_62;
    float t03 = src_data_03 + src_data_13 + src_data_23 + src_data_33 + src_data_43 + src_data_53 + src_data_63;
    float t04 = src_data_04 + src_data_14 + src_data_24 + src_data_34 + src_data_44 + src_data_54 + src_data_64;
    float t05 = src_data_05 + src_data_15 + src_data_25 + src_data_35 + src_data_45 + src_data_55 + src_data_65;
    float t06 = src_data_06 + src_data_16 + src_data_26 + src_data_36 + src_data_46 + src_data_56 + src_data_66;
    float t07 = src_data_07 + src_data_17 + src_data_27 + src_data_37 + src_data_47 + src_data_57 + src_data_67;

    const float t10 = 0.5f * d01 + d11 + 1.5f * d21 + src_data_70;
    const float t11 = 0.5f * d02 + d12 + 1.5f * d22 + src_data_71;
    const float t12 = 0.5f * d03 + d13 + 1.5f * d23 + src_data_72;
    const float t13 = 0.5f * d04 + d14 + 1.5f * d24 + src_data_73;
    const float t14 = 0.5f * d05 + d15 + 1.5f * d25 + src_data_74;
    const float t15 = 0.5f * d06 + d16 + 1.5f * d26 + src_data_75;
    const float t16 = 0.5f * d07 + d17 + 1.5f * d27 + src_data_76;
    const float t17 = 0.5f * d08 + d18 + 1.5f * d28 + src_data_77;

    float s11 = t01 - t02;
    float s12 = t11 - t12;
    float s21 = t03 - t04;
    float s22 = t13 - t14;
    float s31 = t05 - t06;
    float s32 = t15 - t16;

    float m00 = t00 + t01 + t02 + t03 + t04 + t05 + t06;
    const float m01 = 0.5f * s11 + s21 + 1.5f * s31 + t07;
    float m10 = t10 + t11 + t12 + t13 + t14 + t15 + t16;
    const float m11 = 0.5f * s12 + s22 + 1.5f * s32 + t17;

    (dst_data + i)[0] = m00 + bias_data[i];
    (dst_data + i + C4NUM)[0] = m01 + bias_data[i];
    (dst_data + i + dst_step * C4NUM)[0] = m10 + bias_data[i];
    (dst_data + i + dst_step * C4NUM + C4NUM)[0] = m11 + bias_data[i];
  }
#endif
}

void OutputTransform8x3Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                            int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src_data_00 = vld1q_f32(src_data + 0 * src_step);
  float32x4_t src_data_01 = vld1q_f32(src_data + 1 * src_step);
  float32x4_t src_data_02 = vld1q_f32(src_data + 2 * src_step);
  float32x4_t src_data_03 = vld1q_f32(src_data + 3 * src_step);
  float32x4_t src_data_04 = vld1q_f32(src_data + 4 * src_step);
  float32x4_t src_data_05 = vld1q_f32(src_data + 5 * src_step);
  float32x4_t src_data_06 = vld1q_f32(src_data + 6 * src_step);
  float32x4_t src_data_07 = vld1q_f32(src_data + 7 * src_step);
  float32x4_t src_data_10 = vld1q_f32(src_data + 8 * src_step);
  float32x4_t src_data_11 = vld1q_f32(src_data + 9 * src_step);
  float32x4_t src_data_12 = vld1q_f32(src_data + 10 * src_step);
  float32x4_t src_data_13 = vld1q_f32(src_data + 11 * src_step);
  float32x4_t src_data_14 = vld1q_f32(src_data + 12 * src_step);
  float32x4_t src_data_15 = vld1q_f32(src_data + 13 * src_step);
  float32x4_t src_data_16 = vld1q_f32(src_data + 14 * src_step);
  float32x4_t src_data_17 = vld1q_f32(src_data + 15 * src_step);
  float32x4_t src_data_20 = vld1q_f32(src_data + 16 * src_step);
  float32x4_t src_data_21 = vld1q_f32(src_data + 17 * src_step);
  float32x4_t src_data_22 = vld1q_f32(src_data + 18 * src_step);
  float32x4_t src_data_23 = vld1q_f32(src_data + 19 * src_step);
  float32x4_t src_data_24 = vld1q_f32(src_data + 20 * src_step);
  float32x4_t src_data_25 = vld1q_f32(src_data + 21 * src_step);
  float32x4_t src_data_26 = vld1q_f32(src_data + 22 * src_step);
  float32x4_t src_data_27 = vld1q_f32(src_data + 23 * src_step);
  float32x4_t src_data_30 = vld1q_f32(src_data + 24 * src_step);
  float32x4_t src_data_31 = vld1q_f32(src_data + 25 * src_step);
  float32x4_t src_data_32 = vld1q_f32(src_data + 26 * src_step);
  float32x4_t src_data_33 = vld1q_f32(src_data + 27 * src_step);
  float32x4_t src_data_34 = vld1q_f32(src_data + 28 * src_step);
  float32x4_t src_data_35 = vld1q_f32(src_data + 29 * src_step);
  float32x4_t src_data_36 = vld1q_f32(src_data + 30 * src_step);
  float32x4_t src_data_37 = vld1q_f32(src_data + 31 * src_step);
  float32x4_t src_data_40 = vld1q_f32(src_data + 32 * src_step);
  float32x4_t src_data_41 = vld1q_f32(src_data + 33 * src_step);
  float32x4_t src_data_42 = vld1q_f32(src_data + 34 * src_step);
  float32x4_t src_data_43 = vld1q_f32(src_data + 35 * src_step);
  float32x4_t src_data_44 = vld1q_f32(src_data + 36 * src_step);
  float32x4_t src_data_45 = vld1q_f32(src_data + 37 * src_step);
  float32x4_t src_data_46 = vld1q_f32(src_data + 38 * src_step);
  float32x4_t src_data_47 = vld1q_f32(src_data + 39 * src_step);
  float32x4_t src_data_50 = vld1q_f32(src_data + 40 * src_step);
  float32x4_t src_data_51 = vld1q_f32(src_data + 41 * src_step);
  float32x4_t src_data_52 = vld1q_f32(src_data + 42 * src_step);
  float32x4_t src_data_53 = vld1q_f32(src_data + 43 * src_step);
  float32x4_t src_data_54 = vld1q_f32(src_data + 44 * src_step);
  float32x4_t src_data_55 = vld1q_f32(src_data + 45 * src_step);
  float32x4_t src_data_56 = vld1q_f32(src_data + 46 * src_step);
  float32x4_t src_data_57 = vld1q_f32(src_data + 47 * src_step);
  float32x4_t src_data_60 = vld1q_f32(src_data + 48 * src_step);
  float32x4_t src_data_61 = vld1q_f32(src_data + 49 * src_step);
  float32x4_t src_data_62 = vld1q_f32(src_data + 50 * src_step);
  float32x4_t src_data_63 = vld1q_f32(src_data + 51 * src_step);
  float32x4_t src_data_64 = vld1q_f32(src_data + 52 * src_step);
  float32x4_t src_data_65 = vld1q_f32(src_data + 53 * src_step);
  float32x4_t src_data_66 = vld1q_f32(src_data + 54 * src_step);
  float32x4_t src_data_67 = vld1q_f32(src_data + 55 * src_step);
  float32x4_t src_data_70 = vld1q_f32(src_data + 56 * src_step);
  float32x4_t src_data_71 = vld1q_f32(src_data + 57 * src_step);
  float32x4_t src_data_72 = vld1q_f32(src_data + 58 * src_step);
  float32x4_t src_data_73 = vld1q_f32(src_data + 59 * src_step);
  float32x4_t src_data_74 = vld1q_f32(src_data + 60 * src_step);
  float32x4_t src_data_75 = vld1q_f32(src_data + 61 * src_step);
  float32x4_t src_data_76 = vld1q_f32(src_data + 62 * src_step);
  float32x4_t src_data_77 = vld1q_f32(src_data + 63 * src_step);

  float32x4_t d01 = vsubq_f32(src_data_10, src_data_20);
  float32x4_t d02 = vsubq_f32(src_data_11, src_data_21);
  float32x4_t d03 = vsubq_f32(src_data_12, src_data_22);
  float32x4_t d04 = vsubq_f32(src_data_13, src_data_23);
  float32x4_t d05 = vsubq_f32(src_data_14, src_data_24);
  float32x4_t d06 = vsubq_f32(src_data_15, src_data_25);
  float32x4_t d07 = vsubq_f32(src_data_16, src_data_26);
  float32x4_t d08 = vsubq_f32(src_data_17, src_data_27);

  float32x4_t d11 = vsubq_f32(src_data_30, src_data_40);
  float32x4_t d12 = vsubq_f32(src_data_31, src_data_41);
  float32x4_t d13 = vsubq_f32(src_data_32, src_data_42);
  float32x4_t d14 = vsubq_f32(src_data_33, src_data_43);
  float32x4_t d15 = vsubq_f32(src_data_34, src_data_44);
  float32x4_t d16 = vsubq_f32(src_data_35, src_data_45);
  float32x4_t d17 = vsubq_f32(src_data_36, src_data_46);
  float32x4_t d18 = vsubq_f32(src_data_37, src_data_47);

  float32x4_t d21 = vsubq_f32(src_data_50, src_data_60);
  float32x4_t d22 = vsubq_f32(src_data_51, src_data_61);
  float32x4_t d23 = vsubq_f32(src_data_52, src_data_62);
  float32x4_t d24 = vsubq_f32(src_data_53, src_data_63);
  float32x4_t d25 = vsubq_f32(src_data_54, src_data_64);
  float32x4_t d26 = vsubq_f32(src_data_55, src_data_65);
  float32x4_t d27 = vsubq_f32(src_data_56, src_data_66);
  float32x4_t d28 = vsubq_f32(src_data_57, src_data_67);

  float32x4_t d31 = vaddq_f32(src_data_10, src_data_20);
  float32x4_t d32 = vaddq_f32(src_data_11, src_data_21);
  float32x4_t d33 = vaddq_f32(src_data_12, src_data_22);
  float32x4_t d34 = vaddq_f32(src_data_13, src_data_23);
  float32x4_t d35 = vaddq_f32(src_data_14, src_data_24);
  float32x4_t d36 = vaddq_f32(src_data_15, src_data_25);
  float32x4_t d37 = vaddq_f32(src_data_16, src_data_26);
  float32x4_t d38 = vaddq_f32(src_data_17, src_data_27);

  float32x4_t d41 = vaddq_f32(src_data_30, src_data_40);
  float32x4_t d42 = vaddq_f32(src_data_31, src_data_41);
  float32x4_t d43 = vaddq_f32(src_data_32, src_data_42);
  float32x4_t d44 = vaddq_f32(src_data_33, src_data_43);
  float32x4_t d45 = vaddq_f32(src_data_34, src_data_44);
  float32x4_t d46 = vaddq_f32(src_data_35, src_data_45);
  float32x4_t d47 = vaddq_f32(src_data_36, src_data_46);
  float32x4_t d48 = vaddq_f32(src_data_37, src_data_47);

  float32x4_t d51 = vaddq_f32(src_data_50, src_data_60);
  float32x4_t d52 = vaddq_f32(src_data_51, src_data_61);
  float32x4_t d53 = vaddq_f32(src_data_52, src_data_62);
  float32x4_t d54 = vaddq_f32(src_data_53, src_data_63);
  float32x4_t d55 = vaddq_f32(src_data_54, src_data_64);
  float32x4_t d56 = vaddq_f32(src_data_55, src_data_65);
  float32x4_t d57 = vaddq_f32(src_data_56, src_data_66);
  float32x4_t d58 = vaddq_f32(src_data_57, src_data_67);

  float32x4_t t00 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_00, src_data_10), src_data_20), src_data_30), src_data_40),
      src_data_50),
    src_data_60);
  float32x4_t t01 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_01, src_data_11), src_data_21), src_data_31), src_data_41),
      src_data_51),
    src_data_61);
  float32x4_t t02 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_02, src_data_12), src_data_22), src_data_32), src_data_42),
      src_data_52),
    src_data_62);
  float32x4_t t03 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_03, src_data_13), src_data_23), src_data_33), src_data_43),
      src_data_53),
    src_data_63);
  float32x4_t t04 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_04, src_data_14), src_data_24), src_data_34), src_data_44),
      src_data_54),
    src_data_64);
  float32x4_t t05 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_05, src_data_15), src_data_25), src_data_35), src_data_45),
      src_data_55),
    src_data_65);
  float32x4_t t06 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_06, src_data_16), src_data_26), src_data_36), src_data_46),
      src_data_56),
    src_data_66);
  float32x4_t t07 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_07, src_data_17), src_data_27), src_data_37), src_data_47),
      src_data_57),
    src_data_67);

  float32x4_t t10 = vaddq_f32(vaddq_f32(vmulq_n_f32(d01, 0.5), d11), vmulq_n_f32(d21, 1.5));
  float32x4_t t11 = vaddq_f32(vaddq_f32(vmulq_n_f32(d02, 0.5), d12), vmulq_n_f32(d22, 1.5));
  float32x4_t t12 = vaddq_f32(vaddq_f32(vmulq_n_f32(d03, 0.5), d13), vmulq_n_f32(d23, 1.5));
  float32x4_t t13 = vaddq_f32(vaddq_f32(vmulq_n_f32(d04, 0.5), d14), vmulq_n_f32(d24, 1.5));
  float32x4_t t14 = vaddq_f32(vaddq_f32(vmulq_n_f32(d05, 0.5), d15), vmulq_n_f32(d25, 1.5));
  float32x4_t t15 = vaddq_f32(vaddq_f32(vmulq_n_f32(d06, 0.5), d16), vmulq_n_f32(d26, 1.5));
  float32x4_t t16 = vaddq_f32(vaddq_f32(vmulq_n_f32(d07, 0.5), d17), vmulq_n_f32(d27, 1.5));
  float32x4_t t17 = vaddq_f32(vaddq_f32(vmulq_n_f32(d08, 0.5), d18), vmulq_n_f32(d28, 1.5));

  float32x4_t t20 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d31, 0.25), d41), vmulq_n_f32(d51, 2.25)), src_data_70);
  float32x4_t t21 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d32, 0.25), d42), vmulq_n_f32(d52, 2.25)), src_data_71);
  float32x4_t t22 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d33, 0.25), d43), vmulq_n_f32(d53, 2.25)), src_data_72);
  float32x4_t t23 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d34, 0.25), d44), vmulq_n_f32(d54, 2.25)), src_data_73);
  float32x4_t t24 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d35, 0.25), d45), vmulq_n_f32(d55, 2.25)), src_data_74);
  float32x4_t t25 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d36, 0.25), d46), vmulq_n_f32(d56, 2.25)), src_data_75);
  float32x4_t t26 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d37, 0.25), d47), vmulq_n_f32(d57, 2.25)), src_data_76);
  float32x4_t t27 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d38, 0.25), d48), vmulq_n_f32(d58, 2.25)), src_data_77);

  float32x4_t s11 = vsubq_f32(t01, t02);
  float32x4_t s12 = vsubq_f32(t11, t12);
  float32x4_t s13 = vsubq_f32(t21, t22);

  float32x4_t s21 = vsubq_f32(t03, t04);
  float32x4_t s22 = vsubq_f32(t13, t14);
  float32x4_t s23 = vsubq_f32(t23, t24);

  float32x4_t s31 = vsubq_f32(t05, t06);
  float32x4_t s32 = vsubq_f32(t15, t16);
  float32x4_t s33 = vsubq_f32(t25, t26);

  float32x4_t s41 = vaddq_f32(t01, t02);
  float32x4_t s42 = vaddq_f32(t11, t12);
  float32x4_t s43 = vaddq_f32(t21, t22);

  float32x4_t s51 = vaddq_f32(t03, t04);
  float32x4_t s52 = vaddq_f32(t13, t14);
  float32x4_t s53 = vaddq_f32(t23, t24);

  float32x4_t s61 = vaddq_f32(t05, t06);
  float32x4_t s62 = vaddq_f32(t15, t16);
  float32x4_t s63 = vaddq_f32(t25, t26);

  float32x4_t m00 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t00, t01), t02), t03), t04), t05), t06);
  float32x4_t m01 = vaddq_f32(vaddq_f32(vmulq_n_f32(s11, 0.5), s21), vmulq_n_f32(s31, 1.5));
  float32x4_t m02 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s41, 0.25), s51), vmulq_n_f32(s61, 2.25)), t07);

  float32x4_t m10 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t10, t11), t12), t13), t14), t15), t16);
  float32x4_t m11 = vaddq_f32(vaddq_f32(vmulq_n_f32(s12, 0.5), s22), vmulq_n_f32(s32, 1.5));
  float32x4_t m12 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s42, 0.25), s52), vmulq_n_f32(s62, 2.25)), t17);

  float32x4_t m20 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t20, t21), t22), t23), t24), t25), t26);
  float32x4_t m21 = vaddq_f32(vaddq_f32(vmulq_n_f32(s13, 0.5), s23), vmulq_n_f32(s33, 1.5));
  float32x4_t m22 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s43, 0.25), s53), vmulq_n_f32(s63, 2.25)), t27);

  float32x4_t bias_ptr = vld1q_f32(bias_data);
  vst1q_f32(dst_data, vaddq_f32(m00, bias_ptr));
  vst1q_f32(dst_data + C4NUM, vaddq_f32(m01, bias_ptr));
  vst1q_f32(dst_data + 2 * C4NUM, vaddq_f32(m02, bias_ptr));

  vst1q_f32(dst_data + dst_step * C4NUM, vaddq_f32(m10, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM + C4NUM, vaddq_f32(m11, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM + 2 * C4NUM, vaddq_f32(m12, bias_ptr));

  vst1q_f32(dst_data + 2 * dst_step * C4NUM, vaddq_f32(m20, bias_ptr));
  vst1q_f32(dst_data + 2 * dst_step * C4NUM + C4NUM, vaddq_f32(m21, bias_ptr));
  vst1q_f32(dst_data + 2 * dst_step * C4NUM + 2 * C4NUM, vaddq_f32(m22, bias_ptr));
#else
  for (int i = 0; i < C4NUM; i++) {
    float src_data_00 = src_data[i];
    float src_data_01 = src_data[i + src_step];
    float src_data_02 = src_data[i + 2 * src_step];
    float src_data_03 = src_data[i + 3 * src_step];
    float src_data_04 = src_data[i + 4 * src_step];
    float src_data_05 = src_data[i + 5 * src_step];
    float src_data_06 = src_data[i + 6 * src_step];
    float src_data_07 = src_data[i + 7 * src_step];
    float src_data_10 = src_data[i + 8 * src_step];
    float src_data_11 = src_data[i + 9 * src_step];
    float src_data_12 = src_data[i + 10 * src_step];
    float src_data_13 = src_data[i + 11 * src_step];
    float src_data_14 = src_data[i + 12 * src_step];
    float src_data_15 = src_data[i + 13 * src_step];
    float src_data_16 = src_data[i + 14 * src_step];
    float src_data_17 = src_data[i + 15 * src_step];
    float src_data_20 = src_data[i + 16 * src_step];
    float src_data_21 = src_data[i + 17 * src_step];
    float src_data_22 = src_data[i + 18 * src_step];
    float src_data_23 = src_data[i + 19 * src_step];
    float src_data_24 = src_data[i + 20 * src_step];
    float src_data_25 = src_data[i + 21 * src_step];
    float src_data_26 = src_data[i + 22 * src_step];
    float src_data_27 = src_data[i + 23 * src_step];
    float src_data_30 = src_data[i + 24 * src_step];
    float src_data_31 = src_data[i + 25 * src_step];
    float src_data_32 = src_data[i + 26 * src_step];
    float src_data_33 = src_data[i + 27 * src_step];
    float src_data_34 = src_data[i + 28 * src_step];
    float src_data_35 = src_data[i + 29 * src_step];
    float src_data_36 = src_data[i + 30 * src_step];
    float src_data_37 = src_data[i + 31 * src_step];
    float src_data_40 = src_data[i + 32 * src_step];
    float src_data_41 = src_data[i + 33 * src_step];
    float src_data_42 = src_data[i + 34 * src_step];
    float src_data_43 = src_data[i + 35 * src_step];
    float src_data_44 = src_data[i + 36 * src_step];
    float src_data_45 = src_data[i + 37 * src_step];
    float src_data_46 = src_data[i + 38 * src_step];
    float src_data_47 = src_data[i + 39 * src_step];
    float src_data_50 = src_data[i + 40 * src_step];
    float src_data_51 = src_data[i + 41 * src_step];
    float src_data_52 = src_data[i + 42 * src_step];
    float src_data_53 = src_data[i + 43 * src_step];
    float src_data_54 = src_data[i + 44 * src_step];
    float src_data_55 = src_data[i + 45 * src_step];
    float src_data_56 = src_data[i + 46 * src_step];
    float src_data_57 = src_data[i + 47 * src_step];
    float src_data_60 = src_data[i + 48 * src_step];
    float src_data_61 = src_data[i + 49 * src_step];
    float src_data_62 = src_data[i + 50 * src_step];
    float src_data_63 = src_data[i + 51 * src_step];
    float src_data_64 = src_data[i + 52 * src_step];
    float src_data_65 = src_data[i + 53 * src_step];
    float src_data_66 = src_data[i + 54 * src_step];
    float src_data_67 = src_data[i + 55 * src_step];
    float src_data_70 = src_data[i + 56 * src_step];
    float src_data_71 = src_data[i + 57 * src_step];
    float src_data_72 = src_data[i + 58 * src_step];
    float src_data_73 = src_data[i + 59 * src_step];
    float src_data_74 = src_data[i + 60 * src_step];
    float src_data_75 = src_data[i + 61 * src_step];
    float src_data_76 = src_data[i + 62 * src_step];
    float src_data_77 = src_data[i + 63 * src_step];

    float d01 = src_data_10 - src_data_20;
    float d02 = src_data_11 - src_data_21;
    float d03 = src_data_12 - src_data_22;
    float d04 = src_data_13 - src_data_23;
    float d05 = src_data_14 - src_data_24;
    float d06 = src_data_15 - src_data_25;
    float d07 = src_data_16 - src_data_26;
    float d08 = src_data_17 - src_data_27;

    float d11 = src_data_30 - src_data_40;
    float d12 = src_data_31 - src_data_41;
    float d13 = src_data_32 - src_data_42;
    float d14 = src_data_33 - src_data_43;
    float d15 = src_data_34 - src_data_44;
    float d16 = src_data_35 - src_data_45;
    float d17 = src_data_36 - src_data_46;
    float d18 = src_data_37 - src_data_47;

    float d21 = src_data_50 - src_data_60;
    float d22 = src_data_51 - src_data_61;
    float d23 = src_data_52 - src_data_62;
    float d24 = src_data_53 - src_data_63;
    float d25 = src_data_54 - src_data_64;
    float d26 = src_data_55 - src_data_65;
    float d27 = src_data_56 - src_data_66;
    float d28 = src_data_57 - src_data_67;

    float d31 = src_data_10 + src_data_20;
    float d32 = src_data_11 + src_data_21;
    float d33 = src_data_12 + src_data_22;
    float d34 = src_data_13 + src_data_23;
    float d35 = src_data_14 + src_data_24;
    float d36 = src_data_15 + src_data_25;
    float d37 = src_data_16 + src_data_26;
    float d38 = src_data_17 + src_data_27;

    float d41 = src_data_30 + src_data_40;
    float d42 = src_data_31 + src_data_41;
    float d43 = src_data_32 + src_data_42;
    float d44 = src_data_33 + src_data_43;
    float d45 = src_data_34 + src_data_44;
    float d46 = src_data_35 + src_data_45;
    float d47 = src_data_36 + src_data_46;
    float d48 = src_data_37 + src_data_47;

    float d51 = src_data_50 + src_data_60;
    float d52 = src_data_51 + src_data_61;
    float d53 = src_data_52 + src_data_62;
    float d54 = src_data_53 + src_data_63;
    float d55 = src_data_54 + src_data_64;
    float d56 = src_data_55 + src_data_65;
    float d57 = src_data_56 + src_data_66;
    float d58 = src_data_57 + src_data_67;

    float t00 = src_data_00 + src_data_10 + src_data_20 + src_data_30 + src_data_40 + src_data_50 + src_data_60;
    float t01 = src_data_01 + src_data_11 + src_data_21 + src_data_31 + src_data_41 + src_data_51 + src_data_61;
    float t02 = src_data_02 + src_data_12 + src_data_22 + src_data_32 + src_data_42 + src_data_52 + src_data_62;
    float t03 = src_data_03 + src_data_13 + src_data_23 + src_data_33 + src_data_43 + src_data_53 + src_data_63;
    float t04 = src_data_04 + src_data_14 + src_data_24 + src_data_34 + src_data_44 + src_data_54 + src_data_64;
    float t05 = src_data_05 + src_data_15 + src_data_25 + src_data_35 + src_data_45 + src_data_55 + src_data_65;
    float t06 = src_data_06 + src_data_16 + src_data_26 + src_data_36 + src_data_46 + src_data_56 + src_data_66;
    float t07 = src_data_07 + src_data_17 + src_data_27 + src_data_37 + src_data_47 + src_data_57 + src_data_67;

    const float t10 = 0.5f * d01 + d11 + 1.5f * d21;
    const float t11 = 0.5f * d02 + d12 + 1.5f * d22;
    const float t12 = 0.5f * d03 + d13 + 1.5f * d23;
    const float t13 = 0.5f * d04 + d14 + 1.5f * d24;
    const float t14 = 0.5f * d05 + d15 + 1.5f * d25;
    const float t15 = 0.5f * d06 + d16 + 1.5f * d26;
    const float t16 = 0.5f * d07 + d17 + 1.5f * d27;
    const float t17 = 0.5f * d08 + d18 + 1.5f * d28;

    const float t20 = 0.25f * d31 + d41 + 2.25f * d51 + src_data_70;
    const float t21 = 0.25f * d32 + d42 + 2.25f * d52 + src_data_71;
    const float t22 = 0.25f * d33 + d43 + 2.25f * d53 + src_data_72;
    const float t23 = 0.25f * d34 + d44 + 2.25f * d54 + src_data_73;
    const float t24 = 0.25f * d35 + d45 + 2.25f * d55 + src_data_74;
    const float t25 = 0.25f * d36 + d46 + 2.25f * d56 + src_data_75;
    const float t26 = 0.25f * d37 + d47 + 2.25f * d57 + src_data_76;
    const float t27 = 0.25f * d38 + d48 + 2.25f * d58 + src_data_77;

    float s11 = t01 - t02;
    float s12 = t11 - t12;
    float s13 = t21 - t22;

    float s21 = t03 - t04;
    float s22 = t13 - t14;
    float s23 = t23 - t24;

    float s31 = t05 - t06;
    float s32 = t15 - t16;
    float s33 = t25 - t26;

    float s41 = t01 + t02;
    float s42 = t11 + t12;
    float s43 = t21 + t22;

    float s51 = t03 + t04;
    float s52 = t13 + t14;
    float s53 = t23 + t24;

    float s61 = t05 + t06;
    float s62 = t15 + t16;
    float s63 = t25 + t26;

    float m00 = t00 + t01 + t02 + t03 + t04 + t05 + t06;
    const float m01 = 0.5f * s11 + s21 + 1.5f * s31;
    const float m02 = 0.25f * s41 + s51 + 2.25f * s61 + t07;

    float m10 = t10 + t11 + t12 + t13 + t14 + t15 + t16;
    const float m11 = 0.5f * s12 + s22 + 1.5f * s32;
    const float m12 = 0.25f * s42 + s52 + 2.25f * s62 + t17;

    float m20 = t20 + t21 + t22 + t23 + t24 + t25 + t26;
    const float m21 = 0.5f * s13 + s23 + 1.5f * s33;
    const float m22 = 0.25f * s43 + s53 + 2.25f * s63 + t27;

    (dst_data + i)[0] = m00 + bias_data[i];
    (dst_data + i + C4NUM)[0] = m01 + bias_data[i];
    (dst_data + i + 2 * C4NUM)[0] = m02 + bias_data[i];

    (dst_data + i + dst_step * C4NUM)[0] = m10 + bias_data[i];
    (dst_data + i + dst_step * C4NUM + C4NUM)[0] = m11 + bias_data[i];
    (dst_data + i + dst_step * C4NUM + 2 * C4NUM)[0] = m12 + bias_data[i];

    (dst_data + i + 2 * dst_step * C4NUM)[0] = m20 + bias_data[i];
    (dst_data + i + 2 * dst_step * C4NUM + C4NUM)[0] = m21 + bias_data[i];
    (dst_data + i + 2 * dst_step * C4NUM + 2 * C4NUM)[0] = m22 + bias_data[i];
  }
#endif
}

void OutputTransform8x4Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                            int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src_data_00 = vld1q_f32(src_data + 0 * src_step);
  float32x4_t src_data_01 = vld1q_f32(src_data + 1 * src_step);
  float32x4_t src_data_02 = vld1q_f32(src_data + 2 * src_step);
  float32x4_t src_data_03 = vld1q_f32(src_data + 3 * src_step);
  float32x4_t src_data_04 = vld1q_f32(src_data + 4 * src_step);
  float32x4_t src_data_05 = vld1q_f32(src_data + 5 * src_step);
  float32x4_t src_data_06 = vld1q_f32(src_data + 6 * src_step);
  float32x4_t src_data_07 = vld1q_f32(src_data + 7 * src_step);
  float32x4_t src_data_10 = vld1q_f32(src_data + 8 * src_step);
  float32x4_t src_data_11 = vld1q_f32(src_data + 9 * src_step);
  float32x4_t src_data_12 = vld1q_f32(src_data + 10 * src_step);
  float32x4_t src_data_13 = vld1q_f32(src_data + 11 * src_step);
  float32x4_t src_data_14 = vld1q_f32(src_data + 12 * src_step);
  float32x4_t src_data_15 = vld1q_f32(src_data + 13 * src_step);
  float32x4_t src_data_16 = vld1q_f32(src_data + 14 * src_step);
  float32x4_t src_data_17 = vld1q_f32(src_data + 15 * src_step);
  float32x4_t src_data_20 = vld1q_f32(src_data + 16 * src_step);
  float32x4_t src_data_21 = vld1q_f32(src_data + 17 * src_step);
  float32x4_t src_data_22 = vld1q_f32(src_data + 18 * src_step);
  float32x4_t src_data_23 = vld1q_f32(src_data + 19 * src_step);
  float32x4_t src_data_24 = vld1q_f32(src_data + 20 * src_step);
  float32x4_t src_data_25 = vld1q_f32(src_data + 21 * src_step);
  float32x4_t src_data_26 = vld1q_f32(src_data + 22 * src_step);
  float32x4_t src_data_27 = vld1q_f32(src_data + 23 * src_step);
  float32x4_t src_data_30 = vld1q_f32(src_data + 24 * src_step);
  float32x4_t src_data_31 = vld1q_f32(src_data + 25 * src_step);
  float32x4_t src_data_32 = vld1q_f32(src_data + 26 * src_step);
  float32x4_t src_data_33 = vld1q_f32(src_data + 27 * src_step);
  float32x4_t src_data_34 = vld1q_f32(src_data + 28 * src_step);
  float32x4_t src_data_35 = vld1q_f32(src_data + 29 * src_step);
  float32x4_t src_data_36 = vld1q_f32(src_data + 30 * src_step);
  float32x4_t src_data_37 = vld1q_f32(src_data + 31 * src_step);
  float32x4_t src_data_40 = vld1q_f32(src_data + 32 * src_step);
  float32x4_t src_data_41 = vld1q_f32(src_data + 33 * src_step);
  float32x4_t src_data_42 = vld1q_f32(src_data + 34 * src_step);
  float32x4_t src_data_43 = vld1q_f32(src_data + 35 * src_step);
  float32x4_t src_data_44 = vld1q_f32(src_data + 36 * src_step);
  float32x4_t src_data_45 = vld1q_f32(src_data + 37 * src_step);
  float32x4_t src_data_46 = vld1q_f32(src_data + 38 * src_step);
  float32x4_t src_data_47 = vld1q_f32(src_data + 39 * src_step);
  float32x4_t src_data_50 = vld1q_f32(src_data + 40 * src_step);
  float32x4_t src_data_51 = vld1q_f32(src_data + 41 * src_step);
  float32x4_t src_data_52 = vld1q_f32(src_data + 42 * src_step);
  float32x4_t src_data_53 = vld1q_f32(src_data + 43 * src_step);
  float32x4_t src_data_54 = vld1q_f32(src_data + 44 * src_step);
  float32x4_t src_data_55 = vld1q_f32(src_data + 45 * src_step);
  float32x4_t src_data_56 = vld1q_f32(src_data + 46 * src_step);
  float32x4_t src_data_57 = vld1q_f32(src_data + 47 * src_step);
  float32x4_t src_data_60 = vld1q_f32(src_data + 48 * src_step);
  float32x4_t src_data_61 = vld1q_f32(src_data + 49 * src_step);
  float32x4_t src_data_62 = vld1q_f32(src_data + 50 * src_step);
  float32x4_t src_data_63 = vld1q_f32(src_data + 51 * src_step);
  float32x4_t src_data_64 = vld1q_f32(src_data + 52 * src_step);
  float32x4_t src_data_65 = vld1q_f32(src_data + 53 * src_step);
  float32x4_t src_data_66 = vld1q_f32(src_data + 54 * src_step);
  float32x4_t src_data_67 = vld1q_f32(src_data + 55 * src_step);
  float32x4_t src_data_70 = vld1q_f32(src_data + 56 * src_step);
  float32x4_t src_data_71 = vld1q_f32(src_data + 57 * src_step);
  float32x4_t src_data_72 = vld1q_f32(src_data + 58 * src_step);
  float32x4_t src_data_73 = vld1q_f32(src_data + 59 * src_step);
  float32x4_t src_data_74 = vld1q_f32(src_data + 60 * src_step);
  float32x4_t src_data_75 = vld1q_f32(src_data + 61 * src_step);
  float32x4_t src_data_76 = vld1q_f32(src_data + 62 * src_step);
  float32x4_t src_data_77 = vld1q_f32(src_data + 63 * src_step);

  float32x4_t d01 = vsubq_f32(src_data_10, src_data_20);
  float32x4_t d02 = vsubq_f32(src_data_11, src_data_21);
  float32x4_t d03 = vsubq_f32(src_data_12, src_data_22);
  float32x4_t d04 = vsubq_f32(src_data_13, src_data_23);
  float32x4_t d05 = vsubq_f32(src_data_14, src_data_24);
  float32x4_t d06 = vsubq_f32(src_data_15, src_data_25);
  float32x4_t d07 = vsubq_f32(src_data_16, src_data_26);
  float32x4_t d08 = vsubq_f32(src_data_17, src_data_27);

  float32x4_t d11 = vsubq_f32(src_data_30, src_data_40);
  float32x4_t d12 = vsubq_f32(src_data_31, src_data_41);
  float32x4_t d13 = vsubq_f32(src_data_32, src_data_42);
  float32x4_t d14 = vsubq_f32(src_data_33, src_data_43);
  float32x4_t d15 = vsubq_f32(src_data_34, src_data_44);
  float32x4_t d16 = vsubq_f32(src_data_35, src_data_45);
  float32x4_t d17 = vsubq_f32(src_data_36, src_data_46);
  float32x4_t d18 = vsubq_f32(src_data_37, src_data_47);

  float32x4_t d21 = vsubq_f32(src_data_50, src_data_60);
  float32x4_t d22 = vsubq_f32(src_data_51, src_data_61);
  float32x4_t d23 = vsubq_f32(src_data_52, src_data_62);
  float32x4_t d24 = vsubq_f32(src_data_53, src_data_63);
  float32x4_t d25 = vsubq_f32(src_data_54, src_data_64);
  float32x4_t d26 = vsubq_f32(src_data_55, src_data_65);
  float32x4_t d27 = vsubq_f32(src_data_56, src_data_66);
  float32x4_t d28 = vsubq_f32(src_data_57, src_data_67);

  float32x4_t d31 = vaddq_f32(src_data_10, src_data_20);
  float32x4_t d32 = vaddq_f32(src_data_11, src_data_21);
  float32x4_t d33 = vaddq_f32(src_data_12, src_data_22);
  float32x4_t d34 = vaddq_f32(src_data_13, src_data_23);
  float32x4_t d35 = vaddq_f32(src_data_14, src_data_24);
  float32x4_t d36 = vaddq_f32(src_data_15, src_data_25);
  float32x4_t d37 = vaddq_f32(src_data_16, src_data_26);
  float32x4_t d38 = vaddq_f32(src_data_17, src_data_27);

  float32x4_t d41 = vaddq_f32(src_data_30, src_data_40);
  float32x4_t d42 = vaddq_f32(src_data_31, src_data_41);
  float32x4_t d43 = vaddq_f32(src_data_32, src_data_42);
  float32x4_t d44 = vaddq_f32(src_data_33, src_data_43);
  float32x4_t d45 = vaddq_f32(src_data_34, src_data_44);
  float32x4_t d46 = vaddq_f32(src_data_35, src_data_45);
  float32x4_t d47 = vaddq_f32(src_data_36, src_data_46);
  float32x4_t d48 = vaddq_f32(src_data_37, src_data_47);

  float32x4_t d51 = vaddq_f32(src_data_50, src_data_60);
  float32x4_t d52 = vaddq_f32(src_data_51, src_data_61);
  float32x4_t d53 = vaddq_f32(src_data_52, src_data_62);
  float32x4_t d54 = vaddq_f32(src_data_53, src_data_63);
  float32x4_t d55 = vaddq_f32(src_data_54, src_data_64);
  float32x4_t d56 = vaddq_f32(src_data_55, src_data_65);
  float32x4_t d57 = vaddq_f32(src_data_56, src_data_66);
  float32x4_t d58 = vaddq_f32(src_data_57, src_data_67);

  float32x4_t t00 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_00, src_data_10), src_data_20), src_data_30), src_data_40),
      src_data_50),
    src_data_60);
  float32x4_t t01 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_01, src_data_11), src_data_21), src_data_31), src_data_41),
      src_data_51),
    src_data_61);
  float32x4_t t02 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_02, src_data_12), src_data_22), src_data_32), src_data_42),
      src_data_52),
    src_data_62);
  float32x4_t t03 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_03, src_data_13), src_data_23), src_data_33), src_data_43),
      src_data_53),
    src_data_63);
  float32x4_t t04 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_04, src_data_14), src_data_24), src_data_34), src_data_44),
      src_data_54),
    src_data_64);
  float32x4_t t05 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_05, src_data_15), src_data_25), src_data_35), src_data_45),
      src_data_55),
    src_data_65);
  float32x4_t t06 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_06, src_data_16), src_data_26), src_data_36), src_data_46),
      src_data_56),
    src_data_66);
  float32x4_t t07 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_07, src_data_17), src_data_27), src_data_37), src_data_47),
      src_data_57),
    src_data_67);

  float32x4_t t10 = vaddq_f32(vaddq_f32(vmulq_n_f32(d01, 0.5), d11), vmulq_n_f32(d21, 1.5));
  float32x4_t t11 = vaddq_f32(vaddq_f32(vmulq_n_f32(d02, 0.5), d12), vmulq_n_f32(d22, 1.5));
  float32x4_t t12 = vaddq_f32(vaddq_f32(vmulq_n_f32(d03, 0.5), d13), vmulq_n_f32(d23, 1.5));
  float32x4_t t13 = vaddq_f32(vaddq_f32(vmulq_n_f32(d04, 0.5), d14), vmulq_n_f32(d24, 1.5));
  float32x4_t t14 = vaddq_f32(vaddq_f32(vmulq_n_f32(d05, 0.5), d15), vmulq_n_f32(d25, 1.5));
  float32x4_t t15 = vaddq_f32(vaddq_f32(vmulq_n_f32(d06, 0.5), d16), vmulq_n_f32(d26, 1.5));
  float32x4_t t16 = vaddq_f32(vaddq_f32(vmulq_n_f32(d07, 0.5), d17), vmulq_n_f32(d27, 1.5));
  float32x4_t t17 = vaddq_f32(vaddq_f32(vmulq_n_f32(d08, 0.5), d18), vmulq_n_f32(d28, 1.5));

  float32x4_t t20 = vaddq_f32(vaddq_f32(vmulq_n_f32(d31, 0.25), d41), vmulq_n_f32(d51, 2.25));
  float32x4_t t21 = vaddq_f32(vaddq_f32(vmulq_n_f32(d32, 0.25), d42), vmulq_n_f32(d52, 2.25));
  float32x4_t t22 = vaddq_f32(vaddq_f32(vmulq_n_f32(d33, 0.25), d43), vmulq_n_f32(d53, 2.25));
  float32x4_t t23 = vaddq_f32(vaddq_f32(vmulq_n_f32(d34, 0.25), d44), vmulq_n_f32(d54, 2.25));
  float32x4_t t24 = vaddq_f32(vaddq_f32(vmulq_n_f32(d35, 0.25), d45), vmulq_n_f32(d55, 2.25));
  float32x4_t t25 = vaddq_f32(vaddq_f32(vmulq_n_f32(d36, 0.25), d46), vmulq_n_f32(d56, 2.25));
  float32x4_t t26 = vaddq_f32(vaddq_f32(vmulq_n_f32(d37, 0.25), d47), vmulq_n_f32(d57, 2.25));
  float32x4_t t27 = vaddq_f32(vaddq_f32(vmulq_n_f32(d38, 0.25), d48), vmulq_n_f32(d58, 2.25));

  float32x4_t t30 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d01, 0.125), d11), vmulq_n_f32(d21, 3.375)), src_data_70);
  float32x4_t t31 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d02, 0.125), d12), vmulq_n_f32(d22, 3.375)), src_data_71);
  float32x4_t t32 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d03, 0.125), d13), vmulq_n_f32(d23, 3.375)), src_data_72);
  float32x4_t t33 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d04, 0.125), d14), vmulq_n_f32(d24, 3.375)), src_data_73);
  float32x4_t t34 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d05, 0.125), d15), vmulq_n_f32(d25, 3.375)), src_data_74);
  float32x4_t t35 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d06, 0.125), d16), vmulq_n_f32(d26, 3.375)), src_data_75);
  float32x4_t t36 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d07, 0.125), d17), vmulq_n_f32(d27, 3.375)), src_data_76);
  float32x4_t t37 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d08, 0.125), d18), vmulq_n_f32(d28, 3.375)), src_data_77);

  float32x4_t s11 = vsubq_f32(t01, t02);
  float32x4_t s12 = vsubq_f32(t11, t12);
  float32x4_t s13 = vsubq_f32(t21, t22);
  float32x4_t s14 = vsubq_f32(t31, t32);

  float32x4_t s21 = vsubq_f32(t03, t04);
  float32x4_t s22 = vsubq_f32(t13, t14);
  float32x4_t s23 = vsubq_f32(t23, t24);
  float32x4_t s24 = vsubq_f32(t33, t34);

  float32x4_t s31 = vsubq_f32(t05, t06);
  float32x4_t s32 = vsubq_f32(t15, t16);
  float32x4_t s33 = vsubq_f32(t25, t26);
  float32x4_t s34 = vsubq_f32(t35, t36);

  float32x4_t s41 = vaddq_f32(t01, t02);
  float32x4_t s42 = vaddq_f32(t11, t12);
  float32x4_t s43 = vaddq_f32(t21, t22);
  float32x4_t s44 = vaddq_f32(t31, t32);

  float32x4_t s51 = vaddq_f32(t03, t04);
  float32x4_t s52 = vaddq_f32(t13, t14);
  float32x4_t s53 = vaddq_f32(t23, t24);
  float32x4_t s54 = vaddq_f32(t33, t34);

  float32x4_t s61 = vaddq_f32(t05, t06);
  float32x4_t s62 = vaddq_f32(t15, t16);
  float32x4_t s63 = vaddq_f32(t25, t26);
  float32x4_t s64 = vaddq_f32(t35, t36);

  float32x4_t m00 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t00, t01), t02), t03), t04), t05), t06);
  float32x4_t m01 = vaddq_f32(vaddq_f32(vmulq_n_f32(s11, 0.5), s21), vmulq_n_f32(s31, 1.5));
  float32x4_t m02 = vaddq_f32(vaddq_f32(vmulq_n_f32(s41, 0.25), s51), vmulq_n_f32(s61, 2.25));
  float32x4_t m03 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s11, 0.125), s21), vmulq_n_f32(s31, 3.375)), t07);

  float32x4_t m10 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t10, t11), t12), t13), t14), t15), t16);
  float32x4_t m11 = vaddq_f32(vaddq_f32(vmulq_n_f32(s12, 0.5), s22), vmulq_n_f32(s32, 1.5));
  float32x4_t m12 = vaddq_f32(vaddq_f32(vmulq_n_f32(s42, 0.25), s52), vmulq_n_f32(s62, 2.25));
  float32x4_t m13 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s12, 0.125), s22), vmulq_n_f32(s32, 3.375)), t17);

  float32x4_t m20 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t20, t21), t22), t23), t24), t25), t26);
  float32x4_t m21 = vaddq_f32(vaddq_f32(vmulq_n_f32(s13, 0.5), s23), vmulq_n_f32(s33, 1.5));
  float32x4_t m22 = vaddq_f32(vaddq_f32(vmulq_n_f32(s43, 0.25), s53), vmulq_n_f32(s63, 2.25));
  float32x4_t m23 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s13, 0.125), s23), vmulq_n_f32(s33, 3.375)), t27);

  float32x4_t m30 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t30, t31), t32), t33), t34), t35), t36);
  float32x4_t m31 = vaddq_f32(vaddq_f32(vmulq_n_f32(s14, 0.5), s24), vmulq_n_f32(s34, 1.5));
  float32x4_t m32 = vaddq_f32(vaddq_f32(vmulq_n_f32(s44, 0.25), s54), vmulq_n_f32(s64, 2.25));
  float32x4_t m33 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s14, 0.125), s24), vmulq_n_f32(s34, 3.375)), t37);

  float32x4_t bias_ptr = vld1q_f32(bias_data);
  vst1q_f32(dst_data, vaddq_f32(m00, bias_ptr));
  vst1q_f32(dst_data + C4NUM, vaddq_f32(m01, bias_ptr));
  vst1q_f32(dst_data + 2 * C4NUM, vaddq_f32(m02, bias_ptr));
  vst1q_f32(dst_data + 3 * C4NUM, vaddq_f32(m03, bias_ptr));

  vst1q_f32(dst_data + dst_step * C4NUM, vaddq_f32(m10, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM + C4NUM, vaddq_f32(m11, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM + 2 * C4NUM, vaddq_f32(m12, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM + 3 * C4NUM, vaddq_f32(m13, bias_ptr));

  vst1q_f32(dst_data + 2 * dst_step * C4NUM, vaddq_f32(m20, bias_ptr));
  vst1q_f32(dst_data + 2 * dst_step * C4NUM + C4NUM, vaddq_f32(m21, bias_ptr));
  vst1q_f32(dst_data + 2 * dst_step * C4NUM + 2 * C4NUM, vaddq_f32(m22, bias_ptr));
  vst1q_f32(dst_data + 2 * dst_step * C4NUM + 3 * C4NUM, vaddq_f32(m23, bias_ptr));

  vst1q_f32(dst_data + 3 * dst_step * C4NUM, vaddq_f32(m30, bias_ptr));
  vst1q_f32(dst_data + 3 * dst_step * C4NUM + C4NUM, vaddq_f32(m31, bias_ptr));
  vst1q_f32(dst_data + 3 * dst_step * C4NUM + 2 * C4NUM, vaddq_f32(m32, bias_ptr));
  vst1q_f32(dst_data + 3 * dst_step * C4NUM + 3 * C4NUM, vaddq_f32(m33, bias_ptr));
#else
  for (int i = 0; i < C4NUM; i++) {
    float src_data_00 = src_data[i];
    float src_data_01 = src_data[i + src_step];
    float src_data_02 = src_data[i + 2 * src_step];
    float src_data_03 = src_data[i + 3 * src_step];
    float src_data_04 = src_data[i + 4 * src_step];
    float src_data_05 = src_data[i + 5 * src_step];
    float src_data_06 = src_data[i + 6 * src_step];
    float src_data_07 = src_data[i + 7 * src_step];
    float src_data_10 = src_data[i + 8 * src_step];
    float src_data_11 = src_data[i + 9 * src_step];
    float src_data_12 = src_data[i + 10 * src_step];
    float src_data_13 = src_data[i + 11 * src_step];
    float src_data_14 = src_data[i + 12 * src_step];
    float src_data_15 = src_data[i + 13 * src_step];
    float src_data_16 = src_data[i + 14 * src_step];
    float src_data_17 = src_data[i + 15 * src_step];
    float src_data_20 = src_data[i + 16 * src_step];
    float src_data_21 = src_data[i + 17 * src_step];
    float src_data_22 = src_data[i + 18 * src_step];
    float src_data_23 = src_data[i + 19 * src_step];
    float src_data_24 = src_data[i + 20 * src_step];
    float src_data_25 = src_data[i + 21 * src_step];
    float src_data_26 = src_data[i + 22 * src_step];
    float src_data_27 = src_data[i + 23 * src_step];
    float src_data_30 = src_data[i + 24 * src_step];
    float src_data_31 = src_data[i + 25 * src_step];
    float src_data_32 = src_data[i + 26 * src_step];
    float src_data_33 = src_data[i + 27 * src_step];
    float src_data_34 = src_data[i + 28 * src_step];
    float src_data_35 = src_data[i + 29 * src_step];
    float src_data_36 = src_data[i + 30 * src_step];
    float src_data_37 = src_data[i + 31 * src_step];
    float src_data_40 = src_data[i + 32 * src_step];
    float src_data_41 = src_data[i + 33 * src_step];
    float src_data_42 = src_data[i + 34 * src_step];
    float src_data_43 = src_data[i + 35 * src_step];
    float src_data_44 = src_data[i + 36 * src_step];
    float src_data_45 = src_data[i + 37 * src_step];
    float src_data_46 = src_data[i + 38 * src_step];
    float src_data_47 = src_data[i + 39 * src_step];
    float src_data_50 = src_data[i + 40 * src_step];
    float src_data_51 = src_data[i + 41 * src_step];
    float src_data_52 = src_data[i + 42 * src_step];
    float src_data_53 = src_data[i + 43 * src_step];
    float src_data_54 = src_data[i + 44 * src_step];
    float src_data_55 = src_data[i + 45 * src_step];
    float src_data_56 = src_data[i + 46 * src_step];
    float src_data_57 = src_data[i + 47 * src_step];
    float src_data_60 = src_data[i + 48 * src_step];
    float src_data_61 = src_data[i + 49 * src_step];
    float src_data_62 = src_data[i + 50 * src_step];
    float src_data_63 = src_data[i + 51 * src_step];
    float src_data_64 = src_data[i + 52 * src_step];
    float src_data_65 = src_data[i + 53 * src_step];
    float src_data_66 = src_data[i + 54 * src_step];
    float src_data_67 = src_data[i + 55 * src_step];
    float src_data_70 = src_data[i + 56 * src_step];
    float src_data_71 = src_data[i + 57 * src_step];
    float src_data_72 = src_data[i + 58 * src_step];
    float src_data_73 = src_data[i + 59 * src_step];
    float src_data_74 = src_data[i + 60 * src_step];
    float src_data_75 = src_data[i + 61 * src_step];
    float src_data_76 = src_data[i + 62 * src_step];
    float src_data_77 = src_data[i + 63 * src_step];

    float d01 = src_data_10 - src_data_20;
    float d02 = src_data_11 - src_data_21;
    float d03 = src_data_12 - src_data_22;
    float d04 = src_data_13 - src_data_23;
    float d05 = src_data_14 - src_data_24;
    float d06 = src_data_15 - src_data_25;
    float d07 = src_data_16 - src_data_26;
    float d08 = src_data_17 - src_data_27;

    float d11 = src_data_30 - src_data_40;
    float d12 = src_data_31 - src_data_41;
    float d13 = src_data_32 - src_data_42;
    float d14 = src_data_33 - src_data_43;
    float d15 = src_data_34 - src_data_44;
    float d16 = src_data_35 - src_data_45;
    float d17 = src_data_36 - src_data_46;
    float d18 = src_data_37 - src_data_47;

    float d21 = src_data_50 - src_data_60;
    float d22 = src_data_51 - src_data_61;
    float d23 = src_data_52 - src_data_62;
    float d24 = src_data_53 - src_data_63;
    float d25 = src_data_54 - src_data_64;
    float d26 = src_data_55 - src_data_65;
    float d27 = src_data_56 - src_data_66;
    float d28 = src_data_57 - src_data_67;

    float d31 = src_data_10 + src_data_20;
    float d32 = src_data_11 + src_data_21;
    float d33 = src_data_12 + src_data_22;
    float d34 = src_data_13 + src_data_23;
    float d35 = src_data_14 + src_data_24;
    float d36 = src_data_15 + src_data_25;
    float d37 = src_data_16 + src_data_26;
    float d38 = src_data_17 + src_data_27;

    float d41 = src_data_30 + src_data_40;
    float d42 = src_data_31 + src_data_41;
    float d43 = src_data_32 + src_data_42;
    float d44 = src_data_33 + src_data_43;
    float d45 = src_data_34 + src_data_44;
    float d46 = src_data_35 + src_data_45;
    float d47 = src_data_36 + src_data_46;
    float d48 = src_data_37 + src_data_47;

    float d51 = src_data_50 + src_data_60;
    float d52 = src_data_51 + src_data_61;
    float d53 = src_data_52 + src_data_62;
    float d54 = src_data_53 + src_data_63;
    float d55 = src_data_54 + src_data_64;
    float d56 = src_data_55 + src_data_65;
    float d57 = src_data_56 + src_data_66;
    float d58 = src_data_57 + src_data_67;

    float t00 = src_data_00 + src_data_10 + src_data_20 + src_data_30 + src_data_40 + src_data_50 + src_data_60;
    float t01 = src_data_01 + src_data_11 + src_data_21 + src_data_31 + src_data_41 + src_data_51 + src_data_61;
    float t02 = src_data_02 + src_data_12 + src_data_22 + src_data_32 + src_data_42 + src_data_52 + src_data_62;
    float t03 = src_data_03 + src_data_13 + src_data_23 + src_data_33 + src_data_43 + src_data_53 + src_data_63;
    float t04 = src_data_04 + src_data_14 + src_data_24 + src_data_34 + src_data_44 + src_data_54 + src_data_64;
    float t05 = src_data_05 + src_data_15 + src_data_25 + src_data_35 + src_data_45 + src_data_55 + src_data_65;
    float t06 = src_data_06 + src_data_16 + src_data_26 + src_data_36 + src_data_46 + src_data_56 + src_data_66;
    float t07 = src_data_07 + src_data_17 + src_data_27 + src_data_37 + src_data_47 + src_data_57 + src_data_67;

    const float t10 = 0.5f * d01 + d11 + 1.5f * d21;
    const float t11 = 0.5f * d02 + d12 + 1.5f * d22;
    const float t12 = 0.5f * d03 + d13 + 1.5f * d23;
    const float t13 = 0.5f * d04 + d14 + 1.5f * d24;
    const float t14 = 0.5f * d05 + d15 + 1.5f * d25;
    const float t15 = 0.5f * d06 + d16 + 1.5f * d26;
    const float t16 = 0.5f * d07 + d17 + 1.5f * d27;
    const float t17 = 0.5f * d08 + d18 + 1.5f * d28;

    const float t20 = 0.25f * d31 + d41 + 2.25f * d51;
    const float t21 = 0.25f * d32 + d42 + 2.25f * d52;
    const float t22 = 0.25f * d33 + d43 + 2.25f * d53;
    const float t23 = 0.25f * d34 + d44 + 2.25f * d54;
    const float t24 = 0.25f * d35 + d45 + 2.25f * d55;
    const float t25 = 0.25f * d36 + d46 + 2.25f * d56;
    const float t26 = 0.25f * d37 + d47 + 2.25f * d57;
    const const float t27 = 0.25f * d38 + d48 + 2.25f * d58;

    const float t30 = 0.125f * d01 + d11 + 3.375f * d21 + src_data_70;
    const float t31 = 0.125f * d02 + d12 + 3.375f * d22 + src_data_71;
    const float t32 = 0.125f * d03 + d13 + 3.375f * d23 + src_data_72;
    const float t33 = 0.125f * d04 + d14 + 3.375f * d24 + src_data_73;
    const float t34 = 0.125f * d05 + d15 + 3.375f * d25 + src_data_74;
    const float t35 = 0.125f * d06 + d16 + 3.375f * d26 + src_data_75;
    const float t36 = 0.125f * d07 + d17 + 3.375f * d27 + src_data_76;
    const float t37 = 0.125f * d08 + d18 + 3.375f * d28 + src_data_77;

    float s11 = t01 - t02;
    float s12 = t11 - t12;
    float s13 = t21 - t22;
    float s14 = t31 - t32;

    float s21 = t03 - t04;
    float s22 = t13 - t14;
    float s23 = t23 - t24;
    float s24 = t33 - t34;

    float s31 = t05 - t06;
    float s32 = t15 - t16;
    float s33 = t25 - t26;
    float s34 = t35 - t36;

    float s41 = t01 + t02;
    float s42 = t11 + t12;
    float s43 = t21 + t22;
    float s44 = t31 + t32;

    float s51 = t03 + t04;
    float s52 = t13 + t14;
    float s53 = t23 + t24;
    float s54 = t33 + t34;

    float s61 = t05 + t06;
    float s62 = t15 + t16;
    float s63 = t25 + t26;
    float s64 = t35 + t36;

    float m00 = t00 + t01 + t02 + t03 + t04 + t05 + t06;
    const float m01 = 0.5f * s11 + s21 + 1.5f * s31;
    const float m02 = 0.25f * s41 + s51 + 2.25f * s61;
    const float m03 = 0.125f * s11 + s21 + 3.375f * s31 + t07;

    float m10 = t10 + t11 + t12 + t13 + t14 + t15 + t16;
    const float m11 = 0.5f * s12 + s22 + 1.5f * s32;
    const float m12 = 0.25f * s42 + s52 + 2.25f * s62;
    const float m13 = 0.125f * s12 + s22 + 3.375f * s32 + t17;

    float m20 = t20 + t21 + t22 + t23 + t24 + t25 + t26;
    const float m21 = 0.5f * s13 + s23 + 1.5f * s33;
    const float m22 = 0.25f * s43 + s53 + 2.25f * s63;
    const float m23 = 0.125f * s13 + s23 + 3.375f * s33 + t27;

    float m30 = t30 + t31 + t32 + t33 + t34 + t35 + t36;
    const float m31 = 0.5f * s14 + s24 + 1.5f * s34;
    const float m32 = 0.25f * s44 + s54 + 2.25f * s64;
    const float m33 = 0.125f * s14 + s24 + 3.375f * s34 + t37;

    (dst_data + i)[0] = m00 + bias_data[i];
    (dst_data + i + C4NUM)[0] = m01 + bias_data[i];
    (dst_data + i + 2 * C4NUM)[0] = m02 + bias_data[i];
    (dst_data + i + 3 * C4NUM)[0] = m03 + bias_data[i];

    (dst_data + i + dst_step * C4NUM)[0] = m10 + bias_data[i];
    (dst_data + i + dst_step * C4NUM + C4NUM)[0] = m11 + bias_data[i];
    (dst_data + i + dst_step * C4NUM + 2 * C4NUM)[0] = m12 + bias_data[i];
    (dst_data + i + dst_step * C4NUM + 3 * C4NUM)[0] = m13 + bias_data[i];

    (dst_data + i + 2 * dst_step * C4NUM)[0] = m20 + bias_data[i];
    (dst_data + i + 2 * dst_step * C4NUM + C4NUM)[0] = m21 + bias_data[i];
    (dst_data + i + 2 * dst_step * C4NUM + 2 * C4NUM)[0] = m22 + bias_data[i];
    (dst_data + i + 2 * dst_step * C4NUM + 3 * C4NUM)[0] = m23 + bias_data[i];

    (dst_data + i + 3 * dst_step * C4NUM)[0] = m30 + bias_data[i];
    (dst_data + i + 3 * dst_step * C4NUM + C4NUM)[0] = m31 + bias_data[i];
    (dst_data + i + 3 * dst_step * C4NUM + 2 * C4NUM)[0] = m32 + bias_data[i];
    (dst_data + i + 3 * dst_step * C4NUM + 3 * C4NUM)[0] = m33 + bias_data[i];
  }
#endif
}

void OutputTransform8x5Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                            int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src_data_00 = vld1q_f32(src_data + 0 * src_step);
  float32x4_t src_data_01 = vld1q_f32(src_data + 1 * src_step);
  float32x4_t src_data_02 = vld1q_f32(src_data + 2 * src_step);
  float32x4_t src_data_03 = vld1q_f32(src_data + 3 * src_step);
  float32x4_t src_data_04 = vld1q_f32(src_data + 4 * src_step);
  float32x4_t src_data_05 = vld1q_f32(src_data + 5 * src_step);
  float32x4_t src_data_06 = vld1q_f32(src_data + 6 * src_step);
  float32x4_t src_data_07 = vld1q_f32(src_data + 7 * src_step);
  float32x4_t src_data_10 = vld1q_f32(src_data + 8 * src_step);
  float32x4_t src_data_11 = vld1q_f32(src_data + 9 * src_step);
  float32x4_t src_data_12 = vld1q_f32(src_data + 10 * src_step);
  float32x4_t src_data_13 = vld1q_f32(src_data + 11 * src_step);
  float32x4_t src_data_14 = vld1q_f32(src_data + 12 * src_step);
  float32x4_t src_data_15 = vld1q_f32(src_data + 13 * src_step);
  float32x4_t src_data_16 = vld1q_f32(src_data + 14 * src_step);
  float32x4_t src_data_17 = vld1q_f32(src_data + 15 * src_step);
  float32x4_t src_data_20 = vld1q_f32(src_data + 16 * src_step);
  float32x4_t src_data_21 = vld1q_f32(src_data + 17 * src_step);
  float32x4_t src_data_22 = vld1q_f32(src_data + 18 * src_step);
  float32x4_t src_data_23 = vld1q_f32(src_data + 19 * src_step);
  float32x4_t src_data_24 = vld1q_f32(src_data + 20 * src_step);
  float32x4_t src_data_25 = vld1q_f32(src_data + 21 * src_step);
  float32x4_t src_data_26 = vld1q_f32(src_data + 22 * src_step);
  float32x4_t src_data_27 = vld1q_f32(src_data + 23 * src_step);
  float32x4_t src_data_30 = vld1q_f32(src_data + 24 * src_step);
  float32x4_t src_data_31 = vld1q_f32(src_data + 25 * src_step);
  float32x4_t src_data_32 = vld1q_f32(src_data + 26 * src_step);
  float32x4_t src_data_33 = vld1q_f32(src_data + 27 * src_step);
  float32x4_t src_data_34 = vld1q_f32(src_data + 28 * src_step);
  float32x4_t src_data_35 = vld1q_f32(src_data + 29 * src_step);
  float32x4_t src_data_36 = vld1q_f32(src_data + 30 * src_step);
  float32x4_t src_data_37 = vld1q_f32(src_data + 31 * src_step);
  float32x4_t src_data_40 = vld1q_f32(src_data + 32 * src_step);
  float32x4_t src_data_41 = vld1q_f32(src_data + 33 * src_step);
  float32x4_t src_data_42 = vld1q_f32(src_data + 34 * src_step);
  float32x4_t src_data_43 = vld1q_f32(src_data + 35 * src_step);
  float32x4_t src_data_44 = vld1q_f32(src_data + 36 * src_step);
  float32x4_t src_data_45 = vld1q_f32(src_data + 37 * src_step);
  float32x4_t src_data_46 = vld1q_f32(src_data + 38 * src_step);
  float32x4_t src_data_47 = vld1q_f32(src_data + 39 * src_step);
  float32x4_t src_data_50 = vld1q_f32(src_data + 40 * src_step);
  float32x4_t src_data_51 = vld1q_f32(src_data + 41 * src_step);
  float32x4_t src_data_52 = vld1q_f32(src_data + 42 * src_step);
  float32x4_t src_data_53 = vld1q_f32(src_data + 43 * src_step);
  float32x4_t src_data_54 = vld1q_f32(src_data + 44 * src_step);
  float32x4_t src_data_55 = vld1q_f32(src_data + 45 * src_step);
  float32x4_t src_data_56 = vld1q_f32(src_data + 46 * src_step);
  float32x4_t src_data_57 = vld1q_f32(src_data + 47 * src_step);
  float32x4_t src_data_60 = vld1q_f32(src_data + 48 * src_step);
  float32x4_t src_data_61 = vld1q_f32(src_data + 49 * src_step);
  float32x4_t src_data_62 = vld1q_f32(src_data + 50 * src_step);
  float32x4_t src_data_63 = vld1q_f32(src_data + 51 * src_step);
  float32x4_t src_data_64 = vld1q_f32(src_data + 52 * src_step);
  float32x4_t src_data_65 = vld1q_f32(src_data + 53 * src_step);
  float32x4_t src_data_66 = vld1q_f32(src_data + 54 * src_step);
  float32x4_t src_data_67 = vld1q_f32(src_data + 55 * src_step);
  float32x4_t src_data_70 = vld1q_f32(src_data + 56 * src_step);
  float32x4_t src_data_71 = vld1q_f32(src_data + 57 * src_step);
  float32x4_t src_data_72 = vld1q_f32(src_data + 58 * src_step);
  float32x4_t src_data_73 = vld1q_f32(src_data + 59 * src_step);
  float32x4_t src_data_74 = vld1q_f32(src_data + 60 * src_step);
  float32x4_t src_data_75 = vld1q_f32(src_data + 61 * src_step);
  float32x4_t src_data_76 = vld1q_f32(src_data + 62 * src_step);
  float32x4_t src_data_77 = vld1q_f32(src_data + 63 * src_step);

  float32x4_t d01 = vsubq_f32(src_data_10, src_data_20);
  float32x4_t d02 = vsubq_f32(src_data_11, src_data_21);
  float32x4_t d03 = vsubq_f32(src_data_12, src_data_22);
  float32x4_t d04 = vsubq_f32(src_data_13, src_data_23);
  float32x4_t d05 = vsubq_f32(src_data_14, src_data_24);
  float32x4_t d06 = vsubq_f32(src_data_15, src_data_25);
  float32x4_t d07 = vsubq_f32(src_data_16, src_data_26);
  float32x4_t d08 = vsubq_f32(src_data_17, src_data_27);

  float32x4_t d11 = vsubq_f32(src_data_30, src_data_40);
  float32x4_t d12 = vsubq_f32(src_data_31, src_data_41);
  float32x4_t d13 = vsubq_f32(src_data_32, src_data_42);
  float32x4_t d14 = vsubq_f32(src_data_33, src_data_43);
  float32x4_t d15 = vsubq_f32(src_data_34, src_data_44);
  float32x4_t d16 = vsubq_f32(src_data_35, src_data_45);
  float32x4_t d17 = vsubq_f32(src_data_36, src_data_46);
  float32x4_t d18 = vsubq_f32(src_data_37, src_data_47);

  float32x4_t d21 = vsubq_f32(src_data_50, src_data_60);
  float32x4_t d22 = vsubq_f32(src_data_51, src_data_61);
  float32x4_t d23 = vsubq_f32(src_data_52, src_data_62);
  float32x4_t d24 = vsubq_f32(src_data_53, src_data_63);
  float32x4_t d25 = vsubq_f32(src_data_54, src_data_64);
  float32x4_t d26 = vsubq_f32(src_data_55, src_data_65);
  float32x4_t d27 = vsubq_f32(src_data_56, src_data_66);
  float32x4_t d28 = vsubq_f32(src_data_57, src_data_67);

  float32x4_t d31 = vaddq_f32(src_data_10, src_data_20);
  float32x4_t d32 = vaddq_f32(src_data_11, src_data_21);
  float32x4_t d33 = vaddq_f32(src_data_12, src_data_22);
  float32x4_t d34 = vaddq_f32(src_data_13, src_data_23);
  float32x4_t d35 = vaddq_f32(src_data_14, src_data_24);
  float32x4_t d36 = vaddq_f32(src_data_15, src_data_25);
  float32x4_t d37 = vaddq_f32(src_data_16, src_data_26);
  float32x4_t d38 = vaddq_f32(src_data_17, src_data_27);

  float32x4_t d41 = vaddq_f32(src_data_30, src_data_40);
  float32x4_t d42 = vaddq_f32(src_data_31, src_data_41);
  float32x4_t d43 = vaddq_f32(src_data_32, src_data_42);
  float32x4_t d44 = vaddq_f32(src_data_33, src_data_43);
  float32x4_t d45 = vaddq_f32(src_data_34, src_data_44);
  float32x4_t d46 = vaddq_f32(src_data_35, src_data_45);
  float32x4_t d47 = vaddq_f32(src_data_36, src_data_46);
  float32x4_t d48 = vaddq_f32(src_data_37, src_data_47);

  float32x4_t d51 = vaddq_f32(src_data_50, src_data_60);
  float32x4_t d52 = vaddq_f32(src_data_51, src_data_61);
  float32x4_t d53 = vaddq_f32(src_data_52, src_data_62);
  float32x4_t d54 = vaddq_f32(src_data_53, src_data_63);
  float32x4_t d55 = vaddq_f32(src_data_54, src_data_64);
  float32x4_t d56 = vaddq_f32(src_data_55, src_data_65);
  float32x4_t d57 = vaddq_f32(src_data_56, src_data_66);
  float32x4_t d58 = vaddq_f32(src_data_57, src_data_67);

  float32x4_t t00 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_00, src_data_10), src_data_20), src_data_30), src_data_40),
      src_data_50),
    src_data_60);
  float32x4_t t01 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_01, src_data_11), src_data_21), src_data_31), src_data_41),
      src_data_51),
    src_data_61);
  float32x4_t t02 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_02, src_data_12), src_data_22), src_data_32), src_data_42),
      src_data_52),
    src_data_62);
  float32x4_t t03 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_03, src_data_13), src_data_23), src_data_33), src_data_43),
      src_data_53),
    src_data_63);
  float32x4_t t04 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_04, src_data_14), src_data_24), src_data_34), src_data_44),
      src_data_54),
    src_data_64);
  float32x4_t t05 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_05, src_data_15), src_data_25), src_data_35), src_data_45),
      src_data_55),
    src_data_65);
  float32x4_t t06 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_06, src_data_16), src_data_26), src_data_36), src_data_46),
      src_data_56),
    src_data_66);
  float32x4_t t07 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_07, src_data_17), src_data_27), src_data_37), src_data_47),
      src_data_57),
    src_data_67);

  float32x4_t t10 = vaddq_f32(vaddq_f32(vmulq_n_f32(d01, 0.5), d11), vmulq_n_f32(d21, 1.5));
  float32x4_t t11 = vaddq_f32(vaddq_f32(vmulq_n_f32(d02, 0.5), d12), vmulq_n_f32(d22, 1.5));
  float32x4_t t12 = vaddq_f32(vaddq_f32(vmulq_n_f32(d03, 0.5), d13), vmulq_n_f32(d23, 1.5));
  float32x4_t t13 = vaddq_f32(vaddq_f32(vmulq_n_f32(d04, 0.5), d14), vmulq_n_f32(d24, 1.5));
  float32x4_t t14 = vaddq_f32(vaddq_f32(vmulq_n_f32(d05, 0.5), d15), vmulq_n_f32(d25, 1.5));
  float32x4_t t15 = vaddq_f32(vaddq_f32(vmulq_n_f32(d06, 0.5), d16), vmulq_n_f32(d26, 1.5));
  float32x4_t t16 = vaddq_f32(vaddq_f32(vmulq_n_f32(d07, 0.5), d17), vmulq_n_f32(d27, 1.5));
  float32x4_t t17 = vaddq_f32(vaddq_f32(vmulq_n_f32(d08, 0.5), d18), vmulq_n_f32(d28, 1.5));

  float32x4_t t20 = vaddq_f32(vaddq_f32(vmulq_n_f32(d31, 0.25), d41), vmulq_n_f32(d51, 2.25));
  float32x4_t t21 = vaddq_f32(vaddq_f32(vmulq_n_f32(d32, 0.25), d42), vmulq_n_f32(d52, 2.25));
  float32x4_t t22 = vaddq_f32(vaddq_f32(vmulq_n_f32(d33, 0.25), d43), vmulq_n_f32(d53, 2.25));
  float32x4_t t23 = vaddq_f32(vaddq_f32(vmulq_n_f32(d34, 0.25), d44), vmulq_n_f32(d54, 2.25));
  float32x4_t t24 = vaddq_f32(vaddq_f32(vmulq_n_f32(d35, 0.25), d45), vmulq_n_f32(d55, 2.25));
  float32x4_t t25 = vaddq_f32(vaddq_f32(vmulq_n_f32(d36, 0.25), d46), vmulq_n_f32(d56, 2.25));
  float32x4_t t26 = vaddq_f32(vaddq_f32(vmulq_n_f32(d37, 0.25), d47), vmulq_n_f32(d57, 2.25));
  float32x4_t t27 = vaddq_f32(vaddq_f32(vmulq_n_f32(d38, 0.25), d48), vmulq_n_f32(d58, 2.25));

  float32x4_t t30 = vaddq_f32(vaddq_f32(vmulq_n_f32(d01, 0.125), d11), vmulq_n_f32(d21, 3.375));
  float32x4_t t31 = vaddq_f32(vaddq_f32(vmulq_n_f32(d02, 0.125), d12), vmulq_n_f32(d22, 3.375));
  float32x4_t t32 = vaddq_f32(vaddq_f32(vmulq_n_f32(d03, 0.125), d13), vmulq_n_f32(d23, 3.375));
  float32x4_t t33 = vaddq_f32(vaddq_f32(vmulq_n_f32(d04, 0.125), d14), vmulq_n_f32(d24, 3.375));
  float32x4_t t34 = vaddq_f32(vaddq_f32(vmulq_n_f32(d05, 0.125), d15), vmulq_n_f32(d25, 3.375));
  float32x4_t t35 = vaddq_f32(vaddq_f32(vmulq_n_f32(d06, 0.125), d16), vmulq_n_f32(d26, 3.375));
  float32x4_t t36 = vaddq_f32(vaddq_f32(vmulq_n_f32(d07, 0.125), d17), vmulq_n_f32(d27, 3.375));
  float32x4_t t37 = vaddq_f32(vaddq_f32(vmulq_n_f32(d08, 0.125), d18), vmulq_n_f32(d28, 3.375));

  float32x4_t t40 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d31, 0.0625), d41), vmulq_n_f32(d51, 5.0625)), src_data_70);
  float32x4_t t41 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d32, 0.0625), d42), vmulq_n_f32(d52, 5.0625)), src_data_71);
  float32x4_t t42 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d33, 0.0625), d43), vmulq_n_f32(d53, 5.0625)), src_data_72);
  float32x4_t t43 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d34, 0.0625), d44), vmulq_n_f32(d54, 5.0625)), src_data_73);
  float32x4_t t44 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d35, 0.0625), d45), vmulq_n_f32(d55, 5.0625)), src_data_74);
  float32x4_t t45 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d36, 0.0625), d46), vmulq_n_f32(d56, 5.0625)), src_data_75);
  float32x4_t t46 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d37, 0.0625), d47), vmulq_n_f32(d57, 5.0625)), src_data_76);
  float32x4_t t47 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d38, 0.0625), d48), vmulq_n_f32(d58, 5.0625)), src_data_77);

  float32x4_t s11 = vsubq_f32(t01, t02);
  float32x4_t s12 = vsubq_f32(t11, t12);
  float32x4_t s13 = vsubq_f32(t21, t22);
  float32x4_t s14 = vsubq_f32(t31, t32);
  float32x4_t s15 = vsubq_f32(t41, t42);

  float32x4_t s21 = vsubq_f32(t03, t04);
  float32x4_t s22 = vsubq_f32(t13, t14);
  float32x4_t s23 = vsubq_f32(t23, t24);
  float32x4_t s24 = vsubq_f32(t33, t34);
  float32x4_t s25 = vsubq_f32(t43, t44);

  float32x4_t s31 = vsubq_f32(t05, t06);
  float32x4_t s32 = vsubq_f32(t15, t16);
  float32x4_t s33 = vsubq_f32(t25, t26);
  float32x4_t s34 = vsubq_f32(t35, t36);
  float32x4_t s35 = vsubq_f32(t45, t46);

  float32x4_t s41 = vaddq_f32(t01, t02);
  float32x4_t s42 = vaddq_f32(t11, t12);
  float32x4_t s43 = vaddq_f32(t21, t22);
  float32x4_t s44 = vaddq_f32(t31, t32);
  float32x4_t s45 = vaddq_f32(t41, t42);

  float32x4_t s51 = vaddq_f32(t03, t04);
  float32x4_t s52 = vaddq_f32(t13, t14);
  float32x4_t s53 = vaddq_f32(t23, t24);
  float32x4_t s54 = vaddq_f32(t33, t34);
  float32x4_t s55 = vaddq_f32(t43, t44);

  float32x4_t s61 = vaddq_f32(t05, t06);
  float32x4_t s62 = vaddq_f32(t15, t16);
  float32x4_t s63 = vaddq_f32(t25, t26);
  float32x4_t s64 = vaddq_f32(t35, t36);
  float32x4_t s65 = vaddq_f32(t45, t46);

  float32x4_t m00 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t00, t01), t02), t03), t04), t05), t06);
  float32x4_t m01 = vaddq_f32(vaddq_f32(vmulq_n_f32(s11, 0.5), s21), vmulq_n_f32(s31, 1.5));
  float32x4_t m02 = vaddq_f32(vaddq_f32(vmulq_n_f32(s41, 0.25), s51), vmulq_n_f32(s61, 2.25));
  float32x4_t m03 = vaddq_f32(vaddq_f32(vmulq_n_f32(s11, 0.125), s21), vmulq_n_f32(s31, 3.375));
  float32x4_t m04 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s41, 0.0625), s51), vmulq_n_f32(s61, 5.0625)), t07);

  float32x4_t m10 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t10, t11), t12), t13), t14), t15), t16);
  float32x4_t m11 = vaddq_f32(vaddq_f32(vmulq_n_f32(s12, 0.5), s22), vmulq_n_f32(s32, 1.5));
  float32x4_t m12 = vaddq_f32(vaddq_f32(vmulq_n_f32(s42, 0.25), s52), vmulq_n_f32(s62, 2.25));
  float32x4_t m13 = vaddq_f32(vaddq_f32(vmulq_n_f32(s12, 0.125), s22), vmulq_n_f32(s32, 3.375));
  float32x4_t m14 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s42, 0.0625), s52), vmulq_n_f32(s62, 5.0625)), t17);

  float32x4_t m20 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t20, t21), t22), t23), t24), t25), t26);
  float32x4_t m21 = vaddq_f32(vaddq_f32(vmulq_n_f32(s13, 0.5), s23), vmulq_n_f32(s33, 1.5));
  float32x4_t m22 = vaddq_f32(vaddq_f32(vmulq_n_f32(s43, 0.25), s53), vmulq_n_f32(s63, 2.25));
  float32x4_t m23 = vaddq_f32(vaddq_f32(vmulq_n_f32(s13, 0.125), s23), vmulq_n_f32(s33, 3.375));
  float32x4_t m24 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s43, 0.0625), s53), vmulq_n_f32(s63, 5.0625)), t27);

  float32x4_t m30 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t30, t31), t32), t33), t34), t35), t36);
  float32x4_t m31 = vaddq_f32(vaddq_f32(vmulq_n_f32(s14, 0.5), s24), vmulq_n_f32(s34, 1.5));
  float32x4_t m32 = vaddq_f32(vaddq_f32(vmulq_n_f32(s44, 0.25), s54), vmulq_n_f32(s64, 2.25));
  float32x4_t m33 = vaddq_f32(vaddq_f32(vmulq_n_f32(s14, 0.125), s24), vmulq_n_f32(s34, 3.375));
  float32x4_t m34 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s44, 0.0625), s54), vmulq_n_f32(s64, 5.0625)), t37);

  float32x4_t m40 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t40, t41), t42), t43), t44), t45), t46);
  float32x4_t m41 = vaddq_f32(vaddq_f32(vmulq_n_f32(s15, 0.5), s25), vmulq_n_f32(s35, 1.5));
  float32x4_t m42 = vaddq_f32(vaddq_f32(vmulq_n_f32(s45, 0.25), s55), vmulq_n_f32(s65, 2.25));
  float32x4_t m43 = vaddq_f32(vaddq_f32(vmulq_n_f32(s15, 0.125), s25), vmulq_n_f32(s35, 3.375));
  float32x4_t m44 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s45, 0.0625), s55), vmulq_n_f32(s65, 5.0625)), t47);

  float32x4_t bias_ptr = vld1q_f32(bias_data);
  vst1q_f32(dst_data, vaddq_f32(m00, bias_ptr));
  vst1q_f32(dst_data + C4NUM, vaddq_f32(m01, bias_ptr));
  vst1q_f32(dst_data + 2 * C4NUM, vaddq_f32(m02, bias_ptr));
  vst1q_f32(dst_data + 3 * C4NUM, vaddq_f32(m03, bias_ptr));
  vst1q_f32(dst_data + 4 * C4NUM, vaddq_f32(m04, bias_ptr));

  vst1q_f32(dst_data + dst_step * C4NUM, vaddq_f32(m10, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM + C4NUM, vaddq_f32(m11, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM + 2 * C4NUM, vaddq_f32(m12, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM + 3 * C4NUM, vaddq_f32(m13, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM + 4 * C4NUM, vaddq_f32(m14, bias_ptr));

  vst1q_f32(dst_data + 2 * dst_step * C4NUM, vaddq_f32(m20, bias_ptr));
  vst1q_f32(dst_data + 2 * dst_step * C4NUM + C4NUM, vaddq_f32(m21, bias_ptr));
  vst1q_f32(dst_data + 2 * dst_step * C4NUM + 2 * C4NUM, vaddq_f32(m22, bias_ptr));
  vst1q_f32(dst_data + 2 * dst_step * C4NUM + 3 * C4NUM, vaddq_f32(m23, bias_ptr));
  vst1q_f32(dst_data + 2 * dst_step * C4NUM + 4 * C4NUM, vaddq_f32(m24, bias_ptr));

  vst1q_f32(dst_data + 3 * dst_step * C4NUM, vaddq_f32(m30, bias_ptr));
  vst1q_f32(dst_data + 3 * dst_step * C4NUM + C4NUM, vaddq_f32(m31, bias_ptr));
  vst1q_f32(dst_data + 3 * dst_step * C4NUM + 2 * C4NUM, vaddq_f32(m32, bias_ptr));
  vst1q_f32(dst_data + 3 * dst_step * C4NUM + 3 * C4NUM, vaddq_f32(m33, bias_ptr));
  vst1q_f32(dst_data + 3 * dst_step * C4NUM + 4 * C4NUM, vaddq_f32(m34, bias_ptr));

  vst1q_f32(dst_data + 4 * dst_step * C4NUM, vaddq_f32(m40, bias_ptr));
  vst1q_f32(dst_data + 4 * dst_step * C4NUM + C4NUM, vaddq_f32(m41, bias_ptr));
  vst1q_f32(dst_data + 4 * dst_step * C4NUM + 2 * C4NUM, vaddq_f32(m42, bias_ptr));
  vst1q_f32(dst_data + 4 * dst_step * C4NUM + 3 * C4NUM, vaddq_f32(m43, bias_ptr));
  vst1q_f32(dst_data + 4 * dst_step * C4NUM + 4 * C4NUM, vaddq_f32(m44, bias_ptr));
#else
  for (int i = 0; i < C4NUM; i++) {
    float src_data_00 = src_data[i];
    float src_data_01 = src_data[i + src_step];
    float src_data_02 = src_data[i + 2 * src_step];
    float src_data_03 = src_data[i + 3 * src_step];
    float src_data_04 = src_data[i + 4 * src_step];
    float src_data_05 = src_data[i + 5 * src_step];
    float src_data_06 = src_data[i + 6 * src_step];
    float src_data_07 = src_data[i + 7 * src_step];
    float src_data_10 = src_data[i + 8 * src_step];
    float src_data_11 = src_data[i + 9 * src_step];
    float src_data_12 = src_data[i + 10 * src_step];
    float src_data_13 = src_data[i + 11 * src_step];
    float src_data_14 = src_data[i + 12 * src_step];
    float src_data_15 = src_data[i + 13 * src_step];
    float src_data_16 = src_data[i + 14 * src_step];
    float src_data_17 = src_data[i + 15 * src_step];
    float src_data_20 = src_data[i + 16 * src_step];
    float src_data_21 = src_data[i + 17 * src_step];
    float src_data_22 = src_data[i + 18 * src_step];
    float src_data_23 = src_data[i + 19 * src_step];
    float src_data_24 = src_data[i + 20 * src_step];
    float src_data_25 = src_data[i + 21 * src_step];
    float src_data_26 = src_data[i + 22 * src_step];
    float src_data_27 = src_data[i + 23 * src_step];
    float src_data_30 = src_data[i + 24 * src_step];
    float src_data_31 = src_data[i + 25 * src_step];
    float src_data_32 = src_data[i + 26 * src_step];
    float src_data_33 = src_data[i + 27 * src_step];
    float src_data_34 = src_data[i + 28 * src_step];
    float src_data_35 = src_data[i + 29 * src_step];
    float src_data_36 = src_data[i + 30 * src_step];
    float src_data_37 = src_data[i + 31 * src_step];
    float src_data_40 = src_data[i + 32 * src_step];
    float src_data_41 = src_data[i + 33 * src_step];
    float src_data_42 = src_data[i + 34 * src_step];
    float src_data_43 = src_data[i + 35 * src_step];
    float src_data_44 = src_data[i + 36 * src_step];
    float src_data_45 = src_data[i + 37 * src_step];
    float src_data_46 = src_data[i + 38 * src_step];
    float src_data_47 = src_data[i + 39 * src_step];
    float src_data_50 = src_data[i + 40 * src_step];
    float src_data_51 = src_data[i + 41 * src_step];
    float src_data_52 = src_data[i + 42 * src_step];
    float src_data_53 = src_data[i + 43 * src_step];
    float src_data_54 = src_data[i + 44 * src_step];
    float src_data_55 = src_data[i + 45 * src_step];
    float src_data_56 = src_data[i + 46 * src_step];
    float src_data_57 = src_data[i + 47 * src_step];
    float src_data_60 = src_data[i + 48 * src_step];
    float src_data_61 = src_data[i + 49 * src_step];
    float src_data_62 = src_data[i + 50 * src_step];
    float src_data_63 = src_data[i + 51 * src_step];
    float src_data_64 = src_data[i + 52 * src_step];
    float src_data_65 = src_data[i + 53 * src_step];
    float src_data_66 = src_data[i + 54 * src_step];
    float src_data_67 = src_data[i + 55 * src_step];
    float src_data_70 = src_data[i + 56 * src_step];
    float src_data_71 = src_data[i + 57 * src_step];
    float src_data_72 = src_data[i + 58 * src_step];
    float src_data_73 = src_data[i + 59 * src_step];
    float src_data_74 = src_data[i + 60 * src_step];
    float src_data_75 = src_data[i + 61 * src_step];
    float src_data_76 = src_data[i + 62 * src_step];
    float src_data_77 = src_data[i + 63 * src_step];

    float d01 = src_data_10 - src_data_20;
    float d02 = src_data_11 - src_data_21;
    float d03 = src_data_12 - src_data_22;
    float d04 = src_data_13 - src_data_23;
    float d05 = src_data_14 - src_data_24;
    float d06 = src_data_15 - src_data_25;
    float d07 = src_data_16 - src_data_26;
    float d08 = src_data_17 - src_data_27;

    float d11 = src_data_30 - src_data_40;
    float d12 = src_data_31 - src_data_41;
    float d13 = src_data_32 - src_data_42;
    float d14 = src_data_33 - src_data_43;
    float d15 = src_data_34 - src_data_44;
    float d16 = src_data_35 - src_data_45;
    float d17 = src_data_36 - src_data_46;
    float d18 = src_data_37 - src_data_47;

    float d21 = src_data_50 - src_data_60;
    float d22 = src_data_51 - src_data_61;
    float d23 = src_data_52 - src_data_62;
    float d24 = src_data_53 - src_data_63;
    float d25 = src_data_54 - src_data_64;
    float d26 = src_data_55 - src_data_65;
    float d27 = src_data_56 - src_data_66;
    float d28 = src_data_57 - src_data_67;

    float d31 = src_data_10 + src_data_20;
    float d32 = src_data_11 + src_data_21;
    float d33 = src_data_12 + src_data_22;
    float d34 = src_data_13 + src_data_23;
    float d35 = src_data_14 + src_data_24;
    float d36 = src_data_15 + src_data_25;
    float d37 = src_data_16 + src_data_26;
    float d38 = src_data_17 + src_data_27;

    float d41 = src_data_30 + src_data_40;
    float d42 = src_data_31 + src_data_41;
    float d43 = src_data_32 + src_data_42;
    float d44 = src_data_33 + src_data_43;
    float d45 = src_data_34 + src_data_44;
    float d46 = src_data_35 + src_data_45;
    float d47 = src_data_36 + src_data_46;
    float d48 = src_data_37 + src_data_47;

    float d51 = src_data_50 + src_data_60;
    float d52 = src_data_51 + src_data_61;
    float d53 = src_data_52 + src_data_62;
    float d54 = src_data_53 + src_data_63;
    float d55 = src_data_54 + src_data_64;
    float d56 = src_data_55 + src_data_65;
    float d57 = src_data_56 + src_data_66;
    float d58 = src_data_57 + src_data_67;

    float t00 = src_data_00 + src_data_10 + src_data_20 + src_data_30 + src_data_40 + src_data_50 + src_data_60;
    float t01 = src_data_01 + src_data_11 + src_data_21 + src_data_31 + src_data_41 + src_data_51 + src_data_61;
    float t02 = src_data_02 + src_data_12 + src_data_22 + src_data_32 + src_data_42 + src_data_52 + src_data_62;
    float t03 = src_data_03 + src_data_13 + src_data_23 + src_data_33 + src_data_43 + src_data_53 + src_data_63;
    float t04 = src_data_04 + src_data_14 + src_data_24 + src_data_34 + src_data_44 + src_data_54 + src_data_64;
    float t05 = src_data_05 + src_data_15 + src_data_25 + src_data_35 + src_data_45 + src_data_55 + src_data_65;
    float t06 = src_data_06 + src_data_16 + src_data_26 + src_data_36 + src_data_46 + src_data_56 + src_data_66;
    float t07 = src_data_07 + src_data_17 + src_data_27 + src_data_37 + src_data_47 + src_data_57 + src_data_67;

    const float t10 = 0.5f * d01 + d11 + 1.5f * d21;
    const float t11 = 0.5f * d02 + d12 + 1.5f * d22;
    const float t12 = 0.5f * d03 + d13 + 1.5f * d23;
    const float t13 = 0.5f * d04 + d14 + 1.5f * d24;
    const float t14 = 0.5f * d05 + d15 + 1.5f * d25;
    const float t15 = 0.5f * d06 + d16 + 1.5f * d26;
    const float t16 = 0.5f * d07 + d17 + 1.5f * d27;
    const float t17 = 0.5f * d08 + d18 + 1.5f * d28;

    const float t20 = 0.25f * d31 + d41 + 2.25f * d51;
    const float t21 = 0.25f * d32 + d42 + 2.25f * d52;
    const float t22 = 0.25f * d33 + d43 + 2.25f * d53;
    const float t23 = 0.25f * d34 + d44 + 2.25f * d54;
    const float t24 = 0.25f * d35 + d45 + 2.25f * d55;
    const float t25 = 0.25f * d36 + d46 + 2.25f * d56;
    const float t26 = 0.25f * d37 + d47 + 2.25f * d57;
    const float t27 = 0.25f * d38 + d48 + 2.25f * d58;

    const float t30 = 0.125f * d01 + d11 + 3.375f * d21;
    const float t31 = 0.125f * d02 + d12 + 3.375f * d22;
    const float t32 = 0.125f * d03 + d13 + 3.375f * d23;
    const float t33 = 0.125f * d04 + d14 + 3.375f * d24;
    const float t34 = 0.125f * d05 + d15 + 3.375f * d25;
    const float t35 = 0.125f * d06 + d16 + 3.375f * d26;
    const float t36 = 0.125f * d07 + d17 + 3.375f * d27;
    const float t37 = 0.125f * d08 + d18 + 3.375f * d28;

    const float t40 = 0.0625f * d31 + d41 + 5.0625f * d51 + src_data_70;
    const float t41 = 0.0625f * d32 + d42 + 5.0625f * d52 + src_data_71;
    const float t42 = 0.0625f * d33 + d43 + 5.0625f * d53 + src_data_72;
    const float t43 = 0.0625f * d34 + d44 + 5.0625f * d54 + src_data_73;
    const float t44 = 0.0625f * d35 + d45 + 5.0625f * d55 + src_data_74;
    const float t45 = 0.0625f * d36 + d46 + 5.0625f * d56 + src_data_75;
    const float t46 = 0.0625f * d37 + d47 + 5.0625f * d57 + src_data_76;
    const float t47 = 0.0625f * d38 + d48 + 5.0625f * d58 + src_data_77;

    float s11 = t01 - t02;
    float s12 = t11 - t12;
    float s13 = t21 - t22;
    float s14 = t31 - t32;
    float s15 = t41 - t42;

    float s21 = t03 - t04;
    float s22 = t13 - t14;
    float s23 = t23 - t24;
    float s24 = t33 - t34;
    float s25 = t43 - t44;

    float s31 = t05 - t06;
    float s32 = t15 - t16;
    float s33 = t25 - t26;
    float s34 = t35 - t36;
    float s35 = t45 - t46;

    float s41 = t01 + t02;
    float s42 = t11 + t12;
    float s43 = t21 + t22;
    float s44 = t31 + t32;
    float s45 = t41 + t42;

    float s51 = t03 + t04;
    float s52 = t13 + t14;
    float s53 = t23 + t24;
    float s54 = t33 + t34;
    float s55 = t43 + t44;

    float s61 = t05 + t06;
    float s62 = t15 + t16;
    float s63 = t25 + t26;
    float s64 = t35 + t36;
    float s65 = t45 + t46;

    float m00 = t00 + t01 + t02 + t03 + t04 + t05 + t06;
    const float m01 = 0.5f * s11 + s21 + 1.5f * s31;
    const float m02 = 0.25f * s41 + s51 + 2.25f * s61;
    const float m03 = 0.125f * s11 + s21 + 3.375f * s31;
    const float m04 = 0.0625f * s41 + s51 + 5.0625f * s61 + t07;

    float m10 = t10 + t11 + t12 + t13 + t14 + t15 + t16;
    const float m11 = 0.5f * s12 + s22 + 1.5f * s32;
    const float m12 = 0.25f * s42 + s52 + 2.25f * s62;
    const float m13 = 0.125f * s12 + s22 + 3.375f * s32;
    const float m14 = 0.0625f * s42 + s52 + 5.0625f * s62 + t17;

    float m20 = t20 + t21 + t22 + t23 + t24 + t25 + t26;
    const float m21 = 0.5f * s13 + s23 + 1.5f * s33;
    const float m22 = 0.25f * s43 + s53 + 2.25f * s63;
    const float m23 = 0.125f * s13 + s23 + 3.375f * s33;
    const float m24 = 0.0625f * s43 + s53 + 5.0625f * s63 + t27;

    float m30 = t30 + t31 + t32 + t33 + t34 + t35 + t36;
    const float m31 = 0.5f * s14 + s24 + 1.5f * s34;
    const float m32 = 0.25f * s44 + s54 + 2.25f * s64;
    const float m33 = 0.125f * s14 + s24 + 3.375f * s34;
    const float m34 = 0.0625f * s44 + s54 + 5.0625f * s64 + t37;

    float m40 = t40 + t41 + t42 + t43 + t44 + t45 + t46;
    const float m41 = 0.5f * s15 + s25 + 1.5f * s35;
    const float m42 = 0.25f * s45 + s55 + 2.25f * s65;
    const float m43 = 0.125f * s15 + s25 + 3.375f * s35;
    const float m44 = 0.0625f * s45 + s55 + 5.0625f * s65 + t47;

    (dst_data + i)[0] = m00 + bias_data[i];
    (dst_data + i + C4NUM)[0] = m01 + bias_data[i];
    (dst_data + i + 2 * C4NUM)[0] = m02 + bias_data[i];
    (dst_data + i + 3 * C4NUM)[0] = m03 + bias_data[i];
    (dst_data + i + 4 * C4NUM)[0] = m04 + bias_data[i];

    (dst_data + i + dst_step * C4NUM)[0] = m10 + bias_data[i];
    (dst_data + i + dst_step * C4NUM + C4NUM)[0] = m11 + bias_data[i];
    (dst_data + i + dst_step * C4NUM + 2 * C4NUM)[0] = m12 + bias_data[i];
    (dst_data + i + dst_step * C4NUM + 3 * C4NUM)[0] = m13 + bias_data[i];
    (dst_data + i + dst_step * C4NUM + 4 * C4NUM)[0] = m14 + bias_data[i];

    (dst_data + i + 2 * dst_step * C4NUM)[0] = m20 + bias_data[i];
    (dst_data + i + 2 * dst_step * C4NUM + C4NUM)[0] = m21 + bias_data[i];
    (dst_data + i + 2 * dst_step * C4NUM + 2 * C4NUM)[0] = m22 + bias_data[i];
    (dst_data + i + 2 * dst_step * C4NUM + 3 * C4NUM)[0] = m23 + bias_data[i];
    (dst_data + i + 2 * dst_step * C4NUM + 4 * C4NUM)[0] = m24 + bias_data[i];

    (dst_data + i + 3 * dst_step * C4NUM)[0] = m30 + bias_data[i];
    (dst_data + i + 3 * dst_step * C4NUM + C4NUM)[0] = m31 + bias_data[i];
    (dst_data + i + 3 * dst_step * C4NUM + 2 * C4NUM)[0] = m32 + bias_data[i];
    (dst_data + i + 3 * dst_step * C4NUM + 3 * C4NUM)[0] = m33 + bias_data[i];
    (dst_data + i + 3 * dst_step * C4NUM + 4 * C4NUM)[0] = m34 + bias_data[i];

    (dst_data + i + 4 * dst_step * C4NUM)[0] = m40 + bias_data[i];
    (dst_data + i + 4 * dst_step * C4NUM + C4NUM)[0] = m41 + bias_data[i];
    (dst_data + i + 4 * dst_step * C4NUM + 2 * C4NUM)[0] = m42 + bias_data[i];
    (dst_data + i + 4 * dst_step * C4NUM + 3 * C4NUM)[0] = m43 + bias_data[i];
    (dst_data + i + 4 * dst_step * C4NUM + 4 * C4NUM)[0] = m44 + bias_data[i];
  }
#endif
}

void OutputTransform8x6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                            int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src_data_00 = vld1q_f32(src_data + 0 * src_step);
  float32x4_t src_data_01 = vld1q_f32(src_data + 1 * src_step);
  float32x4_t src_data_02 = vld1q_f32(src_data + 2 * src_step);
  float32x4_t src_data_03 = vld1q_f32(src_data + 3 * src_step);
  float32x4_t src_data_04 = vld1q_f32(src_data + 4 * src_step);
  float32x4_t src_data_05 = vld1q_f32(src_data + 5 * src_step);
  float32x4_t src_data_06 = vld1q_f32(src_data + 6 * src_step);
  float32x4_t src_data_07 = vld1q_f32(src_data + 7 * src_step);
  float32x4_t src_data_10 = vld1q_f32(src_data + 8 * src_step);
  float32x4_t src_data_11 = vld1q_f32(src_data + 9 * src_step);
  float32x4_t src_data_12 = vld1q_f32(src_data + 10 * src_step);
  float32x4_t src_data_13 = vld1q_f32(src_data + 11 * src_step);
  float32x4_t src_data_14 = vld1q_f32(src_data + 12 * src_step);
  float32x4_t src_data_15 = vld1q_f32(src_data + 13 * src_step);
  float32x4_t src_data_16 = vld1q_f32(src_data + 14 * src_step);
  float32x4_t src_data_17 = vld1q_f32(src_data + 15 * src_step);
  float32x4_t src_data_20 = vld1q_f32(src_data + 16 * src_step);
  float32x4_t src_data_21 = vld1q_f32(src_data + 17 * src_step);
  float32x4_t src_data_22 = vld1q_f32(src_data + 18 * src_step);
  float32x4_t src_data_23 = vld1q_f32(src_data + 19 * src_step);
  float32x4_t src_data_24 = vld1q_f32(src_data + 20 * src_step);
  float32x4_t src_data_25 = vld1q_f32(src_data + 21 * src_step);
  float32x4_t src_data_26 = vld1q_f32(src_data + 22 * src_step);
  float32x4_t src_data_27 = vld1q_f32(src_data + 23 * src_step);
  float32x4_t src_data_30 = vld1q_f32(src_data + 24 * src_step);
  float32x4_t src_data_31 = vld1q_f32(src_data + 25 * src_step);
  float32x4_t src_data_32 = vld1q_f32(src_data + 26 * src_step);
  float32x4_t src_data_33 = vld1q_f32(src_data + 27 * src_step);
  float32x4_t src_data_34 = vld1q_f32(src_data + 28 * src_step);
  float32x4_t src_data_35 = vld1q_f32(src_data + 29 * src_step);
  float32x4_t src_data_36 = vld1q_f32(src_data + 30 * src_step);
  float32x4_t src_data_37 = vld1q_f32(src_data + 31 * src_step);
  float32x4_t src_data_40 = vld1q_f32(src_data + 32 * src_step);
  float32x4_t src_data_41 = vld1q_f32(src_data + 33 * src_step);
  float32x4_t src_data_42 = vld1q_f32(src_data + 34 * src_step);
  float32x4_t src_data_43 = vld1q_f32(src_data + 35 * src_step);
  float32x4_t src_data_44 = vld1q_f32(src_data + 36 * src_step);
  float32x4_t src_data_45 = vld1q_f32(src_data + 37 * src_step);
  float32x4_t src_data_46 = vld1q_f32(src_data + 38 * src_step);
  float32x4_t src_data_47 = vld1q_f32(src_data + 39 * src_step);
  float32x4_t src_data_50 = vld1q_f32(src_data + 40 * src_step);
  float32x4_t src_data_51 = vld1q_f32(src_data + 41 * src_step);
  float32x4_t src_data_52 = vld1q_f32(src_data + 42 * src_step);
  float32x4_t src_data_53 = vld1q_f32(src_data + 43 * src_step);
  float32x4_t src_data_54 = vld1q_f32(src_data + 44 * src_step);
  float32x4_t src_data_55 = vld1q_f32(src_data + 45 * src_step);
  float32x4_t src_data_56 = vld1q_f32(src_data + 46 * src_step);
  float32x4_t src_data_57 = vld1q_f32(src_data + 47 * src_step);
  float32x4_t src_data_60 = vld1q_f32(src_data + 48 * src_step);
  float32x4_t src_data_61 = vld1q_f32(src_data + 49 * src_step);
  float32x4_t src_data_62 = vld1q_f32(src_data + 50 * src_step);
  float32x4_t src_data_63 = vld1q_f32(src_data + 51 * src_step);
  float32x4_t src_data_64 = vld1q_f32(src_data + 52 * src_step);
  float32x4_t src_data_65 = vld1q_f32(src_data + 53 * src_step);
  float32x4_t src_data_66 = vld1q_f32(src_data + 54 * src_step);
  float32x4_t src_data_67 = vld1q_f32(src_data + 55 * src_step);
  float32x4_t src_data_70 = vld1q_f32(src_data + 56 * src_step);
  float32x4_t src_data_71 = vld1q_f32(src_data + 57 * src_step);
  float32x4_t src_data_72 = vld1q_f32(src_data + 58 * src_step);
  float32x4_t src_data_73 = vld1q_f32(src_data + 59 * src_step);
  float32x4_t src_data_74 = vld1q_f32(src_data + 60 * src_step);
  float32x4_t src_data_75 = vld1q_f32(src_data + 61 * src_step);
  float32x4_t src_data_76 = vld1q_f32(src_data + 62 * src_step);
  float32x4_t src_data_77 = vld1q_f32(src_data + 63 * src_step);

  float32x4_t d01 = vsubq_f32(src_data_10, src_data_20);
  float32x4_t d02 = vsubq_f32(src_data_11, src_data_21);
  float32x4_t d03 = vsubq_f32(src_data_12, src_data_22);
  float32x4_t d04 = vsubq_f32(src_data_13, src_data_23);
  float32x4_t d05 = vsubq_f32(src_data_14, src_data_24);
  float32x4_t d06 = vsubq_f32(src_data_15, src_data_25);
  float32x4_t d07 = vsubq_f32(src_data_16, src_data_26);
  float32x4_t d08 = vsubq_f32(src_data_17, src_data_27);

  float32x4_t d11 = vsubq_f32(src_data_30, src_data_40);
  float32x4_t d12 = vsubq_f32(src_data_31, src_data_41);
  float32x4_t d13 = vsubq_f32(src_data_32, src_data_42);
  float32x4_t d14 = vsubq_f32(src_data_33, src_data_43);
  float32x4_t d15 = vsubq_f32(src_data_34, src_data_44);
  float32x4_t d16 = vsubq_f32(src_data_35, src_data_45);
  float32x4_t d17 = vsubq_f32(src_data_36, src_data_46);
  float32x4_t d18 = vsubq_f32(src_data_37, src_data_47);

  float32x4_t d21 = vsubq_f32(src_data_50, src_data_60);
  float32x4_t d22 = vsubq_f32(src_data_51, src_data_61);
  float32x4_t d23 = vsubq_f32(src_data_52, src_data_62);
  float32x4_t d24 = vsubq_f32(src_data_53, src_data_63);
  float32x4_t d25 = vsubq_f32(src_data_54, src_data_64);
  float32x4_t d26 = vsubq_f32(src_data_55, src_data_65);
  float32x4_t d27 = vsubq_f32(src_data_56, src_data_66);
  float32x4_t d28 = vsubq_f32(src_data_57, src_data_67);

  float32x4_t d31 = vaddq_f32(src_data_10, src_data_20);
  float32x4_t d32 = vaddq_f32(src_data_11, src_data_21);
  float32x4_t d33 = vaddq_f32(src_data_12, src_data_22);
  float32x4_t d34 = vaddq_f32(src_data_13, src_data_23);
  float32x4_t d35 = vaddq_f32(src_data_14, src_data_24);
  float32x4_t d36 = vaddq_f32(src_data_15, src_data_25);
  float32x4_t d37 = vaddq_f32(src_data_16, src_data_26);
  float32x4_t d38 = vaddq_f32(src_data_17, src_data_27);

  float32x4_t d41 = vaddq_f32(src_data_30, src_data_40);
  float32x4_t d42 = vaddq_f32(src_data_31, src_data_41);
  float32x4_t d43 = vaddq_f32(src_data_32, src_data_42);
  float32x4_t d44 = vaddq_f32(src_data_33, src_data_43);
  float32x4_t d45 = vaddq_f32(src_data_34, src_data_44);
  float32x4_t d46 = vaddq_f32(src_data_35, src_data_45);
  float32x4_t d47 = vaddq_f32(src_data_36, src_data_46);
  float32x4_t d48 = vaddq_f32(src_data_37, src_data_47);

  float32x4_t d51 = vaddq_f32(src_data_50, src_data_60);
  float32x4_t d52 = vaddq_f32(src_data_51, src_data_61);
  float32x4_t d53 = vaddq_f32(src_data_52, src_data_62);
  float32x4_t d54 = vaddq_f32(src_data_53, src_data_63);
  float32x4_t d55 = vaddq_f32(src_data_54, src_data_64);
  float32x4_t d56 = vaddq_f32(src_data_55, src_data_65);
  float32x4_t d57 = vaddq_f32(src_data_56, src_data_66);
  float32x4_t d58 = vaddq_f32(src_data_57, src_data_67);

  float32x4_t t00 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_00, src_data_10), src_data_20), src_data_30), src_data_40),
      src_data_50),
    src_data_60);
  float32x4_t t01 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_01, src_data_11), src_data_21), src_data_31), src_data_41),
      src_data_51),
    src_data_61);
  float32x4_t t02 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_02, src_data_12), src_data_22), src_data_32), src_data_42),
      src_data_52),
    src_data_62);
  float32x4_t t03 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_03, src_data_13), src_data_23), src_data_33), src_data_43),
      src_data_53),
    src_data_63);
  float32x4_t t04 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_04, src_data_14), src_data_24), src_data_34), src_data_44),
      src_data_54),
    src_data_64);
  float32x4_t t05 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_05, src_data_15), src_data_25), src_data_35), src_data_45),
      src_data_55),
    src_data_65);
  float32x4_t t06 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_06, src_data_16), src_data_26), src_data_36), src_data_46),
      src_data_56),
    src_data_66);
  float32x4_t t07 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_07, src_data_17), src_data_27), src_data_37), src_data_47),
      src_data_57),
    src_data_67);

  float32x4_t t10 = vaddq_f32(vaddq_f32(vmulq_n_f32(d01, 0.5), d11), vmulq_n_f32(d21, 1.5));
  float32x4_t t11 = vaddq_f32(vaddq_f32(vmulq_n_f32(d02, 0.5), d12), vmulq_n_f32(d22, 1.5));
  float32x4_t t12 = vaddq_f32(vaddq_f32(vmulq_n_f32(d03, 0.5), d13), vmulq_n_f32(d23, 1.5));
  float32x4_t t13 = vaddq_f32(vaddq_f32(vmulq_n_f32(d04, 0.5), d14), vmulq_n_f32(d24, 1.5));
  float32x4_t t14 = vaddq_f32(vaddq_f32(vmulq_n_f32(d05, 0.5), d15), vmulq_n_f32(d25, 1.5));
  float32x4_t t15 = vaddq_f32(vaddq_f32(vmulq_n_f32(d06, 0.5), d16), vmulq_n_f32(d26, 1.5));
  float32x4_t t16 = vaddq_f32(vaddq_f32(vmulq_n_f32(d07, 0.5), d17), vmulq_n_f32(d27, 1.5));
  float32x4_t t17 = vaddq_f32(vaddq_f32(vmulq_n_f32(d08, 0.5), d18), vmulq_n_f32(d28, 1.5));

  float32x4_t t20 = vaddq_f32(vaddq_f32(vmulq_n_f32(d31, 0.25), d41), vmulq_n_f32(d51, 2.25));
  float32x4_t t21 = vaddq_f32(vaddq_f32(vmulq_n_f32(d32, 0.25), d42), vmulq_n_f32(d52, 2.25));
  float32x4_t t22 = vaddq_f32(vaddq_f32(vmulq_n_f32(d33, 0.25), d43), vmulq_n_f32(d53, 2.25));
  float32x4_t t23 = vaddq_f32(vaddq_f32(vmulq_n_f32(d34, 0.25), d44), vmulq_n_f32(d54, 2.25));
  float32x4_t t24 = vaddq_f32(vaddq_f32(vmulq_n_f32(d35, 0.25), d45), vmulq_n_f32(d55, 2.25));
  float32x4_t t25 = vaddq_f32(vaddq_f32(vmulq_n_f32(d36, 0.25), d46), vmulq_n_f32(d56, 2.25));
  float32x4_t t26 = vaddq_f32(vaddq_f32(vmulq_n_f32(d37, 0.25), d47), vmulq_n_f32(d57, 2.25));
  float32x4_t t27 = vaddq_f32(vaddq_f32(vmulq_n_f32(d38, 0.25), d48), vmulq_n_f32(d58, 2.25));

  float32x4_t t30 = vaddq_f32(vaddq_f32(vmulq_n_f32(d01, 0.125), d11), vmulq_n_f32(d21, 3.375));
  float32x4_t t31 = vaddq_f32(vaddq_f32(vmulq_n_f32(d02, 0.125), d12), vmulq_n_f32(d22, 3.375));
  float32x4_t t32 = vaddq_f32(vaddq_f32(vmulq_n_f32(d03, 0.125), d13), vmulq_n_f32(d23, 3.375));
  float32x4_t t33 = vaddq_f32(vaddq_f32(vmulq_n_f32(d04, 0.125), d14), vmulq_n_f32(d24, 3.375));
  float32x4_t t34 = vaddq_f32(vaddq_f32(vmulq_n_f32(d05, 0.125), d15), vmulq_n_f32(d25, 3.375));
  float32x4_t t35 = vaddq_f32(vaddq_f32(vmulq_n_f32(d06, 0.125), d16), vmulq_n_f32(d26, 3.375));
  float32x4_t t36 = vaddq_f32(vaddq_f32(vmulq_n_f32(d07, 0.125), d17), vmulq_n_f32(d27, 3.375));
  float32x4_t t37 = vaddq_f32(vaddq_f32(vmulq_n_f32(d08, 0.125), d18), vmulq_n_f32(d28, 3.375));

  float32x4_t t40 = vaddq_f32(vaddq_f32(vmulq_n_f32(d31, 0.0625), d41), vmulq_n_f32(d51, 5.0625));
  float32x4_t t41 = vaddq_f32(vaddq_f32(vmulq_n_f32(d32, 0.0625), d42), vmulq_n_f32(d52, 5.0625));
  float32x4_t t42 = vaddq_f32(vaddq_f32(vmulq_n_f32(d33, 0.0625), d43), vmulq_n_f32(d53, 5.0625));
  float32x4_t t43 = vaddq_f32(vaddq_f32(vmulq_n_f32(d34, 0.0625), d44), vmulq_n_f32(d54, 5.0625));
  float32x4_t t44 = vaddq_f32(vaddq_f32(vmulq_n_f32(d35, 0.0625), d45), vmulq_n_f32(d55, 5.0625));
  float32x4_t t45 = vaddq_f32(vaddq_f32(vmulq_n_f32(d36, 0.0625), d46), vmulq_n_f32(d56, 5.0625));
  float32x4_t t46 = vaddq_f32(vaddq_f32(vmulq_n_f32(d37, 0.0625), d47), vmulq_n_f32(d57, 5.0625));
  float32x4_t t47 = vaddq_f32(vaddq_f32(vmulq_n_f32(d38, 0.0625), d48), vmulq_n_f32(d58, 5.0625));

  float32x4_t t50 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d01, 0.03125), d11), vmulq_n_f32(d21, 7.59375)), src_data_70);
  float32x4_t t51 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d02, 0.03125), d12), vmulq_n_f32(d22, 7.59375)), src_data_71);
  float32x4_t t52 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d03, 0.03125), d13), vmulq_n_f32(d23, 7.59375)), src_data_72);
  float32x4_t t53 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d04, 0.03125), d14), vmulq_n_f32(d24, 7.59375)), src_data_73);
  float32x4_t t54 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d05, 0.03125), d15), vmulq_n_f32(d25, 7.59375)), src_data_74);
  float32x4_t t55 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d06, 0.03125), d16), vmulq_n_f32(d26, 7.59375)), src_data_75);
  float32x4_t t56 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d07, 0.03125), d17), vmulq_n_f32(d27, 7.59375)), src_data_76);
  float32x4_t t57 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d08, 0.03125), d18), vmulq_n_f32(d28, 7.59375)), src_data_77);

  float32x4_t s11 = vsubq_f32(t01, t02);
  float32x4_t s12 = vsubq_f32(t11, t12);
  float32x4_t s13 = vsubq_f32(t21, t22);
  float32x4_t s14 = vsubq_f32(t31, t32);
  float32x4_t s15 = vsubq_f32(t41, t42);
  float32x4_t s16 = vsubq_f32(t51, t52);

  float32x4_t s21 = vsubq_f32(t03, t04);
  float32x4_t s22 = vsubq_f32(t13, t14);
  float32x4_t s23 = vsubq_f32(t23, t24);
  float32x4_t s24 = vsubq_f32(t33, t34);
  float32x4_t s25 = vsubq_f32(t43, t44);
  float32x4_t s26 = vsubq_f32(t53, t54);

  float32x4_t s31 = vsubq_f32(t05, t06);
  float32x4_t s32 = vsubq_f32(t15, t16);
  float32x4_t s33 = vsubq_f32(t25, t26);
  float32x4_t s34 = vsubq_f32(t35, t36);
  float32x4_t s35 = vsubq_f32(t45, t46);
  float32x4_t s36 = vsubq_f32(t55, t56);

  float32x4_t s41 = vaddq_f32(t01, t02);
  float32x4_t s42 = vaddq_f32(t11, t12);
  float32x4_t s43 = vaddq_f32(t21, t22);
  float32x4_t s44 = vaddq_f32(t31, t32);
  float32x4_t s45 = vaddq_f32(t41, t42);
  float32x4_t s46 = vaddq_f32(t51, t52);

  float32x4_t s51 = vaddq_f32(t03, t04);
  float32x4_t s52 = vaddq_f32(t13, t14);
  float32x4_t s53 = vaddq_f32(t23, t24);
  float32x4_t s54 = vaddq_f32(t33, t34);
  float32x4_t s55 = vaddq_f32(t43, t44);
  float32x4_t s56 = vaddq_f32(t53, t54);

  float32x4_t s61 = vaddq_f32(t05, t06);
  float32x4_t s62 = vaddq_f32(t15, t16);
  float32x4_t s63 = vaddq_f32(t25, t26);
  float32x4_t s64 = vaddq_f32(t35, t36);
  float32x4_t s65 = vaddq_f32(t45, t46);
  float32x4_t s66 = vaddq_f32(t55, t56);

  float32x4_t m00 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t00, t01), t02), t03), t04), t05), t06);
  float32x4_t m01 = vaddq_f32(vaddq_f32(vmulq_n_f32(s11, 0.5), s21), vmulq_n_f32(s31, 1.5));
  float32x4_t m02 = vaddq_f32(vaddq_f32(vmulq_n_f32(s41, 0.25), s51), vmulq_n_f32(s61, 2.25));
  float32x4_t m03 = vaddq_f32(vaddq_f32(vmulq_n_f32(s11, 0.125), s21), vmulq_n_f32(s31, 3.375));
  float32x4_t m04 = vaddq_f32(vaddq_f32(vmulq_n_f32(s41, 0.0625), s51), vmulq_n_f32(s61, 5.0625));
  float32x4_t m05 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s11, 0.03125), s21), vmulq_n_f32(s31, 7.59375)), t07);

  float32x4_t m10 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t10, t11), t12), t13), t14), t15), t16);
  float32x4_t m11 = vaddq_f32(vaddq_f32(vmulq_n_f32(s12, 0.5), s22), vmulq_n_f32(s32, 1.5));
  float32x4_t m12 = vaddq_f32(vaddq_f32(vmulq_n_f32(s42, 0.25), s52), vmulq_n_f32(s62, 2.25));
  float32x4_t m13 = vaddq_f32(vaddq_f32(vmulq_n_f32(s12, 0.125), s22), vmulq_n_f32(s32, 3.375));
  float32x4_t m14 = vaddq_f32(vaddq_f32(vmulq_n_f32(s42, 0.0625), s52), vmulq_n_f32(s62, 5.0625));
  float32x4_t m15 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s12, 0.03125), s22), vmulq_n_f32(s32, 7.59375)), t17);

  float32x4_t m20 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t20, t21), t22), t23), t24), t25), t26);
  float32x4_t m21 = vaddq_f32(vaddq_f32(vmulq_n_f32(s13, 0.5), s23), vmulq_n_f32(s33, 1.5));
  float32x4_t m22 = vaddq_f32(vaddq_f32(vmulq_n_f32(s43, 0.25), s53), vmulq_n_f32(s63, 2.25));
  float32x4_t m23 = vaddq_f32(vaddq_f32(vmulq_n_f32(s13, 0.125), s23), vmulq_n_f32(s33, 3.375));
  float32x4_t m24 = vaddq_f32(vaddq_f32(vmulq_n_f32(s43, 0.0625), s53), vmulq_n_f32(s63, 5.0625));
  float32x4_t m25 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s13, 0.03125), s23), vmulq_n_f32(s33, 7.59375)), t27);

  float32x4_t m30 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t30, t31), t32), t33), t34), t35), t36);
  float32x4_t m31 = vaddq_f32(vaddq_f32(vmulq_n_f32(s14, 0.5), s24), vmulq_n_f32(s34, 1.5));
  float32x4_t m32 = vaddq_f32(vaddq_f32(vmulq_n_f32(s44, 0.25), s54), vmulq_n_f32(s64, 2.25));
  float32x4_t m33 = vaddq_f32(vaddq_f32(vmulq_n_f32(s14, 0.125), s24), vmulq_n_f32(s34, 3.375));
  float32x4_t m34 = vaddq_f32(vaddq_f32(vmulq_n_f32(s44, 0.0625), s54), vmulq_n_f32(s64, 5.0625));
  float32x4_t m35 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s14, 0.03125), s24), vmulq_n_f32(s34, 7.59375)), t37);

  float32x4_t m40 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t40, t41), t42), t43), t44), t45), t46);
  float32x4_t m41 = vaddq_f32(vaddq_f32(vmulq_n_f32(s15, 0.5), s25), vmulq_n_f32(s35, 1.5));
  float32x4_t m42 = vaddq_f32(vaddq_f32(vmulq_n_f32(s45, 0.25), s55), vmulq_n_f32(s65, 2.25));
  float32x4_t m43 = vaddq_f32(vaddq_f32(vmulq_n_f32(s15, 0.125), s25), vmulq_n_f32(s35, 3.375));
  float32x4_t m44 = vaddq_f32(vaddq_f32(vmulq_n_f32(s45, 0.0625), s55), vmulq_n_f32(s65, 5.0625));
  float32x4_t m45 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s15, 0.03125), s25), vmulq_n_f32(s35, 7.59375)), t47);

  float32x4_t m50 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t50, t51), t52), t53), t54), t55), t56);
  float32x4_t m51 = vaddq_f32(vaddq_f32(vmulq_n_f32(s16, 0.5), s26), vmulq_n_f32(s36, 1.5));
  float32x4_t m52 = vaddq_f32(vaddq_f32(vmulq_n_f32(s46, 0.25), s56), vmulq_n_f32(s66, 2.25));
  float32x4_t m53 = vaddq_f32(vaddq_f32(vmulq_n_f32(s16, 0.125), s26), vmulq_n_f32(s36, 3.375));
  float32x4_t m54 = vaddq_f32(vaddq_f32(vmulq_n_f32(s46, 0.0625), s56), vmulq_n_f32(s66, 5.0625));
  float32x4_t m55 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s16, 0.03125), s26), vmulq_n_f32(s36, 7.59375)), t57);

  float32x4_t bias_ptr = vld1q_f32(bias_data);
  vst1q_f32(dst_data, vaddq_f32(m00, bias_ptr));
  vst1q_f32(dst_data + C4NUM, vaddq_f32(m01, bias_ptr));
  vst1q_f32(dst_data + 2 * C4NUM, vaddq_f32(m02, bias_ptr));
  vst1q_f32(dst_data + 3 * C4NUM, vaddq_f32(m03, bias_ptr));
  vst1q_f32(dst_data + 4 * C4NUM, vaddq_f32(m04, bias_ptr));
  vst1q_f32(dst_data + 5 * C4NUM, vaddq_f32(m05, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM, vaddq_f32(m10, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM + C4NUM, vaddq_f32(m11, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM + 2 * C4NUM, vaddq_f32(m12, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM + 3 * C4NUM, vaddq_f32(m13, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM + 4 * C4NUM, vaddq_f32(m14, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM + 5 * C4NUM, vaddq_f32(m15, bias_ptr));
  vst1q_f32(dst_data + 2 * dst_step * C4NUM, vaddq_f32(m20, bias_ptr));
  vst1q_f32(dst_data + 2 * dst_step * C4NUM + C4NUM, vaddq_f32(m21, bias_ptr));
  vst1q_f32(dst_data + 2 * dst_step * C4NUM + 2 * C4NUM, vaddq_f32(m22, bias_ptr));
  vst1q_f32(dst_data + 2 * dst_step * C4NUM + 3 * C4NUM, vaddq_f32(m23, bias_ptr));
  vst1q_f32(dst_data + 2 * dst_step * C4NUM + 4 * C4NUM, vaddq_f32(m24, bias_ptr));
  vst1q_f32(dst_data + 2 * dst_step * C4NUM + 5 * C4NUM, vaddq_f32(m25, bias_ptr));
  vst1q_f32(dst_data + 3 * dst_step * C4NUM, vaddq_f32(m30, bias_ptr));
  vst1q_f32(dst_data + 3 * dst_step * C4NUM + C4NUM, vaddq_f32(m31, bias_ptr));
  vst1q_f32(dst_data + 3 * dst_step * C4NUM + 2 * C4NUM, vaddq_f32(m32, bias_ptr));
  vst1q_f32(dst_data + 3 * dst_step * C4NUM + 3 * C4NUM, vaddq_f32(m33, bias_ptr));
  vst1q_f32(dst_data + 3 * dst_step * C4NUM + 4 * C4NUM, vaddq_f32(m34, bias_ptr));
  vst1q_f32(dst_data + 3 * dst_step * C4NUM + 5 * C4NUM, vaddq_f32(m35, bias_ptr));
  vst1q_f32(dst_data + 4 * dst_step * C4NUM, vaddq_f32(m40, bias_ptr));
  vst1q_f32(dst_data + 4 * dst_step * C4NUM + C4NUM, vaddq_f32(m41, bias_ptr));
  vst1q_f32(dst_data + 4 * dst_step * C4NUM + 2 * C4NUM, vaddq_f32(m42, bias_ptr));
  vst1q_f32(dst_data + 4 * dst_step * C4NUM + 3 * C4NUM, vaddq_f32(m43, bias_ptr));
  vst1q_f32(dst_data + 4 * dst_step * C4NUM + 4 * C4NUM, vaddq_f32(m44, bias_ptr));
  vst1q_f32(dst_data + 4 * dst_step * C4NUM + 5 * C4NUM, vaddq_f32(m45, bias_ptr));
  vst1q_f32(dst_data + 5 * dst_step * C4NUM, vaddq_f32(m50, bias_ptr));
  vst1q_f32(dst_data + 5 * dst_step * C4NUM + C4NUM, vaddq_f32(m51, bias_ptr));
  vst1q_f32(dst_data + 5 * dst_step * C4NUM + 2 * C4NUM, vaddq_f32(m52, bias_ptr));
  vst1q_f32(dst_data + 5 * dst_step * C4NUM + 3 * C4NUM, vaddq_f32(m53, bias_ptr));
  vst1q_f32(dst_data + 5 * dst_step * C4NUM + 4 * C4NUM, vaddq_f32(m54, bias_ptr));
  vst1q_f32(dst_data + 5 * dst_step * C4NUM + 5 * C4NUM, vaddq_f32(m55, bias_ptr));
#else
  for (int i = 0; i < C4NUM; i++) {
    float src_data_00 = src_data[i];
    float src_data_01 = src_data[i + src_step];
    float src_data_02 = src_data[i + 2 * src_step];
    float src_data_03 = src_data[i + 3 * src_step];
    float src_data_04 = src_data[i + 4 * src_step];
    float src_data_05 = src_data[i + 5 * src_step];
    float src_data_06 = src_data[i + 6 * src_step];
    float src_data_07 = src_data[i + 7 * src_step];
    float src_data_10 = src_data[i + 8 * src_step];
    float src_data_11 = src_data[i + 9 * src_step];
    float src_data_12 = src_data[i + 10 * src_step];
    float src_data_13 = src_data[i + 11 * src_step];
    float src_data_14 = src_data[i + 12 * src_step];
    float src_data_15 = src_data[i + 13 * src_step];
    float src_data_16 = src_data[i + 14 * src_step];
    float src_data_17 = src_data[i + 15 * src_step];
    float src_data_20 = src_data[i + 16 * src_step];
    float src_data_21 = src_data[i + 17 * src_step];
    float src_data_22 = src_data[i + 18 * src_step];
    float src_data_23 = src_data[i + 19 * src_step];
    float src_data_24 = src_data[i + 20 * src_step];
    float src_data_25 = src_data[i + 21 * src_step];
    float src_data_26 = src_data[i + 22 * src_step];
    float src_data_27 = src_data[i + 23 * src_step];
    float src_data_30 = src_data[i + 24 * src_step];
    float src_data_31 = src_data[i + 25 * src_step];
    float src_data_32 = src_data[i + 26 * src_step];
    float src_data_33 = src_data[i + 27 * src_step];
    float src_data_34 = src_data[i + 28 * src_step];
    float src_data_35 = src_data[i + 29 * src_step];
    float src_data_36 = src_data[i + 30 * src_step];
    float src_data_37 = src_data[i + 31 * src_step];
    float src_data_40 = src_data[i + 32 * src_step];
    float src_data_41 = src_data[i + 33 * src_step];
    float src_data_42 = src_data[i + 34 * src_step];
    float src_data_43 = src_data[i + 35 * src_step];
    float src_data_44 = src_data[i + 36 * src_step];
    float src_data_45 = src_data[i + 37 * src_step];
    float src_data_46 = src_data[i + 38 * src_step];
    float src_data_47 = src_data[i + 39 * src_step];
    float src_data_50 = src_data[i + 40 * src_step];
    float src_data_51 = src_data[i + 41 * src_step];
    float src_data_52 = src_data[i + 42 * src_step];
    float src_data_53 = src_data[i + 43 * src_step];
    float src_data_54 = src_data[i + 44 * src_step];
    float src_data_55 = src_data[i + 45 * src_step];
    float src_data_56 = src_data[i + 46 * src_step];
    float src_data_57 = src_data[i + 47 * src_step];
    float src_data_60 = src_data[i + 48 * src_step];
    float src_data_61 = src_data[i + 49 * src_step];
    float src_data_62 = src_data[i + 50 * src_step];
    float src_data_63 = src_data[i + 51 * src_step];
    float src_data_64 = src_data[i + 52 * src_step];
    float src_data_65 = src_data[i + 53 * src_step];
    float src_data_66 = src_data[i + 54 * src_step];
    float src_data_67 = src_data[i + 55 * src_step];
    float src_data_70 = src_data[i + 56 * src_step];
    float src_data_71 = src_data[i + 57 * src_step];
    float src_data_72 = src_data[i + 58 * src_step];
    float src_data_73 = src_data[i + 59 * src_step];
    float src_data_74 = src_data[i + 60 * src_step];
    float src_data_75 = src_data[i + 61 * src_step];
    float src_data_76 = src_data[i + 62 * src_step];
    float src_data_77 = src_data[i + 63 * src_step];

    float d01 = src_data_10 - src_data_20;
    float d02 = src_data_11 - src_data_21;
    float d03 = src_data_12 - src_data_22;
    float d04 = src_data_13 - src_data_23;
    float d05 = src_data_14 - src_data_24;
    float d06 = src_data_15 - src_data_25;
    float d07 = src_data_16 - src_data_26;
    float d08 = src_data_17 - src_data_27;

    float d11 = src_data_30 - src_data_40;
    float d12 = src_data_31 - src_data_41;
    float d13 = src_data_32 - src_data_42;
    float d14 = src_data_33 - src_data_43;
    float d15 = src_data_34 - src_data_44;
    float d16 = src_data_35 - src_data_45;
    float d17 = src_data_36 - src_data_46;
    float d18 = src_data_37 - src_data_47;

    float d21 = src_data_50 - src_data_60;
    float d22 = src_data_51 - src_data_61;
    float d23 = src_data_52 - src_data_62;
    float d24 = src_data_53 - src_data_63;
    float d25 = src_data_54 - src_data_64;
    float d26 = src_data_55 - src_data_65;
    float d27 = src_data_56 - src_data_66;
    float d28 = src_data_57 - src_data_67;

    float d31 = src_data_10 + src_data_20;
    float d32 = src_data_11 + src_data_21;
    float d33 = src_data_12 + src_data_22;
    float d34 = src_data_13 + src_data_23;
    float d35 = src_data_14 + src_data_24;
    float d36 = src_data_15 + src_data_25;
    float d37 = src_data_16 + src_data_26;
    float d38 = src_data_17 + src_data_27;

    float d41 = src_data_30 + src_data_40;
    float d42 = src_data_31 + src_data_41;
    float d43 = src_data_32 + src_data_42;
    float d44 = src_data_33 + src_data_43;
    float d45 = src_data_34 + src_data_44;
    float d46 = src_data_35 + src_data_45;
    float d47 = src_data_36 + src_data_46;
    float d48 = src_data_37 + src_data_47;

    float d51 = src_data_50 + src_data_60;
    float d52 = src_data_51 + src_data_61;
    float d53 = src_data_52 + src_data_62;
    float d54 = src_data_53 + src_data_63;
    float d55 = src_data_54 + src_data_64;
    float d56 = src_data_55 + src_data_65;
    float d57 = src_data_56 + src_data_66;
    float d58 = src_data_57 + src_data_67;

    float t00 = src_data_00 + src_data_10 + src_data_20 + src_data_30 + src_data_40 + src_data_50 + src_data_60;
    float t01 = src_data_01 + src_data_11 + src_data_21 + src_data_31 + src_data_41 + src_data_51 + src_data_61;
    float t02 = src_data_02 + src_data_12 + src_data_22 + src_data_32 + src_data_42 + src_data_52 + src_data_62;
    float t03 = src_data_03 + src_data_13 + src_data_23 + src_data_33 + src_data_43 + src_data_53 + src_data_63;
    float t04 = src_data_04 + src_data_14 + src_data_24 + src_data_34 + src_data_44 + src_data_54 + src_data_64;
    float t05 = src_data_05 + src_data_15 + src_data_25 + src_data_35 + src_data_45 + src_data_55 + src_data_65;
    float t06 = src_data_06 + src_data_16 + src_data_26 + src_data_36 + src_data_46 + src_data_56 + src_data_66;
    float t07 = src_data_07 + src_data_17 + src_data_27 + src_data_37 + src_data_47 + src_data_57 + src_data_67;

    const float t10 = 0.5f * d01 + d11 + 1.5f * d21;
    const float t11 = 0.5f * d02 + d12 + 1.5f * d22;
    const float t12 = 0.5f * d03 + d13 + 1.5f * d23;
    const float t13 = 0.5f * d04 + d14 + 1.5f * d24;
    const float t14 = 0.5f * d05 + d15 + 1.5f * d25;
    const float t15 = 0.5f * d06 + d16 + 1.5f * d26;
    const float t16 = 0.5f * d07 + d17 + 1.5f * d27;
    const float t17 = 0.5f * d08 + d18 + 1.5f * d28;

    const float t20 = 0.25f * d31 + d41 + 2.25f * d51;
    const float t21 = 0.25f * d32 + d42 + 2.25f * d52;
    const float t22 = 0.25f * d33 + d43 + 2.25f * d53;
    const float t23 = 0.25f * d34 + d44 + 2.25f * d54;
    const float t24 = 0.25f * d35 + d45 + 2.25f * d55;
    const float t25 = 0.25f * d36 + d46 + 2.25f * d56;
    const float t26 = 0.25f * d37 + d47 + 2.25f * d57;
    const float t27 = 0.25f * d38 + d48 + 2.25f * d58;

    const float t30 = 0.125f * d01 + d11 + 3.375f * d21;
    const float t31 = 0.125f * d02 + d12 + 3.375f * d22;
    const float t32 = 0.125f * d03 + d13 + 3.375f * d23;
    const float t33 = 0.125f * d04 + d14 + 3.375f * d24;
    const float t34 = 0.125f * d05 + d15 + 3.375f * d25;
    const float t35 = 0.125f * d06 + d16 + 3.375f * d26;
    const float t36 = 0.125f * d07 + d17 + 3.375f * d27;
    const float t37 = 0.125f * d08 + d18 + 3.375f * d28;

    const float t40 = 0.0625f * d31 + d41 + 5.0625f * d51;
    const float t41 = 0.0625f * d32 + d42 + 5.0625f * d52;
    const float t42 = 0.0625f * d33 + d43 + 5.0625f * d53;
    const float t43 = 0.0625f * d34 + d44 + 5.0625f * d54;
    const float t44 = 0.0625f * d35 + d45 + 5.0625f * d55;
    const float t45 = 0.0625f * d36 + d46 + 5.0625f * d56;
    const float t46 = 0.0625f * d37 + d47 + 5.0625f * d57;
    const float t47 = 0.0625f * d38 + d48 + 5.0625f * d58;

    const float t50 = 0.03125f * d01 + d11 + 7.59375f * d21 + src_data_70;
    const float t51 = 0.03125f * d02 + d12 + 7.59375f * d22 + src_data_71;
    const float t52 = 0.03125f * d03 + d13 + 7.59375f * d23 + src_data_72;
    const float t53 = 0.03125f * d04 + d14 + 7.59375f * d24 + src_data_73;
    const float t54 = 0.03125f * d05 + d15 + 7.59375f * d25 + src_data_74;
    const const float t55 = 0.03125f * d06 + d16 + 7.59375f * d26 + src_data_75;
    const float t56 = 0.03125f * d07 + d17 + 7.59375f * d27 + src_data_76;
    const float t57 = 0.03125f * d08 + d18 + 7.59375f * d28 + src_data_77;

    float s11 = t01 - t02;
    float s12 = t11 - t12;
    float s13 = t21 - t22;
    float s14 = t31 - t32;
    float s15 = t41 - t42;
    float s16 = t51 - t52;

    float s21 = t03 - t04;
    float s22 = t13 - t14;
    float s23 = t23 - t24;
    float s24 = t33 - t34;
    float s25 = t43 - t44;
    float s26 = t53 - t54;

    float s31 = t05 - t06;
    float s32 = t15 - t16;
    float s33 = t25 - t26;
    float s34 = t35 - t36;
    float s35 = t45 - t46;
    float s36 = t55 - t56;

    float s41 = t01 + t02;
    float s42 = t11 + t12;
    float s43 = t21 + t22;
    float s44 = t31 + t32;
    float s45 = t41 + t42;
    float s46 = t51 + t52;

    float s51 = t03 + t04;
    float s52 = t13 + t14;
    float s53 = t23 + t24;
    float s54 = t33 + t34;
    float s55 = t43 + t44;
    float s56 = t53 + t54;

    float s61 = t05 + t06;
    float s62 = t15 + t16;
    float s63 = t25 + t26;
    float s64 = t35 + t36;
    float s65 = t45 + t46;
    float s66 = t55 + t56;

    float m00 = t00 + t01 + t02 + t03 + t04 + t05 + t06;
    const float m01 = 0.5f * s11 + s21 + 1.5f * s31;
    const float m02 = 0.25f * s41 + s51 + 2.25f * s61;
    const float m03 = 0.125f * s11 + s21 + 3.375f * s31;
    const float m04 = 0.0625f * s41 + s51 + 5.0625f * s61;
    const float m05 = 0.03125f * s11 + s21 + 7.59375f * s31 + t07;

    float m10 = t10 + t11 + t12 + t13 + t14 + t15 + t16;
    const float m11 = 0.5f * s12 + s22 + 1.5f * s32;
    const float m12 = 0.25f * s42 + s52 + 2.25f * s62;
    const float m13 = 0.125f * s12 + s22 + 3.375f * s32;
    const float m14 = 0.0625f * s42 + s52 + 5.0625f * s62;
    const float m15 = 0.03125f * s12 + s22 + 7.59375f * s32 + t17;

    float m20 = t20 + t21 + t22 + t23 + t24 + t25 + t26;
    const float m21 = 0.5f * s13 + s23 + 1.5f * s33;
    const float m22 = 0.25f * s43 + s53 + 2.25f * s63;
    const float m23 = 0.125f * s13 + s23 + 3.375f * s33;
    const float m24 = 0.0625f * s43 + s53 + 5.0625f * s63;
    const float m25 = 0.03125f * s13 + s23 + 7.59375f * s33 + t27;

    float m30 = t30 + t31 + t32 + t33 + t34 + t35 + t36;
    const float m31 = 0.5f * s14 + s24 + 1.5f * s34;
    const float m32 = 0.25f * s44 + s54 + 2.25f * s64;
    const float m33 = 0.125f * s14 + s24 + 3.375f * s34;
    const float m34 = 0.0625f * s44 + s54 + 5.0625f * s64;
    const float m35 = 0.03125f * s14 + s24 + 7.59375f * s34 + t37;

    float m40 = t40 + t41 + t42 + t43 + t44 + t45 + t46;
    const float m41 = 0.5f * s15 + s25 + 1.5f * s35;
    const float m42 = 0.25f * s45 + s55 + 2.25f * s65;
    const float m43 = 0.125f * s15 + s25 + 3.375f * s35;
    const float m44 = 0.0625f * s45 + s55 + 5.0625f * s65;
    const float m45 = 0.03125f * s15 + s25 + 7.59375f * s35 + t47;

    float m50 = t50 + t51 + t52 + t53 + t54 + t55 + t56;
    const float m51 = 0.5f * s16 + s26 + 1.5f * s36;
    const float m52 = 0.25f * s46 + s56 + 2.25f * s66;
    const float m53 = 0.125f * s16 + s26 + 3.375f * s36;
    const float m54 = 0.0625f * s46 + s56 + 5.0625f * s66;
    const float m55 = 0.03125f * s16 + s26 + 7.59375f * s36 + t57;

    (dst_data + i)[0] = m00 + bias_data[i];
    (dst_data + i + C4NUM)[0] = m01 + bias_data[i];
    (dst_data + i + 2 * C4NUM)[0] = m02 + bias_data[i];
    (dst_data + i + 3 * C4NUM)[0] = m03 + bias_data[i];
    (dst_data + i + 4 * C4NUM)[0] = m04 + bias_data[i];
    (dst_data + i + 5 * C4NUM)[0] = m05 + bias_data[i];

    (dst_data + i + dst_step * C4NUM)[0] = m10 + bias_data[i];
    (dst_data + i + dst_step * C4NUM + C4NUM)[0] = m11 + bias_data[i];
    (dst_data + i + dst_step * C4NUM + 2 * C4NUM)[0] = m12 + bias_data[i];
    (dst_data + i + dst_step * C4NUM + 3 * C4NUM)[0] = m13 + bias_data[i];
    (dst_data + i + dst_step * C4NUM + 4 * C4NUM)[0] = m14 + bias_data[i];
    (dst_data + i + dst_step * C4NUM + 5 * C4NUM)[0] = m15 + bias_data[i];

    (dst_data + i + 2 * dst_step * C4NUM)[0] = m20 + bias_data[i];
    (dst_data + i + 2 * dst_step * C4NUM + C4NUM)[0] = m21 + bias_data[i];
    (dst_data + i + 2 * dst_step * C4NUM + 2 * C4NUM)[0] = m22 + bias_data[i];
    (dst_data + i + 2 * dst_step * C4NUM + 3 * C4NUM)[0] = m23 + bias_data[i];
    (dst_data + i + 2 * dst_step * C4NUM + 4 * C4NUM)[0] = m24 + bias_data[i];
    (dst_data + i + 2 * dst_step * C4NUM + 5 * C4NUM)[0] = m25 + bias_data[i];

    (dst_data + i + 3 * dst_step * C4NUM)[0] = m30 + bias_data[i];
    (dst_data + i + 3 * dst_step * C4NUM + C4NUM)[0] = m31 + bias_data[i];
    (dst_data + i + 3 * dst_step * C4NUM + 2 * C4NUM)[0] = m32 + bias_data[i];
    (dst_data + i + 3 * dst_step * C4NUM + 3 * C4NUM)[0] = m33 + bias_data[i];
    (dst_data + i + 3 * dst_step * C4NUM + 4 * C4NUM)[0] = m34 + bias_data[i];
    (dst_data + i + 3 * dst_step * C4NUM + 5 * C4NUM)[0] = m35 + bias_data[i];

    (dst_data + i + 4 * dst_step * C4NUM)[0] = m40 + bias_data[i];
    (dst_data + i + 4 * dst_step * C4NUM + C4NUM)[0] = m41 + bias_data[i];
    (dst_data + i + 4 * dst_step * C4NUM + 2 * C4NUM)[0] = m42 + bias_data[i];
    (dst_data + i + 4 * dst_step * C4NUM + 3 * C4NUM)[0] = m43 + bias_data[i];
    (dst_data + i + 4 * dst_step * C4NUM + 4 * C4NUM)[0] = m44 + bias_data[i];
    (dst_data + i + 4 * dst_step * C4NUM + 5 * C4NUM)[0] = m45 + bias_data[i];

    (dst_data + i + 5 * dst_step * C4NUM)[0] = m50 + bias_data[i];
    (dst_data + i + 5 * dst_step * C4NUM + C4NUM)[0] = m51 + bias_data[i];
    (dst_data + i + 5 * dst_step * C4NUM + 2 * C4NUM)[0] = m52 + bias_data[i];
    (dst_data + i + 5 * dst_step * C4NUM + 3 * C4NUM)[0] = m53 + bias_data[i];
    (dst_data + i + 5 * dst_step * C4NUM + 4 * C4NUM)[0] = m54 + bias_data[i];
    (dst_data + i + 5 * dst_step * C4NUM + 5 * C4NUM)[0] = m55 + bias_data[i];
  }
#endif
}

void OutputTransform8x7Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                            int dst_step) {
#ifdef ENABLE_ARM
  float32x4_t src_data_00 = vld1q_f32(src_data + 0 * src_step);
  float32x4_t src_data_01 = vld1q_f32(src_data + 1 * src_step);
  float32x4_t src_data_02 = vld1q_f32(src_data + 2 * src_step);
  float32x4_t src_data_03 = vld1q_f32(src_data + 3 * src_step);
  float32x4_t src_data_04 = vld1q_f32(src_data + 4 * src_step);
  float32x4_t src_data_05 = vld1q_f32(src_data + 5 * src_step);
  float32x4_t src_data_06 = vld1q_f32(src_data + 6 * src_step);
  float32x4_t src_data_07 = vld1q_f32(src_data + 7 * src_step);
  float32x4_t src_data_10 = vld1q_f32(src_data + 8 * src_step);
  float32x4_t src_data_11 = vld1q_f32(src_data + 9 * src_step);
  float32x4_t src_data_12 = vld1q_f32(src_data + 10 * src_step);
  float32x4_t src_data_13 = vld1q_f32(src_data + 11 * src_step);
  float32x4_t src_data_14 = vld1q_f32(src_data + 12 * src_step);
  float32x4_t src_data_15 = vld1q_f32(src_data + 13 * src_step);
  float32x4_t src_data_16 = vld1q_f32(src_data + 14 * src_step);
  float32x4_t src_data_17 = vld1q_f32(src_data + 15 * src_step);
  float32x4_t src_data_20 = vld1q_f32(src_data + 16 * src_step);
  float32x4_t src_data_21 = vld1q_f32(src_data + 17 * src_step);
  float32x4_t src_data_22 = vld1q_f32(src_data + 18 * src_step);
  float32x4_t src_data_23 = vld1q_f32(src_data + 19 * src_step);
  float32x4_t src_data_24 = vld1q_f32(src_data + 20 * src_step);
  float32x4_t src_data_25 = vld1q_f32(src_data + 21 * src_step);
  float32x4_t src_data_26 = vld1q_f32(src_data + 22 * src_step);
  float32x4_t src_data_27 = vld1q_f32(src_data + 23 * src_step);
  float32x4_t src_data_30 = vld1q_f32(src_data + 24 * src_step);
  float32x4_t src_data_31 = vld1q_f32(src_data + 25 * src_step);
  float32x4_t src_data_32 = vld1q_f32(src_data + 26 * src_step);
  float32x4_t src_data_33 = vld1q_f32(src_data + 27 * src_step);
  float32x4_t src_data_34 = vld1q_f32(src_data + 28 * src_step);
  float32x4_t src_data_35 = vld1q_f32(src_data + 29 * src_step);
  float32x4_t src_data_36 = vld1q_f32(src_data + 30 * src_step);
  float32x4_t src_data_37 = vld1q_f32(src_data + 31 * src_step);
  float32x4_t src_data_40 = vld1q_f32(src_data + 32 * src_step);
  float32x4_t src_data_41 = vld1q_f32(src_data + 33 * src_step);
  float32x4_t src_data_42 = vld1q_f32(src_data + 34 * src_step);
  float32x4_t src_data_43 = vld1q_f32(src_data + 35 * src_step);
  float32x4_t src_data_44 = vld1q_f32(src_data + 36 * src_step);
  float32x4_t src_data_45 = vld1q_f32(src_data + 37 * src_step);
  float32x4_t src_data_46 = vld1q_f32(src_data + 38 * src_step);
  float32x4_t src_data_47 = vld1q_f32(src_data + 39 * src_step);
  float32x4_t src_data_50 = vld1q_f32(src_data + 40 * src_step);
  float32x4_t src_data_51 = vld1q_f32(src_data + 41 * src_step);
  float32x4_t src_data_52 = vld1q_f32(src_data + 42 * src_step);
  float32x4_t src_data_53 = vld1q_f32(src_data + 43 * src_step);
  float32x4_t src_data_54 = vld1q_f32(src_data + 44 * src_step);
  float32x4_t src_data_55 = vld1q_f32(src_data + 45 * src_step);
  float32x4_t src_data_56 = vld1q_f32(src_data + 46 * src_step);
  float32x4_t src_data_57 = vld1q_f32(src_data + 47 * src_step);
  float32x4_t src_data_60 = vld1q_f32(src_data + 48 * src_step);
  float32x4_t src_data_61 = vld1q_f32(src_data + 49 * src_step);
  float32x4_t src_data_62 = vld1q_f32(src_data + 50 * src_step);
  float32x4_t src_data_63 = vld1q_f32(src_data + 51 * src_step);
  float32x4_t src_data_64 = vld1q_f32(src_data + 52 * src_step);
  float32x4_t src_data_65 = vld1q_f32(src_data + 53 * src_step);
  float32x4_t src_data_66 = vld1q_f32(src_data + 54 * src_step);
  float32x4_t src_data_67 = vld1q_f32(src_data + 55 * src_step);
  float32x4_t src_data_70 = vld1q_f32(src_data + 56 * src_step);
  float32x4_t src_data_71 = vld1q_f32(src_data + 57 * src_step);
  float32x4_t src_data_72 = vld1q_f32(src_data + 58 * src_step);
  float32x4_t src_data_73 = vld1q_f32(src_data + 59 * src_step);
  float32x4_t src_data_74 = vld1q_f32(src_data + 60 * src_step);
  float32x4_t src_data_75 = vld1q_f32(src_data + 61 * src_step);
  float32x4_t src_data_76 = vld1q_f32(src_data + 62 * src_step);
  float32x4_t src_data_77 = vld1q_f32(src_data + 63 * src_step);

  float32x4_t d01 = vsubq_f32(src_data_10, src_data_20);
  float32x4_t d02 = vsubq_f32(src_data_11, src_data_21);
  float32x4_t d03 = vsubq_f32(src_data_12, src_data_22);
  float32x4_t d04 = vsubq_f32(src_data_13, src_data_23);
  float32x4_t d05 = vsubq_f32(src_data_14, src_data_24);
  float32x4_t d06 = vsubq_f32(src_data_15, src_data_25);
  float32x4_t d07 = vsubq_f32(src_data_16, src_data_26);
  float32x4_t d08 = vsubq_f32(src_data_17, src_data_27);

  float32x4_t d11 = vsubq_f32(src_data_30, src_data_40);
  float32x4_t d12 = vsubq_f32(src_data_31, src_data_41);
  float32x4_t d13 = vsubq_f32(src_data_32, src_data_42);
  float32x4_t d14 = vsubq_f32(src_data_33, src_data_43);
  float32x4_t d15 = vsubq_f32(src_data_34, src_data_44);
  float32x4_t d16 = vsubq_f32(src_data_35, src_data_45);
  float32x4_t d17 = vsubq_f32(src_data_36, src_data_46);
  float32x4_t d18 = vsubq_f32(src_data_37, src_data_47);

  float32x4_t d21 = vsubq_f32(src_data_50, src_data_60);
  float32x4_t d22 = vsubq_f32(src_data_51, src_data_61);
  float32x4_t d23 = vsubq_f32(src_data_52, src_data_62);
  float32x4_t d24 = vsubq_f32(src_data_53, src_data_63);
  float32x4_t d25 = vsubq_f32(src_data_54, src_data_64);
  float32x4_t d26 = vsubq_f32(src_data_55, src_data_65);
  float32x4_t d27 = vsubq_f32(src_data_56, src_data_66);
  float32x4_t d28 = vsubq_f32(src_data_57, src_data_67);

  float32x4_t d31 = vaddq_f32(src_data_10, src_data_20);
  float32x4_t d32 = vaddq_f32(src_data_11, src_data_21);
  float32x4_t d33 = vaddq_f32(src_data_12, src_data_22);
  float32x4_t d34 = vaddq_f32(src_data_13, src_data_23);
  float32x4_t d35 = vaddq_f32(src_data_14, src_data_24);
  float32x4_t d36 = vaddq_f32(src_data_15, src_data_25);
  float32x4_t d37 = vaddq_f32(src_data_16, src_data_26);
  float32x4_t d38 = vaddq_f32(src_data_17, src_data_27);

  float32x4_t d41 = vaddq_f32(src_data_30, src_data_40);
  float32x4_t d42 = vaddq_f32(src_data_31, src_data_41);
  float32x4_t d43 = vaddq_f32(src_data_32, src_data_42);
  float32x4_t d44 = vaddq_f32(src_data_33, src_data_43);
  float32x4_t d45 = vaddq_f32(src_data_34, src_data_44);
  float32x4_t d46 = vaddq_f32(src_data_35, src_data_45);
  float32x4_t d47 = vaddq_f32(src_data_36, src_data_46);
  float32x4_t d48 = vaddq_f32(src_data_37, src_data_47);

  float32x4_t d51 = vaddq_f32(src_data_50, src_data_60);
  float32x4_t d52 = vaddq_f32(src_data_51, src_data_61);
  float32x4_t d53 = vaddq_f32(src_data_52, src_data_62);
  float32x4_t d54 = vaddq_f32(src_data_53, src_data_63);
  float32x4_t d55 = vaddq_f32(src_data_54, src_data_64);
  float32x4_t d56 = vaddq_f32(src_data_55, src_data_65);
  float32x4_t d57 = vaddq_f32(src_data_56, src_data_66);
  float32x4_t d58 = vaddq_f32(src_data_57, src_data_67);

  float32x4_t t00 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_00, src_data_10), src_data_20), src_data_30), src_data_40),
      src_data_50),
    src_data_60);
  float32x4_t t01 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_01, src_data_11), src_data_21), src_data_31), src_data_41),
      src_data_51),
    src_data_61);
  float32x4_t t02 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_02, src_data_12), src_data_22), src_data_32), src_data_42),
      src_data_52),
    src_data_62);
  float32x4_t t03 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_03, src_data_13), src_data_23), src_data_33), src_data_43),
      src_data_53),
    src_data_63);
  float32x4_t t04 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_04, src_data_14), src_data_24), src_data_34), src_data_44),
      src_data_54),
    src_data_64);
  float32x4_t t05 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_05, src_data_15), src_data_25), src_data_35), src_data_45),
      src_data_55),
    src_data_65);
  float32x4_t t06 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_06, src_data_16), src_data_26), src_data_36), src_data_46),
      src_data_56),
    src_data_66);
  float32x4_t t07 = vaddq_f32(
    vaddq_f32(
      vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(src_data_07, src_data_17), src_data_27), src_data_37), src_data_47),
      src_data_57),
    src_data_67);

  float32x4_t t10 = vaddq_f32(vaddq_f32(vmulq_n_f32(d01, 0.5), d11), vmulq_n_f32(d21, 1.5));
  float32x4_t t11 = vaddq_f32(vaddq_f32(vmulq_n_f32(d02, 0.5), d12), vmulq_n_f32(d22, 1.5));
  float32x4_t t12 = vaddq_f32(vaddq_f32(vmulq_n_f32(d03, 0.5), d13), vmulq_n_f32(d23, 1.5));
  float32x4_t t13 = vaddq_f32(vaddq_f32(vmulq_n_f32(d04, 0.5), d14), vmulq_n_f32(d24, 1.5));
  float32x4_t t14 = vaddq_f32(vaddq_f32(vmulq_n_f32(d05, 0.5), d15), vmulq_n_f32(d25, 1.5));
  float32x4_t t15 = vaddq_f32(vaddq_f32(vmulq_n_f32(d06, 0.5), d16), vmulq_n_f32(d26, 1.5));
  float32x4_t t16 = vaddq_f32(vaddq_f32(vmulq_n_f32(d07, 0.5), d17), vmulq_n_f32(d27, 1.5));
  float32x4_t t17 = vaddq_f32(vaddq_f32(vmulq_n_f32(d08, 0.5), d18), vmulq_n_f32(d28, 1.5));

  float32x4_t t20 = vaddq_f32(vaddq_f32(vmulq_n_f32(d31, 0.25), d41), vmulq_n_f32(d51, 2.25));
  float32x4_t t21 = vaddq_f32(vaddq_f32(vmulq_n_f32(d32, 0.25), d42), vmulq_n_f32(d52, 2.25));
  float32x4_t t22 = vaddq_f32(vaddq_f32(vmulq_n_f32(d33, 0.25), d43), vmulq_n_f32(d53, 2.25));
  float32x4_t t23 = vaddq_f32(vaddq_f32(vmulq_n_f32(d34, 0.25), d44), vmulq_n_f32(d54, 2.25));
  float32x4_t t24 = vaddq_f32(vaddq_f32(vmulq_n_f32(d35, 0.25), d45), vmulq_n_f32(d55, 2.25));
  float32x4_t t25 = vaddq_f32(vaddq_f32(vmulq_n_f32(d36, 0.25), d46), vmulq_n_f32(d56, 2.25));
  float32x4_t t26 = vaddq_f32(vaddq_f32(vmulq_n_f32(d37, 0.25), d47), vmulq_n_f32(d57, 2.25));
  float32x4_t t27 = vaddq_f32(vaddq_f32(vmulq_n_f32(d38, 0.25), d48), vmulq_n_f32(d58, 2.25));

  float32x4_t t30 = vaddq_f32(vaddq_f32(vmulq_n_f32(d01, 0.125), d11), vmulq_n_f32(d21, 3.375));
  float32x4_t t31 = vaddq_f32(vaddq_f32(vmulq_n_f32(d02, 0.125), d12), vmulq_n_f32(d22, 3.375));
  float32x4_t t32 = vaddq_f32(vaddq_f32(vmulq_n_f32(d03, 0.125), d13), vmulq_n_f32(d23, 3.375));
  float32x4_t t33 = vaddq_f32(vaddq_f32(vmulq_n_f32(d04, 0.125), d14), vmulq_n_f32(d24, 3.375));
  float32x4_t t34 = vaddq_f32(vaddq_f32(vmulq_n_f32(d05, 0.125), d15), vmulq_n_f32(d25, 3.375));
  float32x4_t t35 = vaddq_f32(vaddq_f32(vmulq_n_f32(d06, 0.125), d16), vmulq_n_f32(d26, 3.375));
  float32x4_t t36 = vaddq_f32(vaddq_f32(vmulq_n_f32(d07, 0.125), d17), vmulq_n_f32(d27, 3.375));
  float32x4_t t37 = vaddq_f32(vaddq_f32(vmulq_n_f32(d08, 0.125), d18), vmulq_n_f32(d28, 3.375));

  float32x4_t t40 = vaddq_f32(vaddq_f32(vmulq_n_f32(d31, 0.0625), d41), vmulq_n_f32(d51, 5.0625));
  float32x4_t t41 = vaddq_f32(vaddq_f32(vmulq_n_f32(d32, 0.0625), d42), vmulq_n_f32(d52, 5.0625));
  float32x4_t t42 = vaddq_f32(vaddq_f32(vmulq_n_f32(d33, 0.0625), d43), vmulq_n_f32(d53, 5.0625));
  float32x4_t t43 = vaddq_f32(vaddq_f32(vmulq_n_f32(d34, 0.0625), d44), vmulq_n_f32(d54, 5.0625));
  float32x4_t t44 = vaddq_f32(vaddq_f32(vmulq_n_f32(d35, 0.0625), d45), vmulq_n_f32(d55, 5.0625));
  float32x4_t t45 = vaddq_f32(vaddq_f32(vmulq_n_f32(d36, 0.0625), d46), vmulq_n_f32(d56, 5.0625));
  float32x4_t t46 = vaddq_f32(vaddq_f32(vmulq_n_f32(d37, 0.0625), d47), vmulq_n_f32(d57, 5.0625));
  float32x4_t t47 = vaddq_f32(vaddq_f32(vmulq_n_f32(d38, 0.0625), d48), vmulq_n_f32(d58, 5.0625));

  float32x4_t t50 = vaddq_f32(vaddq_f32(vmulq_n_f32(d01, 0.03125), d11), vmulq_n_f32(d21, 7.59375));
  float32x4_t t51 = vaddq_f32(vaddq_f32(vmulq_n_f32(d02, 0.03125), d12), vmulq_n_f32(d22, 7.59375));
  float32x4_t t52 = vaddq_f32(vaddq_f32(vmulq_n_f32(d03, 0.03125), d13), vmulq_n_f32(d23, 7.59375));
  float32x4_t t53 = vaddq_f32(vaddq_f32(vmulq_n_f32(d04, 0.03125), d14), vmulq_n_f32(d24, 7.59375));
  float32x4_t t54 = vaddq_f32(vaddq_f32(vmulq_n_f32(d05, 0.03125), d15), vmulq_n_f32(d25, 7.59375));
  float32x4_t t55 = vaddq_f32(vaddq_f32(vmulq_n_f32(d06, 0.03125), d16), vmulq_n_f32(d26, 7.59375));
  float32x4_t t56 = vaddq_f32(vaddq_f32(vmulq_n_f32(d07, 0.03125), d17), vmulq_n_f32(d27, 7.59375));
  float32x4_t t57 = vaddq_f32(vaddq_f32(vmulq_n_f32(d08, 0.03125), d18), vmulq_n_f32(d28, 7.59375));

  float32x4_t t60 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d31, 0.015625), d41), vmulq_n_f32(d51, 11.390625)), src_data_70);
  float32x4_t t61 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d32, 0.015625), d42), vmulq_n_f32(d52, 11.390625)), src_data_71);
  float32x4_t t62 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d33, 0.015625), d43), vmulq_n_f32(d53, 11.390625)), src_data_72);
  float32x4_t t63 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d34, 0.015625), d44), vmulq_n_f32(d54, 11.390625)), src_data_73);
  float32x4_t t64 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d35, 0.015625), d45), vmulq_n_f32(d55, 11.390625)), src_data_74);
  float32x4_t t65 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d36, 0.015625), d46), vmulq_n_f32(d56, 11.390625)), src_data_75);
  float32x4_t t66 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d37, 0.015625), d47), vmulq_n_f32(d57, 11.390625)), src_data_76);
  float32x4_t t67 =
    vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(d38, 0.015625), d48), vmulq_n_f32(d58, 11.390625)), src_data_77);

  float32x4_t s11 = vsubq_f32(t01, t02);
  float32x4_t s12 = vsubq_f32(t11, t12);
  float32x4_t s13 = vsubq_f32(t21, t22);
  float32x4_t s14 = vsubq_f32(t31, t32);
  float32x4_t s15 = vsubq_f32(t41, t42);
  float32x4_t s16 = vsubq_f32(t51, t52);
  float32x4_t s17 = vsubq_f32(t61, t62);

  float32x4_t s21 = vsubq_f32(t03, t04);
  float32x4_t s22 = vsubq_f32(t13, t14);
  float32x4_t s23 = vsubq_f32(t23, t24);
  float32x4_t s24 = vsubq_f32(t33, t34);
  float32x4_t s25 = vsubq_f32(t43, t44);
  float32x4_t s26 = vsubq_f32(t53, t54);
  float32x4_t s27 = vsubq_f32(t63, t64);

  float32x4_t s31 = vsubq_f32(t05, t06);
  float32x4_t s32 = vsubq_f32(t15, t16);
  float32x4_t s33 = vsubq_f32(t25, t26);
  float32x4_t s34 = vsubq_f32(t35, t36);
  float32x4_t s35 = vsubq_f32(t45, t46);
  float32x4_t s36 = vsubq_f32(t55, t56);
  float32x4_t s37 = vsubq_f32(t65, t66);

  float32x4_t s41 = vaddq_f32(t01, t02);
  float32x4_t s42 = vaddq_f32(t11, t12);
  float32x4_t s43 = vaddq_f32(t21, t22);
  float32x4_t s44 = vaddq_f32(t31, t32);
  float32x4_t s45 = vaddq_f32(t41, t42);
  float32x4_t s46 = vaddq_f32(t51, t52);
  float32x4_t s47 = vaddq_f32(t61, t62);

  float32x4_t s51 = vaddq_f32(t03, t04);
  float32x4_t s52 = vaddq_f32(t13, t14);
  float32x4_t s53 = vaddq_f32(t23, t24);
  float32x4_t s54 = vaddq_f32(t33, t34);
  float32x4_t s55 = vaddq_f32(t43, t44);
  float32x4_t s56 = vaddq_f32(t53, t54);
  float32x4_t s57 = vaddq_f32(t63, t64);

  float32x4_t s61 = vaddq_f32(t05, t06);
  float32x4_t s62 = vaddq_f32(t15, t16);
  float32x4_t s63 = vaddq_f32(t25, t26);
  float32x4_t s64 = vaddq_f32(t35, t36);
  float32x4_t s65 = vaddq_f32(t45, t46);
  float32x4_t s66 = vaddq_f32(t55, t56);
  float32x4_t s67 = vaddq_f32(t65, t66);

  float32x4_t m00 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t00, t01), t02), t03), t04), t05), t06);
  float32x4_t m01 = vaddq_f32(vaddq_f32(vmulq_n_f32(s11, 0.5), s21), vmulq_n_f32(s31, 1.5));
  float32x4_t m02 = vaddq_f32(vaddq_f32(vmulq_n_f32(s41, 0.25), s51), vmulq_n_f32(s61, 2.25));
  float32x4_t m03 = vaddq_f32(vaddq_f32(vmulq_n_f32(s11, 0.125), s21), vmulq_n_f32(s31, 3.375));
  float32x4_t m04 = vaddq_f32(vaddq_f32(vmulq_n_f32(s41, 0.0625), s51), vmulq_n_f32(s61, 5.0625));
  float32x4_t m05 = vaddq_f32(vaddq_f32(vmulq_n_f32(s11, 0.03125), s21), vmulq_n_f32(s31, 7.59375));
  float32x4_t m06 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s41, 0.015625), s51), vmulq_n_f32(s61, 11.390625)), t07);

  float32x4_t m10 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t10, t11), t12), t13), t14), t15), t16);
  float32x4_t m11 = vaddq_f32(vaddq_f32(vmulq_n_f32(s12, 0.5), s22), vmulq_n_f32(s32, 1.5));
  float32x4_t m12 = vaddq_f32(vaddq_f32(vmulq_n_f32(s42, 0.25), s52), vmulq_n_f32(s62, 2.25));
  float32x4_t m13 = vaddq_f32(vaddq_f32(vmulq_n_f32(s12, 0.125), s22), vmulq_n_f32(s32, 3.375));
  float32x4_t m14 = vaddq_f32(vaddq_f32(vmulq_n_f32(s42, 0.0625), s52), vmulq_n_f32(s62, 5.0625));
  float32x4_t m15 = vaddq_f32(vaddq_f32(vmulq_n_f32(s12, 0.03125), s22), vmulq_n_f32(s32, 7.59375));
  float32x4_t m16 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s42, 0.015625), s52), vmulq_n_f32(s62, 11.390625)), t17);

  float32x4_t m20 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t20, t21), t22), t23), t24), t25), t26);
  float32x4_t m21 = vaddq_f32(vaddq_f32(vmulq_n_f32(s13, 0.5), s23), vmulq_n_f32(s33, 1.5));
  float32x4_t m22 = vaddq_f32(vaddq_f32(vmulq_n_f32(s43, 0.25), s53), vmulq_n_f32(s63, 2.25));
  float32x4_t m23 = vaddq_f32(vaddq_f32(vmulq_n_f32(s13, 0.125), s23), vmulq_n_f32(s33, 3.375));
  float32x4_t m24 = vaddq_f32(vaddq_f32(vmulq_n_f32(s43, 0.0625), s53), vmulq_n_f32(s63, 5.0625));
  float32x4_t m25 = vaddq_f32(vaddq_f32(vmulq_n_f32(s13, 0.03125), s23), vmulq_n_f32(s33, 7.59375));
  float32x4_t m26 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s43, 0.015625), s53), vmulq_n_f32(s63, 11.390625)), t27);

  float32x4_t m30 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t30, t31), t32), t33), t34), t35), t36);
  float32x4_t m31 = vaddq_f32(vaddq_f32(vmulq_n_f32(s14, 0.5), s24), vmulq_n_f32(s34, 1.5));
  float32x4_t m32 = vaddq_f32(vaddq_f32(vmulq_n_f32(s44, 0.25), s54), vmulq_n_f32(s64, 2.25));
  float32x4_t m33 = vaddq_f32(vaddq_f32(vmulq_n_f32(s14, 0.125), s24), vmulq_n_f32(s34, 3.375));
  float32x4_t m34 = vaddq_f32(vaddq_f32(vmulq_n_f32(s44, 0.0625), s54), vmulq_n_f32(s64, 5.0625));
  float32x4_t m35 = vaddq_f32(vaddq_f32(vmulq_n_f32(s14, 0.03125), s24), vmulq_n_f32(s34, 7.59375));
  float32x4_t m36 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s44, 0.015625), s54), vmulq_n_f32(s64, 11.390625)), t37);

  float32x4_t m40 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t40, t41), t42), t43), t44), t45), t46);
  float32x4_t m41 = vaddq_f32(vaddq_f32(vmulq_n_f32(s15, 0.5), s25), vmulq_n_f32(s35, 1.5));
  float32x4_t m42 = vaddq_f32(vaddq_f32(vmulq_n_f32(s45, 0.25), s55), vmulq_n_f32(s65, 2.25));
  float32x4_t m43 = vaddq_f32(vaddq_f32(vmulq_n_f32(s15, 0.125), s25), vmulq_n_f32(s35, 3.375));
  float32x4_t m44 = vaddq_f32(vaddq_f32(vmulq_n_f32(s45, 0.0625), s55), vmulq_n_f32(s65, 5.0625));
  float32x4_t m45 = vaddq_f32(vaddq_f32(vmulq_n_f32(s15, 0.03125), s25), vmulq_n_f32(s35, 7.59375));
  float32x4_t m46 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s45, 0.015625), s55), vmulq_n_f32(s65, 11.390625)), t47);

  float32x4_t m50 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t50, t51), t52), t53), t54), t55), t56);
  float32x4_t m51 = vaddq_f32(vaddq_f32(vmulq_n_f32(s16, 0.5), s26), vmulq_n_f32(s36, 1.5));
  float32x4_t m52 = vaddq_f32(vaddq_f32(vmulq_n_f32(s46, 0.25), s56), vmulq_n_f32(s66, 2.25));
  float32x4_t m53 = vaddq_f32(vaddq_f32(vmulq_n_f32(s16, 0.125), s26), vmulq_n_f32(s36, 3.375));
  float32x4_t m54 = vaddq_f32(vaddq_f32(vmulq_n_f32(s46, 0.0625), s56), vmulq_n_f32(s66, 5.0625));
  float32x4_t m55 = vaddq_f32(vaddq_f32(vmulq_n_f32(s16, 0.03125), s26), vmulq_n_f32(s36, 7.59375));
  float32x4_t m56 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s46, 0.015625), s56), vmulq_n_f32(s66, 11.390625)), t57);

  float32x4_t m60 = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(t60, t61), t62), t63), t64), t65), t66);
  float32x4_t m61 = vaddq_f32(vaddq_f32(vmulq_n_f32(s17, 0.5), s27), vmulq_n_f32(s37, 1.5));
  float32x4_t m62 = vaddq_f32(vaddq_f32(vmulq_n_f32(s47, 0.25), s57), vmulq_n_f32(s67, 2.25));
  float32x4_t m63 = vaddq_f32(vaddq_f32(vmulq_n_f32(s17, 0.125), s27), vmulq_n_f32(s37, 3.375));
  float32x4_t m64 = vaddq_f32(vaddq_f32(vmulq_n_f32(s47, 0.0625), s57), vmulq_n_f32(s67, 5.0625));
  float32x4_t m65 = vaddq_f32(vaddq_f32(vmulq_n_f32(s17, 0.03125), s27), vmulq_n_f32(s37, 7.59375));
  float32x4_t m66 = vaddq_f32(vaddq_f32(vaddq_f32(vmulq_n_f32(s47, 0.015625), s57), vmulq_n_f32(s67, 11.390625)), t67);

  float32x4_t bias_ptr = vld1q_f32(bias_data);
  vst1q_f32(dst_data, vaddq_f32(m00, bias_ptr));
  vst1q_f32(dst_data + C4NUM, vaddq_f32(m01, bias_ptr));
  vst1q_f32(dst_data + 2 * C4NUM, vaddq_f32(m02, bias_ptr));
  vst1q_f32(dst_data + 3 * C4NUM, vaddq_f32(m03, bias_ptr));
  vst1q_f32(dst_data + 4 * C4NUM, vaddq_f32(m04, bias_ptr));
  vst1q_f32(dst_data + 5 * C4NUM, vaddq_f32(m05, bias_ptr));
  vst1q_f32(dst_data + 6 * C4NUM, vaddq_f32(m06, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM, vaddq_f32(m10, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM + C4NUM, vaddq_f32(m11, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM + 2 * C4NUM, vaddq_f32(m12, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM + 3 * C4NUM, vaddq_f32(m13, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM + 4 * C4NUM, vaddq_f32(m14, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM + 5 * C4NUM, vaddq_f32(m15, bias_ptr));
  vst1q_f32(dst_data + dst_step * C4NUM + 6 * C4NUM, vaddq_f32(m16, bias_ptr));
  vst1q_f32(dst_data + 2 * dst_step * C4NUM, vaddq_f32(m20, bias_ptr));
  vst1q_f32(dst_data + 2 * dst_step * C4NUM + C4NUM, vaddq_f32(m21, bias_ptr));
  vst1q_f32(dst_data + 2 * dst_step * C4NUM + 2 * C4NUM, vaddq_f32(m22, bias_ptr));
  vst1q_f32(dst_data + 2 * dst_step * C4NUM + 3 * C4NUM, vaddq_f32(m23, bias_ptr));
  vst1q_f32(dst_data + 2 * dst_step * C4NUM + 4 * C4NUM, vaddq_f32(m24, bias_ptr));
  vst1q_f32(dst_data + 2 * dst_step * C4NUM + 5 * C4NUM, vaddq_f32(m25, bias_ptr));
  vst1q_f32(dst_data + 2 * dst_step * C4NUM + 6 * C4NUM, vaddq_f32(m26, bias_ptr));
  vst1q_f32(dst_data + 3 * dst_step * C4NUM, vaddq_f32(m30, bias_ptr));
  vst1q_f32(dst_data + 3 * dst_step * C4NUM + C4NUM, vaddq_f32(m31, bias_ptr));
  vst1q_f32(dst_data + 3 * dst_step * C4NUM + 2 * C4NUM, vaddq_f32(m32, bias_ptr));
  vst1q_f32(dst_data + 3 * dst_step * C4NUM + 3 * C4NUM, vaddq_f32(m33, bias_ptr));
  vst1q_f32(dst_data + 3 * dst_step * C4NUM + 4 * C4NUM, vaddq_f32(m34, bias_ptr));
  vst1q_f32(dst_data + 3 * dst_step * C4NUM + 5 * C4NUM, vaddq_f32(m35, bias_ptr));
  vst1q_f32(dst_data + 3 * dst_step * C4NUM + 6 * C4NUM, vaddq_f32(m36, bias_ptr));
  vst1q_f32(dst_data + 4 * dst_step * C4NUM, vaddq_f32(m40, bias_ptr));
  vst1q_f32(dst_data + 4 * dst_step * C4NUM + C4NUM, vaddq_f32(m41, bias_ptr));
  vst1q_f32(dst_data + 4 * dst_step * C4NUM + 2 * C4NUM, vaddq_f32(m42, bias_ptr));
  vst1q_f32(dst_data + 4 * dst_step * C4NUM + 3 * C4NUM, vaddq_f32(m43, bias_ptr));
  vst1q_f32(dst_data + 4 * dst_step * C4NUM + 4 * C4NUM, vaddq_f32(m44, bias_ptr));
  vst1q_f32(dst_data + 4 * dst_step * C4NUM + 5 * C4NUM, vaddq_f32(m45, bias_ptr));
  vst1q_f32(dst_data + 4 * dst_step * C4NUM + 6 * C4NUM, vaddq_f32(m46, bias_ptr));
  vst1q_f32(dst_data + 5 * dst_step * C4NUM, vaddq_f32(m50, bias_ptr));
  vst1q_f32(dst_data + 5 * dst_step * C4NUM + C4NUM, vaddq_f32(m51, bias_ptr));
  vst1q_f32(dst_data + 5 * dst_step * C4NUM + 2 * C4NUM, vaddq_f32(m52, bias_ptr));
  vst1q_f32(dst_data + 5 * dst_step * C4NUM + 3 * C4NUM, vaddq_f32(m53, bias_ptr));
  vst1q_f32(dst_data + 5 * dst_step * C4NUM + 4 * C4NUM, vaddq_f32(m54, bias_ptr));
  vst1q_f32(dst_data + 5 * dst_step * C4NUM + 5 * C4NUM, vaddq_f32(m55, bias_ptr));
  vst1q_f32(dst_data + 5 * dst_step * C4NUM + 6 * C4NUM, vaddq_f32(m56, bias_ptr));
  vst1q_f32(dst_data + 6 * dst_step * C4NUM, vaddq_f32(m60, bias_ptr));
  vst1q_f32(dst_data + 6 * dst_step * C4NUM + C4NUM, vaddq_f32(m61, bias_ptr));
  vst1q_f32(dst_data + 6 * dst_step * C4NUM + 2 * C4NUM, vaddq_f32(m62, bias_ptr));
  vst1q_f32(dst_data + 6 * dst_step * C4NUM + 3 * C4NUM, vaddq_f32(m63, bias_ptr));
  vst1q_f32(dst_data + 6 * dst_step * C4NUM + 4 * C4NUM, vaddq_f32(m64, bias_ptr));
  vst1q_f32(dst_data + 6 * dst_step * C4NUM + 5 * C4NUM, vaddq_f32(m65, bias_ptr));
  vst1q_f32(dst_data + 6 * dst_step * C4NUM + 6 * C4NUM, vaddq_f32(m66, bias_ptr));
#else
  for (int i = 0; i < C4NUM; i++) {
    float src_data_00 = src_data[i];
    float src_data_01 = src_data[i + src_step];
    float src_data_02 = src_data[i + 2 * src_step];
    float src_data_03 = src_data[i + 3 * src_step];
    float src_data_04 = src_data[i + 4 * src_step];
    float src_data_05 = src_data[i + 5 * src_step];
    float src_data_06 = src_data[i + 6 * src_step];
    float src_data_07 = src_data[i + 7 * src_step];
    float src_data_10 = src_data[i + 8 * src_step];
    float src_data_11 = src_data[i + 9 * src_step];
    float src_data_12 = src_data[i + 10 * src_step];
    float src_data_13 = src_data[i + 11 * src_step];
    float src_data_14 = src_data[i + 12 * src_step];
    float src_data_15 = src_data[i + 13 * src_step];
    float src_data_16 = src_data[i + 14 * src_step];
    float src_data_17 = src_data[i + 15 * src_step];
    float src_data_20 = src_data[i + 16 * src_step];
    float src_data_21 = src_data[i + 17 * src_step];
    float src_data_22 = src_data[i + 18 * src_step];
    float src_data_23 = src_data[i + 19 * src_step];
    float src_data_24 = src_data[i + 20 * src_step];
    float src_data_25 = src_data[i + 21 * src_step];
    float src_data_26 = src_data[i + 22 * src_step];
    float src_data_27 = src_data[i + 23 * src_step];
    float src_data_30 = src_data[i + 24 * src_step];
    float src_data_31 = src_data[i + 25 * src_step];
    float src_data_32 = src_data[i + 26 * src_step];
    float src_data_33 = src_data[i + 27 * src_step];
    float src_data_34 = src_data[i + 28 * src_step];
    float src_data_35 = src_data[i + 29 * src_step];
    float src_data_36 = src_data[i + 30 * src_step];
    float src_data_37 = src_data[i + 31 * src_step];
    float src_data_40 = src_data[i + 32 * src_step];
    float src_data_41 = src_data[i + 33 * src_step];
    float src_data_42 = src_data[i + 34 * src_step];
    float src_data_43 = src_data[i + 35 * src_step];
    float src_data_44 = src_data[i + 36 * src_step];
    float src_data_45 = src_data[i + 37 * src_step];
    float src_data_46 = src_data[i + 38 * src_step];
    float src_data_47 = src_data[i + 39 * src_step];
    float src_data_50 = src_data[i + 40 * src_step];
    float src_data_51 = src_data[i + 41 * src_step];
    float src_data_52 = src_data[i + 42 * src_step];
    float src_data_53 = src_data[i + 43 * src_step];
    float src_data_54 = src_data[i + 44 * src_step];
    float src_data_55 = src_data[i + 45 * src_step];
    float src_data_56 = src_data[i + 46 * src_step];
    float src_data_57 = src_data[i + 47 * src_step];
    float src_data_60 = src_data[i + 48 * src_step];
    float src_data_61 = src_data[i + 49 * src_step];
    float src_data_62 = src_data[i + 50 * src_step];
    float src_data_63 = src_data[i + 51 * src_step];
    float src_data_64 = src_data[i + 52 * src_step];
    float src_data_65 = src_data[i + 53 * src_step];
    float src_data_66 = src_data[i + 54 * src_step];
    float src_data_67 = src_data[i + 55 * src_step];
    float src_data_70 = src_data[i + 56 * src_step];
    float src_data_71 = src_data[i + 57 * src_step];
    float src_data_72 = src_data[i + 58 * src_step];
    float src_data_73 = src_data[i + 59 * src_step];
    float src_data_74 = src_data[i + 60 * src_step];
    float src_data_75 = src_data[i + 61 * src_step];
    float src_data_76 = src_data[i + 62 * src_step];
    float src_data_77 = src_data[i + 63 * src_step];

    float d01 = src_data_10 - src_data_20;
    float d02 = src_data_11 - src_data_21;
    float d03 = src_data_12 - src_data_22;
    float d04 = src_data_13 - src_data_23;
    float d05 = src_data_14 - src_data_24;
    float d06 = src_data_15 - src_data_25;
    float d07 = src_data_16 - src_data_26;
    float d08 = src_data_17 - src_data_27;

    float d11 = src_data_30 - src_data_40;
    float d12 = src_data_31 - src_data_41;
    float d13 = src_data_32 - src_data_42;
    float d14 = src_data_33 - src_data_43;
    float d15 = src_data_34 - src_data_44;
    float d16 = src_data_35 - src_data_45;
    float d17 = src_data_36 - src_data_46;
    float d18 = src_data_37 - src_data_47;

    float d21 = src_data_50 - src_data_60;
    float d22 = src_data_51 - src_data_61;
    float d23 = src_data_52 - src_data_62;
    float d24 = src_data_53 - src_data_63;
    float d25 = src_data_54 - src_data_64;
    float d26 = src_data_55 - src_data_65;
    float d27 = src_data_56 - src_data_66;
    float d28 = src_data_57 - src_data_67;

    float d31 = src_data_10 + src_data_20;
    float d32 = src_data_11 + src_data_21;
    float d33 = src_data_12 + src_data_22;
    float d34 = src_data_13 + src_data_23;
    float d35 = src_data_14 + src_data_24;
    float d36 = src_data_15 + src_data_25;
    float d37 = src_data_16 + src_data_26;
    float d38 = src_data_17 + src_data_27;

    float d41 = src_data_30 + src_data_40;
    float d42 = src_data_31 + src_data_41;
    float d43 = src_data_32 + src_data_42;
    float d44 = src_data_33 + src_data_43;
    float d45 = src_data_34 + src_data_44;
    float d46 = src_data_35 + src_data_45;
    float d47 = src_data_36 + src_data_46;
    float d48 = src_data_37 + src_data_47;

    float d51 = src_data_50 + src_data_60;
    float d52 = src_data_51 + src_data_61;
    float d53 = src_data_52 + src_data_62;
    float d54 = src_data_53 + src_data_63;
    float d55 = src_data_54 + src_data_64;
    float d56 = src_data_55 + src_data_65;
    float d57 = src_data_56 + src_data_66;
    float d58 = src_data_57 + src_data_67;

    float t00 = src_data_00 + src_data_10 + src_data_20 + src_data_30 + src_data_40 + src_data_50 + src_data_60;
    float t01 = src_data_01 + src_data_11 + src_data_21 + src_data_31 + src_data_41 + src_data_51 + src_data_61;
    float t02 = src_data_02 + src_data_12 + src_data_22 + src_data_32 + src_data_42 + src_data_52 + src_data_62;
    float t03 = src_data_03 + src_data_13 + src_data_23 + src_data_33 + src_data_43 + src_data_53 + src_data_63;
    float t04 = src_data_04 + src_data_14 + src_data_24 + src_data_34 + src_data_44 + src_data_54 + src_data_64;
    float t05 = src_data_05 + src_data_15 + src_data_25 + src_data_35 + src_data_45 + src_data_55 + src_data_65;
    float t06 = src_data_06 + src_data_16 + src_data_26 + src_data_36 + src_data_46 + src_data_56 + src_data_66;
    float t07 = src_data_07 + src_data_17 + src_data_27 + src_data_37 + src_data_47 + src_data_57 + src_data_67;

    const float t10 = 0.5f * d01 + d11 + 1.5f * d21;
    const float t11 = 0.5f * d02 + d12 + 1.5f * d22;
    const float t12 = 0.5f * d03 + d13 + 1.5f * d23;
    const float t13 = 0.5f * d04 + d14 + 1.5f * d24;
    const float t14 = 0.5f * d05 + d15 + 1.5f * d25;
    const float t15 = 0.5f * d06 + d16 + 1.5f * d26;
    const float t16 = 0.5f * d07 + d17 + 1.5f * d27;
    const float t17 = 0.5f * d08 + d18 + 1.5f * d28;

    const float t20 = 0.25f * d31 + d41 + 2.25f * d51;
    const float t21 = 0.25f * d32 + d42 + 2.25f * d52;
    const float t22 = 0.25f * d33 + d43 + 2.25f * d53;
    const float t23 = 0.25f * d34 + d44 + 2.25f * d54;
    const float t24 = 0.25f * d35 + d45 + 2.25f * d55;
    const float t25 = 0.25f * d36 + d46 + 2.25f * d56;
    const float t26 = 0.25f * d37 + d47 + 2.25f * d57;
    const float t27 = 0.25f * d38 + d48 + 2.25f * d58;

    const float t30 = 0.125f * d01 + d11 + 3.375f * d21;
    const float t31 = 0.125f * d02 + d12 + 3.375f * d22;
    const float t32 = 0.125f * d03 + d13 + 3.375f * d23;
    const float t33 = 0.125f * d04 + d14 + 3.375f * d24;
    const float t34 = 0.125f * d05 + d15 + 3.375f * d25;
    const float t35 = 0.125f * d06 + d16 + 3.375f * d26;
    const float t36 = 0.125f * d07 + d17 + 3.375f * d27;
    const float t37 = 0.125f * d08 + d18 + 3.375f * d28;

    const float t40 = 0.0625f * d31 + d41 + 5.0625f * d51;
    const float t41 = 0.0625f * d32 + d42 + 5.0625f * d52;
    const float t42 = 0.0625f * d33 + d43 + 5.0625f * d53;
    const float t43 = 0.0625f * d34 + d44 + 5.0625f * d54;
    const float t44 = 0.0625f * d35 + d45 + 5.0625f * d55;
    const float t45 = 0.0625f * d36 + d46 + 5.0625f * d56;
    const float t46 = 0.0625f * d37 + d47 + 5.0625f * d57;
    const float t47 = 0.0625f * d38 + d48 + 5.0625f * d58;

    const float t50 = 0.03125f * d01 + d11 + 7.59375f * d21;
    const float t51 = 0.03125f * d02 + d12 + 7.59375f * d22;
    const float t52 = 0.03125f * d03 + d13 + 7.59375f * d23;
    const float t53 = 0.03125f * d04 + d14 + 7.59375f * d24;
    const float t54 = 0.03125f * d05 + d15 + 7.59375f * d25;
    const float t55 = 0.03125f * d06 + d16 + 7.59375f * d26;
    const float t56 = 0.03125f * d07 + d17 + 7.59375f * d27;
    const float t57 = 0.03125f * d08 + d18 + 7.59375f * d28;

    const float t60 = 0.015625f * d31 + d41 + 11.390625f * d51 + src_data_70;
    const float t61 = 0.015625f * d32 + d42 + 11.390625f * d52 + src_data_71;
    const float t62 = 0.015625f * d33 + d43 + 11.390625f * d53 + src_data_72;
    const float t63 = 0.015625f * d34 + d44 + 11.390625f * d54 + src_data_73;
    const float t64 = 0.015625f * d35 + d45 + 11.390625f * d55 + src_data_74;
    const float t65 = 0.015625f * d36 + d46 + 11.390625f * d56 + src_data_75;
    const float t66 = 0.015625f * d37 + d47 + 11.390625f * d57 + src_data_76;
    const float t67 = 0.015625f * d38 + d48 + 11.390625f * d58 + src_data_77;

    float s11 = t01 - t02;
    float s12 = t11 - t12;
    float s13 = t21 - t22;
    float s14 = t31 - t32;
    float s15 = t41 - t42;
    float s16 = t51 - t52;
    float s17 = t61 - t62;

    float s21 = t03 - t04;
    float s22 = t13 - t14;
    float s23 = t23 - t24;
    float s24 = t33 - t34;
    float s25 = t43 - t44;
    float s26 = t53 - t54;
    float s27 = t63 - t64;

    float s31 = t05 - t06;
    float s32 = t15 - t16;
    float s33 = t25 - t26;
    float s34 = t35 - t36;
    float s35 = t45 - t46;
    float s36 = t55 - t56;
    float s37 = t56 - t66;

    float s41 = t01 + t02;
    float s42 = t11 + t12;
    float s43 = t21 + t22;
    float s44 = t31 + t32;
    float s45 = t41 + t42;
    float s46 = t51 + t52;
    float s47 = t61 + t62;

    float s51 = t03 + t04;
    float s52 = t13 + t14;
    float s53 = t23 + t24;
    float s54 = t33 + t34;
    float s55 = t43 + t44;
    float s56 = t53 + t54;
    float s57 = t63 + t64;

    float s61 = t05 + t06;
    float s62 = t15 + t16;
    float s63 = t25 + t26;
    float s64 = t35 + t36;
    float s65 = t45 + t46;
    float s66 = t55 + t56;
    float s67 = t65 + t66;

    float m00 = t00 + t01 + t02 + t03 + t04 + t05 + t06;
    const float m01 = 0.5f * s11 + s21 + 1.5f * s31;
    const float m02 = 0.25f * s41 + s51 + 2.25f * s61;
    const float m03 = 0.125f * s11 + s21 + 3.375f * s31;
    const float m04 = 0.0625f * s41 + s51 + 5.0625f * s61;
    const float m05 = 0.03125f * s11 + s21 + 7.59375f * s31;
    const float m06 = 0.015625f * s41 + s51 + 11.390625f * s61 + t07;

    float m10 = t10 + t11 + t12 + t13 + t14 + t15 + t16;
    const float m11 = 0.5f * s12 + s22 + 1.5f * s32;
    const float m12 = 0.25f * s42 + s52 + 2.25f * s62;
    const float m13 = 0.125f * s12 + s22 + 3.375f * s32;
    const float m14 = 0.0625f * s42 + s52 + 5.0625f * s62;
    const float m15 = 0.03125f * s12 + s22 + 7.59375f * s32;
    const float m16 = 0.015625f * s42 + s52 + 11.390625f * s62 + t17;

    float m20 = t20 + t21 + t22 + t23 + t24 + t25 + t26;
    const float m21 = 0.5f * s13 + s23 + 1.5f * s33;
    const float m22 = 0.25f * s43 + s53 + 2.25f * s63;
    const float m23 = 0.125f * s13 + s23 + 3.375f * s33;
    const float m24 = 0.0625f * s43 + s53 + 5.0625f * s63;
    const float m25 = 0.03125f * s13 + s23 + 7.59375f * s33;
    const float m26 = 0.015625f * s43 + s53 + 11.390625f * s63 + t27;

    float m30 = t30 + t31 + t32 + t33 + t34 + t35 + t36;
    const float m31 = 0.5f * s14 + s24 + 1.5f * s34;
    const float m32 = 0.25f * s44 + s54 + 2.25f * s64;
    const float m33 = 0.125f * s14 + s24 + 3.375f * s34;
    const float m34 = 0.0625f * s44 + s54 + 5.0625f * s64;
    const float m35 = 0.03125f * s14 + s24 + 7.59375f * s34;
    const float m36 = 0.015625f * s44 + s54 + 11.390625f * s64 + t37;

    float m40 = t40 + t41 + t42 + t43 + t44 + t45 + t46;
    const float m41 = 0.5f * s15 + s25 + 1.5f * s35;
    const float m42 = 0.25f * s45 + s55 + 2.25f * s65;
    const float m43 = 0.125f * s15 + s25 + 3.375f * s35;
    const float m44 = 0.0625f * s45 + s55 + 5.0625f * s65;
    const float m45 = 0.03125f * s15 + s25 + 7.59375f * s35;
    const float m46 = 0.015625f * s45 + s55 + 11.390625f * s65 + t47;

    float m50 = t50 + t51 + t52 + t53 + t54 + t55 + t56;
    const float m51 = 0.5f * s16 + s26 + 1.5f * s36;
    const float m52 = 0.25f * s46 + s56 + 2.25f * s66;
    const float m53 = 0.125f * s16 + s26 + 3.375f * s36;
    const float m54 = 0.0625f * s46 + s56 + 5.0625f * s66;
    const float m55 = 0.03125f * s16 + s26 + 7.59375f * s36;
    const float m56 = 0.015625f * s46 + s56 + 11.390625f * s66 + t57;

    float m60 = t60 + t61 + t62 + t63 + t64 + t65 + t66;
    const float m61 = 0.5f * s17 + s27 + 1.5f * s37;
    const float m62 = 0.25f * s47 + s57 + 2.25f * s67;
    const float m63 = 0.125f * s17 + s27 + 3.375f * s37;
    const float m64 = 0.0625f * s47 + s57 + 5.0625f * s67;
    const float m65 = 0.03125f * s17 + s27 + 7.59375f * s37;
    const float m66 = 0.015625f * s47 + s57 + 11.390625f * s67 + t67;

    (dst_data + i)[0] = m00 + bias_data[i];
    (dst_data + i + C4NUM)[0] = m01 + bias_data[i];
    (dst_data + i + 2 * C4NUM)[0] = m02 + bias_data[i];
    (dst_data + i + 3 * C4NUM)[0] = m03 + bias_data[i];
    (dst_data + i + 4 * C4NUM)[0] = m04 + bias_data[i];
    (dst_data + i + 5 * C4NUM)[0] = m05 + bias_data[i];
    (dst_data + i + 6 * C4NUM)[0] = m06 + bias_data[i];

    (dst_data + i + dst_step * C4NUM)[0] = m10 + bias_data[i];
    (dst_data + i + dst_step * C4NUM + C4NUM)[0] = m11 + bias_data[i];
    (dst_data + i + dst_step * C4NUM + 2 * C4NUM)[0] = m12 + bias_data[i];
    (dst_data + i + dst_step * C4NUM + 3 * C4NUM)[0] = m13 + bias_data[i];
    (dst_data + i + dst_step * C4NUM + 4 * C4NUM)[0] = m14 + bias_data[i];
    (dst_data + i + dst_step * C4NUM + 5 * C4NUM)[0] = m15 + bias_data[i];
    (dst_data + i + dst_step * C4NUM + 6 * C4NUM)[0] = m16 + bias_data[i];

    (dst_data + i + 2 * dst_step * C4NUM)[0] = m20 + bias_data[i];
    (dst_data + i + 2 * dst_step * C4NUM + C4NUM)[0] = m21 + bias_data[i];
    (dst_data + i + 2 * dst_step * C4NUM + 2 * C4NUM)[0] = m22 + bias_data[i];
    (dst_data + i + 2 * dst_step * C4NUM + 3 * C4NUM)[0] = m23 + bias_data[i];
    (dst_data + i + 2 * dst_step * C4NUM + 4 * C4NUM)[0] = m24 + bias_data[i];
    (dst_data + i + 2 * dst_step * C4NUM + 5 * C4NUM)[0] = m25 + bias_data[i];
    (dst_data + i + 2 * dst_step * C4NUM + 6 * C4NUM)[0] = m26 + bias_data[i];

    (dst_data + i + 3 * dst_step * C4NUM)[0] = m30 + bias_data[i];
    (dst_data + i + 3 * dst_step * C4NUM + C4NUM)[0] = m31 + bias_data[i];
    (dst_data + i + 3 * dst_step * C4NUM + 2 * C4NUM)[0] = m32 + bias_data[i];
    (dst_data + i + 3 * dst_step * C4NUM + 3 * C4NUM)[0] = m33 + bias_data[i];
    (dst_data + i + 3 * dst_step * C4NUM + 4 * C4NUM)[0] = m34 + bias_data[i];
    (dst_data + i + 3 * dst_step * C4NUM + 5 * C4NUM)[0] = m35 + bias_data[i];
    (dst_data + i + 3 * dst_step * C4NUM + 6 * C4NUM)[0] = m36 + bias_data[i];

    (dst_data + i + 4 * dst_step * C4NUM)[0] = m40 + bias_data[i];
    (dst_data + i + 4 * dst_step * C4NUM + C4NUM)[0] = m41 + bias_data[i];
    (dst_data + i + 4 * dst_step * C4NUM + 2 * C4NUM)[0] = m42 + bias_data[i];
    (dst_data + i + 4 * dst_step * C4NUM + 3 * C4NUM)[0] = m43 + bias_data[i];
    (dst_data + i + 4 * dst_step * C4NUM + 4 * C4NUM)[0] = m44 + bias_data[i];
    (dst_data + i + 4 * dst_step * C4NUM + 5 * C4NUM)[0] = m45 + bias_data[i];
    (dst_data + i + 4 * dst_step * C4NUM + 6 * C4NUM)[0] = m46 + bias_data[i];

    (dst_data + i + 5 * dst_step * C4NUM)[0] = m50 + bias_data[i];
    (dst_data + i + 5 * dst_step * C4NUM + C4NUM)[0] = m51 + bias_data[i];
    (dst_data + i + 5 * dst_step * C4NUM + 2 * C4NUM)[0] = m52 + bias_data[i];
    (dst_data + i + 5 * dst_step * C4NUM + 3 * C4NUM)[0] = m53 + bias_data[i];
    (dst_data + i + 5 * dst_step * C4NUM + 4 * C4NUM)[0] = m54 + bias_data[i];
    (dst_data + i + 5 * dst_step * C4NUM + 5 * C4NUM)[0] = m55 + bias_data[i];
    (dst_data + i + 5 * dst_step * C4NUM + 6 * C4NUM)[0] = m56 + bias_data[i];

    (dst_data + i + 6 * dst_step * C4NUM)[0] = m60 + bias_data[i];
    (dst_data + i + 6 * dst_step * C4NUM + C4NUM)[0] = m61 + bias_data[i];
    (dst_data + i + 6 * dst_step * C4NUM + 2 * C4NUM)[0] = m62 + bias_data[i];
    (dst_data + i + 6 * dst_step * C4NUM + 3 * C4NUM)[0] = m63 + bias_data[i];
    (dst_data + i + 6 * dst_step * C4NUM + 4 * C4NUM)[0] = m64 + bias_data[i];
    (dst_data + i + 6 * dst_step * C4NUM + 5 * C4NUM)[0] = m65 + bias_data[i];
    (dst_data + i + 6 * dst_step * C4NUM + 6 * C4NUM)[0] = m66 + bias_data[i];
  }
#endif
}

// Reference to the paper "Fast Algorithms for Convolutional Neural Networks"
// Utilize cost model to compute performance gain.
// If the gain is greater than got from Im2col, winograd algorithm will be chosen.
int SelectOutputUnit(ConvParameter *conv_param) {
  int input_batch = conv_param->input_batch_;
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int in_channel = conv_param->input_channel_;
  int out_h = conv_param->output_h_;
  int out_w = conv_param->output_w_;
  int out_channel = conv_param->output_channel_;
  int out_plane = out_h * out_w;

  int max_unit = sqrt((float)(out_plane));
  max_unit = max_unit > MIN_UNIT ? max_unit : MIN_UNIT;
  max_unit = max_unit < MAX_UNIT ? max_unit : MAX_UNIT;
  int output_unit = 1;
  float ratio = 0.0f;
  // cost of conventional convolution multiplications
  float ori_cost = out_plane * out_channel * in_channel * kernel_h * kernel_w;

  for (int u = MIN_UNIT; u < max_unit; u++) {
    int input_unit = u + kernel_h - 1;
    if (input_unit != 4 && input_unit != 8) {
      continue;
    }
    // don't count filter transform cost, because it can be processed once offline.
    const float input_trans_unit_cost = 2 * input_unit * input_unit * input_unit * in_channel;
    float gemm_unit_cost = input_unit * input_unit * in_channel * out_channel;
    float output_trans_unit_cost = input_unit * u * (u + input_unit) * out_channel;
    // equation (23) in papar
    float winograd_cost = (input_trans_unit_cost + gemm_unit_cost + output_trans_unit_cost) *
                          (UP_DIV(out_w, u) * (UP_DIV(out_h, u))) * input_batch;
    float reduce_rate = ori_cost / winograd_cost;
    if (reduce_rate > ratio && reduce_rate > 1) {
      ratio = reduce_rate;
      output_unit = u;
    }
  }
  // If output_unit is 1, then it is conventional convolution
  return output_unit;
}

InputTransformUnitFunc GetInputTransFunc(int input_unit) {
  if (input_unit == 4) {
    return InputTransform4x4Unit;
  } else if (input_unit == 8) {
    return InputTransform8x8Unit;
  } else {
    printf("Only support 4 or 8 for input unit.");
    return NULL;
  }
}

OutputTransformUnitFunc GetOutputTransFunc(int input_unit, int output_unit) {
  if (input_unit == 4 && output_unit == 2) {
    return OutputTransform4x2Unit;
  } else if (input_unit == 4 && output_unit == 3) {
    return OutputTransform4x3Unit;
  } else if (input_unit == 8) {
    return outputTransformUnit[output_unit];
  } else {
    printf(".");
    return NULL;
  }
}

void CheckIfUseWinograd(bool *use_winograd, int *output_unit, ConvParameter *conv_param,
                        InputTransformUnitFunc input_trans_func, OutputTransformUnitFunc output_trans_func) {
  if (conv_param->kernel_w_ == conv_param->kernel_h_ && conv_param->dilation_h_ == 1 && conv_param->dilation_w_ == 1 &&
      conv_param->stride_h_ == 1 && conv_param->stride_w_ == 1) {
    *output_unit = SelectOutputUnit(conv_param);
    if (*output_unit > 1) {
      *use_winograd = true;
      int input_unit = conv_param->kernel_h_ + *output_unit - 1;
      input_trans_func = GetInputTransFunc(input_unit);
      if (input_trans_func == NULL) {
        *use_winograd = false;
      }
      output_trans_func = GetOutputTransFunc(input_unit, *output_unit);
      if (output_trans_func == NULL) {
        *use_winograd = false;
      }
    } else {
      *use_winograd = false;
    }
  } else {
    *use_winograd = false;
  }
}
