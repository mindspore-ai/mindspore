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

#include "nnacl/int8/conv3x3_int8.h"

void Conv3x3Int8InputUnit(int16_t *tmp_data, int16_t *trans_input_data, size_t step, int input_zp) {
#ifdef ENABLE_ARM
  int16x8_t zp = vdupq_n_s16(input_zp);

  int16x8_t d00 = vsubq_s16(vld1q_s16(tmp_data), zp);
  int16x8_t d01 = vsubq_s16(vld1q_s16(tmp_data + 8), zp);
  int16x8_t d02 = vsubq_s16(vld1q_s16(tmp_data + 2 * 8), zp);
  int16x8_t d03 = vsubq_s16(vld1q_s16(tmp_data + 3 * 8), zp);

  int16x8_t d10 = vsubq_s16(vld1q_s16(tmp_data + 4 * 8), zp);
  int16x8_t d11 = vsubq_s16(vld1q_s16(tmp_data + 5 * 8), zp);
  int16x8_t d12 = vsubq_s16(vld1q_s16(tmp_data + 6 * 8), zp);
  int16x8_t d13 = vsubq_s16(vld1q_s16(tmp_data + 7 * 8), zp);

  int16x8_t d20 = vsubq_s16(vld1q_s16(tmp_data + 8 * 8), zp);
  int16x8_t d21 = vsubq_s16(vld1q_s16(tmp_data + 9 * 8), zp);
  int16x8_t d22 = vsubq_s16(vld1q_s16(tmp_data + 10 * 8), zp);
  int16x8_t d23 = vsubq_s16(vld1q_s16(tmp_data + 11 * 8), zp);

  int16x8_t d30 = vsubq_s16(vld1q_s16(tmp_data + 12 * 8), zp);
  int16x8_t d31 = vsubq_s16(vld1q_s16(tmp_data + 13 * 8), zp);
  int16x8_t d32 = vsubq_s16(vld1q_s16(tmp_data + 14 * 8), zp);
  int16x8_t d33 = vsubq_s16(vld1q_s16(tmp_data + 15 * 8), zp);

  int16x8_t t00 = vsubq_s16(d00, d20);
  int16x8_t t01 = vsubq_s16(d01, d21);
  int16x8_t t02 = vsubq_s16(d02, d22);
  int16x8_t t03 = vsubq_s16(d03, d23);

  int16x8_t t10 = vaddq_s16(d10, d20);
  int16x8_t t11 = vaddq_s16(d11, d21);
  int16x8_t t12 = vaddq_s16(d12, d22);
  int16x8_t t13 = vaddq_s16(d13, d23);

  int16x8_t t20 = vsubq_s16(d20, d10);
  int16x8_t t21 = vsubq_s16(d21, d11);
  int16x8_t t22 = vsubq_s16(d22, d12);
  int16x8_t t23 = vsubq_s16(d23, d13);

  int16x8_t t30 = vsubq_s16(d10, d30);
  int16x8_t t31 = vsubq_s16(d11, d31);
  int16x8_t t32 = vsubq_s16(d12, d32);
  int16x8_t t33 = vsubq_s16(d13, d33);

  int16x8_t m00 = vsubq_s16(t00, t02);
  int16x8_t m01 = vaddq_s16(t01, t02);
  int16x8_t m02 = vsubq_s16(t02, t01);
  int16x8_t m03 = vsubq_s16(t01, t03);

  int16x8_t m10 = vsubq_s16(t10, t12);
  int16x8_t m11 = vaddq_s16(t11, t12);
  int16x8_t m12 = vsubq_s16(t12, t11);
  int16x8_t m13 = vsubq_s16(t11, t13);

  int16x8_t m20 = vsubq_s16(t20, t22);
  int16x8_t m21 = vaddq_s16(t21, t22);
  int16x8_t m22 = vsubq_s16(t22, t21);
  int16x8_t m23 = vsubq_s16(t21, t23);

  int16x8_t m30 = vsubq_s16(t30, t32);
  int16x8_t m31 = vaddq_s16(t31, t32);
  int16x8_t m32 = vsubq_s16(t32, t31);
  int16x8_t m33 = vsubq_s16(t31, t33);

  vst1q_s16(trans_input_data, m00);
  vst1q_s16(trans_input_data + step, m01);
  vst1q_s16(trans_input_data + 2 * step, m02);
  vst1q_s16(trans_input_data + 3 * step, m03);

  vst1q_s16(trans_input_data + 4 * step, m10);
  vst1q_s16(trans_input_data + 5 * step, m11);
  vst1q_s16(trans_input_data + 6 * step, m12);
  vst1q_s16(trans_input_data + 7 * step, m13);

  vst1q_s16(trans_input_data + 8 * step, m20);
  vst1q_s16(trans_input_data + 9 * step, m21);
  vst1q_s16(trans_input_data + 10 * step, m22);
  vst1q_s16(trans_input_data + 11 * step, m23);

  vst1q_s16(trans_input_data + 12 * step, m30);
  vst1q_s16(trans_input_data + 13 * step, m31);
  vst1q_s16(trans_input_data + 14 * step, m32);
  vst1q_s16(trans_input_data + 15 * step, m33);
#else
  for (int i = 0; i < C8NUM; i++) {
    int16_t *local_ptr = tmp_data + i;
    int16_t d00 = local_ptr[0] - input_zp;
    int16_t d01 = (local_ptr + C8NUM)[0] - input_zp;
    int16_t d02 = (local_ptr + 2 * C8NUM)[0] - input_zp;
    int16_t d03 = (local_ptr + 3 * C8NUM)[0] - input_zp;

    int16_t d10 = (local_ptr + 4 * C8NUM)[0] - input_zp;
    int16_t d11 = (local_ptr + 5 * C8NUM)[0] - input_zp;
    int16_t d12 = (local_ptr + 6 * C8NUM)[0] - input_zp;
    int16_t d13 = (local_ptr + 7 * C8NUM)[0] - input_zp;

    int16_t d20 = (local_ptr + 8 * C8NUM)[0] - input_zp;
    int16_t d21 = (local_ptr + 9 * C8NUM)[0] - input_zp;
    int16_t d22 = (local_ptr + 10 * C8NUM)[0] - input_zp;
    int16_t d23 = (local_ptr + 11 * C8NUM)[0] - input_zp;

    int16_t d30 = (local_ptr + 12 * C8NUM)[0] - input_zp;
    int16_t d31 = (local_ptr + 13 * C8NUM)[0] - input_zp;
    int16_t d32 = (local_ptr + 14 * C8NUM)[0] - input_zp;
    int16_t d33 = (local_ptr + 15 * C8NUM)[0] - input_zp;

    int16_t t00 = d00 - d20;
    int16_t t01 = d01 - d21;
    int16_t t02 = d02 - d22;
    int16_t t03 = d03 - d23;

    int16_t t10 = d10 + d20;
    int16_t t11 = d11 + d21;
    int16_t t12 = d12 + d22;
    int16_t t13 = d13 + d23;

    int16_t t20 = d20 - d10;
    int16_t t21 = d21 - d11;
    int16_t t22 = d22 - d12;
    int16_t t23 = d23 - d13;

    int16_t t30 = d10 - d30;
    int16_t t31 = d11 - d31;
    int16_t t32 = d12 - d32;
    int16_t t33 = d13 - d33;

    int16_t m00 = t00 - t02;
    int16_t m01 = t01 + t02;
    int16_t m02 = t02 - t01;
    int16_t m03 = t01 - t03;

    int16_t m10 = t10 - t12;
    int16_t m11 = t11 + t12;
    int16_t m12 = t12 - t11;
    int16_t m13 = t11 - t13;

    int16_t m20 = t20 - t22;
    int16_t m21 = t21 + t22;
    int16_t m22 = t22 - t21;
    int16_t m23 = t21 - t23;

    int16_t m30 = t30 - t32;
    int16_t m31 = t31 + t32;
    int16_t m32 = t32 - t31;
    int16_t m33 = t31 - t33;

    (trans_input_data + i)[0] = m00;
    (trans_input_data + i + step)[0] = m01;
    (trans_input_data + i + 2 * step)[0] = m02;
    (trans_input_data + i + 3 * step)[0] = m03;

    (trans_input_data + i + 4 * step)[0] = m10;
    (trans_input_data + i + 5 * step)[0] = m11;
    (trans_input_data + i + 6 * step)[0] = m12;
    (trans_input_data + i + 7 * step)[0] = m13;

    (trans_input_data + i + 8 * step)[0] = m20;
    (trans_input_data + i + 9 * step)[0] = m21;
    (trans_input_data + i + 10 * step)[0] = m22;
    (trans_input_data + i + 11 * step)[0] = m23;

    (trans_input_data + i + 12 * step)[0] = m30;
    (trans_input_data + i + 13 * step)[0] = m31;
    (trans_input_data + i + 14 * step)[0] = m32;
    (trans_input_data + i + 15 * step)[0] = m33;
  }
#endif
}

void Conv3x3Int8FilterTransform(const int16_t *weight_data, int16_t *trans_weight, int iC8, int output_channel,
                                int kernel_plane) {
  const int input_unit = 4;
  int dst_step = iC8 * C8NUM * C4NUM;
  for (int o = 0; o < output_channel; o++) {
    int oc4_block_num = o / C4NUM;
    int oc4_block_rem = o % C4NUM;
    int src_oc_offset = o * iC8 * C8NUM * kernel_plane;
    int dst_oc_offset = oc4_block_num * C4NUM * iC8 * C8NUM * input_unit * input_unit + oc4_block_rem;
    for (int i = 0; i < iC8; i++) {
      const int16_t *src_ic8_ptr = weight_data + src_oc_offset + i * kernel_plane * C8NUM;
      int16_t *dst_ic8_ptr = trans_weight + dst_oc_offset + i * C4NUM * C8NUM;
#ifdef ENABLE_ARM
      int16x8_t g00 = vld1q_s16(src_ic8_ptr);
      int16x8_t g01 = vld1q_s16(src_ic8_ptr + 8);
      int16x8_t g02 = vld1q_s16(src_ic8_ptr + 2 * 8);
      int16x8_t g10 = vld1q_s16(src_ic8_ptr + 3 * 8);
      int16x8_t g11 = vld1q_s16(src_ic8_ptr + 4 * 8);
      int16x8_t g12 = vld1q_s16(src_ic8_ptr + 5 * 8);
      int16x8_t g20 = vld1q_s16(src_ic8_ptr + 6 * 8);
      int16x8_t g21 = vld1q_s16(src_ic8_ptr + 7 * 8);
      int16x8_t g22 = vld1q_s16(src_ic8_ptr + 8 * 8);

      int16x8_t dst00 = vmulq_n_s16(g00, 2);
      int16x8_t dst01 = vmulq_n_s16(g01, 2);
      int16x8_t dst02 = vmulq_n_s16(g02, 2);

      int16x8_t dst10 = vaddq_s16(vaddq_s16(g00, g10), g20);
      int16x8_t dst11 = vaddq_s16(vaddq_s16(g01, g11), g21);
      int16x8_t dst12 = vaddq_s16(vaddq_s16(g02, g12), g22);

      int16x8_t dst20 = vaddq_s16(vsubq_s16(g00, g10), g20);
      int16x8_t dst21 = vaddq_s16(vsubq_s16(g01, g11), g21);
      int16x8_t dst22 = vaddq_s16(vsubq_s16(g02, g12), g22);

      int16x8_t dst30 = vmulq_n_s16(g20, 2);
      int16x8_t dst31 = vmulq_n_s16(g21, 2);
      int16x8_t dst32 = vmulq_n_s16(g22, 2);

      int16x8_t m00 = vmulq_n_s16(dst00, 2);
      int16x8_t m01 = vaddq_s16(vaddq_s16(dst00, dst01), dst02);
      int16x8_t m02 = vaddq_s16(vsubq_s16(dst00, dst01), dst02);
      int16x8_t m03 = vmulq_n_s16(dst02, 2);

      int16x8_t m10 = vmulq_n_s16(dst10, 2);
      int16x8_t m11 = vaddq_s16(vaddq_s16(dst10, dst11), dst12);
      int16x8_t m12 = vaddq_s16(vsubq_s16(dst10, dst11), dst12);
      int16x8_t m13 = vmulq_n_s16(dst12, 2);

      int16x8_t m20 = vmulq_n_s16(dst20, 2);
      int16x8_t m21 = vaddq_s16(vaddq_s16(dst20, dst21), dst22);
      int16x8_t m22 = vaddq_s16(vsubq_s16(dst20, dst21), dst22);
      int16x8_t m23 = vmulq_n_s16(dst22, 2);

      int16x8_t m30 = vmulq_n_s16(dst30, 2);
      int16x8_t m31 = vaddq_s16(vaddq_s16(dst30, dst31), dst32);
      int16x8_t m32 = vaddq_s16(vsubq_s16(dst30, dst31), dst32);
      int16x8_t m33 = vmulq_n_s16(dst32, 2);

      dst_ic8_ptr[0] = m00[0];
      dst_ic8_ptr[4] = m00[1];
      dst_ic8_ptr[8] = m00[2];
      dst_ic8_ptr[12] = m00[3];
      dst_ic8_ptr[16] = m00[4];
      dst_ic8_ptr[20] = m00[5];
      dst_ic8_ptr[24] = m00[6];
      dst_ic8_ptr[28] = m00[7];

      dst_ic8_ptr[0 + dst_step] = m01[0];
      dst_ic8_ptr[4 + dst_step] = m01[1];
      dst_ic8_ptr[8 + dst_step] = m01[2];
      dst_ic8_ptr[12 + dst_step] = m01[3];
      dst_ic8_ptr[16 + dst_step] = m01[4];
      dst_ic8_ptr[20 + dst_step] = m01[5];
      dst_ic8_ptr[24 + dst_step] = m01[6];
      dst_ic8_ptr[28 + dst_step] = m01[7];

      dst_ic8_ptr[0 + 2 * dst_step] = m02[0];
      dst_ic8_ptr[4 + 2 * dst_step] = m02[1];
      dst_ic8_ptr[8 + 2 * dst_step] = m02[2];
      dst_ic8_ptr[12 + 2 * dst_step] = m02[3];
      dst_ic8_ptr[16 + 2 * dst_step] = m02[4];
      dst_ic8_ptr[20 + 2 * dst_step] = m02[5];
      dst_ic8_ptr[24 + 2 * dst_step] = m02[6];
      dst_ic8_ptr[28 + 2 * dst_step] = m02[7];

      dst_ic8_ptr[0 + 3 * dst_step] = m03[0];
      dst_ic8_ptr[4 + 3 * dst_step] = m03[1];
      dst_ic8_ptr[8 + 3 * dst_step] = m03[2];
      dst_ic8_ptr[12 + 3 * dst_step] = m03[3];
      dst_ic8_ptr[16 + 3 * dst_step] = m03[4];
      dst_ic8_ptr[20 + 3 * dst_step] = m03[5];
      dst_ic8_ptr[24 + 3 * dst_step] = m03[6];
      dst_ic8_ptr[28 + 3 * dst_step] = m03[7];

      dst_ic8_ptr[0 + 4 * dst_step] = m10[0];
      dst_ic8_ptr[4 + 4 * dst_step] = m10[1];
      dst_ic8_ptr[8 + 4 * dst_step] = m10[2];
      dst_ic8_ptr[12 + 4 * dst_step] = m10[3];
      dst_ic8_ptr[16 + 4 * dst_step] = m10[4];
      dst_ic8_ptr[20 + 4 * dst_step] = m10[5];
      dst_ic8_ptr[24 + 4 * dst_step] = m10[6];
      dst_ic8_ptr[28 + 4 * dst_step] = m10[7];

      dst_ic8_ptr[0 + 5 * dst_step] = m11[0];
      dst_ic8_ptr[4 + 5 * dst_step] = m11[1];
      dst_ic8_ptr[8 + 5 * dst_step] = m11[2];
      dst_ic8_ptr[12 + 5 * dst_step] = m11[3];
      dst_ic8_ptr[16 + 5 * dst_step] = m11[4];
      dst_ic8_ptr[20 + 5 * dst_step] = m11[5];
      dst_ic8_ptr[24 + 5 * dst_step] = m11[6];
      dst_ic8_ptr[28 + 5 * dst_step] = m11[7];

      dst_ic8_ptr[0 + 6 * dst_step] = m12[0];
      dst_ic8_ptr[4 + 6 * dst_step] = m12[1];
      dst_ic8_ptr[8 + 6 * dst_step] = m12[2];
      dst_ic8_ptr[12 + 6 * dst_step] = m12[3];
      dst_ic8_ptr[16 + 6 * dst_step] = m12[4];
      dst_ic8_ptr[20 + 6 * dst_step] = m12[5];
      dst_ic8_ptr[24 + 6 * dst_step] = m12[6];
      dst_ic8_ptr[28 + 6 * dst_step] = m12[7];

      dst_ic8_ptr[0 + 7 * dst_step] = m13[0];
      dst_ic8_ptr[4 + 7 * dst_step] = m13[1];
      dst_ic8_ptr[8 + 7 * dst_step] = m13[2];
      dst_ic8_ptr[12 + 7 * dst_step] = m13[3];
      dst_ic8_ptr[16 + 7 * dst_step] = m13[4];
      dst_ic8_ptr[20 + 7 * dst_step] = m13[5];
      dst_ic8_ptr[24 + 7 * dst_step] = m13[6];
      dst_ic8_ptr[28 + 7 * dst_step] = m13[7];

      dst_ic8_ptr[0 + 8 * dst_step] = m20[0];
      dst_ic8_ptr[4 + 8 * dst_step] = m20[1];
      dst_ic8_ptr[8 + 8 * dst_step] = m20[2];
      dst_ic8_ptr[12 + 8 * dst_step] = m20[3];
      dst_ic8_ptr[16 + 8 * dst_step] = m20[4];
      dst_ic8_ptr[20 + 8 * dst_step] = m20[5];
      dst_ic8_ptr[24 + 8 * dst_step] = m20[6];
      dst_ic8_ptr[28 + 8 * dst_step] = m20[7];

      dst_ic8_ptr[0 + 9 * dst_step] = m21[0];
      dst_ic8_ptr[4 + 9 * dst_step] = m21[1];
      dst_ic8_ptr[8 + 9 * dst_step] = m21[2];
      dst_ic8_ptr[12 + 9 * dst_step] = m21[3];
      dst_ic8_ptr[16 + 9 * dst_step] = m21[4];
      dst_ic8_ptr[20 + 9 * dst_step] = m21[5];
      dst_ic8_ptr[24 + 9 * dst_step] = m21[6];
      dst_ic8_ptr[28 + 9 * dst_step] = m21[7];

      dst_ic8_ptr[0 + 10 * dst_step] = m22[0];
      dst_ic8_ptr[4 + 10 * dst_step] = m22[1];
      dst_ic8_ptr[8 + 10 * dst_step] = m22[2];
      dst_ic8_ptr[12 + 10 * dst_step] = m22[3];
      dst_ic8_ptr[16 + 10 * dst_step] = m22[4];
      dst_ic8_ptr[20 + 10 * dst_step] = m22[5];
      dst_ic8_ptr[24 + 10 * dst_step] = m22[6];
      dst_ic8_ptr[28 + 10 * dst_step] = m22[7];

      dst_ic8_ptr[0 + 11 * dst_step] = m23[0];
      dst_ic8_ptr[4 + 11 * dst_step] = m23[1];
      dst_ic8_ptr[8 + 11 * dst_step] = m23[2];
      dst_ic8_ptr[12 + 11 * dst_step] = m23[3];
      dst_ic8_ptr[16 + 11 * dst_step] = m23[4];
      dst_ic8_ptr[20 + 11 * dst_step] = m23[5];
      dst_ic8_ptr[24 + 11 * dst_step] = m23[6];
      dst_ic8_ptr[28 + 11 * dst_step] = m23[7];

      dst_ic8_ptr[0 + 12 * dst_step] = m30[0];
      dst_ic8_ptr[4 + 12 * dst_step] = m30[1];
      dst_ic8_ptr[8 + 12 * dst_step] = m30[2];
      dst_ic8_ptr[12 + 12 * dst_step] = m30[3];
      dst_ic8_ptr[16 + 12 * dst_step] = m30[4];
      dst_ic8_ptr[20 + 12 * dst_step] = m30[5];
      dst_ic8_ptr[24 + 12 * dst_step] = m30[6];
      dst_ic8_ptr[28 + 12 * dst_step] = m30[7];

      dst_ic8_ptr[0 + 13 * dst_step] = m31[0];
      dst_ic8_ptr[4 + 13 * dst_step] = m31[1];
      dst_ic8_ptr[8 + 13 * dst_step] = m31[2];
      dst_ic8_ptr[12 + 13 * dst_step] = m31[3];
      dst_ic8_ptr[16 + 13 * dst_step] = m31[4];
      dst_ic8_ptr[20 + 13 * dst_step] = m31[5];
      dst_ic8_ptr[24 + 13 * dst_step] = m31[6];
      dst_ic8_ptr[28 + 13 * dst_step] = m31[7];

      dst_ic8_ptr[0 + 14 * dst_step] = m32[0];
      dst_ic8_ptr[4 + 14 * dst_step] = m32[1];
      dst_ic8_ptr[8 + 14 * dst_step] = m32[2];
      dst_ic8_ptr[12 + 14 * dst_step] = m32[3];
      dst_ic8_ptr[16 + 14 * dst_step] = m32[4];
      dst_ic8_ptr[20 + 14 * dst_step] = m32[5];
      dst_ic8_ptr[24 + 14 * dst_step] = m32[6];
      dst_ic8_ptr[28 + 14 * dst_step] = m32[7];

      dst_ic8_ptr[0 + 15 * dst_step] = m33[0];
      dst_ic8_ptr[4 + 15 * dst_step] = m33[1];
      dst_ic8_ptr[8 + 15 * dst_step] = m33[2];
      dst_ic8_ptr[12 + 15 * dst_step] = m33[3];
      dst_ic8_ptr[16 + 15 * dst_step] = m33[4];
      dst_ic8_ptr[20 + 15 * dst_step] = m33[5];
      dst_ic8_ptr[24 + 15 * dst_step] = m33[6];
      dst_ic8_ptr[28 + 15 * dst_step] = m33[7];
#else
      for (int j = 0; j < C8NUM; j++) {
        const int16_t *local_ptr = src_ic8_ptr + j;
        int16_t dst00 = local_ptr[0] * 2;
        int16_t dst01 = (local_ptr + 8)[0] * 2;
        int16_t dst02 = (local_ptr + 16)[0] * 2;

        int16_t dst10 = local_ptr[0] + (local_ptr + 24)[0] + (local_ptr + 48)[0];
        int16_t dst11 = (local_ptr + 8)[0] + (local_ptr + 32)[0] + (local_ptr + 56)[0];
        int16_t dst12 = (local_ptr + 16)[0] + (local_ptr + 40)[0] + (local_ptr + 64)[0];

        int16_t dst20 = local_ptr[0] - (local_ptr + 24)[0] + (local_ptr + 48)[0];
        int16_t dst21 = (local_ptr + 8)[0] - (local_ptr + 32)[0] + (local_ptr + 56)[0];
        int16_t dst22 = (local_ptr + 16)[0] - (local_ptr + 40)[0] + (local_ptr + 64)[0];

        int16_t dst30 = (local_ptr + 48)[0] * 2;
        int16_t dst31 = (local_ptr + 56)[0] * 2;
        int16_t dst32 = (local_ptr + 64)[0] * 2;

        int16_t m00 = dst00 * 2;
        int16_t m01 = dst00 + dst01 + dst02;
        int16_t m02 = dst00 - dst01 + dst02;
        int16_t m03 = dst02 * 2;

        int16_t m10 = dst10 * 2;
        int16_t m11 = dst10 + dst11 + dst12;
        int16_t m12 = dst10 - dst11 + dst12;
        int16_t m13 = dst12 * 2;

        int16_t m20 = dst20 * 2;
        int16_t m21 = dst20 + dst21 + dst22;
        int16_t m22 = dst20 - dst21 + dst22;
        int16_t m23 = dst22 * 2;

        int16_t m30 = dst30 * 2;
        int16_t m31 = dst30 + dst31 + dst32;
        int16_t m32 = dst30 - dst31 + dst32;
        int16_t m33 = dst32 * 2;

        *(dst_ic8_ptr + j * 4) = m00;
        *(dst_ic8_ptr + j * 4 + dst_step) = m01;
        *(dst_ic8_ptr + j * 4 + 2 * dst_step) = m02;
        *(dst_ic8_ptr + j * 4 + 3 * dst_step) = m03;

        *(dst_ic8_ptr + j * 4 + 4 * dst_step) = m10;
        *(dst_ic8_ptr + j * 4 + 5 * dst_step) = m11;
        *(dst_ic8_ptr + j * 4 + 6 * dst_step) = m12;
        *(dst_ic8_ptr + j * 4 + 7 * dst_step) = m13;

        *(dst_ic8_ptr + j * 4 + 8 * dst_step) = m20;
        *(dst_ic8_ptr + j * 4 + 9 * dst_step) = m21;
        *(dst_ic8_ptr + j * 4 + 10 * dst_step) = m22;
        *(dst_ic8_ptr + j * 4 + 11 * dst_step) = m23;

        *(dst_ic8_ptr + j * 4 + 12 * dst_step) = m30;
        *(dst_ic8_ptr + j * 4 + 13 * dst_step) = m31;
        *(dst_ic8_ptr + j * 4 + 14 * dst_step) = m32;
        *(dst_ic8_ptr + j * 4 + 15 * dst_step) = m33;
      }
#endif
    }
  }
}

void Conv3x3Int8OutputUnit(const int32_t *gemm_out, const int32_t *bias_data, int8_t *output_data, bool h_not_bound,
                           bool w_not_bound, int output_w, int real_num, int oc_start, ConvParameter *conv_param) {
  int32_t *left_shift = conv_param->conv_quant_arg_.left_shift_;
  int32_t *right_shift = conv_param->conv_quant_arg_.right_shift_;
  int32_t *quant_multiplier = conv_param->conv_quant_arg_.quant_multiplier_;
  int output_zp = conv_param->conv_quant_arg_.output_quant_args_[0].zp_;
  int out_min = conv_param->conv_quant_arg_.out_act_min_[0];
  int out_max = conv_param->conv_quant_arg_.out_act_max_[0];

#ifdef ENABLE_ARM
  int32x4_t bias_ptr = vld1q_s32(bias_data);

  int32x4_t s00 = vld1q_s32(gemm_out);
  int32x4_t s01 = vld1q_s32(gemm_out + 4);
  int32x4_t s02 = vld1q_s32(gemm_out + 8);
  int32x4_t s03 = vld1q_s32(gemm_out + 12);

  int32x4_t s10 = vld1q_s32(gemm_out + 16);
  int32x4_t s11 = vld1q_s32(gemm_out + 20);
  int32x4_t s12 = vld1q_s32(gemm_out + 24);
  int32x4_t s13 = vld1q_s32(gemm_out + 28);

  int32x4_t s20 = vld1q_s32(gemm_out + 32);
  int32x4_t s21 = vld1q_s32(gemm_out + 36);
  int32x4_t s22 = vld1q_s32(gemm_out + 40);
  int32x4_t s23 = vld1q_s32(gemm_out + 44);

  int32x4_t s30 = vld1q_s32(gemm_out + 48);
  int32x4_t s31 = vld1q_s32(gemm_out + 52);
  int32x4_t s32 = vld1q_s32(gemm_out + 56);
  int32x4_t s33 = vld1q_s32(gemm_out + 60);

  int32x4_t t00 = vshrq_n_s32(vaddq_s32(vaddq_s32(s00, s10), s20), 1);
  int32x4_t t01 = vshrq_n_s32(vaddq_s32(vaddq_s32(s01, s11), s21), 1);
  int32x4_t t02 = vshrq_n_s32(vaddq_s32(vaddq_s32(s02, s12), s22), 1);
  int32x4_t t03 = vshrq_n_s32(vaddq_s32(vaddq_s32(s03, s13), s23), 1);

  int32x4_t t10 = vshrq_n_s32(vsubq_s32(vsubq_s32(s10, s20), s30), 1);
  int32x4_t t11 = vshrq_n_s32(vsubq_s32(vsubq_s32(s11, s21), s31), 1);
  int32x4_t t12 = vshrq_n_s32(vsubq_s32(vsubq_s32(s12, s22), s32), 1);
  int32x4_t t13 = vshrq_n_s32(vsubq_s32(vsubq_s32(s13, s23), s33), 1);

  int32x4_t d00 = vaddq_s32(vshrq_n_s32(vaddq_s32(vaddq_s32(t00, t01), t02), 1), bias_ptr);
  int32x4_t d01 = vaddq_s32(vshrq_n_s32(vsubq_s32(vsubq_s32(t01, t02), t03), 1), bias_ptr);

  int32x4_t d10 = vaddq_s32(vshrq_n_s32(vaddq_s32(vaddq_s32(t10, t11), t12), 1), bias_ptr);
  int32x4_t d11 = vaddq_s32(vshrq_n_s32(vsubq_s32(vsubq_s32(t11, t12), t13), 1), bias_ptr);

  int32x4_t out_multiplier;
  int32x4_t ls;
  int32x4_t rs;
  if ((conv_param->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL)) {
    out_multiplier = vld1q_s32(quant_multiplier + oc_start);
    ls = vld1q_s32(left_shift + oc_start);
    rs = vld1q_s32(right_shift + oc_start);
  } else {
    out_multiplier = vdupq_n_s32(quant_multiplier[0]);
    ls = vdupq_n_s32(left_shift[0]);
    rs = vdupq_n_s32(right_shift[0]);
  }
  int32x4_t out_zp = vdupq_n_s32(output_zp);
  int32x4_t output_min = vdupq_n_s32(out_min);
  int32x4_t output_max = vdupq_n_s32(out_max);

  d00 = vqshlq_s32(d00, ls);
  d00 = vqrdmulhq_s32(d00, out_multiplier);
  int32x4_t carry = vandq_s32(d00, rs);
  carry = vshrq_n_s32(carry, 31);
  d00 = vqaddq_s32(d00, carry);
  d00 = vqrshlq_s32(d00, rs);
  d00 = vaddq_s32(d00, out_zp);
  d00 = vmaxq_s32(d00, output_min);
  d00 = vminq_s32(d00, output_max);

  d01 = vqshlq_s32(d01, ls);
  d01 = vqrdmulhq_s32(d01, out_multiplier);
  carry = vandq_s32(d01, rs);
  carry = vshrq_n_s32(carry, 31);
  d01 = vqaddq_s32(d01, carry);
  d01 = vqrshlq_s32(d01, rs);
  d01 = vaddq_s32(d01, out_zp);
  d01 = vmaxq_s32(d01, output_min);
  d01 = vminq_s32(d01, output_max);

  d10 = vqshlq_s32(d10, ls);
  d10 = vqrdmulhq_s32(d10, out_multiplier);
  carry = vandq_s32(d10, rs);
  carry = vshrq_n_s32(carry, 31);
  d10 = vqaddq_s32(d10, carry);
  d10 = vqrshlq_s32(d10, rs);
  d10 = vaddq_s32(d10, out_zp);
  d10 = vmaxq_s32(d10, output_min);
  d10 = vminq_s32(d10, output_max);

  d11 = vqshlq_s32(d11, ls);
  d11 = vqrdmulhq_s32(d11, out_multiplier);
  carry = vandq_s32(d11, rs);
  carry = vshrq_n_s32(carry, 31);
  d11 = vqaddq_s32(d11, carry);
  d11 = vqrshlq_s32(d11, rs);
  d11 = vaddq_s32(d11, out_zp);
  d11 = vmaxq_s32(d11, output_min);
  d11 = vminq_s32(d11, output_max);

  (output_data)[0] = (int8_t)d00[0];
  (output_data + 1)[0] = (int8_t)d00[1];
  (output_data + 2)[0] = (int8_t)d00[2];
  (output_data + 3)[0] = (int8_t)d00[3];

  if (w_not_bound) {
    *(output_data + 4) = (int8_t)d01[0];
    *(output_data + 5) = (int8_t)d01[1];
    *(output_data + 6) = (int8_t)d01[2];
    *(output_data + 7) = (int8_t)d01[3];
  }
  if (h_not_bound) {
    *(output_data + output_w * 4) = (int8_t)d10[0];
    *(output_data + output_w * 4 + 1) = (int8_t)d10[1];
    *(output_data + output_w * 4 + 2) = (int8_t)d10[2];
    *(output_data + output_w * 4 + 3) = (int8_t)d10[3];
    if (w_not_bound) {
      *(output_data + output_w * 4 + 4) = (int8_t)d11[0];
      *(output_data + output_w * 4 + 5) = (int8_t)d11[1];
      *(output_data + output_w * 4 + 6) = (int8_t)d11[2];
      *(output_data + output_w * 4 + 7) = (int8_t)d11[3];
    }
  }
#else
  if ((conv_param->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL)) {
    for (int i = 0; i < C4NUM; i++) {
      const int32_t *local_ptr = gemm_out + i;
      const int32_t *bias_ptr = bias_data + i;

      int32_t s00 = local_ptr[0];
      int32_t s01 = (local_ptr + 4)[0];
      int32_t s02 = (local_ptr + 8)[0];
      int32_t s03 = (local_ptr + 12)[0];

      int32_t s10 = (local_ptr + 16)[0];
      int32_t s11 = (local_ptr + 20)[0];
      int32_t s12 = (local_ptr + 24)[0];
      int32_t s13 = (local_ptr + 28)[0];

      int32_t s20 = (local_ptr + 32)[0];
      int32_t s21 = (local_ptr + 36)[0];
      int32_t s22 = (local_ptr + 40)[0];
      int32_t s23 = (local_ptr + 44)[0];

      int32_t s30 = (local_ptr + 48)[0];
      int32_t s31 = (local_ptr + 52)[0];
      int32_t s32 = (local_ptr + 56)[0];
      int32_t s33 = (local_ptr + 60)[0];

      int32_t t00 = (s00 + s10 + s20) / 2;
      int32_t t01 = (s01 + s11 + s21) / 2;
      int32_t t02 = (s02 + s12 + s22) / 2;
      int32_t t03 = (s03 + s13 + s23) / 2;

      int32_t t10 = (s10 - s20 - s30) / 2;
      int32_t t11 = (s11 - s21 - s31) / 2;
      int32_t t12 = (s12 - s22 - s32) / 2;
      int32_t t13 = (s13 - s23 - s33) / 2;

      int32_t d00 = (t00 + t01 + t02) / 2 + bias_ptr[0];
      int32_t d01 = (t01 - t02 - t03) / 2 + bias_ptr[0];

      int32_t d10 = (t10 + t11 + t12) / 2 + bias_ptr[0];
      int32_t d11 = (t11 - t12 - t13) / 2 + bias_ptr[0];

      int oc_index = oc_start + i;
      d00 = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(d00 * (1 << (unsigned int)left_shift[oc_index]), quant_multiplier[oc_index]),
        -right_shift[oc_index]);
      d00 += output_zp;
      d00 = d00 > out_min ? d00 : out_min;
      d00 = d00 < out_max ? d00 : out_max;

      d01 = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(d01 * (1 << (unsigned int)left_shift[oc_index]), quant_multiplier[oc_index]),
        -right_shift[oc_index]);
      d01 += output_zp;
      d01 = d01 > out_min ? d01 : out_min;
      d01 = d01 < out_max ? d01 : out_max;

      d10 = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(d10 * (1 << (unsigned int)left_shift[oc_index]), quant_multiplier[oc_index]),
        -right_shift[oc_index]);
      d10 += output_zp;
      d10 = d10 > out_min ? d10 : out_min;
      d10 = d10 < out_max ? d10 : out_max;

      d11 = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(d11 * (1 << (unsigned int)left_shift[oc_index]), quant_multiplier[oc_index]),
        -right_shift[oc_index]);
      d11 += output_zp;
      d11 = d11 > out_min ? d11 : out_min;
      d11 = d11 < out_max ? d11 : out_max;

      (output_data + i)[0] = (int8_t)d00;
      if (w_not_bound) {
        (output_data + i + C4NUM)[0] = (int8_t)d01;
      }
      if (h_not_bound) {
        (output_data + i + output_w * C4NUM)[0] = (int8_t)d10;
        if (w_not_bound) {
          (output_data + i + output_w * C4NUM + C4NUM)[0] = (int8_t)d11;
        }
      }
    }
  } else {
    for (int i = 0; i < C4NUM; i++) {
      const int32_t *local_ptr = gemm_out + i;
      const int32_t *bias_ptr = bias_data + i;

      int32_t s00 = local_ptr[0];
      int32_t s01 = (local_ptr + 4)[0];
      int32_t s02 = (local_ptr + 8)[0];
      int32_t s03 = (local_ptr + 12)[0];

      int32_t s10 = (local_ptr + 16)[0];
      int32_t s11 = (local_ptr + 20)[0];
      int32_t s12 = (local_ptr + 24)[0];
      int32_t s13 = (local_ptr + 28)[0];

      int32_t s20 = (local_ptr + 32)[0];
      int32_t s21 = (local_ptr + 36)[0];
      int32_t s22 = (local_ptr + 40)[0];
      int32_t s23 = (local_ptr + 44)[0];

      int32_t s30 = (local_ptr + 48)[0];
      int32_t s31 = (local_ptr + 52)[0];
      int32_t s32 = (local_ptr + 56)[0];
      int32_t s33 = (local_ptr + 60)[0];

      int32_t t00 = (s00 + s10 + s20) / 2;
      int32_t t01 = (s01 + s11 + s21) / 2;
      int32_t t02 = (s02 + s12 + s22) / 2;
      int32_t t03 = (s03 + s13 + s23) / 2;

      int32_t t10 = (s10 - s20 - s30) / 2;
      int32_t t11 = (s11 - s21 - s31) / 2;
      int32_t t12 = (s12 - s22 - s32) / 2;
      int32_t t13 = (s13 - s23 - s33) / 2;

      int32_t d00 = (t00 + t01 + t02) / 2 + bias_ptr[0];
      int32_t d01 = (t01 - t02 - t03) / 2 + bias_ptr[0];

      int32_t d10 = (t10 + t11 + t12) / 2 + bias_ptr[0];
      int32_t d11 = (t11 - t12 - t13) / 2 + bias_ptr[0];

      d00 = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(d00 * (1 << (unsigned int)left_shift[0]), quant_multiplier[0]),
        -right_shift[0]);
      d00 += output_zp;
      d00 = d00 > out_min ? d00 : out_min;
      d00 = d00 < out_max ? d00 : out_max;

      d01 = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(d01 * (1 << (unsigned int)left_shift[0]), quant_multiplier[0]),
        -right_shift[0]);
      d01 += output_zp;
      d01 = d01 > out_min ? d01 : out_min;
      d01 = d01 < out_max ? d01 : out_max;

      d10 = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(d10 * (1 << (unsigned int)left_shift[0]), quant_multiplier[0]),
        -right_shift[0]);
      d10 += output_zp;
      d10 = d10 > out_min ? d10 : out_min;
      d10 = d10 < out_max ? d10 : out_max;

      d11 = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(d11 * (1 << (unsigned int)left_shift[0]), quant_multiplier[0]),
        -right_shift[0]);
      d11 += output_zp;
      d11 = d11 > out_min ? d11 : out_min;
      d11 = d11 < out_max ? d11 : out_max;

      (output_data + i)[0] = (int8_t)d00;
      if (w_not_bound) {
        (output_data + i + C4NUM)[0] = (int8_t)d01;
      }
      if (h_not_bound) {
        (output_data + i + output_w * C4NUM)[0] = (int8_t)d10;
        if (w_not_bound) {
          (output_data + i + output_w * C4NUM + C4NUM)[0] = (int8_t)d11;
        }
      }
    }
  }
#endif
}

void Conv3x3Int8OutputTransform(const int32_t *gemm_out, int8_t *out_data, const int32_t *bias_data, int start_index,
                                int real_cal_num, int out_w_block, ConvParameter *conv_param) {
  int output_channel = conv_param->output_channel_;
  int output_w = conv_param->output_w_;
  int output_h = conv_param->output_h_;
  const int oc4 = UP_DIV(output_channel, C4NUM);
  const int input_unit = 4;
  if (out_w_block == 0) {
    return;
  }
  for (int i = 0; i < real_cal_num; i++) {
    int out_w_index = (start_index + i) % out_w_block;
    int out_h_index = (start_index + i) / out_w_block;
    int src_tile_offset = i * oc4 * C4NUM * input_unit * input_unit;
    int dst_tile_offset = C4NUM * (out_w_index * OUPUT_UNIT + out_h_index * OUPUT_UNIT * output_w);

    for (int j = 0; j < oc4; j++) {
      int src_oc4_offset = src_tile_offset + j * input_unit * input_unit * C4NUM;
      int dst_oc4_offset = dst_tile_offset + j * C4NUM * output_h * output_w;
      const int32_t *src_ptr = gemm_out + src_oc4_offset;
      const int32_t *bias_ptr = bias_data + j * C4NUM;
      int8_t *dst_ptr = out_data + dst_oc4_offset;

      // output transform
      int real_num = (output_channel - j * C4NUM) < C4NUM ? (output_channel - j * C4NUM) : C4NUM;
      bool w_not_bound = out_w_index * OUPUT_UNIT + 1 < output_w;
      bool h_not_bound = out_h_index * OUPUT_UNIT + 1 < output_h;
      Conv3x3Int8OutputUnit(src_ptr, bias_ptr, dst_ptr, h_not_bound, w_not_bound, output_w, real_num, j * C4NUM,
                            conv_param);
    }
  }
}

void Conv3x3Int8InputTransform(const int16_t *input_data, int16_t *trans_input, int16_t *tmp_data, int start_index,
                               int real_cal_num, int out_w_block, ConvParameter *conv_param) {
  // input data format : nhwc
  int input_channel = conv_param->input_channel_;
  int input_width = conv_param->input_w_;
  int input_height = conv_param->input_h_;
  int pad_w = conv_param->pad_l_;
  int pad_h = conv_param->pad_u_;
  ConvQuantArg quant_arg = conv_param->conv_quant_arg_;
  int input_zp = quant_arg.input_quant_args_[0].zp_;
  const int ic8 = UP_DIV(input_channel, C8NUM);
  const int input_unit = 4;
  if (out_w_block == 0) {
    return;
  }
  for (int cal_id = 0; cal_id < real_cal_num; cal_id++) {
    int x_id = start_index + cal_id;
    int origin_x = (x_id % out_w_block) * OUPUT_UNIT - pad_w;
    int origin_y = (x_id / out_w_block) * OUPUT_UNIT - pad_h;
    int real_x_start = origin_x > 0 ? 0 : -origin_x;
    int real_x_end = (origin_x + input_unit) < input_width ? input_unit : (input_width - origin_x);
    int real_y_start = origin_y > 0 ? 0 : -origin_y;
    int real_y_end = (origin_y + input_unit) < input_height ? input_unit : (input_height - origin_y);

    int src_plane_offset = C8NUM * (origin_y * input_width + origin_x);
    int dst_plane_offset = cal_id * C8NUM;
    for (int ic = 0; ic < ic8; ic++) {
      // copy data from origin input to tmp buffer
      for (int i = 0; i < input_unit * input_unit * TILE_NUM; i++) tmp_data[i] = input_zp;

      int src_c8_offset = src_plane_offset + ic * C8NUM * input_height * input_width;
      for (int j = real_y_start; j < real_y_end; j++) {
        const int16_t *src = input_data + src_c8_offset + C8NUM * (j * input_width + real_x_start);
        int16_t *dst = tmp_data + C8NUM * (C4NUM * j + real_x_start);
        memcpy(dst, src, (real_x_end - real_x_start) * C8NUM * sizeof(int16_t));
      }
      // input transform
      int dst_ic8_offset = dst_plane_offset + ic * TILE_NUM * C8NUM;
      size_t dst_step = ic8 * C8NUM * TILE_NUM;
      int16_t *trans_input_ptr = trans_input + dst_ic8_offset;
      Conv3x3Int8InputUnit(tmp_data, trans_input_ptr, dst_step, input_zp);
    }
  }
}

void Conv3x3Int8Gemm(int32_t *dst, const int16_t *src, const int16_t *weight, int oc, int ic8, size_t real_cal_num) {
  int oc4 = UP_DIV(oc, C4NUM);
#ifdef ENABLE_ARM
  IndirectGemmInt16to32_8x4(dst, src, weight, 16, ic8, oc4, oc4 * 4 * 16 * sizeof(int32_t));
#else
  const int input_unit_square = 16;
  for (int c = 0; c < oc4; c++) {
    int filter_oc_offset = c * input_unit_square * ic8 * C8NUM * C4NUM;
    int dst_oc_offset = c * input_unit_square * C4NUM;
    for (int n = 0; n < real_cal_num; n++) {
      int src_tile_offset = n * C8NUM;
      int dst_tile_offset = dst_oc_offset + n * oc4 * C4NUM * input_unit_square;
      for (int i = 0; i < 4; i++) {
        int filter_h_offset = filter_oc_offset + i * 4 * ic8 * C8NUM * C4NUM;
        int src_h_offset = src_tile_offset + i * C8NUM * ic8 * C8NUM * C4NUM;
        int dst_h_offset = dst_tile_offset + i * 4 * 4;
        for (int m = 0; m < 4; m++) {
          int filter_w_offset = filter_h_offset + m * 4 * C8NUM * ic8;
          int src_w_offset = src_h_offset + m * 8 * ic8 * C8NUM;
          int dst_w_offset = dst_h_offset + m * C4NUM;

          int32_t acc[4] = {0};
          for (int z = 0; z < 4; z++) {
            int filter_offset = filter_w_offset + z;
            for (int j = 0; j < ic8; j++) {
              int filter_c8_offset = filter_offset + j * 4 * 8;
              int src_c8_offset = src_w_offset + j * 8 * 8;

              for (int k = 0; k < 8; k++) {
                const int16_t *w_ptr = weight + filter_c8_offset + k * 4;
                const int16_t *input_ptr = src + src_c8_offset + k;
                acc[z] += w_ptr[0] * input_ptr[0];
              }
            }
            (dst + dst_w_offset + z)[0] = acc[z];
          }
        }
      }
    }
  }
#endif
}

// int8 convolution 3x3
void Conv3x3Int8(int16_t *input_data, int16_t *transed_weight, const int32_t *bias_data, int8_t *output_data,
                 int16_t *tile_buffer, int16_t *block_unit_buffer, int32_t *tmp_dst_buffer, int8_t *tmp_out,
                 int task_id, ConvParameter *conv_param) {
  int ic8 = UP_DIV(conv_param->input_channel_, C8NUM);
  int out_w_block = UP_DIV(conv_param->output_w_, OUPUT_UNIT);
  int out_h_block = UP_DIV(conv_param->output_h_, OUPUT_UNIT);
  int output_count = out_w_block * out_h_block;
  int output_tile_count = UP_DIV(output_count, TILE_NUM);
  int oc4 = UP_DIV(conv_param->output_channel_, C4NUM);
  int tile_buffer_offset = TILE_NUM * 16 * ic8 * C8NUM;
  const int block_unit_buffer_offset = 16 * C8NUM;
  int tmp_dst_buffer_offset = TILE_NUM * 16 * oc4 * C4NUM;

  for (int batch = 0; batch < conv_param->input_batch_; batch++) {
    int in_batch_offset = batch * ic8 * C8NUM * conv_param->input_h_ * conv_param->input_w_;
    int tmp_out_batch_offset = batch * oc4 * C4NUM * conv_param->output_w_ * conv_param->output_h_;
    for (int thread_id = task_id; thread_id < output_tile_count; thread_id += conv_param->thread_num_) {
      int start_index = thread_id * TILE_NUM;
      int real_cal_num = (output_count - start_index) < TILE_NUM ? (output_count - start_index) : TILE_NUM;

      Conv3x3Int8InputTransform(input_data + in_batch_offset, tile_buffer + task_id * tile_buffer_offset,
                                block_unit_buffer + task_id * block_unit_buffer_offset, start_index, real_cal_num,
                                out_w_block, conv_param);

      Conv3x3Int8Gemm(tmp_dst_buffer + task_id * tmp_dst_buffer_offset, tile_buffer + task_id * tile_buffer_offset,
                      transed_weight, conv_param->output_channel_, ic8, real_cal_num);

      Conv3x3Int8OutputTransform(tmp_dst_buffer + task_id * tmp_dst_buffer_offset, tmp_out + tmp_out_batch_offset,
                                 bias_data, start_index, real_cal_num, out_w_block, conv_param);
    }
  }
}
