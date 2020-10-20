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

#include "nnacl/fp16/winograd_transform_fp16.h"

// for fp16 convolution 3x3 filter/input/output transform F(4,3)
void Conv3x3Fp16InputUnit(float16_t *tmp_data, float16_t *trans_input_data, size_t step) {
  float16x8_t d00 = vld1q_f16(tmp_data);
  float16x8_t d01 = vld1q_f16(tmp_data + 8);
  float16x8_t d02 = vld1q_f16(tmp_data + 2 * 8);
  float16x8_t d03 = vld1q_f16(tmp_data + 3 * 8);
  float16x8_t d04 = vld1q_f16(tmp_data + 4 * 8);
  float16x8_t d05 = vld1q_f16(tmp_data + 5 * 8);

  float16x8_t d10 = vld1q_f16(tmp_data + 6 * 8);
  float16x8_t d11 = vld1q_f16(tmp_data + 7 * 8);
  float16x8_t d12 = vld1q_f16(tmp_data + 8 * 8);
  float16x8_t d13 = vld1q_f16(tmp_data + 9 * 8);
  float16x8_t d14 = vld1q_f16(tmp_data + 10 * 8);
  float16x8_t d15 = vld1q_f16(tmp_data + 11 * 8);

  float16x8_t d20 = vld1q_f16(tmp_data + 12 * 8);
  float16x8_t d21 = vld1q_f16(tmp_data + 13 * 8);
  float16x8_t d22 = vld1q_f16(tmp_data + 14 * 8);
  float16x8_t d23 = vld1q_f16(tmp_data + 15 * 8);
  float16x8_t d24 = vld1q_f16(tmp_data + 16 * 8);
  float16x8_t d25 = vld1q_f16(tmp_data + 17 * 8);

  float16x8_t d30 = vld1q_f16(tmp_data + 18 * 8);
  float16x8_t d31 = vld1q_f16(tmp_data + 19 * 8);
  float16x8_t d32 = vld1q_f16(tmp_data + 20 * 8);
  float16x8_t d33 = vld1q_f16(tmp_data + 21 * 8);
  float16x8_t d34 = vld1q_f16(tmp_data + 22 * 8);
  float16x8_t d35 = vld1q_f16(tmp_data + 23 * 8);

  float16x8_t d40 = vld1q_f16(tmp_data + 24 * 8);
  float16x8_t d41 = vld1q_f16(tmp_data + 25 * 8);
  float16x8_t d42 = vld1q_f16(tmp_data + 26 * 8);
  float16x8_t d43 = vld1q_f16(tmp_data + 27 * 8);
  float16x8_t d44 = vld1q_f16(tmp_data + 28 * 8);
  float16x8_t d45 = vld1q_f16(tmp_data + 29 * 8);

  float16x8_t d50 = vld1q_f16(tmp_data + 30 * 8);
  float16x8_t d51 = vld1q_f16(tmp_data + 31 * 8);
  float16x8_t d52 = vld1q_f16(tmp_data + 32 * 8);
  float16x8_t d53 = vld1q_f16(tmp_data + 33 * 8);
  float16x8_t d54 = vld1q_f16(tmp_data + 34 * 8);
  float16x8_t d55 = vld1q_f16(tmp_data + 35 * 8);

  float16x8_t t00 = vaddq_f16(vsubq_f16(vmulq_n_f16(d00, 4), vmulq_n_f16(d20, 5)), d40);
  float16x8_t t01 = vaddq_f16(vsubq_f16(vmulq_n_f16(d01, 4), vmulq_n_f16(d21, 5)), d41);
  float16x8_t t02 = vaddq_f16(vsubq_f16(vmulq_n_f16(d02, 4), vmulq_n_f16(d22, 5)), d42);
  float16x8_t t03 = vaddq_f16(vsubq_f16(vmulq_n_f16(d03, 4), vmulq_n_f16(d23, 5)), d43);
  float16x8_t t04 = vaddq_f16(vsubq_f16(vmulq_n_f16(d04, 4), vmulq_n_f16(d24, 5)), d44);
  float16x8_t t05 = vaddq_f16(vsubq_f16(vmulq_n_f16(d05, 4), vmulq_n_f16(d25, 5)), d45);

  float16x8_t t10 = vaddq_f16(vaddq_f16(d30, d40), vmulq_n_f16(vaddq_f16(d10, d20), -4));
  float16x8_t t11 = vaddq_f16(vaddq_f16(d31, d41), vmulq_n_f16(vaddq_f16(d11, d21), -4));
  float16x8_t t12 = vaddq_f16(vaddq_f16(d32, d42), vmulq_n_f16(vaddq_f16(d12, d22), -4));
  float16x8_t t13 = vaddq_f16(vaddq_f16(d33, d43), vmulq_n_f16(vaddq_f16(d13, d23), -4));
  float16x8_t t14 = vaddq_f16(vaddq_f16(d34, d44), vmulq_n_f16(vaddq_f16(d14, d24), -4));
  float16x8_t t15 = vaddq_f16(vaddq_f16(d35, d45), vmulq_n_f16(vaddq_f16(d15, d25), -4));

  float16x8_t t20 = vaddq_f16(vsubq_f16(d40, d30), vmulq_n_f16(vsubq_f16(d10, d20), 4));
  float16x8_t t21 = vaddq_f16(vsubq_f16(d41, d31), vmulq_n_f16(vsubq_f16(d11, d21), 4));
  float16x8_t t22 = vaddq_f16(vsubq_f16(d42, d32), vmulq_n_f16(vsubq_f16(d12, d22), 4));
  float16x8_t t23 = vaddq_f16(vsubq_f16(d43, d33), vmulq_n_f16(vsubq_f16(d13, d23), 4));
  float16x8_t t24 = vaddq_f16(vsubq_f16(d44, d34), vmulq_n_f16(vsubq_f16(d14, d24), 4));
  float16x8_t t25 = vaddq_f16(vsubq_f16(d45, d35), vmulq_n_f16(vsubq_f16(d15, d25), 4));

  float16x8_t t30 = vaddq_f16(vsubq_f16(d40, d20), vmulq_n_f16(vsubq_f16(d30, d10), 2));
  float16x8_t t31 = vaddq_f16(vsubq_f16(d41, d21), vmulq_n_f16(vsubq_f16(d31, d11), 2));
  float16x8_t t32 = vaddq_f16(vsubq_f16(d42, d22), vmulq_n_f16(vsubq_f16(d32, d12), 2));
  float16x8_t t33 = vaddq_f16(vsubq_f16(d43, d23), vmulq_n_f16(vsubq_f16(d33, d13), 2));
  float16x8_t t34 = vaddq_f16(vsubq_f16(d44, d24), vmulq_n_f16(vsubq_f16(d34, d14), 2));
  float16x8_t t35 = vaddq_f16(vsubq_f16(d45, d25), vmulq_n_f16(vsubq_f16(d35, d15), 2));

  float16x8_t t40 = vaddq_f16(vsubq_f16(d40, d20), vmulq_n_f16(vsubq_f16(d10, d30), 2));
  float16x8_t t41 = vaddq_f16(vsubq_f16(d41, d21), vmulq_n_f16(vsubq_f16(d11, d31), 2));
  float16x8_t t42 = vaddq_f16(vsubq_f16(d42, d22), vmulq_n_f16(vsubq_f16(d12, d32), 2));
  float16x8_t t43 = vaddq_f16(vsubq_f16(d43, d23), vmulq_n_f16(vsubq_f16(d13, d33), 2));
  float16x8_t t44 = vaddq_f16(vsubq_f16(d44, d24), vmulq_n_f16(vsubq_f16(d14, d34), 2));
  float16x8_t t45 = vaddq_f16(vsubq_f16(d45, d25), vmulq_n_f16(vsubq_f16(d15, d35), 2));

  float16x8_t t50 = vaddq_f16(vsubq_f16(vmulq_n_f16(d10, 4), vmulq_n_f16(d30, 5)), d50);
  float16x8_t t51 = vaddq_f16(vsubq_f16(vmulq_n_f16(d11, 4), vmulq_n_f16(d31, 5)), d51);
  float16x8_t t52 = vaddq_f16(vsubq_f16(vmulq_n_f16(d12, 4), vmulq_n_f16(d32, 5)), d52);
  float16x8_t t53 = vaddq_f16(vsubq_f16(vmulq_n_f16(d13, 4), vmulq_n_f16(d33, 5)), d53);
  float16x8_t t54 = vaddq_f16(vsubq_f16(vmulq_n_f16(d14, 4), vmulq_n_f16(d34, 5)), d54);
  float16x8_t t55 = vaddq_f16(vsubq_f16(vmulq_n_f16(d15, 4), vmulq_n_f16(d35, 5)), d55);

  float16x8_t m00 = vaddq_f16(vsubq_f16(vmulq_n_f16(t00, 4), vmulq_n_f16(t02, 5)), t04);
  float16x8_t m01 = vaddq_f16(vaddq_f16(t03, t04), vmulq_n_f16(vaddq_f16(t01, t02), -4));
  float16x8_t m02 = vaddq_f16(vsubq_f16(t04, t03), vmulq_n_f16(vsubq_f16(t01, t02), 4));
  float16x8_t m03 = vaddq_f16(vsubq_f16(t04, t02), vmulq_n_f16(vsubq_f16(t03, t01), 2));
  float16x8_t m04 = vaddq_f16(vsubq_f16(t04, t02), vmulq_n_f16(vsubq_f16(t01, t03), 2));
  float16x8_t m05 = vaddq_f16(vsubq_f16(vmulq_n_f16(t01, 4), vmulq_n_f16(t03, 5)), t05);

  float16x8_t m10 = vaddq_f16(vsubq_f16(vmulq_n_f16(t10, 4), vmulq_n_f16(t12, 5)), t14);
  float16x8_t m11 = vaddq_f16(vaddq_f16(t13, t14), vmulq_n_f16(vaddq_f16(t11, t12), -4));
  float16x8_t m12 = vaddq_f16(vsubq_f16(t14, t13), vmulq_n_f16(vsubq_f16(t11, t12), 4));
  float16x8_t m13 = vaddq_f16(vsubq_f16(t14, t12), vmulq_n_f16(vsubq_f16(t13, t11), 2));
  float16x8_t m14 = vaddq_f16(vsubq_f16(t14, t12), vmulq_n_f16(vsubq_f16(t11, t13), 2));
  float16x8_t m15 = vaddq_f16(vsubq_f16(vmulq_n_f16(t11, 4), vmulq_n_f16(t13, 5)), t15);

  float16x8_t m20 = vaddq_f16(vsubq_f16(vmulq_n_f16(t20, 4), vmulq_n_f16(t22, 5)), t24);
  float16x8_t m21 = vaddq_f16(vaddq_f16(t23, t24), vmulq_n_f16(vaddq_f16(t21, t22), -4));
  float16x8_t m22 = vaddq_f16(vsubq_f16(t24, t23), vmulq_n_f16(vsubq_f16(t21, t22), 4));
  float16x8_t m23 = vaddq_f16(vsubq_f16(t24, t22), vmulq_n_f16(vsubq_f16(t23, t21), 2));
  float16x8_t m24 = vaddq_f16(vsubq_f16(t24, t22), vmulq_n_f16(vsubq_f16(t21, t23), 2));
  float16x8_t m25 = vaddq_f16(vsubq_f16(vmulq_n_f16(t21, 4), vmulq_n_f16(t23, 5)), t25);

  float16x8_t m30 = vaddq_f16(vsubq_f16(vmulq_n_f16(t30, 4), vmulq_n_f16(t32, 5)), t34);
  float16x8_t m31 = vaddq_f16(vaddq_f16(t33, t34), vmulq_n_f16(vaddq_f16(t31, t32), -4));
  float16x8_t m32 = vaddq_f16(vsubq_f16(t34, t33), vmulq_n_f16(vsubq_f16(t31, t32), 4));
  float16x8_t m33 = vaddq_f16(vsubq_f16(t34, t32), vmulq_n_f16(vsubq_f16(t33, t31), 2));
  float16x8_t m34 = vaddq_f16(vsubq_f16(t34, t32), vmulq_n_f16(vsubq_f16(t31, t33), 2));
  float16x8_t m35 = vaddq_f16(vsubq_f16(vmulq_n_f16(t31, 4), vmulq_n_f16(t33, 5)), t35);

  float16x8_t m40 = vaddq_f16(vsubq_f16(vmulq_n_f16(t40, 4), vmulq_n_f16(t42, 5)), t44);
  float16x8_t m41 = vaddq_f16(vaddq_f16(t43, t44), vmulq_n_f16(vaddq_f16(t41, t42), -4));
  float16x8_t m42 = vaddq_f16(vsubq_f16(t44, t43), vmulq_n_f16(vsubq_f16(t41, t42), 4));
  float16x8_t m43 = vaddq_f16(vsubq_f16(t44, t42), vmulq_n_f16(vsubq_f16(t43, t41), 2));
  float16x8_t m44 = vaddq_f16(vsubq_f16(t44, t42), vmulq_n_f16(vsubq_f16(t41, t43), 2));
  float16x8_t m45 = vaddq_f16(vsubq_f16(vmulq_n_f16(t41, 4), vmulq_n_f16(t43, 5)), t45);

  float16x8_t m50 = vaddq_f16(vsubq_f16(vmulq_n_f16(t50, 4), vmulq_n_f16(t52, 5)), t54);
  float16x8_t m51 = vaddq_f16(vaddq_f16(t53, t54), vmulq_n_f16(vaddq_f16(t51, t52), -4));
  float16x8_t m52 = vaddq_f16(vsubq_f16(t54, t53), vmulq_n_f16(vsubq_f16(t51, t52), 4));
  float16x8_t m53 = vaddq_f16(vsubq_f16(t54, t52), vmulq_n_f16(vsubq_f16(t53, t51), 2));
  float16x8_t m54 = vaddq_f16(vsubq_f16(t54, t52), vmulq_n_f16(vsubq_f16(t51, t53), 2));
  float16x8_t m55 = vaddq_f16(vsubq_f16(vmulq_n_f16(t51, 4), vmulq_n_f16(t53, 5)), t55);

  vst1_f16(trans_input_data, vget_low_f16(m00));
  vst1_f16(trans_input_data + 64, vget_high_f16(m00));
  vst1_f16(trans_input_data + step, vget_low_f16(m01));
  vst1_f16(trans_input_data + step + 64, vget_high_f16(m01));
  vst1_f16(trans_input_data + 2 * step, vget_low_f16(m02));
  vst1_f16(trans_input_data + 2 * step + 64, vget_high_f16(m02));
  vst1_f16(trans_input_data + 3 * step, vget_low_f16(m03));
  vst1_f16(trans_input_data + 3 * step + 64, vget_high_f16(m03));
  vst1_f16(trans_input_data + 4 * step, vget_low_f16(m04));
  vst1_f16(trans_input_data + 4 * step + 64, vget_high_f16(m04));
  vst1_f16(trans_input_data + 5 * step, vget_low_f16(m05));
  vst1_f16(trans_input_data + 5 * step + 64, vget_high_f16(m05));

  vst1_f16(trans_input_data + 6 * step, vget_low_f16(m10));
  vst1_f16(trans_input_data + 6 * step + 64, vget_high_f16(m10));
  vst1_f16(trans_input_data + 7 * step, vget_low_f16(m11));
  vst1_f16(trans_input_data + 7 * step + 64, vget_high_f16(m11));
  vst1_f16(trans_input_data + 8 * step, vget_low_f16(m12));
  vst1_f16(trans_input_data + 8 * step + 64, vget_high_f16(m12));
  vst1_f16(trans_input_data + 9 * step, vget_low_f16(m13));
  vst1_f16(trans_input_data + 9 * step + 64, vget_high_f16(m13));
  vst1_f16(trans_input_data + 10 * step, vget_low_f16(m14));
  vst1_f16(trans_input_data + 10 * step + 64, vget_high_f16(m14));
  vst1_f16(trans_input_data + 11 * step, vget_low_f16(m15));
  vst1_f16(trans_input_data + 11 * step + 64, vget_high_f16(m15));

  vst1_f16(trans_input_data + 12 * step, vget_low_f16(m20));
  vst1_f16(trans_input_data + 12 * step + 64, vget_high_f16(m20));
  vst1_f16(trans_input_data + 13 * step, vget_low_f16(m21));
  vst1_f16(trans_input_data + 13 * step + 64, vget_high_f16(m21));
  vst1_f16(trans_input_data + 14 * step, vget_low_f16(m22));
  vst1_f16(trans_input_data + 14 * step + 64, vget_high_f16(m22));
  vst1_f16(trans_input_data + 15 * step, vget_low_f16(m23));
  vst1_f16(trans_input_data + 15 * step + 64, vget_high_f16(m23));
  vst1_f16(trans_input_data + 16 * step, vget_low_f16(m24));
  vst1_f16(trans_input_data + 16 * step + 64, vget_high_f16(m24));
  vst1_f16(trans_input_data + 17 * step, vget_low_f16(m25));
  vst1_f16(trans_input_data + 17 * step + 64, vget_high_f16(m25));

  vst1_f16(trans_input_data + 18 * step, vget_low_f16(m30));
  vst1_f16(trans_input_data + 18 * step + 64, vget_high_f16(m30));
  vst1_f16(trans_input_data + 19 * step, vget_low_f16(m31));
  vst1_f16(trans_input_data + 19 * step + 64, vget_high_f16(m31));
  vst1_f16(trans_input_data + 20 * step, vget_low_f16(m32));
  vst1_f16(trans_input_data + 20 * step + 64, vget_high_f16(m32));
  vst1_f16(trans_input_data + 21 * step, vget_low_f16(m33));
  vst1_f16(trans_input_data + 21 * step + 64, vget_high_f16(m33));
  vst1_f16(trans_input_data + 22 * step, vget_low_f16(m34));
  vst1_f16(trans_input_data + 22 * step + 64, vget_high_f16(m34));
  vst1_f16(trans_input_data + 23 * step, vget_low_f16(m35));
  vst1_f16(trans_input_data + 23 * step + 64, vget_high_f16(m35));

  vst1_f16(trans_input_data + 24 * step, vget_low_f16(m40));
  vst1_f16(trans_input_data + 24 * step + 64, vget_high_f16(m40));
  vst1_f16(trans_input_data + 25 * step, vget_low_f16(m41));
  vst1_f16(trans_input_data + 25 * step + 64, vget_high_f16(m41));
  vst1_f16(trans_input_data + 26 * step, vget_low_f16(m42));
  vst1_f16(trans_input_data + 26 * step + 64, vget_high_f16(m42));
  vst1_f16(trans_input_data + 27 * step, vget_low_f16(m43));
  vst1_f16(trans_input_data + 27 * step + 64, vget_high_f16(m43));
  vst1_f16(trans_input_data + 28 * step, vget_low_f16(m44));
  vst1_f16(trans_input_data + 28 * step + 64, vget_high_f16(m44));
  vst1_f16(trans_input_data + 29 * step, vget_low_f16(m45));
  vst1_f16(trans_input_data + 29 * step + 64, vget_high_f16(m45));

  vst1_f16(trans_input_data + 30 * step, vget_low_f16(m50));
  vst1_f16(trans_input_data + 30 * step + 64, vget_high_f16(m50));
  vst1_f16(trans_input_data + 31 * step, vget_low_f16(m51));
  vst1_f16(trans_input_data + 31 * step + 64, vget_high_f16(m51));
  vst1_f16(trans_input_data + 32 * step, vget_low_f16(m52));
  vst1_f16(trans_input_data + 32 * step + 64, vget_high_f16(m52));
  vst1_f16(trans_input_data + 33 * step, vget_low_f16(m53));
  vst1_f16(trans_input_data + 33 * step + 64, vget_high_f16(m53));
  vst1_f16(trans_input_data + 34 * step, vget_low_f16(m54));
  vst1_f16(trans_input_data + 34 * step + 64, vget_high_f16(m54));
  vst1_f16(trans_input_data + 35 * step, vget_low_f16(m55));
  vst1_f16(trans_input_data + 35 * step + 64, vget_high_f16(m55));
}

void Conv3x3Fp16InputTransform(const float16_t *input_data, float16_t *trans_input, float16_t *tmp_data,
                               int start_index, int real_cal_num, int out_w_block, ConvParameter *conv_param) {
  // input data format : nhwc
  const int output_unit = 4;
  int input_channel = conv_param->input_channel_;
  int input_width = conv_param->input_w_;
  int input_height = conv_param->input_h_;
  int pad_w = conv_param->pad_l_;
  int pad_h = conv_param->pad_u_;
  int ic8 = UP_DIV(input_channel, C8NUM);
  if (out_w_block == 0) {
    return;
  }
  for (int cal_id = 0; cal_id < real_cal_num; cal_id++) {
    int x_id = start_index + cal_id;
    int origin_x = (x_id % out_w_block) * output_unit - pad_w;
    int origin_y = (x_id / out_w_block) * output_unit - pad_h;
    int real_x_start = origin_x > 0 ? 0 : -origin_x;
    int real_x_end = (origin_x + 6) < input_width ? 6 : (input_width - origin_x);
    int real_y_start = origin_y > 0 ? 0 : -origin_y;
    int real_y_end = (origin_y + 6) < input_height ? 6 : (input_height - origin_y);

    int src_plane_offset = ic8 * C8NUM * (origin_y * input_width + origin_x);
    int dst_plane_offset = cal_id * C4NUM;
    for (int ic = 0; ic < ic8; ic++) {
      // clear tmp buffer
      memset(tmp_data, 0, 6 * 6 * C8NUM * sizeof(float16_t));

      // get real input block with padding
      int src_ic4_offset = src_plane_offset + ic * C8NUM;
      for (int interval = real_y_start; interval < real_y_end; interval++) {
        int src_y_offset = src_ic4_offset + (interval * input_width + real_x_start) * ic8 * C8NUM;
        int dst_y_offset = interval * 6 * C8NUM + real_x_start * C8NUM;
        for (int j = 0; j < (real_x_end - real_x_start); j++) {
          int src_x_offset = src_y_offset + j * ic8 * C8NUM;
          int dst_x_offset = dst_y_offset + j * C8NUM;
          float16_t *src_addr = (float16_t *)(input_data) + src_x_offset;
          float16_t *dst_addr = tmp_data + dst_x_offset;
          vst1q_f16(dst_addr, vld1q_f16(src_addr));
        }
      }

      // input transform
      int dst_ic4_offset = dst_plane_offset + ic * 16 * C8NUM;
      size_t dst_step = ic8 * C8NUM * 16;
      float16_t *trans_input_ptr = trans_input + dst_ic4_offset;
      Conv3x3Fp16InputUnit(tmp_data, trans_input_ptr, dst_step);
    }
  }
}

void Conv3x3Fp16FilterTransform(const float16_t *weight_data, float16_t *trans_weight, int iC4, int output_channel,
                                int kernel_plane) {
  int dst_step = iC4 * C4NUM * 8;
  for (int o = 0; o < output_channel; o++) {
    int oc8_block_num = o / C8NUM;
    int oc8_block_rem = o % C8NUM;
    int src_oc_offset = o * iC4 * C4NUM * kernel_plane;
    int dst_oc_offset = oc8_block_num * C8NUM * iC4 * C4NUM * 36 + oc8_block_rem;
    for (int i = 0; i < iC4; i++) {
      const float16_t *src_ic4_ptr = weight_data + src_oc_offset + i * kernel_plane * C4NUM;
      float16_t *dst_ic4_ptr = trans_weight + dst_oc_offset + i * 8 * C4NUM;
      float16x4_t g00 = vld1_f16(src_ic4_ptr);
      float16x4_t g01 = vld1_f16(src_ic4_ptr + 4);
      float16x4_t g02 = vld1_f16(src_ic4_ptr + 2 * 4);
      float16x4_t g10 = vld1_f16(src_ic4_ptr + 3 * 4);
      float16x4_t g11 = vld1_f16(src_ic4_ptr + 4 * 4);
      float16x4_t g12 = vld1_f16(src_ic4_ptr + 5 * 4);
      float16x4_t g20 = vld1_f16(src_ic4_ptr + 6 * 4);
      float16x4_t g21 = vld1_f16(src_ic4_ptr + 7 * 4);
      float16x4_t g22 = vld1_f16(src_ic4_ptr + 8 * 4);

      float16x4_t dst00 = vmul_n_f16(g00, 0.25);
      float16x4_t dst01 = vmul_n_f16(g01, 0.25);
      float16x4_t dst02 = vmul_n_f16(g02, 0.25);

      float16x4_t dst10 = vmul_n_f16(vadd_f16(g00, vadd_f16(g10, g20)), -0.1666666666667);
      float16x4_t dst11 = vmul_n_f16(vadd_f16(g01, vadd_f16(g11, g21)), -0.1666666666667);
      float16x4_t dst12 = vmul_n_f16(vadd_f16(g02, vadd_f16(g12, g22)), -0.1666666666667);

      float16x4_t dst20 = vmul_n_f16(vsub_f16(vadd_f16(g00, g20), g10), -0.1666666666667);
      float16x4_t dst21 = vmul_n_f16(vsub_f16(vadd_f16(g01, g21), g11), -0.1666666666667);
      float16x4_t dst22 = vmul_n_f16(vsub_f16(vadd_f16(g02, g22), g12), -0.1666666666667);

      float16x4_t dst30 = vadd_f16(vmul_n_f16(g10, 0.08333333333333),
                                   vadd_f16(vmul_n_f16(g00, 0.04166666666667), vmul_n_f16(g20, 0.1666666666667)));
      float16x4_t dst31 = vadd_f16(vmul_n_f16(g11, 0.08333333333333),
                                   vadd_f16(vmul_n_f16(g01, 0.04166666666667), vmul_n_f16(g21, 0.1666666666667)));
      float16x4_t dst32 = vadd_f16(vmul_n_f16(g12, 0.08333333333333),
                                   vadd_f16(vmul_n_f16(g02, 0.04166666666667), vmul_n_f16(g22, 0.1666666666667)));

      float16x4_t dst40 = vsub_f16(vadd_f16(vmul_n_f16(g00, 0.04166666666667), vmul_n_f16(g20, 0.1666666666667)),
                                   vmul_n_f16(g10, 0.08333333333333));
      float16x4_t dst41 = vsub_f16(vadd_f16(vmul_n_f16(g01, 0.04166666666667), vmul_n_f16(g21, 0.1666666666667)),
                                   vmul_n_f16(g11, 0.08333333333333));
      float16x4_t dst42 = vsub_f16(vadd_f16(vmul_n_f16(g02, 0.04166666666667), vmul_n_f16(g22, 0.1666666666667)),
                                   vmul_n_f16(g12, 0.08333333333333));

      float16x4_t dst50 = g20;
      float16x4_t dst51 = g21;
      float16x4_t dst52 = g22;

      float16x4_t m00 = vmul_n_f16(dst00, 0.25);
      float16x4_t m01 = vmul_n_f16(vadd_f16(dst00, vadd_f16(dst01, dst02)), -0.1666666666667);
      float16x4_t m02 = vmul_n_f16(vsub_f16(vadd_f16(dst00, dst02), dst01), -0.1666666666667);
      float16x4_t m03 = vadd_f16(vmul_n_f16(dst01, 0.08333333333333),
                                 vadd_f16(vmul_n_f16(dst00, 0.04166666666667), vmul_n_f16(dst02, 0.1666666666667)));
      float16x4_t m04 = vsub_f16(vadd_f16(vmul_n_f16(dst00, 0.04166666666667), vmul_n_f16(dst02, 0.1666666666667)),
                                 vmul_n_f16(dst01, 0.08333333333333));
      float16x4_t m05 = dst02;

      float16x4_t m10 = vmul_n_f16(dst10, 0.25);
      float16x4_t m11 = vmul_n_f16(vadd_f16(dst10, vadd_f16(dst11, dst12)), -0.1666666666667);
      float16x4_t m12 = vmul_n_f16(vsub_f16(vadd_f16(dst10, dst12), dst11), -0.1666666666667);
      float16x4_t m13 = vadd_f16(vmul_n_f16(dst11, 0.08333333333333),
                                 vadd_f16(vmul_n_f16(dst10, 0.04166666666667), vmul_n_f16(dst12, 0.1666666666667)));
      float16x4_t m14 = vsub_f16(vadd_f16(vmul_n_f16(dst10, 0.04166666666667), vmul_n_f16(dst12, 0.1666666666667)),
                                 vmul_n_f16(dst11, 0.08333333333333));
      float16x4_t m15 = dst12;

      float16x4_t m20 = vmul_n_f16(dst20, 0.25);
      float16x4_t m21 = vmul_n_f16(vadd_f16(dst20, vadd_f16(dst21, dst22)), -0.1666666666667);
      float16x4_t m22 = vmul_n_f16(vsub_f16(vadd_f16(dst20, dst22), dst21), -0.1666666666667);
      float16x4_t m23 = vadd_f16(vmul_n_f16(dst21, 0.08333333333333),
                                 vadd_f16(vmul_n_f16(dst20, 0.04166666666667), vmul_n_f16(dst22, 0.1666666666667)));
      float16x4_t m24 = vsub_f16(vadd_f16(vmul_n_f16(dst20, 0.04166666666667), vmul_n_f16(dst22, 0.1666666666667)),
                                 vmul_n_f16(dst21, 0.08333333333333));
      float16x4_t m25 = dst22;

      float16x4_t m30 = vmul_n_f16(dst30, 0.25);
      float16x4_t m31 = vmul_n_f16(vadd_f16(dst30, vadd_f16(dst31, dst32)), -0.1666666666667);
      float16x4_t m32 = vmul_n_f16(vsub_f16(vadd_f16(dst30, dst32), dst31), -0.1666666666667);
      float16x4_t m33 = vadd_f16(vmul_n_f16(dst31, 0.08333333333333),
                                 vadd_f16(vmul_n_f16(dst30, 0.04166666666667), vmul_n_f16(dst32, 0.1666666666667)));
      float16x4_t m34 = vsub_f16(vadd_f16(vmul_n_f16(dst30, 0.04166666666667), vmul_n_f16(dst32, 0.1666666666667)),
                                 vmul_n_f16(dst31, 0.08333333333333));
      float16x4_t m35 = dst32;

      float16x4_t m40 = vmul_n_f16(dst40, 0.25);
      float16x4_t m41 = vmul_n_f16(vadd_f16(dst40, vadd_f16(dst41, dst42)), -0.1666666666667);
      float16x4_t m42 = vmul_n_f16(vsub_f16(vadd_f16(dst40, dst42), dst41), -0.1666666666667);
      float16x4_t m43 = vadd_f16(vmul_n_f16(dst41, 0.08333333333333),
                                 vadd_f16(vmul_n_f16(dst40, 0.04166666666667), vmul_n_f16(dst42, 0.1666666666667)));
      float16x4_t m44 = vsub_f16(vadd_f16(vmul_n_f16(dst40, 0.04166666666667), vmul_n_f16(dst42, 0.1666666666667)),
                                 vmul_n_f16(dst41, 0.08333333333333));
      float16x4_t m45 = dst42;

      float16x4_t m50 = vmul_n_f16(dst50, 0.25);
      float16x4_t m51 = vmul_n_f16(vadd_f16(dst50, vadd_f16(dst51, dst52)), -0.1666666666667);
      float16x4_t m52 = vmul_n_f16(vsub_f16(vadd_f16(dst50, dst52), dst51), -0.1666666666667);
      float16x4_t m53 = vadd_f16(vmul_n_f16(dst51, 0.08333333333333),
                                 vadd_f16(vmul_n_f16(dst50, 0.04166666666667), vmul_n_f16(dst52, 0.1666666666667)));
      float16x4_t m54 = vsub_f16(vadd_f16(vmul_n_f16(dst50, 0.04166666666667), vmul_n_f16(dst52, 0.1666666666667)),
                                 vmul_n_f16(dst51, 0.08333333333333));
      float16x4_t m55 = dst52;

      for (int j = 0; j < 4; j++) {
        dst_ic4_ptr[j * 8] = m00[j];
        dst_ic4_ptr[j * 8 + dst_step] = m01[j];
        dst_ic4_ptr[j * 8 + 2 * dst_step] = m02[j];
        dst_ic4_ptr[j * 8 + 3 * dst_step] = m03[j];
        dst_ic4_ptr[j * 8 + 4 * dst_step] = m04[j];
        dst_ic4_ptr[j * 8 + 5 * dst_step] = m05[j];
        dst_ic4_ptr[j * 8 + 6 * dst_step] = m10[j];
        dst_ic4_ptr[j * 8 + 7 * dst_step] = m11[j];
        dst_ic4_ptr[j * 8 + 8 * dst_step] = m12[j];
        dst_ic4_ptr[j * 8 + 9 * dst_step] = m13[j];
        dst_ic4_ptr[j * 8 + 10 * dst_step] = m14[j];
        dst_ic4_ptr[j * 8 + 11 * dst_step] = m15[j];
        dst_ic4_ptr[j * 8 + 12 * dst_step] = m20[j];
        dst_ic4_ptr[j * 8 + 13 * dst_step] = m21[j];
        dst_ic4_ptr[j * 8 + 14 * dst_step] = m22[j];
        dst_ic4_ptr[j * 8 + 15 * dst_step] = m23[j];
        dst_ic4_ptr[j * 8 + 16 * dst_step] = m24[j];
        dst_ic4_ptr[j * 8 + 17 * dst_step] = m25[j];
        dst_ic4_ptr[j * 8 + 18 * dst_step] = m30[j];
        dst_ic4_ptr[j * 8 + 19 * dst_step] = m31[j];
        dst_ic4_ptr[j * 8 + 20 * dst_step] = m32[j];
        dst_ic4_ptr[j * 8 + 21 * dst_step] = m33[j];
        dst_ic4_ptr[j * 8 + 22 * dst_step] = m34[j];
        dst_ic4_ptr[j * 8 + 23 * dst_step] = m35[j];
        dst_ic4_ptr[j * 8 + 24 * dst_step] = m40[j];
        dst_ic4_ptr[j * 8 + 25 * dst_step] = m41[j];
        dst_ic4_ptr[j * 8 + 26 * dst_step] = m42[j];
        dst_ic4_ptr[j * 8 + 27 * dst_step] = m43[j];
        dst_ic4_ptr[j * 8 + 28 * dst_step] = m44[j];
        dst_ic4_ptr[j * 8 + 29 * dst_step] = m45[j];
        dst_ic4_ptr[j * 8 + 30 * dst_step] = m50[j];
        dst_ic4_ptr[j * 8 + 31 * dst_step] = m51[j];
        dst_ic4_ptr[j * 8 + 32 * dst_step] = m52[j];
        dst_ic4_ptr[j * 8 + 33 * dst_step] = m53[j];
        dst_ic4_ptr[j * 8 + 34 * dst_step] = m54[j];
        dst_ic4_ptr[j * 8 + 35 * dst_step] = m55[j];
      }
    }
  }
}

void Conv3x3Fp16OutputUnit(const float16_t *gemm_out, const float16_t *bias_data, float16_t *output_data,
                           int output_w) {
  float16x8_t s00 = vld1q_f16(gemm_out);
  float16x8_t s01 = vld1q_f16(gemm_out + 8);
  float16x8_t s02 = vld1q_f16(gemm_out + 16);
  float16x8_t s03 = vld1q_f16(gemm_out + 24);
  float16x8_t s04 = vld1q_f16(gemm_out + 32);
  float16x8_t s05 = vld1q_f16(gemm_out + 40);

  float16x8_t s10 = vld1q_f16(gemm_out + 48);
  float16x8_t s11 = vld1q_f16(gemm_out + 56);
  float16x8_t s12 = vld1q_f16(gemm_out + 64);
  float16x8_t s13 = vld1q_f16(gemm_out + 72);
  float16x8_t s14 = vld1q_f16(gemm_out + 80);
  float16x8_t s15 = vld1q_f16(gemm_out + 88);

  float16x8_t s20 = vld1q_f16(gemm_out + 96);
  float16x8_t s21 = vld1q_f16(gemm_out + 104);
  float16x8_t s22 = vld1q_f16(gemm_out + 112);
  float16x8_t s23 = vld1q_f16(gemm_out + 120);
  float16x8_t s24 = vld1q_f16(gemm_out + 128);
  float16x8_t s25 = vld1q_f16(gemm_out + 136);

  float16x8_t s30 = vld1q_f16(gemm_out + 144);
  float16x8_t s31 = vld1q_f16(gemm_out + 152);
  float16x8_t s32 = vld1q_f16(gemm_out + 160);
  float16x8_t s33 = vld1q_f16(gemm_out + 168);
  float16x8_t s34 = vld1q_f16(gemm_out + 176);
  float16x8_t s35 = vld1q_f16(gemm_out + 184);

  float16x8_t s40 = vld1q_f16(gemm_out + 192);
  float16x8_t s41 = vld1q_f16(gemm_out + 200);
  float16x8_t s42 = vld1q_f16(gemm_out + 208);
  float16x8_t s43 = vld1q_f16(gemm_out + 216);
  float16x8_t s44 = vld1q_f16(gemm_out + 224);
  float16x8_t s45 = vld1q_f16(gemm_out + 232);

  float16x8_t s50 = vld1q_f16(gemm_out + 240);
  float16x8_t s51 = vld1q_f16(gemm_out + 248);
  float16x8_t s52 = vld1q_f16(gemm_out + 256);
  float16x8_t s53 = vld1q_f16(gemm_out + 264);
  float16x8_t s54 = vld1q_f16(gemm_out + 272);
  float16x8_t s55 = vld1q_f16(gemm_out + 280);

  float16x8_t t00 = vaddq_f16(vaddq_f16(vaddq_f16(s00, s10), vaddq_f16(s20, s30)), s40);
  float16x8_t t01 = vaddq_f16(vaddq_f16(vaddq_f16(s01, s11), vaddq_f16(s21, s31)), s41);
  float16x8_t t02 = vaddq_f16(vaddq_f16(vaddq_f16(s02, s12), vaddq_f16(s22, s32)), s42);
  float16x8_t t03 = vaddq_f16(vaddq_f16(vaddq_f16(s03, s13), vaddq_f16(s23, s33)), s43);
  float16x8_t t04 = vaddq_f16(vaddq_f16(vaddq_f16(s04, s14), vaddq_f16(s24, s34)), s44);
  float16x8_t t05 = vaddq_f16(vaddq_f16(vaddq_f16(s05, s15), vaddq_f16(s25, s35)), s45);

  float16x8_t t10 = vaddq_f16(vsubq_f16(s10, s20), vmulq_n_f16(vsubq_f16(s30, s40), 2));
  float16x8_t t11 = vaddq_f16(vsubq_f16(s11, s21), vmulq_n_f16(vsubq_f16(s31, s41), 2));
  float16x8_t t12 = vaddq_f16(vsubq_f16(s12, s22), vmulq_n_f16(vsubq_f16(s32, s42), 2));
  float16x8_t t13 = vaddq_f16(vsubq_f16(s13, s23), vmulq_n_f16(vsubq_f16(s33, s43), 2));
  float16x8_t t14 = vaddq_f16(vsubq_f16(s14, s24), vmulq_n_f16(vsubq_f16(s34, s44), 2));
  float16x8_t t15 = vaddq_f16(vsubq_f16(s15, s25), vmulq_n_f16(vsubq_f16(s35, s45), 2));

  float16x8_t t20 = vaddq_f16(vaddq_f16(s10, s20), vmulq_n_f16(vaddq_f16(s30, s40), 4));
  float16x8_t t21 = vaddq_f16(vaddq_f16(s11, s21), vmulq_n_f16(vaddq_f16(s31, s41), 4));
  float16x8_t t22 = vaddq_f16(vaddq_f16(s12, s22), vmulq_n_f16(vaddq_f16(s32, s42), 4));
  float16x8_t t23 = vaddq_f16(vaddq_f16(s13, s23), vmulq_n_f16(vaddq_f16(s33, s43), 4));
  float16x8_t t24 = vaddq_f16(vaddq_f16(s14, s24), vmulq_n_f16(vaddq_f16(s34, s44), 4));
  float16x8_t t25 = vaddq_f16(vaddq_f16(s15, s25), vmulq_n_f16(vaddq_f16(s35, s45), 4));

  float16x8_t t30 = vaddq_f16(vaddq_f16(vsubq_f16(s10, s20), vmulq_n_f16(vsubq_f16(s30, s40), 8)), s50);
  float16x8_t t31 = vaddq_f16(vaddq_f16(vsubq_f16(s11, s21), vmulq_n_f16(vsubq_f16(s31, s41), 8)), s51);
  float16x8_t t32 = vaddq_f16(vaddq_f16(vsubq_f16(s12, s22), vmulq_n_f16(vsubq_f16(s32, s42), 8)), s52);
  float16x8_t t33 = vaddq_f16(vaddq_f16(vsubq_f16(s13, s23), vmulq_n_f16(vsubq_f16(s33, s43), 8)), s53);
  float16x8_t t34 = vaddq_f16(vaddq_f16(vsubq_f16(s14, s24), vmulq_n_f16(vsubq_f16(s34, s44), 8)), s54);
  float16x8_t t35 = vaddq_f16(vaddq_f16(vsubq_f16(s15, s25), vmulq_n_f16(vsubq_f16(s35, s45), 8)), s55);

  float16x8_t bias_ptr = vld1q_f16(bias_data);
  float16x8_t d00 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t00, t01), vaddq_f16(t02, t03)), t04), bias_ptr);
  float16x8_t d01 = vaddq_f16(vaddq_f16(vsubq_f16(t01, t02), vmulq_n_f16(vsubq_f16(t03, t04), 2)), bias_ptr);
  float16x8_t d02 = vaddq_f16(vaddq_f16(vaddq_f16(t01, t02), vmulq_n_f16(vaddq_f16(t03, t04), 4)), bias_ptr);
  float16x8_t d03 =
    vaddq_f16(vaddq_f16(vaddq_f16(vsubq_f16(t01, t02), vmulq_n_f16(vsubq_f16(t03, t04), 8)), t05), bias_ptr);

  float16x8_t d10 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t10, t11), vaddq_f16(t12, t13)), t14), bias_ptr);
  float16x8_t d11 = vaddq_f16(vaddq_f16(vsubq_f16(t11, t12), vmulq_n_f16(vsubq_f16(t13, t14), 2)), bias_ptr);
  float16x8_t d12 = vaddq_f16(vaddq_f16(vaddq_f16(t11, t12), vmulq_n_f16(vaddq_f16(t13, t14), 4)), bias_ptr);
  float16x8_t d13 =
    vaddq_f16(vaddq_f16(vaddq_f16(vsubq_f16(t11, t12), vmulq_n_f16(vsubq_f16(t13, t14), 8)), t15), bias_ptr);

  float16x8_t d20 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t20, t21), vaddq_f16(t22, t23)), t24), bias_ptr);
  float16x8_t d21 = vaddq_f16(vaddq_f16(vsubq_f16(t21, t22), vmulq_n_f16(vsubq_f16(t23, t24), 2)), bias_ptr);
  float16x8_t d22 = vaddq_f16(vaddq_f16(vaddq_f16(t21, t22), vmulq_n_f16(vaddq_f16(t23, t24), 4)), bias_ptr);
  float16x8_t d23 =
    vaddq_f16(vaddq_f16(vaddq_f16(vsubq_f16(t21, t22), vmulq_n_f16(vsubq_f16(t23, t24), 8)), t25), bias_ptr);

  float16x8_t d30 = vaddq_f16(vaddq_f16(vaddq_f16(vaddq_f16(t30, t31), vaddq_f16(t32, t33)), t34), bias_ptr);
  float16x8_t d31 = vaddq_f16(vaddq_f16(vsubq_f16(t31, t32), vmulq_n_f16(vsubq_f16(t33, t34), 2)), bias_ptr);
  float16x8_t d32 = vaddq_f16(vaddq_f16(vaddq_f16(t31, t32), vmulq_n_f16(vaddq_f16(t33, t34), 4)), bias_ptr);
  float16x8_t d33 =
    vaddq_f16(vaddq_f16(vaddq_f16(vsubq_f16(t31, t32), vmulq_n_f16(vsubq_f16(t33, t34), 8)), t35), bias_ptr);

  vst1q_f16(output_data, d00);
  vst1q_f16(output_data + 8, d01);
  vst1q_f16(output_data + 16, d02);
  vst1q_f16(output_data + 24, d03);

  vst1q_f16(output_data + output_w * 8, d10);
  vst1q_f16(output_data + output_w * 8 + 8, d11);
  vst1q_f16(output_data + output_w * 8 + 16, d12);
  vst1q_f16(output_data + output_w * 8 + 24, d13);

  vst1q_f16(output_data + 2 * output_w * 8, d20);
  vst1q_f16(output_data + 2 * output_w * 8 + 8, d21);
  vst1q_f16(output_data + 2 * output_w * 8 + 16, d22);
  vst1q_f16(output_data + 2 * output_w * 8 + 24, d23);

  vst1q_f16(output_data + 3 * output_w * 8, d30);
  vst1q_f16(output_data + 3 * output_w * 8 + 8, d31);
  vst1q_f16(output_data + 3 * output_w * 8 + 16, d32);
  vst1q_f16(output_data + 3 * output_w * 8 + 24, d33);
}

void Conv3x3Fp16OutputTransform(const float16_t *gemm_out, float16_t *out_data, const float16_t *bias_data,
                                int start_index, int real_cal_num, int out_w_block, ConvParameter *conv_param) {
  int output_channel = conv_param->output_channel_;
  int output_h = conv_param->output_h_;
  int out_h_block = UP_DIV(output_h, C4NUM);
  int oc8 = UP_DIV(output_channel, C8NUM);
  if (out_w_block == 0) {
    return;
  }
  for (int i = 0; i < real_cal_num; i++) {
    int out_w_index = (start_index + i) % out_w_block;
    int out_h_index = (start_index + i) / out_w_block;
    int src_tile_offset = i * oc8 * C8NUM * 36;
    int dst_tile_offset = C8NUM * (out_w_index * C4NUM + out_h_index * C4NUM * out_w_block * C4NUM);

    for (int j = 0; j < oc8; j++) {
      int src_oc8_offset = src_tile_offset + j * 36 * C8NUM;
      int dst_oc8_offset = dst_tile_offset + j * C8NUM * out_h_block * out_w_block * C4NUM * C4NUM;
      const float16_t *src_ptr = gemm_out + src_oc8_offset;
      const float16_t *bias_ptr = bias_data + j * C8NUM;
      float16_t *dst_ptr = out_data + dst_oc8_offset;

      // output transform
      Conv3x3Fp16OutputUnit(src_ptr, bias_ptr, dst_ptr, out_w_block * C4NUM);
    }
  }
}

// fp16 common winograd
void WinogradInputTransformFp16(const float16_t *input_data, float16_t *trans_input, float16_t *tmp_data, int cal_num,
                                int out_tile_index, int out_w_block_num, ConvParameter *conv_param,
                                InputTransFp16Func func) {
  const int tile_num = 16;
  int input_unit = conv_param->input_unit_;
  int output_unit = conv_param->output_unit_;
  int in_channel = conv_param->input_channel_;
  int ic8 = UP_DIV(in_channel, C8NUM);
  int pad_h = conv_param->pad_u_;
  int pad_w = conv_param->pad_l_;
  int input_h = conv_param->input_h_;
  int input_w = conv_param->input_w_;
  if (out_w_block_num == 0) {
    return;
  }
  for (int c = 0; c < cal_num; c++) {  // actual tiled number
    int src_x_s = (out_tile_index % out_w_block_num) * output_unit - pad_w;
    int src_y_s = (out_tile_index / out_w_block_num) * output_unit - pad_h;
    int interval_x_s = src_x_s > 0 ? 0 : -src_x_s;
    int interval_y_s = src_y_s > 0 ? 0 : -src_y_s;
    int src_x_e = src_x_s + input_unit;
    int src_y_e = src_y_s + input_unit;
    int interval_x_e = src_x_e < input_w ? input_unit : (input_w - src_x_s);
    int interval_y_e = src_y_e < input_h ? input_unit : (input_h - src_y_s);

    int src_plane_offset = in_channel * (src_y_s * input_w + src_x_s);
    int dst_plane_offset = c * in_channel;
    for (int ic = 0; ic < ic8; ic++) {
      // clear tmp buffer
      memset(tmp_data, 0, input_unit * input_unit * C8NUM * sizeof(float16_t));

      int real_c = in_channel - ic * C8NUM;
      real_c = real_c > C8NUM ? C8NUM : real_c;
      int src_ic8_offset = src_plane_offset + ic * C8NUM;

      // get real input block with padding
      if (real_c == C8NUM) {
        for (int interval = interval_y_s; interval < interval_y_e; interval++) {
          int src_y_offset = src_ic8_offset + (interval * input_w + interval_x_s) * in_channel;
          int dst_y_offset = interval * input_unit * C8NUM + interval_x_s * C8NUM;
          for (int j = 0; j < (interval_x_e - interval_x_s); j++) {
            int src_x_offset = src_y_offset + j * in_channel;
            int dst_x_offset = dst_y_offset + j * C8NUM;
            const float16_t *src_addr = input_data + src_x_offset;
            float16_t *dst_addr = tmp_data + dst_x_offset;
#ifdef ENABLE_NEON
            vst1q_f16(dst_addr, vld1q_f16(src_addr));
#else
            for (int k = 0; k < C8NUM; k++) {
              dst_addr[k] = src_addr[k];
            }
#endif
          }
        }
      } else if (real_c < 8 && real_c >= 4) {
        for (int interval = interval_y_s; interval < interval_y_e; interval++) {
          int src_y_offset = src_ic8_offset + (interval * input_w + interval_x_s) * in_channel;
          int dst_y_offset = interval * input_unit * C8NUM + interval_x_s * C8NUM;
          for (int j = 0; j < (interval_x_e - interval_x_s); j++) {
            int src_x_offset = src_y_offset + j * in_channel;
            int dst_x_offset = dst_y_offset + j * C8NUM;
            const float16_t *src_addr = input_data + src_x_offset;
            float16_t *dst_addr = tmp_data + dst_x_offset;
            int rc = real_c - 4;
#ifdef ENABLE_NEON
            vst1_f16(dst_addr, vld1_f16(src_addr));
#else
            for (int k = 0; k < C4NUM; k++) {
              dst_addr[k] = src_addr[k];
            }
#endif
            src_addr += 4;
            dst_addr += 4;
            for (int i = 0; i < rc; ++i) {
              dst_addr[i] = src_addr[i];
            }
          }
        }
      } else {
        for (int interval = interval_y_s; interval < interval_y_e; interval++) {
          int src_y_offset = src_ic8_offset + (interval * input_w + interval_x_s) * in_channel;
          int dst_y_offset = interval * input_unit * C8NUM + interval_x_s * C8NUM;
          for (int j = 0; j < (interval_x_e - interval_x_s); j++) {
            int src_x_offset = src_y_offset + j * in_channel;
            int dst_x_offset = dst_y_offset + j * C8NUM;
            const float16_t *src_addr = input_data + src_x_offset;
            float16_t *dst_addr = tmp_data + dst_x_offset;
            for (int k = 0; k < real_c; k++) {
              dst_addr[k] = src_addr[k];
            }
          }
        }
      }

      // input transform
      int dst_ic8_offset = dst_plane_offset + ic * C8NUM;
      size_t dst_step = in_channel * tile_num;
      float16_t *trans_input_ptr = trans_input + dst_ic8_offset;
      func(tmp_data, trans_input_ptr, C8NUM, dst_step, real_c);
    }
    out_tile_index++;
  }  // cal_tile_num loop
}

void WinogradOutputTransformFp16(const float16_t *gemm_out, float16_t *tmp_out_data, const float16_t *bias_data,
                                 int cal_num, int out_tile_index, int output_unit_num, ConvParameter *conv_param,
                                 OutputTransFp16Func func) {
  int output_unit = conv_param->output_unit_;
  int output_w = conv_param->output_w_;
  int output_h = conv_param->output_h_;
  int output_channel = conv_param->output_channel_;
  int oc8 = UP_DIV(output_channel, C8NUM);
  int input_unit = conv_param->input_unit_;
  if (output_unit_num == 0) {
    return;
  }
  for (int i = 0; i < cal_num; i++) {
    int dst_x_s = out_tile_index % output_unit_num;
    int dst_y_s = out_tile_index / output_unit_num;
    int r_w = output_w - dst_x_s * output_unit;
    r_w = r_w > output_unit ? output_unit : r_w;
    int r_h = output_h - dst_y_s * output_unit;
    r_h = r_h > output_unit ? output_unit : r_h;
    int tmp_ix = dst_x_s * output_unit;
    dst_x_s = tmp_ix > output_w ? output_w : tmp_ix;
    int tmp_iy = dst_y_s * output_unit;
    dst_y_s = tmp_iy > output_h ? output_h : tmp_iy;

    int src_tile_offset = i * oc8 * C8NUM * input_unit * input_unit;
    int dst_tile_offset = output_channel * (dst_x_s + dst_y_s * output_w);

    for (int j = 0; j < oc8; j++) {
      int r_c = output_channel - j * C8NUM;
      r_c = r_c > C8NUM ? C8NUM : r_c;
      int src_oc8_offset = src_tile_offset + j * input_unit * input_unit * C8NUM;
      int dst_oc8_offset = dst_tile_offset + j * C8NUM;
      const float16_t *src_ptr = gemm_out + src_oc8_offset;
      const float16_t *bias_ptr = bias_data + j * C8NUM;
      float16_t *dst_ptr = tmp_out_data + dst_oc8_offset;
      func(src_ptr, dst_ptr, bias_ptr, C8NUM, output_w, output_channel, r_w, r_h, r_c);
    }
    out_tile_index++;
  }
}

int WinogradWeightTransformFp16(const float16_t *weight_data, float16_t *winograd_data, float *matrix_g,
                                float *matrix_gt, int oc_block, int input_unit, int kernel_unit, int filter_channel,
                                int filter_batch, bool pack) {
  // original weight format : ohwi
  int oc_block_num = UP_DIV(filter_batch, oc_block);
  int block_stride = filter_channel * oc_block;
  int block_num_stride = block_stride * oc_block_num;

  float16_t *matrix_gt_data_fp16 = (float16_t *)(malloc(input_unit * kernel_unit * sizeof(float16_t)));
  if (matrix_gt_data_fp16 == NULL) {
    return NNACL_ERRCODE_OP_FP16_WINOGRAD_GENERATOR;
  }
  Float32ToFloat16(matrix_gt, matrix_gt_data_fp16, input_unit * kernel_unit);

  // trans_filter = G*g*GT (g represents weight_data) = [(g * (G)T)T * (G)T]T
  // separate into two steps ===> tmp = (g * (G)T)T ===> out = [tmp * (G)T]T
  float16_t *tmp_data = (float16_t *)(malloc(filter_channel * input_unit * kernel_unit * sizeof(float16_t)));
  if (tmp_data == NULL) {
    free(matrix_gt_data_fp16);
    return NNACL_ERRCODE_OP_FP16_WINOGRAD_GENERATOR;
  }
  float16_t *trans_out_data = (float16_t *)(malloc(filter_channel * input_unit * input_unit * sizeof(float16_t)));
  if (trans_out_data == NULL) {
    free(tmp_data);
    free(matrix_gt_data_fp16);
    return NNACL_ERRCODE_OP_FP16_WINOGRAD_GENERATOR;
  }

#ifndef ENABLE_ARM64
  float16_t *tmp_data1 = (float16_t *)(malloc(filter_channel * input_unit * kernel_unit * sizeof(float16_t)));
  if (tmp_data1 == NULL) {
    free(tmp_data);
    free(matrix_gt_data_fp16);
    free(trans_out_data);
    return NNACL_ERRCODE_OP_FP16_WINOGRAD_GENERATOR;
  }
  float16_t *trans_out_data1 = (float16_t *)(malloc(filter_channel * input_unit * input_unit * sizeof(float16_t)));
  if (trans_out_data1 == NULL) {
    free(tmp_data);
    free(tmp_data1);
    free(matrix_gt_data_fp16);
    free(trans_out_data);
    return NNACL_ERRCODE_OP_FP16_WINOGRAD_GENERATOR;
  }
#endif

  int input_oz_offset = kernel_unit * kernel_unit * filter_channel;
  for (int i = 0; i < filter_batch; i++) {
    int out_c_block = i / oc_block;
    int out_c_res = i % oc_block;
    int output_oz_offset = out_c_block * block_stride + out_c_res;

#ifndef ENABLE_ARM64
    // tmp_data = g * GT
    MatrixMultiplyWinogradFp16(weight_data + i * input_oz_offset, matrix_gt_data_fp16, tmp_data, kernel_unit,
                               kernel_unit, input_unit, filter_channel);
    // tmp_data1 = (tmp_data)T
    PackHWCToWHCFp16(tmp_data, tmp_data1, kernel_unit, input_unit, filter_channel);
    // trans_out_data1 = tmp * GT
    MatrixMultiplyWinogradFp16(tmp_data1, matrix_gt_data_fp16, trans_out_data1, input_unit, kernel_unit, input_unit,
                               filter_channel);
    // trans_out_data = (trans_out_data1)T
    PackHWCToWHCFp16(trans_out_data1, trans_out_data, input_unit, input_unit, filter_channel);
#else
    // tmp = (g * GT)T
    MatrixMultiplyWinogradFp16(weight_data + i * input_oz_offset, matrix_gt_data_fp16, tmp_data, kernel_unit,
                               kernel_unit, input_unit, filter_channel);
    // trans = (tmp * GT)T
    MatrixMultiplyWinogradFp16(tmp_data, matrix_gt_data_fp16, trans_out_data, input_unit, kernel_unit, input_unit,
                               filter_channel);
#endif

    if (pack) {
      int in_offset = 0;
      for (int j = 0; j < input_unit; ++j) {
        for (int k = 0; k < input_unit; ++k) {
          for (int c = 0; c < filter_channel; ++c) {
            *(winograd_data + output_oz_offset + c * oc_block) = trans_out_data[in_offset + c];
          }
          in_offset += filter_channel;
          output_oz_offset += block_num_stride;
        }
      }
    } else {
      memcpy(winograd_data + i * filter_channel * input_unit * input_unit, trans_out_data,
             filter_channel * input_unit * input_unit * sizeof(float16_t));
    }
  }

#ifndef ENABLE_ARM64
  free(tmp_data1);
  free(trans_out_data1);
#endif
  free(tmp_data);
  free(trans_out_data);
  free(matrix_gt_data_fp16);
  return NNACL_OK;
}
