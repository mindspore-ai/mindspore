/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
// * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "nnacl/fp32/prelu_fp32.h"
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif

void PRelu(float *input, float *output, const PReluParameter *prelu_param_, int plane) {
#ifdef ENABLE_ARM
  float32x4_t zero_value = vdupq_n_f32(0);
#endif
  int plane_tile = plane / TILE_NUM * TILE_NUM;
  int channel_num = prelu_param_->channel_num_;
  int plane_index = 0;
  for (; plane_index < plane_tile; plane_index += TILE_NUM) {
    float *in_plane_ptr = input + plane_index * channel_num;
    float *out_plane_ptr = output + plane_index * channel_num;
    int channel_index = 0;
#ifdef ENABLE_ARM
    float *negetive_slope_value = prelu_param_->slope_;
    int div_channel = prelu_param_->channel_num_ / C4NUM * C4NUM;
    for (; channel_index < div_channel; channel_index += C4NUM) {
      float32x4_t slope_value = vld1q_f32(negetive_slope_value + channel_index);
      float32x4_t v1 = vld1q_f32(in_plane_ptr + channel_index + 0 * channel_num);
      float32x4_t v2 = vld1q_f32(in_plane_ptr + channel_index + 1 * channel_num);
      float32x4_t v3 = vld1q_f32(in_plane_ptr + channel_index + 2 * channel_num);
      float32x4_t v4 = vld1q_f32(in_plane_ptr + channel_index + 3 * channel_num);
      float32x4_t v5 = vld1q_f32(in_plane_ptr + channel_index + 4 * channel_num);
      float32x4_t v6 = vld1q_f32(in_plane_ptr + channel_index + 5 * channel_num);
      float32x4_t v7 = vld1q_f32(in_plane_ptr + channel_index + 6 * channel_num);
      float32x4_t v8 = vld1q_f32(in_plane_ptr + channel_index + 7 * channel_num);

      float32x4_t r1 = vaddq_f32(vmulq_f32(vminq_f32(v1, zero_value), slope_value), vmaxq_f32(v1, zero_value));
      float32x4_t r2 = vaddq_f32(vmulq_f32(vminq_f32(v2, zero_value), slope_value), vmaxq_f32(v2, zero_value));
      float32x4_t r3 = vaddq_f32(vmulq_f32(vminq_f32(v3, zero_value), slope_value), vmaxq_f32(v3, zero_value));
      float32x4_t r4 = vaddq_f32(vmulq_f32(vminq_f32(v4, zero_value), slope_value), vmaxq_f32(v4, zero_value));
      float32x4_t r5 = vaddq_f32(vmulq_f32(vminq_f32(v5, zero_value), slope_value), vmaxq_f32(v5, zero_value));
      float32x4_t r6 = vaddq_f32(vmulq_f32(vminq_f32(v6, zero_value), slope_value), vmaxq_f32(v6, zero_value));
      float32x4_t r7 = vaddq_f32(vmulq_f32(vminq_f32(v7, zero_value), slope_value), vmaxq_f32(v7, zero_value));
      float32x4_t r8 = vaddq_f32(vmulq_f32(vminq_f32(v8, zero_value), slope_value), vmaxq_f32(v8, zero_value));

      vst1q_f32(out_plane_ptr + channel_index + 0 * channel_num, r1);
      vst1q_f32(out_plane_ptr + channel_index + 1 * channel_num, r2);
      vst1q_f32(out_plane_ptr + channel_index + 2 * channel_num, r3);
      vst1q_f32(out_plane_ptr + channel_index + 3 * channel_num, r4);
      vst1q_f32(out_plane_ptr + channel_index + 4 * channel_num, r5);
      vst1q_f32(out_plane_ptr + channel_index + 5 * channel_num, r6);
      vst1q_f32(out_plane_ptr + channel_index + 6 * channel_num, r7);
      vst1q_f32(out_plane_ptr + channel_index + 7 * channel_num, r8);
    }
#endif
    for (; channel_index < channel_num; channel_index++) {
      float *in_c = in_plane_ptr + channel_index;
      float *out_c = out_plane_ptr + channel_index;
      for (int tile_i = 0; tile_i < TILE_NUM; tile_i++) {
        float *in_tile = in_c + tile_i * channel_num;
        float *out_tile = out_c + tile_i * channel_num;
        const float in_data = in_tile[0];
        out_tile[0] = (in_data < 0 ? in_data : 0) * prelu_param_->slope_[channel_index] + (in_data > 0 ? in_data : 0);
      }
    }
  }

  for (; plane_index < plane; plane_index++) {
    float *in_plane_ptr = input + plane_index * channel_num;
    float *out_plane_ptr = output + plane_index * channel_num;
    for (int channel_index = 0; channel_index < channel_num; channel_index++) {
      const float in_data = in_plane_ptr[channel_index];
      out_plane_ptr[channel_index] =
        (in_data < 0 ? in_data : 0) * prelu_param_->slope_[channel_index] + (in_data > 0 ? in_data : 0);
    }
  }
}

void PReluShareChannel(float *input, float *output, const PReluParameter *prelu_param_, int task_id) {
  for (int j = task_id; j < prelu_param_->tile_block_; j += prelu_param_->op_parameter_.thread_num_) {
    int cal_index;
#ifdef ENABLE_NEON
    float32x4_t slope_value = vdupq_n_f32(prelu_param_->slope_[0]);
    float32x4_t zero_value = vdupq_n_f32(0);
#endif
#ifdef ENABLE_ARM64
    cal_index = j * 64;

#elif ENABLE_ARM32
    cal_index = j * 32;
#else
    cal_index = j * 32;
    const int cal_per_time = 32;
#endif
    float *input_ptr = input + cal_index;
    float *output_ptr = input + cal_index;
#ifdef ENABLE_ARM64
    float32x4_t v1 = vld1q_f32(input_ptr);
    float32x4_t v2 = vld1q_f32(input_ptr + 4);
    float32x4_t v3 = vld1q_f32(input_ptr + 8);
    float32x4_t v4 = vld1q_f32(input_ptr + 12);
    float32x4_t v5 = vld1q_f32(input_ptr + 16);
    float32x4_t v6 = vld1q_f32(input_ptr + 20);
    float32x4_t v7 = vld1q_f32(input_ptr + 24);
    float32x4_t v8 = vld1q_f32(input_ptr + 28);
    float32x4_t v9 = vld1q_f32(input_ptr + 32);
    float32x4_t v10 = vld1q_f32(input_ptr + 36);
    float32x4_t v11 = vld1q_f32(input_ptr + 40);
    float32x4_t v12 = vld1q_f32(input_ptr + 44);
    float32x4_t v13 = vld1q_f32(input_ptr + 48);
    float32x4_t v14 = vld1q_f32(input_ptr + 52);
    float32x4_t v15 = vld1q_f32(input_ptr + 56);
    float32x4_t v16 = vld1q_f32(input_ptr + 60);

    float32x4_t t1 = vaddq_f32(vmulq_f32(vminq_f32(v1, zero_value), slope_value), vmaxq_f32(v1, zero_value));
    float32x4_t t2 = vaddq_f32(vmulq_f32(vminq_f32(v2, zero_value), slope_value), vmaxq_f32(v2, zero_value));
    float32x4_t t3 = vaddq_f32(vmulq_f32(vminq_f32(v3, zero_value), slope_value), vmaxq_f32(v3, zero_value));
    float32x4_t t4 = vaddq_f32(vmulq_f32(vminq_f32(v4, zero_value), slope_value), vmaxq_f32(v4, zero_value));
    float32x4_t t5 = vaddq_f32(vmulq_f32(vminq_f32(v5, zero_value), slope_value), vmaxq_f32(v5, zero_value));
    float32x4_t t6 = vaddq_f32(vmulq_f32(vminq_f32(v6, zero_value), slope_value), vmaxq_f32(v6, zero_value));
    float32x4_t t7 = vaddq_f32(vmulq_f32(vminq_f32(v7, zero_value), slope_value), vmaxq_f32(v7, zero_value));
    float32x4_t t8 = vaddq_f32(vmulq_f32(vminq_f32(v8, zero_value), slope_value), vmaxq_f32(v8, zero_value));
    float32x4_t t9 = vaddq_f32(vmulq_f32(vminq_f32(v9, zero_value), slope_value), vmaxq_f32(v9, zero_value));
    float32x4_t t10 = vaddq_f32(vmulq_f32(vminq_f32(v10, zero_value), slope_value), vmaxq_f32(v10, zero_value));
    float32x4_t t11 = vaddq_f32(vmulq_f32(vminq_f32(v11, zero_value), slope_value), vmaxq_f32(v11, zero_value));
    float32x4_t t12 = vaddq_f32(vmulq_f32(vminq_f32(v12, zero_value), slope_value), vmaxq_f32(v12, zero_value));
    float32x4_t t13 = vaddq_f32(vmulq_f32(vminq_f32(v13, zero_value), slope_value), vmaxq_f32(v13, zero_value));
    float32x4_t t14 = vaddq_f32(vmulq_f32(vminq_f32(v14, zero_value), slope_value), vmaxq_f32(v14, zero_value));
    float32x4_t t15 = vaddq_f32(vmulq_f32(vminq_f32(v15, zero_value), slope_value), vmaxq_f32(v15, zero_value));
    float32x4_t t16 = vaddq_f32(vmulq_f32(vminq_f32(v16, zero_value), slope_value), vmaxq_f32(v16, zero_value));

    vst1q_f32(output_ptr, t1);
    vst1q_f32(output_ptr + 4, t2);
    vst1q_f32(output_ptr + 8, t3);
    vst1q_f32(output_ptr + 12, t4);
    vst1q_f32(output_ptr + 16, t5);
    vst1q_f32(output_ptr + 20, t6);
    vst1q_f32(output_ptr + 24, t7);
    vst1q_f32(output_ptr + 28, t8);
    vst1q_f32(output_ptr + 32, t9);
    vst1q_f32(output_ptr + 36, t10);
    vst1q_f32(output_ptr + 40, t11);
    vst1q_f32(output_ptr + 44, t12);
    vst1q_f32(output_ptr + 48, t13);
    vst1q_f32(output_ptr + 52, t14);
    vst1q_f32(output_ptr + 56, t15);
    vst1q_f32(output_ptr + 60, t16);
#elif ENABLE_ARM32
    float32x4_t v1 = vld1q_f32(input_ptr);
    float32x4_t v2 = vld1q_f32(input_ptr + 4);
    float32x4_t v3 = vld1q_f32(input_ptr + 8);
    float32x4_t v4 = vld1q_f32(input_ptr + 12);
    float32x4_t v5 = vld1q_f32(input_ptr + 16);
    float32x4_t v6 = vld1q_f32(input_ptr + 20);
    float32x4_t v7 = vld1q_f32(input_ptr + 24);
    float32x4_t v8 = vld1q_f32(input_ptr + 28);

    float32x4_t t1 = vaddq_f32(vmulq_f32(vminq_f32(v1, zero_value), slope_value), vmaxq_f32(v1, zero_value));
    float32x4_t t2 = vaddq_f32(vmulq_f32(vminq_f32(v2, zero_value), slope_value), vmaxq_f32(v2, zero_value));
    float32x4_t t3 = vaddq_f32(vmulq_f32(vminq_f32(v3, zero_value), slope_value), vmaxq_f32(v3, zero_value));
    float32x4_t t4 = vaddq_f32(vmulq_f32(vminq_f32(v4, zero_value), slope_value), vmaxq_f32(v4, zero_value));
    float32x4_t t5 = vaddq_f32(vmulq_f32(vminq_f32(v5, zero_value), slope_value), vmaxq_f32(v5, zero_value));
    float32x4_t t6 = vaddq_f32(vmulq_f32(vminq_f32(v6, zero_value), slope_value), vmaxq_f32(v6, zero_value));
    float32x4_t t7 = vaddq_f32(vmulq_f32(vminq_f32(v7, zero_value), slope_value), vmaxq_f32(v7, zero_value));
    float32x4_t t8 = vaddq_f32(vmulq_f32(vminq_f32(v8, zero_value), slope_value), vmaxq_f32(v8, zero_value));

    vst1q_f32(output_ptr, t1);
    vst1q_f32(output_ptr + 4, t2);
    vst1q_f32(output_ptr + 8, t3);
    vst1q_f32(output_ptr + 12, t4);
    vst1q_f32(output_ptr + 16, t5);
    vst1q_f32(output_ptr + 20, t6);
    vst1q_f32(output_ptr + 24, t7);
    vst1q_f32(output_ptr + 28, t8);
#else
    for (int i = 0; i < cal_per_time; ++i) {
      float data = input_ptr[i];
      output_ptr[i] = (data < 0 ? data : 0) * prelu_param_->slope_[0] + (data > 0 ? data : 0);
    }
#endif
  }
}
