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
#include "nnacl/fp32/prelu.h"
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif

void PRelu(float *input, float *output, PReluParameter *prelu_param_, int task_id) {
  float *negetive_slope_value = prelu_param_->slope_;
  int c4 = prelu_param_->channel_num_ / C4NUM;
  int channel_num = prelu_param_->channel_num_;
  for (int j = task_id; j < prelu_param_->tile_block_; j += prelu_param_->op_parameter_.thread_num_) {
    float *input_ptr = input + j * TILE_NUM * channel_num;
    float *output_ptr = input_ptr;
#ifdef ENABLE_NEON
    for (int i = 0; i < c4; i++) {
      int c_offset = i * C4NUM;
      float32x4_t slope_value = vld1q_f32(negetive_slope_value + c_offset);
      float32x4_t v1 = vld1q_f32(input_ptr + c_offset);
      float32x4_t v2 = vld1q_f32(input_ptr + c_offset + channel_num);
      float32x4_t v3 = vld1q_f32(input_ptr + c_offset + 2 * channel_num);
      float32x4_t v4 = vld1q_f32(input_ptr + c_offset + 3 * channel_num);
      float32x4_t v5 = vld1q_f32(input_ptr + c_offset + 4 * channel_num);
      float32x4_t v6 = vld1q_f32(input_ptr + c_offset + 5 * channel_num);
      float32x4_t v7 = vld1q_f32(input_ptr + c_offset + 6 * channel_num);
      float32x4_t v8 = vld1q_f32(input_ptr + c_offset + 7 * channel_num);

      float32x4_t t1 = vmulq_f32(v1, slope_value);
      float32x4_t t2 = vmulq_f32(v2, slope_value);
      float32x4_t t3 = vmulq_f32(v3, slope_value);
      float32x4_t t4 = vmulq_f32(v4, slope_value);
      float32x4_t t5 = vmulq_f32(v5, slope_value);
      float32x4_t t6 = vmulq_f32(v6, slope_value);
      float32x4_t t7 = vmulq_f32(v7, slope_value);
      float32x4_t t8 = vmulq_f32(v8, slope_value);

      uint32x4_t flag1 = vclezq_f32(v1);
      uint32x4_t flag2 = vclezq_f32(v2);
      uint32x4_t flag3 = vclezq_f32(v3);
      uint32x4_t flag4 = vclezq_f32(v4);
      uint32x4_t flag5 = vclezq_f32(v5);
      uint32x4_t flag6 = vclezq_f32(v6);
      uint32x4_t flag7 = vclezq_f32(v7);
      uint32x4_t flag8 = vclezq_f32(v8);

      float32x4_t r1 = vbslq_f32(flag1, t1, v1);
      float32x4_t r2 = vbslq_f32(flag2, t2, v2);
      float32x4_t r3 = vbslq_f32(flag3, t3, v3);
      float32x4_t r4 = vbslq_f32(flag4, t4, v4);
      float32x4_t r5 = vbslq_f32(flag5, t5, v5);
      float32x4_t r6 = vbslq_f32(flag6, t6, v6);
      float32x4_t r7 = vbslq_f32(flag7, t7, v7);
      float32x4_t r8 = vbslq_f32(flag8, t8, v8);

      vst1q_f32(output_ptr + c_offset, r1);
      vst1q_f32(output_ptr + c_offset + channel_num, r2);
      vst1q_f32(output_ptr + c_offset + 2 * channel_num, r3);
      vst1q_f32(output_ptr + c_offset + 3 * channel_num, r4);
      vst1q_f32(output_ptr + c_offset + 4 * channel_num, r5);
      vst1q_f32(output_ptr + c_offset + 5 * channel_num, r6);
      vst1q_f32(output_ptr + c_offset + 6 * channel_num, r7);
      vst1q_f32(output_ptr + c_offset + 7 * channel_num, r8);
    }  // c4 -1 loop
#else
    for (int i = 0; i < TILE_NUM; ++i) {
      int tile_offset = i * channel_num;
      for (int k = 0; k < c4; ++k) {
        int c4_offset = tile_offset + k * C4NUM;
        int slope_offset = k * C4NUM;
        for (int l = 0; l < C4NUM; ++l) {
          float in_data = input_ptr[c4_offset + l];
          output_ptr[c4_offset + l] =
            (in_data < 0 ? in_data : 0) * negetive_slope_value[slope_offset + l] + (in_data > 0 ? in_data : 0);
        }
      }
    }  // c4 - 1 loop
#endif
    int c_s = c4 * C4NUM;
    for (int m = 0; m < TILE_NUM; ++m) {
      int offset = m * channel_num;
      for (int k = c_s; k < channel_num; ++k) {
        int c4_offset = offset + k;
        float in_data = input_ptr[c4_offset];
        if (in_data >= 0) {
          output_ptr[c4_offset] = in_data;
        } else {
          output_ptr[c4_offset] = in_data * negetive_slope_value[k];
        }
      }
    }  // res loop
  }
}

void PReluShareChannel(float *input, float *output, PReluParameter *prelu_param_, int task_id) {
  for (int j = task_id; j < prelu_param_->tile_block_; j += prelu_param_->op_parameter_.thread_num_) {
    int cal_index;
    int cal_per_time;
#ifdef ENABLE_NEON
    float32x4_t slope_value = vdupq_n_f32(prelu_param_->slope_[0]);
    float32x4_t zero_value = vdupq_n_f32(0);
#endif
#ifdef ENABLE_ARM64
    cal_index = j * 64;
    cal_per_time = 64;

#elif ENABLE_ARM32
    cal_index = j * 32;
    cal_per_time = 32;
#else
    cal_index = j * 32;
    cal_per_time = 32;
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
