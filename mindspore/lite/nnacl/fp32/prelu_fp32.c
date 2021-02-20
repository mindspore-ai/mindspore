/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

void PRelu(float *input, float *output, const PReluParameter *prelu_param_, int plane) {
  int plane_tile = plane / TILE_NUM * TILE_NUM;
  int channel_num = prelu_param_->channel_num_;
  int plane_index = 0;
  for (; plane_index < plane_tile; plane_index += TILE_NUM) {
    float *in_plane_ptr = input + plane_index * channel_num;
    float *out_plane_ptr = output + plane_index * channel_num;
    int channel_index = 0;
#if defined(ENABLE_AVX)
    MS_FLOAT32X8 zero_value_8 = MS_MOV256_F32(0.0f);
    MS_FLOAT32X8 one_value_8 = MS_MOV256_F32(1.0f);
    float *negetive_slope_value_8 = prelu_param_->slope_;
    int div_channel_c8 = prelu_param_->channel_num_ / C8NUM * C8NUM;
    for (; channel_index < div_channel_c8; channel_index += C8NUM) {
      MS_FLOAT32X8 slope_value_8 = MS_LD256_F32(negetive_slope_value_8 + channel_index);
      LOAD256X8_F32(src, in_plane_ptr + channel_index, channel_num)
      PRELU_CALCULATE_256X8(dst, src)
      STORE256X8_F32(out_plane_ptr + channel_index, channel_num, dst)
    }
#endif
    // note: First AVX processing, then SSE processing on X86 platform
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
    MS_FLOAT32X4 zero_value = MS_MOVQ_F32(0.0f);
    MS_FLOAT32X4 one_value = MS_MOVQ_F32(1.0f);
    float *negetive_slope_value = prelu_param_->slope_;
    int div_channel = prelu_param_->channel_num_ / C4NUM * C4NUM;
    for (; channel_index < div_channel; channel_index += C4NUM) {
      MS_FLOAT32X4 slope_value = MS_LDQ_F32(negetive_slope_value + channel_index);
      LOAD128X8_F32(src, in_plane_ptr + channel_index, channel_num)
      PRELU_CALCULATE_128X8(dst, src)
      STORE128X8_F32(out_plane_ptr + channel_index, channel_num, dst)
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
#if defined(ENABLE_ARM64) || defined(ENABLE_AVX)
    cal_index = j * 64;
#else
    cal_index = j * 32;
#endif

    float *input_ptr = input + cal_index;
    float *output_ptr = input + cal_index;
#if defined(ENABLE_AVX)
    MS_FLOAT32X8 zero_value_8 = MS_MOV256_F32(0);
    MS_FLOAT32X8 one_value_8 = MS_MOV256_F32(1.0f);
    MS_FLOAT32X8 slope_value_8 = MS_MOV256_F32(prelu_param_->slope_[0]);
    LOAD256X8_F32(src, input_ptr, 8)
    PRELU_CALCULATE_256X8(dst, src)
    STORE256X8_F32(output_ptr, 8, dst)
#elif defined(ENABLE_ARM) || (defined(ENABLE_SSE) && !defined(ENABLE_AVX))
    MS_FLOAT32X4 zero_value = MS_MOVQ_F32(0);
    MS_FLOAT32X4 one_value = MS_MOVQ_F32(1.0f);
    MS_FLOAT32X4 slope_value = MS_MOVQ_F32(prelu_param_->slope_[0]);

    LOAD128X8_F32(src, input_ptr, 4)
#ifdef ENABLE_ARM64
    LOAD128X8_F32(src1, input_ptr + 32, 4)
#endif
    PRELU_CALCULATE_128X8(dst, src)
#ifdef ENABLE_ARM64
    PRELU_CALCULATE_128X8(dst1, src1)
#endif
    STORE128X8_F32(output_ptr, 4, dst)
#ifdef ENABLE_ARM64
    STORE128X8_F32(output_ptr + 32, 4, dst1)
#endif

#else
    const int cal_per_time = 32;
    for (int i = 0; i < cal_per_time; ++i) {
      float data = input_ptr[i];
      output_ptr[i] = (data < 0 ? data : 0) * prelu_param_->slope_[0] + (data > 0 ? data : 0);
    }
#endif
  }
}
