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

#include "nnacl/fp32/pooling.h"
#include <float.h>

void AvgPooling(const float *input_ptr, float *output_ptr, PoolingParameter *pooling_param, int task_id) {
  int stride_w = pooling_param->stride_w_;
  int stride_h = pooling_param->stride_h_;
  int pad_w = pooling_param->pad_l_;
  int pad_h = pooling_param->pad_u_;
  int win_w = pooling_param->window_w_;
  int win_h = pooling_param->window_h_;
  int channel = pooling_param->input_channel_;
  int c4 = UP_DIV(channel, C4NUM);
  int in_w = pooling_param->input_w_;
  int in_h = pooling_param->input_h_;
  int output_w = pooling_param->output_w_;
  int output_h = pooling_param->output_h_;
  int output_batch = pooling_param->output_batch_;
  int out_plane = output_w * output_h;
  int out_tile_count = UP_DIV(out_plane, TILE_NUM);
  int thread_num = pooling_param->thread_num_;
  // input channel is equal to output channel

  for (int batch = 0; batch < output_batch; batch++) {
    int in_batch_offset = batch * in_h * in_w * channel;
    int out_batch_offset = batch * output_h * output_w * channel;
    for (int thread_id = task_id; thread_id < out_tile_count; thread_id += thread_num) {
      int cal_start_index = thread_id * TILE_NUM;
      int real_cal_num = (out_plane - cal_start_index) > TILE_NUM ? TILE_NUM : (out_plane - cal_start_index);
      for (int i = 0; i < real_cal_num; i++) {
        int index = cal_start_index + i;
        int out_w_index = index % output_w;
        int out_h_index = index / output_w;
        int in_w_index = out_w_index * stride_w - pad_w;
        int in_h_index = out_h_index * stride_h - pad_h;
        int out_plane_offset = out_batch_offset + index * channel;
        for (int j = 0; j < c4 - 1; j++) {
          int in_channel_offset = in_batch_offset + j * C4NUM;
          int out_channel_offset = out_plane_offset + j * C4NUM;
#ifdef ENABLE_NEON
          float32x4_t tmp_avg = vdupq_n_f32(0);
#else
          float tmp_avg1 = 0;
          float tmp_avg2 = 0;
          float tmp_avg3 = 0;
          float tmp_avg4 = 0;
#endif
          int real_count = 0;
          for (int h = 0; h < win_h; h++) {
            for (int w = 0; w < win_w; w++) {
              if ((in_h_index + h) < 0 || (in_h_index + h) >= in_h || (in_w_index + w) < 0 ||
                  (in_w_index + w) >= in_w) {
                continue;
              } else {
                int in_offset = in_channel_offset + ((in_h_index + h) * in_w + in_w_index + w) * channel;
#ifdef ENABLE_NEON
                tmp_avg = vaddq_f32(tmp_avg, vld1q_f32(input_ptr + in_offset));
#else
                tmp_avg1 += *(input_ptr + in_offset);
                tmp_avg2 += *(input_ptr + in_offset + 1);
                tmp_avg3 += *(input_ptr + in_offset + 2);
                tmp_avg4 += *(input_ptr + in_offset + 3);
#endif
                ++real_count;
              }
            }  // win_w loop
          }    // win_h loop
#ifdef ENABLE_NEON
          vst1q_f32(output_ptr + out_channel_offset, tmp_avg / vdupq_n_f32(real_count));
#else
          *(output_ptr + out_channel_offset) = tmp_avg1 / (float)real_count;
          *(output_ptr + out_channel_offset + 1) = tmp_avg2 / (float)real_count;
          *(output_ptr + out_channel_offset + 2) = tmp_avg3 / (float)real_count;
          *(output_ptr + out_channel_offset + 3) = tmp_avg4 / (float)real_count;
#endif
        }  // ic4-1 loop
        int channel_s = (c4 - 1) * C4NUM;
        for (int k = channel_s; k < channel; k++) {
          int in_channel_offset = in_batch_offset + k;
          int out_channel_offset = out_plane_offset + k;
          float tmp_avg = 0;
          int real_count = 0;
          for (int h = 0; h < win_h; h++) {
            for (int w = 0; w < win_w; w++) {
              if ((in_h_index + h) < 0 || (in_h_index + h) >= in_h || (in_w_index + w) < 0 ||
                  (in_w_index + w) >= in_w) {
                continue;
              } else {
                int in_offset = in_channel_offset + ((in_h_index + h) * in_w + in_w_index + w) * channel;
                tmp_avg += *(input_ptr + in_offset);
                ++real_count;
              }
            }  // win_w loop
          }    // win_h loop
          *(output_ptr + out_channel_offset) = tmp_avg / (float)real_count;
        }  // channel_res loop
      }    // real_cal_num loop
    }      // out_plane loop
  }        // out_batch loop
}

void MaxPooling(const float *input_ptr, float *output_ptr, PoolingParameter *pooling_param, int task_id) {
  int stride_w = pooling_param->stride_w_;
  int stride_h = pooling_param->stride_h_;
  int pad_w = pooling_param->pad_l_;
  int pad_h = pooling_param->pad_u_;
  int win_w = pooling_param->window_w_;
  int win_h = pooling_param->window_h_;
  int channel = pooling_param->input_channel_;
  int in_w = pooling_param->input_w_;
  int in_h = pooling_param->input_h_;
  int output_w = pooling_param->output_w_;
  int output_h = pooling_param->output_h_;
  int output_batch = pooling_param->output_batch_;
  int out_plane = output_w * output_h;
  int out_tile_count = UP_DIV(out_plane, TILE_NUM);
  int thread_num = pooling_param->thread_num_;
  int c4 = UP_DIV(channel, C4NUM);
  // input channel is equal to output channel

  for (int batch = 0; batch < output_batch; batch++) {
    int in_batch_offset = batch * in_h * in_w * channel;
    int out_batch_offset = batch * output_h * output_w * channel;
    for (int thread_id = task_id; thread_id < out_tile_count; thread_id += thread_num) {
      int cal_start_index = thread_id * TILE_NUM;
      int real_cal_num = (out_plane - cal_start_index) > TILE_NUM ? TILE_NUM : (out_plane - cal_start_index);
      for (int i = 0; i < real_cal_num; i++) {
        int index = cal_start_index + i;
        int out_w_index = index % output_w;
        int out_h_index = index / output_w;
        int in_w_index = out_w_index * stride_w - pad_w;
        int in_h_index = out_h_index * stride_h - pad_h;
        int out_plane_offset = out_batch_offset + index * channel;
        for (int j = 0; j < c4 - 1; j++) {
          int in_channel_offset = in_batch_offset + j * C4NUM;
          int out_channel_offset = out_plane_offset + j * C4NUM;
#ifdef ENABLE_NEON
          float32x4_t tmp_max = vdupq_n_f32(-FLT_MAX);
#else
          float tmp_max1 = -FLT_MAX;
          float tmp_max2 = -FLT_MAX;
          float tmp_max3 = -FLT_MAX;
          float tmp_max4 = -FLT_MAX;
#endif
          for (int h = 0; h < win_h; h++) {
            for (int w = 0; w < win_w; w++) {
              if ((in_h_index + h) < 0 || (in_h_index + h) >= in_h || (in_w_index + w) < 0 ||
                  (in_w_index + w) >= in_w) {
                continue;
              } else {
                int in_offset = in_channel_offset + ((in_h_index + h) * in_w + in_w_index + w) * channel;
#ifdef ENABLE_NEON
                tmp_max = vmaxq_f32(tmp_max, vld1q_f32(input_ptr + in_offset));
#else
                tmp_max1 = fmax(tmp_max1, *(input_ptr + in_offset));
                tmp_max2 = fmax(tmp_max2, *(input_ptr + in_offset + 1));
                tmp_max3 = fmax(tmp_max3, *(input_ptr + in_offset + 2));
                tmp_max4 = fmax(tmp_max4, *(input_ptr + in_offset + 3));
#endif
              }
            }  // win_w loop
          }    // win_h loop
#ifdef ENABLE_NEON
          vst1q_f32(output_ptr + out_channel_offset, tmp_max);
#else
          *(output_ptr + out_channel_offset) = tmp_max1;
          *(output_ptr + out_channel_offset + 1) = tmp_max2;
          *(output_ptr + out_channel_offset + 2) = tmp_max3;
          *(output_ptr + out_channel_offset + 3) = tmp_max4;
#endif
        }  // ic4-1 loop
        int channel_s = (c4 - 1) * C4NUM;
        for (int k = channel_s; k < channel; k++) {
          int in_channel_offset = in_batch_offset + k;
          int out_channel_offset = out_plane_offset + k;
          float tmp_max = -FLT_MAX;
          for (int h = 0; h < win_h; h++) {
            for (int w = 0; w < win_w; w++) {
              if ((in_h_index + h) < 0 || (in_h_index + h) >= in_h || (in_w_index + w) < 0 ||
                  (in_w_index + w) >= in_w) {
                continue;
              } else {
                int in_offset = in_channel_offset + ((in_h_index + h) * in_w + in_w_index + w) * channel;
                tmp_max = fmax(tmp_max, *(input_ptr + in_offset));
              }
            }  // win_w loop
          }    // win_h loop
          *(output_ptr + out_channel_offset) = tmp_max;
        }  // channel_res loop
      }    // real_cal_num loop
    }      // out_plane loop
  }        // out_batch loop
}
