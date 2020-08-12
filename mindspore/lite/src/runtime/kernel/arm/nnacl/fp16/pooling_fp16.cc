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
#include "nnacl/fp16/pooling_fp16.h"
#include <float.h>

void AvgPoolingFp16(const float16_t *input_ptr, float16_t *output_ptr, PoolingParameter *pooling_param, int task_id) {
  int stride_w = pooling_param->stride_w_;
  int stride_h = pooling_param->stride_h_;
  int pad_w = pooling_param->pad_l_;
  int pad_h = pooling_param->pad_u_;
  int win_w = pooling_param->window_w_;
  int win_h = pooling_param->window_h_;
  int channel = pooling_param->input_channel_;
  int c8 = channel / C8NUM;
  int c8_res = channel % C8NUM;
  int c4 = c8_res / C4NUM;

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
        for (int j = 0; j < c8; j++) {
          int in_channel_offset = in_batch_offset + j * C8NUM;
          int out_channel_offset = out_plane_offset + j * C8NUM;
#ifdef ENABLE_NEON
          float16x8_t tmp_avg = vdupq_n_f16(0);
#else
          float16_t tmp_avg[8]{0};
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
                tmp_avg = vaddq_f16(tmp_avg, vld1q_f16(input_ptr + in_offset));
#else
                for (int t = 0; t < 8; t++) {
                  tmp_avg[t] += *(input_ptr + in_offset + t);
                }
#endif
                ++real_count;
              }
            }  // win_w loop
          }    // win_h loop
#ifdef ENABLE_NEON
          vst1q_f16(output_ptr + out_channel_offset, tmp_avg / vdupq_n_f16(real_count));
#else
          for (int t = 0; t < C8NUM; ++t) {
            *(output_ptr + out_channel_offset + t) = tmp_avg[t] / (float16_t)real_count;
          }
#endif
        }  // c8 loop

        int c4_offset = c8 * C8NUM;
        for (int l = 0; l < c4; ++l) {
          int in_channel_offset = in_batch_offset + c4_offset + l * C4NUM;
          int out_channel_offset = out_plane_offset + c4_offset + l * C4NUM;
#ifdef ENABLE_NEON
          float16x4_t tmp_avg = vdup_n_f16(0);
#else
          float16_t tmp_avg[4]{0};
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
                tmp_avg = vadd_f16(tmp_avg, vld1_f16(input_ptr + in_offset));
#else
                for (int j = 0; j < C4NUM; ++j) {
                  tmp_avg[j] += *(input_ptr + in_offset);
                }
#endif
                ++real_count;
              }
            }  // win_w loop
          }    // win_h loop
#ifdef ENABLE_NEON
          vst1_f16(output_ptr + out_channel_offset, tmp_avg / vdup_n_f16(real_count));
#else
          for (int t = 0; t < C4NUM; ++t) {
            *(output_ptr + out_channel_offset + t) = tmp_avg[t] / (float16_t)real_count;
          }
#endif
        }  // c4 loop

        int channel_s = c8 * C8NUM + c4 * C4NUM;
        for (int k = channel_s; k < channel; k++) {
          int in_channel_offset = in_batch_offset + k;
          int out_channel_offset = out_plane_offset + k;
          float16_t tmp_avg = 0;
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
          *(output_ptr + out_channel_offset) = tmp_avg / (float16_t)real_count;
        }  // channel_res loop
      }    // real_cal_num loop
    }      // out_plane loop
  }        // out_batch loop
}

void MaxPoolingFp16(const float16_t *input_ptr, float16_t *output_ptr, PoolingParameter *pooling_param, int task_id) {
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
  int c8 = channel / C8NUM;
  int c8_res = channel % C8NUM;
  int c4 = c8_res / C4NUM;
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
        for (int j = 0; j < c8; j++) {
          int in_channel_offset = in_batch_offset + j * C8NUM;
          int out_channel_offset = out_plane_offset + j * C8NUM;
#ifdef ENABLE_NEON
          float16x8_t tmp_max = vdupq_n_f16(-FLT_MAX);
#else
          float16_t tmp_max[8]{-FLT_MAX};
#endif
          for (int h = 0; h < win_h; h++) {
            for (int w = 0; w < win_w; w++) {
              if ((in_h_index + h) < 0 || (in_h_index + h) >= in_h || (in_w_index + w) < 0 ||
                  (in_w_index + w) >= in_w) {
                continue;
              } else {
                int in_offset = in_channel_offset + ((in_h_index + h) * in_w + in_w_index + w) * channel;
#ifdef ENABLE_NEON
                tmp_max = vmaxq_f16(tmp_max, vld1q_f16(input_ptr + in_offset));
#else
                for (int k = 0; k < C8NUM; k++) {
                  tmp_max[k] = fmax(tmp_max[k], *(input_ptr + in_offset + k));
                }
#endif
              }
            }  // win_w loop
          }    // win_h loop
#ifdef ENABLE_NEON
          vst1q_f16(output_ptr + out_channel_offset, tmp_max);
#else
          for (int l = 0; l < C8NUM; ++l) {
            *(output_ptr + out_channel_offset + l) = tmp_max[l];
          }
#endif
        }  // c8 loop

        int c4_offset = c8 * C8NUM;
        for (int j = 0; j < c4; j++) {
          int in_channel_offset = in_batch_offset + c4_offset + j * C4NUM;
          int out_channel_offset = out_plane_offset + c4_offset + j * C4NUM;
#ifdef ENABLE_NEON
          float16x4_t tmp_max = vdup_n_f16(-FLT_MAX);
#else
          float16_t tmp_max[4]{-FLT_MAX};
#endif
          for (int h = 0; h < win_h; h++) {
            for (int w = 0; w < win_w; w++) {
              if ((in_h_index + h) < 0 || (in_h_index + h) >= in_h || (in_w_index + w) < 0 ||
                  (in_w_index + w) >= in_w) {
                continue;
              } else {
                int in_offset = in_channel_offset + ((in_h_index + h) * in_w + in_w_index + w) * channel;
#ifdef ENABLE_NEON
                tmp_max = vmax_f16(tmp_max, vld1_f16(input_ptr + in_offset));
#else
                for (int k = 0; k < C4NUM; k++) {
                  tmp_max[k] = fmax(tmp_max[k], *(input_ptr + in_offset + k));
                }
#endif
              }
            }  // win_w loop
          }    // win_h loop
#ifdef ENABLE_NEON
          vst1_f16(output_ptr + out_channel_offset, tmp_max);
#else
          for (int l = 0; l < C4NUM; ++l) {
            *(output_ptr + out_channel_offset + l) = tmp_max[l];
          }
#endif
        }  // c4 loop

        int channel_s = c8 * C8NUM + c4 * C4NUM;
        for (int k = channel_s; k < channel; k++) {
          int in_channel_offset = in_batch_offset + k;
          int out_channel_offset = out_plane_offset + k;
          float16_t tmp_max = -FLT_MAX;
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
