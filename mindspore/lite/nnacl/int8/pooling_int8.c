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

#include "nnacl/int8/pooling_int8.h"
#include "nnacl/common_func.h"

void AvgPoolingInt8(const int8_t *input_ptr, int8_t *output_ptr, PoolingParameter *pooling_param, int task_id) {
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
  float input_scale = pooling_param->quant_args_[0][0].scale_;
  int input_zp = pooling_param->quant_args_[0][0].zp_;
  float output_scale = pooling_param->quant_args_[1][0].scale_;
  int output_zp = pooling_param->quant_args_[1][0].zp_;
  double real_multiplier = input_scale / output_scale;
  const int8_t out_min = INT8_MIN;
  const int8_t out_max = INT8_MAX;

  for (int batch = 0; batch < output_batch; batch++) {
    int in_batch_offset = batch * in_h * in_w * channel;
    int out_batch_offset = batch * output_h * output_w * channel;
    for (int i = 0; i < out_plane; i++) {
      int out_w_index = i % output_w;
      int out_h_index = i / output_w;
      int in_w_index = out_w_index * stride_w - pad_w;
      int in_h_index = out_h_index * stride_h - pad_h;
      int out_plane_offset = out_batch_offset + i * channel;
      for (int j = 0; j < channel; j++) {
        int in_channel_offset = in_batch_offset + j;
        int out_channel_offset = out_plane_offset + j;
        int16_t tmp_avg = 0;
        int real_count = 0;
        for (int h = 0; h < win_h; h++) {
          for (int w = 0; w < win_w; w++) {
            if ((in_h_index + h) < 0 || (in_h_index + h) >= in_h || (in_w_index + w) < 0 || (in_w_index + w) >= in_w) {
              continue;
            } else {
              int in_offset = in_channel_offset + ((in_h_index + h) * in_w + in_w_index + w) * channel;
              tmp_avg += *(input_ptr + in_offset);
              ++real_count;
            }
          }  // win_w loop
        }    // win_h loop
        int16_t tmp_out = round((float)tmp_avg / (float)real_count);
        tmp_out = (int8_t)(round((tmp_out - input_zp) * real_multiplier) + output_zp);
        int8_t real_out = tmp_out < out_min ? out_min : tmp_out;
        real_out = real_out > out_max ? out_max : real_out;
        *(output_ptr + out_channel_offset) = real_out;
      }  // in_channel loop
    }    // out_plane loop
  }      // out_batch loop
}

void AvgPoolingOptInt8(const int8_t *input_ptr, int8_t *output_ptr, PoolingParameter *pooling_param, int task_id) {
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
  int c8 = UP_DIV(channel, C8NUM);
  const int8_t out_min = INT8_MIN;
  const int8_t out_max = INT8_MAX;

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
        for (int j = 0; j < c8 - 1; j++) {
          int in_channel_offset = in_batch_offset + j * C8NUM;
          int out_channel_offset = out_plane_offset + j * C8NUM;
          int16_t tmp_avg1 = 0;
          int16_t tmp_avg2 = 0;
          int16_t tmp_avg3 = 0;
          int16_t tmp_avg4 = 0;
          int16_t tmp_avg5 = 0;
          int16_t tmp_avg6 = 0;
          int16_t tmp_avg7 = 0;
          int16_t tmp_avg8 = 0;
          int real_count = 0;
          for (int h = 0; h < win_h; h++) {
            for (int w = 0; w < win_w; w++) {
              if ((in_h_index + h) < 0 || (in_h_index + h) >= in_h || (in_w_index + w) < 0 ||
                  (in_w_index + w) >= in_w) {
                continue;
              } else {
                int in_offset = in_channel_offset + ((in_h_index + h) * in_w + in_w_index + w) * channel;
                tmp_avg1 += *(input_ptr + in_offset);
                tmp_avg2 += *(input_ptr + in_offset + 1);
                tmp_avg3 += *(input_ptr + in_offset + 2);
                tmp_avg4 += *(input_ptr + in_offset + 3);
                tmp_avg5 += *(input_ptr + in_offset + 4);
                tmp_avg6 += *(input_ptr + in_offset + 5);
                tmp_avg7 += *(input_ptr + in_offset + 6);
                tmp_avg8 += *(input_ptr + in_offset + 7);
                ++real_count;
              }
            }  // win_w loop
          }    // win_h loop
          int16_t tmp_out1 = round((float)tmp_avg1 / (float)real_count);
          int16_t tmp_out2 = round((float)tmp_avg2 / (float)real_count);
          int16_t tmp_out3 = round((float)tmp_avg3 / (float)real_count);
          int16_t tmp_out4 = round((float)tmp_avg4 / (float)real_count);
          int16_t tmp_out5 = round((float)tmp_avg5 / (float)real_count);
          int16_t tmp_out6 = round((float)tmp_avg6 / (float)real_count);
          int16_t tmp_out7 = round((float)tmp_avg7 / (float)real_count);
          int16_t tmp_out8 = round((float)tmp_avg8 / (float)real_count);
          int16_t real_out1 = tmp_out1 < out_min ? out_min : tmp_out1;
          int16_t real_out2 = tmp_out2 < out_min ? out_min : tmp_out2;
          int16_t real_out3 = tmp_out3 < out_min ? out_min : tmp_out3;
          int16_t real_out4 = tmp_out4 < out_min ? out_min : tmp_out4;
          int16_t real_out5 = tmp_out5 < out_min ? out_min : tmp_out5;
          int16_t real_out6 = tmp_out6 < out_min ? out_min : tmp_out6;
          int16_t real_out7 = tmp_out7 < out_min ? out_min : tmp_out7;
          int16_t real_out8 = tmp_out8 < out_min ? out_min : tmp_out8;
          real_out1 = real_out1 > out_max ? out_max : real_out1;
          real_out2 = real_out2 > out_max ? out_max : real_out2;
          real_out3 = real_out3 > out_max ? out_max : real_out3;
          real_out4 = real_out4 > out_max ? out_max : real_out4;
          real_out5 = real_out5 > out_max ? out_max : real_out5;
          real_out6 = real_out6 > out_max ? out_max : real_out6;
          real_out7 = real_out7 > out_max ? out_max : real_out7;
          real_out8 = real_out8 > out_max ? out_max : real_out8;
          *(output_ptr + out_channel_offset) = (int8_t)real_out1;
          *(output_ptr + out_channel_offset + 1) = (int8_t)real_out2;
          *(output_ptr + out_channel_offset + 2) = (int8_t)real_out3;
          *(output_ptr + out_channel_offset + 3) = (int8_t)real_out4;
          *(output_ptr + out_channel_offset + 4) = (int8_t)real_out5;
          *(output_ptr + out_channel_offset + 5) = (int8_t)real_out6;
          *(output_ptr + out_channel_offset + 6) = (int8_t)real_out7;
          *(output_ptr + out_channel_offset + 7) = (int8_t)real_out8;
        }  // in_channel loop
        int channel_s = (c8 - 1) * C8NUM;
        for (int k = channel_s; k < channel; k++) {
          int in_channel_offset = in_batch_offset + k;
          int out_channel_offset = out_plane_offset + k;
          int16_t tmp_avg = 0;
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
          int16_t tmp_out = round((float)tmp_avg / (float)real_count);
          int16_t real_out = tmp_out < out_min ? out_min : tmp_out;
          real_out = real_out > out_max ? out_max : real_out;
          *(output_ptr + out_channel_offset) = (int8_t)real_out;
        }  // channel_res loop
      }    // out_plane loop
    }      // out_batch loop
  }
}

void MaxPoolingInt8(const int8_t *input_ptr, int8_t *output_ptr, PoolingParameter *pooling_param, int task_id) {
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
  // input channel is equal to output channel
  float input_scale = pooling_param->quant_args_[0][0].scale_;
  int input_zp = pooling_param->quant_args_[0][0].zp_;
  float output_scale = pooling_param->quant_args_[1][0].scale_;
  int output_zp = pooling_param->quant_args_[1][0].zp_;
  double real_multiplier = input_scale / output_scale;

  for (int batch = 0; batch < output_batch; batch++) {
    int in_batch_offset = batch * in_h * in_w * channel;
    int out_batch_offset = batch * output_h * output_w * channel;
    for (int i = 0; i < out_plane; i++) {
      int out_w_index = i % output_w;
      int out_h_index = i / output_w;
      int in_w_index = out_w_index * stride_w - pad_w;
      int in_h_index = out_h_index * stride_h - pad_h;
      int out_plane_offset = out_batch_offset + i * channel;
      for (int j = 0; j < channel; j++) {
        int in_channel_offset = in_batch_offset + j;
        int out_channel_offset = out_plane_offset + j;
        int8_t tmp_max = INT8_MIN;
        for (int h = 0; h < win_h; h++) {
          for (int w = 0; w < win_w; w++) {
            if ((in_h_index + h) < 0 || (in_h_index + h) >= in_h || (in_w_index + w) < 0 || (in_w_index + w) >= in_w) {
              continue;
            } else {
              int in_offset = in_channel_offset + ((in_h_index + h) * in_w + in_w_index + w) * channel;
              tmp_max = MaxInt8(tmp_max, *(input_ptr + in_offset));
            }
          }  // win_w loop
        }    // win_h loop
        *(output_ptr + out_channel_offset) = (int8_t)(round((tmp_max - input_zp) * real_multiplier) + output_zp);
      }  // in_channel loop
    }    // out_plane loop
  }      // out_batch loop
}

void MaxPoolingOptInt8(const int8_t *input_ptr, int8_t *output_ptr, PoolingParameter *pooling_param, int task_id) {
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
  int c16 = UP_DIV(channel, 16);

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
        for (int j = 0; j < c16 - 1; j++) {
          int in_channel_offset = in_batch_offset + j * 16;
          int out_channel_offset = out_plane_offset + j * 16;
#ifdef ENABLE_NEON
          int8x16_t tmp_max = vdupq_n_s8(INT8_MIN);
#else
          int8_t tmp_max1 = INT8_MIN;
          int8_t tmp_max2 = INT8_MIN;
          int8_t tmp_max3 = INT8_MIN;
          int8_t tmp_max4 = INT8_MIN;
          int8_t tmp_max5 = INT8_MIN;
          int8_t tmp_max6 = INT8_MIN;
          int8_t tmp_max7 = INT8_MIN;
          int8_t tmp_max8 = INT8_MIN;
          int8_t tmp_max9 = INT8_MIN;
          int8_t tmp_max10 = INT8_MIN;
          int8_t tmp_max11 = INT8_MIN;
          int8_t tmp_max12 = INT8_MIN;
          int8_t tmp_max13 = INT8_MIN;
          int8_t tmp_max14 = INT8_MIN;
          int8_t tmp_max15 = INT8_MIN;
          int8_t tmp_max16 = INT8_MIN;
#endif
          for (int h = 0; h < win_h; h++) {
            for (int w = 0; w < win_w; w++) {
              if ((in_h_index + h) < 0 || (in_h_index + h) >= in_h || (in_w_index + w) < 0 ||
                  (in_w_index + w) >= in_w) {
                continue;
              } else {
                int in_offset = in_channel_offset + ((in_h_index + h) * in_w + in_w_index + w) * channel;
#ifdef ENABLE_NEON
                tmp_max = vmaxq_s8(tmp_max, vld1q_s8(input_ptr + in_offset));
#else
                tmp_max1 = MaxInt8(tmp_max1, *(input_ptr + in_offset));
                tmp_max2 = MaxInt8(tmp_max2, *(input_ptr + in_offset + 1));
                tmp_max3 = MaxInt8(tmp_max3, *(input_ptr + in_offset + 2));
                tmp_max4 = MaxInt8(tmp_max4, *(input_ptr + in_offset + 3));
                tmp_max5 = MaxInt8(tmp_max5, *(input_ptr + in_offset + 4));
                tmp_max6 = MaxInt8(tmp_max6, *(input_ptr + in_offset + 5));
                tmp_max7 = MaxInt8(tmp_max7, *(input_ptr + in_offset + 6));
                tmp_max8 = MaxInt8(tmp_max8, *(input_ptr + in_offset + 7));
                tmp_max9 = MaxInt8(tmp_max9, *(input_ptr + in_offset + 8));
                tmp_max10 = MaxInt8(tmp_max10, *(input_ptr + in_offset + 9));
                tmp_max11 = MaxInt8(tmp_max11, *(input_ptr + in_offset + 10));
                tmp_max12 = MaxInt8(tmp_max12, *(input_ptr + in_offset + 11));
                tmp_max13 = MaxInt8(tmp_max13, *(input_ptr + in_offset + 12));
                tmp_max14 = MaxInt8(tmp_max14, *(input_ptr + in_offset + 13));
                tmp_max15 = MaxInt8(tmp_max15, *(input_ptr + in_offset + 14));
                tmp_max16 = MaxInt8(tmp_max16, *(input_ptr + in_offset + 15));
#endif
              }
            }  // win_w loop
          }    // win_h loop
#ifdef ENABLE_NEON
          vst1q_s8(output_ptr + out_channel_offset, tmp_max);
#else
          *(output_ptr + out_channel_offset) = tmp_max1;
          *(output_ptr + out_channel_offset + 1) = tmp_max2;
          *(output_ptr + out_channel_offset + 2) = tmp_max3;
          *(output_ptr + out_channel_offset + 3) = tmp_max4;
          *(output_ptr + out_channel_offset + 4) = tmp_max5;
          *(output_ptr + out_channel_offset + 5) = tmp_max6;
          *(output_ptr + out_channel_offset + 6) = tmp_max7;
          *(output_ptr + out_channel_offset + 7) = tmp_max8;
          *(output_ptr + out_channel_offset + 8) = tmp_max9;
          *(output_ptr + out_channel_offset + 9) = tmp_max10;
          *(output_ptr + out_channel_offset + 10) = tmp_max11;
          *(output_ptr + out_channel_offset + 11) = tmp_max12;
          *(output_ptr + out_channel_offset + 12) = tmp_max13;
          *(output_ptr + out_channel_offset + 13) = tmp_max14;
          *(output_ptr + out_channel_offset + 14) = tmp_max15;
          *(output_ptr + out_channel_offset + 15) = tmp_max16;
#endif
        }  // in_channel loop
        int channel_s = (c16 - 1) * 16;
        for (int k = channel_s; k < channel; k++) {
          int in_channel_offset = in_batch_offset + k;
          int out_channel_offset = out_plane_offset + k;
          int8_t tmp_max = INT8_MIN;
          for (int h = 0; h < win_h; h++) {
            for (int w = 0; w < win_w; w++) {
              if ((in_h_index + h) < 0 || (in_h_index + h) >= in_h || (in_w_index + w) < 0 ||
                  (in_w_index + w) >= in_w) {
                continue;
              } else {
                int in_offset = in_channel_offset + ((in_h_index + h) * in_w + in_w_index + w) * channel;
                tmp_max = MaxInt8(tmp_max, *(input_ptr + in_offset));
              }
            }  // win_w loop
          }    // win_h loop
          *(output_ptr + out_channel_offset) = tmp_max;
        }  // channel_res loop
      }    // out_plane loop
    }      // out_batch loop
  }
}
