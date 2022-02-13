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
#include "nnacl/errorcode.h"

int AvgPoolingInt8(const int8_t *input_ptr, int8_t *output_ptr, const PoolingParameter *pooling_param, int task_id) {
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
        if (real_count == 0) {
          return NNACL_ERR;
        }
        int16_t tmp_out = round((float)tmp_avg / (float)real_count);
        tmp_out = (int8_t)(round((tmp_out - input_zp) * real_multiplier) + output_zp);
        int8_t real_out = tmp_out < out_min ? out_min : tmp_out;
        real_out = real_out > out_max ? out_max : real_out;
        *(output_ptr + out_channel_offset) = real_out;
      }  // in_channel loop
    }    // out_plane loop
  }      // out_batch loop
  return NNACL_OK;
}

int AvgPoolingOptInt8(const int8_t *input_ptr, int8_t *output_ptr, const PoolingParameter *pooling_param, int task_id) {
  int win_w = pooling_param->window_w_;
  int win_h = pooling_param->window_h_;
  int channel = pooling_param->input_channel_;
  int c16 = channel / C16NUM;
  int in_w = pooling_param->input_w_;
  int output_w = pooling_param->output_w_;
  int out_plane = output_w * pooling_param->output_h_;
  int out_tile_count = UP_DIV(out_plane, TILE_NUM);
  int thread_num = out_tile_count < pooling_param->thread_num_ ? out_tile_count : pooling_param->thread_num_;
  int input_zp = pooling_param->quant_args_[0][0].zp_;
  int output_zp = pooling_param->quant_args_[1][0].zp_;
  double real_multiplier = pooling_param->quant_args_[0][0].scale_ / pooling_param->quant_args_[1][0].scale_;
  const int8_t out_min = INT8_MIN;
  const int8_t out_max = INT8_MAX;
  NNACL_CHECK_ZERO_RETURN_ERR(output_w);
  for (int batch = 0; batch < pooling_param->output_batch_; batch++) {
    int in_batch_offset = batch * pooling_param->input_h_ * in_w * channel;
    int out_batch_offset = batch * pooling_param->output_h_ * output_w * channel;
    for (int thread_id = task_id; thread_id < out_tile_count; thread_id += thread_num) {
      int cal_start_index = thread_id * TILE_NUM;
      int real_cal_num = (out_plane - cal_start_index) > TILE_NUM ? TILE_NUM : (out_plane - cal_start_index);
      for (int i = 0; i < real_cal_num; i++) {
        int index = cal_start_index + i;
        int out_w_index = index % output_w;
        int out_h_index = index / output_w;
        int in_w_index = out_w_index * pooling_param->stride_w_ - pooling_param->pad_l_;
        int in_h_index = out_h_index * pooling_param->stride_h_ - pooling_param->pad_u_;
        int out_plane_offset = out_batch_offset + index * channel;
        int input_stride = (in_h_index * in_w + in_w_index) * channel;
        int kw_s = MSMAX(0, -in_w_index);
        int kw_e = MSMIN(win_w, in_w - in_w_index);
        int kh_s = MSMAX(0, -in_h_index);
        int kh_e = MSMIN(win_h, pooling_param->input_h_ - in_h_index);
        int real_count = (kw_e - kw_s) * (kh_e - kh_s);
        if (real_count == 0) {
          return NNACL_ERR;
        }

        // 16 channels
        for (int j = 0; j < c16; j++) {
#ifdef ENABLE_NEON
          int16x8_t tmp_avg[2];
          tmp_avg[0] = vmovq_n_s16(0);
          tmp_avg[1] = vmovq_n_s16(0);
#else
          int16_t tmp_avg[16];
          int16_t real_out[16];
          for (int m = 0; m < C16NUM; ++m) {
            tmp_avg[m] = 0;
          }
#endif
          int in_channel_offset = in_batch_offset + j * C16NUM;
          int out_channel_offset = out_plane_offset + j * C16NUM;

          for (int h = kh_s; h < kh_e; h++) {
            for (int w = kw_s; w < kw_e; w++) {
              int in_offset = in_channel_offset + input_stride + (h * in_w + w) * channel;
#ifdef ENABLE_NEON
              int8x16_t in_ptr = vld1q_s8(input_ptr + in_offset);
              int8x8_t in_data1 = vget_low_s8(in_ptr);
              int8x8_t in_data2 = vget_high_s8(in_ptr);
              int16x8_t data1 = vmovl_s8(in_data1);
              int16x8_t data2 = vmovl_s8(in_data2);
              tmp_avg[0] = vaddq_s16(tmp_avg[0], data1);
              tmp_avg[1] = vaddq_s16(tmp_avg[1], data2);
#else
              for (int k = 0; k < C16NUM; ++k) {
                tmp_avg[k] += input_ptr[in_offset + k];
              }
#endif
            }  // win_w loop
          }    // win_h loop
#ifdef ENABLE_NEON
          int16_t tmp_data[8];
          int16_t tmp_out[8];
          int16_t tmp_data1[8];
          int16_t tmp_out1[8];
          for (int l = 0; l < C8NUM; l++) {
            tmp_data[l] = tmp_avg[0][l] + 128 * real_count;
            tmp_out[l] = (tmp_data[l] + real_count / 2) / real_count;
            tmp_out[l] -= 128;
            tmp_out[l] = round((tmp_out[l] - input_zp) * real_multiplier) + output_zp;
          }
          for (int l = 0; l < C8NUM; l++) {
            tmp_data1[l] = tmp_avg[1][l] + 128 * real_count;
            tmp_out1[l] = (tmp_data1[l] + real_count / 2) / real_count;
            tmp_out1[l] -= 128;
            tmp_out1[l] = round((tmp_out1[l] - input_zp) * real_multiplier) + output_zp;
          }
          int8x8_t real_out[2];
          int8x8_t output_min = vdup_n_s8(out_min);
          int8x8_t output_max = vdup_n_s8(out_max);
          real_out[0] = vqmovn_s16(vld1q_s16(tmp_out));
          real_out[0] = vmin_s8(real_out[0], output_max);
          real_out[0] = vmax_s8(real_out[0], output_min);
          vst1_s8(output_ptr + out_channel_offset, real_out[0]);
          real_out[1] = vqmovn_s16(vld1q_s16(tmp_out1));
          real_out[1] = vmin_s8(real_out[1], output_max);
          real_out[1] = vmax_s8(real_out[1], output_min);
          vst1_s8(output_ptr + out_channel_offset + 8, real_out[1]);
#else
          for (int l = 0; l < C16NUM; ++l) {
            int16_t tmp_data = tmp_avg[l] + 128 * real_count;
            real_out[l] = (tmp_data + real_count / 2) / real_count - 128;
            real_out[l] = (int8_t)(round((real_out[l] - input_zp) * real_multiplier) + output_zp);
            real_out[l] = real_out[l] < out_min ? out_min : real_out[l];
            real_out[l] = real_out[l] > out_max ? out_max : real_out[l];
            *(output_ptr + out_channel_offset + l) = (int8_t)real_out[l];
          }
#endif
        }

        // 8 channels
        int channel_16_res = channel - c16 * C16NUM;
        int c8 = channel_16_res / C8NUM;
        int in_c16_offset = in_batch_offset + c16 * C16NUM;
        int out_c16_offset = out_plane_offset + c16 * C16NUM;
        for (int j = 0; j < c8; j++) {
#ifdef ENABLE_NEON
          int16x8_t tmp_avg = vmovq_n_s16(0);
#else
          int16_t tmp_avg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
          int16_t real_out[8];
#endif
          int in_channel_offset = in_c16_offset + j * C8NUM;
          int out_channel_offset = out_c16_offset + j * C8NUM;
          for (int h = kh_s; h < kh_e; h++) {
            for (int w = kw_s; w < kw_e; w++) {
              int in_offset = in_channel_offset + input_stride + (h * in_w + w) * channel;
#ifdef ENABLE_NEON
              int8x8_t in_ptr = vld1_s8(input_ptr + in_offset);
              int16x8_t data = vmovl_s8(in_ptr);
              tmp_avg = vaddq_s16(tmp_avg, data);
#else
              for (int k = 0; k < C8NUM; ++k) {
                tmp_avg[k] += input_ptr[in_offset + k];
              }
#endif
            }  // win_w loop
          }    // win_h loop
#ifdef ENABLE_NEON
          int16_t tmp_data[8];
          int16_t tmp_out[8];
          for (int l = 0; l < C8NUM; l++) {
            tmp_data[l] = tmp_avg[l] + 128 * real_count;
            tmp_out[l] = (tmp_data[l] + real_count / 2) / real_count;
            tmp_out[l] -= 128;
            tmp_out[l] = round((tmp_out[l] - input_zp) * real_multiplier) + output_zp;
          }
          int8x8_t real_out;
          int8x8_t output_min = vdup_n_s8(out_min);
          int8x8_t output_max = vdup_n_s8(out_max);
          real_out = vqmovn_s16(vld1q_s16(tmp_out));
          real_out = vmin_s8(real_out, output_max);
          real_out = vmax_s8(real_out, output_min);
          vst1_s8(output_ptr + out_channel_offset, real_out);
#else
          for (int l = 0; l < C8NUM; ++l) {
            int16_t tmp_data = tmp_avg[l] + 128 * real_count;
            real_out[l] = (tmp_data + real_count / 2) / real_count - 128;
            real_out[l] = (int8_t)(round((real_out[l] - input_zp) * real_multiplier) + output_zp);
            real_out[l] = real_out[l] < out_min ? out_min : real_out[l];
            real_out[l] = real_out[l] > out_max ? out_max : real_out[l];
            *(output_ptr + out_channel_offset + l) = (int8_t)real_out[l];
          }
#endif
        }

        // less than 8 channel
        int channel_8_res = channel_16_res - c8 * C8NUM;
        int in_c8_offset = in_c16_offset + c8 * C8NUM;
        int out_c8_offset = out_c16_offset + c8 * C8NUM;
        for (int k = 0; k < channel_8_res; k++) {
          int in_channel_offset = in_c8_offset + k;
          int out_channel_offset = out_c8_offset + k;
          int16_t tmp_avg = 0;
          for (int h = kh_s; h < kh_e; h++) {
            for (int w = kw_s; w < kw_e; w++) {
              int in_offset = in_channel_offset + input_stride + (h * in_w + w) * channel;
              tmp_avg += input_ptr[in_offset];
            }  // win_w loop
          }    // win_h loop
          int16_t tmp_out = round((float)tmp_avg / (float)real_count + 128) - 128;
          tmp_out = (int8_t)(round((tmp_out - input_zp) * real_multiplier) + output_zp);
          int16_t real_out = tmp_out < out_min ? out_min : tmp_out;
          real_out = real_out > out_max ? out_max : real_out;
          *(output_ptr + out_channel_offset) = (int8_t)real_out;
        }  // channel_res loop
      }    // out_plane loop
    }      // out_batch loop
  }
  return NNACL_OK;
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

void MaxPoolingWithQuantInt8(const int8_t *input_ptr, int8_t *output_ptr, PoolingParameter *pooling_param,
                             int task_id) {
  int channel = pooling_param->input_channel_;
  int in_w = pooling_param->input_w_;
  int in_h = pooling_param->input_h_;
  int output_w = pooling_param->output_w_;
  int out_plane = output_w * pooling_param->output_h_;
  int out_tile_count = UP_DIV(out_plane, TILE_NUM);
  int thread_num = out_tile_count < pooling_param->thread_num_ ? out_tile_count : pooling_param->thread_num_;
  int c16 = UP_DIV(channel, 16);
  // input channel is equal to output channel
  float input_scale = pooling_param->quant_args_[0][0].scale_;
  int input_zp = pooling_param->quant_args_[0][0].zp_;
  float output_scale = pooling_param->quant_args_[1][0].scale_;
  int output_zp = pooling_param->quant_args_[1][0].zp_;
  double real_multiplier = input_scale / output_scale;

  NNACL_CHECK_ZERO_RETURN(output_w);
  for (int batch = 0; batch < pooling_param->output_batch_; batch++) {
    int in_batch_offset = batch * in_h * in_w * channel;
    int out_batch_offset = batch * pooling_param->output_h_ * output_w * channel;
    for (int thread_id = task_id; thread_id < out_tile_count; thread_id += thread_num) {
      int cal_start_index = thread_id * TILE_NUM;
      int real_cal_num = (out_plane - cal_start_index) > TILE_NUM ? TILE_NUM : (out_plane - cal_start_index);
      for (int i = 0; i < real_cal_num; i++) {
        int index = cal_start_index + i;
        int out_w_index = index % output_w;
        int out_h_index = index / output_w;
        int in_w_index = out_w_index * pooling_param->stride_w_ - pooling_param->pad_l_;
        int in_h_index = out_h_index * pooling_param->stride_h_ - pooling_param->pad_u_;
        int out_plane_offset = out_batch_offset + index * channel;
        for (int j = 0; j < c16 - 1; j++) {
          int in_channel_offset = in_batch_offset + j * 16;
          int out_channel_offset = out_plane_offset + j * 16;
#ifdef ENABLE_NEON
          int8x16_t tmp_max = vdupq_n_s8(INT8_MIN);
#else
          int8_t tmp_max[16];
          for (int m = 0; m < C16NUM; ++m) {
            tmp_max[m] = INT8_MIN;
          }
#endif
          for (int h = 0; h < pooling_param->window_h_; h++) {
            for (int w = 0; w < pooling_param->window_w_; w++) {
              if ((in_h_index + h) < 0 || (in_h_index + h) >= in_h || (in_w_index + w) < 0 ||
                  (in_w_index + w) >= in_w) {
                continue;
              } else {
                int in_offset = in_channel_offset + ((in_h_index + h) * in_w + in_w_index + w) * channel;
#ifdef ENABLE_NEON
                tmp_max = vmaxq_s8(tmp_max, vld1q_s8(input_ptr + in_offset));
#else
                for (int k = 0; k < C16NUM; ++k) {
                  tmp_max[k] = MaxInt8(tmp_max[k], *(input_ptr + in_offset + k));
                }
#endif
              }
            }  // win_w loop
          }    // win_h loop
#ifdef ENABLE_NEON
          for (int l = 0; l < C16NUM; ++l) {
            tmp_max[l] = (int8_t)(round((tmp_max[l] - input_zp) * real_multiplier) + output_zp);
          }
          vst1q_s8(output_ptr + out_channel_offset, tmp_max);
#else
          for (int l = 0; l < C16NUM; ++l) {
            *(output_ptr + out_channel_offset + l) =
              (int8_t)(round((tmp_max[l] - input_zp) * real_multiplier) + output_zp);
          }
#endif
        }  // in_channel loop

        // res channel
        int channel_s = (c16 - 1) * 16;
        for (int k = channel_s; k < channel; k++) {
          int in_channel_offset = in_batch_offset + k;
          int out_channel_offset = out_plane_offset + k;
          int8_t tmp_max = INT8_MIN;
          for (int h = 0; h < pooling_param->window_h_; h++) {
            for (int w = 0; w < pooling_param->window_w_; w++) {
              if ((in_h_index + h) < 0 || (in_h_index + h) >= in_h || (in_w_index + w) < 0 ||
                  (in_w_index + w) >= in_w) {
                continue;
              } else {
                int in_offset = in_channel_offset + ((in_h_index + h) * in_w + in_w_index + w) * channel;
                tmp_max = MaxInt8(tmp_max, *(input_ptr + in_offset));
              }
            }  // win_w loop
          }    // win_h loop
          *(output_ptr + out_channel_offset) = (int8_t)(round((tmp_max - input_zp) * real_multiplier) + output_zp);
        }  // channel_res loop
      }    // out_plane loop
    }      // out_batch loop
  }
}

void MaxPoolingOptInt8(const int8_t *input_ptr, int8_t *output_ptr, PoolingParameter *pooling_param, int task_id) {
  int channel = pooling_param->input_channel_;
  int in_w = pooling_param->input_w_;
  int output_w = pooling_param->output_w_;
  int out_plane = output_w * pooling_param->output_h_;
  int out_tile_count = UP_DIV(out_plane, TILE_NUM);
  int thread_num = MSMIN(out_tile_count, pooling_param->thread_num_);
  int8_t out_array[MAX_MAXPOOL_SIZE];

  NNACL_CHECK_ZERO_RETURN(output_w);
  for (int batch = 0; batch < pooling_param->output_batch_; batch++) {
    int in_batch_offset = batch * pooling_param->input_h_ * in_w * channel;
    int out_batch_offset = batch * pooling_param->output_h_ * output_w * channel;
    for (int thread_id = task_id; thread_id < out_tile_count; thread_id += thread_num) {
      int cal_start_index = thread_id * TILE_NUM;
      int real_cal_num = out_plane - cal_start_index;
      real_cal_num = MSMIN(real_cal_num, TILE_NUM);
      for (int i = 0; i < real_cal_num; i++) {
        int index = cal_start_index + i;
        int out_w_index = index % output_w;
        int out_h_index = index / output_w;
        int in_w_index = out_w_index * pooling_param->stride_w_ - pooling_param->pad_l_;
        int in_h_index = out_h_index * pooling_param->stride_h_ - pooling_param->pad_u_;
        const int ky_s = 0 > (-in_h_index) ? 0 : (-in_h_index);
        int ky_e = MSMIN(pooling_param->window_h_, pooling_param->input_h_ - in_h_index);
        const int kx_s = 0 > (-in_w_index) ? 0 : (-in_w_index);
        int kx_e = MSMIN(pooling_param->window_w_, in_w - in_w_index);
        int input_stride = (in_h_index * in_w + in_w_index) * channel + in_batch_offset;
        int out_plane_offset = out_batch_offset + index * channel;

        int c = 0;
        for (; c < channel; c += MAX_MAXPOOL_SIZE) {
          int real_channel = channel - c;
          real_channel = MSMIN(real_channel, MAX_MAXPOOL_SIZE);
          memset(out_array, INT8_MIN, real_channel);
          int8_t *out_data = output_ptr + out_plane_offset + c;
          for (int h = ky_s; h < ky_e; ++h) {
            int in_h_offset = input_stride + h * in_w * channel + c;
            for (int w = kx_s; w < kx_e; ++w) {
              const int8_t *in_data = input_ptr + in_h_offset + w * channel;
              int j = 0;
#ifdef ENABLE_NEON
              const int8_t *tmp_in_data = in_data;
              int c16 = real_channel / 16 * 16;
              int c8 = real_channel / 8 * 8;
              for (; j < c16; j += 16) {
                int8x16_t ori_in = vld1q_s8(tmp_in_data);
                int8x16_t out_array16 = vld1q_s8(out_array + j);
                tmp_in_data += 16;
                out_array16 = vmaxq_s8(ori_in, out_array16);
                vst1q_s8(out_array + j, out_array16);
              }  // 16 channel loop

              for (; j < c8; j += 8) {
                int8x8_t ori_in = vld1_s8(tmp_in_data);
                int8x8_t out_array8 = vld1_s8(out_array + j);
                tmp_in_data += 8;
                out_array8 = vmax_s8(ori_in, out_array8);
                vst1_s8(out_array + j, out_array8);
              }  // 8 channel loop
#endif
              for (; j < real_channel; ++j) {
                out_array[j] = out_array[j] > in_data[j] ? out_array[j] : in_data[j];
              }
            }  // kw loop
          }    // kh loop

          int j = 0;
#ifdef ENABLE_NEON
          int c16 = real_channel / 16 * 16;
          int c8 = real_channel / 8 * 8;
          int8_t *tmp_out_data = out_data;
          for (; j < c16; j += 16) {
            vst1q_s8(tmp_out_data, vld1q_s8(out_array + j));
            tmp_out_data += 16;
          }  // 16 channel loop

          for (; j < c8; j += 8) {
            vst1_s8(tmp_out_data, vld1_s8(out_array + j));
            tmp_out_data += 8;
          }  // 8 channel loop
#endif
          for (; j < real_channel; ++j) {
            out_data[j] = out_array[j];
          }
        }  // 256 channel loop
      }    // out_plane loop
    }      // out_batch loop
  }
}
