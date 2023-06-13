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
#include "nnacl/errorcode.h"

int AvgPoolingFp16(const float16_t *input_ptr, float16_t *output_ptr, const PoolingParameter *pooling_param,
                   const PoolingComputeParam *pooling_args, int task_id, int thread_num) {
  float16_t min = (float16_t)pooling_args->minf;
  float16_t max = (float16_t)pooling_args->maxf;

  int win_w = pooling_args->window_w_;
  int win_h = pooling_args->window_h_;
  int channel = pooling_args->input_channel_;
  int c8 = channel / C8NUM;
  int in_w = pooling_args->input_w_;
  int in_h = pooling_args->input_h_;
  int output_w = pooling_args->output_w_;
  int output_h = pooling_args->output_h_;
  int out_plane = output_w * output_h;
  int out_tile_count = UP_DIV(out_plane, TILE_NUM);

#ifdef ENABLE_NEON
  MS_FLOAT16X8 min_value = MS_MOVQ_F16(min);
  MS_FLOAT16X8 max_value = MS_MOVQ_F16(max);
#endif

  NNACL_CHECK_ZERO_RETURN_ERR(output_w);
  for (int batch = 0; batch < pooling_args->output_batch_; batch++) {
    const float16_t *src_b_ptr = input_ptr + batch * in_h * in_w * channel;
    float16_t *dst_b_ptr = output_ptr + batch * output_h * output_w * channel;
    for (int thread_id = task_id; thread_id < out_tile_count; thread_id += thread_num) {
      int cal_start_index = thread_id * TILE_NUM;
      int real_cal_num = (out_plane - cal_start_index) > TILE_NUM ? TILE_NUM : (out_plane - cal_start_index);
      for (int i = 0; i < real_cal_num; i++) {
        int index = cal_start_index + i;
        int out_w_index = index % output_w;
        int out_h_index = index / output_w;
        int in_w_index = out_w_index * pooling_param->stride_w_ - pooling_param->pad_l_;
        int in_h_index = out_h_index * pooling_param->stride_h_ - pooling_param->pad_u_;

        const float16_t *src_plane_ptr = src_b_ptr;
        float16_t *dst_plane_ptr = dst_b_ptr + index * channel;

        int real_win_h_start = MSMAX(0, -in_h_index);
        int real_win_h_end = MSMIN(win_h, in_h - in_h_index);
        int real_win_w_start = MSMAX(0, -in_w_index);
        int real_win_w_end = MSMIN(win_w, in_w - in_w_index);

        for (int ci = 0; ci < c8; ci++) {
          const float16_t *src_c_ptr = src_plane_ptr + ci * C8NUM;
          float16_t *dst_c_ptr = dst_plane_ptr + ci * C8NUM;
#ifdef ENABLE_NEON
          MS_FLOAT16X8 tmp_avg = MS_MOVQ_F16(0);
#else
          float16_t tmp_avg0 = 0;
          float16_t tmp_avg1 = 0;
          float16_t tmp_avg2 = 0;
          float16_t tmp_avg3 = 0;
          float16_t tmp_avg4 = 0;
          float16_t tmp_avg5 = 0;
          float16_t tmp_avg6 = 0;
          float16_t tmp_avg7 = 0;
#endif
          int real_count = 0;
          for (int h = real_win_h_start; h < real_win_h_end; h++) {
            for (int w = real_win_w_start; w < real_win_w_end; w++) {
              const float16_t *src_win_ptr = src_c_ptr + ((in_h_index + h) * in_w + in_w_index + w) * channel;
#ifdef ENABLE_NEON
              tmp_avg = MS_ADDQ_F16(tmp_avg, MS_LDQ_F16(src_win_ptr));
#else
              tmp_avg0 += src_win_ptr[0];
              tmp_avg1 += src_win_ptr[1];
              tmp_avg2 += src_win_ptr[2];
              tmp_avg3 += src_win_ptr[3];
              tmp_avg4 += src_win_ptr[4];
              tmp_avg5 += src_win_ptr[5];
              tmp_avg6 += src_win_ptr[6];
              tmp_avg7 += src_win_ptr[7];
#endif
              ++real_count;
            }
          }
          if (real_count == 0) {
            return NNACL_ERR;
          }
#ifdef ENABLE_NEON
          tmp_avg = MS_DIVQ_F16(tmp_avg, MS_MOVQ_F16((float16_t)real_count));
          MS_STQ_F16(dst_c_ptr, MS_MINQ_F16(MS_MAXQ_F16(tmp_avg, min_value), max_value));
#else
          dst_c_ptr[0] = MSMIN(MSMAX(tmp_avg0 / (float16_t)real_count, min), max);
          dst_c_ptr[1] = MSMIN(MSMAX(tmp_avg1 / (float16_t)real_count, min), max);
          dst_c_ptr[2] = MSMIN(MSMAX(tmp_avg2 / (float16_t)real_count, min), max);
          dst_c_ptr[3] = MSMIN(MSMAX(tmp_avg3 / (float16_t)real_count, min), max);
          dst_c_ptr[4] = MSMIN(MSMAX(tmp_avg4 / (float16_t)real_count, min), max);
          dst_c_ptr[5] = MSMIN(MSMAX(tmp_avg5 / (float16_t)real_count, min), max);
          dst_c_ptr[6] = MSMIN(MSMAX(tmp_avg6 / (float16_t)real_count, min), max);
          dst_c_ptr[7] = MSMIN(MSMAX(tmp_avg7 / (float16_t)real_count, min), max);
#endif
        }  // c8 loop
        int channel_s = c8 * C8NUM;
        for (int ci = channel_s; ci < channel; ci++) {
          const float16_t *src_c_ptr = src_plane_ptr + ci;
          float16_t *dst_c_ptr = dst_plane_ptr + ci;
          float16_t tmp_avg = 0;
          int real_count = 0;
          for (int h = real_win_h_start; h < real_win_h_end; h++) {
            for (int w = real_win_w_start; w < real_win_w_end; w++) {
              const float16_t *src_win_ptr = src_c_ptr + ((in_h_index + h) * in_w + in_w_index + w) * channel;
              tmp_avg += src_win_ptr[0];
              ++real_count;
            }
          }
          if (real_count == 0) {
            return NNACL_ERR;
          }
          tmp_avg = tmp_avg / (float16_t)real_count;
          tmp_avg = fmax(tmp_avg, min);
          tmp_avg = fmin(tmp_avg, max);
          dst_c_ptr[0] = tmp_avg;
        }  // channel_res loop
      }    // real_cal_num loop
    }      // out_plane loop
  }        // out_batch loop
  return NNACL_OK;
}

void MaxPoolingC8Fp16(const float16_t *input_ptr, float16_t *output_ptr, const PoolingComputeParam *pooling_args,
                      float16_t min, float16_t max, int in_batch_offset, int out_plane_offset, int real_win_h_start,
                      int real_win_h_end, int real_win_w_start, int real_win_w_end, int in_h_index, int in_w_index) {
  int channel = pooling_args->input_channel_;
  int in_w = pooling_args->input_w_;
  int c8 = channel / C8NUM;
#ifdef ENABLE_NEON
  float16x8_t min_value = vdupq_n_f16(min);
  float16x8_t max_value = vdupq_n_f16(max);
#endif
  for (int j = 0; j < c8; j++) {
    int in_channel_offset = in_batch_offset + j * C8NUM;
    int out_channel_offset = out_plane_offset + j * C8NUM;
#ifdef ENABLE_NEON
    float16x8_t tmp_max = vdupq_n_f16(min);
#else
    float16_t tmp_max[8] = {min, min, min, min, min, min, min, min};
#endif
    for (int h = real_win_h_start; h < real_win_h_end; h++) {
      for (int w = real_win_w_start; w < real_win_w_end; w++) {
        int in_offset = in_channel_offset + ((in_h_index + h) * in_w + in_w_index + w) * channel;
#ifdef ENABLE_NEON
        tmp_max = vmaxq_f16(tmp_max, vld1q_f16(input_ptr + in_offset));
#else
        for (int k = 0; k < C8NUM; k++) {
          tmp_max[k] = fmax(tmp_max[k], *(input_ptr + in_offset + k));
        }
#endif
      }  // win_w loop
    }    // win_h loop
#ifdef ENABLE_NEON
    tmp_max = vmaxq_f16(tmp_max, min_value);
    tmp_max = vminq_f16(tmp_max, max_value);
    vst1q_f16(output_ptr + out_channel_offset, tmp_max);
#else
    for (int l = 0; l < C8NUM; ++l) {
      tmp_max[l] = fmax(tmp_max[l], min);
      tmp_max[l] = fmin(tmp_max[l], max);
      *(output_ptr + out_channel_offset + l) = tmp_max[l];
    }
#endif
  }  // c8 loop
}

void MaxPoolingC4Fp16(const float16_t *input_ptr, float16_t *output_ptr, const PoolingComputeParam *pooling_args,
                      float16_t min, float16_t max, int in_batch_offset, int out_plane_offset, int real_win_h_start,
                      int real_win_h_end, int real_win_w_start, int real_win_w_end, int in_h_index, int in_w_index) {
  int channel = pooling_args->input_channel_;
  int in_w = pooling_args->input_w_;
  int c8 = channel / C8NUM;
  int c8_res = channel % C8NUM;
  int c4 = c8_res / C4NUM;
#ifdef ENABLE_NEON
  float16x4_t min_value2 = vdup_n_f16(min);
  float16x4_t max_value2 = vdup_n_f16(max);
#endif
  int c4_offset = c8 * C8NUM;
  for (int j = 0; j < c4; j++) {
    int in_channel_offset = in_batch_offset + c4_offset + j * C4NUM;
    int out_channel_offset = out_plane_offset + c4_offset + j * C4NUM;
#ifdef ENABLE_NEON
    float16x4_t tmp_max = vdup_n_f16(min);
#else
    float16_t tmp_max[4] = {min, min, min, min};
#endif
    for (int h = real_win_h_start; h < real_win_h_end; h++) {
      for (int w = real_win_w_start; w < real_win_w_end; w++) {
        int in_offset = in_channel_offset + ((in_h_index + h) * in_w + in_w_index + w) * channel;
#ifdef ENABLE_NEON
        tmp_max = vmax_f16(tmp_max, vld1_f16(input_ptr + in_offset));
#else
        for (int k = 0; k < C4NUM; k++) {
          tmp_max[k] = fmax(tmp_max[k], *(input_ptr + in_offset + k));
        }
#endif
      }  // win_w loop
    }    // win_h loop
#ifdef ENABLE_NEON
    tmp_max = vmax_f16(tmp_max, min_value2);
    tmp_max = vmin_f16(tmp_max, max_value2);
    vst1_f16(output_ptr + out_channel_offset, tmp_max);
#else
    for (int l = 0; l < C4NUM; ++l) {
      tmp_max[l] = fmax(tmp_max[l], min);
      tmp_max[l] = fmin(tmp_max[l], max);
      output_ptr[out_channel_offset + l] = tmp_max[l];
    }
#endif
  }  // c4 loop
}
void MaxPoolingC1Fp16(const float16_t *input_ptr, float16_t *output_ptr, const PoolingComputeParam *pooling_args,
                      float16_t min, float16_t max, int in_batch_offset, int out_plane_offset, int real_win_h_start,
                      int real_win_h_end, int real_win_w_start, int real_win_w_end, int in_h_index, int in_w_index) {
  int channel = pooling_args->input_channel_;
  int in_w = pooling_args->input_w_;
  int c8 = channel / C8NUM;
  int c8_res = channel % C8NUM;
  int c4 = c8_res / C4NUM;
  int channel_s = c8 * C8NUM + c4 * C4NUM;
  for (int k = channel_s; k < channel; k++) {
    int in_channel_offset = in_batch_offset + k;
    int out_channel_offset = out_plane_offset + k;
    float16_t tmp_max = -FLT_MAX;
    for (int h = real_win_h_start; h < real_win_h_end; h++) {
      for (int w = real_win_w_start; w < real_win_w_end; w++) {
        int in_offset = in_channel_offset + ((in_h_index + h) * in_w + in_w_index + w) * channel;
        tmp_max = fmax(tmp_max, *(input_ptr + in_offset));
      }  // win_w loop
    }    // win_h loop
    tmp_max = fmax(tmp_max, min);
    tmp_max = fmin(tmp_max, max);
    output_ptr[out_channel_offset] = tmp_max;
  }  // channel_res loop
}

void MaxPoolingFp16(const float16_t *input_ptr, float16_t *output_ptr, const PoolingParameter *pooling_param,
                    const PoolingComputeParam *pooling_args, int task_id, int thread_num) {
  float16_t min = (float16_t)pooling_args->minf;
  float16_t max = (float16_t)pooling_args->maxf;

  int stride_w = pooling_param->stride_w_;
  int stride_h = pooling_param->stride_h_;
  int pad_w = pooling_param->pad_l_;
  int pad_h = pooling_param->pad_u_;
  int win_w = pooling_args->window_w_;
  int win_h = pooling_args->window_h_;
  int channel = pooling_args->input_channel_;
  int in_w = pooling_args->input_w_;
  int in_h = pooling_args->input_h_;
  int output_w = pooling_args->output_w_;
  int output_h = pooling_args->output_h_;
  int output_batch = pooling_args->output_batch_;
  int out_plane = output_w * output_h;
  int out_tile_count = UP_DIV(out_plane, TILE_NUM);

  // input channel is equal to output channel
  NNACL_CHECK_ZERO_RETURN(output_w);
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
        int real_win_h_start = MSMAX(0, -in_h_index);
        int real_win_h_end = MSMIN(win_h, in_h - in_h_index);
        int real_win_w_start = MSMAX(0, -in_w_index);
        int real_win_w_end = MSMIN(win_w, in_w - in_w_index);
        MaxPoolingC8Fp16(input_ptr, output_ptr, pooling_args, min, max, in_batch_offset, out_plane_offset,
                         real_win_h_start, real_win_h_end, real_win_w_start, real_win_w_end, in_h_index, in_w_index);
        MaxPoolingC4Fp16(input_ptr, output_ptr, pooling_args, min, max, in_batch_offset, out_plane_offset,
                         real_win_h_start, real_win_h_end, real_win_w_start, real_win_w_end, in_h_index, in_w_index);
        MaxPoolingC1Fp16(input_ptr, output_ptr, pooling_args, min, max, in_batch_offset, out_plane_offset,
                         real_win_h_start, real_win_h_end, real_win_w_start, real_win_w_end, in_h_index, in_w_index);
      }  // real_cal_num loop
    }    // out_plane loop
  }      // out_batch loop
}
