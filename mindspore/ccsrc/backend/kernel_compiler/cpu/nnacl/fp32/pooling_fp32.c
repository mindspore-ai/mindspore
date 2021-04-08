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

#include "nnacl/fp32/pooling_fp32.h"
#include <float.h>
#include "nnacl/errorcode.h"
#include "nnacl/op_base.h"

int AvgPooling(const float *input_ptr, float *output_ptr, const PoolingParameter *pooling_param, int task_id,
               float minf, float maxf) {
  int win_w = pooling_param->window_w_;
  int win_h = pooling_param->window_h_;
  int channel = pooling_param->input_channel_;
  int c4 = channel / C4NUM; /* oc && ic */
  int in_w = pooling_param->input_w_;
  int in_h = pooling_param->input_h_;
  int output_w = pooling_param->output_w_;
  int output_h = pooling_param->output_h_;
  int out_plane = output_w * output_h;
  int out_tile_count = UP_DIV(out_plane, TILE_NUM);

#if defined(ENABLE_NEON) || defined(ENABLE_SSE)
  MS_FLOAT32X4 min_value = MS_MOVQ_F32(minf);
  MS_FLOAT32X4 max_value = MS_MOVQ_F32(maxf);
#endif

  for (int batch = 0; batch < pooling_param->output_batch_; batch++) {
    const float *src_b_ptr = input_ptr + batch * in_h * in_w * channel;
    float *dst_b_ptr = output_ptr + batch * output_h * output_w * channel;
    for (int thread_id = task_id; thread_id < out_tile_count; thread_id += pooling_param->thread_num_) {
      int cal_start_index = thread_id * TILE_NUM;
      int real_cal_num = (out_plane - cal_start_index) > TILE_NUM ? TILE_NUM : (out_plane - cal_start_index);
      for (int i = 0; i < real_cal_num; i++) {
        int index = cal_start_index + i;
        int out_w_index = index % output_w;
        int out_h_index = index / output_w;
        int in_w_index = out_w_index * pooling_param->stride_w_ - pooling_param->pad_l_;
        int in_h_index = out_h_index * pooling_param->stride_h_ - pooling_param->pad_u_;

        const float *src_plane_ptr = src_b_ptr;
        float *dst_plane_ptr = dst_b_ptr + index * channel;

        int real_win_h_start = MSMAX(0, -in_h_index);
        int real_win_h_end = MSMIN(win_h, in_h - in_h_index);
        int real_win_w_start = MSMAX(0, -in_w_index);
        int real_win_w_end = MSMIN(win_w, in_w - in_w_index);

        for (int ci = 0; ci < c4; ci++) {
          const float *src_c_ptr = src_plane_ptr + ci * C4NUM;
          float *dst_c_ptr = dst_plane_ptr + ci * C4NUM;
#if defined(ENABLE_NEON) || defined(ENABLE_SSE)
          MS_FLOAT32X4 tmp_avg = MS_MOVQ_F32(0);
#else
          float tmp_avg1 = 0;
          float tmp_avg2 = 0;
          float tmp_avg3 = 0;
          float tmp_avg4 = 0;
#endif
          int real_count = 0;
          for (int h = real_win_h_start; h < real_win_h_end; h++) {
            for (int w = real_win_w_start; w < real_win_w_end; w++) {
              const float *src_win_ptr = src_c_ptr + ((in_h_index + h) * in_w + in_w_index + w) * channel;
#if defined(ENABLE_NEON) || defined(ENABLE_SSE)
              tmp_avg = MS_ADDQ_F32(tmp_avg, MS_LDQ_F32(src_win_ptr));
#else
              tmp_avg1 += src_win_ptr[0];
              tmp_avg2 += src_win_ptr[1];
              tmp_avg3 += src_win_ptr[2];
              tmp_avg4 += src_win_ptr[3];
#endif
              ++real_count;
            }  // win_w loop
          }    // win_h loop
          if (real_count == 0) {
            return NNACL_ERR;
          }
#if defined(ENABLE_NEON) || defined(ENABLE_SSE)
          tmp_avg = tmp_avg / MS_MOVQ_F32(real_count);
          tmp_avg = MS_MAXQ_F32(tmp_avg, min_value);
          tmp_avg = MS_MINQ_F32(tmp_avg, max_value);
          MS_STQ_F32(dst_c_ptr, tmp_avg);
#else
          tmp_avg1 /= (float)real_count;
          tmp_avg2 /= (float)real_count;
          tmp_avg3 /= (float)real_count;
          tmp_avg4 /= (float)real_count;
          tmp_avg1 = fmax(tmp_avg1, minf);
          tmp_avg2 = fmax(tmp_avg2, minf);
          tmp_avg3 = fmax(tmp_avg3, minf);
          tmp_avg4 = fmax(tmp_avg4, minf);
          tmp_avg1 = fmin(tmp_avg1, maxf);
          tmp_avg2 = fmin(tmp_avg2, maxf);
          tmp_avg3 = fmin(tmp_avg3, maxf);
          tmp_avg4 = fmin(tmp_avg4, maxf);
          dst_c_ptr[0] = tmp_avg1;
          dst_c_ptr[1] = tmp_avg2;
          dst_c_ptr[2] = tmp_avg3;
          dst_c_ptr[3] = tmp_avg4;
#endif
        }  // ic4-1 loop
        int channel_s = c4 * C4NUM;
        for (int ci = channel_s; ci < channel; ci++) {
          const float *src_c_ptr = src_plane_ptr + ci;
          float *dst_c_ptr = dst_plane_ptr + ci;
          float tmp_avg = 0;
          int real_count = 0;
          for (int h = real_win_h_start; h < real_win_h_end; h++) {
            for (int w = real_win_w_start; w < real_win_w_end; w++) {
              const float *src_win_ptr = src_c_ptr + ((in_h_index + h) * in_w + in_w_index + w) * channel;
              tmp_avg += src_win_ptr[0];
              ++real_count;
            }  // win_w loop
          }    // win_h loop
          if (real_count == 0) {
            return NNACL_ERR;
          }
          tmp_avg = tmp_avg / (float)real_count;
          tmp_avg = fmax(tmp_avg, minf);
          tmp_avg = fmin(tmp_avg, maxf);
          dst_c_ptr[0] = tmp_avg;
        }  // channel_res loop
      }    // real_cal_num loop
    }      // out_plane loop
  }        // out_batch loop
  return NNACL_OK;
}

void MaxPooling(const float *input_ptr, float *output_ptr, const PoolingParameter *pooling_param, int task_id,
                float minf, float maxf) {
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
  int c4 = channel / C4NUM; /* oc && ic */

#if defined(ENABLE_NEON) || defined(ENABLE_SSE)
  MS_FLOAT32X4 min_value = MS_MOVQ_F32(minf);
  MS_FLOAT32X4 max_value = MS_MOVQ_F32(maxf);
#endif

  for (int batch = 0; batch < output_batch; batch++) {
    const float *src_b_ptr = input_ptr + batch * in_h * in_w * channel;
    float *dst_b_ptr = output_ptr + batch * output_h * output_w * channel;
    for (int thread_id = task_id; thread_id < out_tile_count; thread_id += pooling_param->thread_num_) {
      int cal_start_index = thread_id * TILE_NUM;
      int real_cal_num = (out_plane - cal_start_index) > TILE_NUM ? TILE_NUM : (out_plane - cal_start_index);
      for (int i = 0; i < real_cal_num; i++) {
        int index = cal_start_index + i;
        int out_w_index = index % output_w;
        int out_h_index = index / output_w;
        int in_w_index = out_w_index * pooling_param->stride_w_ - pooling_param->pad_l_;
        int in_h_index = out_h_index * pooling_param->stride_h_ - pooling_param->pad_u_;

        const float *src_plane_ptr = src_b_ptr;
        float *dst_plane_ptr = dst_b_ptr + index * channel;

        int real_win_h_start = MSMAX(0, -in_h_index);
        int real_win_h_end = MSMIN(win_h, in_h - in_h_index);
        int real_win_w_start = MSMAX(0, -in_w_index);
        int real_win_w_end = MSMIN(win_w, in_w - in_w_index);

        for (int ci = 0; ci < c4; ci++) {
          const float *src_c_ptr = src_plane_ptr + ci * C4NUM;
          float *dst_c_ptr = dst_plane_ptr + ci * C4NUM;
#if defined(ENABLE_NEON) || defined(ENABLE_SSE)
          MS_FLOAT32X4 tmp_max = MS_MOVQ_F32(-FLT_MAX);
#else
          float tmp_max1 = -FLT_MAX;
          float tmp_max2 = -FLT_MAX;
          float tmp_max3 = -FLT_MAX;
          float tmp_max4 = -FLT_MAX;
#endif

          for (int kh = real_win_h_start; kh < real_win_h_end; kh++) {
            for (int kw = real_win_w_start; kw < real_win_w_end; kw++) {
              const float *src_win_ptr = src_c_ptr + ((in_h_index + kh) * in_w + in_w_index + kw) * channel;
#if defined(ENABLE_NEON) || defined(ENABLE_SSE)
              tmp_max = MS_MAXQ_F32(tmp_max, MS_LDQ_F32(src_win_ptr));
#else
              tmp_max1 = fmax(tmp_max1, src_win_ptr[0]);
              tmp_max2 = fmax(tmp_max2, src_win_ptr[1]);
              tmp_max3 = fmax(tmp_max3, src_win_ptr[2]);
              tmp_max4 = fmax(tmp_max4, src_win_ptr[3]);
#endif
            }  // win_w loop
          }    // win_h loop
#if defined(ENABLE_NEON) || defined(ENABLE_SSE)
          tmp_max = MS_MAXQ_F32(tmp_max, min_value);
          tmp_max = MS_MINQ_F32(tmp_max, max_value);
          MS_STQ_F32(dst_c_ptr, tmp_max);
#else
          tmp_max1 = fmax(tmp_max1, minf);
          tmp_max2 = fmax(tmp_max2, minf);
          tmp_max3 = fmax(tmp_max3, minf);
          tmp_max4 = fmax(tmp_max4, minf);
          tmp_max1 = fmin(tmp_max1, maxf);
          tmp_max2 = fmin(tmp_max2, maxf);
          tmp_max3 = fmin(tmp_max3, maxf);
          tmp_max4 = fmin(tmp_max4, maxf);
          dst_c_ptr[0] = tmp_max1;
          dst_c_ptr[1] = tmp_max2;
          dst_c_ptr[2] = tmp_max3;
          dst_c_ptr[3] = tmp_max4;
#endif
        }  // ic4-1 loop
        int channel_s = c4 * C4NUM;
        for (int ci = channel_s; ci < channel; ci++) {
          float *dst_c_ptr = dst_plane_ptr + ci;
          const float *src_c_ptr = src_plane_ptr + ci;
          float tmp_max = -FLT_MAX;

          for (int kh = real_win_h_start; kh < real_win_h_end; kh++) {
            for (int kw = real_win_w_start; kw < real_win_w_end; kw++) {
              const float *src_win_ptr = src_c_ptr + ((in_h_index + kh) * in_w + in_w_index + kw) * channel;
              tmp_max = fmax(tmp_max, src_win_ptr[0]);
            }  // win_w loop
          }    // win_h loop
          tmp_max = fmax(tmp_max, minf);
          tmp_max = fmin(tmp_max, maxf);
          dst_c_ptr[0] = tmp_max;
        }  // channel_res loop
      }    // real_cal_num loop
    }      // out_plane loop
  }        // out_batch loop
}
