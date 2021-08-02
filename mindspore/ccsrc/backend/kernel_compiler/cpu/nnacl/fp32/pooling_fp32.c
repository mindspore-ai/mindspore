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
  int in_w = pooling_param->input_w_;
  int in_h = pooling_param->input_h_;
  int output_w = pooling_param->output_w_;
  int output_h = pooling_param->output_h_;
  int out_plane = output_w * output_h;
  int out_tile_count = UP_DIV(out_plane, TILE_NUM);
  NNACL_CHECK_ZERO_RETURN_ERR(output_w);
#ifdef ENABLE_AVX
  int c8 = channel / C8NUM * C8NUM;
  MS_FLOAT32X8 min_value_8 = MS_MOV256_F32(minf);
  MS_FLOAT32X8 max_value_8 = MS_MOV256_F32(maxf);
#endif
#if defined(ENABLE_NEON) || defined(ENABLE_SSE)
  int c4 = channel / C4NUM * C4NUM;
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
        int ci = 0;
#ifdef ENABLE_AVX
        for (; ci < c8; ci += C8NUM) {
          const float *src_c_ptr = src_plane_ptr + ci;
          float *dst_c_ptr = dst_plane_ptr + ci;
          MS_FLOAT32X8 tmp_avg = MS_MOV256_F32(0);
          int real_count = 0;
          for (int h = real_win_h_start; h < real_win_h_end; h++) {
            for (int w = real_win_w_start; w < real_win_w_end; w++) {
              const float *src_win_ptr = src_c_ptr + ((in_h_index + h) * in_w + in_w_index + w) * channel;
              tmp_avg = MS_ADD256_F32(tmp_avg, MS_LD256_F32(src_win_ptr));
              ++real_count;
            }  // win_w loop
          }    // win_h loop
          if (real_count == 0) {
            return NNACL_ERR;
          }
          tmp_avg = MS_DIV256_F32(tmp_avg, MS_MOV256_F32(real_count));
          tmp_avg = MS_MAX256_F32(tmp_avg, min_value_8);
          tmp_avg = MS_MIN256_F32(tmp_avg, max_value_8);
          MS_ST256_F32(dst_c_ptr, tmp_avg);
        }  // ic8-1 loop
#endif
#if defined(ENABLE_NEON) || defined(ENABLE_SSE)
        for (; ci < c4; ci += C4NUM) {
          const float *src_c_ptr = src_plane_ptr + ci;
          float *dst_c_ptr = dst_plane_ptr + ci;
          MS_FLOAT32X4 tmp_avg = MS_MOVQ_F32(0);
          int real_count = 0;
          for (int h = real_win_h_start; h < real_win_h_end; h++) {
            for (int w = real_win_w_start; w < real_win_w_end; w++) {
              const float *src_win_ptr = src_c_ptr + ((in_h_index + h) * in_w + in_w_index + w) * channel;
              tmp_avg = MS_ADDQ_F32(tmp_avg, MS_LDQ_F32(src_win_ptr));
              ++real_count;
            }  // win_w loop
          }    // win_h loop
          if (real_count == 0) {
            return NNACL_ERR;
          }
          tmp_avg = MS_DIVQ_F32(tmp_avg, MS_MOVQ_F32(real_count));
          tmp_avg = MS_MAXQ_F32(tmp_avg, min_value);
          tmp_avg = MS_MINQ_F32(tmp_avg, max_value);
          MS_STQ_F32(dst_c_ptr, tmp_avg);
        }  // ic4-1 loop
#endif
        for (; ci < channel; ci++) {
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
          tmp_avg = fmaxf(tmp_avg, minf);
          tmp_avg = fminf(tmp_avg, maxf);
          dst_c_ptr[0] = tmp_avg;
        }  // channel_res loop
      }    // real_cal_num loop
    }      // out_plane loop
  }        // out_batch loop
  return NNACL_OK;
}

int MaxPooling(const float *input_ptr, float *output_ptr, const PoolingParameter *pooling_param, int task_id,
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
  NNACL_CHECK_ZERO_RETURN_ERR(output_w);
#ifdef ENABLE_AVX
  int c8 = channel / C8NUM * C8NUM;
  MS_FLOAT32X8 min_value_8 = MS_MOV256_F32(minf);
  MS_FLOAT32X8 max_value_8 = MS_MOV256_F32(maxf);
#endif
#if defined(ENABLE_NEON) || defined(ENABLE_SSE)
  int c4 = channel / C4NUM * C4NUM;
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
        int ci = 0;
#ifdef ENABLE_AVX
        for (; ci < c8; ci += C8NUM) {
          const float *src_c_ptr = src_plane_ptr + ci;
          float *dst_c_ptr = dst_plane_ptr + ci;
          MS_FLOAT32X8 tmp_max = MS_MOV256_F32(-FLT_MAX);
          for (int kh = real_win_h_start; kh < real_win_h_end; kh++) {
            for (int kw = real_win_w_start; kw < real_win_w_end; kw++) {
              const float *src_win_ptr = src_c_ptr + ((in_h_index + kh) * in_w + in_w_index + kw) * channel;
              tmp_max = MS_MAX256_F32(tmp_max, MS_LD256_F32(src_win_ptr));
            }  // win_w loop
          }    // win_h loop
          tmp_max = MS_MAX256_F32(tmp_max, min_value_8);
          tmp_max = MS_MIN256_F32(tmp_max, max_value_8);
          MS_ST256_F32(dst_c_ptr, tmp_max);
        }  // ic8 loop
#endif
#if defined(ENABLE_NEON) || defined(ENABLE_SSE)
        for (; ci < c4; ci += C4NUM) {
          const float *src_c_ptr = src_plane_ptr + ci;
          float *dst_c_ptr = dst_plane_ptr + ci;
          MS_FLOAT32X4 tmp_max = MS_MOVQ_F32(-FLT_MAX);
          for (int kh = real_win_h_start; kh < real_win_h_end; kh++) {
            for (int kw = real_win_w_start; kw < real_win_w_end; kw++) {
              const float *src_win_ptr = src_c_ptr + ((in_h_index + kh) * in_w + in_w_index + kw) * channel;
              tmp_max = MS_MAXQ_F32(tmp_max, MS_LDQ_F32(src_win_ptr));
            }  // win_w loop
          }    // win_h loop
          tmp_max = MS_MAXQ_F32(tmp_max, min_value);
          tmp_max = MS_MINQ_F32(tmp_max, max_value);
          MS_STQ_F32(dst_c_ptr, tmp_max);
        }  // ic4 loop
#endif
        for (; ci < channel; ci++) {
          float *dst_c_ptr = dst_plane_ptr + ci;
          const float *src_c_ptr = src_plane_ptr + ci;
          float tmp_max = -FLT_MAX;
          for (int kh = real_win_h_start; kh < real_win_h_end; kh++) {
            for (int kw = real_win_w_start; kw < real_win_w_end; kw++) {
              const float *src_win_ptr = src_c_ptr + ((in_h_index + kh) * in_w + in_w_index + kw) * channel;
              tmp_max = fmaxf(tmp_max, src_win_ptr[0]);
            }  // win_w loop
          }    // win_h loop
          tmp_max = fmaxf(tmp_max, minf);
          tmp_max = fminf(tmp_max, maxf);
          dst_c_ptr[0] = tmp_max;
        }  // channel_res loop
      }    // real_cal_num loop
    }      // out_plane loop
  }        // out_batch loop
  return NNACL_OK;
}
