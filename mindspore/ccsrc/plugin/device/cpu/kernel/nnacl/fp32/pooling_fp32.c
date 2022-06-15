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
#include "nnacl/pooling_fp32_simd.h"

int AvgPoolingBatch(const float *src_b_ptr, float *dst_b_ptr, const PoolingParameter *pooling_param, int task_id,
                    float minf, float maxf) {
  int win_w = pooling_param->window_w_, win_h = pooling_param->window_h_;
  int in_w = pooling_param->input_w_, in_h = pooling_param->input_h_;
  int output_w = pooling_param->output_w_, output_h = pooling_param->output_h_;
  int channel = pooling_param->input_channel_;
  int out_plane = output_w * output_h;
  int out_tile_count = UP_DIV(out_plane, TILE_NUM);
  NNACL_CHECK_ZERO_RETURN_ERR(output_w);

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

      MS_CHECK_TRUE_RET(real_win_h_end > real_win_h_start, NNACL_ERR);
      MS_CHECK_TRUE_RET(real_win_w_end > real_win_w_start, NNACL_ERR);
      SIMD_RUN_NO_SCALAR(AvgPoolingBatch, ci, src_plane_ptr, channel, dst_plane_ptr, real_win_h_start, real_win_h_end,
                         real_win_w_start, real_win_w_end, in_h_index, in_w, in_w_index, minf, maxf);

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
        MS_CHECK_TRUE_RET(real_count != 0, NNACL_ERR);
        tmp_avg = tmp_avg / (float)real_count;
        tmp_avg = fmaxf(tmp_avg, minf);
        tmp_avg = fminf(tmp_avg, maxf);
        dst_c_ptr[0] = tmp_avg;
      }  // channel_res loop
    }    // real_cal_num loop
  }      // out_plane loop
  return NNACL_OK;
}

int AvgPooling(const float *input_ptr, float *output_ptr, const PoolingParameter *pooling_param, int task_id,
               float minf, float maxf) {
  int in_w = pooling_param->input_w_;
  int in_h = pooling_param->input_h_;
  int output_w = pooling_param->output_w_;
  int output_h = pooling_param->output_h_;
  int channel = pooling_param->input_channel_;
  int output_batch = pooling_param->output_batch_;

  for (int batch = 0; batch < output_batch; batch++) {
    const float *src_b_ptr = input_ptr + batch * in_h * in_w * channel;
    float *dst_b_ptr = output_ptr + batch * output_h * output_w * channel;
    int ret = AvgPoolingBatch(src_b_ptr, dst_b_ptr, pooling_param, task_id, minf, maxf);
    if (ret != NNACL_OK) {
      return ret;
    }
  }
  return NNACL_OK;
}

int AvgPoolingFromNC4HW4ToNHWCLessC(const float *src_b_ptr, float *dst_b_ptr, const PoolingParameter *pooling_param,
                                    int task_id, float minf, float maxf) {
  int in_w = pooling_param->input_w_, in_h = pooling_param->input_h_;
  int win_w = pooling_param->window_w_, win_h = pooling_param->window_h_;
  int output_w = pooling_param->output_w_, output_h = pooling_param->output_h_;
  int channel = pooling_param->input_channel_;

  int out_plane = output_w * output_h;
  int in_plane = in_w * in_h;
  NNACL_CHECK_ZERO_RETURN_ERR(output_w);

#ifdef ENABLE_AVX
  const int c_tile = C8NUM;
  const int once_calc_num = 2;
#elif defined(ENABLE_NEON) || defined(ENABLE_SSE)
  const int c_tile = C4NUM;
  const int once_calc_num = 1;
#else
  const int c_tile = 1;
  const int once_calc_num = 1;
#endif

  const int c_xtile = once_calc_num * c_tile;

  int cur_c = (channel / c_xtile) * c_xtile;
  int last_c_size = channel - cur_c;

  int less_out_plane = out_plane * last_c_size;
  int calc_tile = UP_DIV(less_out_plane, pooling_param->thread_num_);

  int index_begin = task_id * calc_tile;
  int index_end = (index_begin + calc_tile) < less_out_plane ? (index_begin + calc_tile) : less_out_plane;

  int c_start = (index_begin / out_plane) + cur_c;
  int index_less = index_begin % out_plane;
  int h_start = index_less / output_h;
  int w_start = index_less % output_h;

  int c_end = (index_end / out_plane) + cur_c;
  index_less = index_end % out_plane;
  int h_end = index_less / output_h;
  int w_end = index_less % output_h;

  int c = c_start;
  int h = h_start;
  int w = w_start;

  int in_w_cx_line = in_w * last_c_size;
  const float *src_c_ptr = src_b_ptr + c * in_plane;
  for (; c < channel; c += c_xtile) {
    for (; h < output_h; h++) {
      int cur_index_in_h_start = MSMAX(h * pooling_param->stride_h_ - pooling_param->pad_d_, 0);
      int cur_index_in_h_end = MSMIN(cur_index_in_h_start + win_h, in_h);

      for (; w < output_w; w++) {
        MS_CHECK_TRUE_RET((c < c_end || h < h_end || w < w_end), NNACL_OK);
        float tmp_avg = 0.0;

        int cur_index_in_w_start = MSMAX(w * pooling_param->stride_w_ - pooling_param->pad_l_, 0);
        int cur_index_in_w_end = MSMIN(cur_index_in_w_start + win_w, in_w);

        int real_count = (cur_index_in_w_end - cur_index_in_w_start) * (cur_index_in_h_end - cur_index_in_h_start);
        MS_CHECK_TRUE_RET(real_count != 0, NNACL_ERR);

        for (int cur_index_in_h = cur_index_in_h_start; cur_index_in_h < cur_index_in_h_end; cur_index_in_h++) {
          const float *src_c_ptr_h_line = src_c_ptr + cur_index_in_h * in_w_cx_line;
          for (int cur_index_in_w = cur_index_in_w_start; cur_index_in_w < cur_index_in_w_end; cur_index_in_w++) {
            const float *cur_input_index = src_c_ptr_h_line + cur_index_in_w * last_c_size + (c - cur_c);
            tmp_avg += cur_input_index[0];
          }
        }

        float *dst_c_ptr = dst_b_ptr + h * output_w * channel + w * channel + c;
        tmp_avg = tmp_avg / (float)real_count;
        tmp_avg = fminf(tmp_avg, maxf);
        dst_c_ptr[0] = tmp_avg;
      }
      w = 0;
    }
    h = 0;
  }
  return NNACL_OK;
}

int AvgPoolingFromNC4HW4ToNHWCBatch(const float *src_b_ptr, float *dst_b_ptr, const PoolingParameter *pooling_param,
                                    int task_id, float minf, float maxf) {
  int in_w = pooling_param->input_w_, in_h = pooling_param->input_h_;
  int win_w = pooling_param->window_w_, win_h = pooling_param->window_h_;
  int output_w = pooling_param->output_w_, output_h = pooling_param->output_h_;
  int channel = pooling_param->input_channel_;

  int out_plane = output_w * output_h;
  int in_plane = in_w * in_h;
  NNACL_CHECK_ZERO_RETURN_ERR(output_w);

#ifdef ENABLE_AVX
  const int c_tile = C8NUM;
  const int once_calc_num = 2;
  MS_FLOAT32X8 min_value_8 = MS_MOV256_F32(minf);
  MS_FLOAT32X8 max_value_8 = MS_MOV256_F32(maxf);
#elif defined(ENABLE_NEON) || defined(ENABLE_SSE)
  const int c_tile = C4NUM;
  const int once_calc_num = 1;
  MS_FLOAT32X4 min_value = MS_MOVQ_F32(minf);
  MS_FLOAT32X4 max_value = MS_MOVQ_F32(maxf);
#else
  const int c_tile = 1;
  const int once_calc_num = 1;
#endif

  int in_w_cx_line = in_w * c_tile;
  const int c_xtile = once_calc_num * c_tile;
  int c_tile_num = channel / c_xtile;
  int all_out_plane = out_plane * c_tile_num;
  int calc_tile = UP_DIV(all_out_plane, pooling_param->thread_num_);

  int index_begin = task_id * calc_tile;
  int index_end = (index_begin + calc_tile) < all_out_plane ? (index_begin + calc_tile) : all_out_plane;

  int c_start = (index_begin / out_plane) * c_xtile;
  int index_less = index_begin % out_plane;
  int h_start = index_less / output_h;
  int w_start = index_less % output_h;

  int c_end = (index_end / out_plane) * c_xtile;
  index_less = index_end % out_plane;
  int h_end = index_less / output_h;
  int w_end = index_less % output_h;

  int c = c_start;
  int h = h_start;
  int w = w_start;
  for (; c < channel; c += c_xtile) {
    const float *src_c_ptr = src_b_ptr + c * in_plane;
    for (; h < output_h; h++) {
      int cur_index_in_h_start = MSMAX(h * pooling_param->stride_h_ - pooling_param->pad_d_, 0);
      int cur_index_in_h_end = MSMIN(cur_index_in_h_start + win_h, in_h);

      for (; w < output_w; w++) {
        MS_CHECK_TRUE_RET((c < c_end || h < h_end || w < w_end), NNACL_OK);

#ifdef ENABLE_AVX
        MS_FLOAT32X8 tmp_avg = MS_MOV256_F32(0);
        MS_FLOAT32X8 tmp_avg2 = MS_MOV256_F32(0);
#elif defined(ENABLE_NEON) || defined(ENABLE_SSE)
        MS_FLOAT32X4 tmp_avg = MS_MOVQ_F32(0);
#else
        float tmp_avg = 0;
#endif

        int cur_index_in_w_start = MSMAX(w * pooling_param->stride_w_ - pooling_param->pad_l_, 0);
        int cur_index_in_w_end = MSMIN(cur_index_in_w_start + win_w, in_w);

        int real_count = (cur_index_in_w_end - cur_index_in_w_start) * (cur_index_in_h_end - cur_index_in_h_start);
        MS_CHECK_TRUE_RET(real_count != 0, NNACL_ERR);

        for (int cur_index_in_h = cur_index_in_h_start; cur_index_in_h < cur_index_in_h_end; cur_index_in_h++) {
          const float *src_c_ptr_h_line = src_c_ptr + cur_index_in_h * in_w_cx_line;
          for (int cur_index_in_w = cur_index_in_w_start; cur_index_in_w < cur_index_in_w_end; cur_index_in_w++) {
            const float *cur_input_index = src_c_ptr_h_line + cur_index_in_w * c_tile;
#ifdef ENABLE_AVX
            tmp_avg = MS_ADD256_F32(tmp_avg, MS_LD256_F32(cur_input_index));
#elif defined(ENABLE_NEON) || defined(ENABLE_SSE)
            tmp_avg = MS_ADDQ_F32(tmp_avg, MS_LDQ_F32(cur_input_index));
#else
            tmp_avg += cur_input_index[0];
#endif
          }

#ifdef ENABLE_AVX
          const float *src_c2_ptr_h_line = src_c_ptr_h_line + c_tile * in_plane;
          for (int cur_index_in_w = cur_index_in_w_start; cur_index_in_w < cur_index_in_w_end; cur_index_in_w++) {
            const float *cur_input_index = src_c2_ptr_h_line + cur_index_in_w * c_tile;

            tmp_avg2 = MS_ADD256_F32(tmp_avg2, MS_LD256_F32(cur_input_index));
          }
#endif
        }

        float *dst_c_ptr = dst_b_ptr + h * output_w * channel + w * channel + c;
#ifdef ENABLE_AVX
        float *dst_c2_ptr = dst_c_ptr + c_tile;

        tmp_avg = MS_DIV256_F32(tmp_avg, MS_MOV256_F32(real_count));
        tmp_avg = MS_MAX256_F32(tmp_avg, min_value_8);
        tmp_avg = MS_MIN256_F32(tmp_avg, max_value_8);
        MS_ST256_F32(dst_c_ptr, tmp_avg);

        tmp_avg2 = MS_DIV256_F32(tmp_avg2, MS_MOV256_F32(real_count));
        tmp_avg2 = MS_MAX256_F32(tmp_avg2, min_value_8);
        tmp_avg2 = MS_MIN256_F32(tmp_avg2, max_value_8);
        MS_ST256_F32(dst_c2_ptr, tmp_avg2);
#elif defined(ENABLE_NEON) || defined(ENABLE_SSE)
        tmp_avg = MS_DIVQ_F32(tmp_avg, MS_MOVQ_F32(real_count));
        tmp_avg = MS_MAXQ_F32(tmp_avg, min_value);
        tmp_avg = MS_MINQ_F32(tmp_avg, max_value);
        MS_STQ_F32(dst_c_ptr, tmp_avg);
#else
        tmp_avg = tmp_avg / (float)real_count;
        tmp_avg = fmaxf(tmp_avg, minf);
        tmp_avg = fminf(tmp_avg, maxf);
        dst_c_ptr[0] = tmp_avg;
#endif
      }
      w = 0;
    }
    h = 0;
  }

  return NNACL_OK;
}

int AvgPoolingFromNC4HW4ToNHWC(const float *input_ptr, float *output_ptr, const PoolingParameter *pooling_param,
                               int task_id, float minf, float maxf) {
  int in_w = pooling_param->input_w_;
  int in_h = pooling_param->input_h_;
  int output_w = pooling_param->output_w_;
  int output_h = pooling_param->output_h_;
  int channel = pooling_param->input_channel_;
  int output_batch = pooling_param->output_batch_;

  for (int batch = 0; batch < output_batch; batch++) {
    const float *src_b_ptr = input_ptr + batch * in_h * in_w * channel;
    float *dst_b_ptr = output_ptr + batch * output_h * output_w * channel;
    int ret = AvgPoolingFromNC4HW4ToNHWCBatch(src_b_ptr, dst_b_ptr, pooling_param, task_id, minf, maxf);
    if (ret != NNACL_OK) {
      return ret;
    }

    ret = AvgPoolingFromNC4HW4ToNHWCLessC(src_b_ptr, dst_b_ptr, pooling_param, task_id, minf, maxf);
    if (ret != NNACL_OK) {
      return ret;
    }
  }
  return NNACL_OK;
}

int MaxPoolingBatch(const float *src_b_ptr, float *dst_b_ptr, const PoolingParameter *pooling_param, int task_id,
                    float minf, float maxf) {
  int in_w = pooling_param->input_w_, in_h = pooling_param->input_h_;
  int win_w = pooling_param->window_w_, win_h = pooling_param->window_h_;
  int output_w = pooling_param->output_w_, output_h = pooling_param->output_h_;
  int channel = pooling_param->input_channel_;
  int out_plane = output_w * output_h;
  int out_tile_count = UP_DIV(out_plane, TILE_NUM);
  NNACL_CHECK_ZERO_RETURN_ERR(output_w);

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

      SIMD_RUN_NO_SCALAR(MaxPoolingBatch, ci, src_plane_ptr, channel, dst_plane_ptr, real_win_h_start, real_win_h_end,
                         real_win_w_start, real_win_w_end, in_h_index, in_w, in_w_index, minf, maxf);

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
  return NNACL_OK;
}

int MaxPooling(const float *input_ptr, float *output_ptr, const PoolingParameter *pooling_param, int task_id,
               float minf, float maxf) {
  int in_w = pooling_param->input_w_;
  int in_h = pooling_param->input_h_;
  int output_w = pooling_param->output_w_;
  int output_h = pooling_param->output_h_;
  int channel = pooling_param->input_channel_;
  int output_batch = pooling_param->output_batch_;

  for (int batch = 0; batch < output_batch; batch++) {
    const float *src_b_ptr = input_ptr + batch * in_h * in_w * channel;
    float *dst_b_ptr = output_ptr + batch * output_h * output_w * channel;
    int ret = MaxPoolingBatch(src_b_ptr, dst_b_ptr, pooling_param, task_id, minf, maxf);
    if (ret != NNACL_OK) {
      return ret;
    }
  }
  return NNACL_OK;
}

int MaxPoolingFromNC4HW4ToNHWCLessC(const float *src_b_ptr, float *dst_b_ptr, const PoolingParameter *pooling_param,
                                    int task_id, float minf, float maxf) {
  int in_w = pooling_param->input_w_, in_h = pooling_param->input_h_;
  int win_w = pooling_param->window_w_, win_h = pooling_param->window_h_;
  int output_w = pooling_param->output_w_, output_h = pooling_param->output_h_;
  int channel = pooling_param->input_channel_;

  int out_plane = output_w * output_h;
  int in_plane = in_w * in_h;
  NNACL_CHECK_ZERO_RETURN_ERR(output_w);

#ifdef ENABLE_AVX
  const int c_tile = C8NUM;
  const int once_calc_num = 2;
#elif defined(ENABLE_NEON) || defined(ENABLE_SSE)
  const int c_tile = C4NUM;
  const int once_calc_num = 1;
#else
  const int c_tile = 1;
  const int once_calc_num = 1;
#endif

  const int c_xtile = once_calc_num * c_tile;

  int cur_c = (channel / c_xtile) * c_xtile;
  int last_c_size = channel - cur_c;

  int less_out_plane = out_plane * last_c_size;
  int calc_tile = UP_DIV(less_out_plane, pooling_param->thread_num_);

  int index_begin = task_id * calc_tile;
  int index_end = (index_begin + calc_tile) < less_out_plane ? (index_begin + calc_tile) : less_out_plane;

  int c_start = (index_begin / out_plane) + cur_c;
  int index_less = index_begin % out_plane;
  int h_start = index_less / output_h;
  int w_start = index_less % output_h;

  int c_end = (index_end / out_plane) + cur_c;
  index_less = index_end % out_plane;
  int h_end = index_less / output_h;
  int w_end = index_less % output_h;

  int c = c_start;
  int h = h_start;
  int w = w_start;

  int in_w_cx_line = in_w * last_c_size;
  const float *src_c_ptr = src_b_ptr + cur_c * in_plane;
  for (; c < channel; c++) {
    for (; h < output_h; h++) {
      int cur_index_in_h_start = MSMAX(h * pooling_param->stride_h_ - pooling_param->pad_d_, 0);
      int cur_index_in_h_end = MSMIN(cur_index_in_h_start + win_h, in_h);

      for (; w < output_w; w++) {
        MS_CHECK_TRUE_RET((c < c_end || h < h_end || w < w_end), NNACL_OK);
        float tmp_max = -FLT_MAX;

        int cur_index_in_w_start = MSMAX(w * pooling_param->stride_w_ - pooling_param->pad_l_, 0);
        int cur_index_in_w_end = MSMIN(cur_index_in_w_start + win_w, in_w);

        for (int cur_index_in_h = cur_index_in_h_start; cur_index_in_h < cur_index_in_h_end; cur_index_in_h++) {
          const float *src_c_ptr_h_line = src_c_ptr + cur_index_in_h * in_w_cx_line;
          for (int cur_index_in_w = cur_index_in_w_start; cur_index_in_w < cur_index_in_w_end; cur_index_in_w++) {
            const float *cur_input_index = src_c_ptr_h_line + cur_index_in_w * last_c_size + (c - cur_c);
            tmp_max = fmaxf(tmp_max, cur_input_index[0]);
          }
        }

        float *dst_c_ptr = dst_b_ptr + h * output_w * channel + w * channel + c;
        tmp_max = fmaxf(tmp_max, minf);
        tmp_max = fminf(tmp_max, maxf);
        dst_c_ptr[0] = tmp_max;
      }
      w = 0;
    }
    h = 0;
  }
  return NNACL_OK;
}

int MaxPoolingFromNC4HW4ToNHWCBatch(const float *src_b_ptr, float *dst_b_ptr, const PoolingParameter *pooling_param,
                                    int task_id, float minf, float maxf) {
  int in_w = pooling_param->input_w_, in_h = pooling_param->input_h_;
  int win_w = pooling_param->window_w_, win_h = pooling_param->window_h_;
  int output_w = pooling_param->output_w_, output_h = pooling_param->output_h_;
  int channel = pooling_param->input_channel_;

  int out_plane = output_w * output_h;
  int in_plane = in_w * in_h;
  NNACL_CHECK_ZERO_RETURN_ERR(output_w);

#ifdef ENABLE_AVX
  const int c_tile = C8NUM;
  const int once_calc_num = 2;
  MS_FLOAT32X8 min_value_8 = MS_MOV256_F32(minf);
  MS_FLOAT32X8 max_value_8 = MS_MOV256_F32(maxf);
#elif defined(ENABLE_NEON) || defined(ENABLE_SSE)
  const int c_tile = C4NUM;
  const int once_calc_num = 1;
  MS_FLOAT32X4 min_value = MS_MOVQ_F32(minf);
  MS_FLOAT32X4 max_value = MS_MOVQ_F32(maxf);
#else
  const int c_tile = 1;
  const int once_calc_num = 1;
#endif

  int in_w_cx_line = in_w * c_tile;
  const int c_xtile = once_calc_num * c_tile;
  int c_tile_num = channel / c_xtile;
  int all_out_plane = out_plane * c_tile_num;
  int calc_tile = UP_DIV(all_out_plane, pooling_param->thread_num_);

  int index_begin = task_id * calc_tile;
  int index_end = (index_begin + calc_tile) < all_out_plane ? (index_begin + calc_tile) : all_out_plane;

  int c_start = (index_begin / out_plane) * c_xtile;
  int index_less = index_begin % out_plane;
  int h_start = index_less / output_h;
  int w_start = index_less % output_h;

  int c_end = (index_end / out_plane) * c_xtile;
  index_less = index_end % out_plane;
  int h_end = index_less / output_h;
  int w_end = index_less % output_h;

  int c = c_start;
  int h = h_start;
  int w = w_start;
  for (; c < channel; c += c_xtile) {
    const float *src_c_ptr = src_b_ptr + c * in_plane;
    for (; h < output_h; h++) {
      int cur_index_in_h_start = MSMAX(h * pooling_param->stride_h_ - pooling_param->pad_d_, 0);
      int cur_index_in_h_end = MSMIN(cur_index_in_h_start + win_h, in_h);

      for (; w < output_w; w++) {
        MS_CHECK_TRUE_RET((c < c_end || h < h_end || w < w_end), NNACL_OK);

#ifdef ENABLE_AVX
        MS_FLOAT32X8 tmp_max = MS_MOV256_F32(-FLT_MAX);
        MS_FLOAT32X8 tmp_max2 = MS_MOV256_F32(-FLT_MAX);
#elif defined(ENABLE_NEON) || defined(ENABLE_SSE)
        MS_FLOAT32X4 tmp_max = MS_MOVQ_F32(-FLT_MAX);
#else
        float tmp_max = -FLT_MAX;
#endif

        int cur_index_in_w_start = MSMAX(w * pooling_param->stride_w_ - pooling_param->pad_l_, 0);
        int cur_index_in_w_end = MSMIN(cur_index_in_w_start + win_w, in_w);

        for (int cur_index_in_h = cur_index_in_h_start; cur_index_in_h < cur_index_in_h_end; cur_index_in_h++) {
          const float *src_c_ptr_h_line = src_c_ptr + cur_index_in_h * in_w_cx_line;
          for (int cur_index_in_w = cur_index_in_w_start; cur_index_in_w < cur_index_in_w_end; cur_index_in_w++) {
            const float *cur_input_index = src_c_ptr_h_line + cur_index_in_w * c_tile;
#ifdef ENABLE_AVX
            tmp_max = MS_MAX256_F32(tmp_max, MS_LD256_F32(cur_input_index));
#elif defined(ENABLE_NEON) || defined(ENABLE_SSE)
            tmp_max = MS_MAXQ_F32(tmp_max, MS_LDQ_F32(cur_input_index));
#else
            tmp_max = fmaxf(tmp_max, cur_input_index[0]);
#endif
          }

#ifdef ENABLE_AVX
          const float *src_c2_ptr_h_line = src_c_ptr_h_line + c_tile * in_plane;
          for (int cur_index_in_w = cur_index_in_w_start; cur_index_in_w < cur_index_in_w_end; cur_index_in_w++) {
            const float *cur_input_index = src_c2_ptr_h_line + cur_index_in_w * c_tile;

            tmp_max2 = MS_MAX256_F32(tmp_max2, MS_LD256_F32(cur_input_index));
          }
#endif
        }

        float *dst_c_ptr = dst_b_ptr + h * output_w * channel + w * channel + c;
#ifdef ENABLE_AVX
        float *dst_c2_ptr = dst_c_ptr + c_tile;

        tmp_max = MS_MAX256_F32(tmp_max, min_value_8);
        tmp_max = MS_MIN256_F32(tmp_max, max_value_8);
        MS_ST256_F32(dst_c_ptr, tmp_max);

        tmp_max2 = MS_MAX256_F32(tmp_max2, min_value_8);
        tmp_max2 = MS_MIN256_F32(tmp_max2, max_value_8);
        MS_ST256_F32(dst_c2_ptr, tmp_max2);
#elif defined(ENABLE_NEON) || defined(ENABLE_SSE)
        tmp_max = MS_MAXQ_F32(tmp_max, min_value);
        tmp_max = MS_MINQ_F32(tmp_max, max_value);
        MS_STQ_F32(dst_c_ptr, tmp_max);
#else
        tmp_max = fmaxf(tmp_max, minf);
        tmp_max = fminf(tmp_max, maxf);
        dst_c_ptr[0] = tmp_max;
#endif
      }
      w = 0;
    }
    h = 0;
  }

  return NNACL_OK;
}

int MaxPoolingFromNC4HW4ToNHWC(const float *input_ptr, float *output_ptr, const PoolingParameter *pooling_param,
                               int task_id, float minf, float maxf) {
  int in_w = pooling_param->input_w_;
  int in_h = pooling_param->input_h_;
  int output_w = pooling_param->output_w_;
  int output_h = pooling_param->output_h_;
  int channel = pooling_param->input_channel_;
  int output_batch = pooling_param->output_batch_;

  for (int batch = 0; batch < output_batch; batch++) {
    const float *src_b_ptr = input_ptr + batch * in_h * in_w * channel;
    float *dst_b_ptr = output_ptr + batch * output_h * output_w * channel;
    int ret = MaxPoolingFromNC4HW4ToNHWCBatch(src_b_ptr, dst_b_ptr, pooling_param, task_id, minf, maxf);
    if (ret != NNACL_OK) {
      return ret;
    }

    ret = MaxPoolingFromNC4HW4ToNHWCLessC(src_b_ptr, dst_b_ptr, pooling_param, task_id, minf, maxf);
    if (ret != NNACL_OK) {
      return ret;
    }
  }
  return NNACL_OK;
}
