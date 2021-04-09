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
#include <stdint.h>
#include <string.h>
#include <float.h>
#include "nnacl/fp32_grad/pooling_grad.h"

void AvgPoolingGrad(const float *input_ptr, float *output_ptr, int count, PoolingParameter *pooling_param) {
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

  const float kk = 1.0f / (float)(win_h * win_w);
#if ENABLE_ARM
  const float32x4_t factor = vdupq_n_f32(kk);
#endif
  for (int ib = 0; ib < count; ib++) {
    float *out = &output_ptr[(ib * in_h * in_w * channel)];
    const float *inPtr = &input_ptr[(ib * output_h * output_w * channel)];
    // iterate over yt
    for (int yh = 0; yh < output_h; yh++) {
      int over_h = pad_h - yh * stride_h;
      int kh_s = MSMAX(0, over_h);
      int kh_e = MSMIN(win_h, in_h + over_h);
      for (int yw = 0; yw < output_w; yw++) {
        int over_w = pad_w - yw * stride_w;
        int kw_s = MSMAX(0, over_w);
        int kw_e = MSMIN(win_w, in_w + over_w);
        int ic = 0;
        for (; ic < channel - 4; ic += 4) {
          int idx = (yw + yh * output_w) * channel + ic;
#ifdef ENABLE_ARM
          float32x4_t in = vld1q_f32(inPtr + idx);
          float32x4_t delta = vmulq_f32(in, factor);
#else
          float delta[4] = {inPtr[idx], inPtr[idx + 1], inPtr[idx + 2], inPtr[idx + 3]};
          for (int i = 0; i < 4; i++) delta[i] *= kk;
#endif
          for (int kh = kh_s; kh < kh_e; kh++) {
            int xh = yh * stride_h + kh - pad_h;
            for (int kw = kw_s; kw < kw_e; kw++) {
              int xw = yw * stride_w + kw - pad_w;
#ifdef ENABLE_ARM
              float *out_vec = out + (xw + in_w * xh) * channel + ic;
              float32x4_t outr = vld1q_f32(out + (xw + in_w * xh) * channel + ic);
              float32x4_t outs = vaddq_f32(outr, delta);
              vst1q_f32(out_vec, outs);
#else

              for (int i = 0; i < 4; i++) {
                out[(xw + in_w * xh) * channel + ic + i] += ((float *)&delta)[i];
              }
#endif
            }
          }
        }
        for (; ic < channel; ic++) {
          int idx = (yw + yh * output_w) * channel + ic;
          float delta = inPtr[idx] * kk;
          for (int kh = kh_s; kh < kh_e; kh++) {
            int xh = yh * stride_h + kh - pad_h;
            for (int kw = kw_s; kw < kw_e; kw++) {
              int xw = yw * stride_w + kw - pad_w;
              out[(xw + in_w * xh) * channel + ic] += delta;
            }
          }
        }
      }
    }
  }
}

#ifdef ENABLE_ARM
static int32x4_t MaxIndex(float32x4_t in, float32x4_t *max, int32x4_t index, int32x4_t prev_index) {
  uint32x4_t res = vcgtq_f32(in, *max);
  int32x4_t m_index = vbslq_s32(res, index, prev_index);
  *max = vbslq_f32(res, in, *max);
  return m_index;
}
#endif

void MaxPoolingGrad(const float *input_ptr, const float *dy_ptr, float *output_ptr, int output_batch,
                    PoolingParameter *pooling_param) {
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

  for (int ib = 0; ib < output_batch; ib++) {
    float *out = &output_ptr[(ib * in_h * in_w * channel)];
    const float *inPtr = &input_ptr[(ib * in_h * in_w * channel)];
    const float *dyPtr = &dy_ptr[(ib * output_h * output_w * channel)];
    for (int yh = 0; yh < output_h; yh++) {
      int over_h = pad_h - yh * stride_h;
      int kh_s = MSMAX(0, over_h);
      int kh_e = MSMIN(win_h, in_h + over_h);
      for (int yw = 0; yw < output_w; yw++) {
        int over_w = pad_w - yw * stride_w;
        int kw_s = MSMAX(0, over_w);
        int kw_e = MSMIN(win_w, in_w + over_w);
        int ic = 0;
        for (; ic < (channel & ~3); ic += 4) {
          int idx = (yw + yh * output_w) * channel + ic;
#ifdef ENABLE_ARM
          uint32x4_t max_idx = vdupq_n_u32(0);
          float32x4_t max_val = vdupq_n_f32(-FLT_MAX);
          float32x4_t delta = vld1q_f32(dyPtr + idx);
#else
          float delta[4] = {dyPtr[idx], dyPtr[idx + 1], dyPtr[idx + 2], dyPtr[idx + 3]};
          float max_val[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
          int max_idx[4] = {0};
#endif
          for (int kh = kh_s; kh < kh_e; kh++) {
            int xh = yh * stride_h + kh - pad_h;
            for (int kw = kw_s; kw < kw_e; kw++) {
              int xw = yw * stride_w + kw - pad_w;
              int val_idx = (xw + in_w * xh) * channel + ic;
#ifdef ENABLE_ARM
              uint32x4_t index = {val_idx, val_idx + 1, val_idx + 2, val_idx + 3};
              float32x4_t in = vld1q_f32(inPtr + val_idx);
              max_idx = MaxIndex(in, &max_val, index, max_idx);
#else
              float val[4] = {inPtr[val_idx], inPtr[val_idx + 1], inPtr[val_idx + 2], inPtr[val_idx + 3]};
              for (int i = 0; i < 4; i++) {
                if (val[i] > max_val[i]) {
                  max_val[i] = val[i];
                  max_idx[i] = val_idx + i;
                }
              }
#endif
            }
          }
          for (int i = 0; i < 4; i++) {
            out[((int *)&max_idx)[i]] += ((float *)&delta)[i];
          }
        }
        for (; ic < channel; ic++) {
          float max_val = -FLT_MAX;
          int max_idx = 0;
          int idx = (yw + yh * output_w) * channel + ic;
          float delta = dyPtr[idx];
          for (int kh = kh_s; kh < kh_e; kh++) {
            int xh = yh * stride_h + kh - pad_h;
            for (int kw = kw_e; kw < kw_s; kw++) {
              int xw = yw * stride_w + kw - pad_w;
              int val_idx = (xw + in_w * xh) * channel + ic;
              float val = inPtr[val_idx];
              if (val > max_val) {
                max_val = val;
                max_idx = val_idx;
              }
            }
          }
          out[max_idx] += delta;
        }
      }
    }
  }
}
