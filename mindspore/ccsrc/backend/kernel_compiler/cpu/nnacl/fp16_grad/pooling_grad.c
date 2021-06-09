/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "nnacl/fp16_grad/pooling_grad.h"

void AvgPoolingFp16Grad(const float16_t *input_ptr, float16_t *output_ptr, int count, PoolingParameter *pooling_param) {
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

  const float16_t kk = 1.0f / (float16_t)(win_h * win_w);
#if ENABLE_NEON
  const float16x4_t factor = vdup_n_f16(kk);
#endif
  for (int ib = 0; ib < count; ib++) {
    float16_t *out = &output_ptr[(ib * in_h * in_w * channel)];
    const float16_t *inPtr = &input_ptr[(ib * output_h * output_w * channel)];
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
#ifdef ENABLE_NEON
          float16x4_t in = vld1_f16(inPtr + idx);
          float16x4_t delta = vmul_f16(in, factor);
#else
          float16_t delta[4] = {inPtr[idx], inPtr[idx + 1], inPtr[idx + 2], inPtr[idx + 3]};
          for (int i = 0; i < 4; i++) delta[i] *= kk;
#endif
          for (int kh = kh_s; kh < kh_e; kh++) {
            int xh = yh * stride_h + kh - pad_h;
            for (int kw = kw_s; kw < kw_e; kw++) {
              int xw = yw * stride_w + kw - pad_w;
#ifdef ENABLE_NEON
              float16_t *out_vec = out + (xw + in_w * xh) * channel + ic;
              float16x4_t outr = vld1_f16(out + (xw + in_w * xh) * channel + ic);
              float16x4_t outs = vadd_f16(outr, delta);
              vst1_f16(out_vec, outs);
#else

              for (int i = 0; i < 4; i++) {
                out[(xw + in_w * xh) * channel + ic + i] += ((float16_t *)&delta)[i];
              }
#endif
            }
          }
        }
        for (; ic < channel; ic++) {
          int idx = (yw + yh * output_w) * channel + ic;
          float16_t delta = inPtr[idx] * kk;
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

#ifdef ENABLE_NEON
static int32x4_t MaxIndex(float16x4_t in, float16x4_t *max, uint32x4_t index, uint32x4_t prev_index) {
  uint16x4_t res = vcgt_f16(in, *max);
  int16x4_t tmp = vreinterpret_s16_u16(res);
  uint32x4_t res_tmp = vreinterpretq_u32_s32(vmovl_s16(tmp));
  int32x4_t m_index = vbslq_s32(res_tmp, index, prev_index);
  *max = vbsl_f16(res, in, *max);
  return m_index;
}
#endif

void MaxPoolingFp16Grad(const float16_t *input_ptr, const float16_t *dy_ptr, float16_t *output_ptr, int output_batch,
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
    float16_t *out = &output_ptr[(ib * in_h * in_w * channel)];
    const float16_t *inPtr = &input_ptr[(ib * in_h * in_w * channel)];
    const float16_t *dyPtr = &dy_ptr[(ib * output_h * output_w * channel)];
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
#ifdef ENABLE_NEON
          uint32x4_t max_idx = vdupq_n_u32(0);
          float16x4_t max_val = vdup_n_f16(-FLT16_MAX);
          float16x4_t delta = vld1_f16(dyPtr + idx);
#else
          float16_t delta[4] = {dyPtr[idx], dyPtr[idx + 1], dyPtr[idx + 2], dyPtr[idx + 3]};
          float16_t max_val[4] = {-FLT16_MAX, -FLT16_MAX, -FLT16_MAX, -FLT16_MAX};
          uint max_idx[4] = {0};
#endif
          for (int kh = kh_s; kh < kh_e; kh++) {
            int xh = yh * stride_h + kh - pad_h;
            for (int kw = kw_s; kw < kw_e; kw++) {
              int xw = yw * stride_w + kw - pad_w;
              int val_idx = (xw + in_w * xh) * channel + ic;
#ifdef ENABLE_NEON
              uint32x4_t index = {val_idx, val_idx + 1, val_idx + 2, val_idx + 3};
              float16x4_t in = vld1_f16(inPtr + val_idx);
              max_idx = MaxIndex(in, &max_val, index, max_idx);
#else
              float16_t val[4] = {inPtr[val_idx], inPtr[val_idx + 1], inPtr[val_idx + 2], inPtr[val_idx + 3]};
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
            out[((int *)&max_idx)[i]] += ((float16_t *)&delta)[i];
          }
        }
        for (; ic < channel; ic++) {
          float16_t max_val = -FLT16_MAX;
          int max_idx = 0;
          int idx = (yw + yh * output_w) * channel + ic;
          float16_t delta = dyPtr[idx];
          for (int kh = kh_s; kh < kh_e; kh++) {
            int xh = yh * stride_h + kh - pad_h;
            for (int kw = kw_e; kw < kw_s; kw++) {
              int xw = yw * stride_w + kw - pad_w;
              int val_idx = (xw + in_w * xh) * channel + ic;
              float16_t val = inPtr[val_idx];
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
