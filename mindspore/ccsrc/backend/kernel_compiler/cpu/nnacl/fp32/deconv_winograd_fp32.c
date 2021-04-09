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

#include "nnacl/fp32/deconv_winograd_fp32.h"

int PackDeConvWgDataFp32(const float *nhwc_weight, DeConvComputeUnit *unit, const ConvParameter *conv_param,
                         const DeConvParam *deconv_param) {
  int tmp_kernel_plane = unit->w_size_ * unit->h_size_;
  int size = conv_param->input_channel_ * conv_param->output_channel_ * tmp_kernel_plane;
  float *current_unit_weight = (float *)malloc(size * sizeof(float));
  if (current_unit_weight == NULL) {
    return NNACL_NULL_PTR;
  }
  for (int ic = 0; ic < conv_param->input_channel_; ic++) {
    const float *src_ic = nhwc_weight + deconv_param->kernel_plane_ * conv_param->output_channel_ * ic;
    float *dst_ic = current_unit_weight + tmp_kernel_plane * conv_param->output_channel_ * ic;
    for (int uhi = 0; uhi < unit->h_size_; uhi++) {
      for (int uwi = 0; uwi < unit->w_size_; uwi++) {
        int src_h_offset = unit->h_start_ + uhi * conv_param->stride_h_;
        int src_w_offset = unit->w_start_ + uwi * conv_param->stride_w_;
        const float *src_hw =
          src_ic + (src_h_offset * conv_param->kernel_w_ + src_w_offset) * conv_param->output_channel_;
        float *dst_hw = dst_ic + (uhi * unit->w_size_ + uwi) * conv_param->output_channel_;
        memcpy(dst_hw, src_hw, conv_param->output_channel_ * sizeof(float));
      }
    }
  }

  if (unit->use_winograd_) {
    /* Generate winograd  */
    float matrix_g[64], matrix_a[64], matrix_b[64];
    float matrix_gt[64], matrix_at[64], matrix_bt[64];
    int ret = CookToomFilter(matrix_a, matrix_at, matrix_b, matrix_bt, matrix_g, matrix_gt, 0.5f,
                             DECONV_WINOGRAD_DEFAULT_UNIT, unit->h_size_);
    if (ret != NNACL_OK) {
      free(current_unit_weight);
      current_unit_weight = NULL;
      return NNACL_ERRCODE_WINOGRAD_GENERATOR_ERROR;
    }

    /* winograd AT */
    unit->winograd_.AT_ = malloc(unit->winograd_.i_ * unit->winograd_.o_ * sizeof(float));
    if (unit->winograd_.AT_ == NULL) {
      if (current_unit_weight != NULL) {
        free(current_unit_weight);
        current_unit_weight = NULL;
      }
      return NNACL_NULL_PTR;
    }
    memcpy(unit->winograd_.AT_, matrix_at, unit->winograd_.i_ * unit->winograd_.o_ * sizeof(float));

    /* winograd BT */
    unit->winograd_.BT_ = malloc(unit->winograd_.o_ * unit->winograd_.o_ * sizeof(float));
    if (unit->winograd_.BT_ == NULL) {
      if (current_unit_weight != NULL) {
        free(current_unit_weight);
        current_unit_weight = NULL;
      }
      if (unit->winograd_.AT_ != NULL) {
        free(unit->winograd_.AT_);
        unit->winograd_.AT_ = NULL;
      }
      return NNACL_NULL_PTR;
    }
    memcpy(unit->winograd_.BT_, matrix_bt, unit->winograd_.o_ * unit->winograd_.o_ * sizeof(float));

    /* winograd Weight */
    size = conv_param->input_channel_ * conv_param->output_channel_ * unit->winograd_.kh_ * unit->winograd_.kw_;
    float *winograd_unit_weight = (float *)malloc(size * sizeof(float));
    if (winograd_unit_weight == NULL) {
      if (current_unit_weight != NULL) {
        free(current_unit_weight);
        current_unit_weight = NULL;
      }
      if (unit->winograd_.AT_ != NULL) {
        free(unit->winograd_.AT_);
        unit->winograd_.AT_ = NULL;
      }
      if (unit->winograd_.BT_ != NULL) {
        free(unit->winograd_.BT_);
        unit->winograd_.BT_ = NULL;
      }
      return NNACL_NULL_PTR;
    }
    WinogradWeightTransform(current_unit_weight, winograd_unit_weight, matrix_g, matrix_gt, C4NUM, unit->winograd_.kh_,
                            unit->h_size_, conv_param->output_channel_, conv_param->input_channel_, false);

    /* reset weight data & info */
    tmp_kernel_plane = unit->winograd_.kh_ * unit->winograd_.kw_;
    free(current_unit_weight);
    current_unit_weight = NULL;
    current_unit_weight = winograd_unit_weight;
    winograd_unit_weight = NULL;
  }

  /* trans mhwc -> hw1:k1-knc0-c4:k1-knc5-c8:hw2:k1-knc0-c4:k1 */
  float *dst_weight = (float *)unit->weight_;
  size = deconv_param->ic_up4_ * deconv_param->oc_up4_ * tmp_kernel_plane;
  memset(dst_weight, 0, size * sizeof(float));
  for (int ic = 0; ic < conv_param->input_channel_; ic++) {
    for (int oc = 0; oc < conv_param->output_channel_; oc++) {
      int oc4div = oc / C4NUM, oc4mod = oc % C4NUM;
      for (int upi = 0; upi < tmp_kernel_plane; upi++) {
        int src_index = ic * conv_param->output_channel_ * tmp_kernel_plane + upi * conv_param->output_channel_ + oc;
        int dst_index = upi * deconv_param->oc_up4_ * deconv_param->ic_up4_ + oc4div * C4NUM * deconv_param->ic_up4_ +
                        ic * C4NUM + oc4mod;
        dst_weight[dst_index] = current_unit_weight[src_index];
      }
    }
  }

  if (current_unit_weight != NULL) {
    free(current_unit_weight);
    current_unit_weight = NULL;
  }
  return NNACL_OK;
}

void DeConvWgInputPack(const float *src_ptr, float *dst_ptr, int channel, int stride) {
  int ic4div = channel / C4NUM;
  int ic4mod = channel % C4NUM;
  const float *src = src_ptr;
  float *dst = dst_ptr;

  for (int ic = 0; ic < ic4div; ic++) {
#ifdef ENABLE_ARM
    vst1q_f32(dst, vld1q_f32(src));
#else
    memcpy(dst, src, C4NUM * sizeof(float));
#endif
    dst += stride;
    src += C4NUM;
  }

  if (ic4mod != 0) {
    int ic_res = 0;
    for (; ic_res < ic4mod; ic_res++) {
      dst[ic_res] = src[ic_res];
    }
    for (; ic_res < C4NUM; ic_res++) {
      dst[ic_res] = 0;
    }
  }
  return;
}

#if !defined(ENABLE_ARM) && !defined(ENABLE_SSE)
void TiledC4MatmulFp32(float *dst, const float *src, const float *weight, size_t cal_num, size_t ic4, size_t oc4) {
  int dx, sz, dz;
  const int src_depth_step = 4 * DECONV_WINOGRAD_DEFAULT_TILE;
  for (dz = 0; dz < oc4; ++dz) {
    float *dst_z = dst + dz * cal_num;
    const float *weight_dz = weight + dz * ic4 * 16;
    for (dx = 0; dx < DECONV_WINOGRAD_DEFAULT_TILE; ++dx) {
      float *dst_x = dst_z + dx * 4;
      dst_x[0] = 0.0f;
      dst_x[1] = 0.0f;
      dst_x[2] = 0.0f;
      dst_x[3] = 0.0f;
      const float *src_dx = src + 4 * dx;
      for (sz = 0; sz < ic4; ++sz) {
        const float *src_z = src_dx + sz * src_depth_step;
        const float *weight_z = weight_dz + sz * 16;
        for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 4; ++j) {
            dst_x[j] += src_z[i] * weight_z[4 * i + j];
          }
        }
      }
    }
  }
}
#endif

#ifdef ENABLE_ARM32
void DeConvWgMergeArm32(const float *src_ptr, float *dst_ptr, size_t src_step, size_t dst_step) {
  asm volatile(
    "mov r11, %[src_ptr]\n"
    "mov r8, %[dst_ptr]\n"
    "mov r10, r8\n"

    "vld1.32 {q0}, [r11], %[src_step]\n"
    "vld1.32 {q1}, [r8], %[dst_step]\n"
    "vld1.32 {q2}, [r11], %[src_step]\n"
    "vld1.32 {q3}, [r8], %[dst_step]\n"

    "vadd.f32 q0, q0, q1\n"
    "vld1.32 {q8}, [r11], %[src_step]\n"
    "vadd.f32 q2, q2, q3\n"

    "vst1.32 {q0}, [r10], %[dst_step]\n"
    "vst1.32 {q2}, [r10], %[dst_step]\n"

    "vld1.32 {q9}, [r8], %[dst_step]\n"

    "vld1.32 {q10}, [r11], %[src_step]\n"

    "vadd.f32 q8, q8, q9\n"
    "vld1.32 {q11}, [r8], %[dst_step]\n"
    "vadd.f32 q10, q10, q11\n"

    "vld1.32 {q0}, [r11], %[src_step]\n"
    "vst1.32 {q8}, [r10], %[dst_step]\n"
    "vst1.32 {q10}, [r10], %[dst_step]\n"

    "vld1.32 {q1}, [r8], %[dst_step]\n"

    "vld1.32 {q2}, [r11], %[src_step]\n"
    "vld1.32 {q3}, [r8], %[dst_step]\n"

    "vadd.f32 q0, q0, q1\n"
    "vadd.f32 q2, q2, q3\n"

    "vst1.32 {q0}, [r10], %[dst_step]\n"
    "vst1.32 {q2}, [r10], %[dst_step]\n"

    "vld1.32 {q8}, [r11], %[src_step]\n"
    "vld1.32 {q9}, [r8], %[dst_step]\n"

    "vld1.32 {q10}, [r11], %[src_step]\n"
    "vld1.32 {q11}, [r8], %[dst_step]\n"

    "vadd.f32 q8, q8, q9\n"
    "vadd.f32 q10, q10, q11\n"

    "vst1.32 {q8}, [r10], %[dst_step]\n"
    "vst1.32 {q10}, [r10], %[dst_step]\n"

    :
    : [ src_ptr ] "r"(src_ptr), [ dst_ptr ] "r"(dst_ptr), [ src_step ] "r"(src_step), [ dst_step ] "r"(dst_step)
    : "r8", "r10", "r11", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
  return;
}
#endif

void DeConvWgMerge(const float *src, float *dst, size_t src_stride, size_t dst_stride, size_t count) {
  const float *src_ptr = src;
  float *dst_ptr = dst;
  size_t cuont8 = count / C8NUM * C8NUM;
  int i = 0;
  for (; i < cuont8; i += 8) {
#ifdef ENABLE_ARM64
    size_t src_step = src_stride * sizeof(float);
    size_t dst_step = dst_stride * sizeof(float);
    asm volatile(
      "mov x7, %[src_ptr]\n"
      "mov x8, %[dst_ptr]\n"
      "mov x10, x8\n"

      "ld1 {v0.4s}, [x7], %[src_step]\n"
      "ld1 {v1.4s}, [x8], %[dst_step]\n"

      "ld1 {v2.4s}, [x7], %[src_step]\n"
      "ld1 {v3.4s}, [x8], %[dst_step]\n"

      "fadd v0.4s, v0.4s, v1.4s\n"
      "ld1 {v4.4s}, [x7], %[src_step]\n"
      "fadd v2.4s, v2.4s, v3.4s\n"

      "st1 {v0.4s}, [x10], %[dst_step]\n"
      "st1 {v2.4s}, [x10], %[dst_step]\n"

      "ld1 {v5.4s}, [x8], %[dst_step]\n"

      "ld1 {v6.4s}, [x7], %[src_step]\n"

      "fadd v4.4s, v4.4s, v5.4s\n"
      "ld1 {v7.4s}, [x8], %[dst_step]\n"
      "fadd v6.4s, v6.4s, v7.4s\n"

      "ld1 {v0.4s}, [x7], %[src_step]\n"
      "st1 {v4.4s}, [x10], %[dst_step]\n"
      "st1 {v6.4s}, [x10], %[dst_step]\n"

      "ld1 {v1.4s}, [x8], %[dst_step]\n"

      "ld1 {v2.4s}, [x7], %[src_step]\n"
      "ld1 {v3.4s}, [x8], %[dst_step]\n"

      "fadd v0.4s, v0.4s, v1.4s\n"
      "fadd v2.4s, v2.4s, v3.4s\n"

      "st1 {v0.4s}, [x10], %[dst_step]\n"
      "st1 {v2.4s}, [x10], %[dst_step]\n"

      "ld1 {v4.4s}, [x7], %[src_step]\n"
      "ld1 {v5.4s}, [x8], %[dst_step]\n"

      "ld1 {v6.4s}, [x7], %[src_step]\n"
      "ld1 {v7.4s}, [x8], %[dst_step]\n"

      "fadd v4.4s, v4.4s, v5.4s\n"
      "fadd v6.4s, v6.4s, v7.4s\n"

      "st1 {v4.4s}, [x10], %[dst_step]\n"
      "st1 {v6.4s}, [x10], %[dst_step]\n"

      :
      : [ src_ptr ] "r"(src_ptr), [ dst_ptr ] "r"(dst_ptr), [ src_step ] "r"(src_step), [ dst_step ] "r"(dst_step)
      : "x7", "x8", "x10", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#elif ENABLE_ARM32
    size_t src_step = src_stride * sizeof(float);
    size_t dst_step = dst_stride * sizeof(float);
    DeConvWgMergeArm32(src_ptr, dst_ptr, src_step, dst_step);
#else
    for (int j = 0; j < 8; j++) {
      const float *s = src_ptr + j * src_stride;
      float *d = dst_ptr + j * dst_stride;
      for (int k = 0; k < 4; k++) {
        d[k] += s[k];
      }
    }
#endif
    src_ptr += 8 * src_stride;
    dst_ptr += 8 * dst_stride;
  }
  for (; i < count; i++) {
#ifdef ENABLE_ARM
    float32x4_t src_data = vld1q_f32(src_ptr);
    float32x4_t dst_data = vld1q_f32(dst_ptr);
    dst_data = vaddq_f32(src_data, dst_data);
    vst1q_f32(dst_ptr, dst_data);
#else
    for (int j = 0; j < 4; j++) {
      dst_ptr[j] += src_ptr[j];
    }
#endif
    src_ptr += src_stride;
    dst_ptr += dst_stride;
  }
  return;
}

void DeConvWgCalWgFp32(const float *tile_in, float *tile_out, const float *weight_buf, float *tmp_buf,
                       const float *at_buf, float *a_mid_buf, float *trans_a_buf, bool *transferred,
                       const float *bt_buf, float *b_tmp_buf, int unit_size, int w_start, int h_start,
                       const ConvParameter *conv_param, const DeConvParam *deconv_param) {
  int winograd_plane = unit_size * unit_size;
  if (!transferred[unit_size]) {
    WinogradTransLeft(tile_in, at_buf, a_mid_buf, DECONV_WINOGRAD_DEFAULT_UNIT, unit_size, DECONV_WINOGRAD_DEFAULT_UNIT,
                      deconv_param->ic_div4_ * DECONV_WINOGRAD_DEFAULT_TILE);
    WinogradTransRight(a_mid_buf, at_buf, trans_a_buf, unit_size, unit_size, DECONV_WINOGRAD_DEFAULT_UNIT,
                       deconv_param->ic_div4_ * DECONV_WINOGRAD_DEFAULT_TILE);
    transferred[unit_size] = true;
  }

  for (int index = 0; index < winograd_plane; index++) {
    float *src = trans_a_buf + index * DECONV_WINOGRAD_DEFAULT_TILE * deconv_param->ic_up4_;
    float *dst = tmp_buf + index * deconv_param->oc_up4_ * DECONV_WINOGRAD_DEFAULT_TILE;
    const float *weight = weight_buf + index * deconv_param->ic_up4_ * deconv_param->oc_up4_;
    TiledC4MatmulFp32(dst, src, weight, DECONV_WINOGRAD_DEFAULT_TILE * C4NUM, deconv_param->ic_div4_,
                      deconv_param->oc_div4_);
  }

  WinogradTransLeft(tmp_buf, bt_buf, b_tmp_buf, unit_size, unit_size, unit_size,
                    deconv_param->oc_div4_ * DECONV_WINOGRAD_DEFAULT_TILE);
  WinogradTransRight(b_tmp_buf, bt_buf, tmp_buf, unit_size, unit_size, unit_size,
                     deconv_param->oc_div4_ * DECONV_WINOGRAD_DEFAULT_TILE);

  // Add to dest
  for (int uhi = 0; uhi < unit_size; uhi++) {
    int h_index = uhi * conv_param->stride_h_ + h_start;
    for (int uwi = 0; uwi < unit_size; uwi++) {
      int w_index = uwi * conv_param->stride_w_ + w_start;

      float *dst = tile_out + w_index * DECONV_WINOGRAD_DEFAULT_TILE * deconv_param->oc_up4_ +
                   h_index * deconv_param->out_tile_w_ * DECONV_WINOGRAD_DEFAULT_TILE * deconv_param->oc_up4_;
      float *src = tmp_buf + (uwi + uhi * unit_size) * DECONV_WINOGRAD_DEFAULT_TILE * deconv_param->oc_up4_;
      DeConvWgMerge(src, dst, 4, 4, DECONV_WINOGRAD_DEFAULT_TILE * deconv_param->oc_div4_);
    }
  }
  return;
}

void DeConvWgCalCommFp32(const float *tile_in, float *tile_out, const float *weight, float *tmp_buf, int h_start,
                         int w_start, int h_size, int w_size, const ConvParameter *conv_param,
                         const DeConvParam *deconv_param) {
  int count = deconv_param->oc_div4_ * w_size * h_size;
  int in_stride = DECONV_WINOGRAD_DEFAULT_TILE * deconv_param->ic_up4_;
  int out_stride = DECONV_WINOGRAD_DEFAULT_TILE * deconv_param->oc_up4_;

  for (int hi = 0; hi < DECONV_WINOGRAD_DEFAULT_UNIT; hi++) {
    for (int wi = 0; wi < DECONV_WINOGRAD_DEFAULT_UNIT; wi++) {
      const float *src_in = tile_in + (wi + hi * DECONV_WINOGRAD_DEFAULT_UNIT) * in_stride;
      TiledC4MatmulFp32(tmp_buf, src_in, weight, DECONV_WINOGRAD_DEFAULT_TILE * 4, deconv_param->ic_div4_, count);

      for (int uhi = 0; uhi < h_size; uhi++) {
        for (int uwi = 0; uwi < w_size; uwi++) {
          int w_index = (wi + uwi) * conv_param->stride_w_ + w_start;
          int h_index = (hi + uhi) * conv_param->stride_h_ + h_start;
          float *dst = tile_out + h_index * out_stride * deconv_param->out_tile_w_ + w_index * out_stride;
          float *src = tmp_buf + (uwi + uhi * w_size) * out_stride;
          DeConvWgMerge(src, dst, 4, 4, DECONV_WINOGRAD_DEFAULT_TILE * deconv_param->oc_div4_);
        }
      }
    }
  }

  return;
}

void DeconvWg(const float *nhwc_input_, float *tile_in, float *tile_out, int start_index, int calculate_count,
              const ConvParameter *conv_param, DeConvParam *deconv_param, int task_id) {
  /* pack tile input */
  int tile_in_unit_stride = deconv_param->ic_up4_ * DECONV_WINOGRAD_DEFAULT_TILE;
#ifdef ENABLE_ARM
  float32x4_t zero = vdupq_n_f32(0.0f);
#endif
  for (int unit_index = 0; unit_index < calculate_count; unit_index++) {
    int plane_index = start_index + unit_index;
    int w_unit_index = plane_index % deconv_param->in_tile_w_count_;
    int h_unit_index = plane_index / deconv_param->in_tile_w_count_;
    int w_start = w_unit_index * DECONV_WINOGRAD_DEFAULT_UNIT;
    int h_start = h_unit_index * DECONV_WINOGRAD_DEFAULT_UNIT;

    float *dst_unit = tile_in + unit_index * C4NUM;
    for (int hi = 0; hi < DECONV_WINOGRAD_DEFAULT_UNIT; hi++) {
      for (int wi = 0; wi < DECONV_WINOGRAD_DEFAULT_UNIT; wi++) {
        float *dst = dst_unit + (wi + hi * DECONV_WINOGRAD_DEFAULT_UNIT) * tile_in_unit_stride;
        int w_index = w_start + wi;
        int h_index = h_start + hi;
        if (w_index >= conv_param->input_w_ || h_index >= conv_param->input_h_) {
          for (int ic4_index = 0; ic4_index < deconv_param->ic_div4_; ic4_index++) {
#ifdef ENABLE_ARM
            vst1q_f32(dst + ic4_index * DECONV_WINOGRAD_DEFAULT_TILE * C4NUM, zero);
#else
            for (int i = 0; i < 4; i++) {
              dst[C4NUM * DECONV_WINOGRAD_DEFAULT_TILE * ic4_index + i] = 0;
            }
#endif
          }
          continue;
        }

        const float *src = nhwc_input_ + (w_index + h_index * conv_param->input_w_) * conv_param->input_channel_;
        DeConvWgInputPack(src, dst, conv_param->input_channel_, DECONV_WINOGRAD_DEFAULT_TILE * C4NUM);
      }
    }
  }

  /* compute */
  bool transferred[DECONV_WINOGRAD_BUFFER_COUNT] = {false};
  for (int i = 0; i < deconv_param->compute_size_; i++) {
    DeConvComputeUnit *unit = &deconv_param->compute_units_[i];
    if (unit->use_winograd_) {
      float *tmp_buf = (float *)unit->tmp_buffer_ + task_id * unit->winograd_.kh_ * unit->winograd_.kw_ *
                                                      deconv_param->oc_div4_ * DECONV_WINOGRAD_DEFAULT_TILE * C4NUM;

      /* winograd a buffer */
      DeConvWgABuffer *wg_buf = &deconv_param->a_buffer_[unit->winograd_.kh_];
      float *wg_mid_a_buf = (float *)wg_buf->middle_buffer_ + task_id * unit->winograd_.kw_ * unit->winograd_.kh_ *
                                                                DECONV_WINOGRAD_DEFAULT_TILE * deconv_param->ic_up4_;
      float *wg_dst_a_buf = (float *)wg_buf->dest_buffer_ + task_id * unit->winograd_.kw_ * unit->winograd_.kh_ *
                                                              DECONV_WINOGRAD_DEFAULT_TILE * deconv_param->ic_up4_;
      float *tmp_b_buf = (float *)unit->winograd_.b_buffer_ + task_id * unit->winograd_.kh_ * unit->winograd_.kw_ *
                                                                deconv_param->oc_up4_ * DECONV_WINOGRAD_DEFAULT_TILE;
      DeConvWgCalWgFp32(tile_in, tile_out, (float *)unit->weight_, tmp_buf, unit->winograd_.AT_, wg_mid_a_buf,
                        wg_dst_a_buf, transferred, unit->winograd_.BT_, tmp_b_buf, unit->winograd_.kh_, unit->w_start_,
                        unit->h_start_, conv_param, deconv_param);
    } else {
      float *tmp_buf = (float *)unit->tmp_buffer_ + task_id * deconv_param->oc_div4_ * unit->w_size_ * unit->h_size_ *
                                                      DECONV_WINOGRAD_DEFAULT_TILE * C4NUM;
      DeConvWgCalCommFp32(tile_in, tile_out, (float *)unit->weight_, tmp_buf, unit->h_start_, unit->w_start_,
                          unit->h_size_, unit->w_size_, conv_param, deconv_param);
    }
  }
  return;
}

void DeconvWgPost(const float *tile_out, float *nc4hw4_output, const ConvParameter *conv_param,
                  const DeConvParam *deconv_param, int calculate_count, int tile_index) {
  /* merge */
  int src_unit_stride = deconv_param->oc_up4_ * DECONV_WINOGRAD_DEFAULT_TILE;

  int src_stride = DECONV_WINOGRAD_DEFAULT_TILE * C4NUM;
  int dst_stride = conv_param->output_w_ * conv_param->output_h_ * C4NUM;

  for (int index = 0; index < calculate_count; ++index) {
    const float *src_start = tile_out + index * C4NUM;

    int plane_index = tile_index * DECONV_WINOGRAD_DEFAULT_TILE + index;
    int w_unit_index = plane_index % deconv_param->in_tile_w_count_;
    int h_unit_index = plane_index / deconv_param->in_tile_w_count_;
    int w_start = w_unit_index * DECONV_WINOGRAD_DEFAULT_UNIT * conv_param->stride_w_ - conv_param->pad_l_;
    int h_start = h_unit_index * DECONV_WINOGRAD_DEFAULT_UNIT * conv_param->stride_h_ - conv_param->pad_u_;
    float *dst_start = nc4hw4_output + h_start * conv_param->output_w_ * C4NUM + w_start * C4NUM;

    int merge_w_start = MSMAX(-w_start, 0);
    int merge_h_start = MSMAX(-h_start, 0);
    int merge_h_end = MSMIN(deconv_param->out_tile_h_, conv_param->output_h_ - h_start);
    int merge_w_end = MSMIN(deconv_param->out_tile_w_, conv_param->output_w_ - w_start);

    for (int hi = merge_h_start; hi < merge_h_end; hi++) {
      for (int wi = merge_w_start; wi < merge_w_end; wi++) {
        const float *src = src_start + (hi * deconv_param->out_tile_w_ + wi) * src_unit_stride;
        float *dst = dst_start + (hi * conv_param->output_w_ + wi) * C4NUM;
        DeConvWgMerge(src, dst, src_stride, dst_stride, deconv_param->oc_div4_);
      }
    }
  }
  return;
}
