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
#include "nnacl/fp32/instance_norm_fp32.h"
#include <math.h>
#include "nnacl/errorcode.h"
#include "nnacl/op_base.h"
#include "nnacl/intrinsics/ms_simd_instructions.h"

int InstanceNorm(const float *src_data, float *dst_data, const float *gamma_data, const float *beta_data,
                 const InstanceNormParameter *param, size_t task_id) {
  NNACL_CHECK_NULL_RETURN_ERR(src_data);
  NNACL_CHECK_NULL_RETURN_ERR(dst_data);
  NNACL_CHECK_NULL_RETURN_ERR(param->op_parameter_.thread_num_)
  int channel_step = UP_DIV(param->channel_, param->op_parameter_.thread_num_);
  int channel_begin = (int)(task_id)*channel_step;
  int channel_end = MSMIN(channel_begin + channel_step, param->channel_);

  for (int b = 0; b < param->batch_; b++) {
    const float *src_b = src_data + b * param->channel_ * param->inner_size_;
    float *dst_b = dst_data + b * param->channel_ * param->inner_size_;
    for (int c = channel_begin; c < channel_end; c++) {
      const float *src = src_b + c * param->inner_size_;
      float *dst = dst_b + c * param->inner_size_;
      double mean = 0.0f;
      double square_mean = 0.0f;

      int index = 0;
#if defined(ENABLE_AVX)
      for (; index <= param->inner_size_ - C8NUM; index += C8NUM) {
        __m256 srcv = _mm256_loadu_ps(src + index);
        __m256 squarev = _mm256_mul_ps(srcv, srcv);
        __m128 src128 = _mm_add_ps(_mm256_extractf128_ps(srcv, 0), _mm256_extractf128_ps(srcv, 1));
        __m128 square128 = _mm_add_ps(_mm256_extractf128_ps(squarev, 0), _mm256_extractf128_ps(squarev, 1));
        for (int i = 0; i < C4NUM; ++i) {
          mean += src128[i];
          square_mean += square128[i];
        }
      }
#endif

#if defined(ENABLE_NEON) || defined(ENABLE_SSE)
      for (; index <= param->inner_size_ - C4NUM; index += C4NUM) {
        MS_FLOAT32X4 srcv = MS_LDQ_F32(src + index);
        MS_FLOAT32X4 squarev = MS_MULQ_F32(srcv, srcv);
#ifdef ENABLE_ARM64
        mean += vaddvq_f32(srcv);
        square_mean += vaddvq_f32(squarev);
#elif defined(ENABLE_SSE)
        for (int i = 0; i < C4NUM; ++i) {
          mean += MS_F32X4_GETI(srcv, i);
          square_mean += MS_F32X4_GETI(squarev, i);
        }
#else
        float32x2_t src_add2 = vadd_f32(vget_low_f32(srcv), vget_high_f32(srcv));
        float32x2_t src_add4 = vpadd_f32(src_add2, src_add2);
        mean += vget_lane_f32(src_add4, 0);
        float32x2_t square_add2 = vadd_f32(vget_low_f32(squarev), vget_high_f32(squarev));
        float32x2_t square_add4 = vpadd_f32(square_add2, square_add2);
        square_mean += vget_lane_f32(square_add4, 0);
#endif
      }
#endif
      for (; index < param->inner_size_; index++) {
        mean += src[index];
        square_mean += src[index] * src[index];
      }

      mean /= (float)param->inner_size_;
      square_mean /= (float)param->inner_size_;
      const double deno = gamma_data[c] / sqrt(square_mean - mean * mean + param->epsilon_);

      index = 0;
#if defined(ENABLE_AVX)
      MS_FLOAT32X8 meanv8 = MS_MOV256_F32(mean);
      MS_FLOAT32X8 denov8 = MS_MOV256_F32(deno);
      for (; index <= param->inner_size_ - C8NUM; index += C8NUM) {
        MS_FLOAT32X8 srcv8 = MS_LD256_F32(src + index);
        MS_FLOAT32X8 dstv8 =
          MS_ADD256_F32(MS_MUL256_F32(MS_SUB256_F32(srcv8, meanv8), denov8), MS_MOV256_F32(*(beta_data + c)));
        MS_ST256_F32(dst + index, dstv8);
      }
#endif

#if defined(ENABLE_NEON) || defined(ENABLE_SSE)
      MS_FLOAT32X4 meanv4 = MS_MOVQ_F32(mean);
      MS_FLOAT32X4 denov4 = MS_MOVQ_F32(deno);
      for (; index <= param->inner_size_ - C4NUM; index += C4NUM) {
        MS_FLOAT32X4 srcv4 = MS_LDQ_F32(src + index);
        MS_FLOAT32X4 dstv4 =
          MS_ADDQ_F32(MS_MULQ_F32(MS_SUBQ_F32(srcv4, meanv4), denov4), MS_MOVQ_F32(*(beta_data + c)));
        MS_STQ_F32(dst + index, dstv4);
      }
#endif
      for (; index < param->inner_size_; index++) {
        dst[index] = (src[index] - mean) * deno + beta_data[c];
      }
    }
  }
  return NNACL_OK;
}

int InstanceNormNC4HW4(const float *src_data, float *dst_data, const float *gamma_data, const float *beta_data,
                       const InstanceNormParameter *param, size_t task_id) {
  NNACL_CHECK_NULL_RETURN_ERR(src_data);
  NNACL_CHECK_NULL_RETURN_ERR(dst_data);
  NNACL_CHECK_NULL_RETURN_ERR(param->op_parameter_.thread_num_);
  int channel = param->channel_;
  int hw_plane = param->inner_size_;
  int channel_step = UP_DIV(UP_DIV(channel, C4NUM), param->op_parameter_.thread_num_) * C4NUM;
  int channel_begin = (int)(task_id)*channel_step;
  int channel_end = MSMIN(channel_begin + channel_step, channel);
#if defined(ENABLE_SSE) || defined(ENABLE_ARM)
  int c4_down = channel_end / C4NUM * C4NUM;
  MS_FLOAT32X4 hw_planev = MS_MOVQ_F32((float)(hw_plane));
#endif
  for (int b = 0; b < param->batch_; b++) {
    const float *src_b = src_data + b * channel * hw_plane;
    float *dst_b = dst_data + b * channel * hw_plane;
    int c = channel_begin;
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
    for (; c < c4_down; c += C4NUM) {
      const float *src = src_b + c * hw_plane;
      float *dst = dst_b + c;
      MS_FLOAT32X4 mean = MS_MOVQ_F32(0.0f);
      MS_FLOAT32X4 square_mean = MS_MOVQ_F32(0.0f);
      for (int index = 0; index < hw_plane; ++index) {
        MS_FLOAT32X4 srcv = MS_LDQ_F32(src + index * C4NUM);
        MS_FLOAT32X4 squarev = MS_MULQ_F32(srcv, srcv);
        mean = MS_ADDQ_F32(mean, srcv);
        square_mean = MS_ADDQ_F32(square_mean, squarev);
      }
      mean = MS_DIVQ_F32(mean, hw_planev);
      square_mean = MS_DIVQ_F32(square_mean, hw_planev);
      MS_FLOAT32X4 deno =
        MS_ADDQ_F32(MS_SUBQ_F32(square_mean, MS_MULQ_F32(mean, mean)), MS_MOVQ_F32(param->epsilon_));  // question
      deno = MS_DIVQ_F32(MS_MOVQ_F32(1.0f), MS_SQRTFX4_F32(deno));

      MS_FLOAT32X4 gammav = MS_MULQ_F32(MS_LDQ_F32(gamma_data + c), deno);  // deno * gamma_data[c]
      MS_FLOAT32X4 betav = MS_LDQ_F32(beta_data + c);
      for (int index = 0; index < hw_plane; ++index) {
        MS_FLOAT32X4 srcv = MS_LDQ_F32(src + index * C4NUM);
        MS_FLOAT32X4 outv = MS_SUBQ_F32(srcv, mean);
        outv = MS_MULQ_F32(outv, gammav);
        outv = MS_ADDQ_F32(outv, betav);
        MS_STQ_F32(dst + index * channel, outv);
      }
    }
#endif
    for (; c < channel_end; ++c) {
      int c4_down_loop = c / C4NUM * C4NUM;
      int c4_mod = c % C4NUM;
      int c_res = MSMIN(channel_end - c4_down_loop, C4NUM);
      const float *src = src_b + c4_down_loop * hw_plane + c4_mod;
      float *dst = dst_b + c;
      float mean = 0.0f;
      float square_mean = 0.0f;
      for (int index = 0; index < hw_plane; ++index) {
        float tmp = src[index * c_res];
        mean += tmp;
        square_mean += tmp * tmp;
      }
      mean /= (float)hw_plane;
      square_mean /= (float)hw_plane;
      const float deno = gamma_data[c] / sqrtf(square_mean - mean * mean + param->epsilon_);
      for (int index = 0; index < hw_plane; ++index) {
        dst[index * channel] = (src[index * c_res] - mean) * deno + beta_data[c];
      }
    }
  }
  return NNACL_OK;
}
