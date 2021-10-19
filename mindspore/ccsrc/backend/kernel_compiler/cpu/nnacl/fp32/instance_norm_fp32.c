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
  NNACL_CHECK_ZERO_RETURN_ERR(param->op_parameter_.thread_num_);
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
      double squ_m = 0.0f;

      int index = 0;
#if defined(ENABLE_AVX)
      for (; index <= param->inner_size_ - C8NUM; index += C8NUM) {
        __m256 srcv = _mm256_loadu_ps(src + index);
        __m256 squarev = _mm256_mul_ps(srcv, srcv);
        __m128 src128 = _mm_add_ps(_mm256_extractf128_ps(srcv, 0), _mm256_extractf128_ps(srcv, 1));
        __m128 square128 = _mm_add_ps(_mm256_extractf128_ps(squarev, 0), _mm256_extractf128_ps(squarev, 1));
        for (int i = 0; i < C4NUM; ++i) {
          mean += MS_F32X4_GETI(src128, i);
          squ_m += MS_F32X4_GETI(square128, i);
        }
      }
#endif

#if defined(ENABLE_NEON) || defined(ENABLE_SSE)
      for (; index <= param->inner_size_ - C4NUM; index += C4NUM) {
        MS_FLOAT32X4 srcv = MS_LDQ_F32(src + index);
        MS_FLOAT32X4 squarev = MS_MULQ_F32(srcv, srcv);
#ifdef ENABLE_ARM64
        mean += vaddvq_f32(srcv);
        squ_m += vaddvq_f32(squarev);
#elif defined(ENABLE_SSE)
        for (int i = 0; i < C4NUM; ++i) {
          mean += MS_F32X4_GETI(srcv, i);
          squ_m += MS_F32X4_GETI(squarev, i);
        }
#else
        float32x2_t src_add2 = vadd_f32(vget_low_f32(srcv), vget_high_f32(srcv));
        float32x2_t src_add4 = vpadd_f32(src_add2, src_add2);
        mean += vget_lane_f32(src_add4, 0);
        float32x2_t square_add2 = vadd_f32(vget_low_f32(squarev), vget_high_f32(squarev));
        float32x2_t square_add4 = vpadd_f32(square_add2, square_add2);
        squ_m += vget_lane_f32(square_add4, 0);
#endif
      }
#endif
      for (; index < param->inner_size_; index++) {
        mean += src[index];
        squ_m += src[index] * src[index];
      }

      mean /= (float)param->inner_size_;
      squ_m /= (float)param->inner_size_;
      const double deno = gamma_data[c] / sqrt(squ_m - mean * mean + param->epsilon_);

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

#if defined(ENABLE_SSE) || defined(ENABLE_ARM)
void InstanceNormC4HW4ArmSse(const float *src_b, float *dst_b, const float *gamma_data, const float *beta_data,
                             int *c_src, const InstanceNormParameter *param, int channel, int channel_end, int hw_plane,
                             MS_FLOAT32X4 hw_planev) {
  int c = *c_src;
  for (; c <= channel_end - C16NUM; c += C16NUM) {
    const float *src = src_b + c * hw_plane, *src1 = src_b + (c + C4NUM) * hw_plane;
    const float *src2 = src_b + (c + C8NUM) * hw_plane, *src3 = src_b + (c + C12NUM) * hw_plane;
    float *dst = dst_b + c;
    MS_FLOAT32X4 mean = MS_MOVQ_F32(0.0f), mean1 = MS_MOVQ_F32(0.0f);
    MS_FLOAT32X4 mean2 = MS_MOVQ_F32(0.0f), mean3 = MS_MOVQ_F32(0.0f);
    MS_FLOAT32X4 squ_m = MS_MOVQ_F32(0.0f), squ_m1 = MS_MOVQ_F32(0.0f);
    MS_FLOAT32X4 squ_m2 = MS_MOVQ_F32(0.0f), squ_m3 = MS_MOVQ_F32(0.0f);
    for (int index = 0; index < hw_plane; ++index) {
      MS_FLOAT32X4 srcv = MS_LDQ_F32(src + index * C4NUM), srcv1 = MS_LDQ_F32(src1 + index * C4NUM);
      MS_FLOAT32X4 srcv2 = MS_LDQ_F32(src2 + index * C4NUM), srcv3 = MS_LDQ_F32(src3 + index * C4NUM);
      MS_FLOAT32X4 squarev = MS_MULQ_F32(srcv, srcv), squarev1 = MS_MULQ_F32(srcv1, srcv1);
      MS_FLOAT32X4 squarev2 = MS_MULQ_F32(srcv2, srcv2), squarev3 = MS_MULQ_F32(srcv3, srcv3);
      MS_ADDQ_F32_VEC(mean, mean1, mean2, mean3, srcv, srcv1, srcv2, srcv3);
      MS_ADDQ_F32_VEC(squ_m, squ_m1, squ_m2, squ_m3, squarev, squarev1, squarev2, squarev3);
    }
    MS_DIVQ_F32_VEC(mean, mean1, mean2, mean3, hw_planev);
    MS_DIVQ_F32_VEC(squ_m, squ_m1, squ_m2, squ_m3, hw_planev);

    MS_FLOAT32X4 deno = MS_ADDQ_F32(MS_SUBQ_F32(squ_m, MS_MULQ_F32(mean, mean)), MS_MOVQ_F32(param->epsilon_));
    MS_FLOAT32X4 deno1 = MS_ADDQ_F32(MS_SUBQ_F32(squ_m1, MS_MULQ_F32(mean1, mean1)), MS_MOVQ_F32(param->epsilon_));
    MS_FLOAT32X4 deno2 = MS_ADDQ_F32(MS_SUBQ_F32(squ_m2, MS_MULQ_F32(mean2, mean2)), MS_MOVQ_F32(param->epsilon_));
    MS_FLOAT32X4 deno3 = MS_ADDQ_F32(MS_SUBQ_F32(squ_m3, MS_MULQ_F32(mean3, mean3)), MS_MOVQ_F32(param->epsilon_));

    deno = MS_DIVQ_F32(MS_MOVQ_F32(1.0f), MS_SQRTFX4_F32(deno));
    deno1 = MS_DIVQ_F32(MS_MOVQ_F32(1.0f), MS_SQRTFX4_F32(deno1));
    deno2 = MS_DIVQ_F32(MS_MOVQ_F32(1.0f), MS_SQRTFX4_F32(deno2));
    deno3 = MS_DIVQ_F32(MS_MOVQ_F32(1.0f), MS_SQRTFX4_F32(deno3));

    MS_FLOAT32X4 gammav = MS_MULQ_F32(MS_LDQ_F32(gamma_data + c), deno);             // deno * gamma_data[c]
    MS_FLOAT32X4 gammav1 = MS_MULQ_F32(MS_LDQ_F32(gamma_data + c + C4NUM), deno1);   // deno * gamma_data[c]
    MS_FLOAT32X4 gammav2 = MS_MULQ_F32(MS_LDQ_F32(gamma_data + c + C8NUM), deno2);   // deno * gamma_data[c]
    MS_FLOAT32X4 gammav3 = MS_MULQ_F32(MS_LDQ_F32(gamma_data + c + C12NUM), deno3);  // deno * gamma_data[c]
    MS_FLOAT32X4 betav = MS_LDQ_F32(beta_data + c), betav1 = MS_LDQ_F32(beta_data + c + C4NUM);
    MS_FLOAT32X4 betav2 = MS_LDQ_F32(beta_data + c + C8NUM), betav3 = MS_LDQ_F32(beta_data + c + C12NUM);
    for (int index = 0; index < hw_plane; ++index) {
      MS_FLOAT32X4 srcv = MS_LDQ_F32(src + index * C4NUM), srcv1 = MS_LDQ_F32(src1 + index * C4NUM);
      MS_FLOAT32X4 srcv2 = MS_LDQ_F32(src2 + index * C4NUM), srcv3 = MS_LDQ_F32(src3 + index * C4NUM);
      MS_FLOAT32X4 outv = MS_SUBQ_F32(srcv, mean), outv1 = MS_SUBQ_F32(srcv1, mean1);
      MS_FLOAT32X4 outv2 = MS_SUBQ_F32(srcv2, mean2), outv3 = MS_SUBQ_F32(srcv3, mean3);

      outv = MS_MULQ_F32(outv, gammav), outv1 = MS_MULQ_F32(outv1, gammav1);
      outv2 = MS_MULQ_F32(outv2, gammav2), outv3 = MS_MULQ_F32(outv3, gammav3);
      MS_ADDQ_F32_VEC(outv, outv1, outv2, outv3, betav, betav1, betav2, betav3);

      MS_STQ_F32(dst + index * channel, outv), MS_STQ_F32(dst + index * channel + C4NUM, outv1);
      MS_STQ_F32(dst + index * channel + C8NUM, outv2), MS_STQ_F32(dst + index * channel + C12NUM, outv3);
    }
  }
  for (; c <= channel_end - C8NUM; c += C8NUM) {
    const float *src = src_b + c * hw_plane, *src1 = src_b + (c + C4NUM) * hw_plane;
    float *dst = dst_b + c;
    MS_FLOAT32X4 mean = MS_MOVQ_F32(0.0f), mean1 = MS_MOVQ_F32(0.0f);
    MS_FLOAT32X4 squ_m = MS_MOVQ_F32(0.0f), squ_m1 = MS_MOVQ_F32(0.0f);
    for (int index = 0; index < hw_plane; ++index) {
      MS_FLOAT32X4 srcv = MS_LDQ_F32(src + index * C4NUM), srcv1 = MS_LDQ_F32(src1 + index * C4NUM);
      MS_FLOAT32X4 squarev = MS_MULQ_F32(srcv, srcv), squarev1 = MS_MULQ_F32(srcv1, srcv1);
      mean = MS_ADDQ_F32(mean, srcv), mean1 = MS_ADDQ_F32(mean1, srcv1);
      squ_m = MS_ADDQ_F32(squ_m, squarev), squ_m1 = MS_ADDQ_F32(squ_m1, squarev1);
    }

    MS_DIVQ_F32_VEC(mean, mean1, squ_m, squ_m1, hw_planev);
    MS_FLOAT32X4 deno = MS_ADDQ_F32(MS_SUBQ_F32(squ_m, MS_MULQ_F32(mean, mean)), MS_MOVQ_F32(param->epsilon_));
    MS_FLOAT32X4 deno1 = MS_ADDQ_F32(MS_SUBQ_F32(squ_m1, MS_MULQ_F32(mean1, mean1)), MS_MOVQ_F32(param->epsilon_));
    deno = MS_DIVQ_F32(MS_MOVQ_F32(1.0f), MS_SQRTFX4_F32(deno));
    deno1 = MS_DIVQ_F32(MS_MOVQ_F32(1.0f), MS_SQRTFX4_F32(deno1));

    MS_FLOAT32X4 gammav = MS_MULQ_F32(MS_LDQ_F32(gamma_data + c), deno);            // deno * gamma_data[c]
    MS_FLOAT32X4 gammav1 = MS_MULQ_F32(MS_LDQ_F32(gamma_data + c + C4NUM), deno1);  // deno * gamma_data[c]
    MS_FLOAT32X4 betav = MS_LDQ_F32(beta_data + c), betav1 = MS_LDQ_F32(beta_data + c + C4NUM);
    for (int index = 0; index < hw_plane; ++index) {
      MS_FLOAT32X4 srcv = MS_LDQ_F32(src + index * C4NUM), srcv1 = MS_LDQ_F32(src1 + index * C4NUM);
      MS_FLOAT32X4 outv = MS_SUBQ_F32(srcv, mean), outv1 = MS_SUBQ_F32(srcv1, mean1);
      outv = MS_MULQ_F32(outv, gammav), outv1 = MS_MULQ_F32(outv1, gammav1);
      outv = MS_ADDQ_F32(outv, betav), outv1 = MS_ADDQ_F32(outv1, betav1);
      MS_STQ_F32(dst + index * channel, outv);
      MS_STQ_F32(dst + index * channel + C4NUM, outv1);
    }
  }
  for (; c <= channel_end - C4NUM; c += C4NUM) {
    const float *src = src_b + c * hw_plane;
    float *dst = dst_b + c;
    MS_FLOAT32X4 mean = MS_MOVQ_F32(0.0f), squ_m = MS_MOVQ_F32(0.0f);
    for (int index = 0; index < hw_plane; ++index) {
      MS_FLOAT32X4 srcv = MS_LDQ_F32(src + index * C4NUM), squarev = MS_MULQ_F32(srcv, srcv);
      mean = MS_ADDQ_F32(mean, srcv), squ_m = MS_ADDQ_F32(squ_m, squarev);
    }
    mean = MS_DIVQ_F32(mean, hw_planev), squ_m = MS_DIVQ_F32(squ_m, hw_planev);
    MS_FLOAT32X4 deno = MS_ADDQ_F32(MS_SUBQ_F32(squ_m, MS_MULQ_F32(mean, mean)), MS_MOVQ_F32(param->epsilon_));
    deno = MS_DIVQ_F32(MS_MOVQ_F32(1.0f), MS_SQRTFX4_F32(deno));

    MS_FLOAT32X4 gammav = MS_MULQ_F32(MS_LDQ_F32(gamma_data + c), deno), betav = MS_LDQ_F32(beta_data + c);
    for (int index = 0; index < hw_plane; ++index) {
      MS_FLOAT32X4 srcv = MS_LDQ_F32(src + index * C4NUM), outv = MS_SUBQ_F32(srcv, mean);
      MS_STQ_F32(dst + index * channel, MS_ADDQ_F32(MS_MULQ_F32(outv, gammav), betav));
    }
  }
  *c_src = c;
}
#endif

int InstanceNormNC4HW4(const float *src_data, float *dst_data, const float *gamma_data, const float *beta_data,
                       const InstanceNormParameter *param, size_t task_id) {
  NNACL_CHECK_NULL_RETURN_ERR(src_data);
  NNACL_CHECK_NULL_RETURN_ERR(dst_data);
  NNACL_CHECK_ZERO_RETURN_ERR(param->op_parameter_.thread_num_);
  int channel = param->channel_;
  int hw_plane = param->inner_size_;
  int channel_step = UP_DIV(UP_DIV(channel, C4NUM), param->op_parameter_.thread_num_) * C4NUM;
  int channel_begin = (int)(task_id)*channel_step;
  int channel_end = MSMIN(channel_begin + channel_step, channel);
#if defined(ENABLE_SSE) || defined(ENABLE_ARM)
  MS_FLOAT32X4 hw_planev = MS_MOVQ_F32((float)(hw_plane));
#endif
  for (int b = 0; b < param->batch_; b++) {
    const float *src_b = src_data + b * channel * hw_plane;
    float *dst_b = dst_data + b * channel * hw_plane;
    int c = channel_begin;
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
    InstanceNormC4HW4ArmSse(src_b, dst_b, gamma_data, beta_data, &c, param, channel, channel_end, hw_plane, hw_planev);
#endif
    for (; c < channel_end; ++c) {
      int c4_down_loop = c / C4NUM * C4NUM;
      int c4_mod = c % C4NUM;
      int c_res = MSMIN(channel_end - c4_down_loop, C4NUM);
      const float *src = src_b + c4_down_loop * hw_plane + c4_mod;
      float *dst = dst_b + c;
      float mean = 0.0f;
      float squ_m = 0.0f;
      for (int index = 0; index < hw_plane; ++index) {
        float tmp = src[index * c_res];
        mean += tmp;
        squ_m += tmp * tmp;
      }
      mean /= (float)hw_plane;
      squ_m /= (float)hw_plane;
      const float deno = gamma_data[c] / sqrtf(squ_m - mean * mean + param->epsilon_);
      for (int index = 0; index < hw_plane; ++index) {
        dst[index * channel] = (src[index * c_res] - mean) * deno + beta_data[c];
      }
    }
  }
  return NNACL_OK;
}

#ifdef ENABLE_AVX
int InstanceNormNC8HW8(const float *src_data, float *dst_data, const float *gamma_data, const float *beta_data,
                       const InstanceNormParameter *param, size_t task_id) {
  NNACL_CHECK_NULL_RETURN_ERR(src_data);
  NNACL_CHECK_NULL_RETURN_ERR(dst_data);
  NNACL_CHECK_ZERO_RETURN_ERR(param->op_parameter_.thread_num_);
  int channel = param->channel_, hw_plane = param->inner_size_;
  int channel_step = UP_DIV(UP_DIV(channel, C8NUM), param->op_parameter_.thread_num_) * C8NUM;
  int channel_begin = (int)(task_id)*channel_step;
  int channel_end = MSMIN(channel_begin + channel_step, channel);
  MS_FLOAT32X8 hw_planev = MS_MOV256_F32((float)(hw_plane));
  for (int b = 0; b < param->batch_; b++) {
    const float *src_b = src_data + b * channel * hw_plane;
    float *dst_b = dst_data + b * channel * hw_plane;
    int c = channel_begin;
    for (; c <= channel_end - C16NUM; c += C16NUM) {
      const float *src = src_b + c * hw_plane;
      const float *src1 = src_b + (c + C8NUM) * hw_plane;
      float *dst = dst_b + c;
      MS_FLOAT32X8 mean = MS_MOV256_F32(0.0f), mean1 = MS_MOV256_F32(0.0f);
      MS_FLOAT32X8 squ_m = MS_MOV256_F32(0.0f), squ_m1 = MS_MOV256_F32(0.0f);
      for (int index = 0; index < hw_plane; ++index) {
        MS_FLOAT32X8 srcv = MS_LD256_F32(src + index * C8NUM), srcv1 = MS_LD256_F32(src1 + index * C8NUM);
        MS_FLOAT32X8 squarev = MS_MUL256_F32(srcv, srcv), squarev1 = MS_MUL256_F32(srcv1, srcv1);
        mean = MS_ADD256_F32(mean, srcv);
        mean1 = MS_ADD256_F32(mean1, srcv1);
        squ_m = MS_ADD256_F32(squ_m, squarev);
        squ_m1 = MS_ADD256_F32(squ_m1, squarev1);
      }
      mean = MS_DIV256_F32(mean, hw_planev);
      mean1 = MS_DIV256_F32(mean1, hw_planev);
      squ_m = MS_DIV256_F32(squ_m, hw_planev);
      squ_m1 = MS_DIV256_F32(squ_m1, hw_planev);
      MS_FLOAT32X8 deno =
        MS_ADD256_F32(MS_SUB256_F32(squ_m, MS_MUL256_F32(mean, mean)), MS_MOV256_F32(param->epsilon_));
      MS_FLOAT32X8 deno1 =
        MS_ADD256_F32(MS_SUB256_F32(squ_m1, MS_MUL256_F32(mean1, mean1)), MS_MOV256_F32(param->epsilon_));
      deno = MS_DIV256_F32(MS_MOV256_F32(1.0f), MS_SQRTFX8_F32(deno));
      deno1 = MS_DIV256_F32(MS_MOV256_F32(1.0f), MS_SQRTFX8_F32(deno1));

      MS_FLOAT32X8 gammav = MS_MUL256_F32(MS_LD256_F32(gamma_data + c), deno);            // deno * gamma_data[c]
      MS_FLOAT32X8 gammav1 = MS_MUL256_F32(MS_LD256_F32(gamma_data + c + C8NUM), deno1);  // deno1 * gamma_data[c]
      MS_FLOAT32X8 betav = MS_LD256_F32(beta_data + c), betav1 = MS_LD256_F32(beta_data + c + C8NUM);
      for (int index = 0; index < hw_plane; ++index) {
        MS_FLOAT32X8 srcv = MS_LD256_F32(src + index * C8NUM), srcv1 = MS_LD256_F32(src1 + index * C8NUM);
        MS_FLOAT32X8 outv = MS_SUB256_F32(srcv, mean), outv1 = MS_SUB256_F32(srcv1, mean1);
        outv = MS_MUL256_F32(outv, gammav);
        outv1 = MS_MUL256_F32(outv1, gammav1);
        outv = MS_ADD256_F32(outv, betav);
        outv1 = MS_ADD256_F32(outv1, betav1);
        MS_ST256_F32(dst + index * channel, outv);
        MS_ST256_F32(dst + index * channel + C8NUM, outv1);
      }
    }
    for (; c <= channel_end - C8NUM; c += C8NUM) {
      const float *src = src_b + c * hw_plane;
      float *dst = dst_b + c;
      MS_FLOAT32X8 mean = MS_MOV256_F32(0.0f), squ_m = MS_MOV256_F32(0.0f);
      for (int index = 0; index < hw_plane; ++index) {
        MS_FLOAT32X8 srcv = MS_LD256_F32(src + index * C8NUM);
        MS_FLOAT32X8 squarev = MS_MUL256_F32(srcv, srcv);
        mean = MS_ADD256_F32(mean, srcv);
        squ_m = MS_ADD256_F32(squ_m, squarev);
      }
      mean = MS_DIV256_F32(mean, hw_planev);
      squ_m = MS_DIV256_F32(squ_m, hw_planev);
      MS_FLOAT32X8 deno = MS_ADD256_F32(MS_SUB256_F32(squ_m, MS_MUL256_F32(mean, mean)),
                                        MS_MOV256_F32(param->epsilon_));  // 256uestion
      deno = MS_DIV256_F32(MS_MOV256_F32(1.0f), MS_SQRTFX8_F32(deno));

      MS_FLOAT32X8 gammav = MS_MUL256_F32(MS_LD256_F32(gamma_data + c), deno);  // deno * gamma_data[c]
      MS_FLOAT32X8 betav = MS_LD256_F32(beta_data + c);
      for (int index = 0; index < hw_plane; ++index) {
        MS_FLOAT32X8 srcv = MS_LD256_F32(src + index * C8NUM), outv = MS_SUB256_F32(srcv, mean);
        outv = MS_MUL256_F32(outv, gammav);
        outv = MS_ADD256_F32(outv, betav);
        MS_ST256_F32(dst + index * channel, outv);
      }
    }
    for (; c < channel_end; ++c) {
      int c8_down_loop = c / C8NUM * C8NUM, c8_mod = c % C8NUM;
      int c_res = MSMIN(channel_end - c8_down_loop, C8NUM);
      const float *src = src_b + c8_down_loop * hw_plane + c8_mod;
      float *dst = dst_b + c;
      float mean = 0.0f, squ_m = 0.0f;
      for (int index = 0; index < hw_plane; ++index) {
        float tmp = src[index * c_res];
        mean += tmp;
        squ_m += tmp * tmp;
      }
      mean /= (float)hw_plane;
      squ_m /= (float)hw_plane;
      const float deno = gamma_data[c] / sqrtf(squ_m - mean * mean + param->epsilon_);
      for (int index = 0; index < hw_plane; ++index) {
        dst[index * channel] = (src[index * c_res] - mean) * deno + beta_data[c];
      }
    }
  }
  return NNACL_OK;
}
#endif
