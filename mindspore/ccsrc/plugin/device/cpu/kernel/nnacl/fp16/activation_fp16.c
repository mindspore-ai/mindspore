/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "nnacl/fp16/activation_fp16.h"
#include <float.h>
#include "nnacl/fp32/exp_fp32.h"
#include "nnacl/fp16/exp_fp16.h"
#include "nnacl/errorcode.h"

int ReluFp16(const float16_t *src, float16_t *dst, int ele_num) {
  int offset = 0;
#ifdef ENABLE_NEON
  float16x8_t zero = vdupq_n_f16(0);
  for (; offset <= ele_num - C8NUM; offset += C8NUM) {
    float16x8_t src_value = vld1q_f16(src + offset);
    float16x8_t rst_value = vmaxq_f16(src_value, zero);
    vst1q_f16(dst + offset, rst_value);
  }
#endif
  for (; offset < ele_num; offset++) {
    dst[offset] = src[offset] < 0.0f ? 0.0f : src[offset];
  }
  return NNACL_OK;
}

int Relu6Fp16(const float16_t *data, float16_t *dst, int ele_num) {
  int offset = 0;
#ifdef ENABLE_NEON
  float16x8_t zero_data = vdupq_n_f16(0);
  float16x8_t six_data = vdupq_n_f16(6);
  for (; offset <= ele_num - C8NUM; offset += C8NUM) {
    float16x8_t relu6_data = vld1q_f16(data + offset);
    relu6_data = vmaxq_f16(relu6_data, zero_data);
    relu6_data = vminq_f16(relu6_data, six_data);
    vst1q_f16(dst + offset, relu6_data);
  }
#endif
  for (; offset < ele_num; offset++) {
    dst[offset] = data[offset] < 0.0f ? 0.0f : data[offset];
    dst[offset] = dst[offset] > 6.0f ? 6.0f : dst[offset];
  }
  return NNACL_OK;
}

int LReluFp16(const float16_t *src, float16_t *dst, int ele_num, float16_t alpha) {
  int offset = 0;
#ifdef ENABLE_NEON
  float16x8_t zero_data = vdupq_n_f16(0);
  float16x8_t alpha_data = vdupq_n_f16(alpha);
  for (; offset <= ele_num - C8NUM; offset += C8NUM) {
    float16x8_t src_tmp = vld1q_f16(src + offset);
    float16x8_t mul_tmp = vmulq_f16(src_tmp, alpha_data);
    uint16x8_t mask = vcleq_f16(src_tmp, zero_data);
    vst1q_f16(dst + offset, vbslq_f16(mask, mul_tmp, src_tmp));
  }
#endif
  for (; offset < ele_num; ++offset) {
    dst[offset] = src[offset] > (float16_t)0.0f ? src[offset] : (src[offset] * alpha);
  }
  return NNACL_OK;
}

int SigmoidFp16(const float16_t *src, float16_t *dst, int ele_num) {
  int i = 0;
#ifdef ENABLE_NEON
  int count = (ele_num / C4NUM) * C4NUM;
  for (; i < count; i += C4NUM) {
    float32x4_t tmp;
    simd_exp128(vnegq_f32(vcvt_f32_f16(vld1_f16(src + i))), (float *)&tmp);
    vst1_f16(dst + i, vcvt_f16_f32(MS_DIVQ_F32(vdupq_n_f32(1.0f), vaddq_f32(vdupq_n_f32(1.0f), tmp))));
  }
#endif
  for (; i < ele_num; ++i) {
    float temp;
    simd_exp32(-src[i], &temp);
    dst[i] = (float16_t)1.0f / ((float16_t)1.0f + temp);
  }
  return NNACL_OK;
}

float16_t TanhOptFp16(float16_t src) {
  if (src > 5.0f) {
    return 1.0f;
  } else if (src < -5.0f) {
    return -1.0f;
  } else {
    float square = src * src;
    float a = (((square + 378.0f) * square + 17325.0f) * square + 135135.0f) * src;
    float b = ((28.0f * square + 3150.0f) * square + 62370.0f) * square + 135135.0f;
    return a / b;
  }
}

int TanhFp16(const float16_t *src, float16_t *dst, int ele_num) {
  int i = 0;
#ifdef ENABLE_NEON
  static float32x4_t paramv[] = {{378.0f, 378.0f, 378.0f, 378.0f},
                                 {17325.0f, 17325.0f, 17325.0f, 17325.0f},
                                 {135135.0f, 135135.0f, 135135.0f, 135135.0f},
                                 {28.0f, 28.0f, 28.0f, 28.0f},
                                 {3150.0f, 3150.0f, 3150.0f, 3150.0f},
                                 {62370.0f, 62370.0f, 62370.0f, 62370.0f}};
  float32x4_t neg_one = {-1.0f, -1.0f, -1.0f, -1.0f};
  float32x4_t pos_one = {1.0f, 1.0f, 1.0f, 1.0f};
  int count = (ele_num / C4NUM) * C4NUM;
  for (; i < count; i += C4NUM) {
    float32x4_t input = vcvt_f32_f16(vld1_f16(src + i));
    float32x4_t square = vmulq_f32(input, input);
    float32x4_t a = vmulq_f32(
      vaddq_f32(vmulq_f32(vaddq_f32(vmulq_f32(vaddq_f32(square, paramv[0]), square), paramv[1]), square), paramv[2]),
      input);
    float32x4_t b = vaddq_f32(
      vmulq_f32(vaddq_f32(vmulq_f32(vaddq_f32(vmulq_f32(paramv[3], square), paramv[4]), square), paramv[5]), square),
      paramv[2]);
    vst1_f16(dst + i, vcvt_f16_f32(vminq_f32(vmaxq_f32(MS_DIVQ_F32(a, b), neg_one), pos_one)));
  }
#endif
  for (; i < ele_num; ++i) {
    float input = src[i];
    float square = input * input;
    float a = (((square + 378.0f) * square + 17325.0f) * square + 135135.0f) * input;
    float b = ((28.0f * square + 3150.0f) * square + 62370.0f) * square + 135135.0f;
    dst[i] = a / b;
    dst[i] = MSMAX(dst[i], -1);
    dst[i] = MSMIN(dst[i], 1);
  }
  return NNACL_OK;
}

int HSwishFp16(const float16_t *src, float16_t *dst, int ele_num) {
  int i = 0;
#ifdef ENABLE_NEON
  const MS_FLOAT16X8 zero_data = vdupq_n_f16(0);
  const MS_FLOAT16X8 three_data = vdupq_n_f16(3);
  const MS_FLOAT16X8 six_data = vdupq_n_f16(6);
  for (; i <= ele_num - C8NUM; i += C8NUM) {
    MS_FLOAT16X8 in_data = MS_LDQ_F16(src + i);
    MS_FLOAT16X8 tmp = MS_MAXQ_F16(in_data + three_data, zero_data);
    tmp = MS_MINQ_F16(tmp, six_data);
    MS_STQ_F16(dst + i, vmulq_f16(in_data, MS_DIVQ_F16(tmp, six_data)));
  }
#endif
  for (; i < ele_num; ++i) {
    float16_t in = src[i];
    float16_t relu6 = MSMIN(MSMAX(in + 3.0f, 0.0f), 6.0f);
    dst[i] = in * relu6 / (float16_t)6.0f;
  }
  return NNACL_OK;
}

int SwishFp16(const float16_t *src, float16_t *dst, int ele_num) {
  int i = 0;
#ifdef ENABLE_NEON
  float32x4_t const_val = vdupq_n_f32(1.0f);
  for (int num_max = ele_num - C16NUM; i <= num_max; i += C16NUM) {
    float16x4x4_t ins = vld4_f16(src + i);
    float32x4_t in0 = MS_CVT_F32_F16(ins.val[0]);
    float32x4_t in1 = MS_CVT_F32_F16(ins.val[1]);
    float32x4_t in2 = MS_CVT_F32_F16(ins.val[2]);
    float32x4_t in3 = MS_CVT_F32_F16(ins.val[3]);
    float32x4_t exp0 = simd_exp128_f32(vnegq_f32(in0));
    float32x4_t exp1 = simd_exp128_f32(vnegq_f32(in1));
    float32x4_t exp2 = simd_exp128_f32(vnegq_f32(in2));
    float32x4_t exp3 = simd_exp128_f32(vnegq_f32(in3));
    float32x4_t res0 = MS_DIVQ_F32(in0, vaddq_f32(const_val, exp0));
    float32x4_t res1 = MS_DIVQ_F32(in1, vaddq_f32(const_val, exp1));
    float32x4_t res2 = MS_DIVQ_F32(in2, vaddq_f32(const_val, exp2));
    float32x4_t res3 = MS_DIVQ_F32(in3, vaddq_f32(const_val, exp3));
    float16x4x4_t res = {MS_CVT_F16_F32(res0), MS_CVT_F16_F32(res1), MS_CVT_F16_F32(res2), MS_CVT_F16_F32(res3)};
    vst4_f16(dst + i, res);
  }
  for (int num_max = ele_num - C4NUM; i <= num_max; i += C4NUM) {
    float32x4_t in = MS_CVT_F32_F16(vld1_f16(src + i));
    float16x4_t res = MS_CVT_F16_F32(MS_DIVQ_F32(in, vaddq_f32(const_val, simd_exp128_f32(vnegq_f32(in)))));
    vst1_f16(dst + i, res);
  }
#endif
  for (; i < ele_num; ++i) {
    float temp = simd_exp32_f32(-src[i]);
    dst[i] = src[i] / (1.0f + temp);
  }
  return NNACL_OK;
}

int HSigmoidFp16(const float16_t *src, float16_t *dst, int ele_num) {
  int offset = 0;
#ifdef ENABLE_NEON
  const MS_FLOAT16X8 zero_data = vdupq_n_f16(0);
  const MS_FLOAT16X8 three_data = vdupq_n_f16(3);
  const MS_FLOAT16X8 six_data = vdupq_n_f16(6);
  for (; offset <= ele_num - C8NUM; offset += C8NUM) {
    MS_FLOAT16X8 relu6_data = MS_LDQ_F16(src + offset) + three_data;
    relu6_data = MS_MAXQ_F16(relu6_data, zero_data);
    relu6_data = MS_MINQ_F16(relu6_data, six_data);
    MS_STQ_F16(dst + offset, MS_DIVQ_F16(relu6_data, six_data));
  }
#endif

  for (; offset < ele_num; offset++) {
    float16_t tmp = (src[offset] + 3.0 < 0.0) ? 0.0 : src[offset] + 3.0;
    dst[offset] = ((tmp < 6.0) ? tmp : 6.0) / 6.0;
  }

  return NNACL_OK;
}

int HardTanhFp16(const float16_t *src, int length, float16_t *dst, float min_val, float max_val) {
  if (max_val <= min_val) {
    return NNACL_ERR;
  }
  int i = 0;
  if (min_val == FLT_MIN) {
    for (i = 0; i < length; ++i) {
      dst[i] = src[i] > max_val ? max_val : src[i];
    }
  } else if (max_val == FLT_MAX) {
    for (i = 0; i < length; ++i) {
      dst[i] = src[i] < min_val ? min_val : src[i];
    }
  } else {
    for (i = 0; i < length; ++i) {
      dst[i] = src[i] < min_val ? min_val : (src[i] > max_val ? max_val : src[i]);
    }
  }
  return NNACL_OK;
}

int GeluFp16(const float16_t *src, int length, float16_t *dst, bool approximate) {
  if (src == NULL || dst == NULL) {
    return NNACL_ERR;
  }
  int i = 0;
  if (approximate) {
    // dst = 0.5 * x * (1 + tanh((2 / pi) ^ 0.5 * (x + 0.044715x^3)))
#ifdef ENABLE_NEON
    for (int num_max = length - C16NUM; i <= num_max; i += C16NUM) {
      float16x4x4_t ins = vld4_f16(src + i);
      float32x4_t in0 = MS_CVT_F32_F16(ins.val[0]);
      float32x4_t in1 = MS_CVT_F32_F16(ins.val[1]);
      float32x4_t in2 = MS_CVT_F32_F16(ins.val[2]);
      float32x4_t in3 = MS_CVT_F32_F16(ins.val[3]);
      float32x4_t res0 = 0.5f * in0 * (1.0f + MS_TANHX4_F32((0.79788456080287f + 0.035677408136f * in0 * in0) * in0));
      float32x4_t res1 = 0.5f * in1 * (1.0f + MS_TANHX4_F32((0.79788456080287f + 0.035677408136f * in1 * in1) * in1));
      float32x4_t res2 = 0.5f * in2 * (1.0f + MS_TANHX4_F32((0.79788456080287f + 0.035677408136f * in2 * in2) * in2));
      float32x4_t res3 = 0.5f * in3 * (1.0f + MS_TANHX4_F32((0.79788456080287f + 0.035677408136f * in3 * in3) * in3));
      float16x4x4_t res = {
        MS_CVT_F16_F32(res0),
        MS_CVT_F16_F32(res1),
        MS_CVT_F16_F32(res2),
        MS_CVT_F16_F32(res3),
      };
      vst4_f16(dst + i, res);
    }
    for (int num_max = length - C4NUM; i <= num_max; i += C4NUM) {
      float32x4_t in = MS_CVT_F32_F16(vld1_f16(src + i));
      float32x4_t res = 0.5f * in * (1.0f + MS_TANHX4_F32((0.79788456080287f + 0.035677408136f * in * in) * in));
      vst1_f16(dst + i, MS_CVT_F16_F32(res));
    }
#endif
    for (; i < length; i++) {
      dst[i] =
        0.5f * src[i] *
        (1.0f + TanhOptFp16(((float16_t)0.79788456080287f + (float16_t)0.035677408136f * src[i] * src[i]) * src[i]));
    }
  } else {
#ifdef ENABLE_NEON
    int C8 = DOWN_ROUND(length, C8NUM);
    for (; i < C8; i += C8NUM) {
      float16x8_t in = vld1q_f16(src + i);
      const float16x8_t res = 0.5f * in * (1.0f + MS_ERFX8_F16(in / (float16_t)1.4142135623730951f));
      vst1q_f16(dst + i, res);
    }
#endif
    for (; i < length; i++) {
      dst[i] = 0.5f * src[i] * (1.0f + erff(src[i] / 1.4142135623730951f));
    }
  }
  return NNACL_OK;
}

int SoftplusFp16(const float16_t *src, int length, float16_t *dst) {
  int i = 0;
  for (; i < length; ++i) {
    single_exp_fp16(src[i], dst + i);
    dst[i] = log1p(dst[i]);
  }
  return NNACL_OK;
}

int EluFp16(const float16_t *src, int length, float16_t *dst, float16_t alpha) {
  int i = 0;
#ifdef ENABLE_NEON
  float16x8_t one = MS_MOVQ_F16(1.0f);
  for (; i <= length - 8; i += 8) {
    float16x8_t src_tmp = MS_LDQ_F16(src + i);
    float16x8_t exp_tmp = VexpFp16(src_tmp);  // exp(x)
    exp_tmp = MS_SUBQ_F16(exp_tmp, one);      // exp(x) - 1
    float16x8_t elu_tmp = MS_MULQ_N_F16(exp_tmp, alpha);
    uint16x8_t mask = vcleq_f16(src_tmp, MS_MOVQ_F16(0.0f));
    MS_STQ_F16(dst + i, vbslq_f16(mask, elu_tmp, src_tmp));
  }
#endif
  for (; i < length; ++i) {
    dst[i] = src[i] > 0 ? src[i] : (expm1(src[i]) * alpha);
  }
  return NNACL_OK;
}
