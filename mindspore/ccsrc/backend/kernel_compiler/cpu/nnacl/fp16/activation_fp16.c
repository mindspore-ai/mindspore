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
  int i = 0;
#ifdef ENABLE_NEON
  int ele_c8 = DOWN_ROUND(ele_num, C8NUM);
  for (; i < ele_c8; i += C8NUM) {
    float16x8_t src_tmp = vld1q_f16(src + i);
    float16x8_t mul_tmp = vmulq_n_f16(src_tmp, alpha);
    uint16x8_t mask = vcgtq_f16(src_tmp, vdupq_n_f16(0.0f));
    vst1q_f16(dst + i, vbslq_f16(mask, src_tmp, mul_tmp));
  }
#endif
  for (; i < ele_num; ++i) {
    dst[i] = src[i] > (float16_t)0.0f ? src[i] : (src[i] * alpha);
  }
  return NNACL_OK;
}

int SigmoidFp16(const float16_t *src, float16_t *dst, int ele_num) {
  int i = 0;
#ifdef ENABLE_NEON
  int count = (ele_num / C4NUM) * C4NUM;
  for (; i < count; i += C4NUM) {
    float32x4_t tmp;
    simd_exp(vnegq_f32(vcvt_f32_f16(vld1_f16(src + i))), (float *)&tmp);
    vst1_f16(dst + i, vcvt_f16_f32(MS_DIVQ_F32(vdupq_n_f32(1.0f), vaddq_f32(vdupq_n_f32(1.0f), tmp))));
  }
#endif
  for (; i < ele_num; ++i) {
    float temp;
    single_exp(-src[i], &temp);
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
  int ret = SigmoidFp16(src, dst, ele_num);
  if (ret != NNACL_OK) {
    return NNACL_ERR;
  }
  int index = 0;
  for (; index < ele_num; index++) {
    dst[index] = src[index] * dst[index];
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
    int C8 = DOWN_ROUND(length, C8NUM);
    for (; i < C8; i += C8NUM) {
      float16x8_t in = vld1q_f16(src + i);
      float16x8_t res =
        0.5f * in * (1.0f + MS_TANHX8_F16(((float16_t)0.79788456080287f + (float16_t)0.035677408136f * in * in) * in));
      vst1q_f16(dst + i, res);
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

int EluFp16(const float16_t *src, int length, float16_t *dst, float16_t alpha) {
  int i = 0;
#ifdef ENABLE_NEON
  float16x8_t one = MS_MOVQ_F16(1.0f);
  for (; i <= length - 8; i += 8) {
    float16x8_t src_tmp = MS_LDQ_F16(src + i);
    float16x8_t exp_tmp = VexpFp16(src_tmp);  // exp(x)
    exp_tmp = MS_SUBQ_F16(exp_tmp, one);      // exp(x) - 1
    float16x8_t elu_tmp = MS_MULQ_N_F16(exp_tmp, alpha);
    uint16x8_t mask = MS_CMPGTQ_F16(src_tmp, MS_MOVQ_F16(0.0f));
    MS_STQ_F16(dst + i, vbslq_f16(elu_tmp, src_tmp, mask));
  }
#endif
  for (; i < length; ++i) {
    dst[i] = src[i] > 0 ? src[i] : (expm1(src[i]) * alpha);
  }
  return NNACL_OK;
}
