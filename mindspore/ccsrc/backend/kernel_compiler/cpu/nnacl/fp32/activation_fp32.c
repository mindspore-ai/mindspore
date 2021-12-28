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
#include <float.h>
#include "nnacl/fp32/activation_fp32.h"
#include "nnacl/fp32/exp_fp32.h"
#include "nnacl/errorcode.h"

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdFp32ReluCoreCalc(block_size, block_num, src, length, dst, i)                                        \
  do {                                                                                                          \
    MS_FLOAT_32xN(block_num) zero_##block_num = MS_MOVN_F32(block_size, 0.0f);                                  \
    for (int block_max_size = length - block_num + 1; i < block_max_size; i += block_num) {                     \
      MS_ST_F32(block_size, dst + i, MS_MAX_F32(block_size, MS_LD_F32(block_size, src + i), zero_##block_num)); \
    }                                                                                                           \
  } while (0)

int Fp32Relu(const float *src, int length, float *dst) {
  int i = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdFp32ReluCoreCalc, src, length, dst, i);

  for (; i < length; ++i) {
    dst[i] = src[i] > 0 ? src[i] : 0;
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdInt32ReluCoreCalc(block_size, block_num, src, length, dst, i)                                             \
  do {                                                                                                                \
    MS_INT_32xN(block_num) zero_##block_num = MS_MOVN_EPI32(block_size, 0.0f);                                        \
    for (int block_max_size = length - block_num + 1; i < block_max_size; i += block_num) {                           \
      MS_ST_EPI32(block_size, dst + i, MS_MAX_EPI32(block_size, MS_LD_EPI32(block_size, src + i), zero_##block_num)); \
    }                                                                                                                 \
  } while (0)

int Int32Relu(const int32_t *src, int length, int32_t *dst) {
  int i = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdInt32ReluCoreCalc, src, length, dst, i);

  for (; i < length; ++i) {
    dst[i] = src[i] > 0 ? src[i] : 0;
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdFp32Relu6CoreCalc(block_size, block_num, src, length, dst, i)                                          \
  do {                                                                                                             \
    MS_FLOAT_32xN(block_num) zero_##block_num = MS_MOVN_F32(block_size, 0.0f);                                     \
    MS_FLOAT_32xN(block_num) six_##block_num = MS_MOVN_F32(block_size, 6.0f);                                      \
    for (int block_max_size = length - block_num + 1; i < block_max_size; i += block_num) {                        \
      MS_FLOAT_32xN(block_num) dst_tmp = MS_MAX_F32(block_size, MS_LD_F32(block_size, src + i), zero_##block_num); \
      dst_tmp = MS_MIN_F32(block_size, dst_tmp, six_##block_num);                                                  \
      MS_ST_F32(block_size, dst + i, dst_tmp);                                                                     \
    }                                                                                                              \
  } while (0)
int Fp32Relu6(const float *src, int length, float *dst) {
  int i = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdFp32Relu6CoreCalc, src, length, dst, i);

  for (; i < length; ++i) {
    if (src[i] < 0) {
      dst[i] = 0;
    } else {
      dst[i] = src[i] > 6.0f ? 6.0f : src[i];  // relu 6.0
    }
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdLReluCoreCalc(block_size, block_num, src, length, dst, alpha, i)                            \
  for (int block_max_size = length - block_num + 1; i < block_max_size; i += block_num) {               \
    MS_FLOAT_32xN(block_num) src_tmp = MS_LD_F32(block_size, src + i);                                  \
    MS_FLOAT_32xN(block_num) mul_tmp = MS_MUL_N_F32(block_size, src_tmp, alpha);                        \
    MS_MASK##block_size##_TYPE mask = MS_CMPGT_F32(block_size, src_tmp, MS_MOVN_F32(block_size, 0.0f)); \
    MS_ST_F32(block_size, dst + i, MS_BLEND_F32(block_size, mul_tmp, src_tmp, mask));                   \
  }

int LRelu(const float *src, int length, float *dst, float alpha) {
  int i = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdLReluCoreCalc, src, length, dst, alpha, i);

  for (; i < length; ++i) {
    dst[i] = src[i] > 0 ? src[i] : (src[i] * alpha);
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdSigmoidCoreCalc(block_size, block_num, src, length, dst, i)                                                \
  for (int block_max_size = length - block_num + 1; i < block_max_size; i += block_num) {                              \
    MS_EXP_ST_F32(block_size, MS_SUB_F32(block_size, MS_MOVN_F32(block_size, 0.0f), (MS_LD_F32(block_size, src + i))), \
                  dst + i);                                                                                            \
    MS_ST_F32(block_size, dst + i,                                                                                     \
              MS_DIV_F32(block_size, MS_MOVN_F32(block_size, 1.0f),                                                    \
                         MS_ADD_F32(block_size, MS_MOVN_F32(block_size, 1.0f), MS_LD_F32(block_size, dst + i))));      \
  }

int Sigmoid(const float *src, int length, float *dst) {
  int i = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdSigmoidCoreCalc, src, length, dst, i);

  for (; i < length; ++i) {
    simd_exp32(-src[i], dst + i);
    dst[i] = 1.0f / (1.0f + dst[i]);
  }
  return NNACL_OK;
}

float TanhOpt(float src) {
  if (src > 5.0) {  // src > 5.0, tanh(src) = 1.0f
    return 1.0f;
  } else if (src < -5.0) {  // src < -5.0, tanh(src) = -1.0f
    return -1.0f;
  } else {
    float square = src * src;
    float a = (((square + 378.0f) * square + 17325.0f) * square + 135135.0f) * src;
    float b = ((28.0f * square + 3150.0f) * square + 62370.0f) * square + 135135.0f;
    return a / b;
  }
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdTanhCoreCalc(block_size, block_num, src, length, dst, i)                      \
  for (int block_max_size = length - block_num + 1; i < block_max_size; i += block_num) { \
    MS_FLOAT_32xN(block_num) input = MS_LD_F32(block_size, src + i);                      \
    MS_ST_F32(block_size, dst + i, MS_TANHX##block_num##_F32(input));                     \
  }

int Tanh(const float *src, int length, float *dst) {
  int i = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdTanhCoreCalc, src, length, dst, i);

  for (; i < length; ++i) {
    dst[i] = TanhOpt(src[i]);
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdSwishCoreCalc(block_size, block_num, src, length, dst, i)                     \
  for (int block_max_size = length - block_num + 1; i < block_max_size; i += block_num) { \
    MS_FLOAT_32xN(block_num) src_value = MS_LD_F32(block_size, src + i);                  \
    MS_FLOAT_32xN(block_num) sigmoid_value = MS_LD_F32(block_size, dst + i);              \
    MS_FLOAT_32xN(block_num) result = MS_MUL_F32(block_size, src_value, sigmoid_value);   \
    MS_ST_F32(block_size, dst + i, result);                                               \
  }

int Swish(const float *src, int length, float *dst) {
  int ret = Sigmoid(src, length, dst);
  if (ret != NNACL_OK) {
    return NNACL_ERR;
  }
  int i = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdSwishCoreCalc, src, length, dst, i);

  for (; i < length; ++i) {
    dst[i] = src[i] * dst[i];
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdHSwishCoreCalc(block_size, block_num, src, length, dst, i)                                         \
  for (int block_max_size = length - block_num + 1; i < block_max_size; i += block_num) {                      \
    MS_FLOAT_32xN(block_num) src_value = MS_LD_F32(block_size, src + i);                                       \
    MS_FLOAT_32xN(block_num) relu6 = MS_CLAMP_N_F32(block_size, MS_ADD_N_F32(block_size, src_value, 3), 0, 6); \
    MS_ST_F32(block_size, dst + i, MS_DIV_N_F32(block_size, MS_MUL_F32(block_size, src_value, relu6), 6));     \
  }

int HSwish(const float *src, int length, float *dst) {
  int i = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdHSwishCoreCalc, src, length, dst, i);

  for (; i < length; ++i) {
    float in = src[i];
    float relu6 = MSMIN(MSMAX(in + C3NUM, 0), C6NUM);
    dst[i] = in * relu6 / C6NUM;
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdHSigmoidCoreCalc(block_size, block_num, src, length, dst, i)                                       \
  for (int block_max_size = length - block_num + 1; i < block_max_size; i += block_num) {                      \
    MS_FLOAT_32xN(block_num) src_value = MS_LD_F32(block_size, src + i);                                       \
    MS_FLOAT_32xN(block_num) relu6 = MS_CLAMP_N_F32(block_size, MS_ADD_N_F32(block_size, src_value, 3), 0, 6); \
    MS_ST_F32(block_size, dst + i, MS_DIV_N_F32(block_size, relu6, 6));                                        \
  }

int HSigmoid(const float *src, int length, float *dst) {
  int i = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdHSigmoidCoreCalc, src, length, dst, i);

  for (; i < length; ++i) {
    float relu6 = MSMIN(MSMAX(src[i] + C3NUM, 0), C6NUM);
    dst[i] = relu6 / C6NUM;
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdHardTanhCoreCalc1(block_size, block_num, src, length, dst, min_val, max_val, i)            \
  for (int block_max_size = length - block_num + 1; i < block_max_size; i += block_num) {              \
    MS_ST_F32(block_size, dst + i, MS_MIN_N_F32(block_size, MS_LD_F32(block_size, src + i), max_val)); \
  }

#define SimdHardTanhCoreCalc2(block_size, block_num, src, length, dst, min_val, max_val, i)            \
  for (int block_max_size = length - block_num + 1; i < block_max_size; i += block_num) {              \
    MS_ST_F32(block_size, dst + i, MS_MAX_N_F32(block_size, MS_LD_F32(block_size, src + i), min_val)); \
  }

#define SimdHardTanhCoreCalc3(block_size, block_num, src, length, dst, min_val, max_val, i)                       \
  for (int block_max_size = length - block_num + 1; i < block_max_size; i += block_num) {                         \
    MS_ST_F32(block_size, dst + i, MS_CLAMP_N_F32(block_size, MS_LD_F32(block_size, src + i), min_val, max_val)); \
  }

int HardTanh(const float *src, int length, float *dst, float min_val, float max_val) {
  if (max_val <= min_val) {
    return NNACL_ERR;
  }
  int i = 0;
  if (min_val == FLT_MIN) {
    MS_SIMD_RUN_NO_SCALAR(SimdHardTanhCoreCalc1, src, length, dst, min_val, max_val, i);

    for (; i < length; ++i) {
      dst[i] = src[i] > max_val ? max_val : src[i];
    }
  } else if (max_val == FLT_MAX) {
    MS_SIMD_RUN_NO_SCALAR(SimdHardTanhCoreCalc2, src, length, dst, min_val, max_val, i);

    for (; i < length; ++i) {
      dst[i] = src[i] < min_val ? min_val : src[i];
    }
  } else {
    MS_SIMD_RUN_NO_SCALAR(SimdHardTanhCoreCalc3, src, length, dst, min_val, max_val, i);

    for (; i < length; ++i) {
      dst[i] = src[i] < min_val ? min_val : (src[i] > max_val ? max_val : src[i]);
    }
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdSoftplusCoreCalc(block_size, block_num, src, length, dst, i)                                              \
  for (int block_max_size = length - block_num + 1; i < block_max_size; i += block_num) {                             \
    MS_FLOAT_32xN(block_num) in = MS_LD_F32(block_size, src + i);                                                     \
    MS_FLOAT_32xN(block_num) tmp1 = MS_MUL_F32(block_size, MS_MUL_N_F32(block_size, in, 0.035677408136f), in);        \
    MS_FLOAT_32xN(block_num) tmp2 = MS_MUL_F32(block_size, MS_ADD_N_F32(block_size, tmp1, 0.79788456080287f), in);    \
    const MS_FLOAT_32xN(block_num) res = MS_MUL_F32(block_size, MS_MUL_N_F32(block_size, in, 0.5f),                   \
                                                    MS_ADD_N_F32(block_size, MS_TANHX##block_num##_F32(tmp2), 1.0f)); \
    MS_ST_F32(block_size, dst + i, res);                                                                              \
  }

int Gelu(const float *src, int length, float *dst, bool approximate) {
  if (src == NULL || dst == NULL) {
    return NNACL_ERR;
  }
  int i = 0;
  if (approximate) {
    MS_SIMD_RUN_NO_SCALAR(SimdSoftplusCoreCalc, src, length, dst, i);

    // dst = 0.5 * x * (1 + tanh((2 / pi) ^ 0.5 * (x + 0.044715x^3)))
    for (; i < length; i++) {
      dst[i] = 0.5 * src[i] * (1.0 + TanhOpt((0.79788456080287f + 0.035677408136f * src[i] * src[i]) * src[i]));
    }
  } else {
#if defined(ENABLE_AVX512) || defined(ENABLE_AVX) || defined(ENABLE_SSE) || defined(ENABLE_ARM)
    MS_FLOAT32X4 para1 = MS_MOVQ_F32(1.4142135623730951f);
    MS_FLOAT32X4 para2 = MS_MOVQ_F32(1.0f);
    MS_FLOAT32X4 para3 = MS_MOVQ_F32(0.5f);
    int C4 = DOWN_ROUND(length, C4NUM);
    for (; i < C4; i += C4NUM) {
      MS_FLOAT32X4 in = MS_LDQ_F32(src + i);
      MS_FLOAT32X4 res = MS_MULQ_F32(MS_MULQ_F32(para3, in), MS_ADDQ_F32(para2, MS_ERFX4_F32(MS_DIVQ_F32(in, para1))));
      MS_STQ_F32(dst + i, res);
    }
#endif
    for (; i < length; i++) {
      dst[i] =
        0.5 * src[i] * (1.0 + erf(src[i] / 1.4142135623730951f));  // dst = 0.5 * x * (1.0 + x / 1.4142135623730951f))
    }
  }
  return NNACL_OK;
}

int Softplus(const float *src, int length, float *dst) {
  int i = 0;
  for (; i < length; ++i) {
    simd_exp32(src[i], dst + i);
    dst[i] = log1p(dst[i]);
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdEluCoreCalc(block_size, block_num, src, length, dst, alpha, i)                              \
  for (int block_max_size = length - block_num + 1; i < block_max_size; i += block_num) {               \
    MS_FLOAT_32xN(block_num) src_tmp = MS_LD_F32(block_size, src + i);                                  \
    MS_FLOAT_32xN(block_num) exp_tmp = simd_exp##block_size##_f32(src_tmp);                             \
    exp_tmp = MS_SUB_N_F32(block_size, exp_tmp, 1.0f);                                                  \
    MS_FLOAT_32xN(block_num) elu_tmp = MS_MUL_N_F32(block_size, exp_tmp, alpha);                        \
    MS_MASK##block_size##_TYPE mask = MS_CMPLE_F32(block_size, src_tmp, MS_MOVN_F32(block_size, 0.0f)); \
    MS_ST_F32(block_size, dst + i, MS_BLEND_F32(block_size, src_tmp, elu_tmp, mask));                   \
  }

int Elu(const float *src, int length, float *dst, float alpha) {
  int i = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdEluCoreCalc, src, length, dst, alpha, i);

  for (; i < length; ++i) {
    dst[i] = src[i] > 0 ? src[i] : (expm1(src[i]) * alpha);
  }
  return NNACL_OK;
}
