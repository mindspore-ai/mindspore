/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "nnacl/activation_fp32_simd.h"

int Fp32Relu(const float *src, int length, float *dst) {
  int i = 0;

  SIMD_RUN_NO_SCALAR(Fp32Relu, i, src, length, dst);

  for (; i < length; ++i) {
    dst[i] = src[i] > 0 ? src[i] : 0;
  }
  return NNACL_OK;
}

int Int32Relu(const int32_t *src, int length, int32_t *dst) {
  int i = 0;

  SIMD_RUN_NO_SCALAR(Int32Relu, i, src, length, dst);

  for (; i < length; ++i) {
    dst[i] = src[i] > 0 ? src[i] : 0;
  }
  return NNACL_OK;
}

int Fp32Relu6(const float *src, int length, float *dst) {
  int i = 0;

  SIMD_RUN_NO_SCALAR(Fp32Relu6, i, src, length, dst);

  for (; i < length; ++i) {
    if (src[i] < 0) {
      dst[i] = 0;
    } else {
      dst[i] = src[i] > 6.0f ? 6.0f : src[i];  // relu 6.0
    }
  }
  return NNACL_OK;
}

int Fp32Clip(const float *src, int length, float *dst, float min, float max) {
  int i = 0;

  SIMD_RUN_NO_SCALAR(Fp32Clip, i, src, length, dst, min, max);

  for (; i < length; ++i) {
    if (src[i] < min) {
      dst[i] = min;
    } else {
      dst[i] = src[i] > max ? max : src[i];
    }
  }
  return NNACL_OK;
}

int Int32Clip(const int *src, int length, int *dst, int min, int max) {
  int i = 0;

  SIMD_RUN_NO_SCALAR(Int32Clip, i, src, length, dst, min, max);

  for (; i < length; ++i) {
    if (src[i] < min) {
      dst[i] = min;
    } else {
      dst[i] = src[i] > max ? max : src[i];
    }
  }
  return NNACL_OK;
}

int LRelu(const float *src, int length, float *dst, float alpha) {
  int i = 0;

  SIMD_RUN_NO_SCALAR(LRelu, i, src, length, dst, alpha);

  for (; i < length; ++i) {
    dst[i] = src[i] > 0 ? src[i] : (src[i] * alpha);
  }
  return NNACL_OK;
}

int Sigmoid(const float *src, int length, float *dst) {
  int i = 0;

  SIMD_RUN_NO_SCALAR(Sigmoid, i, src, length, dst);

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

int Tanh(const float *src, int length, float *dst) {
  int i = 0;

  SIMD_RUN_NO_SCALAR(Tanh, i, src, length, dst);

  for (; i < length; ++i) {
    dst[i] = TanhOpt(src[i]);
  }
  return NNACL_OK;
}

int Swish(const float *src, int length, float *dst) {
  int i = 0;

  SIMD_RUN_NO_SCALAR(Swish, i, src, length, dst);

  for (; i < length; ++i) {
    simd_exp32(-src[i], dst + i);
    dst[i] = src[i] / (1.0f + dst[i]);
  }
  return NNACL_OK;
}

int HSwish(const float *src, int length, float *dst) {
  int i = 0;

  SIMD_RUN_NO_SCALAR(HSwish, i, src, length, dst);

  for (; i < length; ++i) {
    float in = src[i];
    float relu6 = MSMIN(MSMAX(in + C3NUM, 0), C6NUM);
    dst[i] = in * relu6 / C6NUM;
  }
  return NNACL_OK;
}

int HSigmoid(const float *src, int length, float *dst) {
  int i = 0;

  SIMD_RUN_NO_SCALAR(HSigmoid, i, src, length, dst);

  for (; i < length; ++i) {
    float relu6 = MSMIN(MSMAX(src[i] + C3NUM, 0), C6NUM);
    dst[i] = relu6 / C6NUM;
  }
  return NNACL_OK;
}

int HardTanh(const float *src, int length, float *dst, float min_val, float max_val) {
  if (max_val <= min_val) {
    return NNACL_ERR;
  }
  int i = 0;
  if (min_val == FLT_MIN) {
    SIMD_RUN_NO_SCALAR(HardTanhNoLimitMin, i, src, length, dst, min_val, max_val);

    for (; i < length; ++i) {
      dst[i] = src[i] > max_val ? max_val : src[i];
    }
  } else if (max_val == FLT_MAX) {
    SIMD_RUN_NO_SCALAR(HardTanhNoLimitMax, i, src, length, dst, min_val, max_val);

    for (; i < length; ++i) {
      dst[i] = src[i] < min_val ? min_val : src[i];
    }
  } else {
    SIMD_RUN_NO_SCALAR(HardTanhLimitMinMax, i, src, length, dst, min_val, max_val);

    for (; i < length; ++i) {
      dst[i] = src[i] < min_val ? min_val : (src[i] > max_val ? max_val : src[i]);
    }
  }
  return NNACL_OK;
}

int Gelu(const float *src, int length, float *dst, bool approximate) {
  if (src == NULL || dst == NULL) {
    return NNACL_ERR;
  }
  int i = 0;
  if (approximate) {
    SIMD_RUN_NO_SCALAR(GeluApproximate, i, src, length, dst);

    // dst = 0.5 * x * (1 + tanh((2 / pi) ^ 0.5 * (x + 0.044715x^3)))
    for (; i < length; i++) {
      dst[i] = 0.5 * src[i] * (1.0 + TanhOpt((0.79788456080287f + 0.035677408136f * src[i] * src[i]) * src[i]));
    }
  } else {
    SIMD_RUN_NO_SCALAR(Gelu, i, src, length, dst);

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

int Elu(const float *src, int length, float *dst, float alpha) {
  int i = 0;

  SIMD_RUN_NO_SCALAR(Elu, i, src, length, dst, alpha);

  for (; i < length; ++i) {
    dst[i] = src[i] > 0 ? src[i] : (expm1(src[i]) * alpha);
  }
  return NNACL_OK;
}

void Celu(const float *src, int length, float *dst, float alpha) {
  int i = 0;

  SIMD_RUN_NO_SCALAR(Celu, i, src, length, dst, alpha);

  for (; i < length; ++i) {
    dst[i] = src[i] > 0 ? src[i] : (expm1(src[i] / alpha) * alpha);
  }
  return;
}

int HShrink(const float *src, int length, float *dst, float lambd) {
  int i = 0;
  const float neg_lambd = -1 * lambd;
  SIMD_RUN_NO_SCALAR(HShrink, i, src, length, dst, lambd);

  for (; i < length; ++i) {
    dst[i] = src[i] >= neg_lambd && src[i] <= lambd ? 0 : src[i];
  }
  return NNACL_OK;
}

int SoftShrink(const float *src, int length, float *dst, float lambd) {
  int i = 0;
  const float neg_lambd = -1 * lambd;
  SIMD_RUN_NO_SCALAR(SoftShrink, i, src, length, dst, lambd);

  for (; i < length; ++i) {
    dst[i] = (src[i] > lambd) ? (src[i] - lambd) : ((src[i] < neg_lambd) ? (src[i] + lambd) : (0));
  }
  return NNACL_OK;
}

int SoftsignFp32Opt(const float *src, int length, float *dst) {
  int i = 0;

  SIMD_RUN_NO_SCALAR(SoftsignFp32Opt, i, src, length, dst);
  for (; i < length; ++i) {
    dst[i] = src[i] / (1.0 + fabsf(src[i]));
  }
  return NNACL_OK;
}
