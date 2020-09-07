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

#include "nnacl/fp32/activation.h"
#include "nnacl/errorcode.h"

int Fp32Relu(const float *src, int length, float *dst) {
  for (int i = 0; i < length; ++i) {
    dst[i] = src[i] > 0 ? src[i] : 0;
  }
  return NNACL_OK;
}

int Fp32Relu6(const float *src, int length, float *dst) {
  for (int i = 0; i < length; ++i) {
    if (src[i] < 0) {
      dst[i] = 0;
    } else {
      dst[i] = src[i] > 6.0f ? 6.0f : src[i];
    }
  }
  return NNACL_OK;
}

int LRelu(const float *src, int length, float *dst, float alpha) {
  for (int i = 0; i < length; ++i) {
    dst[i] = src[i] > 0 ? src[i] : (src[i] * alpha);
  }
  return NNACL_OK;
}

int Sigmoid(const float *src, int length, float *dst) {
  const float upper_bound = 16.619047164916992188f;
  const float lower_bound = -9.0f;
  for (int i = 0; i < length; ++i) {
    float input_val = src[i];
    float result;
    if (input_val > upper_bound) {
      result = 1.0f;
    } else if (input_val < lower_bound) {
      result = exp(input_val);
    } else {
      result = 1.0f / (1.0f + exp(-input_val));
    }
    dst[i] = result;
  }
  return NNACL_OK;
}

float TanhOpt(float src) {
  if (src > 5.0) {
    return 1.0f;
  } else if (src < -5.0) {
    return -1.0f;
  } else {
    float square = src * src;
    float a = (((square + 378.0f) * square + 17325.0f) * square + 135135.0f) * src;
    float b = ((28.0f * square + 3150.0f) * square + 62370.0f) * square + 135135.0f;
    return a / b;
  }
}

int Tanh(const float *src, int length, float *dst) {
  for (int i = 0; i < length; ++i) {
    dst[i] = TanhOpt(src[i]);
  }
  return NNACL_OK;
}

int HSwish(const float *src, int length, float *dst) {
  for (int i = 0; i < length; ++i) {
    float in = src[i];
    float relu6 = MSMIN(MSMAX(in + 3, 0), 6);
    dst[i] = in * relu6 / 6;
  }
  return NNACL_OK;
}
