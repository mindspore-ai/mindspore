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
#include "nnacl/activation_grad.h"

int ReluGrad(float *src0, float *src1, int length, float *dst) {
  for (int i = 0; i < length; ++i) {
    dst[i] = src1[i] > 0 ? 1.0f : 0.0f;
  }
  ElementMul(src0, dst, dst, length);
  return NNACL_OK;
}

int Relu6Grad(float *src0, float *src1, int length, float *dst) {
  for (int i = 0; i < length; ++i) {
    if (src1[i] < 0) {
      dst[i] = 0;
    } else {
      dst[i] = src1[i] > 6.0f ? 0.0f : 1.0f;
    }
  }
  ElementMul(src0, dst, dst, length);
  return NNACL_OK;
}

int LReluGrad(float *src0, float *src1, int length, float *dst, float alpha) {
  for (int i = 0; i < length; ++i) {
    dst[i] = src1[i] > 0.0f ? 1.0f : alpha;
  }
  ElementMul(src0, dst, dst, length);
  return NNACL_OK;
}

int SigmoidGrad(float *src0, float *src1, int length, float *dst) {
  for (int i = 0; i < length; ++i) {
    dst[i] = src0[i] * (src1[i] * (1.0f - src1[i]));
  }
  return NNACL_OK;
}

int TanhGrad(float *src0, float *src1, int length, float *dst) {
  for (int i = 0; i < length; ++i) {
    dst[i] = (1.0f - (src1[i] * src1[i])) * src0[i];
  }
  return NNACL_OK;
}

int HSwishGrad(float *src0, float *src1, int length, float *dst) {
  for (int i = 0; i < length; ++i) {
    float tmp = (src1[i] > 3.0f ? 1.0f : (src1[i] < -3.0f ? 0.0f : (2.0f * src1[i] + 3.0f) / 6.0f));
    dst[i] = tmp * src0[i];
  }
  return NNACL_OK;
}

int HSigmoidGrad(float *src0, float *src1, int length, float *dst) {
  for (int i = 0; i < length; ++i) {
    float tmp = (src1[i] > 3.0f ? 1.0f : (src1[i] < -3.0f ? 0.0f : 1.0f / 6.0f));
    dst[i] = tmp * src0[i];
  }
  return NNACL_OK;
}
