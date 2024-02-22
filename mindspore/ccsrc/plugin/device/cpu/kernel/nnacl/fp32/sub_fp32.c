/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "nnacl/fp32/sub_fp32.h"
#include "nnacl/sub_fp32_simd.h"
#include "nnacl/errorcode.h"

int ElementOptSub(const float *in0, const float *in1, float *out, int size, bool first_scalar) {
  int index = 0;
  if (first_scalar) {
    SIMD_RUN_NO_SCALAR(ElementOptSubNum0, index, in0, in1, out, size);
    for (; index < size; index++) {
      out[index] = in0[0] - in1[index];
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptSubNum1, index, in0, in1, out, size);
    for (; index < size; index++) {
      out[index] = in0[index] - in1[0];
    }
  }
  return NNACL_OK;
}

int ElementOptSubExt(const float *in0, const float *in1, const float alpha, float *out, int size, bool first_scalar) {
  int index = 0;
  if (first_scalar) {
    SIMD_RUN_NO_SCALAR(ElementOptSubExtNum0, index, in0, in1, alpha, out, size);
    for (; index < size; index++) {
      out[index] = in0[0] - in1[index] * alpha;
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptSubExtNum1, index, in0, in1, alpha, out, size);
    for (; index < size; index++) {
      out[index] = in0[index] - in1[0] * alpha;
    }
  }
  return NNACL_OK;
}

int ElementSubExt(const float *in0, const float *in1, const float alpha, float *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementSubExt, index, in0, in1, alpha, out, size);
  for (; index < size; index++) {
    out[index] = in0[index] - in1[index] * alpha;
  }
  return NNACL_OK;
}

int ElementOptSubInt(const int32_t *in0, const int32_t *in1, int32_t *out, int size, bool first_scalar) {
  int index = 0;
  if (first_scalar) {
    SIMD_RUN_NO_SCALAR(ElementOptSubIntNum0, index, in0, in1, out, size);
    for (; index < size; index++) {
      out[index] = in0[0] - in1[index];
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptSubIntNum1, index, in0, in1, out, size);
    for (; index < size; index++) {
      out[index] = in0[index] - in1[0];
    }
  }
  return NNACL_OK;
}

int ElementOptSubRelu(const float *in0, const float *in1, float *out, int size, bool first_scalar) {
  int index = 0;
  if (first_scalar) {
    SIMD_RUN_NO_SCALAR(ElementOptSubReluNum0, index, in0, in1, out, size);
    for (; index < size; index++) {
      out[index] = MSMAX(in0[0] - in1[index], 0);
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptSubReluNum1, index, in0, in1, out, size);
    for (; index < size; index++) {
      out[index] = MSMAX(in0[index] - in1[0], 0);
    }
  }
  return NNACL_OK;
}

int ElementOptSubRelu6(const float *in0, const float *in1, float *out, int size, bool first_scalar) {
  int index = 0;
  if (first_scalar) {
    SIMD_RUN_NO_SCALAR(ElementOptSubRelu6Num0, index, in0, in1, out, size);
    for (; index < size; index++) {
      out[index] = MSMIN(MSMAX(in0[0] - in1[index], 0), 6);
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptSubRelu6Num1, index, in0, in1, out, size);
    for (; index < size; index++) {
      out[index] = MSMIN(MSMAX(in0[index] - in1[0], 0), 6);
    }
  }
  return NNACL_OK;
}

int ElementSub(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementSub, index, in0, in1, out, size);
  for (; index < size; index++) {
    out[index] = in0[index] - in1[index];
  }
  return NNACL_OK;
}

int ElementSubInt(const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementSubInt, index, in0, in1, out, size);
  for (; index < size; index++) {
    out[index] = in0[index] - in1[index];
  }
  return NNACL_OK;
}

int ElementSubRelu(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementSubRelu, index, in0, in1, out, size);
  for (; index < size; index++) {
    float res = in0[index] - in1[index];
    out[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

int ElementSubRelu6(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementSubRelu6, index, in0, in1, out, size);
  for (; index < size; index++) {
    out[index] = MSMIN(MSMAX(in0[index] - in1[index], 0), 6);
  }

  return NNACL_OK;
}
