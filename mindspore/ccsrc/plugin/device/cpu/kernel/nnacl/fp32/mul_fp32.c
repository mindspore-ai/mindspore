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
#include "nnacl/fp32/mul_fp32.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "nnacl/mul_fp32_simd.h"
#include "nnacl/errorcode.h"

int BroadcastMul(const float *in0, const float *in1, float *tile_in0, float *tile_in1, float *out, int size,
                 ArithmeticParameter *param) {
  TileDimensionsFp32(in0, in1, tile_in0, tile_in1, param);
  return ElementMul(tile_in0, tile_in1, out, size);
}

int ElementMul(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementMul, index, in0, in1, out, size);
  for (; index < size; index++) {
    out[index] = in0[index] * in1[index];
  }
  return NNACL_OK;
}

int ElementMulRelu(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementMulRelu, index, in0, in1, out, size);
  for (; index < size; index++) {
    float res = in0[index] * in1[index];
    out[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

int ElementMulRelu6(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementMulRelu6, index, in0, in1, out, size);
  for (; index < size; index++) {
    out[index] = MSMIN(MSMAX(in0[index] * in1[index], 0), 6);
  }
  return NNACL_OK;
}

int ElementMulInt(const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementMulInt, index, in0, in1, out, size);
  for (; index < size; index++) {
    out[index] = in0[index] * in1[index];
  }
  return NNACL_OK;
}

int ElementMulReluInt(const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementMulReluInt, index, in0, in1, out, size);
  for (; index < size; index++) {
    int res = in0[index] * in1[index];
    out[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

int ElementMulRelu6Int(const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementMulRelu6Int, index, in0, in1, out, size);
  for (; index < size; index++) {
    out[index] = MSMIN(MSMAX(in0[index] * in1[index], 0), 6);
  }
  return NNACL_OK;
}

int ElementOptMul(const float *in0, const float *in1, float *out, int size, bool first_scalar) {
  int index = 0;
  if (first_scalar) {
    SIMD_RUN_NO_SCALAR(ElementOptMulNum0, index, in0, in1, out, size);

    for (; index < size; index++) {
      out[index] = in0[0] * in1[index];
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptMulNum1, index, in0, in1, out, size);

    for (; index < size; index++) {
      out[index] = in0[index] * in1[0];
    }
  }
  return NNACL_OK;
}

int ElementOptMulRelu(const float *in0, const float *in1, float *out, int size, bool first_scalar) {
  int index = 0;
  if (first_scalar) {
    SIMD_RUN_NO_SCALAR(ElementOptMulReluNum0, index, in0, in1, out, size);
    for (; index < size; index++) {
      out[index] = MSMAX(in0[0] * in1[index], 0);
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptMulReluNum1, index, in0, in1, out, size);
    for (; index < size; index++) {
      out[index] = MSMAX(in0[index] * in1[0], 0);
    }
  }
  return NNACL_OK;
}

int ElementOptMulRelu6(const float *in0, const float *in1, float *out, int size, bool first_scalar) {
  int index = 0;
  if (first_scalar) {
    SIMD_RUN_NO_SCALAR(ElementOptMulRelu6Num0, index, in0, in1, out, size);

    for (; index < size; index++) {
      out[index] = MSMIN(MSMAX(in0[0] * in1[index], 0), 6);
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptMulRelu6Num1, index, in0, in1, out, size);

    for (; index < size; index++) {
      out[index] = MSMIN(MSMAX(in0[index] * in1[0], 0), 6);
    }
  }
  return NNACL_OK;
}

int ElementOptMulInt(const int32_t *in0, const int32_t *in1, int32_t *out, int size, bool first_scalar) {
  int index = 0;
  if (first_scalar) {
    SIMD_RUN_NO_SCALAR(ElementOptMulIntNum0, index, in0, in1, out, size);
    for (; index < size; index++) {
      out[index] = in0[0] * in1[index];
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptMulIntNum1, index, in0, in1, out, size);
    for (; index < size; index++) {
      out[index] = in0[index] * in1[0];
    }
  }
  return NNACL_OK;
}

int ElementOptMulReluInt(const int32_t *in0, const int32_t *in1, int32_t *out, int size, bool first_scalar) {
  int index = 0;
  if (first_scalar) {
    SIMD_RUN_NO_SCALAR(ElementOptMulReluIntNum0, index, in0, in1, out, size);
    for (; index < size; index++) {
      out[index] = MSMAX(in0[0] * in1[index], 0);
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptMulReluIntNum1, index, in0, in1, out, size);
    for (; index < size; index++) {
      out[index] = MSMAX(in0[index] * in1[0], 0);
    }
  }
  return NNACL_OK;
}

int ElementOptMulRelu6Int(const int32_t *in0, const int32_t *in1, int32_t *out, int size, bool first_scalar) {
  int index = 0;
  if (first_scalar) {
    SIMD_RUN_NO_SCALAR(ElementOptMulRelu6IntNum0, index, in0, in1, out, size);
    for (; index < size; index++) {
      out[index] = MSMIN(MSMAX(in0[0] * in1[index], 0), 6);
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptMulRelu6IntNum1, index, in0, in1, out, size);
    for (; index < size; index++) {
      out[index] = MSMIN(MSMAX(in0[index] * in1[0], 0), 6);
    }
  }
  return NNACL_OK;
}
