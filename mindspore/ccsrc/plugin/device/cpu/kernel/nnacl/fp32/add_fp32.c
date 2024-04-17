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

#include "nnacl/fp32/add_fp32.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "nnacl/add_fp32_simd.h"

int ElementOptAdd(const float *in0, const float *in1, float *out, int size, bool first_scalar) {
  int index = 0;
  if (first_scalar) {
    SIMD_RUN_NO_SCALAR(ElementOptAdd, index, in0, in1, out, size);
    for (; index < size; index++) {
      out[index] = in0[0] + in1[index];
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptAdd, index, in1, in0, out, size);
    for (; index < size; index++) {
      out[index] = in0[index] + in1[0];
    }
  }
  return NNACL_OK;
}

int ElementOptAddExt(const float *in0, const float *in1, const float alpha, float *out, int size, bool first_scalar) {
  int index = 0;
  if (first_scalar) {
    SIMD_RUN_NO_SCALAR(ElementOptAddExtNum0, index, in0, in1, alpha, out, size);
    for (; index < size; index++) {
      out[index] = in0[0] + in1[index] * alpha;
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptAddExtNum1, index, in0, in1, alpha, out, size);
    for (; index < size; index++) {
      out[index] = in0[index] + in1[0] * alpha;
    }
  }
  return NNACL_OK;
}

int ElementOptAddInt(const int32_t *in0, const int32_t *in1, int32_t *out, int size, bool first_scalar) {
  int index = 0;
  if (first_scalar) {
    SIMD_RUN_NO_SCALAR(ElementOptAddInt, index, in0, in1, out, size);
    for (; index < size; index++) {
      out[index] = in0[0] + in1[index];
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptAddInt, index, in1, in0, out, size);
    for (; index < size; index++) {
      out[index] = in0[index] + in1[0];
    }
  }
  return NNACL_OK;
}

int ElementOptAddRelu(const float *in0, const float *in1, float *out, int size, bool first_scalar) {
  int index = 0;
  if (first_scalar) {
    SIMD_RUN_NO_SCALAR(ElementOptAddRelu, index, in0, in1, out, size);
    for (; index < size; index++) {
      out[index] = MSMAX(in0[0] + in1[index], 0);
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptAddRelu, index, in1, in0, out, size);
    for (; index < size; index++) {
      out[index] = MSMAX(in0[index] + in1[0], 0);
    }
  }
  return NNACL_OK;
}

int ElementOptAddRelu6(const float *in0, const float *in1, float *out, int size, bool first_scalar) {
  int index = 0;
  if (first_scalar) {
    SIMD_RUN_NO_SCALAR(ElementOptAddRelu6, index, in0, in1, out, size);
    for (; index < size; index++) {
      out[index] = MSMIN(MSMAX(in0[0] + in1[index], 0), 6);
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptAddRelu6, index, in1, in0, out, size);
    for (; index < size; index++) {
      out[index] = MSMIN(MSMAX(in0[index] + in1[0], 0), 6);
    }
  }
  return NNACL_OK;
}

int BroadcastAdd(const float *in0, const float *in1, float *tile_in0, float *tile_in1, float *out, int size,
                 ArithmeticParameter *param) {
  TileDimensionsFp32(in0, in1, tile_in0, tile_in1, param);
  return ElementAdd(tile_in0, tile_in1, out, size);
}

int ElementAdd(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementAdd, index, in0, in1, out, size);
  for (; index < size; index++) {
    out[index] = in0[index] + in1[index];
  }
  return NNACL_OK;
}

int ElementAddExt(const float *in0, const float *in1, const float alpha, float *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementAddExt, index, in0, in1, alpha, out, size);
  for (; index < size; index++) {
    out[index] = in0[index] + in1[index] * alpha;
  }
  return NNACL_OK;
}

int ElementAddRelu(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementAddRelu, index, in0, in1, out, size);
  for (; index < size; index++) {
    float res = in0[index] + in1[index];
    out[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

int ElementAddRelu6(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementAddRelu6, index, in0, in1, out, size);
  for (; index < size; index++) {
    out[index] = MSMIN(MSMAX(in0[index] + in1[index], 0), 6);
  }
  return NNACL_OK;
}

int ElementAddInt(const int32_t *in0, const int32_t *in1, int32_t *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementAddInt, index, in0, in1, out, size);
  for (; index < size; index++) {
    out[index] = in0[index] + in1[index];
  }
  return NNACL_OK;
}
