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

#include "nnacl/fp32/div_fp32.h"
#include <math.h>
#include "nnacl/fp32/arithmetic_fp32.h"
#include "nnacl/div_fp32_simd.h"

int ElementOptDiv(const float *in0, const float *in1, float *out, int size, bool first_scalar) {
  int index = 0;

  if (first_scalar) {
    SIMD_RUN_NO_SCALAR(ElementOptDivNum0, index, in0, in1, out, size);

    for (; index < size; index++) {
      out[index] = in0[0] / in1[index];
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptDivNum1, index, in0, in1, out, size);
    for (; index < size; index++) {
      out[index] = in0[index] / in1[0];
    }
  }
  return NNACL_OK;
}

int ElementOptDivRelu(const float *in0, const float *in1, float *out, int size, bool first_scalar) {
  int index = 0;
  if (first_scalar) {
    SIMD_RUN_NO_SCALAR(ElementOptDivReluNum0, index, in0, in1, out, size);

    for (; index < size; index++) {
      out[index] = in0[0] / in1[index];
      out[index] = out[index] > 0 ? out[index] : 0;
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptDivReluNum1, index, in0, in1, out, size);

    for (; index < size; index++) {
      out[index] = in0[index] / in1[0];
      out[index] = out[index] > 0 ? out[index] : 0;
    }
  }
  return NNACL_OK;
}

int ElementOptDivRelu6(const float *in0, const float *in1, float *out, int size, bool first_scalar) {
  int index = 0;

  if (first_scalar) {
    SIMD_RUN_NO_SCALAR(ElementOptDivRelu6Num0, index, in0, in1, out, size);

    for (; index < size; index++) {
      out[index] = MSMIN(MSMAX(in0[0] / in1[index], RELU6_MIN_VAL), RELU6_MAX_VAL);
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptDivRelu6Num1, index, in0, in1, out, size);

    for (; index < size; index++) {
      out[index] = MSMIN(MSMAX(in0[index] / in1[0], RELU6_MIN_VAL), RELU6_MAX_VAL);
    }
  }
  return NNACL_OK;
}

int ElementOptDivInt(const int32_t *in0, const int32_t *in1, int32_t *out, int size, bool first_scalar) {
  int index = 0;

  if (first_scalar) {
    SIMD_RUN_NO_SCALAR(ElementOptDivIntNum0, index, in0, in1, out, size);

    for (; index < size; index++) {
      NNACL_CHECK_ZERO_RETURN_ERR(in1[index] != 0);
      out[index] = in0[0] / in1[index];
    }
  } else {
    NNACL_CHECK_ZERO_RETURN_ERR(in1[0] != 0);

    SIMD_RUN_NO_SCALAR(ElementOptDivIntNum1, index, in0, in1, out, size);

    for (; index < size; index++) {
      out[index] = in0[index] / in1[0];
    }
  }
  return NNACL_OK;
}

int BroadcastDiv(const float *in0, const float *in1, float *tile_in0, float *tile_in1, float *out, int size,
                 ArithmeticParameter *param) {
  TileDimensionsFp32(in0, in1, tile_in0, tile_in1, param);
  return ElementDiv(tile_in0, tile_in1, out, size);
}

int ElementDiv(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementDiv, index, in0, in1, out, size);
  for (; index < size; index++) {
    out[index] = in0[index] / in1[index];
  }
  return NNACL_OK;
}

int ElementDivRelu(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementDivRelu, index, in0, in1, out, size);
  for (; index < size; index++) {
    float res = in0[index] / in1[index];
    out[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

int ElementDivRelu6(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementDivRelu6, index, in0, in1, out, size);
  for (; index < size; index++) {
    out[index] = MSMIN(MSMAX(in0[index] / in1[index], RELU6_MIN_VAL), RELU6_MAX_VAL);
  }
  return NNACL_OK;
}
