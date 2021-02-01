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

#include "nnacl/fp32/div_fp32.h"
#include <math.h>
#include "nnacl/fp32/arithmetic_fp32.h"

int ElementOptDiv(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param) {
  if (param->in_elements_num0_ == 1) {
    for (int index = 0; index < size; index++) {
      out[index] = in0[0] / in1[index];
    }
  } else {
    if (in1[0] == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
    for (int index = 0; index < size; index++) {
      out[index] = in0[index] / in1[0];
    }
  }
  return NNACL_OK;
}

int ElementOptDivRelu(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param) {
  if (param->in_elements_num0_ == 1) {
    for (int index = 0; index < size; index++) {
      out[index] = in0[0] / in1[index];
      out[index] = out[index] > 0 ? out[index] : 0;
    }
  } else {
    for (int index = 0; index < size; index++) {
      out[index] = in0[index] / in1[0];
      out[index] = out[index] > 0 ? out[index] : 0;
    }
  }
  return NNACL_OK;
}

int ElementOptDivRelu6(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param) {
  if (param->in_elements_num0_ == 1) {
    for (int index = 0; index < size; index++) {
      out[index] = MSMIN(MSMAX(in0[0] / in1[index], 0), 6);
    }
  } else {
    for (int index = 0; index < size; index++) {
      out[index] = MSMIN(MSMAX(in0[index] / in1[0], 0), 6);
    }
  }
  return NNACL_OK;
}

int ElementOptDivInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param) {
  if (param->in_elements_num0_ == 1) {
    for (int index = 0; index < size; index++) {
      out[index] = in0[0] / in1[index];
    }
  } else {
    if (in1[0] == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
    for (int index = 0; index < size; index++) {
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
  for (int i = 0; i < size; i++) {
    out[i] = in0[i] / in1[i];
  }
  return NNACL_OK;
}

int ElementDivRelu(const float *in0, const float *in1, float *out, int size) {
  for (int i = 0; i < size; i++) {
    float res = in0[i] / in1[i];
    out[i] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

int ElementDivRelu6(const float *in0, const float *in1, float *out, int size) {
  for (int i = 0; i < size; i++) {
    out[i] = MSMIN(MSMAX(in0[i] / in1[i], 0), 6);
  }
  return NNACL_OK;
}
