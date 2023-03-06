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

#include "nnacl/fp32/arithmetic_fp32.h"
#include <math.h>
#include "nnacl/arithmetic_fp32_simd.h"

#define ACCURACY_DATA 0.00000001

int ElementFloorMod(const float *in0, const float *in1, float *out, int size) {
  int i = 0;

  SIMD_RUN_X86_NO_SCALAR(ElementFloorMod, i, in0, in1, out, size);  // neon no floor instruction

  for (; i < size; i++) {
    out[i] = in0[i] - floorf(in0[i] / in1[i]) * in1[i];
  }
  return NNACL_OK;
}

int ElementOptFloorMod(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param) {
  int i = 0;

  if (param->in_elements_num0_ == 1) {
    SIMD_RUN_X86_NO_SCALAR(ElementOptFloorModNum0, i, in0, in1, out, size);  // neon no floor instruction
    for (; i < size; i++) {
      out[i] = in0[0] - floorf(in0[0] / in1[i]) * in1[i];
    }
  } else {
    SIMD_RUN_X86_NO_SCALAR(ElementOptFloorModNum1, i, in0, in1, out, size);  // neon no floor instruction
    for (; i < size; i++) {
      out[i] = in0[i] - floorf(in0[i] / in1[0]) * in1[0];
    }
  }

  return NNACL_OK;
}

int ElementFloorModInt(const int *in0, const int *in1, int *out, int size) {
  for (int i = 0; i < size; i++) {
    NNACL_CHECK_ZERO_RETURN_ERR(in1[i]);
    int remainder = in0[i] - (in0[i] / in1[i]) * in1[i];
    out[i] = (remainder != 0) && ((in0[i] > 0) != (in1[i] > 0)) ? remainder + in1[i] : remainder;
  }
  return NNACL_OK;
}

int ElementOptFloorModInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param) {
  int i = 0;
  if (param->in_elements_num0_ == 1) {
    for (; i < size; i++) {
      NNACL_CHECK_ZERO_RETURN_ERR(in1[i]);
      int remainder = in0[0] - (in0[0] / in1[i]) * in1[i];
      out[i] = (remainder != 0) && ((in0[0] > 0) != (in1[i] > 0)) ? remainder + in1[i] : remainder;
    }
  } else {
    NNACL_CHECK_ZERO_RETURN_ERR(in1[0]);
    for (; i < size; i++) {
      int remainder = in0[i] - (in0[i] / in1[0]) * in1[0];
      out[i] = (remainder != 0) && ((in0[i] > 0) != (in1[0] > 0)) ? remainder + in1[0] : remainder;
    }
  }

  return NNACL_OK;
}

int ElementMod(const float *in0, const float *in1, float *out, int size) {
  for (int i = 0; i < size; i++) {
    out[i] = fmodf(in0[i], in1[i]);
  }
  return NNACL_OK;
}

int ElementOptMod(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param) {
  int index = 0;
  if (param->in_elements_num0_ == 1) {
    for (; index < size; index++) {
      out[index] = fmodf(in0[0], in1[index]);
    }
  } else {
    for (; index < size; index++) {
      out[index] = fmodf(in0[index], in1[0]);
    }
  }
  return NNACL_OK;
}

int ElementModInt(const int *in0, const int *in1, int *out, int size) {
  for (int i = 0; i < size; i++) {
    NNACL_CHECK_ZERO_RETURN_ERR(in1[i]);
    out[i] = in0[i] % in1[i];
  }
  return NNACL_OK;
}

int ElementOptModInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param) {
  if (param->in_elements_num0_ == 1) {
    for (int index = 0; index < size; index++) {
      NNACL_CHECK_ZERO_RETURN_ERR(in1[index]);
      out[index] = in0[0] % in1[index];
    }
  } else {
    NNACL_CHECK_ZERO_RETURN_ERR(in1[0]);
    for (int index = 0; index < size; index++) {
      out[index] = in0[index] % in1[0];
    }
  }
  return NNACL_OK;
}

int ElementFloorDiv(const float *in0, const float *in1, float *out, int size) {
  int i = 0;

  SIMD_RUN_X86_NO_SCALAR(ElementFloorDiv, i, in0, in1, out, size);  // neon no floor instruction

  for (; i < size; i++) {
    out[i] = floorf(in0[i] / in1[i]);
  }
  return NNACL_OK;
}

int ElementOptFloorDiv(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param) {
  int i = 0;

  if (param->in_elements_num0_ == 1) {
    SIMD_RUN_X86_NO_SCALAR(ElementOptFloorDivNum0, i, in0, in1, out, size);  // neon no floor instruction

    for (; i < size; i++) {
      out[i] = floorf(in0[0] / in1[i]);
    }
  } else {
    SIMD_RUN_X86_NO_SCALAR(ElementOptFloorDivNum1, i, in0, in1, out, size);  // neon no floor instruction

    for (; i < size; i++) {
      out[i] = floorf(in0[i] / in1[0]);
    }
  }

  return NNACL_OK;
}

int ElementFloorDivInt(const int *in0, const int *in1, int *out, int size) {
  int i = 0;

  SIMD_RUN_NO_SCALAR(ElementFloorDivInt, i, in0, in1, out, size);

  for (; i < size; i++) {
    NNACL_CHECK_ZERO_RETURN_ERR(in1[i]);
    out[i] = in0[i] / in1[i];
  }
  return NNACL_OK;
}

int ElementOptFloorDivInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param) {
  int i = 0;
  if (param->in_elements_num0_ == 1) {
    SIMD_RUN_NO_SCALAR(ElementOptFloorDivIntNum0, i, in0, in1, out, size);

    for (; i < size; i++) {
      NNACL_CHECK_ZERO_RETURN_ERR(in1[i]);
      out[i] = in0[0] / in1[i];
    }
  } else {
    NNACL_CHECK_ZERO_RETURN_ERR(in1[0]);

    SIMD_RUN_NO_SCALAR(ElementOptFloorDivIntNum1, i, in0, in1, out, size);

    for (; i < size; i++) {
      out[i] = in0[i] / in1[0];
    }
  }

  return NNACL_OK;
}

int ElementLogicalAnd(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementLogicalAnd, index, in0, in1, out, size);
  for (; index < size; index++) {
    out[index] = (float)((bool)(in0[index]) & (bool)(in1[index]));
  }
  return NNACL_OK;
}

int ElementOptLogicalAnd(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param) {
  int index = 0;
  SIMD_RUN_NO_SCALAR(ElementOptLogicalAnd, index, in0, in1, out, size, param);
  if (param->in_elements_num0_ == 1) {
    for (; index < size; index++) {
      out[index] = (float)((bool)(in0[0]) & (bool)(in1[index]));
    }
  } else {
    for (; index < size; index++) {
      out[index] = (float)((bool)(in0[index]) & (bool)(in1[0]));
    }
  }

  return NNACL_OK;
}

int ElementLogicalAndInt(const int *in0, const int *in1, int *out, int size) {
  int index = 0;
  for (; index < size; index++) {
    out[index] = (int)((unsigned int)(in0[index]) & (unsigned int)(in1[index]));
  }
  return NNACL_OK;
}

int ElementOptLogicalAndInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param) {
  int index = 0;
  if (param->in_elements_num0_ == 1) {
    for (; index < size; index++) {
      out[index] = (int)((unsigned int)(in0[0]) & (unsigned int)(in1[index]));
    }
  } else {
    for (; index < size; index++) {
      out[index] = (int)((unsigned int)(in0[index]) & (unsigned int)(in1[0]));
    }
  }

  return NNACL_OK;
}

int ElementLogicalAndBool(const bool *in0, const bool *in1, bool *out, int size) {
  int index = 0;
  for (; index < size; index++) {
    out[index] = (bool)((unsigned int)(in0[index]) & (unsigned int)(in1[index]));
  }

  return NNACL_OK;
}

int ElementOptLogicalAndBool(const bool *in0, const bool *in1, bool *out, int size, const ArithmeticParameter *param) {
  int index = 0;
  if (param->in_elements_num0_ == 1) {
    for (; index < size; index++) {
      out[index] = (bool)((unsigned int)(in0[0]) & (unsigned int)(in1[index]));
    }
  } else {
    for (; index < size; index++) {
      out[index] = (bool)((unsigned int)(in0[index]) & (unsigned int)(in1[0]));
    }
  }

  return NNACL_OK;
}

int ElementLogicalOr(const float *in0, const float *in1, float *out, int size) {
  int index = 0;
#ifdef ENABLE_NEON
  float32x4_t vtrue = vdupq_n_f32(1);
  float32x4_t vfalse = vdupq_n_f32(0);
  uint32x4_t mask = vmovq_n_u32(((uint32_t)(1u << 31) - 1));
  uint32x4_t zeros = vdupq_n_u32(0);
  for (; index <= size - 4; index += C4NUM) {
    uint32x4_t vin0 = vandq_u32(vreinterpretq_u32_f32(vld1q_f32(in0 + index)), mask);
    uint32x4_t vin1 = vandq_u32(vreinterpretq_u32_f32(vld1q_f32(in1 + index)), mask);
    float32x4_t vout = vbslq_f32(vceqq_u32(vorrq_u32(vin0, vin1), zeros), vfalse, vtrue);
    vst1q_f32(out + index, vout);
  }
#endif
  for (; index < size; index++) {
    out[index] = (float)((bool)(in0[index]) | (bool)(in1[index]));
  }
  return NNACL_OK;
}

int ElementOptLogicalOr(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param) {
  int index = 0;
  if (param->in_elements_num0_ == 1) {
    for (; index < size; index++) {
      out[index] = (float)((bool)(in0[0]) | (bool)(in1[index]));
    }
  } else {
    for (; index < size; index++) {
      out[index] = (float)((bool)(in0[index]) | (bool)(in1[0]));
    }
  }

  return NNACL_OK;
}

int ElementLogicalOrBool(const bool *in0, const bool *in1, bool *out, int size) {
  int index = 0;
  for (; index < size; index++) {
    out[index] = (bool)(in0[index] | in1[index]);
  }
  return NNACL_OK;
}

int ElementOptLogicalOrBool(const bool *in0, const bool *in1, bool *out, int size, const ArithmeticParameter *param) {
  int index = 0;
  if (param->in_elements_num0_ == 1) {
    for (; index < size; index++) {
      out[index] = (bool)(in0[0] | in1[index]);
    }
  } else {
    for (; index < size; index++) {
      out[index] = (bool)(in0[index] | in1[0]);
    }
  }

  return NNACL_OK;
}

int ElementMaximum(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementMaximum, index, in0, in1, out, size);

  for (; index < size; index++) {
    out[index] = in0[index] > in1[index] ? in0[index] : in1[index];
  }
  return NNACL_OK;
}

int ElementOptMaximum(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param) {
  int index = 0;

  if (param->in_elements_num0_ == 1) {
    SIMD_RUN_NO_SCALAR(ElementOptMaximumNum0, index, in0, in1, out, size);

    for (; index < size; index++) {
      out[index] = in0[0] > in1[index] ? in0[0] : in1[index];
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptMaximumNum1, index, in0, in1, out, size);

    for (; index < size; index++) {
      out[index] = in0[index] > in1[0] ? in0[index] : in1[0];
    }
  }

  return NNACL_OK;
}

int ElementMaximumInt(const int *in0, const int *in1, int *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementMaximumInt, index, in0, in1, out, size);

  for (; index < size; index++) {
    out[index] = in0[index] > in1[index] ? in0[index] : in1[index];
  }
  return NNACL_OK;
}

int ElementOptMaximumInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param) {
  int index = 0;
  if (param->in_elements_num0_ == 1) {
    SIMD_RUN_NO_SCALAR(ElementOptMaximumIntNum0, index, in0, in1, out, size);

    for (; index < size; index++) {
      out[index] = in0[0] > in1[index] ? in0[0] : in1[index];
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptMaximumIntNum1, index, in0, in1, out, size);

    for (; index < size; index++) {
      out[index] = in0[index] > in1[0] ? in0[index] : in1[0];
    }
  }

  return NNACL_OK;
}

int ElementMinimumInt(const int *input0, const int *input1, int *output, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementMinimumInt, index, input0, input1, output, size);

  for (; index < size; index++) {
    output[index] = input0[index] > input1[index] ? input1[index] : input0[index];
  }
  return NNACL_OK;
}

int ElementOptMinimumInt(const int *input0, const int *input1, int *output, int size,
                         const ArithmeticParameter *param) {
  int index = 0;
  if (param->in_elements_num0_ == 1) {
    SIMD_RUN_NO_SCALAR(ElementOptMinimumIntNum0, index, input0, input1, output, size);

    for (; index < size; index++) {
      output[index] = input0[0] > input1[index] ? input1[index] : input0[0];
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptMinimumIntNum1, index, input0, input1, output, size);

    for (; index < size; index++) {
      output[index] = input0[index] > input1[0] ? input1[0] : input0[index];
    }
  }

  return NNACL_OK;
}

int ElementMinimum(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(ElementMinimum, index, in0, in1, out, size);

  for (; index < size; index++) {
    out[index] = in0[index] > in1[index] ? in1[index] : in0[index];
  }
  return NNACL_OK;
}

int ElementOptMinimum(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param) {
  int index = 0;
  if (param->in_elements_num0_ == 1) {
    SIMD_RUN_NO_SCALAR(ElementOptMinimumNum0, index, in0, in1, out, size);

    for (; index < size; index++) {
      out[index] = in0[0] > in1[index] ? in1[index] : in0[0];
    }
  } else {
    SIMD_RUN_NO_SCALAR(ElementOptMinimumNum1, index, in0, in1, out, size);

    for (; index < size; index++) {
      out[index] = in0[index] > in1[0] ? in1[0] : in0[index];
    }
  }

  return NNACL_OK;
}

#undef ACCURACY_DATA

void TileOneDimensionFp32(const float *inData, float *outData, int dim, size_t ndim, const int *inShape,
                          const int *inStrides, const int *outStrides, const int *multiple) {
  int srcDimSize = inShape[dim];
  if (dim == ndim - 1) {
    for (int i = 0; i < multiple[dim]; i++) {
      memcpy(outData, inData, srcDimSize * sizeof(float));
      outData += srcDimSize;
    }
    return;
  }
  for (size_t i = 0; i < srcDimSize; i++) {
    for (size_t j = 0; j < multiple[dim]; j++) {
      TileOneDimensionFp32(inData + inStrides[dim] * i, outData + outStrides[dim] * (i + j * srcDimSize), dim + 1, ndim,
                           inShape, inStrides, outStrides, multiple);
    }
  }
}

void TileDimensionsFp32(const float *data0, const float *data1, float *tile_data0, float *tile_data1,
                        ArithmeticParameter *param) {
  CalcMultiplesAndStrides(param);
  TileOneDimensionFp32(data0, tile_data0, 0, param->ndim_, param->in_shape0_, param->in_strides0_, param->out_strides_,
                       param->multiples0_);
  TileOneDimensionFp32(data1, tile_data1, 0, param->ndim_, param->in_shape1_, param->in_strides1_, param->out_strides_,
                       param->multiples1_);
}

void AssignSubOpt(float *in0, const float *in1, size_t size) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(AssignSubOpt, index, in0, in1, size);

  for (; index < size; index++) {
    in0[index] = in0[index] - in1[index];
  }
  return;
}
