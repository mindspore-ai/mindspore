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

#include "nnacl/fp16/arithmetic_fp16.h"
#include <math.h>
#include "nnacl/arithmetic_common.h"

void TileOneDimensionFp16(float16_t *inData, float16_t *outData, int dim, size_t ndim, int *inShape, int *inStrides,
                          int *outStrides, int *multiple) {
  int srcDimSize = inShape[dim];
  if (dim == ndim - 1) {
    for (int i = 0; i < multiple[dim]; i++) {
      memcpy(outData, inData, srcDimSize * sizeof(float16_t));
      outData += srcDimSize;
    }
    return;
  }
  for (size_t i = 0; i < srcDimSize; i++) {
    for (size_t j = 0; j < multiple[dim]; j++) {
      TileOneDimensionFp16(inData + inStrides[dim] * i, outData + outStrides[dim] * (i + j * srcDimSize), dim + 1, ndim,
                           inShape, inStrides, outStrides, multiple);
    }
  }
}

void TileDimensionsFp16(float16_t *data0, float16_t *data1, float16_t *tile_data0, float16_t *tile_data1,
                        ArithmeticParameter *param) {
  CalcMultiplesAndStrides(param);
  TileOneDimensionFp16(data0, tile_data0, 0, param->ndim_, param->in_shape0_, param->in_strides0_, param->out_strides_,
                       param->multiples0_);
  TileOneDimensionFp16(data1, tile_data1, 0, param->ndim_, param->in_shape1_, param->in_strides1_, param->out_strides_,
                       param->multiples1_);
}

int ElementMulFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;

  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = vld1q_f16(input0);
    float16x8_t vin1 = vld1q_f16(input1);
    float16x8_t vout = vmulq_f16(vin0, vin1);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      output[i] = input0[i] * input1[i];
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = input0[index] * input1[index];
  }

  return NNACL_OK;
}
int ElementOptMulFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                      ArithmeticParameter *param) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = {input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0]};
  float16x8_t vin1_opt = {input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0]};
  float16_t in0_opt = input0[0];
  float16_t in1_opt = input1[0];
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = param->in_elements_num0_ == 1 ? vin0_opt : vld1q_f16(input0);
    float16x8_t vin1 = param->in_elements_num1_ == 1 ? vin1_opt : vld1q_f16(input1);
    float16x8_t vout = vmulq_f16(vin0, vin1);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[i];
      float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[i];
      output[i] = in0 * in1;
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[index];
    float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[index];
    output[index] = in0 * in1;
  }

  return NNACL_OK;
}

int ElementMulReluFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;

#ifdef ENABLE_NEON
  float16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
#endif

  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = vld1q_f16(input0);
    float16x8_t vin1 = vld1q_f16(input1);
    float16x8_t vout = vmulq_f16(vin0, vin1);
    vout = vmaxq_f16(vout, zeros);
    vst1q_f16(output, vout);
#else
    float16_t res;
    for (int i = 0; i < C8NUM; ++i) {
      res = input[i] * input1[i];
      output[i] = res > 0 ? res : 0;
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t res = input0[index] * input1[index];
    output[index] = res > 0 ? res : 0;
  }

  return NNACL_OK;
}
int ElementOptMulReluFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                          ArithmeticParameter *param) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = {input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0]};
  float16x8_t vin1_opt = {input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0]};
  float16_t in0_opt = input0[0];
  float16_t in1_opt = input1[0];
  float16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
#endif

  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = param->in_elements_num0_ == 1 ? vin0_opt : vld1q_f16(input0);
    float16x8_t vin1 = param->in_elements_num1_ == 1 ? vin1_opt : vld1q_f16(input1);
    float16x8_t vout = vmulq_f16(vin0, vin1);
    vout = vmaxq_f16(vout, zeros);
    vst1q_f16(output, vout);
#else
    float16_t res;
    for (int i = 0; i < C8NUM; ++i) {
      float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[i];
      float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[i];
      res = in0 * in1;
      output[i] = res > 0 ? res : 0;
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[index];
    float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[index];
    float16_t res = in0 * in1;
    output[index] = res > 0 ? res : 0;
  }

  return NNACL_OK;
}

int ElementMulRelu6Fp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;

#ifdef ENABLE_NEON
  float16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
  float16x8_t bounds = {6, 6, 6, 6, 6, 6, 6, 6};
#endif

  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = vld1q_f16(input0);
    float16x8_t vin1 = vld1q_f16(input1);
    float16x8_t vout = vmulq_f16(vin0, vin1);
    vout = vminq_f16(vmaxq_f16(vout, zeros), bounds);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      output[i] = MSMIN(MSMAX(input0[i] * input1[i], 0), 6);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = MSMIN(MSMAX(input0[index] * input1[index], 0), 6);
  }

  return NNACL_OK;
}
int ElementOptMulRelu6Fp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                           ArithmeticParameter *param) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = {input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0]};
  float16x8_t vin1_opt = {input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0]};
  float16_t in0_opt = input0[0];
  float16_t in1_opt = input1[0];
  float16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
  float16x8_t bounds = {6, 6, 6, 6, 6, 6, 6, 6};
#endif

  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = param->in_elements_num0_ == 1 ? vin0_opt : vld1q_f16(input0);
    float16x8_t vin1 = param->in_elements_num1_ == 1 ? vin1_opt : vld1q_f16(input1);
    float16x8_t vout = vmulq_f16(vin0, vin1);
    vout = vminq_f16(vmaxq_f16(vout, zeros), bounds);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[i];
      float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[i];
      output[i] = MSMIN(MSMAX(in0 * in1, 0), 6);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[index];
    float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[index];
    output[index] = MSMIN(MSMAX(in0 * in1, 0), 6);
  }

  return NNACL_OK;
}

int ElementAddFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;

  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = vld1q_f16(input0);
    float16x8_t vin1 = vld1q_f16(input1);
    float16x8_t vout = vaddq_f16(vin0, vin1);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      output[i] = input0[i] + input1[i];
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = input0[index] + input1[index];
  }
  return NNACL_OK;
}
int ElementOptAddFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                      ArithmeticParameter *param) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = {input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0]};
  float16x8_t vin1_opt = {input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0]};
  float16_t in0_opt = input0[0];
  float16_t in1_opt = input1[0];
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = param->in_elements_num0_ == 1 ? vin0_opt : vld1q_f16(input0);
    float16x8_t vin1 = param->in_elements_num1_ == 1 ? vin1_opt : vld1q_f16(input1);
    float16x8_t vout = vaddq_f16(vin0, vin1);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[i];
      float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[i];
      output[i] = in0 + in1;
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[index];
    float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[index];
    output[index] = in0 + in1;
  }
  return NNACL_OK;
}

int ElementAddReluFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;

#ifdef ENABLE_NEON
  float16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
#endif

  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = vld1q_f16(input0);
    float16x8_t vin1 = vld1q_f16(input1);
    float16x8_t vout = vaddq_f16(vin0, vin1);
    vout = vmaxq_f16(vout, zeros);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      output[i] = MSMAX(input0[i] + input1[i], 0);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t res = input0[index] + input1[index];
    output[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}
int ElementOptAddReluFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                          ArithmeticParameter *param) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = {input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0]};
  float16x8_t vin1_opt = {input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0]};
  float16_t in0_opt = input0[0];
  float16_t in1_opt = input1[0];
  float16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
#endif

  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = param->in_elements_num0_ == 1 ? vin0_opt : vld1q_f16(input0);
    float16x8_t vin1 = param->in_elements_num1_ == 1 ? vin1_opt : vld1q_f16(input1);
    float16x8_t vout = vaddq_f16(vin0, vin1);
    vout = vmaxq_f16(vout, zeros);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[i];
      float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[i];
      output[i] = MSMAX(in0 + in1, 0);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[index];
    float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[index];
    float16_t res = in0 + in1;
    output[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

int ElementAddRelu6Fp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;

#ifdef ENABLE_NEON
  float16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
  float16x8_t bounds = {6, 6, 6, 6, 6, 6, 6, 6};
#endif

  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = vld1q_f16(input0);
    float16x8_t vin1 = vld1q_f16(input1);
    float16x8_t vout = vaddq_f16(vin0, vin1);
    vout = vminq_f16(vmaxq_f16(vout, zeros), bounds);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      output[i] = MSMIN(MSMAX(input0[i] + input1[i], 0), 6);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = MSMIN(MSMAX(input0[index] + input1[index], 0), 6);
  }

  return NNACL_OK;
}
int ElementOptAddRelu6Fp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                           ArithmeticParameter *param) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = {input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0]};
  float16x8_t vin1_opt = {input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0]};
  float16_t in0_opt = input0[0];
  float16_t in1_opt = input1[0];
  float16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
  float16x8_t bounds = {6, 6, 6, 6, 6, 6, 6, 6};
#endif

  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = param->in_elements_num0_ == 1 ? vin0_opt : vld1q_f16(input0);
    float16x8_t vin1 = param->in_elements_num1_ == 1 ? vin1_opt : vld1q_f16(input1);
    float16x8_t vout = vaddq_f16(vin0, vin1);
    vout = vminq_f16(vmaxq_f16(vout, zeros), bounds);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[i];
      float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[i];
      output[i] = MSMIN(MSMAX(in0 + in1, 0), 6);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[index];
    float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[index];
    output[index] = MSMIN(MSMAX(in0 + in1, 0), 6);
  }

  return NNACL_OK;
}

int ElementSubFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;

  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = vld1q_f16(input0);
    float16x8_t vin1 = vld1q_f16(input1);
    float16x8_t vout = vsubq_f16(vin0, vin1);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      output[i] = input0[i] - input1[i];
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = input0[index] - input1[index];
  }
  return NNACL_OK;
}
int ElementOptSubFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                      ArithmeticParameter *param) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = {input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0]};
  float16x8_t vin1_opt = {input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0]};
  float16_t in0_opt = input0[0];
  float16_t in1_opt = input1[0];
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = param->in_elements_num0_ == 1 ? vin0_opt : vld1q_f16(input0);
    float16x8_t vin1 = param->in_elements_num1_ == 1 ? vin1_opt : vld1q_f16(input1);
    float16x8_t vout = vsubq_f16(vin0, vin1);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[i];
      float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[i];
      output[i] = in0 - in1;
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[index];
    float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[index];
    output[index] = in0 - in1;
  }
  return NNACL_OK;
}

int ElementSubReluFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = vld1q_f16(input0);
    float16x8_t vin1 = vld1q_f16(input1);
    float16x8_t vout = vsubq_f16(vin0, vin1);
    vout = vmaxq_f16(vout, zeros);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      output[i] = MSMAX(input0[i] - input1[i], 0);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t res = input0[index] - input1[index];
    output[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}
int ElementOptSubReluFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                          ArithmeticParameter *param) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = {input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0]};
  float16x8_t vin1_opt = {input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0]};
  float16_t in0_opt = input0[0];
  float16_t in1_opt = input1[0];
  float16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = param->in_elements_num0_ == 1 ? vin0_opt : vld1q_f16(input0);
    float16x8_t vin1 = param->in_elements_num1_ == 1 ? vin1_opt : vld1q_f16(input1);
    float16x8_t vout = vsubq_f16(vin0, vin1);
    vout = vmaxq_f16(vout, zeros);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[i];
      float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[i];
      output[i] = MSMAX(in0 - in1, 0);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[index];
    float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[index];
    float16_t res = in0 - in1;
    output[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

int ElementSubRelu6Fp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
  float16x8_t bounds = {6, 6, 6, 6, 6, 6, 6, 6};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = vld1q_f16(input0);
    float16x8_t vin1 = vld1q_f16(input1);
    float16x8_t vout = vsubq_f16(vin0, vin1);
    vout = vminq_f16(vmaxq_f16(vout, zeros), bounds);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      output[i] = MSMIN(MSMAX(input0[i] - input1[i], 0), 6);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = MSMIN(MSMAX(input0[index] - input1[index], 0), 6);
  }

  return NNACL_OK;
}
int ElementOptSubRelu6Fp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                           ArithmeticParameter *param) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = {input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0]};
  float16x8_t vin1_opt = {input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0]};
  float16_t in0_opt = input0[0];
  float16_t in1_opt = input1[0];
  float16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
  float16x8_t bounds = {6, 6, 6, 6, 6, 6, 6, 6};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = param->in_elements_num0_ == 1 ? vin0_opt : vld1q_f16(input0);
    float16x8_t vin1 = param->in_elements_num1_ == 1 ? vin1_opt : vld1q_f16(input1);
    float16x8_t vout = vsubq_f16(vin0, vin1);
    vout = vminq_f16(vmaxq_f16(vout, zeros), bounds);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[i];
      float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[i];
      output[i] = MSMIN(MSMAX(in0 - in1, 0), 6);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[index];
    float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[index];
    output[index] = MSMIN(MSMAX(in0 - in1, 0), 6);
  }

  return NNACL_OK;
}

int ElementDivFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;

  for (int index = 0; index < block_c8; index += C8NUM) {
    for (int i = 0; i < C8NUM; ++i) {
      if (input1[i] == 0) {
        return NNACL_ERRCODE_DIVISOR_ZERO;
      }
    }
#ifdef ENABLE_NEON
    float16x8_t vin0 = vld1q_f16(input0);
    float16x8_t vin1 = vld1q_f16(input1);
    float16x8_t vout = vdivq_f16(vin0, vin1);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      output[i] = input0[i] / input1[i];
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    if (input1[index] == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
    output[index] = input0[index] / input1[index];
  }
  return NNACL_OK;
}
int ElementOptDivFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                      ArithmeticParameter *param) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = {input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0]};
  float16x8_t vin1_opt = {input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0]};
  float16_t in0_opt = input0[0];
  float16_t in1_opt = input1[0];
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
    if (param->in_elements_num1_ == 1) {
      if (in1_opt == 0) {
        return NNACL_ERRCODE_DIVISOR_ZERO;
      }
    } else {
      for (int i = 0; i < C8NUM; ++i) {
        if (input1[i] == 0) {
          return NNACL_ERRCODE_DIVISOR_ZERO;
        }
      }
    }
#ifdef ENABLE_NEON
    float16x8_t vin0 = param->in_elements_num0_ == 1 ? vin0_opt : vld1q_f16(input0);
    float16x8_t vin1 = param->in_elements_num1_ == 1 ? vin1_opt : vld1q_f16(input1);
    float16x8_t vout = vdivq_f16(vin0, vin1);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[i];
      float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[i];
      output[i] = in0 / in1;
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[index];
    float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[index];
    if (in1 == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
    output[index] = in0 / in1;
  }
  return NNACL_OK;
}

int ElementDivReluFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
    for (int i = 0; i < C8NUM; ++i) {
      if (input1[i] == 0) {
        return NNACL_ERRCODE_DIVISOR_ZERO;
      }
    }
#ifdef ENABLE_NEON
    float16x8_t vin0 = vld1q_f16(input0);
    float16x8_t vin1 = vld1q_f16(input1);
    float16x8_t vout = vdivq_f16(vin0, vin1);
    vout = vmaxq_f16(vout, zeros);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      output[i] = MSMAX(input0[i] / input1[i], 0);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    if (input1[index] == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
    float16_t res = input0[index] / input1[index];
    output[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}
int ElementOptDivReluFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                          ArithmeticParameter *param) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;

#ifdef ENABLE_NEON
  float16x8_t vin0_opt = {input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0]};
  float16x8_t vin1_opt = {input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0]};
  float16_t in0_opt = input0[0];
  float16_t in1_opt = input1[0];
  float16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
    if (param->in_elements_num1_ == 1) {
      if (in1_opt == 0) {
        return NNACL_ERRCODE_DIVISOR_ZERO;
      }
    } else {
      for (int i = 0; i < C8NUM; ++i) {
        if (input1[i] == 0) {
          return NNACL_ERRCODE_DIVISOR_ZERO;
        }
      }
    }
#ifdef ENABLE_NEON
    float16x8_t vin0 = param->in_elements_num0_ == 1 ? vin0_opt : vld1q_f16(input0);
    float16x8_t vin1 = param->in_elements_num1_ == 1 ? vin1_opt : vld1q_f16(input1);
    float16x8_t vout = vdivq_f16(vin0, vin1);
    vout = vmaxq_f16(vout, zeros);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[i];
      float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[i];
      output[i] = MSMAX(in0 / in1, 0);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[index];
    float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[index];
    if (in1 == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
    float16_t res = in0 / in1;
    output[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

int ElementDivRelu6Fp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
  float16x8_t bounds = {6, 6, 6, 6, 6, 6, 6, 6};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
    for (int i = 0; i < C8NUM; ++i) {
      if (input1[i] == 0) {
        return NNACL_ERRCODE_DIVISOR_ZERO;
      }
    }
#ifdef ENABLE_NEON
    float16x8_t vin0 = vld1q_f16(input0);
    float16x8_t vin1 = vld1q_f16(input1);
    float16x8_t vout = vdivq_f16(vin0, vin1);
    vout = vminq_f16(vmaxq_f16(vout, zeros), bounds);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      output[i] = MSMIN(MSMAX(input0[i] / input1[i], 0), 6);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    if (input1[index] == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
    output[index] = MSMIN(MSMAX(input0[index] / input1[index], 0), 6);
  }
  return NNACL_OK;
}
int ElementOptDivRelu6Fp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                           ArithmeticParameter *param) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;

#ifdef ENABLE_NEON
  float16x8_t vin0_opt = {input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0]};
  float16x8_t vin1_opt = {input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0]};
  float16_t in0_opt = input0[0];
  float16_t in1_opt = input1[0];
  float16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
  float16x8_t bounds = {6, 6, 6, 6, 6, 6, 6, 6};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
    if (param->in_elements_num1_ == 1) {
      if (in1_opt == 0) {
        return NNACL_ERRCODE_DIVISOR_ZERO;
      }
    } else {
      for (int i = 0; i < C8NUM; ++i) {
        if (input1[i] == 0) {
          return NNACL_ERRCODE_DIVISOR_ZERO;
        }
      }
    }
#ifdef ENABLE_NEON
    float16x8_t vin0 = param->in_elements_num0_ == 1 ? vin0_opt : vld1q_f16(input0);
    float16x8_t vin1 = param->in_elements_num1_ == 1 ? vin1_opt : vld1q_f16(input1);
    float16x8_t vout = vdivq_f16(vin0, vin1);
    vout = vminq_f16(vmaxq_f16(vout, zeros), bounds);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[i];
      float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[i];
      output[i] = MSMIN(MSMAX(in0 / in1, 0), 6);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[index];
    float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[index];
    if (in1 == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
    output[index] = MSMIN(MSMAX(in0 / in1, 0), 6);
  }
  return NNACL_OK;
}

int ElementFloorModFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  for (int i = 0; i < element_size; ++i) {
    if (input1[i] == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
    output[i] = input0[i] - floorf(input0[i] / input1[i]) * input1[i];
  }
  return NNACL_OK;
}
int ElementOptFloorModFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                           ArithmeticParameter *param) {
  if (param->in_elements_num1_ == 1) {
    if (input1[0] == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
    for (int i = 0; i < element_size; ++i) {
      output[i] = input0[i] - floorf(input0[i] / input1[0]) * input1[0];
    }
  } else {
    for (int i = 0; i < element_size; ++i) {
      if (input1[i] == 0) {
        return NNACL_ERRCODE_DIVISOR_ZERO;
      }
      output[i] = input0[i] - floorf(input0[i] / input1[i]) * input1[i];
    }
  }
  return NNACL_OK;
}

int ElementFloorDivFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  for (int i = 0; i < element_size; ++i) {
    if (input1[i] == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
    output[i] = floorf(input0[i] / input1[i]);
  }
  return NNACL_OK;
}
int ElementOptFloorDivFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                           ArithmeticParameter *param) {
  if (param->in_elements_num1_ == 1) {
    if (input1[0] == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
    for (int i = 0; i < element_size; ++i) {
      output[i] = floorf(input0[i] / input1[0]);
    }
  } else {
    for (int i = 0; i < element_size; ++i) {
      if (input1[i] == 0) {
        return NNACL_ERRCODE_DIVISOR_ZERO;
      }
      output[i] = floorf(input0[i] / input1[i]);
    }
  }
  return NNACL_OK;
}

int ElementLogicalAndFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;

#ifdef ENABLE_NEON
  float16x8_t vtrue = {1, 1, 1, 1, 1, 1, 1, 1};
  float16x8_t vfalse = {0, 0, 0, 0, 0, 0, 0, 0};
  uint16x8_t mask = vmovq_n_u16(((uint16_t)(1u << 15) - 1));
  uint16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    uint16x8_t vin0 = vandq_u16(vreinterpretq_s16_f16(vld1q_f16(input0)), mask);
    uint16x8_t vin1 = vandq_u16(vreinterpretq_s16_f16(vld1q_f16(input1)), mask);
    float16x8_t vout = vbslq_f16(vceqq_u16(vandq_u16(vin0, vin1), zeros), vfalse, vtrue);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      output[i] = (float16_t)((bool)(input0[i]) & (bool)(input1[i]));
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = (float16_t)((bool)(input0[index]) & (bool)(input1[index]));
  }
  return NNACL_OK;
}
int ElementOptLogicalAndFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                             ArithmeticParameter *param) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = {input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0]};
  float16x8_t vin1_opt = {input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0]};
  float16_t in0_opt = input0[0];
  float16_t in1_opt = input1[0];
  float16x8_t vtrue = {1, 1, 1, 1, 1, 1, 1, 1};
  float16x8_t vfalse = {0, 0, 0, 0, 0, 0, 0, 0};
  uint16x8_t mask = vmovq_n_u16(((uint16_t)(1u << 15) - 1));
  uint16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0_ = param->in_elements_num0_ == 1 ? vin0_opt : vld1q_f16(input0);
    float16x8_t vin1_ = param->in_elements_num1_ == 1 ? vin1_opt : vld1q_f16(input1);
    uint16x8_t vin0 = vandq_u16(vreinterpretq_s16_f16(vin0_), mask);
    uint16x8_t vin1 = vandq_u16(vreinterpretq_s16_f16(vin1_), mask);
    float16x8_t vout = vbslq_f16(vceqq_u16(vandq_u16(vin0, vin1), zeros), vfalse, vtrue);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[i];
      float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[i];
      output[i] = (float16_t)((bool)(in0) & (bool)(in1));
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[index];
    float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[index];
    output[index] = (float16_t)((bool)(in0) & (bool)(in1));
  }
  return NNACL_OK;
}

int ElementLogicalOrFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;

#ifdef ENABLE_NEON
  float16x8_t vtrue = {1, 1, 1, 1, 1, 1, 1, 1};
  float16x8_t vfalse = {0, 0, 0, 0, 0, 0, 0, 0};
  uint16x8_t mask = vmovq_n_u16(((uint16_t)(1u << 15) - 1));
  uint16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    uint16x8_t vin0 = vandq_u16(vreinterpretq_s16_f16(vld1q_f16(input0)), mask);
    uint16x8_t vin1 = vandq_u16(vreinterpretq_s16_f16(vld1q_f16(input1)), mask);
    float16x8_t vout = vbslq_f16(vceqq_u16(vorrq_u16(vin0, vin1), zeros), vfalse, vtrue);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      output[i] = (float16_t)((bool)(input0[i]) | (bool)(input1[i]));
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = (float16_t)((bool)(input0[index]) | (bool)(input1[index]));
  }
  return NNACL_OK;
}
int ElementOptLogicalOrFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                            ArithmeticParameter *param) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = {input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0]};
  float16x8_t vin1_opt = {input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0]};
  float16_t in0_opt = input0[0];
  float16_t in1_opt = input1[0];
  float16x8_t vtrue = {1, 1, 1, 1, 1, 1, 1, 1};
  float16x8_t vfalse = {0, 0, 0, 0, 0, 0, 0, 0};
  uint16x8_t mask = vmovq_n_u16(((uint16_t)(1u << 15) - 1));
  uint16x8_t zeros = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0_ = param->in_elements_num0_ == 1 ? vin0_opt : vld1q_f16(input0);
    float16x8_t vin1_ = param->in_elements_num1_ == 1 ? vin1_opt : vld1q_f16(input1);
    uint16x8_t vin0 = vandq_u16(vreinterpretq_s16_f16(vin0_), mask);
    uint16x8_t vin1 = vandq_u16(vreinterpretq_s16_f16(vin1_), mask);
    float16x8_t vout = vbslq_f16(vceqq_u16(vorrq_u16(vin0, vin1), zeros), vfalse, vtrue);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[i];
      float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[i];
      output[i] = (float16_t)((bool)(in0) | (bool)(in1));
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[index];
    float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[index];
    output[index] = (float16_t)((bool)(in0) | (bool)(in1));
  }
  return NNACL_OK;
}

int ElementSquaredDifferenceFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  ElementSubFp16(input0, input1, output, element_size);
  return ElementMulFp16(output, output, output, element_size);
}
int ElementOptSquaredDifferenceFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                                    ArithmeticParameter *param) {
  ElementOptSubFp16(input0, input1, output, element_size, param);
  return ElementMulFp16(output, output, output, element_size);
}

int ElementMaximumFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = vld1q_f16(input0);
    float16x8_t vin1 = vld1q_f16(input1);
    float16x8_t vout = vmaxq_f16(vin0, vin1);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      output[i] = MSMAX(input0[i], input1[i]);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = MSMAX(input0[index], input1[index]);
  }
  return NNACL_OK;
}
int ElementOptMaximumFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                          ArithmeticParameter *param) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = {input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0]};
  float16x8_t vin1_opt = {input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0]};
  float16_t in0_opt = input0[0];
  float16_t in1_opt = input1[0];
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = param->in_elements_num0_ == 1 ? vin0_opt : vld1q_f16(input0);
    float16x8_t vin1 = param->in_elements_num1_ == 1 ? vin1_opt : vld1q_f16(input1);
    float16x8_t vout = vmaxq_f16(vin0, vin1);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[i];
      float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[i];
      output[i] = MSMAX(in0, in1);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[index];
    float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[index];
    output[index] = MSMAX(in0, in1);
  }
  return NNACL_OK;
}

int ElementMinimumFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = vld1q_f16(input0);
    float16x8_t vin1 = vld1q_f16(input1);
    float16x8_t vout = vminq_f16(vin0, vin1);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      output[i] = MSMIN(input0[i], input1[i]);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = MSMIN(input0[index], input1[index]);
  }
  return NNACL_OK;
}
int ElementOptMinimumFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                          ArithmeticParameter *param) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = {input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0]};
  float16x8_t vin1_opt = {input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0]};
  float16_t in0_opt = input0[0];
  float16_t in1_opt = input1[0];
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = param->in_elements_num0_ == 1 ? vin0_opt : vld1q_f16(input0);
    float16x8_t vin1 = param->in_elements_num1_ == 1 ? vin1_opt : vld1q_f16(input1);
    float16x8_t vout = vminq_f16(vin0, vin1);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[i];
      float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[i];
      output[i] = MSMIN(in0, in1);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[index];
    float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[index];
    output[index] = MSMIN(in0, in1);
  }
  return NNACL_OK;
}

int ElementNotEqualFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vtrue = {1, 1, 1, 1, 1, 1, 1, 1};
  float16x8_t vfalse = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = vld1q_f16(input0);
    float16x8_t vin1 = vld1q_f16(input1);
    float16x8_t vout = vbslq_f16(vceqq_f16(vin0, vin1), vfalse, vtrue);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      output[i] = (float16_t)(input0[i] != input1[i]);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = (float16_t)(input0[index] != input1[index]);
  }
  return NNACL_OK;
}
int ElementOptNotEqualFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                           ArithmeticParameter *param) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = {input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0]};
  float16x8_t vin1_opt = {input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0]};
  float16_t in0_opt = input0[0];
  float16_t in1_opt = input1[0];
  float16x8_t vtrue = {1, 1, 1, 1, 1, 1, 1, 1};
  float16x8_t vfalse = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = param->in_elements_num0_ == 1 ? vin0_opt : vld1q_f16(input0);
    float16x8_t vin1 = param->in_elements_num1_ == 1 ? vin1_opt : vld1q_f16(input1);
    float16x8_t vout = vbslq_f16(vceqq_f16(vin0, vin1), vfalse, vtrue);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[i];
      float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[i];
      output[i] = (float16_t)(in0 != in1);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[index];
    float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[index];
    output[index] = (float16_t)(in0 != in1);
  }
  return NNACL_OK;
}

int ElementEqualFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vtrue = {1, 1, 1, 1, 1, 1, 1, 1};
  float16x8_t vfalse = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = vld1q_f16(input0);
    float16x8_t vin1 = vld1q_f16(input1);
    float16x8_t vout = vbslq_f16(vceqq_f16(vin0, vin1), vtrue, vfalse);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      output[i] = (float16_t)(input0[i] == input1[i]);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = (float16_t)(input0[index] == input1[index]);
  }
  return NNACL_OK;
}
int ElementOptEqualFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                        ArithmeticParameter *param) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = {input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0]};
  float16x8_t vin1_opt = {input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0]};
  float16_t in0_opt = input0[0];
  float16_t in1_opt = input1[0];
  float16x8_t vtrue = {1, 1, 1, 1, 1, 1, 1, 1};
  float16x8_t vfalse = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = param->in_elements_num0_ == 1 ? vin0_opt : vld1q_f16(input0);
    float16x8_t vin1 = param->in_elements_num1_ == 1 ? vin1_opt : vld1q_f16(input1);
    float16x8_t vout = vbslq_f16(vceqq_f16(vin0, vin1), vtrue, vfalse);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[i];
      float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[i];
      output[i] = (float16_t)(in0 == in1);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[index];
    float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[index];
    output[index] = (float16_t)(in0 == in1);
  }
  return NNACL_OK;
}

int ElementLessFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vtrue = {1, 1, 1, 1, 1, 1, 1, 1};
  float16x8_t vfalse = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = vld1q_f16(input0);
    float16x8_t vin1 = vld1q_f16(input1);
    float16x8_t vout = vbslq_f16(vcltq_f16(vin0, vin1), vtrue, vfalse);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      output[i] = (float16_t)(input0[i] < input1[i]);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = (float16_t)(input0[index] < input1[index]);
  }
  return NNACL_OK;
}
int ElementOptLessFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                       ArithmeticParameter *param) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = {input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0]};
  float16x8_t vin1_opt = {input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0]};
  float16_t in0_opt = input0[0];
  float16_t in1_opt = input1[0];
  float16x8_t vtrue = {1, 1, 1, 1, 1, 1, 1, 1};
  float16x8_t vfalse = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = param->in_elements_num0_ == 1 ? vin0_opt : vld1q_f16(input0);
    float16x8_t vin1 = param->in_elements_num1_ == 1 ? vin1_opt : vld1q_f16(input1);
    float16x8_t vout = vbslq_f16(vcltq_f16(vin0, vin1), vtrue, vfalse);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[i];
      float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[i];
      output[i] = (float16_t)(in0 < in1);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[index];
    float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[index];
    output[index] = (float16_t)(in0 < in1);
  }
  return NNACL_OK;
}

int ElementLessEqualFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vtrue = {1, 1, 1, 1, 1, 1, 1, 1};
  float16x8_t vfalse = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = vld1q_f16(input0);
    float16x8_t vin1 = vld1q_f16(input1);
    float16x8_t vout = vbslq_f16(vcleq_f16(vin0, vin1), vtrue, vfalse);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      output[i] = (float16_t)(input0[i] <= input1[i]);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = (float16_t)(input0[index] <= input1[index]);
  }
  return NNACL_OK;
}
int ElementOptLessEqualFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                            ArithmeticParameter *param) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = {input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0]};
  float16x8_t vin1_opt = {input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0]};
  float16_t in0_opt = input0[0];
  float16_t in1_opt = input1[0];
  float16x8_t vtrue = {1, 1, 1, 1, 1, 1, 1, 1};
  float16x8_t vfalse = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = param->in_elements_num0_ == 1 ? vin0_opt : vld1q_f16(input0);
    float16x8_t vin1 = param->in_elements_num1_ == 1 ? vin1_opt : vld1q_f16(input1);
    float16x8_t vout = vbslq_f16(vcleq_f16(vin0, vin1), vtrue, vfalse);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[i];
      float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[i];
      output[i] = (float16_t)(in0 <= in1);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[index];
    float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[index];
    output[index] = (float16_t)(in0 <= in1);
  }
  return NNACL_OK;
}

int ElementGreaterFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vtrue = {1, 1, 1, 1, 1, 1, 1, 1};
  float16x8_t vfalse = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = vld1q_f16(input0);
    float16x8_t vin1 = vld1q_f16(input1);
    float16x8_t vout = vbslq_f16(vcgtq_f16(vin0, vin1), vtrue, vfalse);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      output[i] = (float16_t)(input0[i] > input1[i]);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = (float16_t)(input0[index] > input1[index]);
  }
  return NNACL_OK;
}
int ElementOptGreaterFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                          ArithmeticParameter *param) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = {input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0]};
  float16x8_t vin1_opt = {input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0]};
  float16_t in0_opt = input0[0];
  float16_t in1_opt = input1[0];
  float16x8_t vtrue = {1, 1, 1, 1, 1, 1, 1, 1};
  float16x8_t vfalse = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = param->in_elements_num0_ == 1 ? vin0_opt : vld1q_f16(input0);
    float16x8_t vin1 = param->in_elements_num1_ == 1 ? vin1_opt : vld1q_f16(input1);
    float16x8_t vout = vbslq_f16(vcgtq_f16(vin0, vin1), vtrue, vfalse);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[i];
      float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[i];
      output[i] = (float16_t)(in0 > in1);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[index];
    float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[index];
    output[index] = (float16_t)(in0 > in1);
  }
  return NNACL_OK;
}

int ElementGreaterEqualFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vtrue = {1, 1, 1, 1, 1, 1, 1, 1};
  float16x8_t vfalse = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = vld1q_f16(input0);
    float16x8_t vin1 = vld1q_f16(input1);
    float16x8_t vout = vbslq_f16(vcgeq_f16(vin0, vin1), vtrue, vfalse);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      output[i] = (float16_t)(input0[i] >= input1[i]);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = (float16_t)(input0[index] >= input1[index]);
  }
  return NNACL_OK;
}
int ElementOptGreaterEqualFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                               ArithmeticParameter *param) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = {input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0], input0[0]};
  float16x8_t vin1_opt = {input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0], input1[0]};
  float16_t in0_opt = input0[0];
  float16_t in1_opt = input1[0];
  float16x8_t vtrue = {1, 1, 1, 1, 1, 1, 1, 1};
  float16x8_t vfalse = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c8; index += C8NUM) {
#ifdef ENABLE_NEON
    float16x8_t vin0 = param->in_elements_num0_ == 1 ? vin0_opt : vld1q_f16(input0);
    float16x8_t vin1 = param->in_elements_num1_ == 1 ? vin1_opt : vld1q_f16(input1);
    float16x8_t vout = vbslq_f16(vcgeq_f16(vin0, vin1), vtrue, vfalse);
    vst1q_f16(output, vout);
#else
    for (int i = 0; i < C8NUM; ++i) {
      float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[i];
      float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[i];
      output[i] = (float16_t)(in0 >= in1);
    }
#endif
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float16_t in0 = param->in_elements_num0_ == 1 ? in0_opt : input0[index];
    float16_t in1 = param->in_elements_num1_ == 1 ? in1_opt : input1[index];
    output[index] = (float16_t)(in0 >= in1);
  }
  return NNACL_OK;
}
