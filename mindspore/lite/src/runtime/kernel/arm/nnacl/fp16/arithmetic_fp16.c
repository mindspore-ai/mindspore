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

int ElementOptMulFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                      ArithmeticParameter *param) {
  if (param->in_elements_num0_ == 1) {
    for (int i = 0; i < element_size; ++i) {
      output[i] = input0[0] * input1[i];
    }
  } else if (param->in_elements_num1_ == 1) {
    for (int i = 0; i < element_size; ++i) {
      output[i] = input0[i] * input1[0];
    }
  } else {
    for (int i = 0; i < element_size; ++i) {
      output[i] = input0[i] * input1[i];
    }
  }
  return NNACL_OK;
}

int ElementOptSubFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                      ArithmeticParameter *param) {
  if (param->in_elements_num0_ == 1) {
    for (int i = 0; i < element_size; ++i) {
      output[i] = input0[0] - input1[i];
    }
  } else if (param->in_elements_num1_ == 1) {
    for (int i = 0; i < element_size; ++i) {
      output[i] = input0[i] - input1[0];
    }
  } else {
    for (int i = 0; i < element_size; ++i) {
      output[i] = input0[i] - input1[i];
    }
  }
  return NNACL_OK;
}

int ElementOptAddFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                      ArithmeticParameter *param) {
  if (param->in_elements_num0_ == 1) {
    for (int i = 0; i < element_size; ++i) {
      output[i] = input0[0] + input1[i];
    }
  } else if (param->in_elements_num1_ == 1) {
    for (int i = 0; i < element_size; ++i) {
      output[i] = input0[i] + input1[0];
    }
  } else {
    for (int i = 0; i < element_size; ++i) {
      output[i] = input0[i] + input1[i];
    }
  }
  return NNACL_OK;
}

int ElementMulFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;

  for (int index = 0; index < block_c8; index += C8NUM) {
    output[0] = input0[0] * input1[0];
    output[1] = input0[1] * input1[1];
    output[2] = input0[2] * input1[2];
    output[3] = input0[3] * input1[3];
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = input0[index] * input1[index];
  }

  return NNACL_OK;
}

int ElementMulReluFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;

  for (int index = 0; index < block_c8; index += C8NUM) {
    float16_t res = input0[0] * input1[0];
    output[0] = res > 0 ? res : 0;
    res = input0[1] * input1[1];
    output[1] = res > 0 ? res : 0;
    res = input0[2] * input1[2];
    output[2] = res > 0 ? res : 0;
    res = input0[3] * input1[3];
    output[3] = res > 0 ? res : 0;
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

int ElementMulRelu6Fp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;

  for (int index = 0; index < block_c8; index += C8NUM) {
    output[0] = MSMIN(MSMAX(input0[0] * input1[0], 0), 6);
    output[1] = MSMIN(MSMAX(input0[1] * input1[1], 0), 6);
    output[2] = MSMIN(MSMAX(input0[2] * input1[2], 0), 6);
    output[3] = MSMIN(MSMAX(input0[3] * input1[3], 0), 6);
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = MSMIN(MSMAX(input0[index] * input1[index], 0), 6);
  }

  return NNACL_OK;
}

int ElementAddFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;

  for (int index = 0; index < block_c8; index += C8NUM) {
    output[0] = input0[0] + input1[0];
    output[1] = input0[1] + input1[1];
    output[2] = input0[2] + input1[2];
    output[3] = input0[3] + input1[3];
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = input0[index] + input1[index];
  }
  return NNACL_OK;
}

int ElementAddReluFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;

  for (int index = 0; index < block_c8; index += C8NUM) {
    float16_t res = input0[0] + input1[0];
    output[0] = res > 0 ? res : 0;
    res = input0[1] + input1[1];
    output[1] = res > 0 ? res : 0;
    res = input0[2] + input1[2];
    output[2] = res > 0 ? res : 0;
    res = input0[3] + input1[3];
    output[3] = res > 0 ? res : 0;
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

int ElementAddRelu6Fp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;

  for (int index = 0; index < block_c8; index += C8NUM) {
    output[0] = MSMIN(MSMAX(input0[0] + input1[0], 0), 6);
    output[1] = MSMIN(MSMAX(input0[1] + input1[1], 0), 6);
    output[2] = MSMIN(MSMAX(input0[2] + input1[2], 0), 6);
    output[3] = MSMIN(MSMAX(input0[3] + input1[3], 0), 6);
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = MSMIN(MSMAX(input0[index] + input1[index], 0), 6);
  }

  return NNACL_OK;
}

int ElementSubFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;

  for (int index = 0; index < block_c8; index += C8NUM) {
    output[0] = input0[0] - input1[0];
    output[1] = input0[1] - input1[1];
    output[2] = input0[2] - input1[2];
    output[3] = input0[3] - input1[3];
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = input0[index] - input1[index];
  }
  return NNACL_OK;
}

int ElementSubReluFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;

  for (int index = 0; index < block_c8; index += C8NUM) {
    float16_t res = input0[0] - input1[0];
    output[0] = res > 0 ? res : 0;
    res = input0[1] - input1[1];
    output[1] = res > 0 ? res : 0;
    res = input0[2] - input1[2];
    output[2] = res > 0 ? res : 0;
    res = input0[3] - input1[3];
    output[3] = res > 0 ? res : 0;
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

int ElementSubRelu6Fp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size) {
  int block_mod = element_size % C8NUM;
  int block_c8 = element_size - block_mod;

  for (int index = 0; index < block_c8; index += C8NUM) {
    output[0] = MSMIN(MSMAX(input0[0] - input1[0], 0), 6);
    output[1] = MSMIN(MSMAX(input0[1] - input1[1], 0), 6);
    output[2] = MSMIN(MSMAX(input0[2] - input1[2], 0), 6);
    output[3] = MSMIN(MSMAX(input0[3] - input1[3], 0), 6);
    input0 += C8NUM;
    input1 += C8NUM;
    output += C8NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = MSMIN(MSMAX(input0[index] - input1[index], 0), 6);
  }

  return NNACL_OK;
}
