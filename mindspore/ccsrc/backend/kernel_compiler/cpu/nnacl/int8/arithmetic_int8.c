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

#include "nnacl/int8/arithmetic_int8.h"
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/errorcode.h"

void TileOneDimensionInt8(const int8_t *inData, int8_t *outData, int dim, size_t ndim, const int *inShape,
                          const int *inStrides, const int *outStrides, const int *multiple) {
  int srcDimSize = inShape[dim];
  if (dim == ndim - 1) {
    for (int i = 0; i < multiple[dim]; i++) {
      memcpy(outData, inData, srcDimSize * sizeof(int8_t));
      outData += srcDimSize;
    }
    return;
  }
  for (size_t i = 0; i < srcDimSize; i++) {
    for (size_t j = 0; j < multiple[dim]; j++) {
      TileOneDimensionInt8(inData + inStrides[dim] * i, outData + outStrides[dim] * (i + j * srcDimSize), dim + 1, ndim,
                           inShape, inStrides, outStrides, multiple);
    }
  }
}

void TileDimensionsInt8(const int8_t *data0, const int8_t *data1, int8_t *tile_data0, int8_t *tile_data1,
                        ArithmeticParameter *param) {
  CalcMultiplesAndStrides(param);
  TileOneDimensionInt8(data0, tile_data0, 0, param->ndim_, param->in_shape0_, param->in_strides0_, param->out_strides_,
                       param->multiples0_);
  TileOneDimensionInt8(data1, tile_data1, 0, param->ndim_, param->in_shape1_, param->in_strides1_, param->out_strides_,
                       param->multiples1_);
}

#define ACCURACY_DATA 0.00000001

int ElementNotEqualInt8(int8_t *input0, int8_t *input1, uint8_t *output, int element_size,
                        ArithmeticQuantArg *quant_arg) {
  float in0_bias = -quant_arg->in0_args_.zp_ * quant_arg->in0_args_.scale_;
  float in1_bias = -quant_arg->in1_args_.zp_ * quant_arg->in1_args_.scale_;

  for (int index = 0; index < element_size; ++index) {
    float in0_real = input0[index] * quant_arg->in0_args_.scale_ + in0_bias;
    float in1_real = input1[index] * quant_arg->in1_args_.scale_ + in1_bias;
    float minus_inputs = in0_real - in1_real;
    bool out_real = true;
    if (minus_inputs >= -ACCURACY_DATA && minus_inputs <= ACCURACY_DATA) {
      out_real = false;
    }
    output[index] = (uint8_t)out_real;
  }
  return NNACL_OK;
}

int ElementEqualInt8(int8_t *input0, int8_t *input1, uint8_t *output, int element_size, ArithmeticQuantArg *quant_arg) {
  float in0_bias = -quant_arg->in0_args_.zp_ * quant_arg->in0_args_.scale_;
  float in1_bias = -quant_arg->in1_args_.zp_ * quant_arg->in1_args_.scale_;
  for (int index = 0; index < element_size; ++index) {
    float in0_real = input0[index] * quant_arg->in0_args_.scale_ + in0_bias;
    float in1_real = input1[index] * quant_arg->in1_args_.scale_ + in1_bias;
    float minus_inputs = in0_real - in1_real;
    bool out_real = false;
    if (minus_inputs >= -ACCURACY_DATA && minus_inputs <= ACCURACY_DATA) {
      out_real = true;
    }
    output[index] = (uint8_t)out_real;
  }
  return NNACL_OK;
}

int ElementLessInt8(int8_t *input0, int8_t *input1, uint8_t *output, int element_size, ArithmeticQuantArg *quant_arg) {
  float in0_bias = -quant_arg->in0_args_.zp_ * quant_arg->in0_args_.scale_;
  float in1_bias = -quant_arg->in1_args_.zp_ * quant_arg->in1_args_.scale_;
  for (int index = 0; index < element_size; ++index) {
    float in0_real = input0[index] * quant_arg->in0_args_.scale_ + in0_bias;
    float in1_real = input1[index] * quant_arg->in1_args_.scale_ + in1_bias;
    bool out_real = in0_real < in1_real;
    output[index] = (uint8_t)out_real;
  }
  return NNACL_OK;
}

int ElementLessEqualInt8(int8_t *input0, int8_t *input1, uint8_t *output, int element_size,
                         ArithmeticQuantArg *quant_arg) {
  float in0_bias = -quant_arg->in0_args_.zp_ * quant_arg->in0_args_.scale_;
  float in1_bias = -quant_arg->in1_args_.zp_ * quant_arg->in1_args_.scale_;
  for (int index = 0; index < element_size; ++index) {
    float in0_real = input0[index] * quant_arg->in0_args_.scale_ + in0_bias;
    float in1_real = input1[index] * quant_arg->in1_args_.scale_ + in1_bias;
    bool out_real = in0_real <= in1_real;
    output[index] = (uint8_t)out_real;
  }
  return NNACL_OK;
}

int ElementGreaterInt8(int8_t *input0, int8_t *input1, uint8_t *output, int element_size,
                       ArithmeticQuantArg *quant_arg) {
  float in0_bias = -quant_arg->in0_args_.zp_ * quant_arg->in0_args_.scale_;
  float in1_bias = -quant_arg->in1_args_.zp_ * quant_arg->in1_args_.scale_;
  for (int index = 0; index < element_size; ++index) {
    float in0_real = input0[index] * quant_arg->in0_args_.scale_ + in0_bias;
    float in1_real = input1[index] * quant_arg->in1_args_.scale_ + in1_bias;
    bool out_real = in0_real > in1_real;
    output[index] = (uint8_t)out_real;
  }
  return NNACL_OK;
}

int ElementGreaterEqualInt8(int8_t *input0, int8_t *input1, uint8_t *output, int element_size,
                            ArithmeticQuantArg *quant_arg) {
  float in0_bias = -quant_arg->in0_args_.zp_ * quant_arg->in0_args_.scale_;
  float in1_bias = -quant_arg->in1_args_.zp_ * quant_arg->in1_args_.scale_;
  for (int index = 0; index < element_size; ++index) {
    float in0_real = input0[index] * quant_arg->in0_args_.scale_ + in0_bias;
    float in1_real = input1[index] * quant_arg->in1_args_.scale_ + in1_bias;
    bool out_real = in0_real >= in1_real;
    output[index] = (uint8_t)out_real;
  }
  return NNACL_OK;
}

#undef ACCURACY_DATA
