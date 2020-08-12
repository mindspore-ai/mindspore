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

#define ACCURACY_DATA 0.00000001

int ElementNotEqualInt8(int8_t *input0, int8_t *input1, int8_t *output, int element_size,
                        ArithmeticQuantArg *quant_arg) {
  float in0_bias = -quant_arg->in0_args_.zp_ * quant_arg->in0_args_.scale_;
  float in1_bias = -quant_arg->in1_args_.zp_ * quant_arg->in1_args_.scale_;
  float output_inverse_scale = 1.f / quant_arg->out_args_.scale_;
  float out_zp = quant_arg->out_args_.zp_;

  for (int index = 0; index < element_size; ++index) {
    float in0_real = input0[index] * quant_arg->in0_args_.scale_ + in0_bias;
    float in1_real = input1[index] * quant_arg->in1_args_.scale_ + in1_bias;
    float minus_inputs = in0_real - in1_real;
    float out_real = (float)true;
    if (minus_inputs >= -ACCURACY_DATA && minus_inputs <= ACCURACY_DATA) {
      out_real = (float)false;
    }
    output[index] = (int8_t)(out_real * output_inverse_scale + out_zp);
  }
  return NNACL_OK;
}

int ElementEqualInt8(int8_t *input0, int8_t *input1, int8_t *output, int element_size, ArithmeticQuantArg *quant_arg) {
  float in0_bias = -quant_arg->in0_args_.zp_ * quant_arg->in0_args_.scale_;
  float in1_bias = -quant_arg->in1_args_.zp_ * quant_arg->in1_args_.scale_;
  float output_inverse_scale = 1.f / quant_arg->out_args_.scale_;
  float out_zp = quant_arg->out_args_.zp_;
  for (int index = 0; index < element_size; ++index) {
    float in0_real = input0[index] * quant_arg->in0_args_.scale_ + in0_bias;
    float in1_real = input1[index] * quant_arg->in1_args_.scale_ + in1_bias;
    float minus_inputs = in0_real - in1_real;
    float out_real = (float)false;
    if (minus_inputs >= -ACCURACY_DATA && minus_inputs <= ACCURACY_DATA) {
      out_real = (float)true;
    }
    output[index] = (int8_t)(out_real * output_inverse_scale + out_zp);
  }
  return NNACL_OK;
}

int ElementLessInt8(int8_t *input0, int8_t *input1, int8_t *output, int element_size, ArithmeticQuantArg *quant_arg) {
  float in0_bias = -quant_arg->in0_args_.zp_ * quant_arg->in0_args_.scale_;
  float in1_bias = -quant_arg->in1_args_.zp_ * quant_arg->in1_args_.scale_;
  float output_inverse_scale = 1.f / quant_arg->out_args_.scale_;
  float out_zp = quant_arg->out_args_.zp_;
  for (int index = 0; index < element_size; ++index) {
    float in0_real = input0[index] * quant_arg->in0_args_.scale_ + in0_bias;
    float in1_real = input1[index] * quant_arg->in1_args_.scale_ + in1_bias;
    float out_real = (float)(in0_real < in1_real);
    output[index] = (int8_t)(out_real * output_inverse_scale + out_zp);
  }
  return NNACL_OK;
}

int ElementLessEqualInt8(int8_t *input0, int8_t *input1, int8_t *output, int element_size,
                         ArithmeticQuantArg *quant_arg) {
  float in0_bias = -quant_arg->in0_args_.zp_ * quant_arg->in0_args_.scale_;
  float in1_bias = -quant_arg->in1_args_.zp_ * quant_arg->in1_args_.scale_;
  float output_inverse_scale = 1.f / quant_arg->out_args_.scale_;
  float out_zp = quant_arg->out_args_.zp_;

  for (int index = 0; index < element_size; ++index) {
    float in0_real = input0[index] * quant_arg->in0_args_.scale_ + in0_bias;
    float in1_real = input1[index] * quant_arg->in1_args_.scale_ + in1_bias;
    float out_real = (float)(in0_real <= in1_real);
    output[index] = (int8_t)(out_real * output_inverse_scale + out_zp);
  }
  return NNACL_OK;
}

int ElementGreaterInt8(int8_t *input0, int8_t *input1, int8_t *output, int element_size,
                       ArithmeticQuantArg *quant_arg) {
  float in0_bias = -quant_arg->in0_args_.zp_ * quant_arg->in0_args_.scale_;
  float in1_bias = -quant_arg->in1_args_.zp_ * quant_arg->in1_args_.scale_;
  float output_inverse_scale = 1.f / quant_arg->out_args_.scale_;
  float out_zp = quant_arg->out_args_.zp_;

  for (int index = 0; index < element_size; ++index) {
    float in0_real = input0[index] * quant_arg->in0_args_.scale_ + in0_bias;
    float in1_real = input1[index] * quant_arg->in1_args_.scale_ + in1_bias;
    float out_real = (float)(in0_real > in1_real);
    output[index] = (int8_t)(out_real * output_inverse_scale + out_zp);
  }
  return NNACL_OK;
}

int ElementGreaterEqualInt8(int8_t *input0, int8_t *input1, int8_t *output, int element_size,
                            ArithmeticQuantArg *quant_arg) {
  float in0_bias = -quant_arg->in0_args_.zp_ * quant_arg->in0_args_.scale_;
  float in1_bias = -quant_arg->in1_args_.zp_ * quant_arg->in1_args_.scale_;
  float output_inverse_scale = 1.f / quant_arg->out_args_.scale_;
  float out_zp = quant_arg->out_args_.zp_;
  for (int index = 0; index < element_size; ++index) {
    float in0_real = input0[index] * quant_arg->in0_args_.scale_ + in0_bias;
    float in1_real = input1[index] * quant_arg->in1_args_.scale_ + in1_bias;
    float out_real = (float)(in0_real >= in1_real);
    output[index] = (int8_t)(out_real * output_inverse_scale + out_zp);
  }
  return NNACL_OK;
}

#undef ACCURACY_DATA
