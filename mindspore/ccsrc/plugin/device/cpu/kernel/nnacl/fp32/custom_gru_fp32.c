#ifdef ENABLE_ARM64
/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "nnacl/fp32/custom_gru_fp32.h"
#include "nnacl/fp32/activation_fp32.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/fp32/pack_fp32.h"

void CustomGru(float *output, const float *input, const float *weight_input, const float *weight_hidden,
               const float *bias_input, const float *bias_hidden, const float *init_h, float *buffer[4],
               const CustomGruParameter *gru_param) {
  int num_step = gru_param->num_step;
  int batch_size = gru_param->batch_size;
  int input_size = gru_param->input_size;
  int hidden_size = gru_param->hidden_size;
  int output_size = batch_size * hidden_size;
  int double_output_size = output_size * C2NUM;
  int col_align = UP_ROUND(hidden_size, C8NUM);
  int weight_in_offset = col_align * input_size;
  int weight_hidden_offset = col_align * hidden_size;
  float *input_gate = buffer[1];
  float *hidden_gate = buffer[C3NUM];
  for (int i = 0; i < num_step; ++i) {
    if (batch_size != 1) {
      RowMajor2Col12MajorParallel(input + i * batch_size * input_size, buffer[0], batch_size, input_size, 0,
                                  batch_size);
      for (int j = 0; j < C3NUM; ++j) {
        MatMulOpt(buffer[0], weight_input + j * weight_in_offset, input_gate + j * output_size,
                  bias_input + j * col_align, ActType_No, input_size, batch_size, hidden_size, hidden_size,
                  OutType_Nhwc);
      }
      RowMajor2Col12MajorParallel(init_h, buffer[C2NUM], batch_size, hidden_size, 0, batch_size);
      for (int j = 0; j < C3NUM; ++j) {
        MatMulOpt(buffer[C2NUM], weight_hidden + j * weight_hidden_offset, hidden_gate + j * output_size,
                  bias_hidden + j * col_align, ActType_No, hidden_size, batch_size, hidden_size, hidden_size,
                  OutType_Nhwc);
      }
    } else {
      for (int j = 0; j < C3NUM; ++j) {
        MatVecMulPackFp32(input + i * input_size, weight_input + j * weight_in_offset, input_gate + j * output_size,
                          bias_input + j * col_align, ActType_No, input_size, hidden_size);
        MatVecMulPackFp32(init_h, weight_hidden + j * weight_hidden_offset, hidden_gate + j * output_size,
                          bias_hidden + j * col_align, ActType_No, hidden_size, hidden_size);
      }
    }
    ElementAdd(input_gate, hidden_gate, input_gate, double_output_size);
    Sigmoid(input_gate, double_output_size, input_gate);
    ElementMul(input_gate, hidden_gate + double_output_size, input_gate, output_size);
    ElementAdd(input_gate, input_gate + double_output_size, input_gate, output_size);
    Tanh(input_gate, output_size, input_gate);
    ElementSub(init_h, input_gate, hidden_gate, output_size);
    ElementMul(input_gate + output_size, hidden_gate, hidden_gate, output_size);
    ElementAdd(input_gate, hidden_gate, output, output_size);
    init_h = output;
    output += output_size;
  }
}
#endif
