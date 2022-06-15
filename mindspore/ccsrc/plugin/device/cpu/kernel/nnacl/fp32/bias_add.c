/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32/bias_add.h"
#include "nnacl/op_base.h"
#include "nnacl/bias_add_simd.h"

void BiasAddByInnerCore(const float *input, const float *bias, float *output, int64_t num) {
  int64_t index = 0;

  SIMD_RUN_NO_SCALAR(BiasAddByInnerCore, index, input, bias, output, num);

  for (; index < num; ++index) {
    output[index] = input[index] + bias[index];
  }
}

void BiasAddByBatchCore(const float *input, const float *bias, float *output, int64_t num) {
  float *output1 = output;
  float *output2 = output + num;
  float *output3 = output + num * 2;
  float *output4 = output + num * 3;
  int64_t index = 0;

  SIMD_RUN_NO_SCALAR(BiasAddByBatchCore, index, input, bias, output1, output2, output3, output4, num);

  const float *input_data1 = input;
  const float *input_data2 = input + num;
  const float *input_data3 = input + num * 2;
  const float *input_data4 = input + num * 3;
  for (; index < num; ++index) {
    output1[index] = input_data1[index] + bias[index];
    output2[index] = input_data2[index] + bias[index];
    output3[index] = input_data3[index] + bias[index];
    output4[index] = input_data4[index] + bias[index];
  }
}

void DoBiasAddByBatch(const float *input, const float *bias, float *output, int64_t start_inner, int64_t start_outer,
                      int64_t end_inner, int64_t end_outer, int64_t inner_num) {
  const float *cur_bias = bias + start_inner;
  if (start_outer == end_outer) {
    BiasAddByInnerCore(input, cur_bias, output, end_inner - start_inner);
    return;
  }
  if (start_inner != 0) {
    BiasAddByInnerCore(input, cur_bias, output, inner_num - start_inner);
    start_outer += 1;
    input += inner_num - start_inner;
    cur_bias = bias;
    output += inner_num - start_inner;
  }
  int64_t step = C4NUM * inner_num;
  for (; start_outer <= end_outer - C4NUM; start_outer += C4NUM) {
    BiasAddByBatchCore(input, cur_bias, output, inner_num);
    input += step;
    output += step;
  }
  for (; start_outer < end_outer; ++start_outer) {
    BiasAddByInnerCore(input, cur_bias, output, inner_num);
    input += inner_num;
    output += inner_num;
  }
  BiasAddByInnerCore(input, cur_bias, output, end_inner);
}

void DoBiasAddByInner(const float *input, const float *bias, float *output, int64_t start_inner, int64_t start_outer,
                      int64_t end_inner, int64_t end_outer, int64_t inner_num) {
  const float *cur_bias = bias + start_inner;
  if (start_outer == end_outer) {
    BiasAddByInnerCore(input, cur_bias, output, end_inner - start_inner);
    return;
  } else {
    BiasAddByInnerCore(input, cur_bias, output, inner_num - start_inner);
    start_outer += 1;
    input += inner_num - start_inner;
    cur_bias = bias;
    output += inner_num - start_inner;
  }
  if (start_outer == end_outer) {
    BiasAddByInnerCore(input, cur_bias, output, end_inner);
    return;
  } else {
    for (; start_outer < end_outer; ++start_outer) {
      BiasAddByInnerCore(input, cur_bias, output, inner_num);
      input += inner_num;
      output += inner_num;
    }
  }
  BiasAddByInnerCore(input, bias, output, end_inner);
}

void BiasAddOpt(const float *input, const float *bias, float *output, int64_t start, int64_t end, int64_t inner_num,
                bool batch_priority) {
  if (inner_num == 0) {
    return;
  }
  int64_t start_outer = start / inner_num;
  int64_t start_inner = start % inner_num;
  int64_t end_outer = end / inner_num;
  int64_t end_inner = end % inner_num;
  const float *cur_input = input + start;
  float *cur_output = output + start;

  if (batch_priority) {
    DoBiasAddByBatch(cur_input, bias, cur_output, start_inner, start_outer, end_inner, end_outer, inner_num);
  } else {
    DoBiasAddByInner(cur_input, bias, cur_output, start_inner, start_outer, end_inner, end_outer, inner_num);
  }
}
