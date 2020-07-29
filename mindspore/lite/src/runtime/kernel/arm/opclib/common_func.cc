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

#include "src/runtime/kernel/arm/opclib/common_func.h"
#include "src/runtime/kernel/arm/opclib/quantization/fixed_point.h"

#ifndef ENABLE_ARM
void IndirectGemmFp32(float *output, const float *input, const float *weight, const float *bias, size_t step, int ic4,
                      int output_channel, size_t offset, size_t relu, size_t relu6) {
  for (int i = 0; i < TILE_NUM; i++) {
    int input_tile_offset = i * C4NUM;
    int output_tile_offset = i * output_channel;
    for (int j = 0; j < output_channel; j++) {
      int oc8_block = j / C8NUM;
      int oc8_res = j % C8NUM;
      int weight_oc_offset = oc8_block * step * ic4 * C4NUM * C8NUM + oc8_res;
      int out_oc_offset = output_tile_offset + j;

      float acc = 0;
      for (int n = 0; n < step; n++) {
        int input_kw_offset = input_tile_offset + n * ic4 * C4NUM * TILE_NUM;
        int weight_kw_offset = weight_oc_offset + n * ic4 * C4NUM * C8NUM;

        for (int k = 0; k < ic4; k++) {
          int input_ic4_offset = input_kw_offset + k * TILE_NUM * C4NUM;
          int weight_ic4_offset = weight_kw_offset + k * C4NUM * C8NUM;
          for (int m = 0; m < C4NUM; m++) {
            int input_ic_offset = input_ic4_offset + m;
            int weight_ic_offset = weight_ic4_offset + m * C8NUM;
            acc += (weight + weight_ic_offset)[0] * (input + input_ic_offset)[0];
          }
        }
      }
      acc += bias[j];
      if (relu) {
        acc = acc > 0 ? acc : 0;
      } else if (relu6) {
        if (acc < 0) {
          acc = 0;
        } else if (acc > 6) {
          acc = 6;
        } else {
        }
      }
      (output + out_oc_offset)[0] = acc;
    }
  }
}

void IndirectGemmFp32_8x8(float *output, const float *input, const float *weight, const float *bias, size_t step,
                          size_t ic4, size_t output_channel, size_t offset, size_t mode, size_t writeC4, size_t relu,
                          size_t relu6) {
  int oc4 = UP_DIV(output_channel, C4NUM);
  if (mode && writeC4) {
    for (int i = 0; i < TILE_NUM; i++) {
      int input_tile_offset = i * C4NUM;
      int output_tile_offset = i * oc4 * C4NUM * step;
      for (int j = 0; j < output_channel; j++) {
        int oc4_block = j / 4;
        int oc4_res = j % 4;
        int oc8_block = oc4_block / 2;
        int oc8_res = oc4_block % 2;
        int weight_oc_offset = oc8_block * step * ic4 * C4NUM * C8NUM + oc8_res * C4NUM + oc4_res;
        int out_oc_offset = output_tile_offset + oc4_block * step * C4NUM + oc4_res;

        for (int n = 0; n < step; n++) {
          int input_kw_offset = input_tile_offset + n * ic4 * C4NUM * TILE_NUM;
          int weight_kw_offset = weight_oc_offset + n * ic4 * C4NUM * C8NUM;
          int output_kw_offset = out_oc_offset + n * C4NUM;
          float acc = 0;

          for (int k = 0; k < ic4; k++) {
            int input_ic4_offset = input_kw_offset + k * TILE_NUM * C4NUM;
            int weight_ic4_offset = weight_kw_offset + k * C4NUM * C8NUM;
            for (int m = 0; m < 4; m++) {
              int input_ic_offset = input_ic4_offset + m;
              int weight_ic_offset = weight_ic4_offset + m * C8NUM;
              acc += (weight + weight_ic_offset)[0] * (input + input_ic_offset)[0];
            }
          }
          (output + output_kw_offset)[0] = acc;
        }
      }
    }
  } else if (mode) {
    IndirectGemmFp32_Comm(output, input, weight, ic4, C8NUM, output_channel, offset);
  } else {
    IndirectGemmFp32(output, input, weight, bias, step, ic4, output_channel, offset, relu, relu6);
  }
}
#endif

int8_t MinInt8(int8_t a, int8_t b) { return b ^ ((a ^ b) & -(a < b)); }

int8_t MaxInt8(int8_t a, int8_t b) { return a ^ ((a ^ b) & -(a < b)); }

void ReluFp32(float *data, int ele_num) {
  for (int i = 0; i < ele_num; i++) {
    if (data[i] < 0) {
      data[i] = 0;
    } else {
      // do nothing
    }
  }
}

void Relu6Fp32(float *data, int ele_num) {
  for (int i = 0; i < ele_num; i++) {
    if (data[i] < 0) {
      data[i] = 0;
    } else if (data[i] > 6) {
      data[i] = 6;
    } else {
      // do nothing
    }
  }
}

void IndirectGemmFp32_Comm(float *output, const float *input, const float *weight, size_t ic4, size_t hw, size_t oc,
                           size_t offset) {
  for (int r = 0; r < hw; r++) {
    for (int c = 0; c < oc; c++) {
      float value = 0;
      for (int deep = 0; deep < ic4; deep++) {
        int d4mod = deep % 4;
        int d4div = deep / 4;
        int a_index = d4div * 4 * 8 + r * 4 + d4mod;
        int b_index = 8 * deep + c;
        value += input[a_index] * weight[b_index];
      }
      output[r * offset + c] = value;
    }
  }
  return;
}

void PostFuncInt8(const int *in, const int *bias, int8_t *out, int oc, int plane, int plane8, int32_t multiplier,
                  int32_t left_shift, int32_t right_shift, int32_t zp, int8_t mini, int8_t maxi) {
  /*  (int32_t)row8x8-major * multiplier + bias  =>  (int8)relu  =>  (int8_t)row-major  */
  for (int r = 0; r < plane; r++) {
    for (int c = 0; c < oc; c++) {
      int c8div = c / 8, c8mod = c % 8;
      int src_index = c8div * plane8 * 8 + r * 8 + c8mod;
      int dst_index = r * oc + c;
      int32_t value = in[src_index];
      if (bias != nullptr) {
        value = in[src_index] + bias[c];
      }
      value = MultiplyByQuantizedMultiplier(value, multiplier, left_shift, right_shift) + zp;
      value = MSMIN(maxi, value);
      value = MSMAX(mini, value);
      out[dst_index] = (int8_t)value;
    }
  }
  return;
}
