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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_WRAPPER_FP32_DEQUANT_TO_INT8_FP32_WRAPPER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_WRAPPER_FP32_DEQUANT_TO_INT8_FP32_WRAPPER_H_
#include <stdint.h>
#include <string.h>
typedef struct DeQuantArg {
  float scale;
  int32_t zeroPoint;
  float var_corr;
  float mean_corr;
  float *clusters;
  int clusters_nums;
  int bitNum;
} DeQuantArg;

#ifdef __cplusplus
extern "C" {
#endif

void DequantDataPerChannel(const int8_t *quant_src, const DeQuantArg **de_quant_args, size_t de_quant_nums,
                           size_t per_batch_size, float *de_quant_dst);

void DequantData(const int8_t *quant_src, const DeQuantArg **de_quant_args, size_t de_quant_nums, size_t channels,
                 float *de_quant_dst);

void DequantDataPerTensor(const int8_t *quant_src, const DeQuantArg **de_quant_args, size_t de_quant_nums,
                          float *de_quant_dst);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_WRAPPER_FP32_DEQUANT_TO_INT8_FP32_WRAPPER_H_
