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

#ifndef MINDSPORE_NNACL_FP16_WINOGRAD_UTILS_H_
#define MINDSPORE_NNACL_FP16_WINOGRAD_UTILS_H_

#include "nnacl/conv_parameter.h"
#include "nnacl/op_base.h"
#include "nnacl/intrinsics/ms_simd_instructions_fp16.h"

#define MAX_LEN 256

#ifdef __cplusplus
extern "C" {
#endif
typedef void (*InputTransFp16Func)(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step,
                                   int real_c);

typedef void (*OutputTransFp16Func)(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);

#define Load16DataFp16                           \
  src[0] = vld1q_f16(src_data + 0 * src_step);   \
  src[1] = vld1q_f16(src_data + 1 * src_step);   \
  src[2] = vld1q_f16(src_data + 2 * src_step);   \
  src[3] = vld1q_f16(src_data + 3 * src_step);   \
  src[4] = vld1q_f16(src_data + 4 * src_step);   \
  src[5] = vld1q_f16(src_data + 5 * src_step);   \
  src[6] = vld1q_f16(src_data + 6 * src_step);   \
  src[7] = vld1q_f16(src_data + 7 * src_step);   \
  src[8] = vld1q_f16(src_data + 8 * src_step);   \
  src[9] = vld1q_f16(src_data + 9 * src_step);   \
  src[10] = vld1q_f16(src_data + 10 * src_step); \
  src[11] = vld1q_f16(src_data + 11 * src_step); \
  src[12] = vld1q_f16(src_data + 12 * src_step); \
  src[13] = vld1q_f16(src_data + 13 * src_step); \
  src[14] = vld1q_f16(src_data + 14 * src_step); \
  src[15] = vld1q_f16(src_data + 15 * src_step);

#define Load16DataC4Fp16                        \
  src[0] = vld1_f16(src_data + 0 * src_step);   \
  src[1] = vld1_f16(src_data + 1 * src_step);   \
  src[2] = vld1_f16(src_data + 2 * src_step);   \
  src[3] = vld1_f16(src_data + 3 * src_step);   \
  src[4] = vld1_f16(src_data + 4 * src_step);   \
  src[5] = vld1_f16(src_data + 5 * src_step);   \
  src[6] = vld1_f16(src_data + 6 * src_step);   \
  src[7] = vld1_f16(src_data + 7 * src_step);   \
  src[8] = vld1_f16(src_data + 8 * src_step);   \
  src[9] = vld1_f16(src_data + 9 * src_step);   \
  src[10] = vld1_f16(src_data + 10 * src_step); \
  src[11] = vld1_f16(src_data + 11 * src_step); \
  src[12] = vld1_f16(src_data + 12 * src_step); \
  src[13] = vld1_f16(src_data + 13 * src_step); \
  src[14] = vld1_f16(src_data + 14 * src_step); \
  src[15] = vld1_f16(src_data + 15 * src_step);

#define Load36DataFp16                           \
  src[0] = vld1q_f16(src_data + 0 * src_step);   \
  src[1] = vld1q_f16(src_data + 1 * src_step);   \
  src[2] = vld1q_f16(src_data + 2 * src_step);   \
  src[3] = vld1q_f16(src_data + 3 * src_step);   \
  src[4] = vld1q_f16(src_data + 4 * src_step);   \
  src[5] = vld1q_f16(src_data + 5 * src_step);   \
  src[6] = vld1q_f16(src_data + 6 * src_step);   \
  src[7] = vld1q_f16(src_data + 7 * src_step);   \
  src[8] = vld1q_f16(src_data + 8 * src_step);   \
  src[9] = vld1q_f16(src_data + 9 * src_step);   \
  src[10] = vld1q_f16(src_data + 10 * src_step); \
  src[11] = vld1q_f16(src_data + 11 * src_step); \
  src[12] = vld1q_f16(src_data + 12 * src_step); \
  src[13] = vld1q_f16(src_data + 13 * src_step); \
  src[14] = vld1q_f16(src_data + 14 * src_step); \
  src[15] = vld1q_f16(src_data + 15 * src_step); \
  src[16] = vld1q_f16(src_data + 16 * src_step); \
  src[17] = vld1q_f16(src_data + 17 * src_step); \
  src[18] = vld1q_f16(src_data + 18 * src_step); \
  src[19] = vld1q_f16(src_data + 19 * src_step); \
  src[20] = vld1q_f16(src_data + 20 * src_step); \
  src[21] = vld1q_f16(src_data + 21 * src_step); \
  src[22] = vld1q_f16(src_data + 22 * src_step); \
  src[23] = vld1q_f16(src_data + 23 * src_step); \
  src[24] = vld1q_f16(src_data + 24 * src_step); \
  src[25] = vld1q_f16(src_data + 25 * src_step); \
  src[26] = vld1q_f16(src_data + 26 * src_step); \
  src[27] = vld1q_f16(src_data + 27 * src_step); \
  src[28] = vld1q_f16(src_data + 28 * src_step); \
  src[29] = vld1q_f16(src_data + 29 * src_step); \
  src[30] = vld1q_f16(src_data + 30 * src_step); \
  src[31] = vld1q_f16(src_data + 31 * src_step); \
  src[32] = vld1q_f16(src_data + 32 * src_step); \
  src[33] = vld1q_f16(src_data + 33 * src_step); \
  src[34] = vld1q_f16(src_data + 34 * src_step); \
  src[35] = vld1q_f16(src_data + 35 * src_step);

#define Load36DataC4Fp16                        \
  src[0] = vld1_f16(src_data + 0 * src_step);   \
  src[1] = vld1_f16(src_data + 1 * src_step);   \
  src[2] = vld1_f16(src_data + 2 * src_step);   \
  src[3] = vld1_f16(src_data + 3 * src_step);   \
  src[4] = vld1_f16(src_data + 4 * src_step);   \
  src[5] = vld1_f16(src_data + 5 * src_step);   \
  src[6] = vld1_f16(src_data + 6 * src_step);   \
  src[7] = vld1_f16(src_data + 7 * src_step);   \
  src[8] = vld1_f16(src_data + 8 * src_step);   \
  src[9] = vld1_f16(src_data + 9 * src_step);   \
  src[10] = vld1_f16(src_data + 10 * src_step); \
  src[11] = vld1_f16(src_data + 11 * src_step); \
  src[12] = vld1_f16(src_data + 12 * src_step); \
  src[13] = vld1_f16(src_data + 13 * src_step); \
  src[14] = vld1_f16(src_data + 14 * src_step); \
  src[15] = vld1_f16(src_data + 15 * src_step); \
  src[16] = vld1_f16(src_data + 16 * src_step); \
  src[17] = vld1_f16(src_data + 17 * src_step); \
  src[18] = vld1_f16(src_data + 18 * src_step); \
  src[19] = vld1_f16(src_data + 19 * src_step); \
  src[20] = vld1_f16(src_data + 20 * src_step); \
  src[21] = vld1_f16(src_data + 21 * src_step); \
  src[22] = vld1_f16(src_data + 22 * src_step); \
  src[23] = vld1_f16(src_data + 23 * src_step); \
  src[24] = vld1_f16(src_data + 24 * src_step); \
  src[25] = vld1_f16(src_data + 25 * src_step); \
  src[26] = vld1_f16(src_data + 26 * src_step); \
  src[27] = vld1_f16(src_data + 27 * src_step); \
  src[28] = vld1_f16(src_data + 28 * src_step); \
  src[29] = vld1_f16(src_data + 29 * src_step); \
  src[30] = vld1_f16(src_data + 30 * src_step); \
  src[31] = vld1_f16(src_data + 31 * src_step); \
  src[32] = vld1_f16(src_data + 32 * src_step); \
  src[33] = vld1_f16(src_data + 33 * src_step); \
  src[34] = vld1_f16(src_data + 34 * src_step); \
  src[35] = vld1_f16(src_data + 35 * src_step);

#define Load64DataFp16                           \
  src[0] = vld1q_f16(src_data + 0 * src_step);   \
  src[1] = vld1q_f16(src_data + 1 * src_step);   \
  src[2] = vld1q_f16(src_data + 2 * src_step);   \
  src[3] = vld1q_f16(src_data + 3 * src_step);   \
  src[4] = vld1q_f16(src_data + 4 * src_step);   \
  src[5] = vld1q_f16(src_data + 5 * src_step);   \
  src[6] = vld1q_f16(src_data + 6 * src_step);   \
  src[7] = vld1q_f16(src_data + 7 * src_step);   \
  src[8] = vld1q_f16(src_data + 8 * src_step);   \
  src[9] = vld1q_f16(src_data + 9 * src_step);   \
  src[10] = vld1q_f16(src_data + 10 * src_step); \
  src[11] = vld1q_f16(src_data + 11 * src_step); \
  src[12] = vld1q_f16(src_data + 12 * src_step); \
  src[13] = vld1q_f16(src_data + 13 * src_step); \
  src[14] = vld1q_f16(src_data + 14 * src_step); \
  src[15] = vld1q_f16(src_data + 15 * src_step); \
  src[16] = vld1q_f16(src_data + 16 * src_step); \
  src[17] = vld1q_f16(src_data + 17 * src_step); \
  src[18] = vld1q_f16(src_data + 18 * src_step); \
  src[19] = vld1q_f16(src_data + 19 * src_step); \
  src[20] = vld1q_f16(src_data + 20 * src_step); \
  src[21] = vld1q_f16(src_data + 21 * src_step); \
  src[22] = vld1q_f16(src_data + 22 * src_step); \
  src[23] = vld1q_f16(src_data + 23 * src_step); \
  src[24] = vld1q_f16(src_data + 24 * src_step); \
  src[25] = vld1q_f16(src_data + 25 * src_step); \
  src[26] = vld1q_f16(src_data + 26 * src_step); \
  src[27] = vld1q_f16(src_data + 27 * src_step); \
  src[28] = vld1q_f16(src_data + 28 * src_step); \
  src[29] = vld1q_f16(src_data + 29 * src_step); \
  src[30] = vld1q_f16(src_data + 30 * src_step); \
  src[31] = vld1q_f16(src_data + 31 * src_step); \
  src[32] = vld1q_f16(src_data + 32 * src_step); \
  src[33] = vld1q_f16(src_data + 33 * src_step); \
  src[34] = vld1q_f16(src_data + 34 * src_step); \
  src[35] = vld1q_f16(src_data + 35 * src_step); \
  src[36] = vld1q_f16(src_data + 36 * src_step); \
  src[37] = vld1q_f16(src_data + 37 * src_step); \
  src[38] = vld1q_f16(src_data + 38 * src_step); \
  src[39] = vld1q_f16(src_data + 39 * src_step); \
  src[40] = vld1q_f16(src_data + 40 * src_step); \
  src[41] = vld1q_f16(src_data + 41 * src_step); \
  src[42] = vld1q_f16(src_data + 42 * src_step); \
  src[43] = vld1q_f16(src_data + 43 * src_step); \
  src[44] = vld1q_f16(src_data + 44 * src_step); \
  src[45] = vld1q_f16(src_data + 45 * src_step); \
  src[46] = vld1q_f16(src_data + 46 * src_step); \
  src[47] = vld1q_f16(src_data + 47 * src_step); \
  src[48] = vld1q_f16(src_data + 48 * src_step); \
  src[49] = vld1q_f16(src_data + 49 * src_step); \
  src[50] = vld1q_f16(src_data + 50 * src_step); \
  src[51] = vld1q_f16(src_data + 51 * src_step); \
  src[52] = vld1q_f16(src_data + 52 * src_step); \
  src[53] = vld1q_f16(src_data + 53 * src_step); \
  src[54] = vld1q_f16(src_data + 54 * src_step); \
  src[55] = vld1q_f16(src_data + 55 * src_step); \
  src[56] = vld1q_f16(src_data + 56 * src_step); \
  src[57] = vld1q_f16(src_data + 57 * src_step); \
  src[58] = vld1q_f16(src_data + 58 * src_step); \
  src[59] = vld1q_f16(src_data + 59 * src_step); \
  src[60] = vld1q_f16(src_data + 60 * src_step); \
  src[61] = vld1q_f16(src_data + 61 * src_step); \
  src[62] = vld1q_f16(src_data + 62 * src_step); \
  src[63] = vld1q_f16(src_data + 63 * src_step);

#define Load64DataC4Fp16                        \
  src[0] = vld1_f16(src_data + 0 * src_step);   \
  src[1] = vld1_f16(src_data + 1 * src_step);   \
  src[2] = vld1_f16(src_data + 2 * src_step);   \
  src[3] = vld1_f16(src_data + 3 * src_step);   \
  src[4] = vld1_f16(src_data + 4 * src_step);   \
  src[5] = vld1_f16(src_data + 5 * src_step);   \
  src[6] = vld1_f16(src_data + 6 * src_step);   \
  src[7] = vld1_f16(src_data + 7 * src_step);   \
  src[8] = vld1_f16(src_data + 8 * src_step);   \
  src[9] = vld1_f16(src_data + 9 * src_step);   \
  src[10] = vld1_f16(src_data + 10 * src_step); \
  src[11] = vld1_f16(src_data + 11 * src_step); \
  src[12] = vld1_f16(src_data + 12 * src_step); \
  src[13] = vld1_f16(src_data + 13 * src_step); \
  src[14] = vld1_f16(src_data + 14 * src_step); \
  src[15] = vld1_f16(src_data + 15 * src_step); \
  src[16] = vld1_f16(src_data + 16 * src_step); \
  src[17] = vld1_f16(src_data + 17 * src_step); \
  src[18] = vld1_f16(src_data + 18 * src_step); \
  src[19] = vld1_f16(src_data + 19 * src_step); \
  src[20] = vld1_f16(src_data + 20 * src_step); \
  src[21] = vld1_f16(src_data + 21 * src_step); \
  src[22] = vld1_f16(src_data + 22 * src_step); \
  src[23] = vld1_f16(src_data + 23 * src_step); \
  src[24] = vld1_f16(src_data + 24 * src_step); \
  src[25] = vld1_f16(src_data + 25 * src_step); \
  src[26] = vld1_f16(src_data + 26 * src_step); \
  src[27] = vld1_f16(src_data + 27 * src_step); \
  src[28] = vld1_f16(src_data + 28 * src_step); \
  src[29] = vld1_f16(src_data + 29 * src_step); \
  src[30] = vld1_f16(src_data + 30 * src_step); \
  src[31] = vld1_f16(src_data + 31 * src_step); \
  src[32] = vld1_f16(src_data + 32 * src_step); \
  src[33] = vld1_f16(src_data + 33 * src_step); \
  src[34] = vld1_f16(src_data + 34 * src_step); \
  src[35] = vld1_f16(src_data + 35 * src_step); \
  src[36] = vld1_f16(src_data + 36 * src_step); \
  src[37] = vld1_f16(src_data + 37 * src_step); \
  src[38] = vld1_f16(src_data + 38 * src_step); \
  src[39] = vld1_f16(src_data + 39 * src_step); \
  src[40] = vld1_f16(src_data + 40 * src_step); \
  src[41] = vld1_f16(src_data + 41 * src_step); \
  src[42] = vld1_f16(src_data + 42 * src_step); \
  src[43] = vld1_f16(src_data + 43 * src_step); \
  src[44] = vld1_f16(src_data + 44 * src_step); \
  src[45] = vld1_f16(src_data + 45 * src_step); \
  src[46] = vld1_f16(src_data + 46 * src_step); \
  src[47] = vld1_f16(src_data + 47 * src_step); \
  src[48] = vld1_f16(src_data + 48 * src_step); \
  src[49] = vld1_f16(src_data + 49 * src_step); \
  src[50] = vld1_f16(src_data + 50 * src_step); \
  src[51] = vld1_f16(src_data + 51 * src_step); \
  src[52] = vld1_f16(src_data + 52 * src_step); \
  src[53] = vld1_f16(src_data + 53 * src_step); \
  src[54] = vld1_f16(src_data + 54 * src_step); \
  src[55] = vld1_f16(src_data + 55 * src_step); \
  src[56] = vld1_f16(src_data + 56 * src_step); \
  src[57] = vld1_f16(src_data + 57 * src_step); \
  src[58] = vld1_f16(src_data + 58 * src_step); \
  src[59] = vld1_f16(src_data + 59 * src_step); \
  src[60] = vld1_f16(src_data + 60 * src_step); \
  src[61] = vld1_f16(src_data + 61 * src_step); \
  src[62] = vld1_f16(src_data + 62 * src_step); \
  src[63] = vld1_f16(src_data + 63 * src_step);

InputTransFp16Func GetInputTransFp16Func(int input_unit);

void InputTransform4x4UnitFp16(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step, int real_c);

void InputTransform6x6UnitFp16(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step, int real_c);

void InputTransform8x8UnitFp16(const float16_t *src_data, float16_t *dst_data, int src_step, int dst_step, int real_c);

OutputTransFp16Func GetOutputTransFp16Func(int input_unit, int output_unit, ActType act_type);

#define Store4DataFp16                          \
  vst1q_f16(dst_data, m[0]);                    \
  vst1q_f16(dst_data + out_c, m[1]);            \
  vst1q_f16(dst_data + dst_step * out_c, m[2]); \
  vst1q_f16(dst_data + dst_step * out_c + out_c, m[3]);

#define Store4DataC4Fp16                       \
  vst1_f16(dst_data, m[0]);                    \
  vst1_f16(dst_data + out_c, m[1]);            \
  vst1_f16(dst_data + dst_step * out_c, m[2]); \
  vst1_f16(dst_data + dst_step * out_c + out_c, m[3]);

#define Store9DataFp16                                      \
  vst1q_f16(dst_data, m[0]);                                \
  vst1q_f16(dst_data + out_c, m[1]);                        \
  vst1q_f16(dst_data + 2 * out_c, m[2]);                    \
  vst1q_f16(dst_data + dst_step * out_c, m[3]);             \
  vst1q_f16(dst_data + dst_step * out_c + out_c, m[4]);     \
  vst1q_f16(dst_data + dst_step * out_c + 2 * out_c, m[5]); \
  vst1q_f16(dst_data + 2 * dst_step * out_c, m[6]);         \
  vst1q_f16(dst_data + 2 * dst_step * out_c + out_c, m[7]); \
  vst1q_f16(dst_data + 2 * dst_step * out_c + 2 * out_c, m[8]);

#define Store9DataC4Fp16                                   \
  vst1_f16(dst_data, m[0]);                                \
  vst1_f16(dst_data + out_c, m[1]);                        \
  vst1_f16(dst_data + 2 * out_c, m[2]);                    \
  vst1_f16(dst_data + dst_step * out_c, m[3]);             \
  vst1_f16(dst_data + dst_step * out_c + out_c, m[4]);     \
  vst1_f16(dst_data + dst_step * out_c + 2 * out_c, m[5]); \
  vst1_f16(dst_data + 2 * dst_step * out_c, m[6]);         \
  vst1_f16(dst_data + 2 * dst_step * out_c + out_c, m[7]); \
  vst1_f16(dst_data + 2 * dst_step * out_c + 2 * out_c, m[8]);

#define Store16DataFp16                                          \
  vst1q_f16(dst_data, m[0]);                                     \
  vst1q_f16(dst_data + out_c, m[1]);                             \
  vst1q_f16(dst_data + 2 * out_c, m[2]);                         \
  vst1q_f16(dst_data + 3 * out_c, m[3]);                         \
  vst1q_f16(dst_data + dst_step * out_c, m[4]);                  \
  vst1q_f16(dst_data + dst_step * out_c + out_c, m[5]);          \
  vst1q_f16(dst_data + dst_step * out_c + 2 * out_c, m[6]);      \
  vst1q_f16(dst_data + dst_step * out_c + 3 * out_c, m[7]);      \
  vst1q_f16(dst_data + 2 * dst_step * out_c, m[8]);              \
  vst1q_f16(dst_data + 2 * dst_step * out_c + out_c, m[9]);      \
  vst1q_f16(dst_data + 2 * dst_step * out_c + 2 * out_c, m[10]); \
  vst1q_f16(dst_data + 2 * dst_step * out_c + 3 * out_c, m[11]); \
  vst1q_f16(dst_data + 3 * dst_step * out_c, m[12]);             \
  vst1q_f16(dst_data + 3 * dst_step * out_c + out_c, m[13]);     \
  vst1q_f16(dst_data + 3 * dst_step * out_c + 2 * out_c, m[14]); \
  vst1q_f16(dst_data + 3 * dst_step * out_c + 3 * out_c, m[15]);

#define Store16DataC4Fp16                                       \
  vst1_f16(dst_data, m[0]);                                     \
  vst1_f16(dst_data + out_c, m[1]);                             \
  vst1_f16(dst_data + 2 * out_c, m[2]);                         \
  vst1_f16(dst_data + 3 * out_c, m[3]);                         \
  vst1_f16(dst_data + dst_step * out_c, m[4]);                  \
  vst1_f16(dst_data + dst_step * out_c + out_c, m[5]);          \
  vst1_f16(dst_data + dst_step * out_c + 2 * out_c, m[6]);      \
  vst1_f16(dst_data + dst_step * out_c + 3 * out_c, m[7]);      \
  vst1_f16(dst_data + 2 * dst_step * out_c, m[8]);              \
  vst1_f16(dst_data + 2 * dst_step * out_c + out_c, m[9]);      \
  vst1_f16(dst_data + 2 * dst_step * out_c + 2 * out_c, m[10]); \
  vst1_f16(dst_data + 2 * dst_step * out_c + 3 * out_c, m[11]); \
  vst1_f16(dst_data + 3 * dst_step * out_c, m[12]);             \
  vst1_f16(dst_data + 3 * dst_step * out_c + out_c, m[13]);     \
  vst1_f16(dst_data + 3 * dst_step * out_c + 2 * out_c, m[14]); \
  vst1_f16(dst_data + 3 * dst_step * out_c + 3 * out_c, m[15]);

#define Store25DataFp16                                          \
  vst1q_f16(dst_data, m[0]);                                     \
  vst1q_f16(dst_data + out_c, m[1]);                             \
  vst1q_f16(dst_data + 2 * out_c, m[2]);                         \
  vst1q_f16(dst_data + 3 * out_c, m[3]);                         \
  vst1q_f16(dst_data + 4 * out_c, m[4]);                         \
  vst1q_f16(dst_data + dst_step * out_c, m[5]);                  \
  vst1q_f16(dst_data + dst_step * out_c + out_c, m[6]);          \
  vst1q_f16(dst_data + dst_step * out_c + 2 * out_c, m[7]);      \
  vst1q_f16(dst_data + dst_step * out_c + 3 * out_c, m[8]);      \
  vst1q_f16(dst_data + dst_step * out_c + 4 * out_c, m[9]);      \
  vst1q_f16(dst_data + 2 * dst_step * out_c, m[10]);             \
  vst1q_f16(dst_data + 2 * dst_step * out_c + out_c, m[11]);     \
  vst1q_f16(dst_data + 2 * dst_step * out_c + 2 * out_c, m[12]); \
  vst1q_f16(dst_data + 2 * dst_step * out_c + 3 * out_c, m[13]); \
  vst1q_f16(dst_data + 2 * dst_step * out_c + 4 * out_c, m[14]); \
  vst1q_f16(dst_data + 3 * dst_step * out_c, m[15]);             \
  vst1q_f16(dst_data + 3 * dst_step * out_c + out_c, m[16]);     \
  vst1q_f16(dst_data + 3 * dst_step * out_c + 2 * out_c, m[17]); \
  vst1q_f16(dst_data + 3 * dst_step * out_c + 3 * out_c, m[18]); \
  vst1q_f16(dst_data + 3 * dst_step * out_c + 4 * out_c, m[19]); \
  vst1q_f16(dst_data + 4 * dst_step * out_c, m[20]);             \
  vst1q_f16(dst_data + 4 * dst_step * out_c + out_c, m[21]);     \
  vst1q_f16(dst_data + 4 * dst_step * out_c + 2 * out_c, m[22]); \
  vst1q_f16(dst_data + 4 * dst_step * out_c + 3 * out_c, m[23]); \
  vst1q_f16(dst_data + 4 * dst_step * out_c + 4 * out_c, m[24]);

#define Store25DataC4Fp16                                       \
  vst1_f16(dst_data, m[0]);                                     \
  vst1_f16(dst_data + out_c, m[1]);                             \
  vst1_f16(dst_data + 2 * out_c, m[2]);                         \
  vst1_f16(dst_data + 3 * out_c, m[3]);                         \
  vst1_f16(dst_data + 4 * out_c, m[4]);                         \
  vst1_f16(dst_data + dst_step * out_c, m[5]);                  \
  vst1_f16(dst_data + dst_step * out_c + out_c, m[6]);          \
  vst1_f16(dst_data + dst_step * out_c + 2 * out_c, m[7]);      \
  vst1_f16(dst_data + dst_step * out_c + 3 * out_c, m[8]);      \
  vst1_f16(dst_data + dst_step * out_c + 4 * out_c, m[9]);      \
  vst1_f16(dst_data + 2 * dst_step * out_c, m[10]);             \
  vst1_f16(dst_data + 2 * dst_step * out_c + out_c, m[11]);     \
  vst1_f16(dst_data + 2 * dst_step * out_c + 2 * out_c, m[12]); \
  vst1_f16(dst_data + 2 * dst_step * out_c + 3 * out_c, m[13]); \
  vst1_f16(dst_data + 2 * dst_step * out_c + 4 * out_c, m[14]); \
  vst1_f16(dst_data + 3 * dst_step * out_c, m[15]);             \
  vst1_f16(dst_data + 3 * dst_step * out_c + out_c, m[16]);     \
  vst1_f16(dst_data + 3 * dst_step * out_c + 2 * out_c, m[17]); \
  vst1_f16(dst_data + 3 * dst_step * out_c + 3 * out_c, m[18]); \
  vst1_f16(dst_data + 3 * dst_step * out_c + 4 * out_c, m[19]); \
  vst1_f16(dst_data + 4 * dst_step * out_c, m[20]);             \
  vst1_f16(dst_data + 4 * dst_step * out_c + out_c, m[21]);     \
  vst1_f16(dst_data + 4 * dst_step * out_c + 2 * out_c, m[22]); \
  vst1_f16(dst_data + 4 * dst_step * out_c + 3 * out_c, m[23]); \
  vst1_f16(dst_data + 4 * dst_step * out_c + 4 * out_c, m[24]);

void OutputTransform4x2UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform4x2ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform4x2Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform4x3UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform4x3ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform4x3Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);

void OutputTransform6x2UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x2ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x2Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x3UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x3ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x3Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x4UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x4ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x4Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x5UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x5ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x5Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);

void OutputTransform8x2UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x2ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x2Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x3UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x3ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x3Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x4UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x4ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x4Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x5UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x5ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x5Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x6ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x6Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x7UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x7ReluUnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                    int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x7Relu6UnitFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *bias_data,
                                     int src_step, int dst_step, int out_c, int r_w, int r_h, int r_c);

int SelectOutputUnitFp16(const ConvParameter *conv_param);

void CheckIfUseWinogradFp16(bool *use_winograd, int *output_unit, const ConvParameter *conv_param);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP16_WINOGRAD_UTILS_H_
