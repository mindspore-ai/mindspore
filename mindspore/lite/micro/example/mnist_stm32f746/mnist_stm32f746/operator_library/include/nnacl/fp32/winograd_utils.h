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

#ifndef MINDSPORE_LITE_NNACL_WINOGRAD_UTILS_H_
#define MINDSPORE_LITE_NNACL_WINOGRAD_UTILS_H_

#ifdef ENABLE_ARM
#include <arm_neon.h>
#endif
#include "nnacl/conv_parameter.h"
#include "nnacl/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef void (*InputTransFunc)(const float *src_data, float *dst_data, int src_step, int dst_step, int real_c);

typedef void (*OutputTransFunc)(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c);

void GeneralInputTransformUnit(const float *src_data, float *dst_data, const float *matrix_b, const float *matrix_bt,
                               int src_step, int dst_step, int in_unit);

void GeneralOutputTransformUnit(const float *src_data, float *dst_data, const float *bias_data, const float *matrix_a,
                                const float *matrix_at, int src_step, int dst_step, int in_unit, int out_unit);

#define Load16Data                                \
  src[0] = MS_LDQ_F32(src_data + 0 * src_step);   \
  src[1] = MS_LDQ_F32(src_data + 1 * src_step);   \
  src[2] = MS_LDQ_F32(src_data + 2 * src_step);   \
  src[3] = MS_LDQ_F32(src_data + 3 * src_step);   \
  src[4] = MS_LDQ_F32(src_data + 4 * src_step);   \
  src[5] = MS_LDQ_F32(src_data + 5 * src_step);   \
  src[6] = MS_LDQ_F32(src_data + 6 * src_step);   \
  src[7] = MS_LDQ_F32(src_data + 7 * src_step);   \
  src[8] = MS_LDQ_F32(src_data + 8 * src_step);   \
  src[9] = MS_LDQ_F32(src_data + 9 * src_step);   \
  src[10] = MS_LDQ_F32(src_data + 10 * src_step); \
  src[11] = MS_LDQ_F32(src_data + 11 * src_step); \
  src[12] = MS_LDQ_F32(src_data + 12 * src_step); \
  src[13] = MS_LDQ_F32(src_data + 13 * src_step); \
  src[14] = MS_LDQ_F32(src_data + 14 * src_step); \
  src[15] = MS_LDQ_F32(src_data + 15 * src_step);

#define Load36Data                                \
  src[0] = MS_LDQ_F32(src_data + 0 * src_step);   \
  src[1] = MS_LDQ_F32(src_data + 1 * src_step);   \
  src[2] = MS_LDQ_F32(src_data + 2 * src_step);   \
  src[3] = MS_LDQ_F32(src_data + 3 * src_step);   \
  src[4] = MS_LDQ_F32(src_data + 4 * src_step);   \
  src[5] = MS_LDQ_F32(src_data + 5 * src_step);   \
  src[6] = MS_LDQ_F32(src_data + 6 * src_step);   \
  src[7] = MS_LDQ_F32(src_data + 7 * src_step);   \
  src[8] = MS_LDQ_F32(src_data + 8 * src_step);   \
  src[9] = MS_LDQ_F32(src_data + 9 * src_step);   \
  src[10] = MS_LDQ_F32(src_data + 10 * src_step); \
  src[11] = MS_LDQ_F32(src_data + 11 * src_step); \
  src[12] = MS_LDQ_F32(src_data + 12 * src_step); \
  src[13] = MS_LDQ_F32(src_data + 13 * src_step); \
  src[14] = MS_LDQ_F32(src_data + 14 * src_step); \
  src[15] = MS_LDQ_F32(src_data + 15 * src_step); \
  src[16] = MS_LDQ_F32(src_data + 16 * src_step); \
  src[17] = MS_LDQ_F32(src_data + 17 * src_step); \
  src[18] = MS_LDQ_F32(src_data + 18 * src_step); \
  src[19] = MS_LDQ_F32(src_data + 19 * src_step); \
  src[20] = MS_LDQ_F32(src_data + 20 * src_step); \
  src[21] = MS_LDQ_F32(src_data + 21 * src_step); \
  src[22] = MS_LDQ_F32(src_data + 22 * src_step); \
  src[23] = MS_LDQ_F32(src_data + 23 * src_step); \
  src[24] = MS_LDQ_F32(src_data + 24 * src_step); \
  src[25] = MS_LDQ_F32(src_data + 25 * src_step); \
  src[26] = MS_LDQ_F32(src_data + 26 * src_step); \
  src[27] = MS_LDQ_F32(src_data + 27 * src_step); \
  src[28] = MS_LDQ_F32(src_data + 28 * src_step); \
  src[29] = MS_LDQ_F32(src_data + 29 * src_step); \
  src[30] = MS_LDQ_F32(src_data + 30 * src_step); \
  src[31] = MS_LDQ_F32(src_data + 31 * src_step); \
  src[32] = MS_LDQ_F32(src_data + 32 * src_step); \
  src[33] = MS_LDQ_F32(src_data + 33 * src_step); \
  src[34] = MS_LDQ_F32(src_data + 34 * src_step); \
  src[35] = MS_LDQ_F32(src_data + 35 * src_step);

#define Load64Data                                \
  src[0] = MS_LDQ_F32(src_data + 0 * src_step);   \
  src[1] = MS_LDQ_F32(src_data + 1 * src_step);   \
  src[2] = MS_LDQ_F32(src_data + 2 * src_step);   \
  src[3] = MS_LDQ_F32(src_data + 3 * src_step);   \
  src[4] = MS_LDQ_F32(src_data + 4 * src_step);   \
  src[5] = MS_LDQ_F32(src_data + 5 * src_step);   \
  src[6] = MS_LDQ_F32(src_data + 6 * src_step);   \
  src[7] = MS_LDQ_F32(src_data + 7 * src_step);   \
  src[8] = MS_LDQ_F32(src_data + 8 * src_step);   \
  src[9] = MS_LDQ_F32(src_data + 9 * src_step);   \
  src[10] = MS_LDQ_F32(src_data + 10 * src_step); \
  src[11] = MS_LDQ_F32(src_data + 11 * src_step); \
  src[12] = MS_LDQ_F32(src_data + 12 * src_step); \
  src[13] = MS_LDQ_F32(src_data + 13 * src_step); \
  src[14] = MS_LDQ_F32(src_data + 14 * src_step); \
  src[15] = MS_LDQ_F32(src_data + 15 * src_step); \
  src[16] = MS_LDQ_F32(src_data + 16 * src_step); \
  src[17] = MS_LDQ_F32(src_data + 17 * src_step); \
  src[18] = MS_LDQ_F32(src_data + 18 * src_step); \
  src[19] = MS_LDQ_F32(src_data + 19 * src_step); \
  src[20] = MS_LDQ_F32(src_data + 20 * src_step); \
  src[21] = MS_LDQ_F32(src_data + 21 * src_step); \
  src[22] = MS_LDQ_F32(src_data + 22 * src_step); \
  src[23] = MS_LDQ_F32(src_data + 23 * src_step); \
  src[24] = MS_LDQ_F32(src_data + 24 * src_step); \
  src[25] = MS_LDQ_F32(src_data + 25 * src_step); \
  src[26] = MS_LDQ_F32(src_data + 26 * src_step); \
  src[27] = MS_LDQ_F32(src_data + 27 * src_step); \
  src[28] = MS_LDQ_F32(src_data + 28 * src_step); \
  src[29] = MS_LDQ_F32(src_data + 29 * src_step); \
  src[30] = MS_LDQ_F32(src_data + 30 * src_step); \
  src[31] = MS_LDQ_F32(src_data + 31 * src_step); \
  src[32] = MS_LDQ_F32(src_data + 32 * src_step); \
  src[33] = MS_LDQ_F32(src_data + 33 * src_step); \
  src[34] = MS_LDQ_F32(src_data + 34 * src_step); \
  src[35] = MS_LDQ_F32(src_data + 35 * src_step); \
  src[36] = MS_LDQ_F32(src_data + 36 * src_step); \
  src[37] = MS_LDQ_F32(src_data + 37 * src_step); \
  src[38] = MS_LDQ_F32(src_data + 38 * src_step); \
  src[39] = MS_LDQ_F32(src_data + 39 * src_step); \
  src[40] = MS_LDQ_F32(src_data + 40 * src_step); \
  src[41] = MS_LDQ_F32(src_data + 41 * src_step); \
  src[42] = MS_LDQ_F32(src_data + 42 * src_step); \
  src[43] = MS_LDQ_F32(src_data + 43 * src_step); \
  src[44] = MS_LDQ_F32(src_data + 44 * src_step); \
  src[45] = MS_LDQ_F32(src_data + 45 * src_step); \
  src[46] = MS_LDQ_F32(src_data + 46 * src_step); \
  src[47] = MS_LDQ_F32(src_data + 47 * src_step); \
  src[48] = MS_LDQ_F32(src_data + 48 * src_step); \
  src[49] = MS_LDQ_F32(src_data + 49 * src_step); \
  src[50] = MS_LDQ_F32(src_data + 50 * src_step); \
  src[51] = MS_LDQ_F32(src_data + 51 * src_step); \
  src[52] = MS_LDQ_F32(src_data + 52 * src_step); \
  src[53] = MS_LDQ_F32(src_data + 53 * src_step); \
  src[54] = MS_LDQ_F32(src_data + 54 * src_step); \
  src[55] = MS_LDQ_F32(src_data + 55 * src_step); \
  src[56] = MS_LDQ_F32(src_data + 56 * src_step); \
  src[57] = MS_LDQ_F32(src_data + 57 * src_step); \
  src[58] = MS_LDQ_F32(src_data + 58 * src_step); \
  src[59] = MS_LDQ_F32(src_data + 59 * src_step); \
  src[60] = MS_LDQ_F32(src_data + 60 * src_step); \
  src[61] = MS_LDQ_F32(src_data + 61 * src_step); \
  src[62] = MS_LDQ_F32(src_data + 62 * src_step); \
  src[63] = MS_LDQ_F32(src_data + 63 * src_step);

InputTransFunc GetInputTransFunc(int input_unit);

void InputTransform4x4Unit(const float *src_data, float *dst_data, int src_step, int dst_step, int real_c);

void InputTransform6x6Unit(const float *src_data, float *dst_data, int src_step, int dst_step, int real_c);

void InputTransform8x8Unit(const float *src_data, float *dst_data, int src_step, int dst_step, int real_c);

OutputTransFunc GetOutputTransFunc(int input_unit, int output_unit, ActType act_type);

#define Store4Data                               \
  MS_STQ_F32(dst_data, m[0]);                    \
  MS_STQ_F32(dst_data + out_c, m[1]);            \
  MS_STQ_F32(dst_data + dst_step * out_c, m[2]); \
  MS_STQ_F32(dst_data + dst_step * out_c + out_c, m[3]);

#define Store9Data                                           \
  MS_STQ_F32(dst_data, m[0]);                                \
  MS_STQ_F32(dst_data + out_c, m[1]);                        \
  MS_STQ_F32(dst_data + 2 * out_c, m[2]);                    \
  MS_STQ_F32(dst_data + dst_step * out_c, m[3]);             \
  MS_STQ_F32(dst_data + dst_step * out_c + out_c, m[4]);     \
  MS_STQ_F32(dst_data + dst_step * out_c + 2 * out_c, m[5]); \
  MS_STQ_F32(dst_data + 2 * dst_step * out_c, m[6]);         \
  MS_STQ_F32(dst_data + 2 * dst_step * out_c + out_c, m[7]); \
  MS_STQ_F32(dst_data + 2 * dst_step * out_c + 2 * out_c, m[8]);

#define Store16Data                                               \
  MS_STQ_F32(dst_data, m[0]);                                     \
  MS_STQ_F32(dst_data + out_c, m[1]);                             \
  MS_STQ_F32(dst_data + 2 * out_c, m[2]);                         \
  MS_STQ_F32(dst_data + 3 * out_c, m[3]);                         \
  MS_STQ_F32(dst_data + dst_step * out_c, m[4]);                  \
  MS_STQ_F32(dst_data + dst_step * out_c + out_c, m[5]);          \
  MS_STQ_F32(dst_data + dst_step * out_c + 2 * out_c, m[6]);      \
  MS_STQ_F32(dst_data + dst_step * out_c + 3 * out_c, m[7]);      \
  MS_STQ_F32(dst_data + 2 * dst_step * out_c, m[8]);              \
  MS_STQ_F32(dst_data + 2 * dst_step * out_c + out_c, m[9]);      \
  MS_STQ_F32(dst_data + 2 * dst_step * out_c + 2 * out_c, m[10]); \
  MS_STQ_F32(dst_data + 2 * dst_step * out_c + 3 * out_c, m[11]); \
  MS_STQ_F32(dst_data + 3 * dst_step * out_c, m[12]);             \
  MS_STQ_F32(dst_data + 3 * dst_step * out_c + out_c, m[13]);     \
  MS_STQ_F32(dst_data + 3 * dst_step * out_c + 2 * out_c, m[14]); \
  MS_STQ_F32(dst_data + 3 * dst_step * out_c + 3 * out_c, m[15]);

#define Store25Data                                               \
  MS_STQ_F32(dst_data, m[0]);                                     \
  MS_STQ_F32(dst_data + out_c, m[1]);                             \
  MS_STQ_F32(dst_data + 2 * out_c, m[2]);                         \
  MS_STQ_F32(dst_data + 3 * out_c, m[3]);                         \
  MS_STQ_F32(dst_data + 4 * out_c, m[4]);                         \
  MS_STQ_F32(dst_data + dst_step * out_c, m[5]);                  \
  MS_STQ_F32(dst_data + dst_step * out_c + out_c, m[6]);          \
  MS_STQ_F32(dst_data + dst_step * out_c + 2 * out_c, m[7]);      \
  MS_STQ_F32(dst_data + dst_step * out_c + 3 * out_c, m[8]);      \
  MS_STQ_F32(dst_data + dst_step * out_c + 4 * out_c, m[9]);      \
  MS_STQ_F32(dst_data + 2 * dst_step * out_c, m[10]);             \
  MS_STQ_F32(dst_data + 2 * dst_step * out_c + out_c, m[11]);     \
  MS_STQ_F32(dst_data + 2 * dst_step * out_c + 2 * out_c, m[12]); \
  MS_STQ_F32(dst_data + 2 * dst_step * out_c + 3 * out_c, m[13]); \
  MS_STQ_F32(dst_data + 2 * dst_step * out_c + 4 * out_c, m[14]); \
  MS_STQ_F32(dst_data + 3 * dst_step * out_c, m[15]);             \
  MS_STQ_F32(dst_data + 3 * dst_step * out_c + out_c, m[16]);     \
  MS_STQ_F32(dst_data + 3 * dst_step * out_c + 2 * out_c, m[17]); \
  MS_STQ_F32(dst_data + 3 * dst_step * out_c + 3 * out_c, m[18]); \
  MS_STQ_F32(dst_data + 3 * dst_step * out_c + 4 * out_c, m[19]); \
  MS_STQ_F32(dst_data + 4 * dst_step * out_c, m[20]);             \
  MS_STQ_F32(dst_data + 4 * dst_step * out_c + out_c, m[21]);     \
  MS_STQ_F32(dst_data + 4 * dst_step * out_c + 2 * out_c, m[22]); \
  MS_STQ_F32(dst_data + 4 * dst_step * out_c + 3 * out_c, m[23]); \
  MS_STQ_F32(dst_data + 4 * dst_step * out_c + 4 * out_c, m[24]);

void OutputTransform4x2Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c);
void OutputTransform4x2ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform4x2Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform4x3Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c);
void OutputTransform4x3ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform4x3Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c);

void OutputTransform6x2Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x2ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x2Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x3Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x3ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x3Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x4Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x4ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x4Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x5Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x5ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x5Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c);

void OutputTransform8x2Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x2ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x2Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x3Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x3ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x3Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x4Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x4ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x4Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x5Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x5ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x5Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x6ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x6Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x7Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step, int dst_step,
                            int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x7ReluUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x7Relu6Unit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                 int dst_step, int out_c, int r_w, int r_h, int r_c);

int SelectOutputUnit(ConvParameter *conv_param);

bool CheckIfUseWinograd(int *output_unit, ConvParameter *conv_param);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_WINOGRAD_UTILS_H_
