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
#ifdef ENABLE_AVX
#ifndef MINDSPORE_NNACL_WINOGRAD_AVX_H_
#define MINDSPORE_NNACL_WINOGRAD_AVX_H_

#include "nnacl/conv_parameter.h"
#include "nnacl/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef void (*OutputTransFunc)(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                int dst_step, int out_c, int r_w, int r_h, int r_c);

#define LoadAvx16Data                               \
  src[0] = MS_LD256_F32(src_data + 0 * src_step);   \
  src[1] = MS_LD256_F32(src_data + 1 * src_step);   \
  src[2] = MS_LD256_F32(src_data + 2 * src_step);   \
  src[3] = MS_LD256_F32(src_data + 3 * src_step);   \
  src[4] = MS_LD256_F32(src_data + 4 * src_step);   \
  src[5] = MS_LD256_F32(src_data + 5 * src_step);   \
  src[6] = MS_LD256_F32(src_data + 6 * src_step);   \
  src[7] = MS_LD256_F32(src_data + 7 * src_step);   \
  src[8] = MS_LD256_F32(src_data + 8 * src_step);   \
  src[9] = MS_LD256_F32(src_data + 9 * src_step);   \
  src[10] = MS_LD256_F32(src_data + 10 * src_step); \
  src[11] = MS_LD256_F32(src_data + 11 * src_step); \
  src[12] = MS_LD256_F32(src_data + 12 * src_step); \
  src[13] = MS_LD256_F32(src_data + 13 * src_step); \
  src[14] = MS_LD256_F32(src_data + 14 * src_step); \
  src[15] = MS_LD256_F32(src_data + 15 * src_step);

#define LoadAvx36Data                               \
  src[0] = MS_LD256_F32(src_data + 0 * src_step);   \
  src[1] = MS_LD256_F32(src_data + 1 * src_step);   \
  src[2] = MS_LD256_F32(src_data + 2 * src_step);   \
  src[3] = MS_LD256_F32(src_data + 3 * src_step);   \
  src[4] = MS_LD256_F32(src_data + 4 * src_step);   \
  src[5] = MS_LD256_F32(src_data + 5 * src_step);   \
  src[6] = MS_LD256_F32(src_data + 6 * src_step);   \
  src[7] = MS_LD256_F32(src_data + 7 * src_step);   \
  src[8] = MS_LD256_F32(src_data + 8 * src_step);   \
  src[9] = MS_LD256_F32(src_data + 9 * src_step);   \
  src[10] = MS_LD256_F32(src_data + 10 * src_step); \
  src[11] = MS_LD256_F32(src_data + 11 * src_step); \
  src[12] = MS_LD256_F32(src_data + 12 * src_step); \
  src[13] = MS_LD256_F32(src_data + 13 * src_step); \
  src[14] = MS_LD256_F32(src_data + 14 * src_step); \
  src[15] = MS_LD256_F32(src_data + 15 * src_step); \
  src[16] = MS_LD256_F32(src_data + 16 * src_step); \
  src[17] = MS_LD256_F32(src_data + 17 * src_step); \
  src[18] = MS_LD256_F32(src_data + 18 * src_step); \
  src[19] = MS_LD256_F32(src_data + 19 * src_step); \
  src[20] = MS_LD256_F32(src_data + 20 * src_step); \
  src[21] = MS_LD256_F32(src_data + 21 * src_step); \
  src[22] = MS_LD256_F32(src_data + 22 * src_step); \
  src[23] = MS_LD256_F32(src_data + 23 * src_step); \
  src[24] = MS_LD256_F32(src_data + 24 * src_step); \
  src[25] = MS_LD256_F32(src_data + 25 * src_step); \
  src[26] = MS_LD256_F32(src_data + 26 * src_step); \
  src[27] = MS_LD256_F32(src_data + 27 * src_step); \
  src[28] = MS_LD256_F32(src_data + 28 * src_step); \
  src[29] = MS_LD256_F32(src_data + 29 * src_step); \
  src[30] = MS_LD256_F32(src_data + 30 * src_step); \
  src[31] = MS_LD256_F32(src_data + 31 * src_step); \
  src[32] = MS_LD256_F32(src_data + 32 * src_step); \
  src[33] = MS_LD256_F32(src_data + 33 * src_step); \
  src[34] = MS_LD256_F32(src_data + 34 * src_step); \
  src[35] = MS_LD256_F32(src_data + 35 * src_step);

#define LoadAvx64Data                               \
  src[0] = MS_LD256_F32(src_data + 0 * src_step);   \
  src[1] = MS_LD256_F32(src_data + 1 * src_step);   \
  src[2] = MS_LD256_F32(src_data + 2 * src_step);   \
  src[3] = MS_LD256_F32(src_data + 3 * src_step);   \
  src[4] = MS_LD256_F32(src_data + 4 * src_step);   \
  src[5] = MS_LD256_F32(src_data + 5 * src_step);   \
  src[6] = MS_LD256_F32(src_data + 6 * src_step);   \
  src[7] = MS_LD256_F32(src_data + 7 * src_step);   \
  src[8] = MS_LD256_F32(src_data + 8 * src_step);   \
  src[9] = MS_LD256_F32(src_data + 9 * src_step);   \
  src[10] = MS_LD256_F32(src_data + 10 * src_step); \
  src[11] = MS_LD256_F32(src_data + 11 * src_step); \
  src[12] = MS_LD256_F32(src_data + 12 * src_step); \
  src[13] = MS_LD256_F32(src_data + 13 * src_step); \
  src[14] = MS_LD256_F32(src_data + 14 * src_step); \
  src[15] = MS_LD256_F32(src_data + 15 * src_step); \
  src[16] = MS_LD256_F32(src_data + 16 * src_step); \
  src[17] = MS_LD256_F32(src_data + 17 * src_step); \
  src[18] = MS_LD256_F32(src_data + 18 * src_step); \
  src[19] = MS_LD256_F32(src_data + 19 * src_step); \
  src[20] = MS_LD256_F32(src_data + 20 * src_step); \
  src[21] = MS_LD256_F32(src_data + 21 * src_step); \
  src[22] = MS_LD256_F32(src_data + 22 * src_step); \
  src[23] = MS_LD256_F32(src_data + 23 * src_step); \
  src[24] = MS_LD256_F32(src_data + 24 * src_step); \
  src[25] = MS_LD256_F32(src_data + 25 * src_step); \
  src[26] = MS_LD256_F32(src_data + 26 * src_step); \
  src[27] = MS_LD256_F32(src_data + 27 * src_step); \
  src[28] = MS_LD256_F32(src_data + 28 * src_step); \
  src[29] = MS_LD256_F32(src_data + 29 * src_step); \
  src[30] = MS_LD256_F32(src_data + 30 * src_step); \
  src[31] = MS_LD256_F32(src_data + 31 * src_step); \
  src[32] = MS_LD256_F32(src_data + 32 * src_step); \
  src[33] = MS_LD256_F32(src_data + 33 * src_step); \
  src[34] = MS_LD256_F32(src_data + 34 * src_step); \
  src[35] = MS_LD256_F32(src_data + 35 * src_step); \
  src[36] = MS_LD256_F32(src_data + 36 * src_step); \
  src[37] = MS_LD256_F32(src_data + 37 * src_step); \
  src[38] = MS_LD256_F32(src_data + 38 * src_step); \
  src[39] = MS_LD256_F32(src_data + 39 * src_step); \
  src[40] = MS_LD256_F32(src_data + 40 * src_step); \
  src[41] = MS_LD256_F32(src_data + 41 * src_step); \
  src[42] = MS_LD256_F32(src_data + 42 * src_step); \
  src[43] = MS_LD256_F32(src_data + 43 * src_step); \
  src[44] = MS_LD256_F32(src_data + 44 * src_step); \
  src[45] = MS_LD256_F32(src_data + 45 * src_step); \
  src[46] = MS_LD256_F32(src_data + 46 * src_step); \
  src[47] = MS_LD256_F32(src_data + 47 * src_step); \
  src[48] = MS_LD256_F32(src_data + 48 * src_step); \
  src[49] = MS_LD256_F32(src_data + 49 * src_step); \
  src[50] = MS_LD256_F32(src_data + 50 * src_step); \
  src[51] = MS_LD256_F32(src_data + 51 * src_step); \
  src[52] = MS_LD256_F32(src_data + 52 * src_step); \
  src[53] = MS_LD256_F32(src_data + 53 * src_step); \
  src[54] = MS_LD256_F32(src_data + 54 * src_step); \
  src[55] = MS_LD256_F32(src_data + 55 * src_step); \
  src[56] = MS_LD256_F32(src_data + 56 * src_step); \
  src[57] = MS_LD256_F32(src_data + 57 * src_step); \
  src[58] = MS_LD256_F32(src_data + 58 * src_step); \
  src[59] = MS_LD256_F32(src_data + 59 * src_step); \
  src[60] = MS_LD256_F32(src_data + 60 * src_step); \
  src[61] = MS_LD256_F32(src_data + 61 * src_step); \
  src[62] = MS_LD256_F32(src_data + 62 * src_step); \
  src[63] = MS_LD256_F32(src_data + 63 * src_step);

#define StoreAvx4Data                              \
  MS_ST256_F32(dst_data, m[0]);                    \
  MS_ST256_F32(dst_data + out_c, m[1]);            \
  MS_ST256_F32(dst_data + dst_step * out_c, m[2]); \
  MS_ST256_F32(dst_data + dst_step * out_c + out_c, m[3]);

#define StoreAvx9Data                                          \
  MS_ST256_F32(dst_data, m[0]);                                \
  MS_ST256_F32(dst_data + out_c, m[1]);                        \
  MS_ST256_F32(dst_data + 2 * out_c, m[2]);                    \
  MS_ST256_F32(dst_data + dst_step * out_c, m[3]);             \
  MS_ST256_F32(dst_data + dst_step * out_c + out_c, m[4]);     \
  MS_ST256_F32(dst_data + dst_step * out_c + 2 * out_c, m[5]); \
  MS_ST256_F32(dst_data + 2 * dst_step * out_c, m[6]);         \
  MS_ST256_F32(dst_data + 2 * dst_step * out_c + out_c, m[7]); \
  MS_ST256_F32(dst_data + 2 * dst_step * out_c + 2 * out_c, m[8]);

#define StoreAvx16Data                                              \
  MS_ST256_F32(dst_data, m[0]);                                     \
  MS_ST256_F32(dst_data + out_c, m[1]);                             \
  MS_ST256_F32(dst_data + 2 * out_c, m[2]);                         \
  MS_ST256_F32(dst_data + 3 * out_c, m[3]);                         \
  MS_ST256_F32(dst_data + dst_step * out_c, m[4]);                  \
  MS_ST256_F32(dst_data + dst_step * out_c + out_c, m[5]);          \
  MS_ST256_F32(dst_data + dst_step * out_c + 2 * out_c, m[6]);      \
  MS_ST256_F32(dst_data + dst_step * out_c + 3 * out_c, m[7]);      \
  MS_ST256_F32(dst_data + 2 * dst_step * out_c, m[8]);              \
  MS_ST256_F32(dst_data + 2 * dst_step * out_c + out_c, m[9]);      \
  MS_ST256_F32(dst_data + 2 * dst_step * out_c + 2 * out_c, m[10]); \
  MS_ST256_F32(dst_data + 2 * dst_step * out_c + 3 * out_c, m[11]); \
  MS_ST256_F32(dst_data + 3 * dst_step * out_c, m[12]);             \
  MS_ST256_F32(dst_data + 3 * dst_step * out_c + out_c, m[13]);     \
  MS_ST256_F32(dst_data + 3 * dst_step * out_c + 2 * out_c, m[14]); \
  MS_ST256_F32(dst_data + 3 * dst_step * out_c + 3 * out_c, m[15]);

#define StoreAvx25Data                                              \
  MS_ST256_F32(dst_data, m[0]);                                     \
  MS_ST256_F32(dst_data + out_c, m[1]);                             \
  MS_ST256_F32(dst_data + 2 * out_c, m[2]);                         \
  MS_ST256_F32(dst_data + 3 * out_c, m[3]);                         \
  MS_ST256_F32(dst_data + 4 * out_c, m[4]);                         \
  MS_ST256_F32(dst_data + dst_step * out_c, m[5]);                  \
  MS_ST256_F32(dst_data + dst_step * out_c + out_c, m[6]);          \
  MS_ST256_F32(dst_data + dst_step * out_c + 2 * out_c, m[7]);      \
  MS_ST256_F32(dst_data + dst_step * out_c + 3 * out_c, m[8]);      \
  MS_ST256_F32(dst_data + dst_step * out_c + 4 * out_c, m[9]);      \
  MS_ST256_F32(dst_data + 2 * dst_step * out_c, m[10]);             \
  MS_ST256_F32(dst_data + 2 * dst_step * out_c + out_c, m[11]);     \
  MS_ST256_F32(dst_data + 2 * dst_step * out_c + 2 * out_c, m[12]); \
  MS_ST256_F32(dst_data + 2 * dst_step * out_c + 3 * out_c, m[13]); \
  MS_ST256_F32(dst_data + 2 * dst_step * out_c + 4 * out_c, m[14]); \
  MS_ST256_F32(dst_data + 3 * dst_step * out_c, m[15]);             \
  MS_ST256_F32(dst_data + 3 * dst_step * out_c + out_c, m[16]);     \
  MS_ST256_F32(dst_data + 3 * dst_step * out_c + 2 * out_c, m[17]); \
  MS_ST256_F32(dst_data + 3 * dst_step * out_c + 3 * out_c, m[18]); \
  MS_ST256_F32(dst_data + 3 * dst_step * out_c + 4 * out_c, m[19]); \
  MS_ST256_F32(dst_data + 4 * dst_step * out_c, m[20]);             \
  MS_ST256_F32(dst_data + 4 * dst_step * out_c + out_c, m[21]);     \
  MS_ST256_F32(dst_data + 4 * dst_step * out_c + 2 * out_c, m[22]); \
  MS_ST256_F32(dst_data + 4 * dst_step * out_c + 3 * out_c, m[23]); \
  MS_ST256_F32(dst_data + 4 * dst_step * out_c + 4 * out_c, m[24]);

void OutputTransform4x2AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform4x2ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform4x2Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform4x3AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform4x3ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform4x3Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c);

void OutputTransform6x2AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x2ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x2Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x3AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x3ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x3Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x4AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x4ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x4Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x5AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x5ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform6x5Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c);

void OutputTransform8x2AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x2ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x2Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x3AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x3ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x3Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x4AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x4ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x4Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x5AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x5ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x5Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x6ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x6Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x7AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                               int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x7ReluAvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                   int dst_step, int out_c, int r_w, int r_h, int r_c);
void OutputTransform8x7Relu6AvxUnit(const float *src_data, float *dst_data, const float *bias_data, int src_step,
                                    int dst_step, int out_c, int r_w, int r_h, int r_c);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_WINOGRAD_AVX_H_
#endif
