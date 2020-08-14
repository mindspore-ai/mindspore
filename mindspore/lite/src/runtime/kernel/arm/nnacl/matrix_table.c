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
#include "nnacl/matrix_table.h"

void MatrixG4x2(float *matrix_data) {
  matrix_data[0] = 1.0f;
  matrix_data[1] = 0.0f;
  matrix_data[2] = 1.0f;
  matrix_data[3] = 0.5f;
  matrix_data[4] = 1.0f;
  matrix_data[5] = -0.5f;
  matrix_data[6] = 0.0f;
  matrix_data[7] = 1.0f;
}

void MatrixGT2x4(float *matrix_data) {
  matrix_data[0] = 1.0f;
  matrix_data[1] = 1.0f;
  matrix_data[2] = 1.0f;
  matrix_data[3] = 0.0f;
  matrix_data[4] = 0.0f;
  matrix_data[5] = 0.5f;
  matrix_data[6] = -0.5f;
  matrix_data[7] = 1.0f;
}

void MatrixG8x2(float *matrix_data) {
  matrix_data[0] = 1.0f;
  matrix_data[1] = 0.0f;
  matrix_data[2] = 1.0f;
  matrix_data[3] = 0.5f;
  matrix_data[4] = 1.0f;
  matrix_data[5] = -0.5f;
  matrix_data[6] = 1.0f;
  matrix_data[7] = 1.0f;
  matrix_data[8] = 1.0f;
  matrix_data[9] = -1.0f;
  matrix_data[10] = 1.0f;
  matrix_data[11] = 1.5f;
  matrix_data[12] = 1.0f;
  matrix_data[13] = -1.5f;
  matrix_data[14] = 0.0f;
  matrix_data[15] = 1.0f;
}

void MatrixGT2x8(float *matrix_data) {
  matrix_data[0] = 1.0f;
  matrix_data[1] = 1.0f;
  matrix_data[2] = 1.0f;
  matrix_data[3] = 1.5f;
  matrix_data[4] = 1.0f;
  matrix_data[5] = 1.0f;
  matrix_data[6] = 1.0f;
  matrix_data[7] = 0.0f;
  matrix_data[8] = 0.0f;
  matrix_data[9] = 0.5f;
  matrix_data[10] = -0.5f;
  matrix_data[11] = 1.0f;
  matrix_data[12] = -1.0f;
  matrix_data[13] = 1.5f;
  matrix_data[14] = -1.5f;
  matrix_data[15] = 1.0f;
}

void MatrixG8x3(float *matrix_data) {
  matrix_data[0] = 1.0f;
  matrix_data[1] = 0.0f;
  matrix_data[2] = 0.0f;
  matrix_data[3] = 1.0f;
  matrix_data[4] = 0.5f;
  matrix_data[5] = 0.25f;
  matrix_data[6] = 1.0f;
  matrix_data[7] = -0.5f;
  matrix_data[8] = 0.25f;
  matrix_data[9] = 1.0f;
  matrix_data[10] = 1.0f;
  matrix_data[11] = 1.0f;
  matrix_data[12] = 1.0f;
  matrix_data[13] = -1.0f;
  matrix_data[14] = 1.0f;
  matrix_data[15] = 1.0f;
  matrix_data[16] = 1.5f;
  matrix_data[17] = 2.25f;
  matrix_data[18] = 1.0f;
  matrix_data[19] = -1.5f;
  matrix_data[20] = 2.25f;
  matrix_data[21] = 0.0f;
  matrix_data[22] = 0.0f;
  matrix_data[23] = 1.0f;
}

void MatrixGT3x8(float *matrix_data) {
  matrix_data[0] = 1.0f;
  matrix_data[1] = 1.0f;
  matrix_data[2] = 1.0f;
  matrix_data[3] = 1.0f;
  matrix_data[4] = 1.0f;
  matrix_data[5] = 1.0f;
  matrix_data[6] = 1.0f;
  matrix_data[7] = 0.0f;
  matrix_data[8] = 0.0f;
  matrix_data[9] = 0.5f;
  matrix_data[10] = -0.5f;
  matrix_data[11] = 1.0f;
  matrix_data[12] = -1.0f;
  matrix_data[13] = 1.5f;
  matrix_data[14] = -1.5f;
  matrix_data[15] = 0.0f;
  matrix_data[16] = 0.0f;
  matrix_data[17] = 0.25f;
  matrix_data[18] = 0.25f;
  matrix_data[19] = 1.0f;
  matrix_data[20] = 1.0f;
  matrix_data[21] = 2.25f;
  matrix_data[22] = 2.25f;
  matrix_data[23] = 1.0f;
}

void MatrixG8x4(float *matrix_data) {
  matrix_data[0] = 1.0f;
  matrix_data[1] = 0.0f;
  matrix_data[2] = 0.0f;
  matrix_data[3] = 0.0f;
  matrix_data[4] = 1.0f;
  matrix_data[5] = 0.5f;
  matrix_data[6] = 0.25f;
  matrix_data[7] = 0.125f;
  matrix_data[8] = 1.0f;
  matrix_data[9] = -0.5f;
  matrix_data[10] = 0.25f;
  matrix_data[11] = -0.125f;
  matrix_data[12] = 1.0f;
  matrix_data[13] = 1.0f;
  matrix_data[14] = 1.0f;
  matrix_data[15] = 1.0f;
  matrix_data[16] = 1.0f;
  matrix_data[17] = -1.0f;
  matrix_data[18] = 1.0f;
  matrix_data[19] = -1.0f;
  matrix_data[20] = 1.0f;
  matrix_data[21] = 1.5f;
  matrix_data[22] = 2.25f;
  matrix_data[23] = 3.375f;
  matrix_data[24] = 1.0f;
  matrix_data[25] = -1.5f;
  matrix_data[26] = 2.25f;
  matrix_data[27] = -3.375f;
  matrix_data[28] = 0.0f;
  matrix_data[29] = 0.0f;
  matrix_data[30] = 0.0f;
  matrix_data[31] = 1.0f;
}

void MatrixGT4x8(float *matrix_data) {
  matrix_data[0] = 1.0f;
  matrix_data[1] = 1.0f;
  matrix_data[2] = 1.0f;
  matrix_data[3] = 1.0f;
  matrix_data[4] = 1.0f;
  matrix_data[5] = 1.0f;
  matrix_data[6] = 1.0f;
  matrix_data[7] = 0.0f;
  matrix_data[8] = 0.0f;
  matrix_data[9] = 0.5f;
  matrix_data[10] = -0.5f;
  matrix_data[11] = 1.0f;
  matrix_data[12] = -1.0f;
  matrix_data[13] = 1.5f;
  matrix_data[14] = -1.5f;
  matrix_data[15] = 0.0f;
  matrix_data[16] = 0.0f;
  matrix_data[17] = 0.25f;
  matrix_data[18] = 0.25f;
  matrix_data[19] = 1.0f;
  matrix_data[20] = 1.0f;
  matrix_data[21] = 2.25f;
  matrix_data[22] = 2.25f;
  matrix_data[23] = 0.0f;
  matrix_data[24] = 0.0f;
  matrix_data[25] = 0.125f;
  matrix_data[26] = -0.125f;
  matrix_data[27] = 1.0f;
  matrix_data[28] = -1.0f;
  matrix_data[29] = 3.375f;
  matrix_data[30] = -3.375f;
  matrix_data[31] = 1.0f;
}

void MatrixG8x5(float *matrix_data) {
  matrix_data[0] = 1.0f;
  matrix_data[1] = 0.0f;
  matrix_data[2] = 0.0f;
  matrix_data[3] = 0.0f;
  matrix_data[4] = 0.0f;
  matrix_data[5] = 1.0f;
  matrix_data[6] = 0.5f;
  matrix_data[7] = 0.25f;
  matrix_data[8] = 0.125f;
  matrix_data[9] = 0.0625f;
  matrix_data[10] = 1.0f;
  matrix_data[11] = -0.5f;
  matrix_data[12] = 0.25f;
  matrix_data[13] = -0.125f;
  matrix_data[14] = 0.0625f;
  matrix_data[15] = 1.0f;
  matrix_data[16] = 1.0f;
  matrix_data[17] = 1.0f;
  matrix_data[18] = 1.0f;
  matrix_data[19] = 1.0f;
  matrix_data[20] = 1.0f;
  matrix_data[21] = -1.0f;
  matrix_data[22] = 1.0f;
  matrix_data[23] = -1.0f;
  matrix_data[24] = 1.0f;
  matrix_data[25] = 1.0f;
  matrix_data[26] = 1.5f;
  matrix_data[27] = 2.25f;
  matrix_data[28] = 3.375f;
  matrix_data[29] = 5.0625f;
  matrix_data[30] = 1.0f;
  matrix_data[31] = -1.5f;
  matrix_data[32] = 2.25f;
  matrix_data[33] = -3.375f;
  matrix_data[34] = 5.0625f;
  matrix_data[35] = 0.0f;
  matrix_data[36] = 0.0f;
  matrix_data[37] = 0.0f;
  matrix_data[38] = 0.0f;
  matrix_data[39] = 1.0f;
}

void MatrixGT5x8(float *matrix_data) {
  matrix_data[0] = 1.0f;
  matrix_data[1] = 1.0f;
  matrix_data[2] = 1.0f;
  matrix_data[3] = 1.0f;
  matrix_data[4] = 1.0f;
  matrix_data[5] = 1.0f;
  matrix_data[6] = 1.0f;
  matrix_data[7] = 0.0f;
  matrix_data[8] = 0.0f;
  matrix_data[9] = 0.5f;
  matrix_data[10] = -0.5f;
  matrix_data[11] = 1.0f;
  matrix_data[12] = -1.0f;
  matrix_data[13] = 1.5f;
  matrix_data[14] = -1.5f;
  matrix_data[15] = 0.0f;
  matrix_data[16] = 0.0f;
  matrix_data[17] = 0.25f;
  matrix_data[18] = 0.25f;
  matrix_data[19] = 1.0f;
  matrix_data[20] = 1.0f;
  matrix_data[21] = 2.25f;
  matrix_data[22] = 2.25f;
  matrix_data[23] = 0.0f;
  matrix_data[24] = 0.0f;
  matrix_data[25] = 0.125f;
  matrix_data[26] = -0.125f;
  matrix_data[27] = 1.0f;
  matrix_data[28] = -1.0f;
  matrix_data[29] = 3.375f;
  matrix_data[30] = -3.375f;
  matrix_data[31] = 0.0f;
  matrix_data[32] = 0.0f;
  matrix_data[33] = 0.0625f;
  matrix_data[34] = 0.0625f;
  matrix_data[35] = 1.0f;
  matrix_data[36] = 1.0f;
  matrix_data[37] = 5.0625f;
  matrix_data[38] = 5.0625f;
  matrix_data[39] = 1.0f;
}

void MatrixG8x6(float *matrix_data) {
  matrix_data[0] = 1.0f;
  matrix_data[1] = 0.0f;
  matrix_data[2] = 0.0f;
  matrix_data[3] = 0.0f;
  matrix_data[4] = 0.0f;
  matrix_data[5] = 0.0f;
  matrix_data[6] = 1.0f;
  matrix_data[7] = 0.5f;
  matrix_data[8] = 0.25f;
  matrix_data[9] = 0.125f;
  matrix_data[10] = 0.0625f;
  matrix_data[11] = 0.03125f;
  matrix_data[12] = 1.0f;
  matrix_data[13] = -0.5f;
  matrix_data[14] = 0.25f;
  matrix_data[15] = -0.125f;
  matrix_data[16] = 0.0625f;
  matrix_data[17] = -0.03125f;
  matrix_data[18] = 1.0f;
  matrix_data[19] = 1.0f;
  matrix_data[20] = 1.0f;
  matrix_data[21] = 1.0f;
  matrix_data[22] = 1.0f;
  matrix_data[23] = 1.0f;
  matrix_data[24] = 1.0f;
  matrix_data[25] = -1.0f;
  matrix_data[26] = 1.0f;
  matrix_data[27] = -1.0f;
  matrix_data[28] = 1.0f;
  matrix_data[29] = -1.0f;
  matrix_data[30] = 1.0f;
  matrix_data[31] = 1.5f;
  matrix_data[32] = 2.25f;
  matrix_data[33] = 3.375f;
  matrix_data[34] = 5.0625f;
  matrix_data[35] = 7.59375f;
  matrix_data[36] = 1.0f;
  matrix_data[37] = -1.5f;
  matrix_data[38] = 2.25f;
  matrix_data[39] = -3.375f;
  matrix_data[40] = 5.0625f;
  matrix_data[41] = -7.59375f;
  matrix_data[42] = 0.0f;
  matrix_data[43] = 0.0f;
  matrix_data[44] = 0.0f;
  matrix_data[45] = 0.0f;
  matrix_data[46] = 0.0f;
  matrix_data[47] = 1.0f;
}

void MatrixGT6x8(float *matrix_data) {
  matrix_data[0] = 1.0f;
  matrix_data[1] = 1.0f;
  matrix_data[2] = 1.0f;
  matrix_data[3] = 1.0f;
  matrix_data[4] = 1.0f;
  matrix_data[5] = 1.0f;
  matrix_data[6] = 1.0f;
  matrix_data[7] = 0.0f;
  matrix_data[8] = 0.0f;
  matrix_data[9] = 0.5f;
  matrix_data[10] = -0.5f;
  matrix_data[11] = 1.0f;
  matrix_data[12] = -1.0f;
  matrix_data[13] = 1.5f;
  matrix_data[14] = -1.5f;
  matrix_data[15] = 0.0f;
  matrix_data[16] = 0.0f;
  matrix_data[17] = 0.25f;
  matrix_data[18] = 0.25f;
  matrix_data[19] = 1.0f;
  matrix_data[20] = 1.0f;
  matrix_data[21] = 2.25f;
  matrix_data[22] = 2.25f;
  matrix_data[23] = 0.0f;
  matrix_data[24] = 0.0f;
  matrix_data[25] = 0.125f;
  matrix_data[26] = -0.125f;
  matrix_data[27] = 1.0f;
  matrix_data[28] = -1.0f;
  matrix_data[29] = 3.375f;
  matrix_data[30] = -3.375f;
  matrix_data[31] = 0.0f;
  matrix_data[32] = 0.0f;
  matrix_data[33] = 0.0625f;
  matrix_data[34] = 0.0625f;
  matrix_data[35] = 1.0f;
  matrix_data[36] = 1.0f;
  matrix_data[37] = 5.0625f;
  matrix_data[38] = 5.0625f;
  matrix_data[39] = 0.0f;
  matrix_data[40] = 0.0;
  matrix_data[41] = 0.03125f;
  matrix_data[42] = -0.03125f;
  matrix_data[43] = 1.0f;
  matrix_data[44] = -1.0f;
  matrix_data[45] = 7.59375f;
  matrix_data[46] = -7.59375f;
  matrix_data[47] = 0.0f;
  matrix_data[48] = 1.0f;
}

void MatrixG8x7(float *matrix_data) {
  matrix_data[0] = 1.0f;
  matrix_data[1] = 0.0f;
  matrix_data[2] = 0.0f;
  matrix_data[3] = 0.0f;
  matrix_data[4] = 0.0f;
  matrix_data[5] = 0.0f;
  matrix_data[6] = 0.0f;
  matrix_data[7] = 1.0f;
  matrix_data[8] = 0.5f;
  matrix_data[9] = 0.25f;
  matrix_data[10] = 0.125f;
  matrix_data[11] = 0.0625f;
  matrix_data[12] = 0.03125f;
  matrix_data[13] = 0.015625f;
  matrix_data[14] = 1.0f;
  matrix_data[15] = -0.5f;
  matrix_data[16] = 0.25f;
  matrix_data[17] = -0.125f;
  matrix_data[18] = 0.0625f;
  matrix_data[19] = -0.03125f;
  matrix_data[20] = 0.015625f;
  matrix_data[21] = 1.0f;
  matrix_data[22] = 1.0f;
  matrix_data[23] = 1.0f;
  matrix_data[24] = 1.0f;
  matrix_data[25] = 1.0f;
  matrix_data[26] = 1.0f;
  matrix_data[27] = 1.0f;
  matrix_data[28] = 1.0f;
  matrix_data[29] = -1.0f;
  matrix_data[30] = 1.0f;
  matrix_data[31] = -1.0f;
  matrix_data[32] = 1.0f;
  matrix_data[33] = -1.0f;
  matrix_data[34] = 1.0f;
  matrix_data[35] = 1.0f;
  matrix_data[36] = 1.5f;
  matrix_data[37] = 2.25f;
  matrix_data[38] = 3.375f;
  matrix_data[39] = 5.0625f;
  matrix_data[40] = 7.59375f;
  matrix_data[41] = 11.390625f;
  matrix_data[42] = 1.0f;
  matrix_data[43] = -1.5f;
  matrix_data[44] = 2.25f;
  matrix_data[45] = -3.375f;
  matrix_data[46] = 5.0625f;
  matrix_data[47] = -7.59375f;
  matrix_data[48] = 11.390625f;
  matrix_data[49] = 0.0f;
  matrix_data[50] = 0.0f;
  matrix_data[51] = 0.0f;
  matrix_data[52] = 0.0f;
  matrix_data[53] = 0.0f;
  matrix_data[54] = 0.0f;
  matrix_data[55] = 1.0f;
}

void MatrixGT7x8(float *matrix_data) {
  matrix_data[0] = 1.0f;
  matrix_data[1] = 1.0f;
  matrix_data[2] = 1.0f;
  matrix_data[3] = 1.0f;
  matrix_data[4] = 1.0f;
  matrix_data[5] = 1.0f;
  matrix_data[6] = 1.0f;
  matrix_data[7] = 0.0f;
  matrix_data[8] = 0.0f;
  matrix_data[9] = 0.5f;
  matrix_data[10] = -0.5f;
  matrix_data[11] = 1.0f;
  matrix_data[12] = -1.0f;
  matrix_data[13] = 1.5f;
  matrix_data[14] = -1.5f;
  matrix_data[15] = 0.0f;
  matrix_data[16] = 0.0f;
  matrix_data[17] = 0.25f;
  matrix_data[18] = 0.25f;
  matrix_data[19] = 1.0f;
  matrix_data[20] = 1.0f;
  matrix_data[21] = 2.25f;
  matrix_data[22] = 2.25f;
  matrix_data[23] = 0.0f;
  matrix_data[24] = 0.0f;
  matrix_data[25] = 0.125f;
  matrix_data[26] = -0.125f;
  matrix_data[27] = 1.0f;
  matrix_data[28] = -1.0f;
  matrix_data[29] = 3.375f;
  matrix_data[30] = -3.375f;
  matrix_data[31] = 0.0f;
  matrix_data[32] = 0.0f;
  matrix_data[33] = 0.0625f;
  matrix_data[34] = 0.0625f;
  matrix_data[35] = 1.0f;
  matrix_data[36] = 1.0f;
  matrix_data[37] = 5.0625f;
  matrix_data[38] = 5.0625f;
  matrix_data[39] = 0.0f;
  matrix_data[40] = 0.0;
  matrix_data[41] = 0.03125f;
  matrix_data[42] = -0.03125f;
  matrix_data[43] = 1.0f;
  matrix_data[44] = -1.0f;
  matrix_data[45] = 7.59375f;
  matrix_data[46] = -7.59375f;
  matrix_data[47] = 0.0f;
  matrix_data[48] = 0.0f;
  matrix_data[49] = 0.015625f;
  matrix_data[50] = 0.015625f;
  matrix_data[51] = 1.0f;
  matrix_data[52] = 1.0f;
  matrix_data[53] = 11.390625f;
  matrix_data[54] = 11.390625f;
  matrix_data[55] = 1.0f;
}
