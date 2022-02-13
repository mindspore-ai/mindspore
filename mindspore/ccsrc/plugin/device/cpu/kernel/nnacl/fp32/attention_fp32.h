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

#ifndef MINDSPORE_NNACL_FP32_ATTENTION_FP32_H_
#define MINDSPORE_NNACL_FP32_ATTENTION_FP32_H_

#include "nnacl/attention_parameter.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct Matrix {
  float *data_;
  int row_;
  int col_;
  float *packed_data_;
  int packed_row_;
  int packed_col_;
  int batch_;
  bool is_transpose_;
} Matrix;

int InitMatrix(Matrix *matrix, int batch, int row, int col, bool is_trans);

size_t LeftMatrixPackElementSize(Matrix *matrix, int row_tile);

size_t RightMatrixPackElementSize(Matrix *matrix, int col_tile);

int PackLeftMatrix(Matrix *matrix, int row_tile);

int PackRightMatrix(Matrix *matrix, int col_tile);

int PackAttentionBias(Matrix *matrix, int tile);

void QWithPosition(RelativePositionAttentionParameter *param, Matrix *q_mat, const Matrix *wq_mat, Matrix *bq_mat,
                   Matrix *q2wq_mat, Matrix *pu_mat, Matrix *pv_mat, Matrix *q2wq_with_pos_mat,
                   Matrix *q2wq_with_pu_trans_mat, Matrix *q2wq_with_pv_trans_mat);

void KMulWeightK(RelativePositionAttentionParameter *param, Matrix *k_mat, const Matrix *wk_mat, Matrix *bk_mat,
                 Matrix *k2wk_mat, Matrix *k2wk_trans_mat);

void VMulWeightV(RelativePositionAttentionParameter *param, Matrix *v_mat, const Matrix *wv_mat, Matrix *bv_mat,
                 Matrix *v2wv_mat, Matrix *v2wv_trans_mat);

void PMulWeightP(RelativePositionAttentionParameter *param, Matrix *p_mat, const Matrix *wp_mat, Matrix *p2wp_mat,
                 Matrix *p2wp_trans_mat);

void CalculateLogits(RelativePositionAttentionParameter *param, Matrix *q2wq_with_pu_trans_mat,
                     Matrix *q2wq_with_pv_trans_mat, Matrix *k2wk_trans_mat, Matrix *p2wp_trans_mat,
                     Matrix *logits_with_u_mat, Matrix *logits_with_v_mat, Matrix *logits_with_v_pad_mat,
                     Matrix *logits_with_v_shifted_mat, Matrix *logits_mat);

void RelPosAttention(RelativePositionAttentionParameter *param, Matrix *logits_mat, Matrix *softmax_mat,
                     Matrix *v2wv_trans_mat, Matrix *logits2v_mat, Matrix *logits2v_trans_mat, const Matrix *wo_mat,
                     Matrix *bo_mat, Matrix *output_mat);
#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_NNACL_FP32_ATTENTION_FP32_H_
