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

#include "nnacl/fp32/attention_fp32.h"
#include <string.h>
#include <math.h>
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/fp32/add_fp32.h"
#include "nnacl/fp32/transpose_fp32.h"
#include "nnacl/fp32/softmax_fp32.h"
#include "nnacl/errorcode.h"

int InitMatrix(Matrix *matrix, int batch, int row, int col, bool is_trans) {
  if (matrix == NULL) {
    return NNACL_NULL_PTR;
  }
  matrix->batch_ = batch;
  matrix->row_ = row;
  matrix->col_ = col;
  matrix->is_transpose_ = is_trans;
  matrix->data_ = NULL;
  matrix->packed_data_ = NULL;
  return NNACL_OK;
}

size_t LeftMatrixPackElementSize(Matrix *matrix, int row_tile) {
  if (matrix == NULL) {
    return 0;
  }
  int real_row = matrix->is_transpose_ ? matrix->col_ : matrix->row_;
  int deep = matrix->is_transpose_ ? matrix->row_ : matrix->col_;
  bool vec_matmul = real_row == 1;
  int row_align = vec_matmul ? 1 : UP_ROUND(real_row, row_tile);
  int dst_area = row_align * deep;
  matrix->packed_row_ = row_align;
  matrix->packed_col_ = deep;
  return matrix->batch_ * dst_area;
}

size_t RightMatrixPackElementSize(Matrix *matrix, int col_tile) {
  if (matrix == NULL) {
    return 0;
  }
  int deep = matrix->is_transpose_ ? matrix->col_ : matrix->row_;
  int real_col = matrix->is_transpose_ ? matrix->row_ : matrix->col_;
  bool vec_matmul = deep == 1;
  int col_align = vec_matmul ? real_col : UP_ROUND(real_col, col_tile);
  int dst_area = deep * col_align;
  matrix->packed_row_ = deep;
  matrix->packed_col_ = col_align;
  return matrix->batch_ * dst_area;
}

int PackLeftMatrix(Matrix *matrix, int row_tile) {
  if (matrix == NULL || matrix->data_ == NULL || row_tile == 0) {
    return NNACL_NULL_PTR;
  }
  int real_row = matrix->is_transpose_ ? matrix->col_ : matrix->row_;
  int deep = matrix->is_transpose_ ? matrix->row_ : matrix->col_;
  bool vec_matmul = real_row == 1;
  int row_align = vec_matmul ? 1 : UP_ROUND(real_row, row_tile);
  int src_area = matrix->row_ * matrix->col_;
  int dst_area = row_align * deep;
  bool malloced = false;
  if (matrix->packed_data_ == NULL) {
    matrix->packed_data_ = (float *)malloc(dst_area * matrix->batch_ * sizeof(float));
    if (matrix->packed_data_ == NULL) {
      return NNACL_NULL_PTR;
    }
    malloced = true;
  }

  if (vec_matmul) {
    memcpy(matrix->packed_data_, matrix->data_, matrix->batch_ * dst_area * sizeof(float));
  } else {
    for (int i = 0; i < matrix->batch_; i++) {
      const float *cur_src = matrix->data_ + i * src_area;
      float *cur_dst = matrix->packed_data_ + i * dst_area;
      switch (row_tile) {
        case C6NUM:
          if (matrix->is_transpose_) {
            RowMajor2Row6Major(cur_src, cur_dst, real_row, deep);
          } else {
            RowMajor2Col6Major(cur_src, cur_dst, real_row, deep);
          }
          break;
        case C4NUM:
          if (matrix->is_transpose_) {
            RowMajor2Row4Major(cur_src, cur_dst, real_row, deep);
          } else {
            RowMajor2Col4Major(cur_src, cur_dst, real_row, deep);
          }
          break;
        case C12NUM:
          if (matrix->is_transpose_) {
            RowMajor2Row12Major(cur_src, cur_dst, real_row, deep);
          } else {
            RowMajor2Col12Major(cur_src, cur_dst, real_row, deep);
          }
          break;
        default:
          if (malloced) {
            free(matrix->packed_data_);
            matrix->packed_data_ = NULL;
            return NNACL_ERR;
          }
          break;
      }
    }
  }
  matrix->packed_row_ = row_align;
  matrix->packed_col_ = deep;
  return NNACL_OK;
}

int PackRightMatrix(Matrix *matrix, int col_tile) {
  if (matrix == NULL || matrix->data_ == NULL || col_tile == 0) {
    return NNACL_NULL_PTR;
  }
  int deep = matrix->is_transpose_ ? matrix->col_ : matrix->row_;
  int real_col = matrix->is_transpose_ ? matrix->row_ : matrix->col_;
  bool vec_matmul = deep == 1;
  int col_align = vec_matmul ? real_col : UP_ROUND(real_col, col_tile);
  int src_area = matrix->row_ * matrix->col_;
  int dst_area = deep * col_align;
  bool malloced = false;
  if (matrix->packed_data_ == NULL) {
    matrix->packed_data_ = (float *)malloc(dst_area * matrix->batch_ * sizeof(float));
    if (matrix->packed_data_ == NULL) {
      return NNACL_NULL_PTR;
    }
    malloced = true;
  }
  if (vec_matmul) {
    memcpy(matrix->packed_data_, matrix->data_, matrix->batch_ * dst_area * sizeof(float));
  } else {
    for (int i = 0; i < matrix->batch_; i++) {
      const float *cur_src = matrix->data_ + i * src_area;
      float *cur_dst = matrix->packed_data_ + i * dst_area;
      switch (col_tile) {
        case C16NUM:
          if (matrix->is_transpose_) {
            RowMajor2Col16Major(cur_src, cur_dst, deep, real_col);
          } else {
            RowMajor2Row16Major(cur_src, cur_dst, deep, real_col);
          }
          break;
        case C4NUM:
          if (matrix->is_transpose_) {
            RowMajor2Col4Major(cur_src, cur_dst, deep, real_col);
          } else {
            RowMajor2Row4Major(cur_src, cur_dst, deep, real_col);
          }
          break;
        case C8NUM:
          if (matrix->is_transpose_) {
            RowMajor2Col8Major(cur_src, cur_dst, deep, real_col);
          } else {
            RowMajor2Row8Major(cur_src, cur_dst, deep, real_col);
          }
          break;
        default:
          if (malloced) {
            free(matrix->packed_data_);
            matrix->packed_data_ = NULL;
            return NNACL_ERR;
          }
          break;
      }
    }
  }
  matrix->packed_row_ = deep;
  matrix->packed_col_ = col_align;
  return NNACL_OK;
}

int PackAttentionBias(Matrix *matrix, int tile) {
  if (matrix == NULL || matrix->batch_ != 1 || matrix->row_ != 1 || matrix->data_ == NULL) {
    return NNACL_PARAM_INVALID;
  }
  if (tile == 0) {
    return NNACL_OK;
  }
  int size = matrix->col_;
  float *src = matrix->data_;
  int size_align = UP_ROUND(size, tile);
  if (size_align <= 0) {
    return NNACL_ERR;
  }
  matrix->packed_data_ = (float *)malloc(size_align * sizeof(float));
  if (matrix->packed_data_ == NULL) {
    return NNACL_NULL_PTR;
  }
  matrix->packed_row_ = matrix->row_;
  matrix->packed_col_ = size_align;
  memset(matrix->packed_data_, 0, size_align * sizeof(float));
  memcpy(matrix->packed_data_, src, size * sizeof(float));
  return NNACL_OK;
}

static void RelativeShiftPad(const float *input_data, float *output_data, const int *input_shape, int tid,
                             int thread_num) {
  int row = input_shape[0];
  int col = input_shape[1];
  int out_area = row * (col + 1);
  memset(output_data, 0, out_area * sizeof(float));
  for (int r = tid; r < row; r += thread_num) {
    float *dst = output_data + r * (col + 1);
    const float *src = input_data + r * col;
    memcpy(dst, src, col * sizeof(float));
  }
  int tile = row % thread_num;
  for (int r = row - tile; r < row; r++) {
    float *dst = output_data + r * (col + 1);
    const float *src = input_data + r * col;
    memcpy(dst, src, col * sizeof(float));
  }
}

static void RelativeShiftSlice(const float *input_data, float *output_data, const int *input_shape, int tid,
                               int thread_num) {
  int row = input_shape[0];
  int col = input_shape[1];
  int begin = row;
  memset(output_data, 0, row * row * sizeof(float));
  for (int r = tid; r < row; r += thread_num) {
    float *dst = output_data + r * row;
    const float *src = input_data + r * col + begin;
    memcpy(dst, src, (col / 2) * sizeof(float));
  }
  int tile = row % thread_num;
  for (int r = row - tile; r < row; r++) {
    float *dst = output_data + r * row;
    const float *src = input_data + r * col + begin;
    memcpy(dst, src, (col / 2) * sizeof(float));
  }
}

static void RelativeShift(const Matrix *x, float *pad_buf, float *slice_buf) {
  int x_area = x->row_ * x->col_;
  int pad_area = x->row_ * (x->col_ + 1);
  int slice_area = x->row_ * (x->col_ / 2);
  int input_shape[] = {x->row_, x->col_};
  memset(slice_buf, 0, x->batch_ * x->row_ * (x->col_ / 2) * sizeof(float));
  for (int i = 0; i < x->batch_; i++) {
    float *cur_x_data = x->data_ + i * x_area;
    memset(pad_buf, 0, pad_area * sizeof(float));
    // pad: [row, col + 1]
    RelativeShiftPad(cur_x_data, pad_buf, input_shape, 0, 1);
    // reshape: [col + 1, row]
    // slice last row: [col, row]
    // reshape: [row, col]
    // slice col -> [row, row + col / 2]: [row, col / 2]
    float *cur_slice_data = slice_buf + i * slice_area;
    RelativeShiftSlice(pad_buf, cur_slice_data, input_shape, 0, 1);
  }
}

static void ElementOptAddDiv(const float *input0, const float *input1, const float input2, float *output,
                             const int batch, const int area) {
  int index = 0;
  const float mul = 1 / input2;
  for (int b = 0; b < batch; b++) {
    const float *cur_input0 = input0 + b * area;
    const float *cur_input1 = input1 + b * area;
    float *cur_output = output + b * area;
#ifdef ENABLE_NEON
    for (; index <= area - 4; index += C4NUM) {
      float32x4_t vin0 = vld1q_f32(cur_input0 + index);
      float32x4_t vin1 = vld1q_f32(cur_input1 + index);
      float32x4_t vout = vaddq_f32(vin0, vin1);
      vout = vmulq_n_f32(vout, mul);
      vst1q_f32(cur_output + index, vout);
    }
#endif
    for (; index < area; index++) {
      cur_output[index] += (cur_input0[index] + cur_input1[index]) * mul;
    }
  }
}

static bool GetTransposeParameter(TransposeParameter *param, const int in_shape[], int in_shape_len,
                                  const int out_shape[], int out_shape_len, const int perm[], int perm_len) {
  param->num_axes_ = perm_len;
  size_t shape_size = 1;
  for (int i = 0; i < perm_len; i++) {
    param->perm_[i] = perm[i];
    shape_size *= perm[i];  // check overflow
  }
  param->data_num_ = (int)shape_size;  // check overflow
  param->strides_[param->num_axes_ - 1] = 1;
  param->out_strides_[param->num_axes_ - 1] = 1;
  if (param->num_axes_ - 1 >= in_shape_len) {
    return false;
  }
  if (param->num_axes_ - 1 >= out_shape_len) {
    return false;
  }
  for (int i = param->num_axes_ - 2; i >= 0; i--) {
    param->strides_[i] = in_shape[i + 1] * param->strides_[i + 1];
    param->out_strides_[i] = out_shape[i + 1] * param->out_strides_[i + 1];
  }
  return true;
}

void QWithPosition(RelativePositionAttentionParameter *param, Matrix *q_mat, const Matrix *wq_mat, Matrix *bq_mat,
                   Matrix *q2wq_mat, Matrix *pu_mat, Matrix *pv_mat, Matrix *q2wq_with_pos_mat,
                   Matrix *q2wq_with_pu_trans_mat, Matrix *q2wq_with_pv_trans_mat) {
  int num_heads = param->num_heads_;
  int d_model = param->d_model_;
  int batch = param->batch_;
  int depth = d_model / num_heads;
  // Q * WQ
  int q_area = q_mat->packed_row_ * q_mat->packed_col_;
  int wq_area = wq_mat->packed_row_ * wq_mat->packed_col_;
  int q2wq_area = q2wq_mat->row_ * q2wq_mat->col_ * q2wq_mat->batch_ / param->batch_;
  float *q2wq_data = q2wq_mat->data_;
  memset(q2wq_data, 0, param->batch_ * q2wq_area * sizeof(float));
  for (int i = 0; i < param->batch_; i++) {
    float *cur_q = q_mat->packed_data_ + i * q_area;
    float *cur_wq = wq_mat->packed_data_ + i * wq_area;
    float *cur_q2wq = q2wq_data + i * q2wq_area;
    MatMulOpt(cur_q, cur_wq, cur_q2wq, bq_mat->packed_data_, ActType_No, q_mat->col_, q_mat->row_, wq_mat->col_,
              wq_mat->col_, OutType_Nhwc);
  }
  // transpose param init
  TransposeParameter q_with_pos_trans_param;
  int q_with_pos_trans_in_shape[] = {batch, param->q_seq_, num_heads, depth};
  int q_with_pos_trans_out_shape[] = {batch, num_heads, param->q_seq_, depth};
  int q_with_pos_perm[] = {0, 2, 1, 3};
  (void)GetTransposeParameter(&q_with_pos_trans_param, q_with_pos_trans_in_shape, 4, q_with_pos_trans_out_shape, 4,
                              q_with_pos_perm, 4);
  int q2wq_reshaped_area = q2wq_mat->row_ * q2wq_mat->col_;
  // Q_WQ + POS_U
  {
    float *q_with_pu = q2wq_with_pos_mat->data_;
    int q_with_pu_area = q2wq_with_pos_mat->row_ * q2wq_with_pos_mat->col_;
    memset(q_with_pu, 0, q2wq_with_pos_mat->batch_ * q_with_pu_area * sizeof(float));
    for (int i = 0; i < q2wq_with_pos_mat->batch_; i++) {
      float *cur_qw = q2wq_data + i * q2wq_reshaped_area;
      float *cur_q_with_pu = q_with_pu + i * q_with_pu_area;
      ElementAdd(cur_qw, pu_mat->packed_data_, cur_q_with_pu, q_with_pu_area);
    }
    // Q_WITH_U perm [0,2,1,3]
    float *q_with_pu_trans = q2wq_with_pu_trans_mat->data_;
    size_t q_with_pu_trans_data_size = (size_t)(q2wq_with_pu_trans_mat->batch_) *
                                       (size_t)(q2wq_with_pu_trans_mat->row_) * (size_t)(q2wq_with_pu_trans_mat->col_) *
                                       sizeof(float);
    memset(q_with_pu_trans, 0, q_with_pu_trans_data_size);
    TransposeDimsFp32(q_with_pu, q_with_pu_trans, q_with_pos_trans_out_shape, &q_with_pos_trans_param, 0, 1);
  }

  // Q_WQ + POS_V
  {
    float *q_with_pv = q2wq_with_pos_mat->data_;
    int q_with_pv_area = q2wq_with_pos_mat->row_ * q2wq_with_pos_mat->col_;
    memset(q_with_pv, 0, q2wq_with_pos_mat->batch_ * q_with_pv_area * sizeof(float));
    for (int i = 0; i < q2wq_with_pos_mat->batch_; i++) {
      float *cur_qw = q2wq_data + i * q2wq_reshaped_area;
      float *cur_q_with_pv = q_with_pv + i * q_with_pv_area;
      ElementAdd(cur_qw, pv_mat->packed_data_, cur_q_with_pv, q_with_pv_area);
    }
    // Q_WITH_V perm [0,2,1,3]
    float *q_with_pv_trans = q2wq_with_pv_trans_mat->data_;
    size_t q_with_pv_trans_data_size = (size_t)(q2wq_with_pv_trans_mat->batch_) *
                                       (size_t)(q2wq_with_pv_trans_mat->row_) * (size_t)(q2wq_with_pv_trans_mat->col_) *
                                       sizeof(float);
    memset(q_with_pv_trans, 0, q_with_pv_trans_data_size);
    TransposeDimsFp32(q_with_pv, q_with_pv_trans, q_with_pos_trans_out_shape, &q_with_pos_trans_param, 0, 1);
  }
}

void KMulWeightK(RelativePositionAttentionParameter *param, Matrix *k_mat, const Matrix *wk_mat, Matrix *bk_mat,
                 Matrix *k2wk_mat, Matrix *k2wk_trans_mat) {
  int num_heads = param->num_heads_;
  int d_model = param->d_model_;
  int batch = param->batch_;
  int depth = d_model / num_heads;
  // K * WK
  int k_area = k_mat->packed_row_ * k_mat->packed_col_;
  int wk_area = wk_mat->packed_row_ * wk_mat->packed_col_;
  int k2wk_area = k2wk_mat->row_ * k2wk_mat->col_ * k2wk_mat->batch_ / param->batch_;
  float *k2wk = k2wk_mat->data_;
  memset(k2wk, 0, param->batch_ * k2wk_area * sizeof(float));
  for (int i = 0; i < param->batch_; i++) {
    float *cur_k = k_mat->packed_data_ + i * k_area;
    float *cur_wk = wk_mat->packed_data_ + i * wk_area;
    float *cur_k2wk = k2wk + i * k2wk_area;
    MatMulOpt(cur_k, cur_wk, cur_k2wk, bk_mat->packed_data_, ActType_No, k_mat->col_, k_mat->row_, wk_mat->col_,
              wk_mat->col_, OutType_Nhwc);
  }
  // K * WK perm [0,2,3,1]
  float *k2wk_trans_data = k2wk_trans_mat->data_;
  int k2wk_trans_area = k2wk_trans_mat->row_ * k2wk_trans_mat->col_;
  memset(k2wk_trans_data, 0, k2wk_trans_mat->batch_ * k2wk_trans_area * sizeof(float));
  TransposeParameter k2wk_trans_param;
  int k2wk_in_shape[] = {batch, param->k_seq_, num_heads, depth};
  int k2wk_out_shape[] = {batch, num_heads, depth, param->k_seq_};
  int k2wk_perm[] = {0, 2, 3, 1};
  (void)GetTransposeParameter(&k2wk_trans_param, k2wk_in_shape, 4, k2wk_out_shape, 4, k2wk_perm, 4);
  TransposeDimsFp32(k2wk, k2wk_trans_data, k2wk_out_shape, &k2wk_trans_param, 0, 1);
}

void VMulWeightV(RelativePositionAttentionParameter *param, Matrix *v_mat, const Matrix *wv_mat, Matrix *bv_mat,
                 Matrix *v2wv_mat, Matrix *v2wv_trans_mat) {
  int num_heads = param->num_heads_;
  int d_model = param->d_model_;
  int batch = param->batch_;
  int depth = d_model / num_heads;
  // V * WV
  int v_area = v_mat->packed_row_ * v_mat->packed_col_;
  int wv_area = wv_mat->packed_row_ * wv_mat->packed_col_;
  int v2wv_area = v2wv_mat->row_ * v2wv_mat->col_ * v2wv_mat->batch_ / param->batch_;
  float *v2wv = v2wv_mat->data_;
  memset(v2wv, 0, param->batch_ * v2wv_area * sizeof(float));
  for (int i = 0; i < param->batch_; i++) {
    float *cur_v = v_mat->packed_data_ + i * v_area;
    float *cur_wv = wv_mat->packed_data_ + i * wv_area;
    float *cur_v2wv = v2wv + i * v2wv_area;
    MatMulOpt(cur_v, cur_wv, cur_v2wv, bv_mat->packed_data_, ActType_No, v_mat->col_, v_mat->row_, wv_mat->col_,
              wv_mat->col_, OutType_Nhwc);
  }
  // V * WV perm [0,2,1,3]
  float *v2wv_trans_data = v2wv_trans_mat->data_;
  int v2wv_trans_area = v2wv_trans_mat->row_ * v2wv_trans_mat->col_;
  memset(v2wv_trans_data, 0, v2wv_trans_mat->batch_ * v2wv_trans_area * sizeof(float));
  TransposeParameter v2wv_trans_param;
  int v2wv_in_shape[] = {batch, param->v_seq_, num_heads, depth};
  int v2wv_out_shape[] = {batch, num_heads, param->v_seq_, depth};
  int v2wv_perm[] = {0, 2, 1, 3};
  (void)GetTransposeParameter(&v2wv_trans_param, v2wv_in_shape, 4, v2wv_out_shape, 4, v2wv_perm, 4);
  TransposeDimsFp32(v2wv, v2wv_trans_data, v2wv_out_shape, &v2wv_trans_param, 0, 1);
}

void PMulWeightP(RelativePositionAttentionParameter *param, Matrix *p_mat, const Matrix *wp_mat, Matrix *p2wp_mat,
                 Matrix *p2wp_trans_mat) {
  int num_heads = param->num_heads_;
  int d_model = param->d_model_;
  int batch = param->batch_;
  int depth = d_model / num_heads;

  // P * WP
  int p_area = p_mat->packed_row_ * p_mat->packed_col_;
  int wp_area = wp_mat->packed_row_ * wp_mat->packed_col_;
  int p2wp_area = p2wp_mat->row_ * p2wp_mat->col_ * p2wp_mat->batch_ / param->batch_;
  float *p2wp_data = p2wp_mat->data_;
  memset(p2wp_data, 0, param->batch_ * p2wp_area * sizeof(float));
  for (int i = 0; i < param->batch_; i++) {
    float *cur_p = p_mat->packed_data_ + i * p_area;
    float *cur_wp = wp_mat->packed_data_ + i * wp_area;
    float *cur_p2wp = p2wp_data + i * p2wp_area;
    MatMulOpt(cur_p, cur_wp, cur_p2wp, NULL, ActType_No, p_mat->col_, p_mat->row_, wp_mat->col_, wp_mat->col_,
              OutType_Nhwc);
  }
  // P * WP perm [0,2,3,1]
  float *p2wp_trans_data = p2wp_trans_mat->data_;
  int p2wp_trans_area = p2wp_trans_mat->row_ * p2wp_trans_mat->col_;
  memset(p2wp_trans_data, 0, p2wp_trans_mat->batch_ * p2wp_trans_area * sizeof(float));
  TransposeParameter p2wp_trans_param;
  int p2wp_in_shape[] = {batch, param->p_seq_, num_heads, depth};
  int p2wp_out_shape[] = {batch, num_heads, depth, param->p_seq_};
  int p2wp_perm[] = {0, 2, 3, 1};
  (void)GetTransposeParameter(&p2wp_trans_param, p2wp_in_shape, 4, p2wp_out_shape, 4, p2wp_perm, 4);
  TransposeDimsFp32(p2wp_data, p2wp_trans_data, p2wp_out_shape, &p2wp_trans_param, 0, 1);
}

void CalculateLogits(RelativePositionAttentionParameter *param, Matrix *q2wq_with_pu_trans_mat,
                     Matrix *q2wq_with_pv_trans_mat, Matrix *k2wk_trans_mat, Matrix *p2wp_trans_mat,
                     Matrix *logits_with_u_mat, Matrix *logits_with_v_mat, Matrix *logits_with_v_pad_mat,
                     Matrix *logits_with_v_shifted_mat, Matrix *logits_mat) {
  int num_heads = param->num_heads_;
  int d_model = param->d_model_;
  int depth = d_model / num_heads;

  // pack Q_WITH_U as left_matrix
  // since we malloc dst data, pack function can not be failed
  (void)PackLeftMatrix(q2wq_with_pu_trans_mat, param->row_tile_);
  // pack Q_WITH_V as left_matrix
  (void)PackLeftMatrix(q2wq_with_pv_trans_mat, param->row_tile_);
  // pack K * WK as right_matrix
  (void)PackRightMatrix(k2wk_trans_mat, param->col_tile_);
  // pack P * WP as right_matrix
  (void)PackRightMatrix(p2wp_trans_mat, param->col_tile_);

  // q_with_pu * k = logits_with_u
  MatMulOpt(q2wq_with_pu_trans_mat->packed_data_, k2wk_trans_mat->packed_data_, logits_with_u_mat->data_, NULL,
            ActType_No, q2wq_with_pu_trans_mat->col_, logits_with_u_mat->row_, logits_with_u_mat->col_,
            logits_with_u_mat->col_, OutType_Nhwc);

  // q_with_pv * p = logits_with_v
  MatMulOpt(q2wq_with_pv_trans_mat->packed_data_, p2wp_trans_mat->packed_data_, logits_with_v_mat->data_, NULL,
            ActType_No, q2wq_with_pv_trans_mat->col_, logits_with_v_mat->row_, logits_with_v_mat->col_,
            logits_with_v_mat->col_, OutType_Nhwc);
  // relative shift logits_with_v
  float *pad_buf = logits_with_v_pad_mat->data_;
  float *logits_with_v_shifted_data = logits_with_v_shifted_mat->data_;
  RelativeShift(logits_with_v_mat, pad_buf, logits_with_v_shifted_data);
  // logits = (logits_with_u + logits_with_v) / sqrt(depth)
  float *logits_buffer = logits_mat->data_;
  ElementOptAddDiv(logits_with_u_mat->data_, logits_with_v_shifted_data, 1 / sqrt(depth), logits_buffer,
                   logits_with_u_mat->batch_, logits_with_u_mat->row_ * logits_with_u_mat->col_);
}

void RelPosAttention(RelativePositionAttentionParameter *param, Matrix *logits_mat, Matrix *softmax_mat,
                     Matrix *v2wv_trans_mat, Matrix *logits2v_mat, Matrix *logits2v_trans_mat, const Matrix *wo_mat,
                     Matrix *bo_mat, Matrix *output_mat) {
  int num_heads = param->num_heads_;
  int d_model = param->d_model_;
  int batch = param->batch_;
  int depth = d_model / num_heads;
  float *logits_buffer = logits_mat->data_;
  // softmax(logits)
  SoftmaxLastAxis(logits_buffer, softmax_mat->data_, batch * num_heads * softmax_mat->row_, softmax_mat->col_);

  // logits * v
  (void)PackLeftMatrix(softmax_mat, param->row_tile_);
  (void)PackRightMatrix(v2wv_trans_mat, param->col_tile_);
  int softmax_logits_area = softmax_mat->packed_row_ * softmax_mat->packed_col_;
  int v2wv_area = v2wv_trans_mat->packed_row_ * v2wv_trans_mat->packed_col_;
  int logits2v_area = logits2v_mat->row_ * logits2v_mat->col_;
  float *logits2v_data = logits2v_mat->data_;
  memset(logits2v_data, 0, logits2v_mat->batch_ * logits2v_area * sizeof(float));
  for (int i = 0; i < logits2v_mat->batch_; i++) {
    float *cur_logits = softmax_mat->packed_data_ + i * softmax_logits_area;
    float *cur_v2wv = v2wv_trans_mat->packed_data_ + i * v2wv_area;
    float *cur_logits2v = logits2v_data + i * logits2v_area;
    MatMulOpt(cur_logits, cur_v2wv, cur_logits2v, NULL, ActType_No, softmax_mat->col_, softmax_mat->row_,
              v2wv_trans_mat->col_, v2wv_trans_mat->col_, OutType_Nhwc);
  }
  // multi_head output perm [0,2,1,3]
  float *logits2v_trans_data = logits2v_trans_mat->data_;
  int logits2v_trans_area = logits2v_trans_mat->row_ * logits2v_trans_mat->col_;
  memset(logits2v_trans_data, 0, logits2v_trans_mat->batch_ * logits2v_trans_area * sizeof(float));
  TransposeParameter logits2v_trans_param;
  int logits2v_trans_in_shape[] = {batch, num_heads, param->q_seq_, depth};
  int logits2v_trans_out_shape[] = {batch, param->q_seq_, num_heads, depth};
  int logits2v_trans_perm[] = {0, 2, 1, 3};
  (void)GetTransposeParameter(&logits2v_trans_param, logits2v_trans_in_shape, 4, logits2v_trans_out_shape, 4,
                              logits2v_trans_perm, 4);
  TransposeDimsFp32(logits2v_data, logits2v_trans_data, logits2v_trans_out_shape, &logits2v_trans_param, 0, 1);
  // concat = reshape [batch, -1, d_model]
  logits2v_trans_mat->batch_ = batch;
  logits2v_trans_mat->row_ = param->q_seq_;
  logits2v_trans_mat->col_ = param->d_model_;
  // * o
  (void)PackLeftMatrix(logits2v_trans_mat, param->row_tile_);
  int concat_out_area = logits2v_trans_mat->packed_row_ * logits2v_trans_mat->packed_col_;
  int wo_area = wo_mat->packed_row_ * wo_mat->packed_col_;
  int output_area = output_mat->row_ * output_mat->col_;
  for (int i = 0; i < output_mat->batch_; i++) {
    float *cur_concat_out = logits2v_trans_mat->packed_data_ + i * concat_out_area;
    float *cur_wo = wo_mat->packed_data_ + i * wo_area;
    float *cur_output = output_mat->data_ + i * output_area;
    MatMulOpt(cur_concat_out, cur_wo, cur_output, bo_mat->packed_data_, ActType_No, logits2v_trans_mat->col_,
              logits2v_trans_mat->row_, wo_mat->col_, wo_mat->col_, OutType_Nhwc);
  }
}
