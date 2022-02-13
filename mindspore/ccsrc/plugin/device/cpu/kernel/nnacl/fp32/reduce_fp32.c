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

#include "nnacl/fp32/reduce_fp32.h"
#include <float.h>
#include "nnacl/errorcode.h"
#include "nnacl/common_func.h"

#ifdef ENABLE_NNACL_INFER_SHAPE
#include "nnacl/reduce_parameter.h"
#endif

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define ReduceCoreCalc(block_size, block_num, op_name, op_type, outer_src, outer_dst, k)      \
  for (int block_max_size = inner_size - block_num + 1; k < block_max_size; k += block_num) { \
    const op_type *inner_src = outer_src + k;                                                 \
    op_type *inner_dst = outer_dst + k;                                                       \
    op_name##PreDeal(block_size, block_num);                                                  \
    for (int i = 0; i < axis_size; i++) {                                                     \
      op_name##MidCalc(block_size, block_num);                                                \
    }                                                                                         \
    op_name##PostDeal(block_size, block_num);                                                 \
  }

#define RegReduceOp(op_name, op_type)                                                                             \
  int op_name(int outer_size, int inner_size, int axis_size, const op_type *src_data, op_type *dst_data, int tid, \
              int thread_num) {                                                                                   \
    MS_CHECK_TRUE_RET(src_data != NULL && dst_data != NULL, NNACL_NULL_PTR);                                      \
    MS_CHECK_TRUE_RET(thread_num > 0, NNACL_PARAM_INVALID);                                                       \
    for (int j = tid; j < outer_size; j += thread_num) {                                                          \
      const op_type *outer_src = src_data + j * axis_size * inner_size;                                           \
      op_type *outer_dst = dst_data + j * inner_size;                                                             \
      int k = 0;                                                                                                  \
      MS_SIMD_RUN(ReduceCoreCalc, op_name, op_type, outer_src, outer_dst, k);                                     \
    }                                                                                                             \
    return NNACL_OK;                                                                                              \
  }

// ReduceSum
// (c style) ReduceSumPreDeal : float tmp = 0;
#define ReduceSumPreDeal(block_size, block_num) MS_FLOAT_32xN(block_num) tmp = MS_MOVN_F32(block_size, 0);
// (c style) ReduceSumMidCalc : tmp = tmp + *(inner_src + i * inner_size);
#define ReduceSumMidCalc(block_size, block_num) \
  tmp = MS_ADD_F32(block_size, tmp, MS_LD_F32(block_size, inner_src + i * inner_size));
// (c style) ReduceSumPostDeal : *inner_dst = tmp;
#define ReduceSumPostDeal(block_size, block_num) MS_ST_F32(block_size, inner_dst, tmp);
RegReduceOp(ReduceSum, float);

// ReduceMean
// (c style) ReduceMeanPreDeal : int tmp = 0;
#define ReduceMeanPreDeal(block_size, block_num) MS_FLOAT_32xN(block_num) tmp = MS_MOVN_F32(block_size, 0);
// (c style) ReduceMeanMidCalc : tmp = tmp + *(inner_src + i * inner_size);
#define ReduceMeanMidCalc(block_size, block_num) \
  tmp = MS_ADD_F32(block_size, tmp, MS_LD_F32(block_size, inner_src + i * inner_size));
// (c style) ReduceMeanPostDeal : *inner_dst = tmp / axis_size;
#define ReduceMeanPostDeal(block_size, block_num) \
  MS_ST_F32(block_size, inner_dst, MS_DIV_N_F32(block_size, tmp, axis_size));
RegReduceOp(ReduceMean, float);

// ReduceMin
// (c style) ReduceMinPreDeal : float tmp = FLT_MAX;
#define ReduceMinPreDeal(block_size, block_num) MS_FLOAT_32xN(block_num) tmp = MS_MOVN_F32(block_size, FLT_MAX);
// (c style) ReduceMinMidCalc : tmp = fminf(tmp, *(inner_src + i * inner_size));
#define ReduceMinMidCalc(block_size, block_num) \
  tmp = MS_MIN_F32(block_size, tmp, MS_LD_F32(block_size, inner_src + i * inner_size));
// (c style) ReduceMinPostDeal : *inner_dst = tmp;
#define ReduceMinPostDeal(block_size, block_num) MS_ST_F32(block_size, inner_dst, tmp);
RegReduceOp(ReduceMin, float);

// ReduceMax
// (c style) ReduceMaxPreDeal : float tmp = FLT_MIN;
#define ReduceMaxPreDeal(block_size, block_num) MS_FLOAT_32xN(block_num) tmp = MS_MOVN_F32(block_size, FLT_MIN);
// (c style) ReduceMaxMidCalc : tmp = fmaxf(tmp, *(inner_src + i * inner_size));
#define ReduceMaxMidCalc(block_size, block_num) \
  tmp = MS_MAX_F32(block_size, tmp, MS_LD_F32(block_size, inner_src + i * inner_size));
// (c style) ReduceMaxPostDeal : *inner_dst = tmp;
#define ReduceMaxPostDeal(block_size, block_num) MS_ST_F32(block_size, inner_dst, tmp);
RegReduceOp(ReduceMax, float);

// ReduceProd
// (c style) ReduceProdPreDeal : float tmp = 1.0f;
#define ReduceProdPreDeal(block_size, block_num) MS_FLOAT_32xN(block_num) tmp = MS_MOVN_F32(block_size, 1.0f);
// (c style) ReduceProdMidCalc : tmp = tmp * (*(inner_src + i * inner_size));
#define ReduceProdMidCalc(block_size, block_num) \
  tmp = MS_MUL_F32(block_size, tmp, MS_LD_F32(block_size, inner_src + i * inner_size));
// (c style) ReduceProdPostDeal : *inner_dst = tmp;
#define ReduceProdPostDeal(block_size, block_num) MS_ST_F32(block_size, inner_dst, tmp);
RegReduceOp(ReduceProd, float);

// ReduceSumSquare
// (c style) ReduceSumSquarePreDeal : float tmp = 0;
#define ReduceSumSquarePreDeal(block_size, block_num) MS_FLOAT_32xN(block_num) tmp = MS_MOVN_F32(block_size, 0);
// (c style) ReduceSumSquareMidCalc : float val = *(inner_src + i * inner_size); tmp = tmp + val * val;
#define ReduceSumSquareMidCalc(block_size, block_num) \
  tmp = MS_ADD_F32(block_size, tmp, MS_MUL_SQUARE_F32(block_size, MS_LD_F32(block_size, inner_src + i * inner_size)));
// (c style) ReduceSumSquarePostDeal : *inner_dst = tmp;
#define ReduceSumSquarePostDeal(block_size, block_num) MS_ST_F32(block_size, inner_dst, tmp);
RegReduceOp(ReduceSumSquare, float);

// IntReduceSum
// (c style) IntReduceSumPreDeal : int tmp = 0;
#define IntReduceSumPreDeal(block_size, block_num) MS_INT_32xN(block_num) tmp = MS_MOVN_EPI32(block_size, 0);
// (c style) IntReduceSumMidCalc : tmp = tmp + *(inner_src + i * inner_size);
#define IntReduceSumMidCalc(block_size, block_num) \
  tmp = MS_ADD_EPI32(block_size, tmp, MS_LD_EPI32(block_size, inner_src + i * inner_size));
// (c style) IntReduceSumPostDeal : *inner_dst = tmp;
#define IntReduceSumPostDeal(block_size, block_num) MS_ST_EPI32(block_size, inner_dst, tmp);
RegReduceOp(IntReduceSum, int);

// IntReduceMean
// (c style) IntReduceSumPreDeal : int tmp = 0;
#define IntReduceMeanPreDeal(block_size, block_num) MS_INT_32xN(block_num) tmp = MS_MOVN_EPI32(block_size, 0);
// (c style) IntReduceSumMidCalc : tmp = tmp + *(inner_src + i * inner_size);
#define IntReduceMeanMidCalc(block_size, block_num) \
  tmp = MS_ADD_EPI32(block_size, tmp, MS_LD_EPI32(block_size, inner_src + i * inner_size));
// (c style) IntReduceSumPostDeal : *inner_dst = tmp / axis_size;
#define IntReduceMeanPostDeal(block_size, block_num) \
  MS_ST_EPI32(block_size, inner_dst, MS_DIV_N_EPI32(block_size, tmp, axis_size));
RegReduceOp(IntReduceMean, int);

// IntReduceMin
// (c style) IntReduceMinPreDeal : int tmp = INT32_MAX;
#define IntReduceMinPreDeal(block_size, block_num) MS_INT_32xN(block_num) tmp = MS_MOVN_EPI32(block_size, INT32_MAX);
// (c style) IntReduceMinMidCalc : tmp = fminf(tmp, *(inner_src + i * inner_size));
#define IntReduceMinMidCalc(block_size, block_num) \
  tmp = MS_MIN_EPI32(block_size, tmp, MS_LD_EPI32(block_size, inner_src + i * inner_size));
// (c style) IntReduceMinPostDeal : *inner_dst = tmp;
#define IntReduceMinPostDeal(block_size, block_num) MS_ST_EPI32(block_size, inner_dst, tmp);
RegReduceOp(IntReduceMin, int);

// IntReduceMax
// (c style) IntReduceMinPreDeal : int tmp = INT32_MIN;
#define IntReduceMaxPreDeal(block_size, block_num) MS_INT_32xN(block_num) tmp = MS_MOVN_EPI32(block_size, INT32_MIN);
// (c style) IntReduceMinMidCalc : tmp = fmax+f(tmp, *(inner_src + i * inner_size));
#define IntReduceMaxMidCalc(block_size, block_num) \
  tmp = MS_MAX_EPI32(block_size, tmp, MS_LD_EPI32(block_size, inner_src + i * inner_size));
// (c style) IntReduceMinPostDeal : *inner_dst = tmp;
#define IntReduceMaxPostDeal(block_size, block_num) MS_ST_EPI32(block_size, inner_dst, tmp);
RegReduceOp(IntReduceMax, int);

int ReduceAll(int outer_size, int inner_size, int axis_size, const bool *src_data, bool *dst_data, int tid,
              int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  if (thread_num == 0) {
    return NNACL_PARAM_INVALID;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const bool *outer_src = src_data + j * axis_size * inner_size;
    bool *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const bool *inner_src = outer_src + k;
      bool *inner_dst = outer_dst + k;
      bool tmp = true;
      for (i = 0; i < axis_size; i++) {
        tmp = tmp && inner_src[i * inner_size];
      }
      *inner_dst = tmp;
    }
  }
  return NNACL_OK;
}

int IntReduceProd(int outer_size, int inner_size, int axis_size, const int *src_data, int *dst_data, int tid,
                  int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  if (thread_num == 0) {
    return NNACL_PARAM_INVALID;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const int *outer_src = src_data + j * axis_size * inner_size;
    int *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const int *inner_src = outer_src + k;
      int *inner_dst = outer_dst + k;
      int tmp = 1;
      for (i = 0; i < axis_size; i++) {
        if (isMulOverflow(tmp, inner_src[i * inner_size])) {
          return NNACL_ERRCODE_MUL_OVERFLOW;
        }
        tmp *= inner_src[i * inner_size];
      }
      *inner_dst = tmp;
    }
  }
  return NNACL_OK;
}

#ifdef ENABLE_NNACL_INFER_SHAPE
int ReduceInferShape(int **in_shape, size_t *dim_size, int *out_shape, int *in_format, int *out_format,
                     int *in_datatype, int *out_datatype, OpParameter *param) {
  *out_format = in_format[0];
  *out_datatype = in_datatype[0];
  ReduceParameter *reduce_parameter = (ReduceParameter *)param;
  bool keep_dims = reduce_parameter->keep_dims_;
  int num_axes = reduce_parameter->num_axes_;
  int *in_shape0 = in_shape[0];
  int rank = dim_size[0];
  MS_CHECK_TRUE_RET(rank > 0 && rank <= REDUCE_MAX_AXES_NUM, NNACL_PARAM_INVALID);
  int axes[REDUCE_MAX_AXES_NUM];
  int actual_axes_num = num_axes;
  for (int i = 0; i < num_axes; ++i) {
    MS_CHECK_TRUE_RET(reduce_parameter->axes_[i] >= -rank && reduce_parameter->axes_[i] < rank, NNACL_PARAM_INVALID);
    if (reduce_parameter->axes_[i] < 0) {
      axes[i] = reduce_parameter->axes_[i] + rank;
    } else {
      axes[i] = reduce_parameter->axes_[i];
    }
  }
  if (reduce_parameter->reduce_to_end_) {
    MS_CHECK_TRUE_RET(num_axes == 1, NNACL_PARAM_INVALID);
    int begin_axis = axes[0];
    num_axes = rank - begin_axis;
    for (int i = begin_axis + 1; i < rank; ++i) {
      axes[actual_axes_num++] = i;
    }
  }
  if (num_axes == 0) {
    int j = 0;
    for (int i = 0; i < rank; ++i) {
      axes[i] = i;
      if (keep_dims) {
        out_shape[j++] = 1;
      }
    }
    reduce_parameter->num_axes_ = rank;
    for (int i = 0; i < rank; ++i) {
      reduce_parameter->axes_[i] = axes[i];
    }
    return NNACL_OK;
  }
  // reduce on selected axes
  int j = 0;
  for (int i = 0; i < rank; ++i) {
    bool reduce_axis = false;
    for (int idx = 0; idx < num_axes; ++idx) {
      if (axes[idx] == i) {
        reduce_axis = true;
        break;
      }
    }
    if (reduce_axis) {
      if (keep_dims) {
        out_shape[j++] = 1;
      }
    } else {
      out_shape[j++] = in_shape0[i];
    }
  }
  reduce_parameter->num_axes_ = num_axes;
  for (int i = 0; i < num_axes; ++i) {
    reduce_parameter->axes_[i] = axes[i];
  }
  return NNACL_OK;
}
#endif

// [A, B] -> [B]
// col_size : start -> end for parallel
int ReduceSumDim2Axis0(size_t col_size, size_t col_len, size_t row_len, const float *src_data, float *dst_data) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
#ifdef ENABLE_AVX
  size_t block_mod = col_size % C8NUM;
  size_t block_c8 = col_size - block_mod;
#endif
  size_t k = 0;
#ifdef ENABLE_AVX
  for (; k < block_c8; k += C8NUM) {
    MS_FLOAT32X8 tmp = {0, 0, 0, 0, 0, 0, 0, 0};
    const float *inner_src = src_data + k;
    float *inner_dst = dst_data + k;
    for (size_t i = 0; i < row_len; ++i) {
      tmp = MS_ADD256_F32(tmp, MS_LD256_F32(inner_src + i * col_len));
    }
    MS_ST256_F32(inner_dst, tmp);
  }
#endif
  for (; k < col_size; k++) {
    const float *inner_src = src_data + k;
    float *inner_dst = dst_data + k;
    float tmp = 0.0f;
    for (size_t i = 0; i < row_len; i++) {
      tmp += inner_src[i * col_len];
    }
    *inner_dst = tmp;
  }
  return NNACL_OK;
}

// [A, B] -> [A]
int ReduceSumDim2Axis1(size_t col_len, const float *src_data, float *dst_data) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  size_t k = 0;
  float tmp = 0;
#ifdef ENABLE_AVX
  size_t block_mod = col_len % C8NUM;
  size_t block_c8 = col_len - block_mod;
  float tmp_arr[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  MS_FLOAT32X8 tmp_arr_8 = MS_MOV256_F32(tmp_arr[0]);
  for (; k < block_c8; k += C8NUM) {
    MS_FLOAT32X8 src_in = MS_LD256_F32(src_data + k);
    tmp_arr_8 = MS_ADD256_F32(tmp_arr_8, src_in);
  }
  MS_ST256_F32(tmp_arr, tmp_arr_8);
  for (size_t i = 0; i < 8; ++i) {
    tmp += tmp_arr[i];
  }
#endif
  for (; k < col_len; k++) {
    tmp += src_data[k];
  }
  dst_data[0] = tmp;
  return NNACL_OK;
}
