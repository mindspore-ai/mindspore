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
#include "nnacl/reduce_fp32_simd.h"
#ifdef ENABLE_NNACL_INFER_SHAPE
#include "nnacl/reduce_parameter.h"
#endif

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define ReduceCoreCalc(op_name, op_type, outer_src, outer_dst, k) \
  for (; k < inner_size; k++) {                                   \
    const op_type *inner_src = outer_src + k;                     \
    op_name##PreDeal;                                             \
    for (int i = 0; i < axis_size; i++) {                         \
      op_name##MidCalc;                                           \
    }                                                             \
    op_name##PostDeal;                                            \
  }

#define RegReduceOp(op_name, op_type)                                                                             \
  int op_name(int outer_size, int inner_size, int axis_size, const op_type *src_data, op_type *dst_data, int tid, \
              int thread_num) {                                                                                   \
    NNACL_CHECK_TRUE_RET(src_data != NULL && dst_data != NULL, NNACL_NULL_PTR);                                   \
    NNACL_CHECK_TRUE_RET(thread_num > 0, NNACL_PARAM_INVALID);                                                    \
    NNACL_CHECK_TRUE_RET(axis_size > 0, NNACL_ERR);                                                               \
    for (int j = tid; j < outer_size; j += thread_num) {                                                          \
      const op_type *outer_src = src_data + j * axis_size * inner_size;                                           \
      op_type *outer_dst = dst_data + j * inner_size;                                                             \
      int k = 0;                                                                                                  \
      SIMD_RUN_NO_SCALAR(op_name, k, outer_src, outer_dst, inner_size, axis_size);                                \
                                                                                                                  \
      ReduceCoreCalc(op_name, op_type, outer_src, outer_dst, k);                                                  \
    }                                                                                                             \
    return NNACL_OK;                                                                                              \
  }

// ReduceSum
#define ReduceSumPreDeal float tmp = 0;
#define ReduceSumMidCalc tmp += inner_src[i * inner_size];
#define ReduceSumPostDeal outer_dst[k] = tmp;
RegReduceOp(ReduceSum, float);

int ReduceSumByLastAxis(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
                        int thread_num) {
  NNACL_CHECK_TRUE_RET(src_data != NULL && dst_data != NULL, NNACL_NULL_PTR);
  NNACL_CHECK_TRUE_RET(thread_num > 0, NNACL_PARAM_INVALID);
  NNACL_CHECK_TRUE_RET(axis_size > 0, NNACL_ERR);

  for (int j = tid; j < outer_size; j += thread_num) {
    const float *src_tmp = src_data + j * axis_size;

    float tmp = src_tmp[0];
    int i = 1;

    SIMD_RUN_NO_SCALAR(ReduceSumByLastAxis, i, src_tmp, &tmp, axis_size);
    for (; i < axis_size; i++) {
      tmp += src_tmp[i];
    }
    dst_data[j] = tmp;
  }
  return NNACL_OK;
}

// ReduceMean
#define ReduceMeanPreDeal float tmp = 0;
#define ReduceMeanMidCalc tmp += inner_src[i * inner_size];
#define ReduceMeanPostDeal outer_dst[k] = tmp / axis_size;
RegReduceOp(ReduceMean, float);

// ReduceMin
#define ReduceMinPreDeal float tmp = FLT_MAX;
#define ReduceMinMidCalc tmp = fminf(tmp, inner_src[i * inner_size]);
#define ReduceMinPostDeal outer_dst[k] = tmp;
RegReduceOp(ReduceMin, float);

// ReduceMax
#define ReduceMaxPreDeal float tmp = FLT_MIN;
#define ReduceMaxMidCalc tmp = fmaxf(tmp, inner_src[i * inner_size]);
#define ReduceMaxPostDeal outer_dst[k] = tmp;
RegReduceOp(ReduceMax, float);

int ReduceMaxByLastAxis(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
                        int thread_num) {
  NNACL_CHECK_TRUE_RET(src_data != NULL && dst_data != NULL, NNACL_NULL_PTR);
  NNACL_CHECK_TRUE_RET(thread_num > 0, NNACL_PARAM_INVALID);
  NNACL_CHECK_TRUE_RET(axis_size > 0, NNACL_ERR);

  for (int j = tid; j < outer_size; j += thread_num) {
    const float *src_tmp = src_data + j * axis_size;

    float tmp = src_tmp[0];
    int i = 1;

    SIMD_RUN_NO_SCALAR(ReduceMaxByLastAxis, i, src_tmp, &tmp, axis_size);
    for (; i < axis_size; i++) {
      tmp = fmaxf(tmp, src_tmp[i]);
    }
    dst_data[j] = tmp;
  }
  return NNACL_OK;
}

// ReduceProd
#define ReduceProdPreDeal float tmp = 1.0f;
#define ReduceProdMidCalc tmp *= inner_src[i * inner_size];
#define ReduceProdPostDeal outer_dst[k] = tmp;
RegReduceOp(ReduceProd, float);

// ReduceSumSquare
#define ReduceSumSquarePreDeal float tmp = 0;
#define ReduceSumSquareMidCalc tmp += (inner_src[i * inner_size] * inner_src[i * inner_size]);
#define ReduceSumSquarePostDeal outer_dst[k] = tmp;
RegReduceOp(ReduceSumSquare, float);

// ReduceL2Norm
#define ReduceL2NormPreDeal float tmp = 0;
#define ReduceL2NormMidCalc tmp += (inner_src[i * inner_size] * inner_src[i * inner_size]);
#define ReduceL2NormPostDeal outer_dst[k] = sqrt(tmp);
RegReduceOp(ReduceL2Norm, float);

// IntReduceSum
#define IntReduceSumPreDeal int tmp = 0;
#define IntReduceSumMidCalc tmp += inner_src[i * inner_size];
#define IntReduceSumPostDeal outer_dst[k] = tmp;
RegReduceOp(IntReduceSum, int32_t);

// IntReduceMean
#define IntReduceMeanPreDeal int tmp = 0;
#define IntReduceMeanMidCalc tmp += inner_src[i * inner_size];
#define IntReduceMeanPostDeal outer_dst[k] = tmp / axis_size;
RegReduceOp(IntReduceMean, int32_t);

// IntReduceMin
#define IntReduceMinPreDeal int tmp = INT32_MAX;
#define IntReduceMinMidCalc tmp = MSMIN(tmp, inner_src[i * inner_size]);
#define IntReduceMinPostDeal outer_dst[k] = tmp;
RegReduceOp(IntReduceMin, int32_t);

// IntReduceMax
#define IntReduceMaxPreDeal int tmp = INT32_MIN;
#define IntReduceMaxMidCalc tmp = MSMAX(tmp, inner_src[i * inner_size]);
#define IntReduceMaxPostDeal outer_dst[k] = tmp;
RegReduceOp(IntReduceMax, int32_t);

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

int IntReduceProd(int outer_size, int inner_size, int axis_size, const int32_t *src_data, int32_t *dst_data, int tid,
                  int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  if (thread_num == 0) {
    return NNACL_PARAM_INVALID;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const int32_t *outer_src = src_data + j * axis_size * inner_size;
    int32_t *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const int32_t *inner_src = outer_src + k;
      int32_t *inner_dst = outer_dst + k;
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
int ReduceInferShape(int32_t **in_shape, size_t *dim_size, int32_t *out_shape, int32_t *in_format, int32_t *out_format,
                     int32_t *in_datatype, int32_t *out_datatype, OpParameter *param) {
  *out_format = in_format[0];
  *out_datatype = in_datatype[0];
  ReduceParameter *reduce_parameter = (ReduceParameter *)param;
  bool keep_dims = reduce_parameter->keep_dims_;
  int num_axes = reduce_parameter->num_axes_;
  int32_t *in_shape0 = in_shape[0];
  int rank = dim_size[0];
  NNACL_CHECK_TRUE_RET(rank > 0 && rank <= REDUCE_MAX_AXES_NUM, NNACL_PARAM_INVALID);
  int axes[REDUCE_MAX_AXES_NUM];
  int actual_axes_num = num_axes;
  for (int i = 0; i < num_axes; ++i) {
    NNACL_CHECK_TRUE_RET(reduce_parameter->axes_[i] >= -rank && reduce_parameter->axes_[i] < rank, NNACL_PARAM_INVALID);
    if (reduce_parameter->axes_[i] < 0) {
      axes[i] = reduce_parameter->axes_[i] + rank;
    } else {
      axes[i] = reduce_parameter->axes_[i];
    }
  }
  if (reduce_parameter->reduce_to_end_) {
    NNACL_CHECK_TRUE_RET(num_axes == 1, NNACL_PARAM_INVALID);
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

  size_t k = 0;
  SIMD_RUN_NO_SCALAR(ReduceSumDim2Axis0, k, col_size, col_len, row_len, src_data, dst_data);
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

int ReduceMeanWithAxis(const float *src_data, float *mean, int64_t size) {
  if (size == 0 || src_data == NULL) {
    return NNACL_NULL_PTR;
  }
  float sum = 0.0;
  int64_t i = 0;
  SIMD_RUN_NO_SCALAR(ReduceSumByLastAxis, i, src_data, &sum, 0);
  for (; i < size; ++i) {
    sum += src_data[i];
  }
  *mean = sum / size;
  return NNACL_OK;
}

int ReduceDeviation(const float *src_data, int64_t size, float mean, float *deviation) {
  if (size == 0 || src_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int64_t i = 0;
  SIMD_RUN_NO_SCALAR(FloatReduceDeviation, i, src_data, mean, size, deviation);
  for (; i < size; ++i) {
    float tmp = src_data[i] - mean;
    tmp = tmp * tmp;
    *deviation += tmp;
  }
  return NNACL_OK;
}
