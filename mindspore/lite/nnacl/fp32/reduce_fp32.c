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

int ReduceMean(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
               int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const float *outer_src = src_data + j * axis_size * inner_size;
    float *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const float *inner_src = outer_src + k;
      float *inner_dst = outer_dst + k;
      float tmp = 0.0f;
      for (i = 0; i < axis_size; i++) {
        tmp += inner_src[i * inner_size];
      }
      *inner_dst = tmp / (float)axis_size;
    }
  }
  return NNACL_OK;
}

int IntReduceMean(int outer_size, int inner_size, int axis_size, const int *src_data, int *dst_data, int tid,
                  int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j;
#ifdef ENABLE_NEON
  int block_mod = inner_size % C4NUM;
  int block_c4 = inner_size - block_mod;
#endif
  for (j = tid; j < outer_size; j += thread_num) {
    const int *outer_src = src_data + j * axis_size * inner_size;
    int *outer_dst = dst_data + j * inner_size;
    int k = 0;
#ifdef ENABLE_NEON
    for (; k < block_c4; k += C4NUM) {
      const int *inner_src = outer_src + k;
      int *inner_dst = outer_dst + k;
      int32x4_t tmp = {0, 0, 0, 0};
      for (i = 0; i < axis_size; i++) {
        tmp = vaddq_s32(tmp, vld1q_s32(inner_src + i * inner_size));
      }
      tmp[0] /= axis_size;
      tmp[1] /= axis_size;
      tmp[2] /= axis_size;
      tmp[3] /= axis_size;
      vst1q_s32(inner_dst, tmp);
    }
#endif
    for (; k < inner_size; k++) {
      const int *inner_src = outer_src + k;
      int *inner_dst = outer_dst + k;
      int tmp = 0;
      for (i = 0; i < axis_size; i++) {
        tmp += inner_src[i * inner_size];
      }
      *inner_dst = tmp / axis_size;
    }
  }
  return NNACL_OK;
}

int ReduceSum(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
              int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j;
#ifdef ENABLE_NEON
  int block_mod = inner_size % C4NUM;
  int block_c4 = inner_size - block_mod;
#endif
  for (j = tid; j < outer_size; j += thread_num) {
    const float *outer_src = src_data + j * axis_size * inner_size;
    float *outer_dst = dst_data + j * inner_size;
    int k = 0;
#ifdef ENABLE_NEON
    for (; k < block_c4; k += C4NUM) {
      const float *inner_src = outer_src + k;
      float *inner_dst = outer_dst + k;
      float32x4_t tmp = {0, 0, 0, 0};
      for (i = 0; i < axis_size; i++) {
        tmp = vaddq_f32(tmp, vld1q_f32(inner_src + i * inner_size));
      }
      vst1q_f32(inner_dst, tmp);
    }
#endif
    for (; k < inner_size; k++) {
      const float *inner_src = outer_src + k;
      float *inner_dst = outer_dst + k;
      float tmp = 0.0f;
      for (i = 0; i < axis_size; i++) {
        tmp += inner_src[i * inner_size];
      }
      *inner_dst = tmp;
    }
  }
  return NNACL_OK;
}

int IntReduceSum(int outer_size, int inner_size, int axis_size, const int *src_data, int *dst_data, int tid,
                 int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j;
#ifdef ENABLE_NEON
  int block_mod = inner_size % C4NUM;
  int block_c4 = inner_size - block_mod;
#endif
  for (j = tid; j < outer_size; j += thread_num) {
    const int *outer_src = src_data + j * axis_size * inner_size;
    int *outer_dst = dst_data + j * inner_size;
    int k = 0;
#ifdef ENABLE_NEON
    for (; k < block_c4; k += C4NUM) {
      const int *inner_src = outer_src + k;
      int *inner_dst = outer_dst + k;
      int32x4_t tmp = {0, 0, 0, 0};
      for (i = 0; i < axis_size; i++) {
        tmp = vaddq_s32(tmp, vld1q_s32(inner_src + i * inner_size));
      }
      vst1q_s32(inner_dst, tmp);
    }
#endif
    for (; k < inner_size; k++) {
      const int *inner_src = outer_src + k;
      int *inner_dst = outer_dst + k;
      int tmp = 0;
      for (i = 0; i < axis_size; i++) {
        tmp += inner_src[i * inner_size];
      }
      *inner_dst = tmp;
    }
  }
  return NNACL_OK;
}

int ReduceMax(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
              int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const float *outer_src = src_data + j * axis_size * inner_size;
    float *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const float *inner_src = outer_src + k;
      float *inner_dst = outer_dst + k;
      float tmp = -FLT_MAX;
      for (i = 0; i < axis_size; i++) {
        tmp = tmp > inner_src[i * inner_size] ? tmp : inner_src[i * inner_size];
      }
      *inner_dst = tmp;
    }
  }
  return NNACL_OK;
}

int IntReduceMax(int outer_size, int inner_size, int axis_size, const int *src_data, int *dst_data, int tid,
                 int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const int *outer_src = src_data + j * axis_size * inner_size;
    int *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const int *inner_src = outer_src + k;
      int *inner_dst = outer_dst + k;
      int tmp = -INT_MAX;
      for (i = 0; i < axis_size; i++) {
        tmp = tmp > inner_src[i * inner_size] ? tmp : inner_src[i * inner_size];
      }
      *inner_dst = tmp;
    }
  }
  return NNACL_OK;
}

int ReduceMin(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
              int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const float *outer_src = src_data + j * axis_size * inner_size;
    float *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const float *inner_src = outer_src + k;
      float *inner_dst = outer_dst + k;
      float tmp = FLT_MAX;
      for (i = 0; i < axis_size; i++) {
        tmp = tmp < inner_src[i * inner_size] ? tmp : inner_src[i * inner_size];
      }
      *inner_dst = tmp;
    }
  }
  return NNACL_OK;
}

int IntReduceMin(int outer_size, int inner_size, int axis_size, const int *src_data, int *dst_data, int tid,
                 int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const int *outer_src = src_data + j * axis_size * inner_size;
    int *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const int *inner_src = outer_src + k;
      int *inner_dst = outer_dst + k;
      int tmp = INT32_MAX;
      for (i = 0; i < axis_size; i++) {
        tmp = tmp < inner_src[i * inner_size] ? tmp : inner_src[i * inner_size];
      }
      *inner_dst = tmp;
    }
  }
  return NNACL_OK;
}

int ReduceAll(int outer_size, int inner_size, int axis_size, const bool *src_data, bool *dst_data, int tid,
              int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
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

int ReduceProd(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
               int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const float *outer_src = src_data + j * axis_size * inner_size;
    float *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const float *inner_src = outer_src + k;
      float *inner_dst = outer_dst + k;
      float tmp = 1.0f;
      for (i = 0; i < axis_size; i++) {
        tmp *= inner_src[i * inner_size];
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

int ReduceSumSquare(int outer_size, int inner_size, int axis_size, const float *src_data, float *dst_data, int tid,
                    int thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int i, j, k;
  for (j = tid; j < outer_size; j += thread_num) {
    const float *outer_src = src_data + j * axis_size * inner_size;
    float *outer_dst = dst_data + j * inner_size;
    for (k = 0; k < inner_size; k++) {
      const float *inner_src = outer_src + k;
      float *inner_dst = outer_dst + k;
      float tmp = 0.0f;
      for (i = 0; i < axis_size; i++) {
        tmp += inner_src[i * inner_size] * inner_src[i * inner_size];
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
  if (rank <= 0 || rank > REDUCE_MAX_AXES_NUM) {
    return NNACL_PARAM_INVALID;
  }
  int axes[REDUCE_MAX_AXES_NUM];
  int actual_axes_num = num_axes;
  for (int i = 0; i < num_axes; ++i) {
    if (reduce_parameter->axes_[i] < -rank || reduce_parameter->axes_[i] >= rank) {
      return NNACL_PARAM_INVALID;
    }
    if (reduce_parameter->axes_[i] < 0) {
      axes[i] = reduce_parameter->axes_[i] + rank;
    } else {
      axes[i] = reduce_parameter->axes_[i];
    }
  }
  if (reduce_parameter->reduce_to_end_) {
    if (num_axes != 1) {
      return NNACL_PARAM_INVALID;
    }
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
