/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include <stdio.h>
#include "nnacl/base/gather_d_base.h"

int CheckIndexValue_int32_t(int32_t *index, const int max_index, const size_t *index_shape,
                            const size_t index_shape_size) {
  // check index
  size_t index_size = 1;
  for (size_t i = 0; i < index_shape_size; ++i) {
    index_size *= index_shape[i];
  }

  for (size_t i = 0; i < index_size; ++i) {
    if (index[i] >= max_index || index[i] < -max_index) {
      return NNACL_ERR;
    }
    if (index[i] < 0) {
      index[i] = max_index + index[i];
    }
  }
  return NNACL_OK;
}

int CheckIndexValue_int64_t(int64_t *index, const int max_index, const size_t *index_shape,
                            const size_t index_shape_size) {
  // check index
  size_t index_size = 1;
  for (size_t i = 0; i < index_shape_size; ++i) {
    index_size *= index_shape[i];
  }
  for (size_t i = 0; i < index_size; ++i) {
    if (index[i] >= max_index || index[i] < -max_index) {
      return NNACL_ERR;
    }
    if (index[i] < 0) {
      index[i] = max_index + index[i];
    }
  }
  return NNACL_OK;
}

int InitCalVec(size_t *in_strides, size_t *out_strides, size_t *pos, const size_t *input_shape,
               const size_t input_shape_size, const size_t *output_shape, const size_t output_shape_size) {
  // in_strides
  NNACL_CHECK_NULL_RETURN_ERR(in_strides);
  for (size_t i = 0; i < input_shape_size; ++i) {
    in_strides[i] = 1;
  }
  for (int i = (int)input_shape_size - 2; i >= 0; --i) {
    in_strides[i] = input_shape[i + 1] * in_strides[i + 1];
  }

  // out_strides
  NNACL_CHECK_NULL_RETURN_ERR(out_strides);
  for (size_t i = 0; i < output_shape_size; ++i) {
    out_strides[i] = 1;
  }
  for (int i = (int)output_shape_size - 2; i >= 0; --i) {
    out_strides[i] = output_shape[i + 1] * out_strides[i + 1];
  }

  NNACL_CHECK_NULL_RETURN_ERR(pos);
  for (size_t i = 0; i < output_shape_size; ++i) {
    pos[i] = 0;
  }
  return NNACL_OK;
}

#define COPY_TASK_IMPL(type0, type1)                                                                                   \
  int CopyTask_Input_##type0##_Index_##type1(                                                                          \
    type0 *output, const type0 *input, const type1 *index, size_t cur_dim, size_t *pos, const int dim,                 \
    const size_t *output_shape, const size_t output_shape_size, const size_t *in_strides, const size_t *out_strides) { \
    if (pos == NULL || out_strides == NULL || in_strides == NULL) {                                                    \
      return NNACL_NULL_PTR;                                                                                           \
    }                                                                                                                  \
    for (size_t i = 0; i < output_shape[cur_dim]; ++i) {                                                               \
      pos[cur_dim] = i;                                                                                                \
      if (cur_dim == (int)output_shape_size - 1) {                                                                     \
        size_t input_offset = 0;                                                                                       \
        size_t out_offset = 0;                                                                                         \
        for (size_t j = 0; j < output_shape_size; ++j) {                                                               \
          out_offset += pos[j] * out_strides[j];                                                                       \
        }                                                                                                              \
        size_t cur_index = pos[dim];                                                                                   \
        pos[dim] = index[out_offset];                                                                                  \
        for (size_t j = 0; j < output_shape_size; ++j) {                                                               \
          input_offset += pos[j] * in_strides[j];                                                                      \
        }                                                                                                              \
        ((type0 *)output)[out_offset] = ((const type0 *)input)[input_offset];                                          \
        pos[dim] = cur_index;                                                                                          \
      } else {                                                                                                         \
        CopyTask_Input_##type0##_Index_##type1(output, input, index, cur_dim + 1, pos, dim, output_shape,              \
                                               output_shape_size, in_strides, out_strides);                            \
      }                                                                                                                \
    }                                                                                                                  \
    return NNACL_OK;                                                                                                   \
  }

COPY_TASK_IMPL(bool, int32_t)
COPY_TASK_IMPL(bool, int64_t)
COPY_TASK_IMPL(int16_t, int32_t)
COPY_TASK_IMPL(int16_t, int64_t)
COPY_TASK_IMPL(int32_t, int32_t)
COPY_TASK_IMPL(int32_t, int64_t)
COPY_TASK_IMPL(int64_t, int32_t)
COPY_TASK_IMPL(int64_t, int64_t)
COPY_TASK_IMPL(float, int32_t)
COPY_TASK_IMPL(float, int64_t)
#ifdef ENABLE_FP16
COPY_TASK_IMPL(float16_t, int32_t)
COPY_TASK_IMPL(float16_t, int64_t)
#endif

#define GATHER_D_IMPL(type0, type1)                                                                                  \
  GATHER_D_IMPL_DECLARATION(type0, type1) {                                                                          \
    if (output == NULL || input == NULL || index == NULL || input_shape == NULL || output_shape == NULL) {           \
      return NNACL_NULL_PTR;                                                                                         \
    }                                                                                                                \
    int max_index = input_shape[dim];                                                                                \
    int ret = CheckIndexValue_##type1(index, max_index, output_shape, output_shape_size);                            \
    if (ret != NNACL_OK) {                                                                                           \
      return ret;                                                                                                    \
    }                                                                                                                \
    size_t in_strides[MAX_SHAPE_SIZE];                                                                               \
    size_t out_strides[MAX_SHAPE_SIZE];                                                                              \
    size_t pos[MAX_SHAPE_SIZE];                                                                                      \
    ret = InitCalVec(in_strides, out_strides, pos, input_shape, input_shape_size, output_shape, output_shape_size);  \
    if (ret != NNACL_OK) {                                                                                           \
      return ret;                                                                                                    \
    }                                                                                                                \
    ret = CopyTask_Input_##type0##_Index_##type1(output, input, index, 0, pos, dim, output_shape, output_shape_size, \
                                                 in_strides, out_strides);                                           \
    return ret;                                                                                                      \
  }

GATHER_D_IMPL(bool, int32_t)
GATHER_D_IMPL(bool, int64_t)
GATHER_D_IMPL(int16_t, int32_t)
GATHER_D_IMPL(int16_t, int64_t)
GATHER_D_IMPL(int32_t, int32_t)
GATHER_D_IMPL(int32_t, int64_t)
GATHER_D_IMPL(int64_t, int32_t)
GATHER_D_IMPL(int64_t, int64_t)
GATHER_D_IMPL(float, int32_t)
GATHER_D_IMPL(float, int64_t)
#ifdef ENABLE_FP16
GATHER_D_IMPL(float16_t, int32_t)
GATHER_D_IMPL(float16_t, int64_t)
#endif
