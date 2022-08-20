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
#ifndef MINDSPORE_NNACL_BASE_SCALE_BASE_H_
#define MINDSPORE_NNACL_BASE_SCALE_BASE_H_

#define ScalarNoAct(src) src
#define ScalarRelu(src) MSMAX(src, 0.0f)
#define ScalarRelu6(src) MSMIN(MSMAX(src, 0.0f), 6.0f)

// Pattern: the shapes of src, scale and bias can be converted to like {[a, b, c], [b, 1], [b, 1]};
#define ScalePatternOneTemplate(FUNC, SUB_FUNC, ACT, TYPE)                                                      \
  void FUNC(const TYPE *src, const TYPE *scale, const TYPE *bias, TYPE *out, const ScaleParameter *scale_param, \
            const int block_info[C4NUM]) {                                                                      \
    int inner_size = scale_param->inner_size_;                                                                  \
    int pre = inner_size - block_info[C3NUM];                                                                   \
    int post = block_info[1] - block_info[0] - pre;                                                             \
    int middle = 0;                                                                                             \
    if (post > 0) {                                                                                             \
      middle = post / inner_size;                                                                               \
      post = post % inner_size;                                                                                 \
    } else {                                                                                                    \
      pre += post;                                                                                              \
      post = 0;                                                                                                 \
    }                                                                                                           \
    int middle_begin = block_info[C2NUM];                                                                       \
    if (pre > 0) {                                                                                              \
      int i = 0;                                                                                                \
      SIMD_RUN_AVX(SUB_FUNC, i, src, scale[middle_begin], bias[middle_begin], out, pre);                        \
      SIMD_RUN_SSE(SUB_FUNC, i, src, scale[middle_begin], bias[middle_begin], out, pre);                        \
      SIMD_RUN_NEON(SUB_FUNC, i, src, scale[middle_begin], bias[middle_begin], out, pre);                       \
      for (; i < pre; ++i) {                                                                                    \
        out[i] = ACT(src[i] * scale[middle_begin] + bias[middle_begin]);                                        \
      }                                                                                                         \
      src += pre;                                                                                               \
      out += pre;                                                                                               \
      ++middle_begin;                                                                                           \
    }                                                                                                           \
    int middle_size = scale_param->axis_size_;                                                                  \
    for (; middle > 0; --middle) {                                                                              \
      middle_begin = middle_begin == middle_size ? 0 : middle_begin;                                            \
      int i = 0;                                                                                                \
      SIMD_RUN_AVX(SUB_FUNC, i, src, scale[middle_begin], bias[middle_begin], out, inner_size);                 \
      SIMD_RUN_SSE(SUB_FUNC, i, src, scale[middle_begin], bias[middle_begin], out, inner_size);                 \
      SIMD_RUN_NEON(SUB_FUNC, i, src, scale[middle_begin], bias[middle_begin], out, inner_size);                \
      for (; i < inner_size; ++i) {                                                                             \
        out[i] = ACT(src[i] * scale[middle_begin] + bias[middle_begin]);                                        \
      }                                                                                                         \
      src += inner_size;                                                                                        \
      out += inner_size;                                                                                        \
      ++middle_begin;                                                                                           \
    }                                                                                                           \
    if (post > 0) {                                                                                             \
      middle_begin = middle_begin == middle_size ? 0 : middle_begin;                                            \
      int i = 0;                                                                                                \
      SIMD_RUN_AVX(SUB_FUNC, i, src, scale[middle_begin], bias[middle_begin], out, post);                       \
      SIMD_RUN_SSE(SUB_FUNC, i, src, scale[middle_begin], bias[middle_begin], out, post);                       \
      SIMD_RUN_NEON(SUB_FUNC, i, src, scale[middle_begin], bias[middle_begin], out, post);                      \
      for (; i < post; ++i) {                                                                                   \
        out[i] = ACT(src[i] * scale[middle_begin] + bias[middle_begin]);                                        \
      }                                                                                                         \
    }                                                                                                           \
  }

// Pattern: the shapes of src, scale and bias can be converted to like {[a, b], [b], [b]}
#define ScalePatternTwoTemplate(FUNC, SUB_FUNC, ACT, TYPE)                                                      \
  void FUNC(const TYPE *src, const TYPE *scale, const TYPE *bias, TYPE *out, const ScaleParameter *scale_param, \
            const int block_info[C4NUM]) {                                                                      \
    int middle_size = scale_param->axis_size_;                                                                  \
    int pre = middle_size - block_info[C2NUM];                                                                  \
    int post = block_info[1] - block_info[0] - pre;                                                             \
    int middle = 0;                                                                                             \
    if (post > 0) {                                                                                             \
      middle = post / middle_size;                                                                              \
      post = post % middle_size;                                                                                \
    } else {                                                                                                    \
      pre += post;                                                                                              \
      post = 0;                                                                                                 \
    }                                                                                                           \
    int middle_begin = block_info[C2NUM];                                                                       \
    if (pre > 0) {                                                                                              \
      int i = 0;                                                                                                \
      SIMD_RUN_AVX(SUB_FUNC, i, src, scale + middle_begin, bias + middle_begin, out, pre);                      \
      SIMD_RUN_SSE(SUB_FUNC, i, src, scale + middle_begin, bias + middle_begin, out, pre);                      \
      SIMD_RUN_NEON(SUB_FUNC, i, src, scale + middle_begin, bias + middle_begin, out, pre);                     \
      for (; i < pre; ++i) {                                                                                    \
        out[i] = ACT(src[i] * scale[middle_begin + i] + bias[middle_begin + i]);                                \
      }                                                                                                         \
      src += pre;                                                                                               \
      out += pre;                                                                                               \
    }                                                                                                           \
    for (; middle > 0; --middle) {                                                                              \
      int i = 0;                                                                                                \
      SIMD_RUN_AVX(SUB_FUNC, i, src, scale, bias, out, middle_size);                                            \
      SIMD_RUN_SSE(SUB_FUNC, i, src, scale, bias, out, middle_size);                                            \
      SIMD_RUN_NEON(SUB_FUNC, i, src, scale, bias, out, middle_size);                                           \
      for (; i < middle_size; ++i) {                                                                            \
        out[i] = ACT(src[i] * scale[i] + bias[i]);                                                              \
      }                                                                                                         \
      src += middle_size;                                                                                       \
      out += middle_size;                                                                                       \
      ++middle_begin;                                                                                           \
    }                                                                                                           \
    if (post > 0) {                                                                                             \
      int i = 0;                                                                                                \
      SIMD_RUN_AVX(SUB_FUNC, i, src, scale, bias, out, post);                                                   \
      SIMD_RUN_SSE(SUB_FUNC, i, src, scale, bias, out, post);                                                   \
      SIMD_RUN_NEON(SUB_FUNC, i, src, scale, bias, out, post);                                                  \
      for (; i < post; ++i) {                                                                                   \
        out[i] = ACT(src[i] * scale[i] + bias[i]);                                                              \
      }                                                                                                         \
    }                                                                                                           \
  }

// Pattern: the shapes of src, scale and bias can be converted to like {[a, b, c], [b, 1], [c]};
#define ScalePatternThirdTemplate(FUNC, SUB_FUNC, ACT, TYPE)                                                    \
  void FUNC(const TYPE *src, const TYPE *scale, const TYPE *bias, TYPE *out, const ScaleParameter *scale_param, \
            const int block_info[C4NUM]) {                                                                      \
    int inner_size = scale_param->inner_size_;                                                                  \
    int inner_begin = block_info[C3NUM];                                                                        \
    int pre = inner_size - inner_begin;                                                                         \
    int post = block_info[1] - block_info[0] - pre;                                                             \
    int middle = 0;                                                                                             \
    if (post > 0) {                                                                                             \
      middle = post / inner_size;                                                                               \
      post = post % inner_size;                                                                                 \
    } else {                                                                                                    \
      pre += post;                                                                                              \
      post = 0;                                                                                                 \
    }                                                                                                           \
    int middle_begin = block_info[C2NUM];                                                                       \
    if (pre > 0) {                                                                                              \
      int i = 0;                                                                                                \
      SIMD_RUN_AVX(SUB_FUNC, i, src, scale[middle_begin], bias + inner_begin, out, pre);                        \
      SIMD_RUN_SSE(SUB_FUNC, i, src, scale[middle_begin], bias + inner_begin, out, pre);                        \
      SIMD_RUN_NEON(SUB_FUNC, i, src, scale[middle_begin], bias + inner_begin, out, pre);                       \
      for (; i < pre; ++i) {                                                                                    \
        out[i] = ACT(src[i] * scale[middle_begin] + bias[inner_begin + i]);                                     \
      }                                                                                                         \
      src += pre;                                                                                               \
      out += pre;                                                                                               \
      ++middle_begin;                                                                                           \
    }                                                                                                           \
    int middle_size = scale_param->axis_size_;                                                                  \
    for (; middle > 0; --middle) {                                                                              \
      middle_begin = middle_begin == middle_size ? 0 : middle_begin;                                            \
      int i = 0;                                                                                                \
      SIMD_RUN_AVX(SUB_FUNC, i, src, scale[middle_begin], bias, out, inner_size);                               \
      SIMD_RUN_SSE(SUB_FUNC, i, src, scale[middle_begin], bias, out, inner_size);                               \
      SIMD_RUN_NEON(SUB_FUNC, i, src, scale[middle_begin], bias, out, inner_size);                              \
      for (; i < inner_size; ++i) {                                                                             \
        out[i] = ACT(src[i] * scale[middle_begin] + bias[i]);                                                   \
      }                                                                                                         \
      src += inner_size;                                                                                        \
      out += inner_size;                                                                                        \
      ++middle_begin;                                                                                           \
    }                                                                                                           \
    if (post > 0) {                                                                                             \
      middle_begin = middle_begin == middle_size ? 0 : middle_begin;                                            \
      int i = 0;                                                                                                \
      SIMD_RUN_AVX(SUB_FUNC, i, src, scale[middle_begin], bias, out, post);                                     \
      SIMD_RUN_SSE(SUB_FUNC, i, src, scale[middle_begin], bias, out, post);                                     \
      SIMD_RUN_NEON(SUB_FUNC, i, src, scale[middle_begin], bias, out, post);                                    \
      for (; i < post; ++i) {                                                                                   \
        out[i] = ACT(src[i] * scale[middle_begin] + bias[i]);                                                   \
      }                                                                                                         \
    }                                                                                                           \
  }

#endif  // MINDSPORE_NNACL_BASE_SCALE_BASE_H_
