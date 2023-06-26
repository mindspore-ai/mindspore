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

#include "nnacl/fp32/arg_min_max_fp32.h"
#include <float.h>

#define ARG_MIN_MAX_FUNC(data_type)                                                                                \
  int ArgCompareDesc32##data_type(const void *a, const void *b) {                                                  \
    DATA_TYPE b_value = ((ArgElement *)b)->data_.UNION_DATA;                                                       \
    DATA_TYPE a_value = ((ArgElement *)a)->data_.UNION_DATA;                                                       \
    if (b_value > a_value) {                                                                                       \
      return 1;                                                                                                    \
    }                                                                                                              \
    if (b_value < a_value) {                                                                                       \
      return -1;                                                                                                   \
    }                                                                                                              \
    return 0;                                                                                                      \
  }                                                                                                                \
  int ArgCompareAsc32##data_type(const void *a, const void *b) {                                                   \
    DATA_TYPE a_value = ((ArgElement *)a)->data_.UNION_DATA;                                                       \
    DATA_TYPE b_value = ((ArgElement *)b)->data_.UNION_DATA;                                                       \
    if (b_value > a_value) {                                                                                       \
      return -1;                                                                                                   \
    }                                                                                                              \
    if (b_value < a_value) {                                                                                       \
      return 1;                                                                                                    \
    }                                                                                                              \
    return 0;                                                                                                      \
  }                                                                                                                \
                                                                                                                   \
  void ArgMaxTopK1##data_type(const DATA_TYPE *input, void *output, DATA_TYPE *output_value,                       \
                              const ArgMinMaxComputeParam *param, int pre_axis_count, int axis_count,              \
                              int after_axis_count) {                                                              \
    bool out_value = param->out_value_;                                                                            \
    DATA_TYPE *outputfp32 = (DATA_TYPE *)output;                                                                   \
    int32_t *outputint = (int32_t *)output;                                                                        \
    for (int i = 0; i < pre_axis_count; ++i) {                                                                     \
      int output_offset = i * after_axis_count;                                                                    \
      int input_offset = output_offset * axis_count;                                                               \
      for (int j = 0; j < after_axis_count; ++j) {                                                                 \
        DATA_TYPE value = MIN_VALUE;                                                                               \
        int index = 0;                                                                                             \
        for (int k = 0; k < axis_count; ++k) {                                                                     \
          DATA_TYPE value_tmp = input[input_offset + k * after_axis_count + j];                                    \
          if (value_tmp > value) {                                                                                 \
            value = value_tmp;                                                                                     \
            index = k;                                                                                             \
          }                                                                                                        \
        }                                                                                                          \
        if (out_value) {                                                                                           \
          outputfp32[output_offset + j] = value;                                                                   \
        } else {                                                                                                   \
          outputint[output_offset + j] = index;                                                                    \
        }                                                                                                          \
        if (output_value != NULL) {                                                                                \
          output_value[output_offset + j] = value;                                                                 \
        }                                                                                                          \
      }                                                                                                            \
    }                                                                                                              \
  }                                                                                                                \
                                                                                                                   \
  void ArgMinTopK1##data_type(const DATA_TYPE *input, void *output, DATA_TYPE *output_value,                       \
                              const ArgMinMaxComputeParam *param, int pre_axis_count, int axis_count,              \
                              int after_axis_count) {                                                              \
    bool out_value = param->out_value_;                                                                            \
    DATA_TYPE *outputfp32 = (DATA_TYPE *)output;                                                                   \
    int32_t *outputint = (int32_t *)output;                                                                        \
    for (int i = 0; i < pre_axis_count; ++i) {                                                                     \
      int output_offset = i * after_axis_count;                                                                    \
      int input_offset = output_offset * axis_count;                                                               \
      for (int j = 0; j < after_axis_count; ++j) {                                                                 \
        DATA_TYPE value = MAX_VALUE;                                                                               \
        int index = 0;                                                                                             \
        for (int k = 0; k < axis_count; ++k) {                                                                     \
          DATA_TYPE value_tmp = input[input_offset + k * after_axis_count + j];                                    \
          if (value_tmp < value) {                                                                                 \
            value = value_tmp;                                                                                     \
            index = k;                                                                                             \
          }                                                                                                        \
        }                                                                                                          \
        if (out_value) {                                                                                           \
          outputfp32[output_offset + j] = value;                                                                   \
        } else {                                                                                                   \
          outputint[output_offset + j] = index;                                                                    \
        }                                                                                                          \
        if (output_value != NULL) {                                                                                \
          output_value[output_offset + j] = value;                                                                 \
        }                                                                                                          \
      }                                                                                                            \
    }                                                                                                              \
  }                                                                                                                \
                                                                                                                   \
  void ArgMinMaxDim0##data_type(const DATA_TYPE *input, void *output, DATA_TYPE *output_value,                     \
                                const int32_t *in_shape, const ArgMinMaxComputeParam *param,                       \
                                COMPARE_FUNCTION compare_func) {                                                   \
    DATA_TYPE *outputfp32 = (DATA_TYPE *)output;                                                                   \
    int32_t *outputint = (int32_t *)output;                                                                        \
    for (int32_t i = 0; i < param->in_strides_[0]; ++i) {                                                          \
      for (int j = 0; j < in_shape[0]; ++j) {                                                                      \
        int offset = param->in_strides_[0] * j + i;                                                                \
        param->arg_elements_[j].index_ = (uint32_t)j;                                                              \
        param->arg_elements_[j].data_.UNION_DATA = input[offset];                                                  \
      }                                                                                                            \
      qsort(param->arg_elements_, in_shape[0], sizeof(ArgElement), *compare_func);                                 \
      for (int j = 0; j < param->topk_; ++j) {                                                                     \
        int out_offset = j * param->out_strides_[0] + i;                                                           \
        if (param->out_value_) {                                                                                   \
          outputfp32[out_offset] = param->arg_elements_[j].data_.UNION_DATA;                                       \
        } else {                                                                                                   \
          outputint[out_offset] = param->arg_elements_[j].index_;                                                  \
        }                                                                                                          \
        if (output_value != NULL) {                                                                                \
          output_value[out_offset] = param->arg_elements_[j].data_.UNION_DATA;                                     \
        }                                                                                                          \
      }                                                                                                            \
    }                                                                                                              \
    return;                                                                                                        \
  }                                                                                                                \
                                                                                                                   \
  void ArgMinMaxDim1##data_type(const DATA_TYPE *input, void *output, DATA_TYPE *output_value,                     \
                                const int32_t *in_shape, const ArgMinMaxComputeParam *param,                       \
                                COMPARE_FUNCTION compare_func) {                                                   \
    DATA_TYPE *outputfp32 = (DATA_TYPE *)output;                                                                   \
    int32_t *outputint = (int32_t *)output;                                                                        \
    int in_shape1 = in_shape[1];                                                                                   \
    for (int i = 0; i < in_shape[0]; ++i) {                                                                        \
      int in_dim0_offset = i * param->in_strides_[0];                                                              \
      int out_dim0_offset = i * param->out_strides_[0];                                                            \
      for (int j = 0; j < param->in_strides_[1]; ++j) {                                                            \
        for (int k = 0; k < in_shape1; ++k) {                                                                      \
          int offset = param->in_strides_[1] * k + in_dim0_offset + j;                                             \
          param->arg_elements_[k].index_ = (uint32_t)k;                                                            \
          param->arg_elements_[k].data_.UNION_DATA = input[offset];                                                \
        }                                                                                                          \
        qsort(param->arg_elements_, in_shape1, sizeof(ArgElement), *compare_func);                                 \
        for (int k = 0; k < param->topk_; ++k) {                                                                   \
          int out_offset = out_dim0_offset + j + k * param->out_strides_[1];                                       \
          if (param->out_value_) {                                                                                 \
            outputfp32[out_offset] = param->arg_elements_[k].data_.UNION_DATA;                                     \
          } else {                                                                                                 \
            outputint[out_offset] = param->arg_elements_[k].index_;                                                \
          }                                                                                                        \
          if (output_value != NULL) {                                                                              \
            output_value[out_offset] = param->arg_elements_[k].data_.UNION_DATA;                                   \
          }                                                                                                        \
        }                                                                                                          \
      }                                                                                                            \
    }                                                                                                              \
    return;                                                                                                        \
  }                                                                                                                \
                                                                                                                   \
  void ArgMinMaxDim2##data_type(const DATA_TYPE *input, void *output, DATA_TYPE *output_value,                     \
                                const int32_t *in_shape, const ArgMinMaxComputeParam *param,                       \
                                COMPARE_FUNCTION compare_func) {                                                   \
    int in_shape1 = in_shape[1];                                                                                   \
    int in_shape2 = in_shape[2];                                                                                   \
    DATA_TYPE *outputfp32 = (DATA_TYPE *)output;                                                                   \
    int32_t *outputint = (int32_t *)output;                                                                        \
    for (int i = 0; i < in_shape[0]; ++i) {                                                                        \
      int in_dim0_offset = i * param->in_strides_[0];                                                              \
      int out_dim0_offset = i * param->out_strides_[0];                                                            \
      for (int j = 0; j < in_shape1; ++j) {                                                                        \
        int in_dim1_offset = j * param->in_strides_[1] + in_dim0_offset;                                           \
        int out_dim1_offset = j * param->out_strides_[1] + out_dim0_offset;                                        \
        for (int k = 0; k < param->in_strides_[2]; ++k) {                                                          \
          for (int l = 0; l < in_shape2; ++l) {                                                                    \
            int offset = param->in_strides_[2] * l + k + in_dim1_offset;                                           \
            param->arg_elements_[l].index_ = (uint32_t)l;                                                          \
            param->arg_elements_[l].data_.UNION_DATA = input[offset];                                              \
          }                                                                                                        \
          qsort(param->arg_elements_, in_shape2, sizeof(ArgElement), *compare_func);                               \
          for (int l = 0; l < param->topk_; ++l) {                                                                 \
            int out_offset = out_dim1_offset + k + l * param->out_strides_[2];                                     \
            if (param->out_value_) {                                                                               \
              outputfp32[out_offset] = param->arg_elements_[l].data_.UNION_DATA;                                   \
            } else {                                                                                               \
              outputint[out_offset] = param->arg_elements_[l].index_;                                              \
            }                                                                                                      \
            if (output_value != NULL) {                                                                            \
              output_value[out_offset] = param->arg_elements_[l].data_.UNION_DATA;                                 \
            }                                                                                                      \
          }                                                                                                        \
        }                                                                                                          \
      }                                                                                                            \
    }                                                                                                              \
  }                                                                                                                \
                                                                                                                   \
  void ArgMinMaxDim3##data_type(const DATA_TYPE *input, void *output, DATA_TYPE *output_value,                     \
                                const int32_t *in_shape, const ArgMinMaxComputeParam *param,                       \
                                COMPARE_FUNCTION compare_func) {                                                   \
    int in_shape1 = in_shape[1];                                                                                   \
    int in_shape2 = in_shape[2];                                                                                   \
    int in_shape3 = in_shape[3];                                                                                   \
    DATA_TYPE *outputfp32 = (DATA_TYPE *)output;                                                                   \
    int32_t *outputint = (int32_t *)output;                                                                        \
    for (int i = 0; i < in_shape[0]; ++i) {                                                                        \
      int in_dim0_offset = i * param->in_strides_[0];                                                              \
      int out_dim0_offset = i * param->out_strides_[0];                                                            \
      for (int j = 0; j < in_shape1; ++j) {                                                                        \
        int in_dim1_offset = j * param->in_strides_[1] + in_dim0_offset;                                           \
        int out_dim1_offset = j * param->out_strides_[1] + out_dim0_offset;                                        \
        for (int k = 0; k < in_shape2; ++k) {                                                                      \
          int in_dim2_offset = k * param->in_strides_[2] + in_dim1_offset;                                         \
          int out_dim2_offset = k * param->out_strides_[2] + out_dim1_offset;                                      \
          for (int l = 0; l < in_shape3; ++l) {                                                                    \
            int offset = l + in_dim2_offset;                                                                       \
            param->arg_elements_[l].index_ = (uint32_t)l;                                                          \
            param->arg_elements_[l].data_.UNION_DATA = input[offset];                                              \
          }                                                                                                        \
          qsort(param->arg_elements_, in_shape3, sizeof(ArgElement), *compare_func);                               \
          for (int l = 0; l < param->topk_; ++l) {                                                                 \
            int out_offset = out_dim2_offset + l;                                                                  \
            if (param->out_value_) {                                                                               \
              outputfp32[out_offset] = param->arg_elements_[l].data_.UNION_DATA;                                   \
            } else {                                                                                               \
              outputint[out_offset] = (int)(param->arg_elements_[l].index_);                                       \
            }                                                                                                      \
            if (output_value != NULL) {                                                                            \
              output_value[out_offset] = param->arg_elements_[l].data_.UNION_DATA;                                 \
            }                                                                                                      \
          }                                                                                                        \
        }                                                                                                          \
      }                                                                                                            \
    }                                                                                                              \
  }                                                                                                                \
                                                                                                                   \
  void ArgMinMax##data_type##32(const DATA_TYPE *input, void *output, DATA_TYPE *output_value,                     \
                                const int32_t *in_shape, const ArgMinMaxComputeParam *param) {                     \
    if (param->topk_ == 1) {                                                                                       \
      int pre_axis_count = 1;                                                                                      \
      int axis_count = 1;                                                                                          \
      int after_axis_count = 1;                                                                                    \
      ComputeAxisDims(in_shape, param->dims_size_, param->axis_, &pre_axis_count, &axis_count, &after_axis_count); \
                                                                                                                   \
      if (param->get_max_) {                                                                                       \
        ArgMaxTopK1##data_type(input, output, output_value, param, pre_axis_count, axis_count, after_axis_count);  \
      } else {                                                                                                     \
        ArgMinTopK1##data_type(input, output, output_value, param, pre_axis_count, axis_count, after_axis_count);  \
      }                                                                                                            \
      return;                                                                                                      \
    }                                                                                                              \
                                                                                                                   \
    COMPARE_FUNCTION compare_function = NULL;                                                                      \
    if (param->get_max_) {                                                                                         \
      compare_function = ArgCompareDesc32##data_type;                                                              \
    } else {                                                                                                       \
      compare_function = ArgCompareAsc32##data_type;                                                               \
    }                                                                                                              \
                                                                                                                   \
    switch (param->axis_) {                                                                                        \
      case 0:                                                                                                      \
        ArgMinMaxDim0##data_type(input, output, output_value, in_shape, param, compare_function);                  \
        break;                                                                                                     \
      case 1:                                                                                                      \
        ArgMinMaxDim1##data_type(input, output, output_value, in_shape, param, compare_function);                  \
        break;                                                                                                     \
      case 2:                                                                                                      \
        ArgMinMaxDim2##data_type(input, output, output_value, in_shape, param, compare_function);                  \
        break;                                                                                                     \
      case 3:                                                                                                      \
        ArgMinMaxDim3##data_type(input, output, output_value, in_shape, param, compare_function);                  \
        break;                                                                                                     \
    }                                                                                                              \
    return;                                                                                                        \
  }

#define DATA_TYPE float
#define MIN_VALUE -FLT_MAX
#define MAX_VALUE FLT_MAX
#define UNION_DATA f_data_
ARG_MIN_MAX_FUNC(Fp)
#undef DATA_TYPE
#undef MIN_VALUE
#undef MAX_VALUE
#undef UNION_DATA

#define DATA_TYPE int32_t
#define MIN_VALUE INT32_MIN
#define MAX_VALUE INT32_MAX
#define UNION_DATA i_data_
ARG_MIN_MAX_FUNC(Int)
#undef DATA_TYPE
#undef MIN_VALUE
#undef MAX_VALUE
#undef UNION_DATA
