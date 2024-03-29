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

#include "nnacl/fp16/arg_min_max_fp16.h"

int ArgCompareAscFp16(const void *a, const void *b) {
  float16_t a_value = ((ArgElement *)a)->data_.f16_data_;
  float16_t b_value = ((ArgElement *)b)->data_.f16_data_;
  if (b_value > a_value) {
    return -1;
  }
  if (b_value < a_value) {
    return 1;
  }

  return 0;
}

int ArgCompareDescFp16(const void *a, const void *b) {
  float16_t b_value = ((ArgElement *)b)->data_.f16_data_;
  float16_t a_value = ((ArgElement *)a)->data_.f16_data_;
  if (b_value > a_value) {
    return 1;
  }
  if (b_value < a_value) {
    return -1;
  }

  return 0;
}

void ArgMaxTopK1Fp16(const float16_t *input, void *output, float16_t *output_value, const ArgMinMaxComputeParam *param,
                     int pre_axis_count, int axis_count, int after_axis_count) {
  bool out_value = param->out_value_;
  float16_t *outputfp16 = (float16_t *)output;
  int *outputint = (int *)output;
  for (int i = 0; i < pre_axis_count; ++i) {
    size_t output_offset = i * after_axis_count;
    size_t input_offset = output_offset * axis_count;
    for (int j = 0; j < after_axis_count; ++j) {
      float16_t value = -FLT_MAX;
      int index = 0;
      for (int k = 0; k < axis_count; ++k) {
        float16_t value_tmp = input[input_offset + k * after_axis_count + j];
        if (value_tmp > value) {
          value = value_tmp;
          index = k;
        }
      }
      if (out_value) {
        outputfp16[output_offset + j] = value;
      } else {
        outputint[output_offset + j] = index;
      }
      if (output_value != NULL) {
        output_value[output_offset + j] = value;
      }
    }
  }
}

void ArgMinTopK1Fp16(const float16_t *input, void *output, float16_t *output_value, const ArgMinMaxComputeParam *param,
                     int pre_axis_count, int axis_count, int after_axis_count) {
  bool out_value = param->out_value_;
  float16_t *outputfp16 = (float16_t *)output;
  int *outputint = (int *)output;
  for (int i = 0; i < pre_axis_count; ++i) {
    size_t output_offset = i * after_axis_count;
    size_t input_offset = output_offset * axis_count;
    for (int j = 0; j < after_axis_count; ++j) {
      float16_t value = FLT_MAX;
      int index = 0;
      for (int k = 0; k < axis_count; ++k) {
        float16_t value_tmp = input[input_offset + k * after_axis_count + j];
        if (value_tmp < value) {
          value = value_tmp;
          index = k;
        }
      }
      if (out_value) {
        outputfp16[output_offset + j] = value;
      } else {
        outputint[output_offset + j] = index;
      }
      if (output_value != NULL) {
        output_value[output_offset + j] = value;
      }
    }
  }
}

void ArgMinMaxDim0Fp16(const float16_t *input, void *output, float16_t *output_value, const int *in_shape,
                       const ArgMinMaxComputeParam *param, COMPARE_FUNCTION compare_func) {
  float16_t *outputfp16 = (float16_t *)output;
  int *outputint = (int *)output;
  for (int32_t i = 0; i < param->in_strides_[0]; ++i) {
    for (int j = 0; j < in_shape[0]; ++j) {
      size_t offset = param->in_strides_[0] * j + i;
      param->arg_elements_[j].index_ = j;
      param->arg_elements_[j].data_.f16_data_ = input[offset];
    }
    qsort(param->arg_elements_, in_shape[0], sizeof(ArgElement), *compare_func);
    for (int j = 0; j < param->topk_; ++j) {
      size_t out_offset = j * param->out_strides_[0] + i;
      if (param->out_value_) {
        outputfp16[out_offset] = param->arg_elements_[j].data_.f16_data_;
      } else {
        outputint[out_offset] = param->arg_elements_[j].index_;
      }
      if (output_value != NULL) {
        output_value[out_offset] = param->arg_elements_[j].data_.f16_data_;
      }
    }
  }
  return;
}

void ArgMinMaxDim1Fp16(const float16_t *input, void *output, float16_t *output_value, const int *in_shape,
                       const ArgMinMaxComputeParam *param, COMPARE_FUNCTION compare_func) {
  int in_shape1 = in_shape[1];
  float16_t *outputfp16 = (float16_t *)output;
  int *outputint = (int *)output;
  for (int i = 0; i < in_shape[0]; ++i) {
    size_t in_dim0_offset = i * param->in_strides_[0];
    size_t out_dim0_offset = i * param->out_strides_[0];
    for (int j = 0; j < param->in_strides_[1]; ++j) {
      for (int k = 0; k < in_shape1; ++k) {
        size_t offset = param->in_strides_[1] * k + in_dim0_offset + j;
        param->arg_elements_[k].index_ = k;
        param->arg_elements_[k].data_.f16_data_ = input[offset];
      }
      qsort(param->arg_elements_, in_shape1, sizeof(ArgElement), *compare_func);
      for (int k = 0; k < param->topk_; ++k) {
        size_t out_offset = out_dim0_offset + j + k * param->out_strides_[1];
        if (param->out_value_) {
          outputfp16[out_offset] = param->arg_elements_[k].data_.f16_data_;
        } else {
          outputint[out_offset] = param->arg_elements_[k].index_;
        }
        if (output_value != NULL) {
          output_value[out_offset] = param->arg_elements_[k].data_.f16_data_;
        }
      }
    }
  }
  return;
}

void ArgMinMaxDim2Fp16(const float16_t *input, float16_t *output, float16_t *output_value, const int *in_shape,
                       const ArgMinMaxComputeParam *param, COMPARE_FUNCTION compare_func) {
  int in_shape1 = in_shape[1];
  int in_shape2 = in_shape[2];
  float *outputfp16 = (float *)output;
  int *outputint = (int *)output;
  for (int i = 0; i < in_shape[0]; ++i) {
    size_t in_dim0_offset = i * param->in_strides_[0];
    size_t out_dim0_offset = i * param->out_strides_[0];
    for (int j = 0; j < in_shape1; ++j) {
      size_t in_dim1_offset = j * param->in_strides_[1] + in_dim0_offset;
      size_t out_dim1_offset = j * param->out_strides_[1] + out_dim0_offset;
      for (int k = 0; k < param->in_strides_[2]; ++k) {
        for (int l = 0; l < in_shape2; ++l) {
          size_t offset = param->in_strides_[2] * l + k + in_dim1_offset;
          param->arg_elements_[l].index_ = l;
          param->arg_elements_[l].data_.f16_data_ = input[offset];
        }
        qsort(param->arg_elements_, in_shape2, sizeof(ArgElement), *compare_func);
        for (int l = 0; l < param->topk_; ++l) {
          size_t out_offset = out_dim1_offset + k + l * param->out_strides_[2];
          if (param->out_value_) {
            outputfp16[out_offset] = param->arg_elements_[l].data_.f16_data_;
          } else {
            outputint[out_offset] = param->arg_elements_[l].index_;
          }
          if (output_value != NULL) {
            output_value[out_offset] = param->arg_elements_[l].data_.f16_data_;
          }
        }
      }
    }
  }
}

void ArgMinMaxDim3Fp16(const float16_t *input, float16_t *output, float16_t *output_value, const int *in_shape,
                       const ArgMinMaxComputeParam *param, COMPARE_FUNCTION compare_func) {
  int in_shape1 = in_shape[1];
  int in_shape2 = in_shape[2];
  int in_shape3 = in_shape[3];
  float *outputfp16 = (float *)output;
  int *outputint = (int *)output;
  for (int i = 0; i < in_shape[0]; ++i) {
    size_t in_dim0_offset = i * param->in_strides_[0];
    size_t out_dim0_offset = i * param->out_strides_[0];
    for (int j = 0; j < in_shape1; ++j) {
      size_t in_dim1_offset = j * param->in_strides_[1] + in_dim0_offset;
      size_t out_dim1_offset = j * param->out_strides_[1] + out_dim0_offset;
      for (int k = 0; k < in_shape2; ++k) {
        size_t in_dim2_offset = k * param->in_strides_[2] + in_dim1_offset;
        size_t out_dim2_offset = k * param->out_strides_[2] + out_dim1_offset;
        for (int l = 0; l < in_shape3; ++l) {
          size_t offset = l + in_dim2_offset;
          param->arg_elements_[l].index_ = l;
          param->arg_elements_[l].data_.f16_data_ = input[offset];
        }
        qsort(param->arg_elements_, in_shape3, sizeof(ArgElement), *compare_func);
        for (int l = 0; l < param->topk_; ++l) {
          size_t out_offset = out_dim2_offset + l;
          if (param->out_value_) {
            outputfp16[out_offset] = param->arg_elements_[l].data_.f16_data_;
          } else {
            outputint[out_offset] = param->arg_elements_[l].index_;
          }
          if (output_value != NULL) {
            output_value[out_offset] = param->arg_elements_[l].data_.f16_data_;
          }
        }
      }
    }
  }
}

void ArgMinMaxFp16(const float16_t *input, void *output, float16_t *output_value, const int *in_shape,
                   const ArgMinMaxComputeParam *param) {
  if (param->topk_ == 1) {
    int pre_axis_count = 1;
    int axis_count = 1;
    int after_axis_count = 1;
    ComputeAxisDims(in_shape, param->dims_size_, param->axis_, &pre_axis_count, &axis_count, &after_axis_count);

    if (param->get_max_) {
      ArgMaxTopK1Fp16(input, output, output_value, param, pre_axis_count, axis_count, after_axis_count);
    } else {
      ArgMinTopK1Fp16(input, output, output_value, param, pre_axis_count, axis_count, after_axis_count);
    }
    return;
  }

  COMPARE_FUNCTION compare_function = NULL;
  if (param->get_max_) {
    compare_function = ArgCompareDescFp16;
  } else {
    compare_function = ArgCompareAscFp16;
  }

  switch (param->axis_) {
    case 0:
      ArgMinMaxDim0Fp16(input, output, output_value, in_shape, param, compare_function);
      break;
    case 1:
      ArgMinMaxDim1Fp16(input, output, output_value, in_shape, param, compare_function);
      break;
    case 2:
      ArgMinMaxDim2Fp16(input, output, output_value, in_shape, param, compare_function);
      break;
    case 3:
      ArgMinMaxDim3Fp16(input, output, output_value, in_shape, param, compare_function);
      break;
  }
  return;
}
