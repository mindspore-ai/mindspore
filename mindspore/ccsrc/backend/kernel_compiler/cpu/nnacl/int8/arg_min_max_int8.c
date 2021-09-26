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
#include "nnacl/int8/arg_min_max_int8.h"
#include <float.h>

void CalcParameter(const int *shape, int dims_number, int axis, int *pre_axis_count, int *axis_count,
                   int *after_axis_count) {
  *pre_axis_count = 1;
  for (int i = 0; i < axis; ++i) {
    *pre_axis_count = (*pre_axis_count) * shape[i];
  }

  *axis_count = shape[axis];

  *after_axis_count = 1;
  for (int i = axis + 1; i < dims_number; ++i) {
    *after_axis_count = (*after_axis_count) * shape[i];
  }
}

void DoArgMinMaxQuant(const int8_t *input, int8_t *output, const ArgMinMaxParameter *param, int pre_axis_count,
                      int axis_count, int after_axis_count, const QuantArg *in_quant_arg,
                      const QuantArg *out_quant_arg) {
  bool out_value = param->out_value_;
  const float output_inverse_scale = 1.f / out_quant_arg->scale_;
  float bias = -in_quant_arg->zp_ * in_quant_arg->scale_;
  int32_t output_zp = out_quant_arg->zp_;
  for (int i = 0; i < pre_axis_count; ++i) {
    int output_offset = i * after_axis_count;
    int input_offset = output_offset * axis_count;
    for (int j = 0; j < after_axis_count; ++j) {
      float value = -FLT_MAX;
      if (!param->get_max_) {
        value = FLT_MAX;
      }
      float index = 0.0f;
      for (int k = 0; k < axis_count; ++k) {
        float value_tmp = input[input_offset + k * after_axis_count + j] * in_quant_arg->scale_ + bias;
        if (param->get_max_) {
          if (value_tmp > value) {
            value = value_tmp;
            index = k;
          }
        } else {
          if (value_tmp < value) {
            value = value_tmp;
            index = k;
          }
        }
      }
      float real_out = out_value ? value : index;
      output[output_offset + j] = real_out * output_inverse_scale + output_zp;
    }
  }
}

void Int8ArgMinMaxQuant(const int8_t *input, int8_t *output, const int *in_shape, const ArgMinMaxParameter *param,
                        const QuantArg *in_quant_arg, const QuantArg *out_quant_arg) {
  int pre_axis_count = 1;
  int axis_count = 1;
  int after_axis_count = 1;
  CalcParameter(in_shape, param->dims_size_, param->axis_, &pre_axis_count, &axis_count, &after_axis_count);
  DoArgMinMaxQuant(input, output, param, pre_axis_count, axis_count, after_axis_count, in_quant_arg, out_quant_arg);
  return;
}

int ArgCompareAscInt8(const void *a, const void *b) {
  return ((ArgElement *)a)->data_.f_data_ - ((ArgElement *)b)->data_.f_data_;
}

int ArgCompareDescInt8(const void *a, const void *b) {
  return ((ArgElement *)b)->data_.f_data_ - ((ArgElement *)a)->data_.f_data_;
}

int8_t GetInt8Output(float real_out, float output_inverse_scale, int32_t output_zp) {
  return real_out * output_inverse_scale + output_zp;
}

void Int8ArgMinMaxDim0(const int8_t *input, int8_t *output, const int *in_shape, ArgMinMaxParameter *param,
                       const QuantArg *in_quant_arg, const QuantArg *out_quant_arg) {
  bool out_value = param->out_value_;
  const float output_inverse_scale = 1.f / out_quant_arg->scale_;
  float bias = -in_quant_arg->zp_ * in_quant_arg->scale_;
  int32_t output_zp = out_quant_arg->zp_;
  for (int32_t i = 0; i < param->in_strides_[0]; ++i) {
    for (int j = 0; j < in_shape[0]; ++j) {
      int offset = param->in_strides_[0] * j + i;
      param->arg_elements_[j].index_ = (uint32_t)j;
      param->arg_elements_[j].data_.f_data_ = input[offset] * in_quant_arg->scale_ + bias;
    }
    if (param->get_max_) {
      qsort(param->arg_elements_, in_shape[0], sizeof(ArgElement), ArgCompareDescInt8);
    } else {
      qsort(param->arg_elements_, in_shape[0], sizeof(ArgElement), ArgCompareAscInt8);
    }

    for (int j = 0; j < param->topk_; ++j) {
      int out_offset = j * param->out_strides_[0] + i;
      float real_out = out_value ? param->arg_elements_[j].data_.f_data_ : param->arg_elements_[j].index_;
      output[out_offset] = GetInt8Output(real_out, output_inverse_scale, output_zp);
    }
  }
}

void Int8ArgMinMaxDim1(const int8_t *input, int8_t *output, const int *in_shape, ArgMinMaxParameter *param,
                       const QuantArg *in_quant_arg, const QuantArg *out_quant_arg) {
  bool out_value = param->out_value_;
  const float output_inverse_scale = 1.f / out_quant_arg->scale_;
  float bias = -in_quant_arg->zp_ * in_quant_arg->scale_;
  int32_t output_zp = out_quant_arg->zp_;
  int in_shape1 = in_shape[1];
  for (int i = 0; i < in_shape[0]; ++i) {
    int in_dim0_offset = i * param->in_strides_[0];
    int out_dim0_offset = i * param->out_strides_[0];
    for (int j = 0; j < param->in_strides_[1]; ++j) {
      for (int k = 0; k < in_shape1; ++k) {
        int offset = param->in_strides_[1] * k + in_dim0_offset + j;
        param->arg_elements_[k].index_ = (size_t)k;
        param->arg_elements_[k].data_.f_data_ = input[offset] * in_quant_arg->scale_ + bias;
      }
      if (param->get_max_) {
        qsort(param->arg_elements_, in_shape1, sizeof(ArgElement), ArgCompareDescInt8);
      } else {
        qsort(param->arg_elements_, in_shape1, sizeof(ArgElement), ArgCompareAscInt8);
      }

      for (int k = 0; k < param->topk_; ++k) {
        int out_offset = out_dim0_offset + j + k * param->out_strides_[1];
        float real_out = out_value ? param->arg_elements_[k].data_.f_data_ : param->arg_elements_[k].index_;
        output[out_offset] = GetInt8Output(real_out, output_inverse_scale, output_zp);
      }
    }
  }
}

void Int8ArgMinMaxDim2(const int8_t *input, int8_t *output, const int *in_shape, ArgMinMaxParameter *param,
                       const QuantArg *in_quant_arg, const QuantArg *out_quant_arg) {
  bool out_value = param->out_value_;
  const float output_inverse_scale = 1.f / out_quant_arg->scale_;
  float bias = -in_quant_arg->zp_ * in_quant_arg->scale_;
  int32_t output_zp = out_quant_arg->zp_;
  int in_shape1 = in_shape[1];
  int in_shape2 = in_shape[2];
  for (int i = 0; i < in_shape[0]; ++i) {
    int in_dim0_offset = i * param->in_strides_[0];
    int out_dim0_offset = i * param->out_strides_[0];
    for (int j = 0; j < in_shape1; ++j) {
      int in_dim1_offset = j * param->in_strides_[1] + in_dim0_offset;
      int out_dim1_offset = j * param->out_strides_[1] + out_dim0_offset;
      for (int k = 0; k < param->in_strides_[2]; ++k) {
        for (int l = 0; l < in_shape2; ++l) {
          int offset = param->in_strides_[2] * l + k + in_dim1_offset;
          param->arg_elements_[l].index_ = (uint32_t)l;
          param->arg_elements_[l].data_.f_data_ = input[offset] * in_quant_arg->scale_ + bias;
        }
        if (param->get_max_) {
          qsort(param->arg_elements_, in_shape2, sizeof(ArgElement), ArgCompareDescInt8);
        } else {
          qsort(param->arg_elements_, in_shape2, sizeof(ArgElement), ArgCompareAscInt8);
        }
        for (int l = 0; l < param->topk_; ++l) {
          int out_offset = out_dim1_offset + k + l * param->out_strides_[2];
          float real_out = out_value ? param->arg_elements_[l].data_.f_data_ : param->arg_elements_[l].index_;
          output[out_offset] = GetInt8Output(real_out, output_inverse_scale, output_zp);
        }
      }
    }
  }
}

void Int8ArgMinMaxDim3(const int8_t *input, int8_t *output, const int *in_shape, ArgMinMaxParameter *param,
                       const QuantArg *in_quant_arg, const QuantArg *out_quant_arg) {
  bool out_value = param->out_value_;
  const float output_inverse_scale = 1.f / out_quant_arg->scale_;
  float bias = -in_quant_arg->zp_ * in_quant_arg->scale_;
  int32_t output_zp = out_quant_arg->zp_;
  int in_shape1 = in_shape[1];
  int in_shape2 = in_shape[2];
  int in_shape3 = in_shape[3];
  for (int i = 0; i < in_shape[0]; ++i) {
    int in_dim0_offset = i * param->in_strides_[0];
    int out_dim0_offset = i * param->out_strides_[0];
    for (int j = 0; j < in_shape1; ++j) {
      int in_dim1_offset = j * param->in_strides_[1] + in_dim0_offset;
      int out_dim1_offset = j * param->out_strides_[1] + out_dim0_offset;
      for (int k = 0; k < in_shape2; ++k) {
        int in_dim2_offset = k * param->in_strides_[2] + in_dim1_offset;
        int out_dim2_offset = k * param->out_strides_[2] + out_dim1_offset;
        for (int l = 0; l < in_shape3; ++l) {
          int offset = l + in_dim2_offset;
          param->arg_elements_[l].index_ = (uint32_t)l;
          param->arg_elements_[l].data_.f_data_ = input[offset] * in_quant_arg->scale_ + bias;
        }
        if (param->get_max_) {
          qsort(param->arg_elements_, in_shape3, sizeof(ArgElement), ArgCompareDescInt8);
        } else {
          qsort(param->arg_elements_, in_shape3, sizeof(ArgElement), ArgCompareAscInt8);
        }
        for (int l = 0; l < param->topk_; ++l) {
          int out_offset = out_dim2_offset + l;
          float real_out = out_value ? param->arg_elements_[l].data_.f_data_ : param->arg_elements_[l].index_;
          output[out_offset] = GetInt8Output(real_out, output_inverse_scale, output_zp);
        }
      }
    }
  }
}
