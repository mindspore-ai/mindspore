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
#include "nnacl/fp32/arg_min_max.h"
#include <stdlib.h>
#include <float.h>

int ArgCompareAscFp32(const void *a, const void *b) {
  return ((ArgElement *)a)->data_.f_data_ - ((ArgElement *)b)->data_.f_data_;
}

int ArgCompareDescFp32(const void *a, const void *b) {
  // cmp funtion of qsort must return int type
  auto b_value = ((ArgElement *)b)->data_.f_data_;
  auto a_value = ((ArgElement *)a)->data_.f_data_;
  int res = b_value > a_value ? 1 : -1;
  return res;
}

void ArgMaxDim0OutValue(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  for (int32_t i = 0; i < param->in_strides_[0]; ++i) {
    for (int j = 0; j < in_shape[0]; ++j) {
      size_t offset = param->in_strides_[0] * j + i;
      param->arg_elements_[j].index_ = j;
      param->arg_elements_[j].data_.f_data_ = input[offset];
    }
    qsort(param->arg_elements_, in_shape[0], sizeof(ArgElement), ArgCompareDescFp32);
    for (int j = 0; j < param->topk_; ++j) {
      size_t out_offset = j * param->out_strides_[0] + i;
      output[out_offset] = param->arg_elements_[j].data_.f_data_;
    }
  }
}

void ArgMaxDim0OutIndex(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  for (int32_t i = 0; i < param->in_strides_[0]; ++i) {
    for (int j = 0; j < in_shape[0]; ++j) {
      size_t offset = param->in_strides_[0] * j + i;
      param->arg_elements_[j].index_ = j;
      param->arg_elements_[j].data_.f_data_ = input[offset];
    }
    qsort(param->arg_elements_, in_shape[0], sizeof(ArgElement), ArgCompareDescFp32);
    for (int j = 0; j < param->topk_; ++j) {
      size_t out_offset = j * param->out_strides_[0] + i;
      output[out_offset] = param->arg_elements_[j].index_;
    }
  }
}

void ArgMinDim0OutValue(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  for (int32_t i = 0; i < param->in_strides_[0]; ++i) {
    for (int j = 0; j < in_shape[0]; ++j) {
      size_t offset = param->in_strides_[0] * j + i;
      param->arg_elements_[j].index_ = j;
      param->arg_elements_[j].data_.f_data_ = input[offset];
    }
    qsort(param->arg_elements_, in_shape[0], sizeof(ArgElement), ArgCompareAscFp32);
    for (int j = 0; j < param->topk_; ++j) {
      size_t out_offset = j * param->out_strides_[0] + i;
      output[out_offset] = param->arg_elements_[j].data_.f_data_;
    }
  }
}

void ArgMinDim0OutIndex(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  for (int32_t i = 0; i < param->in_strides_[0]; ++i) {
    for (int j = 0; j < in_shape[0]; ++j) {
      size_t offset = param->in_strides_[0] * j + i;
      param->arg_elements_[j].index_ = j;
      param->arg_elements_[j].data_.f_data_ = input[offset];
    }
    qsort(param->arg_elements_, in_shape[0], sizeof(ArgElement), ArgCompareAscFp32);
    for (int j = 0; j < param->topk_; ++j) {
      size_t out_offset = j * param->out_strides_[0] + i;
      output[out_offset] = param->arg_elements_[j].index_;
    }
  }
}

void ArgMaxDim1OutValue(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  int in_shape1 = in_shape[1];
  for (int i = 0; i < in_shape[0]; ++i) {
    size_t in_dim0_offset = i * param->in_strides_[0];
    size_t out_dim0_offset = i * param->out_strides_[0];
    for (int j = 0; j < param->in_strides_[1]; ++j) {
      for (int k = 0; k < in_shape1; ++k) {
        size_t offset = param->in_strides_[1] * k + in_dim0_offset + j;
        param->arg_elements_[k].index_ = k;
        param->arg_elements_[k].data_.f_data_ = input[offset];
      }
      qsort(param->arg_elements_, in_shape1, sizeof(ArgElement), ArgCompareDescFp32);
      for (int k = 0; k < param->topk_; ++k) {
        size_t out_offset = out_dim0_offset + j + k * param->out_strides_[1];
        output[out_offset] = param->arg_elements_[k].data_.f_data_;
      }
    }
  }
}

void ArgMaxDim1OutIndex(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  int in_shape1 = in_shape[1];
  for (int i = 0; i < in_shape[0]; ++i) {
    size_t in_dim0_offset = i * param->in_strides_[0];
    size_t out_dim0_offset = i * param->out_strides_[0];
    for (int j = 0; j < param->in_strides_[1]; ++j) {
      for (int k = 0; k < in_shape1; ++k) {
        size_t offset = param->in_strides_[1] * k + in_dim0_offset + j;
        param->arg_elements_[k].index_ = k;
        param->arg_elements_[k].data_.f_data_ = input[offset];
      }
      qsort(param->arg_elements_, in_shape1, sizeof(ArgElement), ArgCompareDescFp32);
      for (int k = 0; k < param->topk_; ++k) {
        size_t out_offset = out_dim0_offset + j + k * param->out_strides_[1];
        output[out_offset] = param->arg_elements_[k].index_;
      }
    }
  }
}

void ArgMinDim1OutValue(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  int in_shape1 = in_shape[1];
  for (int i = 0; i < in_shape[0]; ++i) {
    size_t in_dim0_offset = i * param->in_strides_[0];
    size_t out_dim0_offset = i * param->out_strides_[0];
    for (int j = 0; j < param->in_strides_[1]; ++j) {
      for (int k = 0; k < in_shape1; ++k) {
        size_t offset = param->in_strides_[1] * k + in_dim0_offset + j;
        param->arg_elements_[k].index_ = k;
        param->arg_elements_[k].data_.f_data_ = input[offset];
      }
      qsort(param->arg_elements_, in_shape1, sizeof(ArgElement), ArgCompareAscFp32);
      for (int k = 0; k < param->topk_; ++k) {
        size_t out_offset = out_dim0_offset + j + k * param->out_strides_[1];
        output[out_offset] = param->arg_elements_[k].data_.f_data_;
      }
    }
  }
}

void ArgMinDim1OutIndex(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  int in_shape1 = in_shape[1];
  for (int i = 0; i < in_shape[0]; ++i) {
    size_t in_dim0_offset = i * param->in_strides_[0];
    size_t out_dim0_offset = i * param->out_strides_[0];
    for (int j = 0; j < param->in_strides_[1]; ++j) {
      for (int k = 0; k < in_shape1; ++k) {
        size_t offset = param->in_strides_[1] * k + in_dim0_offset + j;
        param->arg_elements_[k].index_ = k;
        param->arg_elements_[k].data_.f_data_ = input[offset];
      }
      qsort(param->arg_elements_, in_shape1, sizeof(ArgElement), ArgCompareAscFp32);
      for (int k = 0; k < param->topk_; ++k) {
        size_t out_offset = out_dim0_offset + j + k * param->out_strides_[1];
        output[out_offset] = param->arg_elements_[k].index_;
      }
    }
  }
}

void ArgMaxDim2OutValue(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  int in_shape1 = in_shape[1];
  int in_shape2 = in_shape[2];
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
          param->arg_elements_[l].data_.f_data_ = input[offset];
        }
        qsort(param->arg_elements_, in_shape2, sizeof(ArgElement), ArgCompareDescFp32);
        for (int l = 0; l < param->topk_; ++l) {
          size_t out_offset = out_dim1_offset + k + l * param->out_strides_[2];
          output[out_offset] = param->arg_elements_[l].data_.f_data_;
        }
      }
    }
  }
}

void ArgMaxDim2OutIndex(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  int in_shape1 = in_shape[1];
  int in_shape2 = in_shape[2];
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
          param->arg_elements_[l].data_.f_data_ = input[offset];
        }
        qsort(param->arg_elements_, in_shape2, sizeof(ArgElement), ArgCompareDescFp32);
        for (int l = 0; l < param->topk_; ++l) {
          size_t out_offset = out_dim1_offset + k + l * param->out_strides_[2];
          output[out_offset] = param->arg_elements_[l].index_;
        }
      }
    }
  }
}

void ArgMinDim2OutValue(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  int in_shape1 = in_shape[1];
  int in_shape2 = in_shape[2];
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
          param->arg_elements_[l].data_.f_data_ = input[offset];
        }
        qsort(param->arg_elements_, in_shape2, sizeof(ArgElement), ArgCompareAscFp32);
        for (int l = 0; l < param->topk_; ++l) {
          size_t out_offset = out_dim1_offset + k + l * param->out_strides_[2];
          output[out_offset] = param->arg_elements_[l].data_.f_data_;
        }
      }
    }
  }
}

void ArgMinDim2OutIndex(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  int in_shape1 = in_shape[1];
  int in_shape2 = in_shape[2];
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
          param->arg_elements_[l].data_.f_data_ = input[offset];
        }
        qsort(param->arg_elements_, in_shape2, sizeof(ArgElement), ArgCompareAscFp32);
        for (int l = 0; l < param->topk_; ++l) {
          size_t out_offset = out_dim1_offset + k + l * param->out_strides_[2];
          output[out_offset] = param->arg_elements_[l].index_;
        }
      }
    }
  }
}

void ArgMaxDim3OutValue(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  int in_shape1 = in_shape[1];
  int in_shape2 = in_shape[2];
  int in_shape3 = in_shape[3];
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
          param->arg_elements_[l].data_.f_data_ = input[offset];
        }
        qsort(param->arg_elements_, in_shape3, sizeof(ArgElement), ArgCompareDescFp32);
        for (int l = 0; l < param->topk_; ++l) {
          size_t out_offset = out_dim2_offset + l;
          output[out_offset] = param->arg_elements_[l].data_.f_data_;
        }
      }
    }
  }
}

void ArgMaxDim3OutIndex(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  int in_shape1 = in_shape[1];
  int in_shape2 = in_shape[2];
  int in_shape3 = in_shape[3];
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
          param->arg_elements_[l].data_.f_data_ = input[offset];
        }
        qsort(param->arg_elements_, in_shape3, sizeof(ArgElement), ArgCompareDescFp32);
        for (int l = 0; l < param->topk_; ++l) {
          size_t out_offset = out_dim2_offset + l;
          output[out_offset] = param->arg_elements_[l].index_;
        }
      }
    }
  }
}

void ArgMinDim3OutValue(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  int in_shape1 = in_shape[1];
  int in_shape2 = in_shape[2];
  int in_shape3 = in_shape[3];
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
          param->arg_elements_[l].data_.f_data_ = input[offset];
        }
        qsort(param->arg_elements_, in_shape3, sizeof(ArgElement), ArgCompareAscFp32);
        for (int l = 0; l < param->topk_; ++l) {
          size_t out_offset = out_dim2_offset + l;
          output[out_offset] = param->arg_elements_[l].data_.f_data_;
        }
      }
    }
  }
}

void ArgMinDim3OutIndex(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  int in_shape1 = in_shape[1];
  int in_shape2 = in_shape[2];
  int in_shape3 = in_shape[3];
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
          param->arg_elements_[l].data_.f_data_ = input[offset];
        }
        qsort(param->arg_elements_, in_shape3, sizeof(ArgElement), ArgCompareAscFp32);
        for (int l = 0; l < param->topk_; ++l) {
          size_t out_offset = out_dim2_offset + l;
          output[out_offset] = param->arg_elements_[l].index_;
        }
      }
    }
  }
}

void ArgMaxDim0(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  if (param->out_value_) {
    ArgMaxDim0OutValue(input, output, in_shape, param);
  } else {
    ArgMaxDim0OutIndex(input, output, in_shape, param);
  }
}

void ArgMinDim0(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  if (param->out_value_) {
    ArgMinDim0OutValue(input, output, in_shape, param);
  } else {
    ArgMinDim0OutIndex(input, output, in_shape, param);
  }
}

void ArgMaxDim1(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  if (param->out_value_) {
    ArgMaxDim1OutValue(input, output, in_shape, param);
  } else {
    ArgMaxDim1OutIndex(input, output, in_shape, param);
  }
}

void ArgMinDim1(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  if (param->out_value_) {
    ArgMinDim1OutValue(input, output, in_shape, param);
  } else {
    ArgMinDim1OutIndex(input, output, in_shape, param);
  }
}

void ArgMaxDim2(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  if (param->out_value_) {
    ArgMaxDim2OutValue(input, output, in_shape, param);
  } else {
    ArgMaxDim2OutIndex(input, output, in_shape, param);
  }
}

void ArgMinDim2(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  if (param->out_value_) {
    ArgMinDim2OutValue(input, output, in_shape, param);
  } else {
    ArgMinDim2OutIndex(input, output, in_shape, param);
  }
}

void ArgMaxDim3(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  if (param->out_value_) {
    ArgMaxDim3OutValue(input, output, in_shape, param);
  } else {
    ArgMaxDim3OutIndex(input, output, in_shape, param);
  }
}

void ArgMinDim3(const float *input, float *output, const int *in_shape, ArgMinMaxParameter *param) {
  if (param->out_value_) {
    ArgMinDim3OutValue(input, output, in_shape, param);
  } else {
    ArgMinDim3OutIndex(input, output, in_shape, param);
  }
}

void ArgMax(const float *input, float *output, ArgMinMaxParameter *param, int pre_axis_count, int axis_count,
            int after_axis_count) {
  bool out_value = param->out_value_;
  for (int i = 0; i < pre_axis_count; ++i) {
    size_t output_offset = i * after_axis_count;
    size_t input_offset = output_offset * axis_count;
    for (int j = 0; j < after_axis_count; ++j) {
      float value = -FLT_MAX;
      float index = 0.0f;
      for (int k = 0; k < axis_count; ++k) {
        float value_tmp = input[input_offset + k * after_axis_count + j];
        if (value_tmp > value) {
          value = value_tmp;
          index = k;
        }
      }
      output[output_offset + j] = out_value ? value : index;
    }
  }
}

void ArgMin(const float *input, float *output, ArgMinMaxParameter *param, int pre_axis_count, int axis_count,
            int after_axis_count) {
  bool out_value = param->out_value_;
  for (int i = 0; i < pre_axis_count; ++i) {
    size_t output_offset = i * after_axis_count;
    size_t input_offset = output_offset * axis_count;
    for (int j = 0; j < after_axis_count; ++j) {
      float value = FLT_MAX;
      float index = 0.0f;
      for (int k = 0; k < axis_count; ++k) {
        float value_tmp = input[input_offset + k * after_axis_count + j];
        if (value_tmp < value) {
          value = value_tmp;
          index = k;
        }
      }
      output[output_offset + j] = out_value ? value : index;
    }
  }
}
