/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32/strided_slice_fp32.h"
#include "nnacl/errorcode.h"

void PadStridedSliceParameterTo6D(StridedSliceParameter *param) {
  int32_t begins[DIMENSION_6D];
  int32_t ends[DIMENSION_6D];
  int32_t strides[DIMENSION_6D];
  int32_t input_shape[DIMENSION_6D];
  int32_t i;
  for (i = 0; i < param->num_axes_; ++i) {
    begins[i] = param->begins_[i];
    ends[i] = MSMIN(param->ends_[i], param->in_shape_[i]);
    strides[i] = param->strides_[i];
    input_shape[i] = param->in_shape_[i];
  }
  for (i = param->num_axes_; i < param->in_shape_length_; ++i) {
    input_shape[i] = param->in_shape_[i];
    begins[i] = 0;
    ends[i] = param->in_shape_[i];
    strides[i] = 1;
  }

  int32_t real_index = param->in_shape_length_ - 1;
  for (i = DIMENSION_6D - 1; i >= 0; --i) {
    if (real_index >= 0) {
      param->begins_[i] = begins[real_index];
      param->ends_[i] = ends[real_index];
      param->strides_[i] = strides[real_index];
      param->in_shape_[i] = input_shape[real_index--];
    } else {
      param->begins_[i] = 0;
      param->ends_[i] = 1;
      param->strides_[i] = 1;
      param->in_shape_[i] = 1;
    }
  }
  param->num_axes_ = DIMENSION_6D;
  param->in_shape_length_ = DIMENSION_6D;
}

void ChangeNegToPositive(StridedSliceParameter *param) {
  int i;
  for (i = 0; i < DIMENSION_6D; ++i) {
    if (param->begins_[i] < 0) {
      param->begins_[i] += param->in_shape_[i];
    }
    if (param->ends_[i] < 0) {
      param->ends_[i] += param->in_shape_[i];
    }
  }
}

bool LoopContinue(int stride, int i, int end) { return stride > 0 ? i < end : i > end; }

int DoStridedSlice(const void *in_data, void *out_data, StridedSliceParameter *param) {
  if (in_data == NULL || out_data == NULL || param == NULL) {
    return NNACL_NULL_PTR;
  }
  if (param->num_axes_ > DIMENSION_6D) {
    return NNACL_PARAM_INVALID;
  }

  int *begins = param->begins_;
  int *ends = param->ends_;
  int *strides = param->strides_;
  int *in_shape = param->in_shape_;

  if (param->num_axes_ < DIMENSION_6D) {
    PadStridedSliceParameterTo6D(param);
  }

  size_t dim_offset[DIMENSION_6D - 1];
  dim_offset[4] = in_shape[5];
  dim_offset[3] = in_shape[4] * dim_offset[4];
  dim_offset[2] = in_shape[3] * dim_offset[3];
  dim_offset[1] = in_shape[2] * dim_offset[2];
  dim_offset[0] = in_shape[1] * dim_offset[1];
  size_t out_offset = 0;
  int32_t dim0, dim1, dim2, dim3, dim4, dim5;
  for (dim0 = begins[0]; LoopContinue(strides[0], dim0, ends[0]); dim0 += strides[0]) {
    for (dim1 = begins[1]; LoopContinue(strides[1], dim1, ends[1]); dim1 += strides[1]) {
      for (dim2 = begins[2]; LoopContinue(strides[2], dim2, ends[2]); dim2 += strides[2]) {
        for (dim3 = begins[3]; LoopContinue(strides[3], dim3, ends[3]); dim3 += strides[3]) {
          for (dim4 = begins[4]; LoopContinue(strides[4], dim4, ends[4]); dim4 += strides[4]) {
            for (dim5 = begins[5]; LoopContinue(strides[5], dim5, ends[5]); dim5 += strides[5]) {
              int32_t in_offset = dim0 * dim_offset[0] + dim1 * dim_offset[1] + dim2 * dim_offset[2] +
                                  dim3 * dim_offset[3] + dim4 * dim_offset[4] + dim5;
              if (param->data_type == kDataTypeFloat) {
                *((float *)out_data + out_offset) = *((float *)in_data + in_offset);
              } else if (param->data_type == kDataTypeInt8) {
                *((int8_t *)out_data + out_offset) = *((int8_t *)in_data + in_offset);
              } else if (param->data_type == kDataTypeInt) {
                *((int32_t *)out_data + out_offset) = *((int32_t *)in_data + in_offset);
#ifdef ENABLE_ARM64
              } else if (param->data_type == kDataTypeFloat16) {
                *((float16_t *)out_data + out_offset) = *((float16_t *)in_data + in_offset);
#endif
              } else {
                return NNACL_ERR;
              }
              out_offset++;
            }
          }
        }
      }
    }
  }
  return NNACL_OK;
}

void FastStride(const uint8_t *input, uint8_t *output, int split_len, int stride, size_t outer, size_t inner_size,
                size_t in_offset) {
  for (size_t i = 0; i < outer; ++i) {
    const uint8_t *input_ptr = input + i * in_offset;
    for (int j = 0; j < split_len; ++j) {
      memcpy(output, input_ptr, inner_size);
      output += inner_size;
      input_ptr += inner_size * stride;
    }
  }
}
