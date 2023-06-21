/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "nnacl/nnacl_common.h"
#include "nnacl/errorcode.h"

int PadStridedSliceParameterTo8D(StridedSliceStruct *strided_slice) {
  if (strided_slice->in_shape_size_ > DIMENSION_8D) {
    return NNACL_STRIDED_SLICE_UNSUPPORTED_MAX_8D;
  }

  int32_t begins[DIMENSION_8D];
  int32_t ends[DIMENSION_8D];
  int32_t strides[DIMENSION_8D];
  int32_t input_shape[DIMENSION_8D];
  int32_t i;
  for (i = 0; i < strided_slice->in_shape_size_; ++i) {
    begins[i] = strided_slice->begins_[i];
    ends[i] = MSMIN(strided_slice->ends_[i], strided_slice->in_shape_[i]);
    strides[i] = strided_slice->strides_[i];
    input_shape[i] = strided_slice->in_shape_[i];
  }

  int32_t real_index = strided_slice->in_shape_size_ - 1;
  for (i = DIMENSION_8D - 1; i >= 0; --i) {
    if (real_index >= 0) {
      strided_slice->begins_[i] = begins[real_index];
      strided_slice->ends_[i] = ends[real_index];
      strided_slice->strides_[i] = strides[real_index];
      strided_slice->in_shape_[i] = input_shape[real_index--];
    } else {
      strided_slice->begins_[i] = 0;
      strided_slice->ends_[i] = 1;
      strided_slice->strides_[i] = 1;
      strided_slice->in_shape_[i] = 1;
    }
  }
  strided_slice->in_shape_size_ = DIMENSION_8D;
  return NNACL_OK;
}

bool LoopContinue(int stride, int i, int end) { return stride > 0 ? i < end : i > end; }

int DoStridedSliceIn8D(const void *input, void *output, StridedSliceStruct *strided_slice) {
  NNACL_CHECK_NULL_RETURN_ERR(strided_slice);
  NNACL_CHECK_NULL_RETURN_ERR(input);
  NNACL_CHECK_NULL_RETURN_ERR(output);

  const uint8_t *in = (const uint8_t *)input;
  uint8_t *out = (uint8_t *)output;
  int data_type_size = (int)DataTypeCSize(strided_slice->data_type_);

  int32_t *begins = strided_slice->begins_;
  int32_t *ends = strided_slice->ends_;
  int32_t *strides = strided_slice->strides_;
  int32_t *in_shape = strided_slice->in_shape_;

  int dim_offset[DIMENSION_8D - 1];
  dim_offset[6] = in_shape[7];
  dim_offset[5] = in_shape[6] * dim_offset[6];
  dim_offset[4] = in_shape[5] * dim_offset[5];
  dim_offset[3] = in_shape[4] * dim_offset[4];
  dim_offset[2] = in_shape[3] * dim_offset[3];
  dim_offset[1] = in_shape[2] * dim_offset[2];
  dim_offset[0] = in_shape[1] * dim_offset[1];
  size_t out_offset = 0;
  int32_t dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7;
  for (dim0 = begins[0]; LoopContinue(strides[0], dim0, ends[0]); dim0 += strides[0]) {
    for (dim1 = begins[1]; LoopContinue(strides[1], dim1, ends[1]); dim1 += strides[1]) {
      for (dim2 = begins[2]; LoopContinue(strides[2], dim2, ends[2]); dim2 += strides[2]) {
        for (dim3 = begins[3]; LoopContinue(strides[3], dim3, ends[3]); dim3 += strides[3]) {
          for (dim4 = begins[4]; LoopContinue(strides[4], dim4, ends[4]); dim4 += strides[4]) {
            for (dim5 = begins[5]; LoopContinue(strides[5], dim5, ends[5]); dim5 += strides[5]) {
              for (dim6 = begins[6]; LoopContinue(strides[6], dim6, ends[6]); dim6 += strides[6]) {
                for (dim7 = begins[7]; LoopContinue(strides[7], dim7, ends[7]); dim7 += strides[7]) {
                  int32_t in_offset = dim0 * dim_offset[0] + dim1 * dim_offset[1] + dim2 * dim_offset[2] +
                                      dim3 * dim_offset[3] + dim4 * dim_offset[4] + dim5 * dim_offset[5] +
                                      dim6 * dim_offset[6] + dim7;
                  memcpy(out + out_offset * data_type_size, in + in_offset * data_type_size, data_type_size);
                  out_offset++;
                }
              }
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
  if (stride == 1) {
    size_t unit = split_len * inner_size;
    for (size_t i = 0; i < outer; ++i) {
      memcpy(output, input, unit);
      output += unit;
      input += in_offset;
    }
    return;
  }
  for (size_t i = 0; i < outer; ++i) {
    const uint8_t *input_ptr = input + i * in_offset;
    for (int j = 0; j < split_len; ++j) {
      memcpy(output, input_ptr, inner_size);
      output += inner_size;
      input_ptr += inner_size * stride;
    }
  }
}
