/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32/sparse_reshape_fp32.h"
#include "nnacl/errorcode.h"
#include "nnacl/common_func.h"

int SparseReshapeInferOutputShapeFp32(int32_t *in_inshape_ptr, int32_t *in_outshape_ptr, int32_t *out_outshape_ptr,
                                      size_t input_rank, size_t output_rank) {
  int64_t dense_size = 1;
  int64_t dividend = 1;
  int64_t out_num = 1;
  int64_t ui = -1;
  for (int64_t i = 0; i < input_rank; i++) {
    dense_size *= *(in_inshape_ptr + i);
  }

  for (int64_t d = 0; d < output_rank; d++) {
    const int32_t size = *(in_outshape_ptr + d);
    if (size == -1) {
      if (ui != -1) {
        return NNACL_ERR;
      }
      ui = d;
    } else {
      if (size < 0) {
        return NNACL_ERR;
      }
      dividend *= size;
      *(out_outshape_ptr + d) = size;
      out_num *= size;
    }
  }
  if (ui != -1) {
    const int64_t missing = dense_size / dividend;
    if (dividend * missing != dense_size) {
      return NNACL_ERR;
    }
    out_num *= missing;
    *(out_outshape_ptr + ui) = missing;
  }

  if (out_num != dense_size) {
    return NNACL_ERR;
  }

  return NNACL_OK;
}

int SparseReshapeInOutCoordTrans(int32_t *in_indices_ptr, int32_t *in_inshape_ptr, int32_t *out_outshape_ptr,
                                 int32_t in_indices_num, int32_t *out_indices_ptr, int32_t *in_stride,
                                 int32_t *out_stride, size_t input_rank, size_t output_rank) {
  in_stride[input_rank - 1] = 1;
  for (int64_t d = input_rank - 2; d >= 0; d--) {
    in_stride[d] = in_stride[d + 1] * in_inshape_ptr[d + 1];
  }

  out_stride[output_rank - 1] = 1;
  for (int64_t d = output_rank - 2; d >= 0; d--) {
    out_stride[d] = out_stride[d + 1] * out_outshape_ptr[d + 1];
  }

  for (int i = 0; i < in_indices_num; i++) {
    int32_t ori_index = 0;
    // input rank and output rank is small, do not need to simd accelerate
    for (int32_t j = 0; j < input_rank; j++) {
      ori_index += in_indices_ptr[i * input_rank + j] * in_stride[j];
    }
    for (int32_t j = 0; j < output_rank; j++) {
      int32_t div_val = ori_index / out_stride[j];
      ori_index = ori_index - out_stride[j] * div_val;
      out_indices_ptr[i * output_rank + j] = div_val;
    }
  }

  return NNACL_OK;
}
