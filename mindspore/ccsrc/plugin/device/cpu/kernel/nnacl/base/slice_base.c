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
#include "nnacl/base/slice_base.h"
#include <string.h>

void InitSliceStruct(SliceStruct *slice, TensorC *in_tensor, TensorC *begin_tensor, TensorC *size_tensor) {
  slice->param_length_ = in_tensor->shape_size_;

  int32_t *begin = (int32_t *)begin_tensor->data_;
  int32_t *size = (int32_t *)size_tensor->data_;

  for (int i = 0; i < slice->param_length_; ++i) {
    slice->shape_[i] = in_tensor->shape_[i];
    slice->begin_[i] = begin[i];
    slice->size_[i] = size[i] < 0 ? slice->shape_[i] - slice->begin_[i] : size[i];
    slice->end_[i] = slice->begin_[i] + slice->size_[i];
  }
  return;
}

void PadSliceParameterTo8D(SliceStruct *param) {
  int32_t begin[DIMENSION_8D];
  int32_t end[DIMENSION_8D];
  int32_t slice_size[DIMENSION_8D];
  int32_t data_shape[DIMENSION_8D];
  for (int32_t i = 0; i < param->param_length_; ++i) {
    begin[i] = param->begin_[i];
    end[i] = param->end_[i];
    slice_size[i] = param->size_[i] < 0 ? param->shape_[i] - begin[i] : param->size_[i];
    data_shape[i] = param->shape_[i];
  }
  int32_t real_index = param->param_length_ - 1;
  for (int32_t i = DIMENSION_8D - 1; i >= 0; --i) {
    if (real_index >= 0) {
      param->begin_[i] = begin[real_index];
      param->end_[i] = end[real_index];
      param->size_[i] = slice_size[real_index];
      param->shape_[i] = data_shape[real_index--];
    } else {
      param->begin_[i] = 0;
      param->end_[i] = 1;
      param->size_[i] = 1;
      param->shape_[i] = 1;
    }
  }
  param->param_length_ = DIMENSION_8D;
}

void DoSlice(const void *input, void *output, const SliceStruct *param, int thread_id, int thread_num, int data_size) {
  int8_t *int8_in = (int8_t *)input;
  int8_t *int8_out = (int8_t *)output;

  int out_stride[8];
  out_stride[7] = 1;
  for (int i = 6; i >= 0; --i) {
    out_stride[i] = out_stride[i + 1] * param->size_[i + 1];
  }
  int count_per_thread = UP_DIV(param->size_[5], thread_num);
  int thread_begin = thread_id * count_per_thread;
  int thread_end = MSMIN(param->size_[5], thread_begin + count_per_thread);
  int copy_size = param->size_[7] * data_size;
  int in_stride[8];
  in_stride[7] = 1;
  for (int i = 6; i >= 0; --i) {
    in_stride[i] = param->shape_[i + 1] * in_stride[i + 1];
  }

  for (int ii = 0; ii < param->size_[0]; ++ii) {
    int out_offset0 = ii * out_stride[0];
    int in_offset0 = (ii + param->begin_[0]) * in_stride[0] + param->begin_[7];
    for (int jj = 0; jj < param->size_[1]; ++jj) {
      int out_offset1 = jj * out_stride[1] + out_offset0;
      int in_offset1 = (jj + param->begin_[1]) * in_stride[1] + in_offset0;
      for (int kk = 0; kk < param->size_[2]; ++kk) {
        int out_offset2 = kk * out_stride[2] + out_offset1;
        int in_offset2 = (kk + param->begin_[2]) * in_stride[2] + in_offset1;
        for (int ll = 0; ll < param->size_[3]; ++ll) {
          int out_offset3 = ll * out_stride[3] + out_offset2;
          int in_offset3 = (ll + param->begin_[3]) * in_stride[3] + in_offset2;
          for (int i = 0; i < param->size_[4]; ++i) {
            int out_offset4 = i * out_stride[4] + out_offset3;
            int in_offset4 = (i + param->begin_[4]) * in_stride[4] + in_offset3;
            for (int j = thread_begin; j < thread_end; ++j) {
              int out_offset5 = j * out_stride[5] + out_offset4;
              int in_offset5 = (j + param->begin_[5]) * in_stride[5] + in_offset4;
              for (int k = 0; k < param->size_[6]; ++k) {
                int out_offset6 = k * out_stride[6] + out_offset5;
                int in_offset6 = (k + param->begin_[6]) * in_stride[6] + in_offset5;
                memcpy(int8_out + out_offset6 * data_size, int8_in + in_offset6 * data_size, copy_size);
              }
            }
          }
        }
      }
    }
  }
}

static bool WhetherCopyByAxis(const int32_t *begin, const int32_t *end, const int32_t *shape, int dim) {
  for (int i = dim + 1; i < DIMENSION_8D; ++i) {
    if (begin[i] != 0 || end[i] != shape[i]) return false;
  }
  return true;
}

void DoSliceNoParallel(const void *input, void *output, const SliceStruct *param, int data_size) {
  int8_t *int8_in = (int8_t *)input;
  int8_t *int8_out = (int8_t *)output;

  int copy_size = param->size_[7] * data_size;
  int in_stride[8];
  in_stride[7] = 1;
  for (int i = 6; i >= 0; --i) {
    in_stride[i] = param->shape_[i + 1] * in_stride[i + 1];
  }
  bool axis_copy_flag[DIMENSION_8D] = {false};
  for (int i = 0; i < DIMENSION_8D; ++i) {
    axis_copy_flag[i] = WhetherCopyByAxis(param->begin_, param->end_, param->shape_, i);
  }
  int out_offset = 0;
  for (int32_t dim0 = param->begin_[0]; dim0 < param->end_[0]; ++dim0) {
    int in_offset0 = dim0 * in_stride[0] + param->begin_[7];
#define FAST_COPY_IF_NEED(rank)                                                      \
  if (axis_copy_flag[rank]) {                                                        \
    int left_block_num = param->end_[rank] - dim##rank;                              \
    memcpy(int8_out + out_offset * data_size, int8_in + in_offset##rank * data_size, \
           in_stride[rank] * left_block_num * data_size);                            \
    out_offset += in_stride[rank] * left_block_num;                                  \
    dim##rank += left_block_num;                                                     \
    continue;                                                                        \
  }
    FAST_COPY_IF_NEED(0);
    for (int dim1 = param->begin_[1]; dim1 < param->end_[1]; ++dim1) {
      int in_offset1 = dim1 * in_stride[1] + in_offset0;
      FAST_COPY_IF_NEED(1);
      for (int32_t dim2 = param->begin_[2]; dim2 < param->end_[2]; ++dim2) {
        int in_offset2 = in_offset1 + dim2 * in_stride[2];
        FAST_COPY_IF_NEED(2);
        for (int32_t dim3 = param->begin_[3]; dim3 < param->end_[3]; ++dim3) {
          int in_offset3 = in_offset2 + dim3 * in_stride[3];
          FAST_COPY_IF_NEED(3);
          for (int32_t dim4 = param->begin_[4]; dim4 < param->end_[4]; ++dim4) {
            int in_offset4 = in_offset3 + dim4 * in_stride[4];
            FAST_COPY_IF_NEED(4);
            for (int32_t dim5 = param->begin_[5]; dim5 < param->end_[5]; ++dim5) {
              int in_offset5 = in_offset4 + dim5 * in_stride[5];
              FAST_COPY_IF_NEED(5);
#undef FAST_COPY_IF_NEED
              for (int32_t dim6 = param->begin_[6]; dim6 < param->end_[6]; ++dim6) {
                int in_offset6 = in_offset5 + dim6 * in_stride[6];
                memcpy(int8_out + out_offset * data_size, int8_in + in_offset6 * data_size, copy_size);
                out_offset += param->size_[7];
              }
            }
          }
        }
      }
    }
  }
}
