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
#include "nnacl/fp32/crop_fp32.h"
#include <string.h>
#include "nnacl/op_base.h"
#include "nnacl/crop_parameter.h"

void Pad4DOffset(const CropParameter *crop_param, int64_t *offset, int length) {
  int axis = crop_param->axis_;
  for (int i = length - 1; i >= 0; --i) {
    int offset_index = i - axis;
    if (offset_index >= 0 && offset_index < COMM_SHAPE_SIZE) {
      offset[i] = crop_param->offset_[offset_index];
    } else {
      offset[i] = 0;
    }
  }
}

void Crop4D(const float *input, float *output, const int *in_shape, const int *out_shape,
            const CropParameter *crop_param, int thread_id) {
  int64_t offset_pad[DIMENSION_4D] = {0};
  Pad4DOffset(crop_param, offset_pad, DIMENSION_4D);
  int out_shape1 = out_shape[1];
  int out_shape2 = out_shape[2];
  int out_shape3 = out_shape[3];
  size_t out_stride2 = out_shape3;
  size_t out_stride1 = out_stride2 * out_shape2;
  size_t out_stride0 = out_stride1 * out_shape1;
  size_t in_stride2 = in_shape[3];
  size_t in_stride1 = in_stride2 * in_shape[2];
  size_t in_stride0 = in_stride1 * in_shape[1];
  size_t copy_size = out_shape3 * sizeof(float);
  if (crop_param->op_parameter_.thread_num_ == 0) {
    return;
  }
  size_t count_per_thread = UP_DIV(out_shape1, crop_param->op_parameter_.thread_num_);
  size_t thread_stride = thread_id * count_per_thread;
  for (int i = 0; i < out_shape[0]; ++i) {
    size_t out_offset0 = i * out_stride0;
    size_t in_offset0 = (i + offset_pad[0]) * in_stride0 + offset_pad[3];
    for (size_t j = 0; j < count_per_thread; ++j) {
      size_t k = j + thread_stride;
      if (k >= out_shape1) {
        break;
      }
      size_t out_offset1 = k * out_stride1 + out_offset0;
      size_t in_offset1 = (k + offset_pad[1]) * in_stride1 + in_offset0;
      for (int l = 0; l < out_shape2; ++l) {
        size_t out_offset = l * out_stride2 + out_offset1;
        size_t in_offset = (l + offset_pad[2]) * in_stride2 + in_offset1;
        memcpy(output + out_offset, input + in_offset, copy_size);
      }
    }
  }
}

void Crop4DNoParallel(const float *input, float *output, const int *in_shape, const int *out_shape,
                      const CropParameter *crop_param) {
  int64_t offset_pad[DIMENSION_4D] = {0};
  Pad4DOffset(crop_param, offset_pad, DIMENSION_4D);
  size_t in_dim2_stride = in_shape[3];
  size_t in_dim1_stride = in_shape[2] * in_dim2_stride;
  size_t in_dim0_stride = in_dim1_stride * in_shape[1];
  size_t offset_3 = offset_pad[3];
  size_t out_offset = 0;
  size_t copy_num = out_shape[3];
  size_t copy_size = copy_num * sizeof(float);
  size_t in_dim0_end = offset_pad[0] + out_shape[0];
  size_t in_dim1_end = offset_pad[1] + out_shape[1];
  size_t in_dim2_end = offset_pad[2] + out_shape[2];
  for (int i = offset_pad[0]; i < in_dim0_end; ++i) {
    size_t dim0_offset = (size_t)i * in_dim0_stride + offset_3;
    for (int j = offset_pad[1]; j < in_dim1_end; ++j) {
      size_t dim1_offset = (size_t)j * in_dim1_stride + dim0_offset;
      for (int k = offset_pad[2]; k < in_dim2_end; ++k) {
        size_t in_offset = dim1_offset + (size_t)k * in_dim2_stride;
        memcpy(output + out_offset, input + in_offset, copy_size);
        out_offset += copy_num;
      }
    }
  }
}
