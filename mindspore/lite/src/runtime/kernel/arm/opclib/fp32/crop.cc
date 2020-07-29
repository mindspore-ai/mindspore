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
#include "src/runtime/kernel/arm/opclib/fp32/crop.h"
#include <string.h>

void Pad4DOffset(CropParameter *crop_param) {
  int64_t offset_tmp[DIMENSION_4D];
  int axis = crop_param->axis_;
  for (int i = 3; i >= 0; --i) {
    int offset_index = i - axis;
    if (offset_index >= 0) {
      offset_tmp[i] = crop_param->offset_[offset_index];
    } else {
      offset_tmp[i] = 0;
    }
  }
  for (int i = 0; i < DIMENSION_4D; ++i) {
    crop_param->offset_[i] = offset_tmp[i];
  }
}

void Crop4D(const float *input, float *output, const int *in_shape, const int *out_shape, CropParameter *crop_param) {
  Pad4DOffset(crop_param);
  size_t in_dim2_stride = in_shape[3];
  size_t in_dim1_stride = in_shape[2] * in_dim2_stride;
  size_t in_dim0_stride = in_dim1_stride * in_shape[1];
  size_t offset_3 = crop_param->offset_[3];
  size_t out_offset = 0;
  size_t copy_num = out_shape[3];
  size_t copy_size = copy_num * sizeof(float);
  size_t in_dim0_end = crop_param->offset_[0] + out_shape[0];
  size_t in_dim1_end = crop_param->offset_[1] + out_shape[1];
  size_t in_dim2_end = crop_param->offset_[2] + out_shape[2];
  for (int i = crop_param->offset_[0]; i < in_dim0_end; ++i) {
    size_t dim0_offset = i * in_dim0_stride + offset_3;
    for (int j = crop_param->offset_[1]; j < in_dim1_end; ++j) {
      size_t dim1_offset = j * in_dim1_stride + dim0_offset;
      for (int k = crop_param->offset_[2]; k < in_dim2_end; ++k) {
        size_t in_offset = dim1_offset + k * in_dim2_stride;
        memcpy(output + out_offset, input + in_offset, copy_size);
        out_offset += copy_num;
      }
    }
  }
}

