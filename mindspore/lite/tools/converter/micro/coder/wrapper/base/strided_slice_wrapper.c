/*
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

#include "wrapper/base/strided_slice_wrapper.h"
#include "nnacl/fp32/strided_slice_fp32.h"

int DoStridedSlice(const void *in_data, void *out_data, StridedSliceParameter *param) {
  StridedSliceStruct strided_slice;
  memcpy(strided_slice.begins_, param->begins_, MAX_SHAPE_SIZE * sizeof(int));
  memcpy(strided_slice.ends_, param->ends_, MAX_SHAPE_SIZE * sizeof(int));
  memcpy(strided_slice.in_shape_, param->in_shape_, MAX_SHAPE_SIZE * sizeof(int));
  memcpy(strided_slice.strides_, param->strides_, MAX_SHAPE_SIZE * sizeof(int));
  strided_slice.in_shape_size_ = param->in_shape_length_;
  strided_slice.data_type_ = param->data_type;

  if (param->num_axes_ < DIMENSION_8D) {
    PadStridedSliceParameterTo8D(&strided_slice);
  }
  return DoStridedSliceIn8D(in_data, out_data, &strided_slice);
}
