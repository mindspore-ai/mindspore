#ifdef BFC_MEMORY
/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32/transpose_server_fp32.h"

#define JUDGEPART(NUM)                         \
  if (dim_start##NUM == overflow_point##NUM) { \
    dim_start##NUM = 0;                        \
  } else {                                     \
    ++dim_start##NUM;                          \
    in_offset += stride##NUM;                  \
    continue;                                  \
  }

void DoTransposeServerDim3(const float *in_data, float *out_data, const int64_t *overflow_points,
                           const int64_t *strides, const TransposeBlockBoundaryInfo *boundary_info) {
  int64_t stride2 = strides[THIRD_INPUT];
  int64_t size = boundary_info->sizes[0];
  int64_t in_offset = boundary_info->in_offsets[0];
  out_data += boundary_info->out_start_offset;
  for (int64_t i = 0; i < size; ++i) {
    out_data[i] = in_data[in_offset + i * stride2];
  }
  int64_t dim_start1 = boundary_info->start_dim[1];
  int64_t overflow_point1 = overflow_points[1];
  int64_t overflow_point2 = overflow_points[THIRD_INPUT];
  int64_t stride0 = strides[0];
  int64_t stride1 = strides[1];
  int64_t last_dim = overflow_point2 + 1;
  out_data += size;
  size = boundary_info->sizes[1];
  in_offset = boundary_info->in_offsets[1];
  for (int64_t i = 0; i < size; i += last_dim) {
    for (int64_t j = 0; j < overflow_point2; ++j) {
      out_data[i + j] = in_data[in_offset];
      in_offset += stride2;
    }
    out_data[i + overflow_point2] = in_data[in_offset];
    JUDGEPART(1)
    in_offset += stride0;
  }
  out_data += size;
  size = boundary_info->sizes[THIRD_INPUT];
  for (int64_t i = 0; i < size; ++i) {
    out_data[i] = in_data[in_offset + i * stride2];
  }
}

void DoTransposeServerDim4(const float *in_data, float *out_data, const int64_t *overflow_points,
                           const int64_t *strides, const TransposeBlockBoundaryInfo *boundary_info) {
  int64_t stride3 = strides[FOURTH_INPUT];
  int64_t size = boundary_info->sizes[0];
  int64_t in_offset = boundary_info->in_offsets[0];
  out_data += boundary_info->out_start_offset;
  for (int64_t i = 0; i < size; ++i) {
    out_data[i] = in_data[in_offset + i * stride3];
  }
  int64_t dim_start1 = boundary_info->start_dim[1];
  int64_t dim_start2 = boundary_info->start_dim[THIRD_INPUT];
  int64_t overflow_point1 = overflow_points[1];
  int64_t overflow_point2 = overflow_points[THIRD_INPUT];
  int64_t overflow_point3 = overflow_points[FOURTH_INPUT];
  int64_t stride0 = strides[0];
  int64_t stride1 = strides[1];
  int64_t stride2 = strides[THIRD_INPUT];
  int64_t last_dim = overflow_point3 + 1;
  out_data += size;
  size = boundary_info->sizes[1];
  in_offset = boundary_info->in_offsets[1];
  for (int64_t i = 0; i < size; i += last_dim) {
    for (int64_t j = 0; j < overflow_point3; ++j) {
      out_data[i + j] = in_data[in_offset];
      in_offset += stride3;
    }
    out_data[i + overflow_point3] = in_data[in_offset];
    JUDGEPART(2)
    JUDGEPART(1)
    in_offset += stride0;
  }
  out_data += size;
  size = boundary_info->sizes[THIRD_INPUT];
  for (int64_t i = 0; i < size; ++i) {
    out_data[i] = in_data[in_offset + i * stride3];
  }
}

void DoTransposeServerDim5(const float *in_data, float *out_data, const int64_t *overflow_points,
                           const int64_t *strides, const TransposeBlockBoundaryInfo *boundary_info) {
  int64_t stride4 = strides[FIFTH_INPUT];
  int64_t size = boundary_info->sizes[0];
  int64_t in_offset = boundary_info->in_offsets[0];
  out_data += boundary_info->out_start_offset;
  for (int64_t i = 0; i < size; ++i) {
    out_data[i] = in_data[in_offset + i * stride4];
  }
  int64_t dim_start1 = boundary_info->start_dim[1];
  int64_t dim_start2 = boundary_info->start_dim[THIRD_INPUT];
  int64_t dim_start3 = boundary_info->start_dim[FOURTH_INPUT];
  int64_t overflow_point1 = overflow_points[1];
  int64_t overflow_point2 = overflow_points[THIRD_INPUT];
  int64_t overflow_point3 = overflow_points[FOURTH_INPUT];
  int64_t overflow_point4 = overflow_points[FIFTH_INPUT];
  int64_t stride0 = strides[0];
  int64_t stride1 = strides[1];
  int64_t stride2 = strides[THIRD_INPUT];
  int64_t stride3 = strides[FOURTH_INPUT];
  int64_t last_dim = overflow_point4 + 1;
  out_data += size;
  size = boundary_info->sizes[1];
  in_offset = boundary_info->in_offsets[1];
  for (int64_t i = 0; i < size; i += last_dim) {
    for (int64_t j = 0; j < overflow_point4; ++j) {
      out_data[i + j] = in_data[in_offset];
      in_offset += stride4;
    }
    out_data[i + overflow_point4] = in_data[in_offset];
    JUDGEPART(3)
    JUDGEPART(2)
    JUDGEPART(1)
    in_offset += stride0;
  }
  out_data += size;
  size = boundary_info->sizes[THIRD_INPUT];
  for (int64_t i = 0; i < size; ++i) {
    out_data[i] = in_data[in_offset + i * stride4];
  }
}

void DoTransposeServerDim6(const float *in_data, float *out_data, const int64_t *overflow_points,
                           const int64_t *strides, const TransposeBlockBoundaryInfo *boundary_info) {
  int64_t stride5 = strides[SIXTH_INPUT];
  int64_t size = boundary_info->sizes[0];
  int64_t in_offset = boundary_info->in_offsets[0];
  out_data += boundary_info->out_start_offset;
  for (int64_t i = 0; i < size; ++i) {
    out_data[i] = in_data[in_offset + i * stride5];
  }
  int64_t dim_start1 = boundary_info->start_dim[1];
  int64_t dim_start2 = boundary_info->start_dim[THIRD_INPUT];
  int64_t dim_start3 = boundary_info->start_dim[FOURTH_INPUT];
  int64_t dim_start4 = boundary_info->start_dim[FIFTH_INPUT];
  int64_t overflow_point1 = overflow_points[1];
  int64_t overflow_point2 = overflow_points[THIRD_INPUT];
  int64_t overflow_point3 = overflow_points[FOURTH_INPUT];
  int64_t overflow_point4 = overflow_points[FIFTH_INPUT];
  int64_t overflow_point5 = overflow_points[SIXTH_INPUT];
  int64_t stride0 = strides[0];
  int64_t stride1 = strides[1];
  int64_t stride2 = strides[THIRD_INPUT];
  int64_t stride3 = strides[FOURTH_INPUT];
  int64_t stride4 = strides[FIFTH_INPUT];
  int64_t last_dim = overflow_point5 + 1;
  out_data += size;
  size = boundary_info->sizes[1];
  in_offset = boundary_info->in_offsets[1];
  for (int64_t i = 0; i < size; i += last_dim) {
    for (int64_t j = 0; j < overflow_point5; ++j) {
      out_data[i + j] = in_data[in_offset];
      in_offset += stride5;
    }
    out_data[i + overflow_point5] = in_data[in_offset];
    JUDGEPART(4)
    JUDGEPART(3)
    JUDGEPART(2)
    JUDGEPART(1)
    in_offset += stride0;
  }
  out_data += size;
  size = boundary_info->sizes[THIRD_INPUT];
  for (int64_t i = 0; i < size; ++i) {
    out_data[i] = in_data[in_offset + i * stride5];
  }
}

void DoTransposeServer(const float *in_data, float *out_data, const int64_t *overflow_points, const int64_t *strides,
                       int axis_num, const TransposeBlockBoundaryInfo *boundary_info) {
  if (axis_num == DIMENSION_3D) {
    DoTransposeServerDim3(in_data, out_data, overflow_points, strides, boundary_info);
    return;
  } else if (axis_num == DIMENSION_4D) {
    DoTransposeServerDim4(in_data, out_data, overflow_points, strides, boundary_info);
    return;
  } else if (axis_num == DIMENSION_5D) {
    DoTransposeServerDim5(in_data, out_data, overflow_points, strides, boundary_info);
    return;
  } else if (axis_num == DIMENSION_6D) {
    DoTransposeServerDim6(in_data, out_data, overflow_points, strides, boundary_info);
    return;
  }
  out_data += boundary_info->out_start_offset;
  int64_t stride = strides[axis_num - 1];
  int64_t size = boundary_info->sizes[0];
  int64_t in_offset = boundary_info->in_offsets[0];
  for (int64_t i = 0; i < size; ++i) {
    out_data[i] = in_data[in_offset + i * stride];
  }
  int64_t dim_info[MAX_TRANSPOSE_DIM_SIZE] = {};
  for (int i = 0; i < axis_num; ++i) {
    dim_info[i] = boundary_info->start_dim[i];
  }
  int64_t last_overflow_point = overflow_points[axis_num - 1];
  int64_t last_dim = last_overflow_point + 1;
  out_data += size;
  size = boundary_info->sizes[1];
  for (int64_t i = 0; i < size; i += last_dim) {
    for (int64_t j = 0; j < last_overflow_point; ++j) {
      out_data[i + j] = in_data[in_offset];
      in_offset += stride;
    }
    out_data[i + last_overflow_point] = in_data[in_offset];
    int j = axis_num - 2;
    while (dim_info[j] == overflow_points[j]) {
      dim_info[j] = 0;
      --j;
    }
    ++dim_info[j];
    in_offset += strides[j];
  }
  out_data += size;
  size = boundary_info->sizes[THIRD_INPUT];
  for (int64_t i = 0; i < size; ++i) {
    out_data[i] = in_data[in_offset + i * stride];
  }
}
#endif
