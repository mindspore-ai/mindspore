/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include <iostream>
#include "backend/kernel_compiler/gpu/cuda_impl/gatherv2.cuh"
#include "runtime/device/gpu/cuda_common.h"
template <typename T, typename S>
__device__ void GatherV2Kernel(T *input, S *indices, T *output, size_t output_dim0, size_t output_dim1,
                               size_t output_dim2, size_t input_dim1) {
  int num = output_dim0 * output_dim1 * output_dim2;
  int i, j, k;
  for (int write_index = blockIdx.x * blockDim.x + threadIdx.x; write_index < num;
       write_index += blockDim.x * gridDim.x) {
    i = write_index / (output_dim1 * output_dim2) % output_dim0;
    j = write_index / output_dim2 % output_dim1;
    k = write_index % output_dim2;

    if ((indices[j] >= 0) && (indices[j] < input_dim1)) {
      int read_index = i * input_dim1 * output_dim2 + indices[j] * output_dim2 + k;
      output[write_index] = input[read_index];
    } else {
      output[write_index] = 0;
    }
  }

  return;
}

template <typename T, typename S>
__global__ void GatherV2StaticShapeWrapper(T *input, S *indices, T *output, size_t output_dim0, size_t output_dim1,
                                           size_t output_dim2, size_t input_dim1) {
  GatherV2Kernel(input, indices, output, output_dim0, output_dim1, output_dim2, input_dim1);
}

template <typename T, typename S>
__global__ void GatherV2DynamicShape(T *input, S *indices, T *output, size_t *input_shape_wksp, size_t input_rank,
                                     size_t *indices_shape_wksp, size_t indices_rank, int64_t *axis_wksp,
                                     size_t *output_shape_wksp, const int max_output_size) {
  int gt_id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t axis = (size_t)(*axis_wksp);

  int output_shape_index = 0;
  size_t output_dim0 = 1;
  for (size_t i = 0; i < axis; i++) {
    output_dim0 *= input_shape_wksp[i];

    if (gt_id == 0) {
      output_shape_wksp[output_shape_index] = input_shape_wksp[i];
      output_shape_index++;
    }
  }

  size_t output_dim1 = 1;
  for (size_t i = 0; i < indices_rank; i++) {
    output_dim1 *= indices_shape_wksp[i];

    if (gt_id == 0) {
      output_shape_wksp[output_shape_index] = indices_shape_wksp[i];
      output_shape_index++;
    }
  }

  size_t output_dim2 = 1;
  for (size_t i = axis + 1; i < input_rank; i++) {
    output_dim2 *= indices_shape_wksp[i];

    if (gt_id == 0) {
      output_shape_wksp[output_shape_index] = input_shape_wksp[i];
      output_shape_index++;
    }
  }

  size_t input_dim1 = (size_t)(input_shape_wksp[axis]);

  GatherV2Kernel(input, indices, output, output_dim0, output_dim1, output_dim2, input_dim1);
}

// entry points from gpu kernel's .h file
template <typename T, typename S>
void CalGatherV2StaticShape(T *input, S *indices, T *output, size_t output_dim0, size_t output_dim1, size_t output_dim2,
                            size_t input_dim1, cudaStream_t stream) {
  int size = output_dim0 * output_dim1 * output_dim2;
  GatherV2StaticShapeWrapper<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, indices, output, output_dim0,
                                                                           output_dim1, output_dim2, input_dim1);
  return;
}

template <typename T, typename S>
void CalGatherV2DynamicShape(T *input, S *indices, T *output, size_t *input_shape_wksp, size_t input_rank,
                             size_t *indices_shape_wksp, size_t indices_rank, int64_t *axis_wksp,
                             size_t *output_shape_wksp, const int max_output_size, cudaStream_t stream) {
  GatherV2DynamicShape<<<GET_BLOCKS(max_output_size), GET_THREADS, 0, stream>>>(
    input, indices, output, input_shape_wksp, input_rank, indices_shape_wksp, indices_rank, axis_wksp,
    output_shape_wksp, max_output_size);
}

// template instantiations
template void CalGatherV2StaticShape<float, int>(float *input, int *indices, float *output, size_t output_dim0,
                                                 size_t output_dim1, size_t output_dim2, size_t input_dim1,
                                                 cudaStream_t stream);

template void CalGatherV2StaticShape<half, int>(half *input, int *indices, half *output, size_t output_dim0,
                                                size_t output_dim1, size_t output_dim2, size_t input_dim1,
                                                cudaStream_t stream);

template void CalGatherV2DynamicShape<float, int>(float *input, int *indices, float *output, size_t *input_shape_wksp,
                                                  size_t input_rank, size_t *indices_shape_wksp, size_t indices_rank,
                                                  int64_t *axis_wksp, size_t *output_shape_wksp,
                                                  const int max_output_size, cudaStream_t stream);

template void CalGatherV2DynamicShape<half, int>(half *input, int *indices, half *output, size_t *input_shape_wksp,
                                                 size_t input_rank, size_t *indices_shape_wksp, size_t indices_rank,
                                                 int64_t *axis_wksp, size_t *output_shape_wksp,
                                                 const int max_output_size, cudaStream_t stream);
