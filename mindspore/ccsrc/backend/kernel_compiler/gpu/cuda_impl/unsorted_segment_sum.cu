/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/cuda_impl/unsorted_segment_sum.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"

template<typename T, typename S>
__global__ void UnsortedSegmentSum(size_t input_dim0, size_t input_dim1, size_t output_dim0, size_t output_dim1,
                       T* input_addr, S* ids_addr, T* output_addr) {
  for (int input_index = blockIdx.x * blockDim.x + threadIdx.x; input_index < input_dim0 * input_dim1;
      input_index += blockDim.x * gridDim.x) {
    size_t j = input_index / input_dim1;
    size_t k = input_index % input_dim1;

    S i = ids_addr[j];
    if (i < 0 || i >= output_dim0) {
      continue;
    }
    size_t output_index = i * output_dim1 + k;
    MsAtomicAdd(output_addr + output_index, input_addr[input_index]);
  }
}

template<typename T, typename S>
void UnsortedSegmentSum(size_t input_dim0, size_t input_dim1, size_t output_dim0, size_t output_dim1,
                        T* input_addr, S* ids_addr, T* output_addr, cudaStream_t stream) {
  int size = input_dim0 * input_dim1;
  UnsortedSegmentSum<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input_dim0, input_dim1,
                                  output_dim0, output_dim1, input_addr, ids_addr, output_addr);
  return;
}

template void UnsortedSegmentSum(size_t input_dim0, size_t input_dim1, size_t output_dim0, size_t output_dim1,
                                 double* input_addr, int* ids_addr, double* output_addr, cudaStream_t stream);
template void UnsortedSegmentSum(size_t input_dim0, size_t input_dim1, size_t output_dim0, size_t output_dim1,
                                 double* input_addr, int64_t* ids_addr, double* output_addr, cudaStream_t stream);

template void UnsortedSegmentSum(size_t input_dim0, size_t input_dim1, size_t output_dim0, size_t output_dim1,
                                 float* input_addr, int* ids_addr, float* output_addr, cudaStream_t stream);
template void UnsortedSegmentSum(size_t input_dim0, size_t input_dim1, size_t output_dim0, size_t output_dim1,
                                 float* input_addr, int64_t* ids_addr, float* output_addr, cudaStream_t stream);

template void UnsortedSegmentSum(size_t input_dim0, size_t input_dim1, size_t output_dim0, size_t output_dim1,
                                 int* input_addr, int* ids_addr, int* output_addr, cudaStream_t stream);
template void UnsortedSegmentSum(size_t input_dim0, size_t input_dim1, size_t output_dim0, size_t output_dim1,
                                 int* input_addr, int64_t* ids_addr, int* output_addr, cudaStream_t stream);



