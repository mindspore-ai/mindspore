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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unsorted_segment_prod.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T, typename S>
__global__ void UnsortedSegmentProdCal(size_t input_dim0, size_t input_dim1, size_t output_dim0, size_t output_dim1,
                                       T *input_addr, S *ids_addr, T *output_addr) {
  for (int input_index = blockIdx.x * blockDim.x + threadIdx.x; input_index < input_dim0 * input_dim1;
       input_index += blockDim.x * gridDim.x) {
    size_t j = input_index / input_dim1;
    size_t k = input_index % input_dim1;

    S i = ids_addr[j];
    if (i < 0 || i >= output_dim0) {
      continue;
    }
    size_t output_index = i * output_dim1 + k;
    MsAtomicMul(output_addr + output_index, input_addr[input_index]);
  }
}

template <typename T>
__global__ void UnsortedSegmentProdInit(size_t size, T *output_addr) {
  const T init_value = static_cast<T>(1);
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < size; index += blockDim.x * gridDim.x) {
    output_addr[index] = init_value;
  }
}

template <typename T, typename S>
void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0, size_t output_dim1, T *input_addr,
                         S *ids_addr, T *output_addr, cudaStream_t stream, const uint32_t &device_id) {
  size_t out_size = output_dim0 * output_dim1;
  UnsortedSegmentProdInit<<<CUDA_BLOCKS(device_id, out_size), CUDA_THREADS(device_id), 0, stream>>>(out_size,
                                                                                                    output_addr);

  size_t in_size = input_dim0 * input_dim1;
  UnsortedSegmentProdCal<<<CUDA_BLOCKS(device_id, in_size), CUDA_THREADS(device_id), 0, stream>>>(
    input_dim0, input_dim1, output_dim0, output_dim1, input_addr, ids_addr, output_addr);
  return;
}

template CUDA_LIB_EXPORT void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0,
                                                  size_t output_dim1, double *input_addr, int *ids_addr,
                                                  double *output_addr, cudaStream_t stream, const uint32_t &device_id);
template CUDA_LIB_EXPORT void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0,
                                                  size_t output_dim1, double *input_addr, int64_t *ids_addr,
                                                  double *output_addr, cudaStream_t stream, const uint32_t &device_id);

template CUDA_LIB_EXPORT void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0,
                                                  size_t output_dim1, float *input_addr, int *ids_addr,
                                                  float *output_addr, cudaStream_t stream, const uint32_t &device_id);
template CUDA_LIB_EXPORT void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0,
                                                  size_t output_dim1, float *input_addr, int64_t *ids_addr,
                                                  float *output_addr, cudaStream_t stream, const uint32_t &device_id);

template CUDA_LIB_EXPORT void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0,
                                                  size_t output_dim1, half *input_addr, int *ids_addr,
                                                  half *output_addr, cudaStream_t stream, const uint32_t &device_id);
template CUDA_LIB_EXPORT void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0,
                                                  size_t output_dim1, half *input_addr, int64_t *ids_addr,
                                                  half *output_addr, cudaStream_t stream, const uint32_t &device_id);

template CUDA_LIB_EXPORT void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0,
                                                  size_t output_dim1, int *input_addr, int *ids_addr, int *output_addr,
                                                  cudaStream_t stream, const uint32_t &device_id);
template CUDA_LIB_EXPORT void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0,
                                                  size_t output_dim1, int *input_addr, int64_t *ids_addr,
                                                  int *output_addr, cudaStream_t stream, const uint32_t &device_id);

template CUDA_LIB_EXPORT void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0,
                                                  size_t output_dim1, uint32_t *input_addr, int *ids_addr,
                                                  uint32_t *output_addr, cudaStream_t stream,
                                                  const uint32_t &device_id);
template CUDA_LIB_EXPORT void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0,
                                                  size_t output_dim1, uint32_t *input_addr, int64_t *ids_addr,
                                                  uint32_t *output_addr, cudaStream_t stream,
                                                  const uint32_t &device_id);

template CUDA_LIB_EXPORT void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0,
                                                  size_t output_dim1, uint8_t *input_addr, int *ids_addr,
                                                  uint8_t *output_addr, cudaStream_t stream, const uint32_t &device_id);
template CUDA_LIB_EXPORT void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0,
                                                  size_t output_dim1, uint8_t *input_addr, int64_t *ids_addr,
                                                  uint8_t *output_addr, cudaStream_t stream, const uint32_t &device_id);

template CUDA_LIB_EXPORT void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0,
                                                  size_t output_dim1, int16_t *input_addr, int *ids_addr,
                                                  int16_t *output_addr, cudaStream_t stream, const uint32_t &device_id);
template CUDA_LIB_EXPORT void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0,
                                                  size_t output_dim1, int16_t *input_addr, int64_t *ids_addr,
                                                  int16_t *output_addr, cudaStream_t stream, const uint32_t &device_id);

template CUDA_LIB_EXPORT void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0,
                                                  size_t output_dim1, uint64_t *input_addr, int *ids_addr,
                                                  uint64_t *output_addr, cudaStream_t stream,
                                                  const uint32_t &device_id);
template CUDA_LIB_EXPORT void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0,
                                                  size_t output_dim1, uint64_t *input_addr, int64_t *ids_addr,
                                                  uint64_t *output_addr, cudaStream_t stream,
                                                  const uint32_t &device_id);

template CUDA_LIB_EXPORT void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0,
                                                  size_t output_dim1, int8_t *input_addr, int *ids_addr,
                                                  int8_t *output_addr, cudaStream_t stream, const uint32_t &device_id);
template CUDA_LIB_EXPORT void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0,
                                                  size_t output_dim1, int8_t *input_addr, int64_t *ids_addr,
                                                  int8_t *output_addr, cudaStream_t stream, const uint32_t &device_id);

template CUDA_LIB_EXPORT void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0,
                                                  size_t output_dim1, uint16_t *input_addr, int *ids_addr,
                                                  uint16_t *output_addr, cudaStream_t stream,
                                                  const uint32_t &device_id);
template CUDA_LIB_EXPORT void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0,
                                                  size_t output_dim1, uint16_t *input_addr, int64_t *ids_addr,
                                                  uint16_t *output_addr, cudaStream_t stream,
                                                  const uint32_t &device_id);

template CUDA_LIB_EXPORT void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0,
                                                  size_t output_dim1, int64_t *input_addr, int *ids_addr,
                                                  int64_t *output_addr, cudaStream_t stream, const uint32_t &device_id);
template CUDA_LIB_EXPORT void UnsortedSegmentProd(size_t input_dim0, size_t input_dim1, size_t output_dim0,
                                                  size_t output_dim1, int64_t *input_addr, int64_t *ids_addr,
                                                  int64_t *output_addr, cudaStream_t stream, const uint32_t &device_id);
