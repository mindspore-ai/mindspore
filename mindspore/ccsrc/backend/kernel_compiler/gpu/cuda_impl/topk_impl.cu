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

#include "backend/kernel_compiler/gpu/cuda_impl/topk_impl.cuh"
#include <limits>
#include <algorithm>

size_t RoundUpPower2(size_t v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

template <typename T>
__inline__ __device__ void Swap(T *lhs, T *rhs) {
  T tmp = lhs[0];
  lhs[0] = rhs[0];
  rhs[0] = tmp;
}

template <typename T, typename S>
__global__ void TopkKernel(const size_t outer, const size_t inner, const size_t ceil_power2, const T *input, const S *k,
                           T *output, S *indices, T *data_buff, S *index_buff) {
  // default: sort with share memory
  extern __shared__ T share_mem[];
  T *data_arr = share_mem;
  S *index_arr = reinterpret_cast<S *>(data_arr + ceil_power2);
  // sort with RAM
  if (data_buff != nullptr && index_buff != nullptr) {
    data_arr = data_buff + blockIdx.x * ceil_power2;
    index_arr = index_buff + blockIdx.x * ceil_power2;
  }

  for (size_t i = threadIdx.x; i < ceil_power2; i += blockDim.x) {
    data_arr[i] = (i < inner) ? input[blockIdx.x * inner + i] : std::numeric_limits<T>::max();
    index_arr[i] = i;
  }
  __syncthreads();

  for (size_t i = 2; i <= ceil_power2; i <<= 1) {
    for (size_t j = (i >> 1); j > 0; j >>= 1) {
      for (size_t tid = threadIdx.x; tid < ceil_power2; tid += blockDim.x) {
        size_t tid_comp = tid ^ j;
        if (tid_comp > tid) {
          if ((tid & i) == 0) {
            if (data_arr[tid] > data_arr[tid_comp]) {
              Swap(&data_arr[tid], &data_arr[tid_comp]);
              Swap(&index_arr[tid], &index_arr[tid_comp]);
            }
          } else {
            if (data_arr[tid] < data_arr[tid_comp]) {
              Swap(&data_arr[tid], &data_arr[tid_comp]);
              Swap(&index_arr[tid], &index_arr[tid_comp]);
            }
          }
        }
      }
      __syncthreads();
    }
  }

  for (size_t tid = threadIdx.x; tid < k[0]; tid += blockDim.x) {
    output[blockIdx.x * k[0] + tid] = data_arr[inner - tid - 1];
    indices[blockIdx.x * k[0] + tid] = index_arr[inner - tid - 1];
  }
}

template <typename T, typename S>
void TopK(const size_t &outer, const size_t &inner, const T *input, const S *k, T *output, S *indices, T *data_buff,
          S *index_buff, cudaStream_t stream) {
  size_t ceil_power2 = RoundUpPower2(inner);
  size_t share_mem = (data_buff == nullptr) ? ceil_power2 * (sizeof(T) + sizeof(S)) : 0;
  size_t thread_num = std::min(ceil_power2, static_cast<size_t>(GET_THREADS));
  TopkKernel<<<outer, thread_num, share_mem, stream>>>(outer, inner, ceil_power2, input, k, output, indices, data_buff,
                                                       index_buff);
}

template <typename T, typename S>
__global__ void BitonicSortByKeyKernel(const size_t outer, const size_t inner, const size_t ceil_power2, T *input,
                                       S *indices, T *data_buff, S *index_buff) {
  // default: sort with share memory
  extern __shared__ T share_mem[];
  T *data_arr = share_mem;
  S *index_arr = reinterpret_cast<S *>(data_arr + ceil_power2);
  // sort with RAM
  if (data_buff != nullptr && index_buff != nullptr) {
    data_arr = data_buff + blockIdx.x * ceil_power2;
    index_arr = index_buff + blockIdx.x * ceil_power2;
  }

  for (size_t i = threadIdx.x; i < ceil_power2; i += blockDim.x) {
    data_arr[i] = (i < inner) ? input[blockIdx.x * inner + i] : std::numeric_limits<T>::max();
    index_arr[i] = (i < inner) ? indices[blockIdx.x * inner + i] : std::numeric_limits<S>::max();
  }
  __syncthreads();

  for (size_t i = 2; i <= ceil_power2; i <<= 1) {
    for (size_t j = (i >> 1); j > 0; j >>= 1) {
      for (size_t tid = threadIdx.x; tid < ceil_power2; tid += blockDim.x) {
        size_t tid_comp = tid ^ j;
        if (tid_comp > tid) {
          if ((tid & i) == 0) {
            if (index_arr[tid] > index_arr[tid_comp]) {
              Swap(&data_arr[tid], &data_arr[tid_comp]);
              Swap(&index_arr[tid], &index_arr[tid_comp]);
            }
          } else {
            if (index_arr[tid] < index_arr[tid_comp]) {
              Swap(&data_arr[tid], &data_arr[tid_comp]);
              Swap(&index_arr[tid], &index_arr[tid_comp]);
            }
          }
        }
      }
      __syncthreads();
    }
  }

  for (size_t tid = threadIdx.x; tid < inner; tid += blockDim.x) {
    input[blockIdx.x * inner + tid] = data_arr[tid];
    indices[blockIdx.x * inner + tid] = index_arr[tid];
  }
}

template <typename T, typename S>
void BitonicSortByKey(const size_t &outer, const size_t &inner, T *input, S *indices, T *data_buff, S *index_buff,
                      cudaStream_t stream) {
  size_t ceil_power2 = RoundUpPower2(inner);
  size_t share_mem = ceil_power2 * (sizeof(T) + sizeof(S));
  if (share_mem > SHARED_MEM_PER_BLOCK) {
    share_mem = 0;
  } else {
    data_buff = nullptr;
    index_buff = nullptr;
  }
  size_t thread_num = std::min(ceil_power2, static_cast<size_t>(GET_THREADS));
  BitonicSortByKeyKernel<<<outer, thread_num, share_mem, stream>>>(outer, inner, ceil_power2, input, indices, data_buff,
                                                                   index_buff);
}

template void TopK(const size_t &outer, const size_t &inner, const float *input_addr, const int *k, float *output,
                   int *indices, float *data_buff, int *index_buff, cudaStream_t stream);
template void BitonicSortByKey(const size_t &outer, const size_t &inner, float *input, int *indices, float *data_buff,
                               int *index_buff, cudaStream_t stream);
