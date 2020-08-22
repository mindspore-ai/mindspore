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

#include "backend/kernel_compiler/gpu/cuda_impl/random_choice_with_mask_impl.cuh"
#include <algorithm>

int RcwmRoundUpPower2(int v) {
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
__global__ void InitArray(const int input_size, const int ceil_power2, const T *input, S *mask_buff, S *rank_buff) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < ceil_power2; pos += blockDim.x * gridDim.x) {
    mask_buff[pos] = (pos < input_size) ? static_cast<S>(input[pos]) : 0;
    rank_buff[pos] = (pos < input_size && input[pos] != false) ? pos : (ceil_power2 + 1);
  }
}

template <size_t blockSize, typename T>
__device__ void WarpReduce(volatile T *sdata, size_t tid) {
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <size_t blockSize, typename T>
__global__ void ReductionSum(T *g_idata, T *g_odata, size_t n) {
  __shared__ T sdata[blockSize];

  size_t tid = threadIdx.x;
  size_t i = blockIdx.x * (blockSize) + tid;
  size_t gridSize = blockSize * gridDim.x;
  sdata[tid] = 0;

  while (i < n) {
    sdata[tid] += g_idata[i];
    i += gridSize;
  }

  __syncthreads();

  if (blockSize >= 1024) {
    if (tid < 512) {
      sdata[tid] += sdata[tid + 512];
    }
    __syncthreads();
  }
  if (blockSize >= 512) {
    if (tid < 256) {
      sdata[tid] += sdata[tid + 256];
    }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) {
      sdata[tid] += sdata[tid + 64];
    }
    __syncthreads();
  }

  if (tid < 32) WarpReduce<blockSize>(sdata, tid);
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <typename T, typename S>
__global__ void Reshape2Index(const int input_size, const int input_shape_size, const int d1, const int d2,
                              const int d3, const int d4, const int d5, const T *input, S *output_index) {
  int pos_array[MAX_DIMENSION];
  int index_pos;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < input_size; pos += blockDim.x * gridDim.x) {
    pos_array[0] = pos / (d2 * d3 * d4 * d5) % d1;
    pos_array[1] = pos / (d3 * d4 * d5) % d2;
    pos_array[2] = pos / (d4 * d5) % d3;
    pos_array[3] = pos / (d5) % d4;
    pos_array[4] = pos % d5;

    index_pos = pos * input_shape_size;
    if (input[pos] == false) {
      for (int i = 0; i < input_shape_size; i++) {
        output_index[index_pos++] = 0;
      }
    } else {
      for (int i = MAX_DIMENSION - input_shape_size; i < MAX_DIMENSION; i++) {
        output_index[index_pos++] = pos_array[i];
      }
    }
  }
}

template <typename T>
__global__ void Copy(const T *src, T *dst, const int n) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < n; pos += blockDim.x * gridDim.x) {
    dst[pos] = src[pos];
  }
}

template <typename T>
__global__ void Sort(const int ceil_power2, T *rank_buff) {
  for (size_t i = 2; i <= ceil_power2; i <<= 1) {
    for (size_t j = (i >> 1); j > 0; j >>= 1) {
      for (size_t tid = threadIdx.x; tid < ceil_power2; tid += blockDim.x) {
        size_t tid_comp = tid ^ j;
        if (tid_comp > tid) {
          if ((tid & i) == 0) {
            if (rank_buff[tid] > rank_buff[tid_comp]) {
              Swap(&rank_buff[tid], &rank_buff[tid_comp]);
            }
          } else {
            if (rank_buff[tid] < rank_buff[tid_comp]) {
              Swap(&rank_buff[tid], &rank_buff[tid_comp]);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

__global__ void SrandInit(const int ceil_power2, curandState *globalState, const int seedc) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < ceil_power2; i += blockDim.x * gridDim.x) {
    curand_init(seedc, threadIdx.x, 0, &globalState[i]);
  }
}

template <typename T>
__global__ void Shuffle(const int ceil_power2, curandState *globalState, T *rank_buff) {
  int limit = ceil_power2 + 1;
  int value;
  size_t i = ceil_power2;
  for (size_t j = (i >> 1); j > 0; j >>= 1) {
    for (size_t tid = threadIdx.x; tid < ceil_power2; tid += blockDim.x) {
      size_t tid_comp = tid ^ j;
      if (tid_comp > tid) {
        value = static_cast<int>(curand(&globalState[tid]));
        if (value & 1) {
          if (rank_buff[tid] != limit && rank_buff[tid_comp] != limit) {
            Swap(&rank_buff[tid], &rank_buff[tid_comp]);
          }
        }
      }
    }
    __syncthreads();
  }
}

template <typename T, typename S>
__global__ void MoveToOutput(const int input_shape_size, const int count, const T *input, S *output_index,
                             T *output_mask, S *index_buff, S *rank_buff, S *Tnum_buff) {
  int Tnum = static_cast<int>(Tnum_buff[0]);
  int idx = 0;
  int pos;
  if (count <= Tnum) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
      idx = rank_buff[i];
      pos = i;
      output_mask[pos] = input[idx];
      pos *= input_shape_size;
      idx *= input_shape_size;
      for (size_t j = 0; j < input_shape_size; j++) {
        output_index[pos] = index_buff[idx];
        pos++;
        idx++;
      }
    }
  } else {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
      if (i < Tnum) {
        idx = rank_buff[i];
        pos = i;
        output_mask[pos] = input[idx];
        pos *= input_shape_size;
        idx *= input_shape_size;
        for (size_t j = 0; j < input_shape_size; j++) {
          output_index[pos] = index_buff[idx];
          pos++;
          idx++;
        }
      } else {
        pos = i;
        output_mask[pos] = static_cast<T>(0);
        pos *= input_shape_size;
        for (size_t j = 0; j < input_shape_size; j++) {
          output_index[pos] = static_cast<S>(0);
          pos++;
        }
      }
    }
  }
}

template <typename T, typename S>
void CalRandomChoiceWithMask(const int &input_size, const int &input_shape_size, const int &d1, const int &d2,
                             const int &d3, const int &d4, const int &d5, const int &seedc, const int &count,
                             const T *input, S *output_index, T *output_mask, S *index_buff, S *mask_buff, S *rank_buff,
                             S *Tnum_buff, S *tmp_buff, curandState *globalState, cudaStream_t stream) {
  int ceil_power2 = RcwmRoundUpPower2(input_size);

  InitArray<<<GET_BLOCKS(input_size), GET_THREADS, 0, stream>>>(input_size, ceil_power2, input, mask_buff, rank_buff);

  size_t BLOCKNUM;
  size_t n = ceil_power2;
  Copy<<<GET_BLOCKS(input_size), GET_THREADS, 0, stream>>>(mask_buff, tmp_buff, ceil_power2);
  do {
    BLOCKNUM = std::ceil(static_cast<float>(n) / BLOCKSIZE);
    ReductionSum<BLOCKSIZE, S><<<BLOCKNUM, BLOCKSIZE, 0, stream>>>(tmp_buff, Tnum_buff, n);
    Copy<<<GET_BLOCKS(input_size), GET_THREADS, 0, stream>>>(Tnum_buff, tmp_buff, BLOCKNUM);
    n = BLOCKNUM;
  } while (n > BLOCKSIZE);
  if (n > 1) ReductionSum<BLOCKSIZE, S><<<1, BLOCKSIZE, 0, stream>>>(Tnum_buff, Tnum_buff, n);

  Reshape2Index<<<GET_BLOCKS(input_size), GET_THREADS, 0, stream>>>(input_size, input_shape_size, d1, d2, d3, d4, d5,
                                                                    input, index_buff);

  Sort<<<1, GET_THREADS, 0, stream>>>(ceil_power2, rank_buff);

  SrandInit<<<GET_BLOCKS(ceil_power2), GET_THREADS, 0, stream>>>(ceil_power2, globalState, seedc);
  Shuffle<<<1, GET_THREADS, 0, stream>>>(ceil_power2, globalState, rank_buff);

  MoveToOutput<<<GET_BLOCKS(count), GET_THREADS, 0, stream>>>(input_shape_size, count, input, output_index, output_mask,
                                                              index_buff, rank_buff, Tnum_buff);
}

template void CalRandomChoiceWithMask(const int &input_size, const int &input_shape_size, const int &d1, const int &d2,
                                      const int &d3, const int &d4, const int &d5, const int &seedc, const int &count,
                                      const bool *input, int *output_index, bool *output_mask, int *index_buff,
                                      int *mask_buff, int *rank_buff, int *Tnum_buff, int *tmp_buff,
                                      curandState *globalState, cudaStream_t stream);
