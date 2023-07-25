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

#include <limits>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/quantile_impl.cuh"

int RoundUpPower2(int v) {
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

template <typename T>
__global__ void DoQuantile(const T *input, const T *q, T *out, T *sort, const int dim, const int x, const int y,
                           const int z, const int each_q_elements, const int output_elements, const int ceil_p_2,
                           int *nan, const bool ignorenan_) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < output_elements; index += blockDim.x * gridDim.x) {
    if (nan[index % each_q_elements] > 0 && (!ignorenan_ || nan[index % each_q_elements] == y)) {
      out[index] = NAN;
    } else {
      size_t q_index = index / each_q_elements;
      size_t start = static_cast<size_t>((index % each_q_elements) / z) * ceil_p_2 * z + (index % each_q_elements) % z;
      T iq = q[q_index];
      int iqy_int = static_cast<int>(iq * static_cast<T>(y - nan[index % each_q_elements] - 1));
      T iqy_T = static_cast<T>(iq * static_cast<T>(y - nan[index % each_q_elements] - 1));
      int step = z * iqy_int;
      int input_index = start + step;
      out[index] = static_cast<T>(sort[input_index] +
                                  (iqy_T - static_cast<T>(iqy_int)) * (sort[input_index + z] - sort[input_index]));
    }
  }
}

template <typename T>
__global__ void Copy(const T *input, T *sort, const int x, const int ceil_p_2, const int y, const int z) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < x * ceil_p_2 * z; pos += blockDim.x * gridDim.x) {
    size_t input_x = static_cast<size_t>(pos / (ceil_p_2 * z));
    size_t input_y = static_cast<size_t>(pos % (ceil_p_2 * z) / z);
    size_t input_z = pos % z;
    sort[pos] = input_y < y ? input[input_x * y * z + input_y * z + input_z] : std::numeric_limits<T>::max();
  }
}

template <typename T>
__global__ void BitonicSort(const int ceil_power2, T *rank_buff, const int clip_num, const int step) {
  for (size_t clip_i = blockIdx.x; clip_i < clip_num; clip_i += gridDim.x) {
    T *rank_buff_offset = rank_buff + static_cast<size_t>(clip_i / step) * ceil_power2 * step + clip_i % step;
    for (size_t i = 2; i <= ceil_power2; i <<= 1) {
      for (size_t j = (i >> 1); j > 0; j >>= 1) {
        for (size_t tid = threadIdx.x; tid < ceil_power2; tid += blockDim.x) {
          size_t tid_comp = tid ^ j;
          if (tid_comp > tid) {
            if ((tid & i) == 0) {
              if (rank_buff_offset[tid * step] > rank_buff_offset[tid_comp * step] ||
                  rank_buff_offset[tid * step] != rank_buff_offset[tid * step]) {
                Swap(&rank_buff_offset[tid * step], &rank_buff_offset[tid_comp * step]);
              }
            } else {
              if (rank_buff_offset[tid * step] < rank_buff_offset[tid_comp * step] ||
                  rank_buff_offset[tid_comp * step] != rank_buff_offset[tid_comp * step]) {
                Swap(&rank_buff_offset[tid * step], &rank_buff_offset[tid_comp * step]);
              }
            }
          }
        }
        __syncthreads();
      }
    }
  }
}

template <typename T>
__global__ void QuantileKernelCheckQ(int num, const T *q, int *flag_in) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += gridDim.x * blockDim.x) {
    if (q[i] < 0 || q[i] > 1) {
      *flag_in = 1;
      return;
    }
  }
}

template <typename T>
__global__ void CountNan(int x, int y, int z, int num, const T *input, int *flag_in) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += gridDim.x * blockDim.x) {
    size_t start = static_cast<size_t>(i / z) * y * z + i % z;
    for (size_t index = start; index < start + z * y; index += z) {
      if ((input[index] != input[index])) {
        flag_in[i] += 1;
      }
    }
  }
}

template <typename T>
__global__ void FlagNan(int x, int y, int z, int num, const T *input, int *flag_in) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += gridDim.x * blockDim.x) {
    if ((input[i] != input[i]) && flag_in[i / (y * z) * z + i % (y * z) % z] != 1) {
      flag_in[i / (y * z) * z + i % (y * z) % z] = 1;
    }
  }
}

template <typename T>
__global__ void CheckNanInit(int x, int y, int z, int num, const T *input, int *flag_in) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += gridDim.x * blockDim.x) {
    flag_in[i] = 0;
  }
}

template <typename T>
CUDA_LIB_EXPORT cudaError_t Quantile(const T *input, const T *q, T *out, T *sort, const int dim, const int x,
                                     const int y, const int z, const int each_q_elements, const int output_elements,
                                     int *flag_in, int *ret_flag_device, int *nan_flags, const bool ignorenan_,
                                     const uint32_t &device_id, cudaStream_t cuda_stream) {
  (void)cudaMemset(ret_flag_device, 0, sizeof(int));
  QuantileKernelCheckQ<<<CUDA_BLOCKS(device_id, output_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    output_elements / each_q_elements, q, ret_flag_device);
  cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));
  (void)cudaMemcpy(flag_in, ret_flag_device, sizeof(int), cudaMemcpyDeviceToHost);
  (void)cudaMemset(nan_flags, 0, sizeof(int));
  CheckNanInit<<<CUDA_BLOCKS(device_id, x * z), CUDA_THREADS(device_id), 0, cuda_stream>>>(x, y, z, x * z, input,
                                                                                           nan_flags);
  if (ignorenan_) {
    CountNan<<<CUDA_BLOCKS(device_id, x * z), CUDA_THREADS(device_id), 0, cuda_stream>>>(x, y, z, x * z, input,
                                                                                         nan_flags);
  } else {
    FlagNan<<<CUDA_BLOCKS(device_id, x * y * z), CUDA_THREADS(device_id), 0, cuda_stream>>>(x, y, z, x * y * z, input,
                                                                                            nan_flags);
  }
  int ceil_p_2 = RoundUpPower2(y);
  int thread = std::min(ceil_p_2, CUDA_THREADS(device_id));
  Copy<<<CUDA_BLOCKS(device_id, x * ceil_p_2 * z), CUDA_THREADS(device_id), 0, cuda_stream>>>(input, sort, x, ceil_p_2,
                                                                                              y, z);
  BitonicSort<<<x * z, thread, 0, cuda_stream>>>(ceil_p_2, sort, x * z, z);
  DoQuantile<<<CUDA_BLOCKS(device_id, output_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, q, out, sort, dim, x, y, z, each_q_elements, output_elements, ceil_p_2, nan_flags, ignorenan_);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t Quantile<float>(const float *input, const float *q, float *out, float *sort,
                                                     const int dim, const int x, const int y, const int z,
                                                     const int each_q_elements, const int output_elements, int *flag_in,
                                                     int *ret_flag_device, int *nan_flags, const bool ignorenan_,
                                                     const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t Quantile<double>(const double *input, const double *q, double *out, double *sort,
                                                      const int dim, const int x, const int y, const int z,
                                                      const int each_q_elements, const int output_elements,
                                                      int *flag_in, int *ret_flag_device, int *nan_flags,
                                                      const bool ignorenan_, const uint32_t &device_id,
                                                      cudaStream_t cuda_stream);
