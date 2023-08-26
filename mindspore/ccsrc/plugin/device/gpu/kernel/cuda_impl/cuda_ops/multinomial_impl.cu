/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "multinomial_impl.cuh"
#include <algorithm>

template <typename T, typename S>
inline T Floor(const T &num, const S &unit) {
  return static_cast<T>(num / unit);
}

template <typename T, typename S>
inline T Ceil(const T &num, const S &unit) {
  return static_cast<T>((num + unit - 1) / unit);
}

__global__ void InitRandStateKernel(uint64_t seed, uint64_t seed_offset, int num, curandState *state) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x) {
    curand_init(seed, i, seed_offset, &state[i]);
  }
}

cudaError_t InitRandState(uint64_t seed, uint64_t seed_offset, int num, curandState *state, cudaStream_t stream) {
  InitRandStateKernel<<<(num + 127) / 128, 128, 0, stream>>>(seed, seed_offset, num, state);
  return GetCudaStatus();
}

template <typename T>
__global__ void CheckZeroKernel(const size_t distributions, const size_t categories, const T *input, T *out) {
  out[0] = 0;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (distributions); pos += blockDim.x * gridDim.x) {
    if (input[(1 + pos) * categories - 1] <= 0) {
      out[0] = 1;
    }
  }
  return;
}

template <typename T>
cudaError_t CheckZero(const size_t distributions, const size_t categories, const T *input, T *output,
                      cudaStream_t cuda_stream) {
  CheckZeroKernel<<<GET_BLOCKS(distributions), GET_THREADS, 0, cuda_stream>>>(distributions, categories, input, output);
  return GetCudaStatus();
}

template <typename T>
__global__ void CheckNonNegKernel(const size_t size, const T *input, T *out) {
  out[0] = 0;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    if (input[pos] < 0) {
      out[0] = 1;
    }
  }
  return;
}

template <typename T>
cudaError_t CheckNonNeg(const size_t size, const T *input, T *output, cudaStream_t cuda_stream) {
  CheckNonNegKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}

template <typename T>
__device__ int BinarySearchForMultinomial(T *start_addr, int size, T rand) {
  int start = 0;
  int end = size;
  while (end - start > 0) {
    int mid = start + (end - start) / 2;
    T mid_val = start_addr[mid];
    if (mid_val < rand) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  if (start == size) {
    start = size - 1;
  }
  return start;
}

template <typename T, typename S>
__global__ void MultinomialKernel(int row, int col, T *probs, curandState *state, int64_t *num_sample, S *output) {
  // Load the probs to shared memory.
  extern __shared__ float accum_probs[];
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int probs_base_index = gid * col;
  if (probs_base_index >= row * col) {
    return;
  }

  int shm_base_index = threadIdx.x * col;
  accum_probs[shm_base_index] = probs[probs_base_index];
  for (int i = 1; i < col; i++) {
    probs_base_index++;
    float prob = static_cast<float>(probs[probs_base_index]);
    CUDA_KERNEL_ASSERT(prob >= 0);
    CUDA_KERNEL_ASSERT(!isnan(prob));
    CUDA_KERNEL_ASSERT(!isinf(prob));
    accum_probs[shm_base_index + i] = accum_probs[shm_base_index + i - 1] + prob;
  }
  __syncthreads();

  // Probs normalization.
  float max_probs = accum_probs[shm_base_index + col - 1];
  for (int i = 0; i < col; i++) {
    accum_probs[shm_base_index + i] /= max_probs;
  }
  __syncthreads();

  // Sample.
  int output_base_index = gid * num_sample[0];
  auto local_state = state[gid];
  for (int i = 0; i < num_sample[0]; i++) {
    float rand = curand_uniform(&local_state);
    output[output_base_index + i] = static_cast<S>(BinarySearchForMultinomial(&accum_probs[shm_base_index], col, rand));
  }
  state[gid] = local_state;
}

template <typename T, typename S>
cudaError_t Multinomial(int row, int col, T *probs, curandState *state, int64_t *num_sample, S *output,
                        cudaStream_t stream) {
  // Every block process several rows. It depends on shared memory usage.
  constexpr int max_shm_used_per_block = 256;
  int block_dim = std::max(Floor(std::min(row, max_shm_used_per_block), col), 1);
  int grid_dim = Ceil(row, block_dim);
  int shm_size = block_dim * col * sizeof(float);

  MultinomialKernel<<<grid_dim, block_dim, shm_size, stream>>>(row, col, probs, state, num_sample, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t Multinomial<float, int64_t>(int row, int col, float *probs, curandState *state,
                                                                 int64_t *num_sample, int64_t *output,
                                                                 cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Multinomial<double, int64_t>(int row, int col, double *probs, curandState *state,
                                                                  int64_t *num_sample, int64_t *output,
                                                                  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Multinomial<half, int64_t>(int row, int col, half *probs, curandState *state,
                                                                int64_t *num_sample, int64_t *output,
                                                                cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Multinomial<int8_t, int64_t>(int row, int col, int8_t *probs, curandState *state,
                                                                  int64_t *num_sample, int64_t *output,
                                                                  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Multinomial<int16_t, int64_t>(int row, int col, int16_t *probs, curandState *state,
                                                                   int64_t *num_sample, int64_t *output,
                                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Multinomial<int32_t, int64_t>(int row, int col, int32_t *probs, curandState *state,
                                                                   int64_t *num_sample, int64_t *output,
                                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Multinomial<int64_t, int64_t>(int row, int col, int64_t *probs, curandState *state,
                                                                   int64_t *num_sample, int64_t *output,
                                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Multinomial<uint8_t, int64_t>(int row, int col, uint8_t *probs, curandState *state,
                                                                   int64_t *num_sample, int64_t *output,
                                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Multinomial<uint16_t, int64_t>(int row, int col, uint16_t *probs,
                                                                    curandState *state, int64_t *num_sample,
                                                                    int64_t *output, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Multinomial<uint32_t, int64_t>(int row, int col, uint32_t *probs,
                                                                    curandState *state, int64_t *num_sample,
                                                                    int64_t *output, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Multinomial<uint64_t, int64_t>(int row, int col, uint64_t *probs,
                                                                    curandState *state, int64_t *num_sample,
                                                                    int64_t *output, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Multinomial<float, int32_t>(int row, int col, float *probs, curandState *state,
                                                                 int64_t *num_sample, int32_t *output,
                                                                 cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Multinomial<double, int32_t>(int row, int col, double *probs, curandState *state,
                                                                  int64_t *num_sample, int32_t *output,
                                                                  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Multinomial<half, int32_t>(int row, int col, half *probs, curandState *state,
                                                                int64_t *num_sample, int32_t *output,
                                                                cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Multinomial<int8_t, int32_t>(int row, int col, int8_t *probs, curandState *state,
                                                                  int64_t *num_sample, int32_t *output,
                                                                  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Multinomial<int16_t, int32_t>(int row, int col, int16_t *probs, curandState *state,
                                                                   int64_t *num_sample, int32_t *output,
                                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Multinomial<int32_t, int32_t>(int row, int col, int32_t *probs, curandState *state,
                                                                   int64_t *num_sample, int32_t *output,
                                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Multinomial<int64_t, int32_t>(int row, int col, int64_t *probs, curandState *state,
                                                                   int64_t *num_sample, int32_t *output,
                                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Multinomial<uint8_t, int32_t>(int row, int col, uint8_t *probs, curandState *state,
                                                                   int64_t *num_sample, int32_t *output,
                                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Multinomial<uint16_t, int32_t>(int row, int col, uint16_t *probs,
                                                                    curandState *state, int64_t *num_sample,
                                                                    int32_t *output, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Multinomial<uint32_t, int32_t>(int row, int col, uint32_t *probs,
                                                                    curandState *state, int64_t *num_sample,
                                                                    int32_t *output, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Multinomial<uint64_t, int32_t>(int row, int col, uint64_t *probs,
                                                                    curandState *state, int64_t *num_sample,
                                                                    int32_t *output, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CheckNonNeg<float>(const size_t size, const float *input, float *output,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CheckZero<float>(const size_t distributions, const size_t categories,
                                                      const float *input, float *output, cudaStream_t cuda_stream);
