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

#include "median_impl.cuh"
#include <iostream>
#include <vector>
#include <algorithm>

constexpr int WARP_SIZE = 32;
constexpr int MAX_THREAD = 1024;
constexpr int RADIX_BITS = 2;
constexpr int RADIX_SIZE = 4;
constexpr int RADIX_MASK = (RADIX_SIZE - 1);

__device__ __forceinline__ unsigned int warp_ballot(int predicate) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 700
  return __ballot(predicate);
#else
  return __ballot_sync(__activemask(), predicate);
#endif
}

template <typename T>
static __device__ __host__ T round_up(T a, T b) {
  return ((a + b - 1) / b) * b;
}

template <typename T>
struct Bitfield {};

template <>
struct Bitfield<unsigned int> {
  static __device__ unsigned int GetBitfield(unsigned int val, int pos, int len) {
#if !defined(__CUDA_ARCH__)
    pos &= 0xff;
    len &= 0xff;

    unsigned int m = (1u << len) - 1u;
    return (val >> pos) & m;
#else
    unsigned int ret;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
    return ret;
#endif
  }

  static __device__ unsigned int SetBitfield(unsigned int val, unsigned int to_insert, int pos, int len) {
#if !defined(__CUDA_ARCH__)
    pos &= 0xff;
    len &= 0xff;

    unsigned int m = (1u << len) - 1u;
    to_insert &= m;
    to_insert <<= pos;
    m <<= pos;

    return (val & ~m) | to_insert;
#else
    unsigned int ret;
    asm("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(ret) : "r"(to_insert), "r"(val), "r"(pos), "r"(len));
    return ret;
#endif
  }
};

template <>
struct Bitfield<uint64_t> {
  static __device__ uint64_t GetBitfield(uint64_t val, int pos, int len) {
#if !defined(__CUDA_ARCH__)
    pos &= 0xff;
    len &= 0xff;

    uint64_t m = (1u << len) - 1u;
    return (val >> pos) & m;
#else
    uint64_t ret;
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(val), "r"(pos), "r"(len));
    return ret;
#endif
  }

  static __device__ uint64_t SetBitfield(uint64_t val, uint64_t to_insert, int pos, int len) {
#if !defined(__CUDA_ARCH__)
    pos &= 0xff;
    len &= 0xff;

    uint64_t m = (1u << len) - 1u;
    to_insert &= m;
    to_insert <<= pos;
    m <<= pos;

    return (val & ~m) | to_insert;
#else
    uint64_t ret;
    asm("bfi.b64 %0, %1, %2, %3, %4;" : "=l"(ret) : "l"(to_insert), "l"(val), "r"(pos), "r"(len));
    return ret;
#endif
  }
};

template <typename T>
struct MedianTypeConfig {
  typedef T RadixType;

  static inline __device__ RadixType Convert(T v) { return v; }

  static inline __device__ T Deconvert(RadixType v) { return v; }
};

template <>
struct MedianTypeConfig<uint8_t> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType Convert(uint8_t v) { return v; }

  static inline __device__ uint8_t Deconvert(RadixType v) { return v; }
};

template <>
struct MedianTypeConfig<int8_t> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType Convert(int8_t v) { return 128u + v; }

  static inline __device__ int8_t Deconvert(RadixType v) { return v - 128; }
};

template <>
struct MedianTypeConfig<uint16_t> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType Convert(uint16_t v) { return v; }

  static inline __device__ uint16_t Deconvert(RadixType v) { return v; }
};

template <>
struct MedianTypeConfig<int16_t> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType Convert(int16_t v) { return 32768u + v; }

  static inline __device__ int16_t Deconvert(RadixType v) { return v - 32768; }
};

template <>
struct MedianTypeConfig<int32_t> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType Convert(int32_t v) { return 2147483648u + v; }

  static inline __device__ int32_t Deconvert(RadixType v) { return v - 2147483648u; }
};

template <>
struct MedianTypeConfig<int64_t> {
  typedef uint64_t RadixType;

  static inline __device__ RadixType Convert(int64_t v) { return 9223372036854775808ull + v; }

  static inline __device__ int64_t Deconvert(RadixType v) { return v - 9223372036854775808ull; }
};

template <>
struct MedianTypeConfig<half> {
  typedef uint32_t RadixType;

  // Converts a half to an unsigned int with the same sorting
  static inline __device__ RadixType Convert(half v) {
    RadixType x = __half_as_ushort(v);
    RadixType mask = (x & 0x00008000) ? 0x0000ffff : 0x00008000;
    return (v == v) ? (x ^ mask) : 0xffff;
  }

  static inline __device__ half Deconvert(RadixType v) {
    RadixType mask = (v & 0x00008000) ? 0x00008000 : 0x0000ffff;
    return __ushort_as_half(v ^ mask);
  }
};

template <>
struct MedianTypeConfig<float> {
  typedef uint32_t RadixType;

  // Converts a float to an unsigned int with the same sorting
  static inline __device__ RadixType Convert(float v) {
    RadixType x = __float_as_int(v);
    RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
    return (v == v) ? (x ^ mask) : 0xffffffff;
  }

  static inline __device__ float Deconvert(RadixType v) {
    RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;
    return __int_as_float(v ^ mask);
  }
};

template <>
struct MedianTypeConfig<double> {
  typedef uint64_t RadixType;

  static inline __device__ RadixType Convert(double v) {
    RadixType x = __double_as_longlong(v);
    RadixType mask = -((x >> 63)) | 0x8000000000000000;
    return (v == v) ? (x ^ mask) : 0xffffffffffffffff;
  }

  static inline __device__ double Deconvert(RadixType v) {
    RadixType mask = ((v >> 63) - 1) | 0x8000000000000000;
    return __longlong_as_double(v ^ mask);
  }
};

// This function counts the distribution of all input values in a slice
template <typename T, typename R_T, typename S, int RadixSize, int RadixBits>
__device__ void CountRadixUsingMask(int counts[RadixSize], int *smem, R_T desired, R_T desired_mask,
                                    int radix_digit_pos, S size, S stride, const T *data) {
#pragma unroll
  for (int i = 0; i < RadixSize; ++i) {
    counts[i] = 0;
  }

  if (threadIdx.x < RadixSize) {
    smem[threadIdx.x] = 0;
  }
  __syncthreads();

  for (S i = threadIdx.x; i < size; i += blockDim.x) {
    R_T val = MedianTypeConfig<T>::Convert(data[i * stride]);

    bool hasVal = ((val & desired_mask) == desired);
    R_T digit_in_radix = Bitfield<R_T>::GetBitfield(val, radix_digit_pos, RadixBits);

#pragma unroll
    for (uint32_t j = 0; j < RadixSize; ++j) {
      bool vote = hasVal && (digit_in_radix == j);
      counts[j] += __popc(warp_ballot(vote));
    }
  }

  if (threadIdx.x % 32 == 0) {
#pragma unroll
    for (uint32_t i = 0; i < RadixSize; ++i) {
      atomicAdd(&smem[i], counts[i]);
    }
  }

  __syncthreads();

#pragma unroll
  for (uint32_t i = 0; i < RadixSize; ++i) {
    counts[i] = smem[i];
  }

  __syncthreads();
}

// This finds the unique value that matches the pattern
template <typename T, typename R_T, typename S>
__device__ T FindPattern(T *smem, const T *data, S size, S stride, R_T desired, R_T desired_mask) {
  if (threadIdx.x < 2) {
    smem[threadIdx.x] = static_cast<T>(0);
  }
  __syncthreads();

  S size_round = round_up(size, static_cast<S>(blockDim.x));
  for (S i = threadIdx.x; i < size_round; i += blockDim.x) {
    bool in_range = (i < size);
    T v = in_range ? data[i * stride] : static_cast<T>(0);

    if (in_range && ((MedianTypeConfig<T>::Convert(v) & desired_mask) == desired)) {
      smem[0] = static_cast<T>(1);
      smem[1] = v;
    }

    __syncthreads();

    T found = smem[0];
    T val = smem[1];

    __syncthreads();

    if (found != static_cast<T>(0)) {
      return val;
    }
  }

  return static_cast<T>(0);
}

// Returns the top-Kth element found in the data using radix selection
template <typename T, typename R_T, typename S>
__device__ void RadixSelect(const T *data, S kth, bool largest, S size, S stride, int *smem, T *top_k) {
  int counts[RADIX_SIZE];
  R_T desired = 0;
  R_T desired_mask = 0;
  int k = kth;

  for (int digit_pos = sizeof(T) * 8 - RADIX_BITS; digit_pos >= 0; digit_pos -= RADIX_BITS) {
    CountRadixUsingMask<T, R_T, S, RADIX_SIZE, RADIX_BITS>(counts, smem, desired, desired_mask, digit_pos, size, stride,
                                                           data);

    auto found_unique = [&](int i, int count) -> bool {
      if (count == 1 && k == 1) {
        desired = Bitfield<R_T>::SetBitfield(desired, i, digit_pos, RADIX_BITS);
        desired_mask = Bitfield<R_T>::SetBitfield(desired_mask, RADIX_MASK, digit_pos, RADIX_BITS);
        *top_k = FindPattern<T, R_T, S>(reinterpret_cast<T *>(smem), data, size, stride, desired, desired_mask);
        return true;
      }
      return false;
    };
    auto found_non_unique = [&](int i, int count) -> bool {
      if (count >= k) {
        desired = Bitfield<R_T>::SetBitfield(desired, i, digit_pos, RADIX_BITS);
        desired_mask = Bitfield<R_T>::SetBitfield(desired_mask, RADIX_MASK, digit_pos, RADIX_BITS);
        return true;
      }
      k -= count;
      return false;
    };

    if (largest) {
#pragma unroll
      for (int i = RADIX_SIZE - 1; i >= 0; --i) {
        int count = counts[i];
        if (found_unique(i, count)) {
          return;
        }
        if (found_non_unique(i, count)) {
          break;
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < RADIX_SIZE; ++i) {
        int count = counts[i];
        if (found_unique(i, count)) {
          return;
        }
        if (found_non_unique(i, count)) {
          break;
        }
      }
    }
  }

  *top_k = MedianTypeConfig<T>::Deconvert(desired);
}

template <typename T, typename S>
__global__ void MedianKernel(const T *input, T *output, S *indices, S size, S num, S stride, bool global_median) {
  __shared__ int smem[WARP_SIZE];

  S slice = blockIdx.y * gridDim.x + blockIdx.x;
  if (slice >= num) {
    return;
  }

  S offset_y = size * gridDim.x;

  S k = (size - 1) / 2;

  // Find the median value
  T median = static_cast<T>(0);
  RadixSelect<T, typename MedianTypeConfig<T>::RadixType, S>(input + blockIdx.y * offset_y + blockIdx.x, k + 1, false,
                                                             size, stride, smem, &median);
  output[slice] = median;

  // Find the index of the median value in the slice
  if (!global_median) {
    for (S i = threadIdx.x; i < size; i += blockDim.x) {
      T val = input[blockIdx.y * offset_y + blockIdx.x + i * stride];
      if (val == median) {
        indices[slice] = i;
        break;
      }
    }
  }
}

template <typename T, typename S>
cudaError_t Median(const T *input_value, T *output, S *indices, const std::vector<int64_t> input_shape,
                   const int64_t axis, bool global_median, cudaStream_t cuda_stream) {
  dim3 threads, grids;
  size_t i = 0;
  for (; i < static_cast<size_t>(axis); i++) {
    grids.y *= input_shape[i];
  }
  size_t size = input_shape.size() == 0 ? 1 : input_shape[axis];
  for (i = axis + 1; i < input_shape.size(); i++) {
    grids.x *= input_shape[i];
  }
  threads.x = std::min(round_up(static_cast<int>(size), WARP_SIZE), MAX_THREAD);

  S num = grids.y * grids.x;
  S stride = grids.x;

  MedianKernel<T, S>
    <<<grids, threads, 0, cuda_stream>>>(input_value, output, indices, size, num, stride, global_median);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t Median<uint8_t, int64_t>(const uint8_t *input_value, uint8_t *output,
                                                              int64_t *indices, const std::vector<int64_t> input_shape,
                                                              const int64_t axis, bool global_median,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t Median<int8_t, int64_t>(const int8_t *input_value, int8_t *output,
                                                             int64_t *indices, const std::vector<int64_t> input_shape,
                                                             const int64_t axis, bool global_median,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t Median<uint16_t, int64_t>(const uint16_t *input_value, uint16_t *output,
                                                               int64_t *indices, const std::vector<int64_t> input_shape,
                                                               const int64_t axis, bool global_median,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t Median<int16_t, int64_t>(const int16_t *input_value, int16_t *output,
                                                              int64_t *indices, const std::vector<int64_t> input_shape,
                                                              const int64_t axis, bool global_median,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t Median<uint32_t, int64_t>(const uint32_t *input_value, uint32_t *output,
                                                               int64_t *indices, const std::vector<int64_t> input_shape,
                                                               const int64_t axis, bool global_median,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t Median<int32_t, int64_t>(const int32_t *input_value, int32_t *output,
                                                              int64_t *indices, const std::vector<int64_t> input_shape,
                                                              const int64_t axis, bool global_median,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t Median<uint64_t, int64_t>(const uint64_t *input_value, uint64_t *output,
                                                               int64_t *indices, const std::vector<int64_t> input_shape,
                                                               const int64_t axis, bool global_median,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t Median<int64_t, int64_t>(const int64_t *input_value, int64_t *output,
                                                              int64_t *indices, const std::vector<int64_t> input_shape,
                                                              const int64_t axis, bool global_median,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t Median<half, int64_t>(const half *input_value, half *output, int64_t *indices,
                                                           const std::vector<int64_t> input_shape, const int64_t axis,
                                                           bool global_median, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t Median<float, int64_t>(const float *input_value, float *output, int64_t *indices,
                                                            const std::vector<int64_t> input_shape, const int64_t axis,
                                                            bool global_median, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t Median<double, int64_t>(const double *input_value, double *output,
                                                             int64_t *indices, const std::vector<int64_t> input_shape,
                                                             const int64_t axis, bool global_median,
                                                             cudaStream_t cuda_stream);
