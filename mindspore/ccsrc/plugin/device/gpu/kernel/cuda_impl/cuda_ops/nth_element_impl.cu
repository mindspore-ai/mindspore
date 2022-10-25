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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/nth_element_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/nth_element_lib.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#define RADIX_SIZE 4
#define RADIX_BITS 2
#define RADIX_MASK 3

template <typename T, typename S, typename V>
__device__ T match(T *smem, T *data, V slice_size, V within_slice_stride, S desired, S desiredMask) {
  if (threadIdx.x < 2) {
    smem[threadIdx.x] = static_cast<T>(0);
  }
  __syncthreads();

  V numIterations = RoundUp(slice_size, static_cast<V>(blockDim.x));
  for (V i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < slice_size);
    T v = inRange ? data[i * within_slice_stride] : static_cast<T>(0);
    if (inRange && ((NthElementType<T>::change(v) & desiredMask) == desired)) {
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

template <typename T, typename S, typename V, typename CountType, int RadixSize, int RadixBits>
__device__ void countRadixUsingMask(CountType counts[RadixSize], CountType *smem, S desired, S desiredMask,
                                    int radixDigitPos, V slice_size, V within_slice_stride, T *data) {
#pragma unroll
  for (int i = 0; i < RadixSize; ++i) {
    counts[i] = 0;
  }
  if (threadIdx.x < RadixSize) {
    smem[threadIdx.x] = 0;
  }
  __syncthreads();
  for (V i = threadIdx.x; i < slice_size; i += blockDim.x) {
    S val = NthElementType<T>::change(data[i * within_slice_stride]);
    bool hasVal = ((val & desiredMask) == desired);
    S digitInRadix = Bitvalue<S>::getBitvalue(val, radixDigitPos, RadixBits);
#pragma unroll
    for (uint32_t j = 0; j < RadixSize; ++j) {
      bool vote = hasVal && (digitInRadix == j);
      counts[j] += __popc(__ballot_sync(__activemask(), vote));
    }
  }

  if (getLaneId() == 0) {
#pragma unroll
    for (uint32_t i = 0; i < RadixSize; ++i) {
      MsAtomicAdd(&smem[i], counts[i]);
    }
  }
  __syncthreads();

#pragma unroll
  for (uint32_t i = 0; i < RadixSize; ++i) {
    counts[i] = smem[i];
  }
  __syncthreads();
}

template <typename T, typename S, typename V>
__device__ void radixSelect(T *data, int32_t k, bool reverse, V slice_size, V within_slice_stride, int *smem, T *topK) {
  // Per-thread buckets into which we accumulate digit counts in our radix
  int counts[RADIX_SIZE];

  S desired = 0;
  S desiredMask = 0;

  int kToFind = k;

#pragma unroll
  for (int digitPos = sizeof(T) * 8 - RADIX_BITS; digitPos >= 0; digitPos -= RADIX_BITS) {
    countRadixUsingMask<T, S, V, int, RADIX_SIZE, RADIX_BITS>(counts, smem, desired, desiredMask, digitPos, slice_size,
                                                              within_slice_stride, data);
    auto found_unique = [&](int i, int count) -> bool {
      if (count == 1 && kToFind == 1) {
        desired = Bitvalue<S>::setBitvalue(desired, i, digitPos, RADIX_BITS);
        desiredMask = Bitvalue<S>::setBitvalue(desiredMask, RADIX_MASK, digitPos, RADIX_BITS);
        *topK =
          match<T, S, V>(reinterpret_cast<T *>(smem), data, slice_size, within_slice_stride, desired, desiredMask);
        return true;
      }
      return false;
    };
    auto found_non_unique = [&](int i, int count) -> bool {
      if (count >= kToFind) {
        desired = Bitvalue<S>::setBitvalue(desired, i, digitPos, RADIX_BITS);
        desiredMask = Bitvalue<S>::setBitvalue(desiredMask, RADIX_MASK, digitPos, RADIX_BITS);
        return true;
      }
      kToFind -= count;
      return false;
    };

    if (reverse) {
      // Process in descending order
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
      // Process in ascending order
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

  *topK = NthElementType<T>::recover(desired);
}

template <typename T, typename S>
__global__ void NthElement(const size_t slices_number, const size_t slice_size, T *input, T *output, const int32_t n,
                           bool reverse) {
  unsigned int within_slice_stride = 1;
  __shared__ int smem[kWarpSize];
  int64_t blockId = getLinearBlockId();
  T *start_input = input + blockId * slice_size;
  int32_t k = n + 1;
  radixSelect<T, S, size_t>(start_input, k, reverse, slice_size, within_slice_stride, smem, &output[blockId]);
  return;
}

template <typename T>
void CalNthElement(const size_t slices_number, const size_t slice_size, T *input, int32_t input_n, T *output,
                   bool reverse, const uint32_t &device_id, cudaStream_t stream) {
  dim3 grid = getGrid(slices_number);
  dim3 block(std::min(RoundUp((int64_t)slice_size, (int64_t)kWarpSize), (int64_t)1024));
  if (std::is_same<T, double>::value || std::is_same<T, int64_t>::value || std::is_same<T, uint64_t>::value) {
    NthElement<T, uint64_t><<<grid, block, 0, stream>>>(slices_number, slice_size, input, output, input_n, reverse);
  } else {
    NthElement<T, uint32_t><<<grid, block, 0, stream>>>(slices_number, slice_size, input, output, input_n, reverse);
  }
  return;
}

template CUDA_LIB_EXPORT void CalNthElement(const size_t slices_number, const size_t slice_size, half *input,
                                            int32_t input_n, half *output, bool reverse, const uint32_t &device_id,
                                            cudaStream_t stream);
template CUDA_LIB_EXPORT void CalNthElement(const size_t slices_number, const size_t slice_size, float *input,
                                            int32_t input_n, float *output, bool reverse, const uint32_t &device_id,
                                            cudaStream_t stream);
template CUDA_LIB_EXPORT void CalNthElement(const size_t slices_number, const size_t slice_size, int8_t *input,
                                            int32_t input_n, int8_t *output, bool reverse, const uint32_t &device_id,
                                            cudaStream_t stream);
template CUDA_LIB_EXPORT void CalNthElement(const size_t slices_number, const size_t slice_size, uint16_t *input,
                                            int32_t input_n, uint16_t *output, bool reverse, const uint32_t &device_id,
                                            cudaStream_t stream);

template CUDA_LIB_EXPORT void CalNthElement(const size_t slices_number, const size_t slice_size, int16_t *input,
                                            int32_t input_n, int16_t *output, bool reverse, const uint32_t &device_id,
                                            cudaStream_t stream);
template CUDA_LIB_EXPORT void CalNthElement(const size_t slices_number, const size_t slice_size, uint8_t *input,
                                            int32_t input_n, uint8_t *output, bool reverse, const uint32_t &device_id,
                                            cudaStream_t stream);

template CUDA_LIB_EXPORT void CalNthElement(const size_t slices_number, const size_t slice_size, int32_t *input,
                                            int32_t input_n, int32_t *output, bool reverse, const uint32_t &device_id,
                                            cudaStream_t stream);

template CUDA_LIB_EXPORT void CalNthElement(const size_t slices_number, const size_t slice_size, int64_t *input,
                                            int32_t input_n, int64_t *output, bool reverse, const uint32_t &device_id,
                                            cudaStream_t stream);
template CUDA_LIB_EXPORT void CalNthElement(const size_t slices_number, const size_t slice_size, double *input,
                                            int32_t input_n, double *output, bool reverse, const uint32_t &device_id,
                                            cudaStream_t stream);
