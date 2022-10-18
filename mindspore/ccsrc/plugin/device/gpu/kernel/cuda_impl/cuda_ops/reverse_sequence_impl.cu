/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/reverse_sequence_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"


template <typename T>
using Complex = mindspore::utils::Complex<T>;

// Util function to convert a 1D input array index to an N-D positional index
// Required since GPU iterates over all values in an ND array as a 1D array
__inline__ __device__ void IdxToPos(size_t idx, size_t *pos, size_t cur_thread_idx, size_t *cum_shape,
                                    size_t shape_size) {
  size_t rem_val = idx;
  for (int i = 0; i < shape_size; i++) {
    pos[cur_thread_idx + i] = rem_val / cum_shape[i];
    rem_val = rem_val % cum_shape[i];
  }
  return;
}

// Util function to convert a N-D positonal index to a 1D index
__inline__ __device__ size_t PosToIdx(size_t *pos, size_t cur_thread_idx, size_t *cum_shape, size_t shape_size) {
  size_t idx = 0;
  for (int i = 0; i < shape_size; i++) {
    idx = idx + (pos[cur_thread_idx + i] * cum_shape[i]);
  }
  return idx;
}

// CumShape takes Shape: (2,2,5) => cumShape (10,5,1) which informs how many values
// each dimension will represent. Required for converting 1d index to positional vector.
// In this example 10 in dim 0 means, an increase of 1 in this dim leads to another 10 values
// in the overall array
__global__ void ComputeCumShape(const size_t *input_shape_ptr, size_t *input_shape_cum_ptr, size_t shape_size) {
  int cur_val = 1;
  for (int i = shape_size - 1; i >= 0; i--) {
    // iterate list in reverse and cummulatively build shape
    input_shape_cum_ptr[i] = cur_val;
    cur_val = cur_val * input_shape_ptr[i];
  }
  return;
}
template <typename T, typename S>
__global__ void ReverseSequence(const size_t size, const T *input, const S *seq_len, const int64_t batch_dim,
                                const int64_t seq_dim, size_t *cur_pos_arr, const size_t *input_shape_ptr,
                                size_t *input_shape_cum_ptr, size_t shape_size, T *output) {
  // calculate which thread this is out of total across all blocks for accessing respective cur_pos_arr memory
  size_t cur_thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  cur_thread_idx = cur_thread_idx * shape_size;
  size_t cur_slice = 0;          // current slice as split by the batch_dim
  size_t cur_slice_seq_len = 0;  // reverse seq length for this slice as provided by user
  size_t new_idx = 0;            // calculate corresponding reverse element from input
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
    IdxToPos(idx, cur_pos_arr, cur_thread_idx, input_shape_cum_ptr, shape_size);
    cur_slice = cur_pos_arr[cur_thread_idx + batch_dim];  // all accesses to cur_pos_arr have to be adjusted per thread
    cur_slice_seq_len = seq_len[cur_slice];
    if (cur_slice_seq_len == 0) {  // adjust length to 1 if 0 provided, same result in both cases
      cur_slice_seq_len = 1;
    }
    if (cur_pos_arr[cur_thread_idx + seq_dim] > (cur_slice_seq_len - 1)) {  // check if within range
      // copy value directly and continue - outside of reversal range
      output[idx] = input[idx];
      continue;
    }
    // find corresponding reverse element in input
    cur_pos_arr[cur_thread_idx + seq_dim] =
      (cur_slice_seq_len - 1) - cur_pos_arr[cur_thread_idx + seq_dim];                 // adjust position to target
    new_idx = PosToIdx(cur_pos_arr, cur_thread_idx, input_shape_cum_ptr, shape_size);  // get the updated index
    output[idx] = input[new_idx];
  }
  return;
}

template <typename T, typename S>
void CalReverseSequence(const size_t size, const T *input, const S *seq_len, const int64_t batch_dim,
                        const int64_t seq_dim, size_t *cur_pos_arr, const size_t *input_shape_ptr,
                        size_t *input_shape_cum_ptr, size_t shape_size, T *output,
                        cudaStream_t cuda_stream) {
  ComputeCumShape<<<1, 1, 0, cuda_stream>>>(input_shape_ptr, input_shape_cum_ptr, shape_size);
  ReverseSequence<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(
    size, input, seq_len, batch_dim, seq_dim, cur_pos_arr, input_shape_ptr, input_shape_cum_ptr, shape_size, output);
  return;
}

template CUDA_LIB_EXPORT void CalReverseSequence<int8_t, int>(const size_t size, const int8_t *input,
                                                              const int *seq_len, const int64_t batch_dim,
                                                              const int64_t seq_dim, size_t *cur_pos_arr,
                                                              const size_t *input_shape_ptr,
                                                              size_t *intput_shape_cum_ptr, size_t shape_size,
                                                              int8_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<int8_t, int64_t>(const size_t size, const int8_t *input,
                                                                  const int64_t *seq_len, const int64_t batch_dim,
                                                                  const int64_t seq_dim, size_t *cur_pos_arr,
                                                                  const size_t *input_shape_ptr,
                                                                  size_t *intput_shape_cum_ptr, size_t shape_size,
                                                                  int8_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<int16_t, int>(const size_t size, const int16_t *input,
                                                               const int *seq_len, const int64_t batch_dim,
                                                               const int64_t seq_dim, size_t *cur_pos_arr,
                                                               const size_t *input_shape_ptr,
                                                               size_t *intput_shape_cum_ptr, size_t shape_size,
                                                               int16_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<int16_t, int64_t>(const size_t size, const int16_t *input,
                                                                   const int64_t *seq_len, const int64_t batch_dim,
                                                                   const int64_t seq_dim, size_t *cur_pos_arr,
                                                                   const size_t *input_shape_ptr,
                                                                   size_t *intput_shape_cum_ptr, size_t shape_size,
                                                                   int16_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<int, int>(const size_t size, const int *input, const int *seq_len,
                                                           const int64_t batch_dim, const int64_t seq_dim,
                                                           size_t *cur_pos_arr, const size_t *input_shape_ptr,
                                                           size_t *intput_shape_cum_ptr, size_t shape_size, int *output,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<int, int64_t>(const size_t size, const int *input,
                                                               const int64_t *seq_len, const int64_t batch_dim,
                                                               const int64_t seq_dim, size_t *cur_pos_arr,
                                                               const size_t *input_shape_ptr,
                                                               size_t *intput_shape_cum_ptr, size_t shape_size,
                                                               int *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<int64_t, int>(const size_t size, const int64_t *input,
                                                               const int *seq_len, const int64_t batch_dim,
                                                               const int64_t seq_dim, size_t *cur_pos_arr,
                                                               const size_t *input_shape_ptr,
                                                               size_t *intput_shape_cum_ptr, size_t shape_size,
                                                               int64_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<int64_t, int64_t>(const size_t size, const int64_t *input,
                                                                   const int64_t *seq_len, const int64_t batch_dim,
                                                                   const int64_t seq_dim, size_t *cur_pos_arr,
                                                                   const size_t *input_shape_ptr,
                                                                   size_t *intput_shape_cum_ptr, size_t shape_size,
                                                                   int64_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<half, int>(const size_t size, const half *input, const int *seq_len,
                                                            const int64_t batch_dim, const int64_t seq_dim,
                                                            size_t *cur_pos_arr, const size_t *input_shape_ptr,
                                                            size_t *intput_shape_cum_ptr, size_t shape_size,
                                                            half *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<half, int64_t>(const size_t size, const half *input,
                                                                const int64_t *seq_len, const int64_t batch_dim,
                                                                const int64_t seq_dim, size_t *cur_pos_arr,
                                                                const size_t *input_shape_ptr,
                                                                size_t *intput_shape_cum_ptr, size_t shape_size,
                                                                half *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<float, int>(const size_t size, const float *input, const int *seq_len,
                                                             const int64_t batch_dim, const int64_t seq_dim,
                                                             size_t *cur_pos_arr, const size_t *input_shape_ptr,
                                                             size_t *intput_shape_cum_ptr, size_t shape_size,
                                                             float *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<float, int64_t>(const size_t size, const float *input,
                                                                 const int64_t *seq_len, const int64_t batch_dim,
                                                                 const int64_t seq_dim, size_t *cur_pos_arr,
                                                                 const size_t *input_shape_ptr,
                                                                 size_t *intput_shape_cum_ptr, size_t shape_size,
                                                                 float *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<double, int>(const size_t size, const double *input,
                                                              const int *seq_len, const int64_t batch_dim,
                                                              const int64_t seq_dim, size_t *cur_pos_arr,
                                                              const size_t *input_shape_ptr,
                                                              size_t *intput_shape_cum_ptr, size_t shape_size,
                                                              double *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<double, int64_t>(const size_t size, const double *input,
                                                                  const int64_t *seq_len, const int64_t batch_dim,
                                                                  const int64_t seq_dim, size_t *cur_pos_arr,
                                                                  const size_t *input_shape_ptr,
                                                                  size_t *intput_shape_cum_ptr, size_t shape_size,
                                                                  double *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<bool, int>(const size_t size, const bool *input, const int *seq_len,
                                                            const int64_t batch_dim, const int64_t seq_dim,
                                                            size_t *cur_pos_arr, const size_t *input_shape_ptr,
                                                            size_t *intput_shape_cum_ptr, size_t shape_size,
                                                            bool *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<bool, int64_t>(const size_t size, const bool *input,
                                                                const int64_t *seq_len, const int64_t batch_dim,
                                                                const int64_t seq_dim, size_t *cur_pos_arr,
                                                                const size_t *input_shape_ptr,
                                                                size_t *intput_shape_cum_ptr, size_t shape_size,
                                                                bool *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<uint8_t, int>(const size_t size, const uint8_t *input,
                                                               const int *seq_len, const int64_t batch_dim,
                                                               const int64_t seq_dim, size_t *cur_pos_arr,
                                                               const size_t *input_shape_ptr,
                                                               size_t *intput_shape_cum_ptr, size_t shape_size,
                                                               uint8_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<uint8_t, int64_t>(const size_t size, const uint8_t *input,
                                                                   const int64_t *seq_len, const int64_t batch_dim,
                                                                   const int64_t seq_dim, size_t *cur_pos_arr,
                                                                   const size_t *input_shape_ptr,
                                                                   size_t *intput_shape_cum_ptr, size_t shape_size,
                                                                   uint8_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<uint16_t, int>(const size_t size, const uint16_t *input,
                                                                const int *seq_len, const int64_t batch_dim,
                                                                const int64_t seq_dim, size_t *cur_pos_arr,
                                                                const size_t *input_shape_ptr,
                                                                size_t *intput_shape_cum_ptr, size_t shape_size,
                                                                uint16_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<uint16_t, int64_t>(const size_t size, const uint16_t *input,
                                                                    const int64_t *seq_len, const int64_t batch_dim,
                                                                    const int64_t seq_dim, size_t *cur_pos_arr,
                                                                    const size_t *input_shape_ptr,
                                                                    size_t *intput_shape_cum_ptr, size_t shape_size,
                                                                    uint16_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<uint32_t, int>(const size_t size, const uint32_t *input,
                                                                const int *seq_len, const int64_t batch_dim,
                                                                const int64_t seq_dim, size_t *cur_pos_arr,
                                                                const size_t *input_shape_ptr,
                                                                size_t *intput_shape_cum_ptr, size_t shape_size,
                                                                uint32_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<uint32_t, int64_t>(const size_t size, const uint32_t *input,
                                                                    const int64_t *seq_len, const int64_t batch_dim,
                                                                    const int64_t seq_dim, size_t *cur_pos_arr,
                                                                    const size_t *input_shape_ptr,
                                                                    size_t *intput_shape_cum_ptr, size_t shape_size,
                                                                    uint32_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<uint64_t, int>(const size_t size, const uint64_t *input,
                                                                const int *seq_len, const int64_t batch_dim,
                                                                const int64_t seq_dim, size_t *cur_pos_arr,
                                                                const size_t *input_shape_ptr,
                                                                size_t *intput_shape_cum_ptr, size_t shape_size,
                                                                uint64_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<uint64_t, int64_t>(const size_t size, const uint64_t *input,
                                                                    const int64_t *seq_len, const int64_t batch_dim,
                                                                    const int64_t seq_dim, size_t *cur_pos_arr,
                                                                    const size_t *input_shape_ptr,
                                                                    size_t *intput_shape_cum_ptr, size_t shape_size,
                                                                    uint64_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<Complex<float>, int>(const size_t size, const Complex<float> *input,
                                                                      const int *seq_len, const int64_t batch_dim,
                                                                      const int64_t seq_dim, size_t *cur_pos_arr,
                                                                      const size_t *input_shape_ptr,
                                                                      size_t *intput_shape_cum_ptr, size_t shape_size,
                                                                      Complex<float> *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<Complex<float>, int64_t>(const size_t size,
                                                                          const Complex<float> *input,
                                                                          const int64_t *seq_len,
                                                                          const int64_t batch_dim,
                                                                          const int64_t seq_dim,
                                                                          size_t *cur_pos_arr,
                                                                          const size_t *input_shape_ptr,
                                                                          size_t *intput_shape_cum_ptr,
                                                                          size_t shape_size,
                                                                          Complex<float> *output,
                                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<Complex<double>, int>(const size_t size, const Complex<double> *input,
                                                                       const int *seq_len, const int64_t batch_dim,
                                                                       const int64_t seq_dim, size_t *cur_pos_arr,
                                                                       const size_t *input_shape_ptr,
                                                                       size_t *intput_shape_cum_ptr, size_t shape_size,
                                                                       Complex<double> *output,
                                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalReverseSequence<Complex<double>, int64_t>(const size_t size,
                                                                           const Complex<double> *input,
                                                                           const int64_t *seq_len,
                                                                           const int64_t batch_dim,
                                                                           const int64_t seq_dim, size_t *cur_pos_arr,
                                                                           const size_t *input_shape_ptr,
                                                                           size_t *intput_shape_cum_ptr,
                                                                           size_t shape_size,
                                                                           Complex<double> *output,
                                                                           cudaStream_t cuda_stream);
