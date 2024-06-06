/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/tracev2_grad_impl.cuh"

template <typename T>
__global__ void Tracev2GradKernel(T *din_addr, const T *dout_addr, const size_t row_st, const size_t col_st,
                                  const size_t diag_count, const size_t row_size, const size_t mat_size,
                                  const size_t batch_size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < batch_size; pos += blockDim.x * gridDim.x) {
    T temp = dout_addr[pos];
    T *din_mat_addr = din_addr + pos * mat_size;
    for (size_t i = 0; i < diag_count; i++) {
      size_t offset = (row_st + i) * row_size + (col_st + i);
      din_mat_addr[offset] = temp;
    }
  }
}

template <typename T>
cudaError_t Tracev2GradCalc(T *din_addr, const T *dout_addr, const size_t row_st, const size_t col_st,
                            const size_t diag_count, const size_t row_size, const size_t mat_size,
                            const size_t batch_size, const uint32_t &device_id, cudaStream_t stream) {
  Tracev2GradKernel<<<CUDA_BLOCKS(device_id, batch_size), CUDA_THREADS(device_id), 0, stream>>>(
    din_addr, dout_addr, row_st, col_st, diag_count, row_size, mat_size, batch_size);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t Tracev2GradCalc(half *din_addr, const half *dout_addr, const size_t row_st,
                                                     const size_t col_st, const size_t diag_count,
                                                     const size_t row_size, const size_t mat_size,
                                                     const size_t batch_size, const uint32_t &device_id,
                                                     cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Tracev2GradCalc(float *din_addr, const float *dout_addr, const size_t row_st,
                                                     const size_t col_st, const size_t diag_count,
                                                     const size_t row_size, const size_t mat_size,
                                                     const size_t batch_size, const uint32_t &device_id,
                                                     cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Tracev2GradCalc(double *din_addr, const double *dout_addr, const size_t row_st,
                                                     const size_t col_st, const size_t diag_count,
                                                     const size_t row_size, const size_t mat_size,
                                                     const size_t batch_size, const uint32_t &device_id,
                                                     cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Tracev2GradCalc(Complex<float> *din_addr, const Complex<float> *dout_addr,
                                                     const size_t row_st, const size_t col_st, const size_t diag_count,
                                                     const size_t row_size, const size_t mat_size,
                                                     const size_t batch_size, const uint32_t &device_id,
                                                     cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Tracev2GradCalc(Complex<double> *din_addr, const Complex<double> *dout_addr,
                                                     const size_t row_st, const size_t col_st, const size_t diag_count,
                                                     const size_t row_size, const size_t mat_size,
                                                     const size_t batch_size, const uint32_t &device_id,
                                                     cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Tracev2GradCalc(int8_t *din_addr, const int8_t *dout_addr, const size_t row_st,
                                                     const size_t col_st, const size_t diag_count,
                                                     const size_t row_size, const size_t mat_size,
                                                     const size_t batch_size, const uint32_t &device_id,
                                                     cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Tracev2GradCalc(int16_t *din_addr, const int16_t *dout_addr, const size_t row_st,
                                                     const size_t col_st, const size_t diag_count,
                                                     const size_t row_size, const size_t mat_size,
                                                     const size_t batch_size, const uint32_t &device_id,
                                                     cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Tracev2GradCalc(int32_t *din_addr, const int32_t *dout_addr, const size_t row_st,
                                                     const size_t col_st, const size_t diag_count,
                                                     const size_t row_size, const size_t mat_size,
                                                     const size_t batch_size, const uint32_t &device_id,
                                                     cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Tracev2GradCalc(int64_t *din_addr, const int64_t *dout_addr, const size_t row_st,
                                                     const size_t col_st, const size_t diag_count,
                                                     const size_t row_size, const size_t mat_size,
                                                     const size_t batch_size, const uint32_t &device_id,
                                                     cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Tracev2GradCalc(uint8_t *din_addr, const uint8_t *dout_addr, const size_t row_st,
                                                     const size_t col_st, const size_t diag_count,
                                                     const size_t row_size, const size_t mat_size,
                                                     const size_t batch_size, const uint32_t &device_id,
                                                     cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Tracev2GradCalc(uint16_t *din_addr, const uint16_t *dout_addr, const size_t row_st,
                                                     const size_t col_st, const size_t diag_count,
                                                     const size_t row_size, const size_t mat_size,
                                                     const size_t batch_size, const uint32_t &device_id,
                                                     cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Tracev2GradCalc(uint32_t *din_addr, const uint32_t *dout_addr, const size_t row_st,
                                                     const size_t col_st, const size_t diag_count,
                                                     const size_t row_size, const size_t mat_size,
                                                     const size_t batch_size, const uint32_t &device_id,
                                                     cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Tracev2GradCalc(uint64_t *din_addr, const uint64_t *dout_addr, const size_t row_st,
                                                     const size_t col_st, const size_t diag_count,
                                                     const size_t row_size, const size_t mat_size,
                                                     const size_t batch_size, const uint32_t &device_id,
                                                     cudaStream_t stream);
