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

#include "angle_impl.cuh"
#include <math.h>

template <typename S>
__global__ void Angle(const size_t size, const Complex<S> *input, S *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += gridDim.x * blockDim.x) {
    output[pos] = atan2(input[pos].imag(), input[pos].real());
  }
  return;
}
template <typename T, typename S>
cudaError_t CalAngle(const size_t size, T *input, S *output, const uint32_t device_id, cudaStream_t cuda_stream) {
  Angle<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, output);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t CalAngle<Complex<float>, float>(const size_t size, Complex<float> *input,
                                                                     float *output, const uint32_t device_id,
                                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAngle<Complex<double>, double>(const size_t size, Complex<double> *input,
                                                                       double *output, const uint32_t device_id,
                                                                       cudaStream_t cuda_stream);
