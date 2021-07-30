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
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/pme/fft_3d_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/pme/pme_common.cuh"

template <typename T>
__global__ static void Split_Complex(const int element_numbers, T *real_part, T *imag_part,
                                     const cufftComplex *complex_element) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < element_numbers) {
    real_part[i] = complex_element[i].x;
    imag_part[i] = complex_element[i].y;
  }
}

template <typename T>
void FFT3D(int Nfft, T *input_tensor, T *complex_fq, T *output_real, T *output_imag,
           const cufftHandle &FFT_plan_r2c, cudaStream_t stream) {
  cufftComplex *COMPLEX_FQ = reinterpret_cast<cufftComplex *>(complex_fq);
  cufftExecR2C(FFT_plan_r2c, input_tensor, COMPLEX_FQ);
  Split_Complex<T><<<Nfft / 1024 + 1, 1024, 0, stream>>>(Nfft, output_real, output_imag, COMPLEX_FQ);
  return;
}

template void FFT3D<float>(int Nfft, float *input_tensor, float *complex_fq, float *output_real,
                           float *output_imag, const cufftHandle &FFT_plan_r2c, cudaStream_t stream);
