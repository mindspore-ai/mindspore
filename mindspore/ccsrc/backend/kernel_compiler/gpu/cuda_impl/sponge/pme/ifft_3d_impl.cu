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
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/pme/ifft_3d_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/pme/pme_common.cuh"

template <typename T>
__global__ static void Merge_Complex(const int element_numbers, T *real_part, T *imag_part,
                                     cufftComplex *complex_element) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < element_numbers) {
    complex_element[i].x = real_part[i];
    complex_element[i].y = imag_part[i];
  }
}

template <typename T>
void IFFT3D(int Nfft, T *input_real, T *input_imag, T *complex_fq, T *output_tensor,
            const cufftHandle &FFT_plan_c2r, cudaStream_t stream) {
  cufftComplex *COMPLEX_FQ = reinterpret_cast<cufftComplex *>(complex_fq);
  Merge_Complex<T><<<Nfft / 1024 + 1, 1024, 0, stream>>>(Nfft, input_real, input_imag, COMPLEX_FQ);
  cufftExecC2R(FFT_plan_c2r, COMPLEX_FQ, output_tensor);
  return;
}

template void IFFT3D<float>(int Nfft, float *input_real, float *input_imag, float *complex_fq,
                            float *output_tensor, const cufftHandle &FFT_plan_c2r, cudaStream_t stream);
