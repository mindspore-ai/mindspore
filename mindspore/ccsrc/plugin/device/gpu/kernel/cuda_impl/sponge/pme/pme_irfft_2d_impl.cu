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
#include "plugin/device/gpu/kernel/cuda_impl/sponge/pme/pme_irfft_2d_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/sponge/pme/pme_common.cuh"

template <typename T>
void PMEIRFFT2D(int Nfft, Complex<T> *input_tensor, T *output_tensor, const cufftHandle &FFT_plan_c2r,
            cudaStream_t stream) {
  cufftExecC2R(FFT_plan_c2r, reinterpret_cast<cufftComplex *>(input_tensor), output_tensor);
  return;
}

template CUDA_LIB_EXPORT
void PMEIRFFT2D<float>(int Nfft, Complex<float> *input_tensor, float *output_tensor,
                       const cufftHandle &FFT_plan_c2r, cudaStream_t stream);
