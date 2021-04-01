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
#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_PME_PME_RECIPROCAL_FORCE_IMPL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_PME_PME_RECIPROCAL_FORCE_IMPL_H_

#include <cufft.h>
#include "runtime/device/gpu/cuda_common.h"

struct _VECTOR {
  float x;
  float y;
  float z;
};
void PMEReciprocalForce(int fftx, int ffty, int fftz, int atom_numbers, float beta, float *PME_BC, int *pme_uxyz,
                        float *pme_frxyz, float *PME_Q, float *pme_fq, int *PME_atom_near, int *pme_kxyz,
                        const int *uint_crd_f, const float *charge, float *force, int PME_Nin, int PME_Nall,
                        int PME_Nfft, const cufftHandle &PME_plan_r2c, const cufftHandle &PME_plan_c2r,
                        const _VECTOR &PME_inverse_box_vector, cudaStream_t stream);

#endif
