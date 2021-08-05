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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_RESTRAIN_RESTRAIN_FORCE_ATOM_ENERGY_VIRIAL_IMPL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_RESTRAIN_RESTRAIN_FORCE_ATOM_ENERGY_VIRIAL_IMPL_H_

#include <curand_kernel.h>
#include "runtime/device/gpu/cuda_common.h"

void restrainforcewithatomenergyandvirial(int restrain_numbers, int atom_numbers, const int *restrain_list,
                                          const float *crd_f, const float *crd_ref_f, const float weight,
                                          const float *boxlength_f, float *atom_ene, float *atom_virial, float *frc_f,
                                          cudaStream_t stream);
#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_RESTRAIN_RESTRAIN_FORCE_ATOM_ENERGY_VIRIAL_IMPL_H_
