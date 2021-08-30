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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common/map_center_of_mass_impl.cuh"

__global__ void Map_Center_Of_Mass(int residue_numbers, int *start, int *end,
                float *scaler, VECTOR *center_of_mass, VECTOR *box_length, VECTOR *no_wrap_crd, VECTOR *crd) {
        VECTOR trans_vec;
        VECTOR com;
        for (int residue_i = blockDim.x*blockIdx.x + threadIdx.x; residue_i < residue_numbers;
             residue_i += gridDim.x * blockDim.x) {
                com = center_of_mass[residue_i];

                trans_vec.x = com.x - floorf(com.x / box_length[0].x) * box_length[0].x;
                trans_vec.y = com.y - floorf(com.y / box_length[0].y) * box_length[0].y;
                trans_vec.z = com.z - floorf(com.z / box_length[0].z) * box_length[0].z;
                trans_vec = scaler[0] * trans_vec - com;

                for (int atom_i = start[residue_i] + threadIdx.y; atom_i < end[residue_i]; atom_i += blockDim.y) {
                    crd[atom_i] = no_wrap_crd[atom_i] + trans_vec;
                }
        }
}

void MapCenterOfMass(int residue_numbers, int *start, int *end, float *center_of_mass_f,
                     float *box_length_f, float *no_wrap_crd_f, float *crd_f, float* scaler, cudaStream_t stream) {
  VECTOR *crd = reinterpret_cast<VECTOR *>(crd_f);
  VECTOR *no_wrap_crd = reinterpret_cast<VECTOR *>(no_wrap_crd_f);
  VECTOR *box_length = reinterpret_cast<VECTOR *>(box_length_f);
  VECTOR *center_of_mass = reinterpret_cast<VECTOR *>(center_of_mass_f);
  Map_Center_Of_Mass<<<20, { 32, 4 } , 0, stream>>>(residue_numbers, start, end, scaler, center_of_mass, box_length,
                                                    no_wrap_crd, crd);
  return;
}

void MapCenterOfMass(int residue_numbers, int *start, int *end, float *center_of_mass_f,
                     float *box_length_f, float *no_wrap_crd_f, float *crd_f, float* scaler, cudaStream_t stream);
