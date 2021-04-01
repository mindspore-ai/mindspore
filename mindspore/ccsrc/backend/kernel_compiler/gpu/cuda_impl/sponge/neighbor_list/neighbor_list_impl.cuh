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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_NEIGHBOR_LIST_IMPL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_NEIGHBOR_LIST_IMPL_H_

struct VECTOR {
  float x;
  float y;
  float z;
};
struct INT_VECTOR {
  int int_x;
  int int_y;
  int int_z;
};
struct UNSIGNED_INT_VECTOR {
  unsigned int uint_x;
  unsigned int uint_y;
  unsigned int uint_z;
};
struct NEIGHBOR_LIST {
  int atom_numbers;
  int *atom_serial;
};
struct GRID_BUCKET {
  int *atom_serial;
};
struct GRID_POINTER {
  int *grid_serial;
};

void Construct_Neighbor_List(int grid_numbers, int max_neighbor_numbers, int *nl_atom_numbers, int *nl_atom_serial,
                             NEIGHBOR_LIST *nl, cudaStream_t stream);

void CopyNeighborListAtomNumber(int atom_numbers, NEIGHBOR_LIST *nl, int *nl_atom_numbers, cudaStream_t stream);

void Neighbor_List_Update(int grid_numbers, int atom_numbers, int* d_refresh_count, int refresh_interval,
                          int not_first_time, float skin, int Nxy, float cutoff_square, float cutoff_with_skin_square,
                          int *grid_N, float *box_length, int *atom_numbers_in_grid_bucket, float *grid_length_inverse,
                          int *atom_in_grid_serial, GRID_BUCKET *bucket, float *crd, float *old_crd,
                          float *crd_to_uint_crd_cof, float *half_crd_to_uint_crd_cof, unsigned int *uint_crd,
                          float *uint_dr_to_dr_cof, GRID_POINTER *gpointer, NEIGHBOR_LIST *d_nl,
                          int *excluded_list_start, int *excluded_list, int *excluded_numbers, float half_skin_square,
                          int *is_need_refresh_neighbor_list, cudaStream_t stream);

#endif
