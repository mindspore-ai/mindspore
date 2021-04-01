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
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/neighbor_list/neighbor_list_impl.cuh"
#include <vector>
__global__ void Copy_List(const int element_numbers, const int *origin_list, int *list) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < element_numbers) {
    list[i] = origin_list[i];
  }
}
__global__ void Copy_List(const int element_numbers, const float *origin_list, float *list) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < element_numbers) {
    list[i] = origin_list[i];
  }
}

__global__ void Crd_To_Uint_Crd(const int atom_numbers, float *scale_factor, const VECTOR *crd,
                                UNSIGNED_INT_VECTOR *uint_crd) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    uint_crd[atom_i].uint_x = crd[atom_i].x * scale_factor[0];
    uint_crd[atom_i].uint_y = crd[atom_i].y * scale_factor[1];
    uint_crd[atom_i].uint_z = crd[atom_i].z * scale_factor[2];
    /*uint_crd[atom_i].uint_x = 2 * uint_crd[atom_i].uint_x;
    uint_crd[atom_i].uint_y = 2 * uint_crd[atom_i].uint_y;
    uint_crd[atom_i].uint_z = 2 * uint_crd[atom_i].uint_z;*/
    uint_crd[atom_i].uint_x = uint_crd[atom_i].uint_x << 1;
    uint_crd[atom_i].uint_y = uint_crd[atom_i].uint_y << 1;
    uint_crd[atom_i].uint_z = uint_crd[atom_i].uint_z << 1;
  }
}

__global__ void Vector_Translation(const int vector_numbers, VECTOR *vec_list, const VECTOR translation_vec) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < vector_numbers) {
    vec_list[i].x = vec_list[i].x + translation_vec.x;
    vec_list[i].y = vec_list[i].y + translation_vec.y;
    vec_list[i].z = vec_list[i].z + translation_vec.z;
  }
}
__global__ void Vector_Translation(const int vector_numbers, VECTOR *vec_list, const VECTOR *translation_vec) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < vector_numbers) {
    vec_list[i].x = vec_list[i].x + translation_vec[0].x;
    vec_list[i].y = vec_list[i].y + translation_vec[0].y;
    vec_list[i].z = vec_list[i].z + translation_vec[0].z;
  }
}
__global__ void Crd_Periodic_Map(const int atom_numbers, VECTOR *crd, const float *box_length) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    if (crd[atom_i].x >= 0) {
      if (crd[atom_i].x < box_length[0]) {
      } else {
        crd[atom_i].x = crd[atom_i].x - box_length[0];
      }
    } else {
      crd[atom_i].x = crd[atom_i].x + box_length[0];
    }

    if (crd[atom_i].y >= 0) {
      if (crd[atom_i].y < box_length[1]) {
      } else {
        crd[atom_i].y = crd[atom_i].y - box_length[1];
      }
    } else {
      crd[atom_i].y = crd[atom_i].y + box_length[1];
    }

    if (crd[atom_i].z >= 0) {
      if (crd[atom_i].z < box_length[2]) {
      } else {
        crd[atom_i].z = crd[atom_i].z - box_length[2];
      }
    } else {
      crd[atom_i].z = crd[atom_i].z + box_length[2];
    }
  }
}

__global__ void Clear_Grid_Bucket(const int grid_numbers, int *atom_numbers_in_grid_bucket, GRID_BUCKET *bucket) {
  int grid_serial = blockDim.x * blockIdx.x + threadIdx.x;
  if (grid_serial < grid_numbers) {
    GRID_BUCKET bucket_i = bucket[grid_serial];
    for (int i = 0; i < atom_numbers_in_grid_bucket[grid_serial]; i = i + 1) {
      bucket_i.atom_serial[i] = -1;
    }
    atom_numbers_in_grid_bucket[grid_serial] = 0;
  }
}

__global__ void Find_Atom_In_Grid_Serial(const int atom_numbers, const float *grid_length_inverse, const VECTOR *crd,
                                         const int *grid_N, const int gridxy, int *atom_in_grid_serial) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    int Nx = static_cast<float>(crd[atom_i].x) * grid_length_inverse[0];  // crd.x must < boxlength.x
    int Ny = static_cast<float>(crd[atom_i].y) * grid_length_inverse[1];
    int Nz = static_cast<float>(crd[atom_i].z) * grid_length_inverse[2];
    Nx = Nx & ((Nx - grid_N[0]) >> 31);
    Ny = Ny & ((Ny - grid_N[1]) >> 31);
    Nz = Nz & ((Nz - grid_N[2]) >> 31);
    atom_in_grid_serial[atom_i] = Nz * gridxy + Ny * grid_N[0] + Nx;
  }
}

__global__ void Put_Atom_In_Grid_Bucket(const int atom_numbers, const int *atom_in_grid_serial, GRID_BUCKET *bucket,
                                        int *atom_numbers_in_grid_bucket) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    int grid_serial = atom_in_grid_serial[atom_i];
    GRID_BUCKET bucket_i = bucket[grid_serial];
    int a = atom_numbers_in_grid_bucket[grid_serial];
    atomicCAS(&bucket_i.atom_serial[a], -1, atom_i);
    if (bucket_i.atom_serial[a] != atom_i) {
      while (true) {
        a = a + 1;
        atomicCAS(&bucket_i.atom_serial[a], -1, atom_i);
        if (bucket_i.atom_serial[a] == atom_i) {
          atomicAdd(&atom_numbers_in_grid_bucket[grid_serial], 1);
          break;
        }
      }
    } else {
      atomicAdd(&atom_numbers_in_grid_bucket[grid_serial], 1);
    }
  }
}
__global__ void Find_atom_neighbors(const int atom_numbers, const UNSIGNED_INT_VECTOR *uint_crd,
                                    const float *uint_dr_to_dr_cof, const int *atom_in_grid_serial,
                                    const GRID_POINTER *gpointer, const GRID_BUCKET *bucket,
                                    const int *atom_numbers_in_grid_bucket, NEIGHBOR_LIST *nl,
                                    const float cutoff_skin_square) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    int grid_serial = atom_in_grid_serial[atom_i];
    int grid_serial2;
    int atom_numbers_in_nl_lin = 0;
    int atom_j;
    int int_x;
    int int_y;
    int int_z;
    UNSIGNED_INT_VECTOR uint_crd_i = uint_crd[atom_i];
    NEIGHBOR_LIST nl_i = nl[atom_i];
    GRID_POINTER gpointer_i = gpointer[grid_serial];
    VECTOR dr;
    float dr2;
    for (int grid_cycle = 0; grid_cycle < 125; grid_cycle = grid_cycle + 1) {
      grid_serial2 = gpointer_i.grid_serial[grid_cycle];
      GRID_BUCKET bucket_i = bucket[grid_serial2];
      for (int i = 0; i < atom_numbers_in_grid_bucket[grid_serial2]; i = i + 1) {
        atom_j = bucket_i.atom_serial[i];
        if (atom_j > atom_i) {
          int_x = uint_crd[atom_j].uint_x - uint_crd_i.uint_x;
          int_y = uint_crd[atom_j].uint_y - uint_crd_i.uint_y;
          int_z = uint_crd[atom_j].uint_z - uint_crd_i.uint_z;
          dr.x = uint_dr_to_dr_cof[0] * int_x;
          dr.y = uint_dr_to_dr_cof[1] * int_y;
          dr.z = uint_dr_to_dr_cof[2] * int_z;
          dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
          if (dr2 < cutoff_skin_square) {
            nl_i.atom_serial[atom_numbers_in_nl_lin] = atom_j;
            atom_numbers_in_nl_lin = atom_numbers_in_nl_lin + 1;
          }
        }
      }
    }  // 124 grid cycle
    nl[atom_i].atom_numbers = atom_numbers_in_nl_lin;
  }
}

__global__ void Is_need_refresh_neighbor_list_cuda(const int atom_numbers, const VECTOR *crd, const VECTOR *old_crd,
                                                   const float half_skin_square, int *need_refresh_flag) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < atom_numbers) {
    VECTOR r1 = crd[i];
    VECTOR r2 = old_crd[i];
    r1.x = r1.x - r2.x;
    r1.y = r1.y - r2.y;
    r1.z = r1.z - r2.z;
    float r1_2 = r1.x * r1.x + r1.y * r1.y + r1.z * r1.z;
    if (r1_2 > half_skin_square) {
      atomicExch(&need_refresh_flag[0], 1);
    }
  }
}

__global__ void Delete_Excluded_Atoms_Serial_In_Neighbor_List(const int atom_numbers, NEIGHBOR_LIST *nl,
                                                              const int *excluded_list_start, const int *excluded_list,
                                                              const int *excluded_atom_numbers) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    int excluded_number = excluded_atom_numbers[atom_i];
    if (excluded_number > 0) {
      int list_start = excluded_list_start[atom_i];
      int atom_min = excluded_list[list_start];
      int list_end = list_start + excluded_number;
      int atom_max = excluded_list[list_end - 1];
      NEIGHBOR_LIST nl_i = nl[atom_i];
      int atomnumbers_in_nl_lin = nl_i.atom_numbers;
      int atom_j;
      int excluded_atom_numbers_lin = list_end - list_start;
      int excluded_atom_numbers_count = 0;
      for (int i = 0; i < atomnumbers_in_nl_lin; i = i + 1) {
        atom_j = nl_i.atom_serial[i];
        if (atom_j < atom_min || atom_j > atom_max) {
          continue;
        } else {
          for (int j = list_start; j < list_end; j = j + 1) {
            if (atom_j == excluded_list[j]) {
              atomnumbers_in_nl_lin = atomnumbers_in_nl_lin - 1;
              nl_i.atom_serial[i] = nl_i.atom_serial[atomnumbers_in_nl_lin];
              excluded_atom_numbers_count = excluded_atom_numbers_count + 1;
              i = i - 1;
            }
          }
          if (excluded_atom_numbers_count < excluded_atom_numbers_lin) {
          } else {
            break;
          }  // break
        }    // in the range of excluded min to max
      }      // cycle for neighbors
      nl[atom_i].atom_numbers = atomnumbers_in_nl_lin;
    }  // if need excluded
  }
}

void Refresh_Neighbor_List(int *refresh_sign, const int thread, const int atom_numbers, VECTOR *crd, VECTOR *old_crd,
                           UNSIGNED_INT_VECTOR *uint_crd, float *crd_to_uint_crd_cof, float *uint_dr_to_dr_cof,
                           int *atom_in_grid_serial, const float skin, float *box_length, const GRID_POINTER *gpointer,
                           GRID_BUCKET *bucket, int *atom_numbers_in_grid_bucket, NEIGHBOR_LIST *d_nl,
                           int *excluded_list_start, int *excluded_list, int *excluded_numbers,
                           float cutoff_skin_square, int grid_numbers, float *grid_length_inverse, int *grid_N, int Nxy,
                           cudaStream_t stream) {
  if (refresh_sign[0] == 1) {
    VECTOR trans_vec = {-skin, -skin, -skin};
    Clear_Grid_Bucket<<<ceilf(static_cast<float>(grid_numbers) / thread), thread, 0, stream>>>(
      grid_numbers, atom_numbers_in_grid_bucket, bucket);

    Vector_Translation<<<ceilf(static_cast<float>(atom_numbers) / thread), thread, 0, stream>>>(atom_numbers, crd,
                                                                                                trans_vec);

    Crd_Periodic_Map<<<ceilf(static_cast<float>(atom_numbers) / thread), thread, 0, stream>>>(atom_numbers, crd,
                                                                                              box_length);

    Find_Atom_In_Grid_Serial<<<ceilf(static_cast<float>(atom_numbers) / thread), thread, 0, stream>>>(
      atom_numbers, grid_length_inverse, crd, grid_N, Nxy, atom_in_grid_serial);

    trans_vec.x = -trans_vec.x;
    trans_vec.y = -trans_vec.y;
    trans_vec.z = -trans_vec.z;

    Vector_Translation<<<ceilf(static_cast<float>(atom_numbers) / thread), thread, 0, stream>>>(atom_numbers, crd,
                                                                                                trans_vec);

    Copy_List<<<ceilf(static_cast<float>(3. * atom_numbers) / thread), thread, 0, stream>>>(
      3 * atom_numbers, reinterpret_cast<float *>(crd), reinterpret_cast<float *>(old_crd));

    Put_Atom_In_Grid_Bucket<<<ceilf(static_cast<float>(atom_numbers) / thread), thread, 0, stream>>>(
      atom_numbers, atom_in_grid_serial, bucket, atom_numbers_in_grid_bucket);

    Crd_To_Uint_Crd<<<ceilf(static_cast<float>(atom_numbers) / thread), thread, 0, stream>>>(
      atom_numbers, crd_to_uint_crd_cof, crd, uint_crd);

    Find_atom_neighbors<<<ceilf(static_cast<float>(atom_numbers) / thread), thread, 0, stream>>>(
      atom_numbers, uint_crd, uint_dr_to_dr_cof, atom_in_grid_serial, gpointer, bucket, atom_numbers_in_grid_bucket,
      d_nl, cutoff_skin_square);

    Delete_Excluded_Atoms_Serial_In_Neighbor_List<<<ceilf(static_cast<float>(atom_numbers) / thread), thread, 0,
                                                    stream>>>(atom_numbers, d_nl, excluded_list_start, excluded_list,
                                                              excluded_numbers);
    refresh_sign[0] = 0;
  }
}

void Refresh_Neighbor_List_First_Time(int *refresh_sign, const int thread, const int atom_numbers, VECTOR *crd,
                                      VECTOR *old_crd, UNSIGNED_INT_VECTOR *uint_crd, float *crd_to_uint_crd_cof,
                                      float *uint_dr_to_dr_cof, int *atom_in_grid_serial, const float skin,
                                      float *box_length, const GRID_POINTER *gpointer, GRID_BUCKET *bucket,
                                      int *atom_numbers_in_grid_bucket, NEIGHBOR_LIST *d_nl, int *excluded_list_start,
                                      int *excluded_list, int *excluded_numbers, float cutoff_skin_square,
                                      int grid_numbers, float *grid_length_inverse, int *grid_N, int Nxy,
                                      cudaStream_t stream) {
  VECTOR trans_vec = {skin, skin, skin};
  Clear_Grid_Bucket<<<ceilf(static_cast<float>(grid_numbers) / 32), 32, 0, stream>>>(
    grid_numbers, atom_numbers_in_grid_bucket, bucket);
  Crd_Periodic_Map<<<ceilf(static_cast<float>(atom_numbers) / 32), 32, 0, stream>>>(atom_numbers, crd, box_length);
  Find_Atom_In_Grid_Serial<<<ceilf(static_cast<float>(atom_numbers) / 32), 32, 0, stream>>>(
    atom_numbers, grid_length_inverse, crd, grid_N, Nxy, atom_in_grid_serial);
  Vector_Translation<<<ceilf(static_cast<float>(atom_numbers) / 32), 32, 0, stream>>>(atom_numbers, crd, trans_vec);
  Copy_List<<<ceilf(static_cast<float>(3. * atom_numbers) / 32), 32, 0, stream>>>(
    3 * atom_numbers, reinterpret_cast<float *>(crd), reinterpret_cast<float *>(old_crd));
  Put_Atom_In_Grid_Bucket<<<ceilf(static_cast<float>(atom_numbers) / 32), 32, 0, stream>>>(
    atom_numbers, atom_in_grid_serial, bucket, atom_numbers_in_grid_bucket);
  Crd_To_Uint_Crd<<<ceilf(static_cast<float>(atom_numbers) / 32), 32, 0, stream>>>(atom_numbers, crd_to_uint_crd_cof,
                                                                                   crd, uint_crd);

  Find_atom_neighbors<<<ceilf(static_cast<float>(atom_numbers) / thread), thread, 0, stream>>>(
    atom_numbers, uint_crd, uint_dr_to_dr_cof, atom_in_grid_serial, gpointer, bucket, atom_numbers_in_grid_bucket, d_nl,
    cutoff_skin_square);
  Delete_Excluded_Atoms_Serial_In_Neighbor_List<<<ceilf(static_cast<float>(atom_numbers) / thread), thread, 0,
                                                  stream>>>(atom_numbers, d_nl, excluded_list_start, excluded_list,
                                                            excluded_numbers);
}

__global__ void construct_neighbor_list_kernel(int atom_numbers, int max_neighbor_numbers, int *nl_atom_numbers,
                                               int *nl_atom_serial, NEIGHBOR_LIST *nl) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers; i += gridDim.x * blockDim.x) {
    nl[i].atom_numbers = nl_atom_numbers[i];
    nl[i].atom_serial = nl_atom_serial + i * max_neighbor_numbers;
  }
}

void Construct_Neighbor_List(int atom_numbers, int max_neighbor_numbers, int *nl_atom_numbers, int *nl_atom_serial,
                             NEIGHBOR_LIST *nl, cudaStream_t stream) {
  construct_neighbor_list_kernel<<<ceilf(static_cast<float>(atom_numbers) / 128), 128, 0, stream>>>(
    atom_numbers, max_neighbor_numbers, nl_atom_numbers, nl_atom_serial, nl);
}

__global__ void copy_neighbor_list_atom_number(int atom_numbers, NEIGHBOR_LIST *nl, int *nl_atom_numbers) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers; i += gridDim.x * blockDim.x) {
    nl_atom_numbers[i] = nl[i].atom_numbers;
  }
}

void CopyNeighborListAtomNumber(int atom_numbers, NEIGHBOR_LIST *nl, int *nl_atom_numbers, cudaStream_t stream) {
  copy_neighbor_list_atom_number<<<ceilf(static_cast<float>(atom_numbers) / 128), 128, 0, stream>>>(atom_numbers, nl,
                                                                                                    nl_atom_numbers);
}

void Refresh_Neighbor_List_No_Check(int grid_numbers, int atom_numbers, float skin, int Nxy, float cutoff_skin_square,
                                    int *grid_N, float *box_length, int *atom_numbers_in_grid_bucket,
                                    float *grid_length_inverse, int *atom_in_grid_serial, GRID_BUCKET *bucket,
                                    VECTOR *crd, VECTOR *old_crd, float *crd_to_uint_crd_cof,
                                    UNSIGNED_INT_VECTOR *uint_crd, float *uint_dr_to_dr_cof, GRID_POINTER *gpointer,
                                    NEIGHBOR_LIST *d_nl, int *excluded_list_start, int *excluded_list,
                                    int *excluded_numbers, cudaStream_t stream) {
  VECTOR trans_vec = {-skin, -skin, -skin};

  Clear_Grid_Bucket<<<ceilf(static_cast<float>(grid_numbers) / 32), 32, 0, stream>>>(
    grid_numbers, atom_numbers_in_grid_bucket, bucket);

  Vector_Translation<<<ceilf(static_cast<float>(atom_numbers) / 32), 32, 0, stream>>>(atom_numbers, crd, trans_vec);

  Crd_Periodic_Map<<<ceilf(static_cast<float>(atom_numbers) / 32), 32, 0, stream>>>(atom_numbers, crd, box_length);

  Find_Atom_In_Grid_Serial<<<ceilf(static_cast<float>(atom_numbers) / 32), 32, 0, stream>>>(
    atom_numbers, grid_length_inverse, crd, grid_N, Nxy, atom_in_grid_serial);
  trans_vec.x = -trans_vec.x;
  trans_vec.y = -trans_vec.y;
  trans_vec.z = -trans_vec.z;
  Vector_Translation<<<ceilf(static_cast<float>(atom_numbers) / 32), 32, 0, stream>>>(atom_numbers, crd, trans_vec);

  cudaMemcpyAsync(old_crd, crd, sizeof(VECTOR) * atom_numbers, cudaMemcpyDeviceToDevice, stream);

  Put_Atom_In_Grid_Bucket<<<ceilf(static_cast<float>(atom_numbers) / 32), 32, 0, stream>>>(
    atom_numbers, atom_in_grid_serial, bucket, atom_numbers_in_grid_bucket);

  Crd_To_Uint_Crd<<<ceilf(static_cast<float>(atom_numbers) / 32), 32, 0, stream>>>(atom_numbers, crd_to_uint_crd_cof,
                                                                                   crd, uint_crd);

  Find_atom_neighbors<<<ceilf(static_cast<float>(atom_numbers) / 32), 32, 0, stream>>>(
    atom_numbers, uint_crd, uint_dr_to_dr_cof, atom_in_grid_serial, gpointer, bucket, atom_numbers_in_grid_bucket, d_nl,
    cutoff_skin_square);

  Delete_Excluded_Atoms_Serial_In_Neighbor_List<<<ceilf(static_cast<float>(atom_numbers) / 32), 32, 0, stream>>>(
    atom_numbers, d_nl, excluded_list_start, excluded_list, excluded_numbers);
}

__global__ void Mul_half(float *src, float *dst) {
  int index = threadIdx.x;
  if (index < 3) {
    dst[index] = src[index] * 0.5;
  }
}

void Neighbor_List_Update(int grid_numbers, int atom_numbers, int *d_refresh_count, int refresh_interval,
                          int not_first_time, float skin, int Nxy, float cutoff_square, float cutoff_with_skin_square,
                          int *grid_N, float *box_length, int *atom_numbers_in_grid_bucket, float *grid_length_inverse,
                          int *atom_in_grid_serial, GRID_BUCKET *bucket, float *crd, float *old_crd,
                          float *crd_to_uint_crd_cof, float *half_crd_to_uint_crd_cof, unsigned int *uint_crd,
                          float *uint_dr_to_dr_cof, GRID_POINTER *gpointer, NEIGHBOR_LIST *d_nl,
                          int *excluded_list_start, int *excluded_list, int *excluded_numbers, float half_skin_square,
                          int *is_need_refresh_neighbor_list, cudaStream_t stream) {
  if (not_first_time) {
    if (refresh_interval > 0) {
      std::vector<int> refresh_count_list(1);
      cudaMemcpyAsync(refresh_count_list.data(), d_refresh_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      int refresh_count = refresh_count_list[0];

      if (refresh_count % refresh_interval == 0) {
        Mul_half<<<1, 3, 0, stream>>>(crd_to_uint_crd_cof, half_crd_to_uint_crd_cof);
        Refresh_Neighbor_List_No_Check(grid_numbers, atom_numbers, skin, Nxy, cutoff_square, grid_N, box_length,
                                       atom_numbers_in_grid_bucket, grid_length_inverse, atom_in_grid_serial, bucket,
                                       reinterpret_cast<VECTOR *>(crd), reinterpret_cast<VECTOR *>(old_crd),
                                       half_crd_to_uint_crd_cof, reinterpret_cast<UNSIGNED_INT_VECTOR *>(uint_crd),
                                       uint_dr_to_dr_cof, gpointer, d_nl, excluded_list_start, excluded_list,
                                       excluded_numbers, stream);
      }
      refresh_count += 1;
      cudaMemcpyAsync(d_refresh_count, &refresh_count, sizeof(int), cudaMemcpyHostToDevice, stream);
    } else {
      Is_need_refresh_neighbor_list_cuda<<<ceilf(static_cast<float>(atom_numbers) / 128), 128, 0, stream>>>(
        atom_numbers, reinterpret_cast<VECTOR *>(crd), reinterpret_cast<VECTOR *>(old_crd), half_skin_square,
        is_need_refresh_neighbor_list);
      Mul_half<<<1, 3, 0, stream>>>(crd_to_uint_crd_cof, half_crd_to_uint_crd_cof);
      Refresh_Neighbor_List(is_need_refresh_neighbor_list, 32, atom_numbers, reinterpret_cast<VECTOR *>(crd),
                            reinterpret_cast<VECTOR *>(old_crd), reinterpret_cast<UNSIGNED_INT_VECTOR *>(uint_crd),
                            half_crd_to_uint_crd_cof, uint_dr_to_dr_cof, atom_in_grid_serial, skin, box_length,
                            gpointer, bucket, atom_numbers_in_grid_bucket, d_nl, excluded_list_start, excluded_list,
                            excluded_numbers, cutoff_with_skin_square, grid_numbers, grid_length_inverse, grid_N, Nxy,
                            stream);
    }
  } else {
    Mul_half<<<1, 3, 0, stream>>>(crd_to_uint_crd_cof, half_crd_to_uint_crd_cof);
    Refresh_Neighbor_List_First_Time(
      is_need_refresh_neighbor_list, 32, atom_numbers, reinterpret_cast<VECTOR *>(crd),
      reinterpret_cast<VECTOR *>(old_crd), reinterpret_cast<UNSIGNED_INT_VECTOR *>(uint_crd), half_crd_to_uint_crd_cof,
      uint_dr_to_dr_cof, atom_in_grid_serial, skin, box_length, gpointer, bucket, atom_numbers_in_grid_bucket, d_nl,
      excluded_list_start, excluded_list, excluded_numbers, cutoff_with_skin_square, grid_numbers, grid_length_inverse,
      grid_N, Nxy, stream);
  }
}
