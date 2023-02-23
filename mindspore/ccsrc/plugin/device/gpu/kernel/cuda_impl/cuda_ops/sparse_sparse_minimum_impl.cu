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
#include "sparse_sparse_minimum_impl.cuh"
#include "include/cuda_fp16.h"
#include "include/cuda_runtime.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"


template <typename T>
__global__ void SparseSparseMinimum1(const T *a_indices, const T *b_indices,
                     int64_t *ab_status, const int64_t rank_1,
                            const int64_t a_indices_num, const int64_t b_indices_num ) {
    for (int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
                x < a_indices_num && y < b_indices_num; x+=blockDim.x * gridDim.x, y+=blockDim.y * gridDim.y) {
        if (x < a_indices_num&&y < b_indices_num) {
            for (int64_t i = 0; i < rank_1; i++) {
                if (a_indices[x*rank_1+i] > b_indices[y*rank_1+i]) {
                    ab_status[y*a_indices_num+x] = 1;
                    return;
                }
                 if (a_indices[x*rank_1+i] == b_indices[y*rank_1+i]) {
                    ab_status[y*a_indices_num+x] = 0;
                    continue;
                }
                 if (a_indices[x*rank_1+i] < b_indices[y*rank_1+i]) {
                    ab_status[y*a_indices_num+x] = -1;
                    return;
                }
            }
        }
    }
}


template <typename T>
__global__ void SparseSparseMinimum2(const T *a_indices, const T *b_indices,
                                int64_t *ab_status, const int64_t a_indices_num, const int64_t b_indices_num,
                                    const int64_t rank_1, int64_t *sum_ptr,
                                     int64_t *ab_stauts1, int64_t *ab_stauts2) {
    int64_t count = 0;
    int64_t i = 0;
    int64_t j = 0;
    while (i < a_indices_num&&j < b_indices_num) {
        if (ab_status[j*a_indices_num+i] == -1) {
            ab_stauts1[count] = 1;
            ab_stauts2[count] = i;
            count++;
            i++;
            continue;
        }
        if (ab_status[j*a_indices_num+i] == 0) {
            ab_stauts1[count] = -i;
            ab_stauts2[count] = j;
            count++;
            i++;
            j++;
            continue;
        }
        if (ab_status[j*a_indices_num+i] == 1) {
            ab_stauts1[count] = 2;
            ab_stauts2[count] = j;
            count++;
            j++;
            continue;
        }
    }
    for (int64_t y1 = i; y1 < a_indices_num; y1++) {
         ab_stauts1[count] = 1;
         ab_stauts2[count] = y1;
         count++;
    }

    for (int64_t y1 = j; y1 < b_indices_num; y1++) {
         ab_stauts1[count] = 2;
         ab_stauts2[count] = y1;
         count++;
    }
    *sum_ptr = count;
}

template <typename T, typename S>
__global__ void SparseSparseMinimum3(const T *a_indices, const S *a_values,
                                  const T *b_indices, const S *b_values,
                                  int64_t *ab_stauts1, int64_t *ab_stauts2,
                                    const int64_t rank_1, T *output_indices,
                                    S *output_values, int64_t limit1) {
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < limit1; x += blockDim.x * gridDim.x) {
        int64_t mid1 = ab_stauts2[x];
        int64_t mid2 = -ab_stauts1[x];
        if (x >= limit1)
            return;
        if (ab_stauts1[x] == 3) {
            return;
        } else if (ab_stauts1[x] == 1) {
            for (int64_t m = 0; m < rank_1; m++) {
                output_indices[x*rank_1+m] = a_indices[mid1*rank_1+m];
            }
            output_values[x] = a_values[mid1] < static_cast<S>(0) ? a_values[mid1] : static_cast<S>(0);
        } else if (ab_stauts1[x] == 2) {
            for (int64_t m = 0; m < rank_1; m++) {
               output_indices[x*rank_1+m] = b_indices[mid1*rank_1+m];
            }
            output_values[x] = b_values[mid1] < static_cast<S>(0) ? b_values[mid1] : static_cast<S>(0);
        } else if (ab_stauts1[x] <= 0) {
            for (int64_t m = 0; m < rank_1; m++) {
               output_indices[x*rank_1+m] = b_indices[mid1*rank_1+m];
            }
            output_values[x] = a_values[mid2] < b_values[mid1] ? a_values[mid2] : b_values[mid1];
        }
    }
}




template <typename T>
__global__ void SparseSparseMinimum3(const T *a_indices, const half *a_values,
                                    const T *b_indices, const half *b_values,
                                    int64_t *ab_stauts1, int64_t *ab_stauts2,
                                    const int64_t rank_1, T *output_indices,
                                    half *output_values, int64_t limit1) {
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < limit1; x += blockDim.x * gridDim.x) {
        int64_t mid1 = ab_stauts2[x];
        int64_t mid2 = -ab_stauts1[x];
        if (ab_stauts1[x] == 3)
            return;
        if (ab_stauts1[x] == 1) {
            for (int64_t m = 0; m < rank_1; m++) {
                output_indices[x*rank_1+m] = a_indices[mid1*rank_1+m];
            }
            output_values[x] = __half2float(a_values[mid1]) < 0 ? a_values[mid1] : __float2half(0.0);
        } else if (ab_stauts1[x] == 2) {
            for (int64_t m = 0; m < rank_1; m++) {
               output_indices[x*rank_1+m] = b_indices[mid1*rank_1+m];
            }
            output_values[x] = __half2float(b_values[mid1]) < 0 ? b_values[mid1] : __float2half(0.0);
        } else if (ab_stauts1[x] <= 0) {
            for (int64_t m = 0; m < rank_1; m++) {
               output_indices[x*rank_1+m] = b_indices[mid1*rank_1+m];
            }
            output_values[x] = a_values[mid2] < b_values[mid1] ? a_values[mid2] : b_values[mid1];
        }
    }
}




template <typename T, typename S>
__global__ void Min_test1(const int64_t a_len, const T *a_indices,
                        const S *a_values,
                        T *output_indices, S *output_values,
                        const int64_t rank_1, int64_t *sum_ptr) {
    *sum_ptr = a_len;
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < a_len; x += blockDim.x * gridDim.x) {
        for (int64_t j = 0; j < rank_1; j++) {
            output_indices[x*rank_1+j] = a_indices[x*rank_1+j];
        }
        output_values[x] = a_values[x] < static_cast<S>(0) ? a_values[x] : static_cast<S>(0);
    }
}

template <typename T>
__global__ void Min_test1(const int64_t a_len, const T *a_indices, const half *a_values,
                        T *output_indices, half *output_values,
                        const int64_t rank_1, int64_t *sum_ptr) {
    *sum_ptr = a_len;
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < a_len; x += blockDim.x * gridDim.x) {
        for (int64_t j = 0; j < rank_1; j++) {
            output_indices[x*rank_1+j] = a_indices[x*rank_1+j];
        }
        output_values[x] = __half2float(a_values[x]) < 0 ? a_values[x] : __float2half(0.0);
    }
}

__global__ void Min_test2(int64_t *sum_ptr) {
    *sum_ptr = 0;
}

template <typename T, typename S>
CUDA_LIB_EXPORT void SparseSparseMinimum(const T *a_indices, const S *a_values, const T *b_indices, const S *b_values,
                               T *sum_indices, S *sum_values, int64_t *ab_status_ptr,
                               int64_t *sum_ptr, const int64_t a_indices_num,
                               const int64_t b_indices_num, const int64_t rank_1,
                               cudaStream_t cuda_stream1, const uint32_t &device_id,
                               int64_t *ab_status_ptr1, int64_t *ab_status_ptr2) {
        if (a_indices_num != 0&&b_indices_num == 0) {
            Min_test1<<<CUDA_BLOCKS(device_id, a_indices_num), CUDA_THREADS(device_id)>>>
            (a_indices_num, a_indices, a_values, sum_indices, sum_values, rank_1, sum_ptr);
            cudaDeviceSynchronize();
            return;
        }
        if (a_indices_num == 0&&b_indices_num != 0) {
            Min_test1<<<CUDA_BLOCKS(device_id, b_indices_num), CUDA_THREADS(device_id)>>>
            (b_indices_num, b_indices, b_values, sum_indices, sum_values, rank_1, sum_ptr);
            cudaDeviceSynchronize();
            return;
        }
        if (a_indices_num == 0&&b_indices_num == 0) {
            Min_test2<<<1, 1>>>(sum_ptr);
            return;
        }
        const int block1 = 32;
        const int block2 = 32;
        const int grid1 = (a_indices_num+block1-1)/block1;
        const int grid2 = (b_indices_num+block2-1)/block2;
        const int grid3 = (a_indices_num+b_indices_num+block1-1)/block1;
        dim3 block12(block1, block2);
        dim3 grid12(grid1, grid2);
        SparseSparseMinimum1<<<grid12, block12>>>(a_indices, b_indices, ab_status_ptr
        , rank_1, a_indices_num, b_indices_num);
        cudaDeviceSynchronize();
        SparseSparseMinimum2<<<1, 1>>>(a_indices, b_indices, ab_status_ptr,
        a_indices_num, b_indices_num, rank_1, sum_ptr, ab_status_ptr1, ab_status_ptr2);
        cudaDeviceSynchronize();
        SparseSparseMinimum3<<<grid3, block1>>>(a_indices, a_values, b_indices,
                                              b_values, ab_status_ptr1, ab_status_ptr2,
                                                rank_1, sum_indices, sum_values,
                                                a_indices_num+b_indices_num);
        cudaDeviceSynchronize();
}

#define GPU_SPARSE_SPARSE_MINIMUM_GRAD_EXPORT_REGISTER(index_type1, val_type1)                      \
  template CUDA_LIB_EXPORT void SparseSparseMinimum<index_type1, val_type1>(const index_type1 *a_indices,   \
    const val_type1 *a_values, const index_type1 *b_indices, const val_type1 *b_values,           \
    index_type1 *sum_indices, val_type1 *sum_values,                                             \
     int64_t *ab_status_ptr, int64_t* sum_ptr,      \
    const int64_t a_indices_num, const int64_t b_indices_num, const int64_t rank_1,                  \
    cudaStream_t cuda_stream1, const uint32_t &device_id, int64_t *ab_status_ptr1, int64_t *ab_status_ptr2);


GPU_SPARSE_SPARSE_MINIMUM_GRAD_EXPORT_REGISTER(int64_t, int8_t)
GPU_SPARSE_SPARSE_MINIMUM_GRAD_EXPORT_REGISTER(int64_t, int16_t)
GPU_SPARSE_SPARSE_MINIMUM_GRAD_EXPORT_REGISTER(int64_t, int32_t)
GPU_SPARSE_SPARSE_MINIMUM_GRAD_EXPORT_REGISTER(int64_t, int64_t)
GPU_SPARSE_SPARSE_MINIMUM_GRAD_EXPORT_REGISTER(int64_t, float)
GPU_SPARSE_SPARSE_MINIMUM_GRAD_EXPORT_REGISTER(int64_t, half)
GPU_SPARSE_SPARSE_MINIMUM_GRAD_EXPORT_REGISTER(int64_t, double)
GPU_SPARSE_SPARSE_MINIMUM_GRAD_EXPORT_REGISTER(int64_t, uint8_t)
GPU_SPARSE_SPARSE_MINIMUM_GRAD_EXPORT_REGISTER(int64_t, uint16_t)
