/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, softwareg
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <iostream>
#include <limits>

#include "non_max_suppression_with_overlaps_impl.cuh"
__device__ __host__ int NumRoundUpPower2(int v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

template <typename T>
__inline__ __device__ void Swap(T *lhs, T *rhs) {
  T tmp = lhs[0];
  lhs[0] = rhs[0];
  rhs[0] = tmp;
}

// Initialize per row mask array to all true
__global__ void RowMaskInit(int numSq, bool *row_mask) {
  for (int mat_pos = blockIdx.x * blockDim.x + threadIdx.x; mat_pos < numSq; mat_pos += blockDim.x * gridDim.x) {
    row_mask[mat_pos] = true;
  }
}

// populated return mask (init to all true) and return index array
__global__ void NmsPreProcess(const int num, bool *sel_boxes) {
  for (int box_num = blockIdx.x * blockDim.x + threadIdx.x; box_num < num; box_num += blockDim.x * gridDim.x) {
    sel_boxes[box_num] = true;
  }
}

// Run parallel NMS pass
// Every position in the row_mask array is updated wit correct IOU decision
// after being init to all True
template <typename T>
__global__ void NmsWithOverlapPass(const int valid_num, const int num, const T *IOU_value, T *overlaps, bool *row_mask,
                                   int *index_buff) {
  int box_i, box_j;  // actual input data indexing
  for (int mask_index = blockIdx.x * blockDim.x + threadIdx.x; mask_index < valid_num * valid_num;
       mask_index += blockDim.x * gridDim.x) {
    box_i = mask_index / valid_num;  // row in 2d row_mask array
    box_j = mask_index % valid_num;  // col in 2d row_mask array
    if (box_j > box_i) {             // skip when box_j index lower/equal to box_i - will
                                     // remain true
      row_mask[mask_index] =
          overlaps[index_buff[valid_num - box_i - 1] + index_buff[valid_num - box_j - 1] * num] > IOU_value[0] ? false
                                                                                                               : true;
    }
  }
}

// Reduce pass runs on 1 block to allow thread sync
__global__ void FillSelBoxes(const int num, bool *sel_boxes, bool *row_mask) {
  // loop over every box in order of high to low confidence score
  for (int i = 0; i < num - 1; ++i) {
    if (!sel_boxes[i]) {
      continue;
    }
    // every thread handles a different set of boxes (per all boxes in order)
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < num; j += blockDim.x * gridDim.x) {
      sel_boxes[j] = sel_boxes[j] && row_mask[i * num + j];
    }
    __syncthreads();  // sync all threads before moving all active threads to
                      // next iteration
  }
}

// MaskToIndex
__global__ void MaskToIndex(const int inputsize, int *maxoutput_size, int *valid_score_num, int *index_buff,
                            int *sel_idx, bool *sel_boxes) {
  int keep_num = 0;
  // loop over every box in order of high to low confidence score
  for (int i = 0; i < inputsize; ++i) {
    if (sel_boxes[i]) {
      sel_idx[keep_num] = index_buff[inputsize - i - 1];
      if (++keep_num == maxoutput_size[0]) break;
    }
  }
  atomicAdd(valid_score_num, keep_num);
}

template <typename T>
__global__ void CountValidNum(const int inner, T *score_buff, T *up_score_buff, int *index_buff, int *valid_score_num,
                              T *score_threshold) {
  for (int i = 0; i < inner; i++) {
    if (score_buff[i] >= score_threshold[0]) {
      up_score_buff[valid_score_num[0]] = score_buff[i];
      index_buff[valid_score_num[0]] = i;
      atomicAdd(valid_score_num, 1);
    }
  }
  int ceil_p_2 = NumRoundUpPower2(valid_score_num[0]);
  for (int i = valid_score_num[0]; i < ceil_p_2; i++) {
    up_score_buff[i] = 1 << 8;
    index_buff[i] = i;
  }
}

// Sorting function based on BitonicSort from TopK kernel
template <typename T>
__global__ void NmsSortByKeyKernel(const int ceil_power2, T *up_score_buff, int *index_buff) {
  for (size_t i = 2; i <= ceil_power2; i <<= 1) {
    for (size_t j = (i >> 1); j > 0; j >>= 1) {
      for (size_t tid = threadIdx.x; tid < ceil_power2; tid += blockDim.x) {
        size_t tid_comp = tid ^ j;
        if (tid_comp > tid) {
          if ((tid & i) == 0) {
            if (up_score_buff[tid] > up_score_buff[tid_comp]) {
              Swap(&up_score_buff[tid], &up_score_buff[tid_comp]);
              Swap(&index_buff[tid], &index_buff[tid_comp]);
            }
          } else {
            if (up_score_buff[tid] < up_score_buff[tid_comp]) {
              Swap(&up_score_buff[tid], &up_score_buff[tid_comp]);
              Swap(&index_buff[tid], &index_buff[tid_comp]);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

void CalPreprocess(const int num, bool *sel_boxes, bool *row_mask, const uint32_t &device_id,
                   cudaStream_t cuda_stream) {
  int total_val = num * num;
  RowMaskInit<<<CUDA_BLOCKS(device_id, total_val), CUDA_THREADS(device_id), 0, cuda_stream>>>(total_val, row_mask);
  NmsPreProcess<<<CUDA_BLOCKS(device_id, num), CUDA_THREADS(device_id), 0, cuda_stream>>>(num, sel_boxes);
}

template <typename T>
int CalSort(const int &num, int *index_buff, T *score_buff, T *up_score_buff, int *valid_score_num, T *score_threshold,
            const uint32_t &device_id, cudaStream_t stream) {
  cudaMemset(valid_score_num, 0, sizeof(int));
  CountValidNum<<<1, 1, 0, stream>>>(num, score_buff, up_score_buff, index_buff, valid_score_num, score_threshold);
  int valid_num_host = 0;
  cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
  cudaMemcpy(&valid_num_host, valid_score_num, sizeof(int), cudaMemcpyDeviceToHost);
  int ceil_p_2 = NumRoundUpPower2(valid_num_host);
  int thread = std::min(ceil_p_2, CUDA_THREADS(device_id));
  NmsSortByKeyKernel<<<1, thread, 0, stream>>>(ceil_p_2, up_score_buff, index_buff);
  return valid_num_host;
}

template <typename T>
void CalNms(const int num, const int total_num, const T *IOU_value, T *overlaps, bool *sel_boxes, bool *row_mask,
            int *index_buff, const uint32_t &device_id, cudaStream_t cuda_stream) {
  // run kernel for every position in row_mask array = (num * num) size
  int row_mask_size = num * num;
  NmsWithOverlapPass<<<CUDA_BLOCKS(device_id, row_mask_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      num, total_num, IOU_value, overlaps, row_mask, index_buff);
  FillSelBoxes<<<1, CUDA_THREADS(device_id), 0, cuda_stream>>>(num, sel_boxes, row_mask);
}

template <typename T>
int CalPostprocess(const int inputsize, int *maxoutput_size, int *valid_score_num, T *score_threshold, int *index_buff,
                   int *sel_idx, bool *sel_boxes, const uint32_t &device_id, cudaStream_t cuda_stream) {
  cudaMemset(valid_score_num, 0, sizeof(int));
  MaskToIndex<<<1, 1, 0, cuda_stream>>>(inputsize, maxoutput_size, valid_score_num, index_buff, sel_idx, sel_boxes);
  int num_output_host = 0;
  cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));
  cudaMemcpy(&num_output_host, valid_score_num, sizeof(int), cudaMemcpyDeviceToHost);
  return num_output_host;
}

template CUDA_LIB_EXPORT int CalSort<half>(const int &inner, int *index_buff, half *score_buff, half *up_score_buff,
                                           int *valid_score_num, half *score_threshold, const uint32_t &device_id,
                                           cudaStream_t stream);

template CUDA_LIB_EXPORT void CalNms<half>(const int num, const int total_num, const half *IOU_value, half *overlaps,
                                           bool *sel_boxes, bool *row_mask, int *index_buff, const uint32_t &device_id,
                                           cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT int CalPostprocess<half>(const int inputsize, int *maxoutput_size, int *valid_score_num,
                                                  half *score_threshold, int *index_buff, int *sel_idx, bool *sel_boxes,
                                                  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT int CalSort<float>(const int &inner, int *index_buff, float *score_buff, float *up_score_buff,
                                            int *valid_score_num, float *score_threshold, const uint32_t &device_id,
                                            cudaStream_t stream);

template CUDA_LIB_EXPORT void CalNms<float>(const int num, const int total_num, const float *IOU_value, float *overlaps,
                                            bool *sel_boxes, bool *row_mask, int *index_buff, const uint32_t &device_id,
                                            cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT int CalPostprocess<float>(const int inputsize, int *maxoutput_size, int *valid_score_num,
                                                   float *score_threshold, int *index_buff, int *sel_idx,
                                                   bool *sel_boxes, const uint32_t &device_id,
                                                   cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT int CalSort<double>(const int &inner, int *index_buff, double *score_buff,
                                             double *up_score_buff, int *valid_score_num, double *score_threshold,
                                             const uint32_t &device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CalNms<double>(const int num, const int total_num, const double *IOU_value,
                                             double *overlaps, bool *sel_boxes, bool *row_mask, int *index_buff,
                                             const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT int CalPostprocess<double>(const int inputsize, int *maxoutput_size, int *valid_score_num,
                                                    double *score_threshold, int *index_buff, int *sel_idx,
                                                    bool *sel_boxes, const uint32_t &device_id,
                                                    cudaStream_t cuda_stream);
