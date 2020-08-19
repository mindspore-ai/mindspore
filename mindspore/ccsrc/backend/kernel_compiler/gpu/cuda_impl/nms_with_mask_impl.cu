/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "nms_with_mask_impl.cuh"
#include <limits>
#include <algorithm>

int NMSRoundUpPower2(int v) {
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

template <typename T>
__global__ void PopulateOutput(T *data_in, T *data_out, int *index_buff, const int num, int box_size_) {
  for (int box_num = blockIdx.x * blockDim.x + threadIdx.x; box_num < num; box_num += blockDim.x * gridDim.x) {
    int correct_index = index_buff[(num - 1) - box_num];  // flip the array around
    for (int x = 0; x < 5; x++) {
      data_out[(box_num * box_size_) + x] = data_in[(correct_index * box_size_) + x];
    }
  }
}

template <typename T>
__inline__ __device__ bool IOUDecision(T *output, int box_A_ix, int box_B_ix, int box_A_start, int box_B_start, T *area,
                                       float IOU_value) {
  T x_1 = max(output[box_A_start + 0], output[box_B_start + 0]);
  T y_1 = max(output[box_A_start + 1], output[box_B_start + 1]);
  T x_2 = min(output[box_A_start + 2], output[box_B_start + 2]);
  T y_2 = min(output[box_A_start + 3], output[box_B_start + 3]);
  T width = max(x_2 - x_1, T(0));  // in case of no overlap
  T height = max(y_2 - y_1, T(0));
  T combined_area = area[box_A_ix] + area[box_B_ix];
  // return decision to keep or remove box
  return !(((width * height) / (combined_area - (width * height))) > IOU_value);
}

template <typename T>
__global__ void Preprocess(const int num, int *sel_idx, T *area, T *output, int box_size_) {
  for (int box_num = blockIdx.x * blockDim.x + threadIdx.x; box_num < num; box_num += blockDim.x * gridDim.x) {
    sel_idx[box_num] = box_num;
    area[box_num] = (output[(box_num * box_size_) + 2] - output[(box_num * box_size_) + 0]) *
                    (output[(box_num * box_size_) + 3] - output[(box_num * box_size_) + 1]);
  }
}

template <typename T>
__global__ void NMSWithMaskKernel(const int num, const float IOU_value, T *output, T *area, bool *sel_boxes,
                                  int box_size_) {
  for (int box_num = blockIdx.x * blockDim.x + threadIdx.x; box_num < num; box_num += blockDim.x * gridDim.x) {
    // represents highest score box in that GPU block
    if (threadIdx.x == 0) {
      sel_boxes[box_num] = true;
      continue;
    }
    int box_start_index = box_num * box_size_;  // start index adjustment
    int block_max_box_num = ((blockIdx.x * blockDim.x) + 0);
    int block_max_box_start_index = block_max_box_num * box_size_;  // start index adjustment
    sel_boxes[box_num] =
      IOUDecision(output, box_num, block_max_box_num, block_max_box_start_index, box_start_index, area,
                  IOU_value);  // update mask
  }
}

template <typename T>
__global__ void FinalPass(const int num, const float IOU_value, T *output, T *area, bool *sel_boxes, int box_size_) {
  int box_i, box_j;                          // access all shared mem meta data with these
  int box_i_start_index, box_j_start_index;  // actual input data indexing
  for (int i = 0; i < num - 1; i++) {
    box_i = i;
    box_i_start_index = box_i * box_size_;  // adjust starting index
    if (sel_boxes[box_i]) {
      for (int j = i + 1; j < num; j++) {
        box_j = j;
        box_j_start_index = box_j * box_size_;
        if (sel_boxes[box_j]) {
          sel_boxes[box_j] = IOUDecision(output, box_i, box_j, box_i_start_index, box_j_start_index, area, IOU_value);
        }
      }
    }
  }
}

template <typename T>
__global__ void NMS_BitonicSortByKeyKernel(const int outer, const int inner, const int ceil_power2, T *input,
                                           T *data_buff, int *index_buff, int box_size_) {
  for (int i = threadIdx.x; i < ceil_power2; i += blockDim.x) {
    data_buff[i] = (i < inner) ? input[(i * box_size_) + 4] : std::numeric_limits<T>::max();
    index_buff[i] = i;
  }
  __syncthreads();

  for (size_t i = 2; i <= ceil_power2; i <<= 1) {
    for (size_t j = (i >> 1); j > 0; j >>= 1) {
      for (size_t tid = threadIdx.x; tid < ceil_power2; tid += blockDim.x) {
        size_t tid_comp = tid ^ j;
        if (tid_comp > tid) {
          if ((tid & i) == 0) {
            if (data_buff[tid] > data_buff[tid_comp]) {
              Swap(&data_buff[tid], &data_buff[tid_comp]);
              Swap(&index_buff[tid], &index_buff[tid_comp]);
            }
          } else {
            if (data_buff[tid] < data_buff[tid_comp]) {
              Swap(&data_buff[tid], &data_buff[tid_comp]);
              Swap(&index_buff[tid], &index_buff[tid_comp]);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

template <typename T>
void CalPreprocess(const int num, int *sel_idx, T *area, T *input, T *output, int *index_buff, int box_size_,
                   cudaStream_t cuda_stream) {
  PopulateOutput<<<GET_BLOCKS(num), GET_THREADS, 0, cuda_stream>>>(input, output, index_buff, num, box_size_);
  Preprocess<<<GET_BLOCKS(num), GET_THREADS, 0, cuda_stream>>>(num, sel_idx, area, output, box_size_);
}

template <typename T>
void CalSortInit(const int &num, T *data_in, T *data_out, int *index_buff, T *data_buff, int box_size_,
                 cudaStream_t stream) {
  int ceil_p_2 = NMSRoundUpPower2(num);
  int thread = std::min(ceil_p_2, GET_THREADS);
  NMS_BitonicSortByKeyKernel<<<1, thread, 0, stream>>>(1, num, ceil_p_2, data_in, data_buff, index_buff, box_size_);
}

template <typename T>
void CalNMSWithMask(const int num, const float IOU_value, T *output, T *area, bool *sel_boxes, int box_size_,
                    cudaStream_t cuda_stream) {
  NMSWithMaskKernel<<<GET_BLOCKS(num), GET_THREADS, 0, cuda_stream>>>(num, IOU_value, output, area, sel_boxes,
                                                                      box_size_);
}

template <typename T>
void CalFinalPass(const int num, const float IOU_value, T *output, T *area, bool *sel_boxes, int box_size_,
                  cudaStream_t cuda_stream) {
  FinalPass<<<1, 1, 0, cuda_stream>>>(num, IOU_value, output, area, sel_boxes, box_size_);
}

template void CalPreprocess<float>(const int num, int *sel_idx, float *area, float *input, float *output,
                                   int *index_buff, int box_size_, cudaStream_t cuda_stream);

template void CalSortInit<float>(const int &inner, float *data_in, float *data_out, int *index_buff, float *data_buff,
                                 int box_size_, cudaStream_t stream);

template void CalNMSWithMask<float>(const int num, const float IOU_value, float *output, float *area, bool *sel_boxes,
                                    int box_size_, cudaStream_t cuda_stream);

template void CalFinalPass<float>(const int num, const float IOU_value, float *output, float *area, bool *sel_boxes,
                                  int box_size_, cudaStream_t cuda_stream);
