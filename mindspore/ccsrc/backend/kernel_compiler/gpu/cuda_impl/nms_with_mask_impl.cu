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

// Initialize per row mask array to all true
__global__ void MaskInit(int numSq, bool *row_mask) {
  for (int mat_pos = blockIdx.x * blockDim.x + threadIdx.x; mat_pos < numSq; mat_pos += blockDim.x * gridDim.x) {
    row_mask[mat_pos] = true;
  }
}

// copy data from input to output array sorted by indices returned from bitonic sort
// flips boxes if asked to,  default - false -> if (x1/y1 > x2/y2)
template <typename T>
__global__ void PopulateOutput(T *data_in, T *data_out, int *index_buff, const int num, int box_size_,
                               bool flip_mode = false) {
  for (int box_num = blockIdx.x * blockDim.x + threadIdx.x; box_num < num; box_num += blockDim.x * gridDim.x) {
    int correct_index = index_buff[(num - 1) - box_num];  // flip the array around
    int correct_arr_start = correct_index * box_size_;
    int current_arr_start = box_num * box_size_;
    if (flip_mode) {  // flip boxes
      // check x
      if (data_in[correct_arr_start + 0] > data_in[correct_arr_start + 2]) {
        data_out[current_arr_start + 0] = data_in[correct_arr_start + 2];
        data_out[current_arr_start + 2] = data_in[correct_arr_start + 0];
      } else {
        data_out[current_arr_start + 0] = data_in[correct_arr_start + 0];
        data_out[current_arr_start + 2] = data_in[correct_arr_start + 2];
      }
      // check y
      if (data_in[correct_arr_start + 1] > data_in[correct_arr_start + 3]) {
        data_out[current_arr_start + 1] = data_in[correct_arr_start + 3];
        data_out[current_arr_start + 3] = data_in[correct_arr_start + 1];
      } else {
        data_out[current_arr_start + 1] = data_in[correct_arr_start + 1];
        data_out[current_arr_start + 3] = data_in[correct_arr_start + 3];
      }
      data_out[current_arr_start + 4] = data_in[correct_arr_start + 4];
    } else {  // default behaviour, don't flip
      for (int x = 0; x < 5; x++) {
        data_out[current_arr_start + x] = data_in[correct_arr_start + x];
      }
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
  return !(((width * height) / (combined_area - (width * height))) >= IOU_value);
}

// calculate areas for boxes -> sorted by output boxes
// populated return mask (init to all true) and return index array
template <typename T>
__global__ void Preprocess(const int num, int *sel_idx, bool *sel_boxes, T *area, T *output, int box_size_) {
  for (int box_num = blockIdx.x * blockDim.x + threadIdx.x; box_num < num; box_num += blockDim.x * gridDim.x) {
    sel_idx[box_num] = box_num;
    sel_boxes[box_num] = true;
    area[box_num] = (output[(box_num * box_size_) + 2] - output[(box_num * box_size_) + 0]) *
                    (output[(box_num * box_size_) + 3] - output[(box_num * box_size_) + 1]);
  }
}

// Run parallel NMS pass
// Every box updates it's own mask in row_mask in sep threads
template <typename T>
__global__ void NMSPass(const int num, const float IOU_value, T *output, T *area, bool *sel_boxes, int box_size_,
                        bool *row_mask) {
  int box_i_start_index, box_j_start_index;  // actual input data indexing
  int mask_offset = 0;
  for (int box_i = blockIdx.x * blockDim.x + threadIdx.x; box_i < num - 1; box_i += blockDim.x * gridDim.x) {
    mask_offset = box_i * num;
    box_i_start_index = box_i * box_size_;  // adjust starting index
    for (int box_j = box_i + 1; box_j < num; box_j++) {
      box_j_start_index = box_j * box_size_;
      row_mask[mask_offset + box_j] =
        IOUDecision(output, box_i, box_j, box_i_start_index, box_j_start_index, area, IOU_value);
    }
  }
}

// Reduce pass runs on 1 block to allow thread sync
__global__ void ReducePass(const int num, bool *sel_boxes, bool *row_mask) {
  // loop over every box in order of high to low confidence score
  for (int i = 0; i < num - 1; ++i) {
    if (!sel_boxes[i]) {
      continue;
    }
    // every thread handles a different set of boxes (per all boxes in order)
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < num; j += blockDim.x * gridDim.x) {
      sel_boxes[j] = sel_boxes[j] && row_mask[i * num + j];
    }
    __syncthreads();  // sync all threads before moving all active threads to next iteration
  }
}

// Sorting function based on BitonicSort from TopK kernel
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
void CalPreprocess(const int num, int *sel_idx, bool *sel_boxes, T *area, T *input, T *output, int *index_buff,
                   int box_size_, bool *row_mask, cudaStream_t cuda_stream) {
  int total_val = num * num;
  MaskInit<<<GET_BLOCKS(total_val), GET_THREADS, 0, cuda_stream>>>(total_val, row_mask);
  // default for flipping boxes -> false (provision available to flip if API updated)
  PopulateOutput<<<GET_BLOCKS(num), GET_THREADS, 0, cuda_stream>>>(input, output, index_buff, num, box_size_, false);
  Preprocess<<<GET_BLOCKS(num), GET_THREADS, 0, cuda_stream>>>(num, sel_idx, sel_boxes, area, output, box_size_);
}

template <typename T>
void CalSort(const int &num, T *data_in, T *data_out, int *index_buff, T *data_buff, int box_size_,
             cudaStream_t stream) {
  int ceil_p_2 = NMSRoundUpPower2(num);
  int thread = std::min(ceil_p_2, GET_THREADS);
  NMS_BitonicSortByKeyKernel<<<1, thread, 0, stream>>>(1, num, ceil_p_2, data_in, data_buff, index_buff, box_size_);
}

template <typename T>
void CalNMS(const int num, const float IOU_value, T *output, T *area, bool *sel_boxes, int box_size_, bool *row_mask,
            cudaStream_t cuda_stream) {
  NMSPass<<<GET_BLOCKS(num), GET_THREADS, 0, cuda_stream>>>(num, IOU_value, output, area, sel_boxes, box_size_,
                                                            row_mask);
  ReducePass<<<1, GET_THREADS, 0, cuda_stream>>>(num, sel_boxes, row_mask);
}

template void CalSort<float>(const int &inner, float *data_in, float *data_out, int *index_buff, float *data_buff,
                             int box_size_, cudaStream_t stream);

template void CalPreprocess<float>(const int num, int *sel_idx, bool *sel_boxes, float *area, float *input,
                                   float *output, int *index_buff, int box_size_, bool *row_mask,
                                   cudaStream_t cuda_stream);

template void CalNMS<float>(const int num, const float IOU_value, float *output, float *area, bool *sel_boxes,
                            int box_size_, bool *row_mask, cudaStream_t cuda_stream);
