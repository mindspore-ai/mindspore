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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/non_max_suppressionv3_impl.cuh"
#include <cub/cub.cuh>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <vector>
#include <limits>
#include <iostream>
#include <algorithm>

constexpr int kNmsBlockDim = 16;
constexpr int kNmsBlockDimMax = 128;
constexpr int kNmsBoxesPerThread = 8 * sizeof(int);

template <typename T>
struct GreaterThanCubOp {
  float threshold_;
  __host__ __device__ __forceinline__ GreaterThanCubOp(float threshold) : threshold_(threshold) {}
  __host__ __device__ __forceinline__ bool operator()(const T &val) const {
    return (static_cast<float>(val) > threshold_);
  }
};

template <typename T>
__inline__ __device__ void Swap(T *lhs, T *rhs) {
  T tmp = lhs[0];
  lhs[0] = rhs[0];
  rhs[0] = tmp;
}

template <typename T>
__inline__ __device__ T max(T x, T y) {
  if (x > y) {
    return x;
  } else {
    return y;
  }
}

template <typename T>
__inline__ __device__ T min(T x, T y) {
  if (x < y) {
    return x;
  } else {
    return y;
  }
}

template <typename T>
__inline__ __device__ void Flipped(T *box) {
  if (box[0] > box[2]) Swap(&box[0], &box[2]);
  if (box[1] > box[3]) Swap(&box[1], &box[3]);
}

template <typename T>
__inline__ __device__ bool IouDecision(T *box_A, T *box_B, T a_area, float IOU_threshold) {
  T b_area = (box_B[2] - box_B[0]) * (box_B[3] - box_B[1]);
  if (a_area == static_cast<T>(0.0) || b_area == static_cast<T>(0.0)) return false;
  T x_1 = max(box_A[0], box_B[0]);
  T y_1 = max(box_A[1], box_B[1]);
  T x_2 = min(box_A[2], box_B[2]);
  T y_2 = min(box_A[3], box_B[3]);
  T width = max(x_2 - x_1, T(0));  // in case of no overlap
  T height = max(y_2 - y_1, T(0));
  T intersection = width * height;

  float aa = static_cast<float>(intersection);
  T bb = a_area + b_area - intersection;
  float bt = static_cast<float>(bb) * IOU_threshold;

  return aa > bt;
}

template <typename T>
__inline__ __device__ void SelectHelper(int i_selected, int i_original, T *original, T *selected) {
  selected[i_selected * 4 + 0] = original[i_original * 4 + 0];
  selected[i_selected * 4 + 1] = original[i_original * 4 + 1];
  selected[i_selected * 4 + 2] = original[i_original * 4 + 2];
  selected[i_selected * 4 + 3] = original[i_original * 4 + 3];
  Flipped(selected + i_selected * 4);
}

template <typename T>
__global__ void IndexMultiSelect(const int num_elements, int *index_buff, T *original, T *selected) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements; idx += blockDim.x * gridDim.x) {
    SelectHelper(idx, static_cast<int>(index_buff[idx]), original, selected);
  }
}

template <typename T>
__global__ void CastFloat(const int num_elements, T *scores, float *scores_float) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements; idx += blockDim.x * gridDim.x) {
    scores_float[idx] = static_cast<float>(scores[idx]);
  }
}

__global__ void SetZeros(const int num_elements, unsigned int *target) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements; idx += blockDim.x * gridDim.x) {
    target[idx] = 0;
  }
}

template <typename T>
bool CheckBitHost(T bit_mask, int bit) {
  return (bit_mask >> (bit % kNmsBoxesPerThread)) & 1;
}

template <typename T>
__launch_bounds__(kNmsBlockDim *kNmsBlockDim, 4) __global__
  void NMSReduce(const int num, int u_num, float iou_threshold, T *boxes_sort, int box_size, unsigned int *sel_mask) {
  __shared__ T shared_i_boxes[kNmsBlockDim * 4];
  // Same thing with areas
  __shared__ T shared_i_areas[kNmsBlockDim];
  // The condition of the for loop is common to all threads in the block.
  // This is necessary to be able to call __syncthreads() inside of the loop.
  for (int i_block_offset = blockIdx.x * blockDim.x; i_block_offset < num; i_block_offset += blockDim.x * gridDim.x) {
    const int i = i_block_offset + threadIdx.x;
    if (i < num) {
      // One 1D line load the boxes for x-dimension.
      if (threadIdx.y == 0) {
        shared_i_boxes[threadIdx.x * 4 + 0] = boxes_sort[i * 4 + 0];
        shared_i_boxes[threadIdx.x * 4 + 1] = boxes_sort[i * 4 + 1];
        shared_i_boxes[threadIdx.x * 4 + 2] = boxes_sort[i * 4 + 2];
        shared_i_boxes[threadIdx.x * 4 + 3] = boxes_sort[i * 4 + 3];
        T area = (boxes_sort[i * 4 + 2] - boxes_sort[i * 4 + 0]) * (boxes_sort[i * 4 + 3] - boxes_sort[i * 4 + 1]);
        shared_i_areas[threadIdx.x] = area;
      }
    }
    __syncthreads();
    for (int j_thread_offset = kNmsBoxesPerThread * (blockIdx.y * blockDim.y + threadIdx.y); j_thread_offset < num;
         j_thread_offset += kNmsBoxesPerThread * blockDim.y * gridDim.y) {
      int above_threshold = 0;
      // Make sure that threads are within valid domain.
      bool valid = false;
      // Loop over the next kNmsBoxesPerThread boxes and set corresponding bit
      // if it is overlapping with current box
      for (int ib = 0; ib < kNmsBoxesPerThread; ++ib) {
        const int j = j_thread_offset + ib;
        if (i >= j || i >= num || j >= num) continue;
        valid = true;
        T *j_box = boxes_sort + j * 4;
        T *i_box = shared_i_boxes + threadIdx.x * 4;
        if (IouDecision(i_box, j_box, shared_i_areas[threadIdx.x], iou_threshold)) {
          // we have score[j] <= score[i]. j > i
          above_threshold |= (1U << ib);
        }
      }
      if (valid) {
        sel_mask[i * u_num + j_thread_offset / kNmsBoxesPerThread] = above_threshold;
      }
    }
    __syncthreads();  // making sure everyone is done reading shared memory.
  }
}

template <typename T>
int CalNms(const int num_input, int *num_keep, float iou_threshold, int max_output_size, T *boxes_sort, int *index_buff,
           int box_size, unsigned int *sel_mask, bool *sel_boxes, int *output_ptr, const uint32_t &device_id,
           cudaStream_t cuda_stream) {
  int u_num = (num_input + kNmsBoxesPerThread - 1) / kNmsBoxesPerThread;
  const int max_nms_mask_size = num_input * u_num;
  int thread_num = 256 < num_input ? 256 : num_input;
  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, device_id);
  int max_blocks = prop.multiProcessorCount;
  int block_num = min(static_cast<int>(((num_input - 1) / thread_num) + 1), max_blocks);
  SetZeros<<<block_num, thread_num, 0, cuda_stream>>>(max_nms_mask_size, sel_mask);
  int num_blocks = (num_input + kNmsBlockDim - 1) / kNmsBlockDim;
  num_blocks = std::max(std::min(num_blocks, kNmsBlockDimMax), 1);
  dim3 blocks(num_blocks, num_blocks);
  dim3 threads(kNmsBlockDim, kNmsBlockDim);
  NMSReduce<<<blocks, threads, 0, cuda_stream>>>(num_input, u_num, iou_threshold, boxes_sort, box_size, sel_mask);

  std::vector<unsigned int> sel_mask_host(num_input * u_num);
  cudaMemcpyAsync(sel_mask_host.data(), sel_mask, num_input * u_num * sizeof(unsigned int), cudaMemcpyDeviceToHost,
                  cuda_stream);
  std::vector<int> local(u_num);
  std::vector<char> sel_boxes_host(num_input);
  for (int box = 0; box < u_num; box += 1) {
    local[box] = 0xFFFFFFFF;
  }
  int accepted_boxes = 0;
  for (int box = 0; box < num_input - 1; ++box) {
    if (!CheckBitHost(local[box / kNmsBoxesPerThread], box)) {
      continue;
    }
    accepted_boxes += 1;
    int offset = box * u_num;

    for (int b = 0; b < u_num; b += 1) {
      local[b] &= ~sel_mask_host[offset + b];
    }
    if (accepted_boxes > max_output_size) break;
  }
  for (int box = 0; box < num_input; box += 1) {
    sel_boxes_host[box] = CheckBitHost(local[box / kNmsBoxesPerThread], box);
  }
  cudaMemcpyAsync(sel_boxes, sel_boxes_host.data(), num_input * sizeof(char), cudaMemcpyHostToDevice, cuda_stream);

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  (void)cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, static_cast<int *>(nullptr),
                                   static_cast<char *>(nullptr), static_cast<int *>(nullptr),
                                   static_cast<int *>(nullptr), num_input, cuda_stream);
  (void)cudaMalloc(&d_temp_storage, temp_storage_bytes);
  (void)cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, index_buff, sel_boxes, output_ptr, num_keep,
                                   num_input, cuda_stream);
  (void)cudaFree(d_temp_storage);

  int num_count = 0;
  cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));
  cudaMemcpyAsync(&num_count, num_keep, sizeof(int), cudaMemcpyDeviceToHost, cuda_stream);
  num_count = max_output_size < num_count ? max_output_size : num_count;
  return num_count;
}

template <typename T, typename M, typename S>
cudaError_t DoNms(const int num_input, int *count, int *num_keep, T *scores, T *boxes_in, M iou_threshold_,
                  M score_threshold_, int *index_buff, S max_output_size_, int box_size, unsigned int *sel_mask,
                  bool *sel_boxes, int *output_ptr, const uint32_t &device_id, cudaStream_t cuda_stream,
                  int *output_size) {
  float iou_threshold = static_cast<float>(iou_threshold_);
  float score_threshold = static_cast<float>(score_threshold_);
  int max_output_size = static_cast<int>(max_output_size_);
  cudaMemset(count, 0, sizeof(int));

  float *scores_float = nullptr;
  size_t scores_float_temp_storage_bytes = num_input * sizeof(float);
  (void)cudaMalloc(&scores_float, scores_float_temp_storage_bytes);
  int thread_num = 256 < num_input ? 256 : num_input;
  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, device_id);
  int max_blocks = prop.multiProcessorCount;
  int block_num = std::min(static_cast<int>(((num_input - 1) / thread_num) + 1), max_blocks);
  CastFloat<<<block_num, thread_num, 0, cuda_stream>>>(num_input, scores, scores_float);

  auto policy = thrust::cuda::par.on(cuda_stream);
  thrust::device_ptr<int> dev_ptr(index_buff);
  thrust::sequence(policy, dev_ptr, dev_ptr + num_input);
  cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));
  size_t cub_sort_temp_storage_bytes = 0;
  (void)cub::DeviceRadixSort::SortPairsDescending(nullptr, cub_sort_temp_storage_bytes,
                                                  static_cast<float *>(nullptr),  // scores
                                                  static_cast<float *>(nullptr),  // sorted scores
                                                  static_cast<int *>(nullptr),    // input indices
                                                  static_cast<int *>(nullptr),    // sorted indices
                                                  num_input,                      // num items
                                                  0, 8 * sizeof(float),           // sort all bits
                                                  cuda_stream);
  float *scores_sorted = nullptr;
  size_t scores_sorted_temp_storage_bytes = num_input * sizeof(float);
  (void)cudaMalloc(&scores_sorted, scores_sorted_temp_storage_bytes);
  int *index_sorted = nullptr;
  size_t index_sorted_temp_storage_bytes = num_input * sizeof(int);
  (void)cudaMalloc(&index_sorted, index_sorted_temp_storage_bytes);
  void *sort_temp_buff = nullptr;
  (void)cudaMalloc(&sort_temp_buff, cub_sort_temp_storage_bytes);
  (void)cub::DeviceRadixSort::SortPairsDescending(sort_temp_buff, cub_sort_temp_storage_bytes, scores_float,
                                                  scores_sorted, index_buff, index_sorted, num_input, 0,
                                                  8 * sizeof(float), cuda_stream);
  cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));

  (void)cudaFree(sort_temp_buff);
  GreaterThanCubOp<T> score_limit(score_threshold);
  void *s_temp_storage = nullptr;
  size_t s_temp_storage_bytes = 0;
  (void)cub::DeviceSelect::If(nullptr, s_temp_storage_bytes, static_cast<float *>(nullptr),
                              static_cast<float *>(nullptr), static_cast<int *>(nullptr), num_input, score_limit,
                              cuda_stream);
  (void)cudaMalloc(&s_temp_storage, s_temp_storage_bytes);
  (void)cub::DeviceSelect::If(s_temp_storage, s_temp_storage_bytes, scores_sorted, scores_float, count, num_input,
                              score_limit, cuda_stream);
  (void)cudaFree(s_temp_storage);
  (void)cudaFree(scores_float);
  (void)cudaFree(scores_sorted);
  T *boxes_sort = nullptr;
  size_t boxes_temp_storage_bytes = num_input * box_size * sizeof(T);
  (void)cudaMalloc(&boxes_sort, boxes_temp_storage_bytes);

  IndexMultiSelect<<<block_num, thread_num, 0, cuda_stream>>>(num_input, index_sorted, boxes_in, boxes_sort);
  cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));

  int num_count = 0;
  cudaMemcpyAsync(&num_count, count, sizeof(int), cudaMemcpyDeviceToHost, cuda_stream);
  const int num_to_keep = num_count;
  if (num_to_keep <= 0) {
    return cudaErrorNotReady;
  }
  *output_size = CalNms(num_to_keep, num_keep, iou_threshold, max_output_size, boxes_sort, index_sorted, box_size,
                        sel_mask, sel_boxes, output_ptr, device_id, reinterpret_cast<cudaStream_t>(cuda_stream));
  (void)cudaFree(boxes_sort);
  (void)cudaFree(index_sorted);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t DoNms<float, float, int>(
  const int num_input, int *count, int *num_keep, float *scores, float *boxes_in, float iou_threshold_,
  float score_threshold_, int *index_buff, int max_output_size_, int box_size, unsigned int *sel_mask, bool *sel_boxes,
  int *output_ptr, const uint32_t &device_id, cudaStream_t cuda_stream, int *output_size);
template CUDA_LIB_EXPORT cudaError_t DoNms<float, float, int64_t>(
  const int num_input, int *count, int *num_keep, float *scores, float *boxes_in, float iou_threshold_,
  float score_threshold_, int *index_buff, int64_t max_output_size_, int box_size, unsigned int *sel_mask,
  bool *sel_boxes, int *output_ptr, const uint32_t &device_id, cudaStream_t cuda_stream, int *output_size);
template CUDA_LIB_EXPORT cudaError_t DoNms<half, float, int>(
  const int num_input, int *count, int *num_keep, half *scores, half *boxes_in, float iou_threshold_,
  float score_threshold_, int *index_buff, int max_output_size_, int box_size, unsigned int *sel_mask, bool *sel_boxes,
  int *output_ptr, const uint32_t &device_id, cudaStream_t cuda_stream, int *output_size);
template CUDA_LIB_EXPORT cudaError_t DoNms<half, float, int64_t>(
  const int num_input, int *count, int *num_keep, half *scores, half *boxes_in, float iou_threshold_,
  float score_threshold_, int *index_buff, int64_t max_output_size_, int box_size, unsigned int *sel_mask,
  bool *sel_boxes, int *output_ptr, const uint32_t &device_id, cudaStream_t cuda_stream, int *output_size);
template CUDA_LIB_EXPORT cudaError_t DoNms<float, half, int>(
  const int num_input, int *count, int *num_keep, float *scores, float *boxes_in, half iou_threshold_,
  half score_threshold_, int *index_buff, int max_output_size_, int box_size, unsigned int *sel_mask, bool *sel_boxes,
  int *output_ptr, const uint32_t &device_id, cudaStream_t cuda_stream, int *output_size);
template CUDA_LIB_EXPORT cudaError_t DoNms<float, half, int64_t>(
  const int num_input, int *count, int *num_keep, float *scores, float *boxes_in, half iou_threshold_,
  half score_threshold_, int *index_buff, int64_t max_output_size_, int box_size, unsigned int *sel_mask,
  bool *sel_boxes, int *output_ptr, const uint32_t &device_id, cudaStream_t cuda_stream, int *output_size);
template CUDA_LIB_EXPORT cudaError_t DoNms<half, half, int>(const int num_input, int *count, int *num_keep,
                                                            half *scores, half *boxes_in, half iou_threshold_,
                                                            half score_threshold_, int *index_buff,
                                                            int max_output_size_, int box_size, unsigned int *sel_mask,
                                                            bool *sel_boxes, int *output_ptr, const uint32_t &device_id,
                                                            cudaStream_t cuda_stream, int *output_size);
template CUDA_LIB_EXPORT cudaError_t DoNms<half, half, int64_t>(
  const int num_input, int *count, int *num_keep, half *scores, half *boxes_in, half iou_threshold_,
  half score_threshold_, int *index_buff, int64_t max_output_size_, int box_size, unsigned int *sel_mask,
  bool *sel_boxes, int *output_ptr, const uint32_t &device_id, cudaStream_t cuda_stream, int *output_size);
