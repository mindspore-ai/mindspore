/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless req_uired by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <vector>
#include <limits>
#include <iostream>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/combined_non_max_suppression_impl.cuh"

constexpr int DIM0 = 0;
constexpr int DIM1 = 1;
constexpr int DIM2 = 2;
constexpr int DIM3 = 3;
constexpr int DIM4 = 4;
constexpr float zero = 0;
constexpr float one = 1;

__inline__ __device__ float IOU(float *boxes_result, int i, int j) {
  float lx, ly, rx, ry;
  float w, h;
  float area;
  float area_a = (boxes_result[i * DIM4 + DIM2] - boxes_result[i * DIM4 + DIM0]) *
                 (boxes_result[i * DIM4 + DIM3] - boxes_result[i * DIM4 + DIM1]);
  float area_b = (boxes_result[j * DIM4 + DIM2] - boxes_result[j * DIM4 + DIM0]) *
                 (boxes_result[j * DIM4 + DIM3] - boxes_result[j * DIM4 + DIM1]);
  if ((area_a == zero) || (area_b == zero)) {
    return zero;
  }
  lx = max(boxes_result[i * DIM4 + DIM0], boxes_result[j * DIM4 + DIM0]);
  ly = max(boxes_result[i * DIM4 + DIM1], boxes_result[j * DIM4 + DIM1]);
  rx = min(boxes_result[i * DIM4 + DIM2], boxes_result[j * DIM4 + DIM2]);
  ry = min(boxes_result[i * DIM4 + DIM3], boxes_result[j * DIM4 + DIM3]);
  w = (rx > lx) ? (rx - lx) : zero;
  h = (ry > ly) ? (ry - ly) : zero;
  area = w * h;
  return area / (area_a + area_b - area);
}

template <typename T>
__global__ void permute(int q, int num_boxes, int batch_size, T *boxes, float *new_boxes) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < q * num_boxes * batch_size * DIM4;
       index += blockDim.x * gridDim.x) {
    int i = index % DIM4;
    int c = (index / DIM4) % q;
    int d = (index / DIM4 / q) % num_boxes;
    int n = index / DIM4 / q / num_boxes;
    int new_index = ((n * q + c) * num_boxes + d) * DIM4 + i;
    float result = boxes[index];
    new_boxes[new_index] = result;
  }
}

__global__ void boxsort(float *boxes_result, float *new_boxes, int batch_size, int q, int num_boxes) {
  for (int box_num = blockIdx.x * blockDim.x + threadIdx.x; box_num < batch_size * q * num_boxes;
       box_num += blockDim.x * gridDim.x) {
    boxes_result[box_num * DIM4 + DIM0] = min(new_boxes[box_num * DIM4 + DIM0], new_boxes[box_num * DIM4 + DIM2]);
    boxes_result[box_num * DIM4 + DIM2] = max(new_boxes[box_num * DIM4 + DIM0], new_boxes[box_num * DIM4 + DIM2]);
    boxes_result[box_num * DIM4 + DIM1] = min(new_boxes[box_num * DIM4 + DIM1], new_boxes[box_num * DIM4 + DIM3]);
    boxes_result[box_num * DIM4 + DIM3] = max(new_boxes[box_num * DIM4 + DIM1], new_boxes[box_num * DIM4 + DIM3]);
  }
}

template <typename T>
__global__ void presort(int num_classes, int num_boxes, int batch_size, T *scores, float *new_scores, int *index,
                        T *score_threshold, bool *sel) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_classes * num_boxes * batch_size;
       idx += blockDim.x * gridDim.x) {
    int c = idx % num_classes;
    int d = (idx / num_classes) % num_boxes;
    int n = idx / (num_classes * num_boxes);
    int new_index = (n * num_classes + c) * num_boxes + d;
    float result = scores[idx];
    new_scores[new_index] = result;
    index[new_index] = new_index;
    sel[new_index] = (new_scores[new_index] > score_threshold[DIM0]) ? true : false;
  }
}

__global__ void Init(int num_classes, int num_boxes, int batch_size, bool *mask) {
  for (int mat_pos = blockIdx.x * blockDim.x + threadIdx.x; mat_pos < batch_size * num_classes * num_boxes * num_boxes;
       mat_pos += blockDim.x * gridDim.x) {
    mask[mat_pos] = true;
  }
}

template <typename T>
__global__ void nms(int batch_size, int num_classes, T *iou_threshold, bool *sel, float *boxes_result, int *index,
                    int q, int num_boxes, bool *mask) {
  int box_i, box_j;
  for (int mask_index = blockIdx.x * blockDim.x + threadIdx.x;
       mask_index < batch_size * num_classes * num_boxes * num_boxes; mask_index += blockDim.x * gridDim.x) {
    box_i = mask_index / num_boxes;                                                     // row in 2d sel_mask array
    box_j = mask_index / (num_boxes * num_boxes) * num_boxes + mask_index % num_boxes;  // col in 2d sel_mask array
    if (box_j > box_i) {
      int idi = index[box_i];
      int idj = index[box_j];  // skip when box_j index lower/equal to box_i - will remain true
      if (q == num_classes) {
        if (IOU(boxes_result, idi, idj) > iou_threshold[0]) {
          mask[mask_index] = false;
        }
      } else {
        if (IOU(boxes_result, idi / (num_classes * num_boxes) * num_boxes + idi % num_boxes,
                idj / (num_classes * num_boxes) * num_boxes + idj % num_boxes) > iou_threshold[0]) {
          mask[mask_index] = false;
        }
      }
    }
  }
}

__global__ void nmsReducePass(int batch_size, int num_classes, bool *sel, int *index, int num_boxes, bool *mask) {
  for (int page = DIM0; page < batch_size * num_classes; page++) {
    for (int i = DIM0; i < num_boxes - DIM1; ++i) {
      int idxi = index[page * num_boxes + i];
      if (!sel[idxi]) {
        continue;
      }
      for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < num_boxes; j += blockDim.x * gridDim.x) {
        int idxj = index[page * num_boxes + j];

        sel[idxj] = sel[idxj] && mask[page * num_boxes * num_boxes + i * num_boxes + j];
      }
    }
  }
}

__global__ void sizeperclass(int batch_size, int num_classes, bool *sel, int num_boxes, int *index,
                             int *max_output_size_per_class) {
  for (int page = blockIdx.x * blockDim.x + threadIdx.x; page < batch_size * num_classes;
       page += blockDim.x * gridDim.x) {
    int class_idx_count = DIM0;
    for (int i = page * num_boxes; i < (page + DIM1) * num_boxes; i++) {
      int number = index[i];
      if (sel[number]) {
        class_idx_count++;
        if (class_idx_count > max_output_size_per_class[DIM0]) {
          sel[number] = false;
        }
      }
    }
  }
}

template <typename T>
__global__ void output(int batch_size, int per_detections, int *index, float *new_scores, bool *sel, float *new_boxes,
                       T *nmsed_classes, T *nmsed_scores, T *nmsed_boxes, int *valid_detections, bool clip_boxes,
                       int num_classes, int num_boxes, int q) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch_size; i += gridDim.x * blockDim.x) {
    int num = DIM0;
    for (int j = i * num_classes * num_boxes; (j < (i + DIM1) * num_classes * num_boxes) && (num < per_detections);
         j++) {
      int idx = index[j];
      float score = new_scores[j];
      if (sel[idx]) {
        int bboxOffset = i * (q * num_boxes);
        int bboxId = (idx % (q * num_boxes) + bboxOffset);
        nmsed_classes[i * per_detections + num] = T((idx % (num_classes * num_boxes)) / num_boxes);
        nmsed_scores[i * per_detections + num] = T(score);
        float xMin = new_boxes[bboxId * DIM4];
        float yMin = new_boxes[bboxId * DIM4 + DIM1];
        float xMax = new_boxes[bboxId * DIM4 + DIM2];
        float yMax = new_boxes[bboxId * DIM4 + DIM3];
        nmsed_boxes[(i * per_detections + num) * DIM4] = T(clip_boxes ? max(min(xMin, one), zero) : xMin);
        nmsed_boxes[(i * per_detections + num) * DIM4 + DIM1] = T(clip_boxes ? max(min(yMin, one), zero) : yMin);
        nmsed_boxes[(i * per_detections + num) * DIM4 + DIM2] = T(clip_boxes ? max(min(xMax, one), zero) : xMax);
        nmsed_boxes[(i * per_detections + num) * DIM4 + DIM3] = T(clip_boxes ? max(min(yMax, one), zero) : yMax);
        num++;
      }
    }
    valid_detections[i] = num;
    while (num < per_detections) {
      nmsed_classes[i * per_detections + num] = T(zero);
      nmsed_scores[i * per_detections + num] = T(zero);
      nmsed_boxes[(i * per_detections + num) * DIM4] = T(zero);
      nmsed_boxes[(i * per_detections + num) * DIM4 + DIM1] = T(zero);
      nmsed_boxes[(i * per_detections + num) * DIM4 + DIM2] = T(zero);
      nmsed_boxes[(i * per_detections + num) * DIM4 + DIM3] = T(zero);
      num++;
    }
  }
}

template <typename T>
cudaError_t CalSort(T *scores, int *index, T *score_threshold, int num_classes, T *boxes, float *new_boxes,
                    float *new_scores, int batch_size, int num_boxes, float *boxes_result, int q, bool *sel,
                    const uint32_t &device_id, cudaStream_t cuda_stream) {
  permute<<<CUDA_BLOCKS(device_id, q * num_boxes * batch_size * DIM4), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    q, num_boxes, batch_size, boxes, new_boxes);
  boxsort<<<CUDA_BLOCKS(device_id, batch_size * q * num_boxes), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    boxes_result, new_boxes, batch_size, q, num_boxes);
  presort<<<CUDA_BLOCKS(device_id, num_classes * num_boxes * batch_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    num_classes, num_boxes, batch_size, scores, new_scores, index, score_threshold, sel);
  auto policy = thrust::cuda::par.on(cuda_stream);
  for (int i = DIM0; i < num_classes * batch_size; i++) {
    thrust::stable_sort_by_key(policy, thrust::device_pointer_cast(new_scores + i * num_boxes),
                               thrust::device_pointer_cast(new_scores + i * num_boxes) + num_boxes,
                               thrust::device_pointer_cast(index + i * num_boxes), thrust::greater<float>());
  }
  return GetCudaStatus();
}

template <typename T>
cudaError_t Calnms(int batch_size, int num_classes, T *iou_threshold, bool *sel, float *boxes_result, int *index, int q,
                   int num_boxes, int *max_output_size_per_class, float *new_scores, bool *mask,
                   const uint32_t &device_id, cudaStream_t cuda_stream) {
  Init<<<CUDA_BLOCKS(device_id, batch_size * num_classes * num_boxes * num_boxes), CUDA_THREADS(device_id), 0,
         cuda_stream>>>(num_classes, num_boxes, batch_size, mask);
  nms<<<CUDA_BLOCKS(device_id, batch_size * num_classes * num_boxes * num_boxes), CUDA_THREADS(device_id), 0,
        cuda_stream>>>(batch_size, num_classes, iou_threshold, sel, boxes_result, index, q, num_boxes, mask);
  nmsReducePass<<<CUDA_BLOCKS(device_id, num_boxes), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    batch_size, num_classes, sel, index, num_boxes, mask);
  sizeperclass<<<CUDA_BLOCKS(device_id, batch_size * num_classes), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    batch_size, num_classes, sel, num_boxes, index, max_output_size_per_class);

  auto policy = thrust::cuda::par.on(cuda_stream);
  for (int i = DIM0; i < batch_size; i++) {
    thrust::stable_sort_by_key(
      policy, thrust::device_pointer_cast(new_scores + i * num_boxes * num_classes),
      thrust::device_pointer_cast(new_scores + i * num_boxes * num_classes) + (num_boxes * num_classes),
      thrust::device_pointer_cast(index + i * num_boxes * num_classes), thrust::greater<float>());
  }
  return GetCudaStatus();
}

template <typename T>
cudaError_t Caloutput(int batch_size, int per_detections, int *index, float *new_scores, bool *sel, float *new_boxes,
                      T *nmsed_classes, T *nmsed_scores, T *nmsed_boxes, int *valid_detections, bool clip_boxes,
                      int num_classes, int num_boxes, int q, const uint32_t &device_id, cudaStream_t cuda_stream) {
  output<<<CUDA_BLOCKS(device_id, batch_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    batch_size, per_detections, index, new_scores, sel, new_boxes, nmsed_classes, nmsed_scores, nmsed_boxes,
    valid_detections, clip_boxes, num_classes, num_boxes, q);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalSort<float>(float *scores, int *index, float *score_threshold, int num_classes,
                                                    float *boxes, float *new_boxes, float *new_scores, int batch_size,
                                                    int num_boxes, float *boxes_result, int q, bool *sel,
                                                    const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t Calnms<float>(int batch_size, int num_classes, float *iou_threshold, bool *sel,
                                                   float *boxes_result, int *index, int q, int num_boxes,
                                                   int *max_output_size_per_class, float *new_scores, bool *mask,
                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t Caloutput<float>(int batch_size, int per_detections, int *index, float *new_scores,
                                                      bool *sel, float *new_boxes, float *nmsed_classes,
                                                      float *nmsed_scores, float *nmsed_boxes, int *valid_detections,
                                                      bool clip_boxes, int num_classes, int num_boxes, int q,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);
