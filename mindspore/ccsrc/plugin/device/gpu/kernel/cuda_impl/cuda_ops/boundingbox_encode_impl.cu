/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/boundingbox_encode_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void BoundingBoxEncodeKernel(const size_t size, const T *anchor_box, const T *groundtruth_box, T *deltas,
                                        const float m1, const float m2, const float m3, const float m4, const float s1,
                                        const float s2, const float s3, const float s4) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    const size_t left_x = i * 4;
    const size_t left_y = i * 4 + 1;
    const size_t right_x = i * 4 + 2;
    const size_t right_y = i * 4 + 3;

    T px = (anchor_box[left_x] + anchor_box[right_x]) * 0.5f;
    T py = (anchor_box[left_y] + anchor_box[right_y]) * 0.5f;
    T pw = anchor_box[right_x] - anchor_box[left_x] + 1.0f;
    T ph = anchor_box[right_y] - anchor_box[left_y] + 1.0f;

    T gx = (groundtruth_box[left_x] + groundtruth_box[right_x]) * 0.5f;
    T gy = (groundtruth_box[left_y] + groundtruth_box[right_y]) * 0.5f;
    T gw = groundtruth_box[right_x] - groundtruth_box[left_x] + 1.0f;
    T gh = groundtruth_box[right_y] - groundtruth_box[left_y] + 1.0f;

    T dx = (gx - px) / pw;
    T dy = (gy - py) / ph;
    T dw = log(gw / pw);
    T dh = log(gh / ph);

    deltas[left_x] = (dx - m1) / s1;
    deltas[left_y] = (dy - m2) / s2;
    deltas[right_x] = (dw - m3) / s3;
    deltas[right_y] = (dh - m4) / s4;
  }
}

template <>
__global__ void BoundingBoxEncodeKernel(const size_t size, const half *anchor_box, const half *groundtruth_box,
                                        half *deltas, const float m1, const float m2, const float m3, const float m4,
                                        const float s1, const float s2, const float s3, const float s4) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    const size_t left_x = i * 4;
    const size_t left_y = i * 4 + 1;
    const size_t right_x = i * 4 + 2;
    const size_t right_y = i * 4 + 3;

    const half alpha = 0.5;
    const half beta = 1.0;

    half px = (anchor_box[left_x] + anchor_box[right_x]) * alpha;
    half py = (anchor_box[left_y] + anchor_box[right_y]) * alpha;
    half pw = anchor_box[right_x] - anchor_box[left_x] + beta;
    half ph = anchor_box[right_y] - anchor_box[left_y] + beta;

    half gx = (groundtruth_box[left_x] + groundtruth_box[right_x]) * alpha;
    half gy = (groundtruth_box[left_y] + groundtruth_box[right_y]) * alpha;
    half gw = groundtruth_box[right_x] - groundtruth_box[left_x] + beta;
    half gh = groundtruth_box[right_y] - groundtruth_box[left_y] + beta;

    float dx = (gx - px) / pw;
    float dy = (gy - py) / ph;
    float dw = log(static_cast<float>(gw / pw));
    float dh = log(static_cast<float>(gh / ph));

    deltas[left_x] = (dx - m1) / s1;
    deltas[left_y] = (dy - m2) / s2;
    deltas[right_x] = (dw - m3) / s3;
    deltas[right_y] = (dh - m4) / s4;
  }
}


template <typename T>
void BoundingBoxEncode(const size_t size, const T *anchor_box, const T *groundtruth_box, T *deltas, const float &m1,
                       const float &m2, const float &m3, const float &m4, const float &s1, const float &s2,
                       const float &s3, const float &s4, cudaStream_t cuda_stream) {
  BoundingBoxEncodeKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, anchor_box, groundtruth_box, deltas,
                                                                             m1, m2, m3, m4, s1, s2, s3, s4);
}

template <>
void BoundingBoxEncode(const size_t size, const half *anchor_box, const half *groundtruth_box, half *deltas,
                       const float &m1, const float &m2, const float &m3, const float &m4, const float &s1,
                       const float &s2, const float &s3, const float &s4, cudaStream_t cuda_stream) {
  BoundingBoxEncodeKernel<half><<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, anchor_box, groundtruth_box,
                                                                                   deltas, m1, m2, m3, m4, s1, s2, s3,
                                                                                   s4);
}

template CUDA_LIB_EXPORT void BoundingBoxEncode<float>(const size_t size, const float *anchor_box,
                                                       const float *groundtruth_box, float *deltas,
                                                       const float &m1, const float &m2,
                                                       const float &m3, const float &m4,
                                                       const float &s1, const float &s2,
                                                       const float &s3, const float &s4,
                                                       cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void BoundingBoxEncode<half>(const size_t size, const half *anchor_box,
                                                      const half *groundtruth_box, half *deltas,
                                                      const float &m1, const float &m2,
                                                      const float &m3, const float &m4,
                                                      const float &s1, const float &s2,
                                                      const float &s3, const float &s4,
                                                      cudaStream_t cuda_stream);
