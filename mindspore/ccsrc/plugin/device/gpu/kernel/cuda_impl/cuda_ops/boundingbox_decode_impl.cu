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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/boundingbox_decode_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void BoundingBoxDecodeKernel(const size_t size, const T *rois, const T *deltas, T *bboxes, const float m1,
                                        const float m2, const float m3, const float m4, const float s1, const float s2,
                                        const float s3, const float s4, const int max_height, const int max_width,
                                        const float ratio_clip) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    const size_t left_x = i * 4;
    const size_t left_y = i * 4 + 1;
    const size_t right_x = i * 4 + 2;
    const size_t right_y = i * 4 + 3;

    T dx = deltas[left_x] * s1 + m1;
    T dy = deltas[left_y] * s2 + m2;
    T dw = deltas[right_x] * s3 + m3;
    T dh = deltas[right_y] * s4 + m4;

    T max_ratio = abs(log(ratio_clip));

    dw = dw > max_ratio ? max_ratio : (dw < (-max_ratio) ? (-max_ratio) : dw);
    dh = dh > max_ratio ? max_ratio : (dh < (-max_ratio) ? (-max_ratio) : dh);

    T px = (rois[left_x] + rois[right_x]) * 0.5f;
    T py = (rois[left_y] + rois[right_y]) * 0.5f;
    T pw = rois[right_x] - rois[left_x] + 1.0f;
    T ph = rois[right_y] - rois[left_y] + 1.0f;

    T gx = px + pw * dx;
    T gy = py + ph * dy;
    T gw = pw * exp(dw);
    T gh = ph * exp(dh);

    T x1 = gx - gw * 0.5f + 0.5f;
    T y1 = gy - gh * 0.5f + 0.5f;
    T x2 = gx + gw * 0.5f - 0.5f;
    T y2 = gy + gh * 0.5f - 0.5f;

    x1 = x1 > max_width ? max_width : (x1 < 0 ? 0 : x1);
    y1 = y1 > max_height ? max_height : (y1 < 0 ? 0 : y1);
    x2 = x2 > max_width ? max_width : (x2 < 0 ? 0 : x2);
    y2 = y2 > max_height ? max_height : (y2 < 0 ? 0 : y2);

    bboxes[left_x] = x1;
    bboxes[left_y] = y1;
    bboxes[right_x] = x2;
    bboxes[right_y] = y2;
  }
}

template <>
__global__ void BoundingBoxDecodeKernel(const size_t size, const half *rois, const half *deltas, half *bboxes,
                                        const float m1, const float m2, const float m3, const float m4,
                                        const float s1, const float s2, const float s3, const float s4,
                                        const int max_height, const int max_width, const float ratio_clip) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    const size_t left_x = i * 4;
    const size_t left_y = i * 4 + 1;
    const size_t right_x = i * 4 + 2;
    const size_t right_y = i * 4 + 3;

    float dx = static_cast<float>(deltas[left_x]) * s1 + m1;
    float dy = static_cast<float>(deltas[left_y]) * s2 + m2;
    float dw = static_cast<float>(deltas[right_x]) * s3 + m3;
    float dh = static_cast<float>(deltas[right_y]) * s4 + m4;

    float max_ratio = abs(log(ratio_clip));

    dw = dw > max_ratio ? max_ratio : (dw < (-max_ratio) ? (-max_ratio) : dw);
    dh = dh > max_ratio ? max_ratio : (dh < (-max_ratio) ? (-max_ratio) : dh);

    float px = static_cast<float>(rois[left_x] + rois[right_x]) * 0.5f;
    float py = static_cast<float>(rois[left_y] + rois[right_y]) * 0.5f;
    float pw = static_cast<float>(rois[right_x] - rois[left_x]) + 1.0f;
    float ph = static_cast<float>(rois[right_y] - rois[left_y]) + 1.0f;

    float gx = px + pw * dx;
    float gy = py + ph * dy;
    float gw = pw * exp(dw);
    float gh = ph * exp(dh);

    float x1 = gx - gw * 0.5f + 0.5f;
    float y1 = gy - gh * 0.5f + 0.5f;
    float x2 = gx + gw * 0.5f - 0.5f;
    float y2 = gy + gh * 0.5f - 0.5f;

    x1 = x1 > max_width ? max_width : (x1 < 0 ? 0 : x1);
    y1 = y1 > max_height ? max_height : (y1 < 0 ? 0 : y1);
    x2 = x2 > max_width ? max_width : (x2 < 0 ? 0 : x2);
    y2 = y2 > max_height ? max_height : (y2 < 0 ? 0 : y2);

    bboxes[left_x] = x1;
    bboxes[left_y] = y1;
    bboxes[right_x] = x2;
    bboxes[right_y] = y2;
  }
}

template <typename T>
void BoundingBoxDecode(const size_t size, const T *rois, const T *deltas, T *bboxes, const float &m1, const float &m2,
                       const float &m3, const float &m4, const float &s1, const float &s2, const float &s3,
                       const float &s4, const int &max_height, const int &max_width, const float &ratio_clip,
                       cudaStream_t cuda_stream) {
  BoundingBoxDecodeKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, rois, deltas, bboxes, m1, m2, m3, m4,
                                                                             s1, s2, s3, s4, max_height, max_width,
                                                                             ratio_clip);
}

template <>
void BoundingBoxDecode(const size_t size, const half *rois, const half *deltas, half *bboxes, const float &m1,
                       const float &m2,
                       const float &m3, const float &m4, const float &s1, const float &s2, const float &s3,
                       const float &s4, const int &max_height, const int &max_width, const float &ratio_clip,
                       cudaStream_t cuda_stream) {
  BoundingBoxDecodeKernel<half><<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, rois, deltas, bboxes, m1, m2,
                                                                                   m3, m4, s1, s2, s3, s4, max_height,
                                                                                   max_width, ratio_clip);
}

template CUDA_LIB_EXPORT void BoundingBoxDecode<float>(const size_t size, const float *rois, const float *deltas,
                                                       float *bboxes,
                                                       const float &m1, const float &m2,
                                                       const float &m3, const float &m4,
                                                       const float &s1, const float &s2,
                                                       const float &s3, const float &s4,
                                                       const int &max_height, const int &max_width,
                                                       const float &ratio_clip, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void BoundingBoxDecode<half>(const size_t size, const half *rois, const half *deltas,
                                                      half *bboxes,
                                                      const float &m1, const float &m2,
                                                      const float &m3, const float &m4,
                                                      const float &s1, const float &s2,
                                                      const float &s3, const float &s4,
                                                      const int &max_height, const int &max_width,
                                                      const float &ratio_clip, cudaStream_t cuda_stream);
