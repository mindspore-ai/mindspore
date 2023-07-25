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

#include "convert_gradient_impl.cuh"

template <typename T>
__global__ void ConvertGradientKernel(const size_t size, const size_t height_h, const size_t height_w,
                                      const size_t batchwidth, const size_t width, T *input_addr, T *output_addr) {
  for (size_t pointIdx = blockIdx.x * blockDim.x + threadIdx.x; pointIdx < (size); pointIdx += blockDim.x * gridDim.x) {
    size_t dst_batchIdx = pointIdx / (height_h * height_w);
    size_t dst_batchIdxX = dst_batchIdx / batchwidth;
    size_t dst_batchIdxY = dst_batchIdx % batchwidth;
    size_t dst_x = (pointIdx - dst_batchIdx * height_h * height_w) / height_w;
    size_t dst_y = (pointIdx - dst_batchIdx * height_h * height_w) % height_w;
    size_t src_coordinate = dst_batchIdxX * height_h * width + dst_x * width + dst_batchIdxY * height_w + dst_y;
    output_addr[pointIdx] = input_addr[src_coordinate];
  }
}

template <typename T>
__global__ void ConvertGradientBackKernel(const size_t size, const size_t height_h, const size_t height_w,
                                          const size_t batchwidth, const size_t width, T *input_addr, T *output_addr) {
  for (size_t pointIdx = blockIdx.x * blockDim.x + threadIdx.x; pointIdx < (size); pointIdx += blockDim.x * gridDim.x) {
    size_t dst_batchIdx = pointIdx / (height_h * height_w);
    size_t dst_batchIdxX = dst_batchIdx / batchwidth;
    size_t dst_batchIdxY = dst_batchIdx % batchwidth;
    size_t dst_x = (pointIdx - dst_batchIdx * height_h * height_w) / height_w;
    size_t dst_y = (pointIdx - dst_batchIdx * height_h * height_w) % height_w;
    size_t src_coordinate = dst_batchIdxX * height_h * width + dst_x * width + dst_batchIdxY * height_w + dst_y;
    output_addr[src_coordinate] = input_addr[pointIdx];
  }
}

template <typename T>
__global__ void ConvertGradientBackKernel(const size_t size, const size_t height_h, const size_t height_w,
                                          const size_t ori_h, const size_t ori_w, const size_t batchwidth,
                                          const size_t width, T *input_addr, T *output_addr) {
  for (size_t pointIdx = blockIdx.x * blockDim.x + threadIdx.x; pointIdx < (size); pointIdx += blockDim.x * gridDim.x) {
    size_t dst_batchIdx = pointIdx / (height_h * height_w);
    size_t dst_batchIdxX = dst_batchIdx / batchwidth;
    size_t dst_batchIdxY = dst_batchIdx % batchwidth;
    size_t dst_x = (pointIdx - dst_batchIdx * height_h * height_w) / height_w;
    size_t dst_y = (pointIdx - dst_batchIdx * height_h * height_w) % height_w;
    size_t src_x = dst_batchIdxX * height_h + dst_x;
    size_t src_y = dst_batchIdxY * height_w + dst_y;
    if (src_x < ori_h && src_y < ori_w) {
      size_t src_coordinate = src_x * ori_w + src_y;
      output_addr[src_coordinate] = input_addr[pointIdx];
    }
  }
}

template <typename T>
cudaError_t ConvertGradient(const size_t size, const size_t height_h, const size_t height_w, const size_t batchwidth,
                            const size_t width, T *input_addr, T *output_addr, cudaStream_t cuda_stream) {
  ConvertGradientKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, height_h, height_w, batchwidth, width,
                                                                           input_addr, output_addr);
  return GetCudaStatus();
}

template <typename T>
cudaError_t ConvertGradientBack(const size_t size, const size_t height_h, const size_t height_w,
                                const size_t batchwidth, const size_t width, T *input_addr, T *output_addr,
                                cudaStream_t cuda_stream) {
  ConvertGradientBackKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, height_h, height_w, batchwidth,
                                                                               width, input_addr, output_addr);
  return GetCudaStatus();
}

template <typename T>
cudaError_t ConvertGradientBack(const size_t size, const size_t height_h, const size_t height_w, const size_t ori_h,
                                const size_t ori_w, const size_t batchwidth, const size_t width, T *input_addr,
                                T *output_addr, cudaStream_t cuda_stream) {
  ConvertGradientBackKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(
    size, height_h, height_w, ori_h, ori_w, batchwidth, width, input_addr, output_addr);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t ConvertGradient<float>(const size_t size, const size_t height_h,
                                                            const size_t height_w, const size_t batchwidth,
                                                            const size_t width, float *input_addr, float *output_addr,
                                                            cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ConvertGradientBack<float>(const size_t size, const size_t height_h,
                                                                const size_t height_w, const size_t batchwidth,
                                                                const size_t width, float *input_addr,
                                                                float *output_addr, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ConvertGradientBack<float>(const size_t size, const size_t height_h,
                                                                const size_t height_w, const size_t ori_h,
                                                                const size_t ori_w, const size_t batchwidth,
                                                                const size_t width, float *input_addr,
                                                                float *output_addr, cudaStream_t cuda_stream);
