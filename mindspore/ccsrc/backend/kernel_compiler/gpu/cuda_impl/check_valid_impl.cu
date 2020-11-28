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

#include "backend/kernel_compiler/gpu/cuda_impl/check_valid_impl.cuh"

template <typename T, typename S>
__global__ void CheckValidKernel(const size_t size, const T *box, const T *img_metas, S *valid) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    const size_t left_x = i * 4;
    const size_t left_y = i * 4 + 1;
    const size_t right_x = i * 4 + 2;
    const size_t right_y = i * 4 + 3;

    S valid_flag = false;
    valid_flag |= !(box[left_x] >= static_cast<T>(0.0));
    valid_flag |= !(box[left_y] >= static_cast<T>(0.0));
    valid_flag |= !(img_metas[1] * img_metas[2] - static_cast<T>(1.0) >= box[right_x]);
    valid_flag |= !(img_metas[0] * img_metas[2] - static_cast<T>(1.0) >= box[right_y]);

    valid[i] = !valid_flag;
  }

  return;
}

template <typename S>
__global__ void CheckValidKernel(const size_t size, const unsigned char *box,
                                 const unsigned char *img_metas, S *valid) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    const size_t right_x = i * 4 + 2;
    const size_t right_y = i * 4 + 3;

    S valid_flag = false;
    valid_flag |= !(img_metas[0] * img_metas[2] >= box[right_x] + 1);
    valid_flag |= !(img_metas[1] * img_metas[2] >= box[right_y] + 1);

    valid[i] = !valid_flag;
  }

  return;
}

template <typename T, typename S>
void CheckValid(const size_t &size, const T *box, const T *img_metas, S *valid, cudaStream_t cuda_stream) {
  CheckValidKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, box, img_metas, valid);
}

template void CheckValid(const size_t &size, const float *box, const float *img_metas, bool *valid,
                         cudaStream_t cuda_stream);
template void CheckValid(const size_t &size, const half *box, const half *img_metas, bool *valid,
                         cudaStream_t cuda_stream);
template void CheckValid(const size_t &size, const short *box, const short *img_metas, bool *valid,  // NOLINT
                         cudaStream_t cuda_stream);
template void CheckValid(const size_t &size, const unsigned char *box, const unsigned char *img_metas, bool *valid,
                         cudaStream_t cuda_stream);
