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

#include "backend/kernel_compiler/gpu/cuda_impl/random_categorical.cuh"

template <typename S>
__global__ void RandomCategorical(const size_t num_samples, double** dev_rand, double** dev_cdf,
    const size_t batch_size, const size_t num_classes, S *output_addr) {
    size_t size = num_samples * batch_size;
    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += gridDim.x * blockDim.x) {
        size_t cur_row = pos / num_samples;
        size_t cur_col = pos % num_samples;
        const double to_find = dev_cdf[cur_row][num_classes-1] * dev_rand[cur_row][cur_col];

        size_t idx = 0;
        while (dev_cdf[cur_row][idx] < to_find) {
            idx++;
        }
        output_addr[pos] = static_cast<S>(idx);
    }
}

template <typename T>
__global__ void GetCdf(const T *logits_addr, double** dev_cdf, const size_t batch_size, const size_t num_classes) {
    size_t size = num_classes * batch_size;
    for (size_t pos= blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += gridDim.x * blockDim.x) {
      size_t cur_row = pos / num_classes;
      size_t cur_col = pos % num_classes;
      if (cur_col != 0) {
          return;
      }
      T max_of_row = logits_addr[pos];
      for (size_t i = 1; i < num_classes; i++) {
          if (logits_addr[pos + i] > max_of_row) {
              max_of_row = logits_addr[pos + i];
          }
      }
      dev_cdf[cur_row][0] = exp(static_cast<double>(logits_addr[pos] - max_of_row));
      for (size_t i = 1; i < num_classes; i++) {
          double tmp = exp(static_cast<double>(logits_addr[pos + i] - max_of_row));
          dev_cdf[cur_row][i] = dev_cdf[cur_row][i - 1] + tmp;
      }
    }
}

template <typename S>
void RandomCategoricalKernel(const size_t num_samples, double** dev_rand, double** dev_cdf,
    const size_t batch_size, const size_t num_classes, S *output_addr, cudaStream_t cuda_stream) {
    size_t size_out = num_samples * batch_size;
    RandomCategorical<<<GET_BLOCKS(size_out), GET_THREADS, 0, cuda_stream>>>(num_samples, dev_rand,
                                                                             dev_cdf, batch_size,
                                                                             num_classes, output_addr);
}

template <typename T>
void GetCdfKernel(const T *logits_addr, double** dev_cdf, const size_t batch_size, const size_t num_classes,
                  cudaStream_t cuda_stream) {
    size_t size_cdf = num_classes * batch_size;
    GetCdf<<<GET_BLOCKS(size_cdf), GET_THREADS, 0, cuda_stream>>>(logits_addr, dev_cdf, batch_size, num_classes);
}

template void GetCdfKernel<half>(const half *logits_addr, double** dev_cdf, const size_t batch_size,
        const size_t num_classes, cudaStream_t cuda_stream);
template void GetCdfKernel<float>(const float *logits_addr, double** dev_cdf, const size_t batch_size,
        const size_t num_classes, cudaStream_t cuda_stream);
template void GetCdfKernel<double>(const double *logits_addr, double** dev_cdf, const size_t batch_size,
        const size_t num_classes, cudaStream_t cuda_stream);

template void RandomCategoricalKernel<int16_t>(const size_t num_samples,
        double** dev_rand, double** dev_cdf, const size_t batch_size, const size_t num_classes,
        int16_t *output_addr, cudaStream_t cuda_stream);
template void RandomCategoricalKernel<int>(const size_t num_samples,
        double** dev_rand, double** dev_cdf, const size_t batch_size, const size_t num_classes,
        int *output_addr, cudaStream_t cuda_stream);
template void RandomCategoricalKernel<int64_t>(const size_t num_samples,
        double** dev_rand, double** dev_cdf, const size_t batch_size, const size_t num_classes,
        int64_t *output_addr, cudaStream_t cuda_stream);
