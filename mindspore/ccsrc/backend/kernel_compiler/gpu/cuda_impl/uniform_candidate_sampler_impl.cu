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

#include "backend/kernel_compiler/gpu/cuda_impl/uniform_candidate_sampler_impl.cuh"

template <typename S>
__global__ void AssignToOutput(const int64_t size, const S prob_val, S *output_array) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output_array[pos] = prob_val;
  }
}

template <typename S>
void CalUniformCandidateSampler(const int64_t true_size, const int64_t num_sampled, const S prob_val,
                                S *true_expected_count, S *sampled_expected_count, cudaStream_t cuda_stream) {
  AssignToOutput<<<GET_BLOCKS(true_size), GET_THREADS, 0, cuda_stream>>>(true_size, prob_val, true_expected_count);
  AssignToOutput<<<GET_BLOCKS(num_sampled), GET_THREADS, 0, cuda_stream>>>(num_sampled, prob_val,
                                                                           sampled_expected_count);
}

template void CalUniformCandidateSampler<float>(const int64_t true_size, const int64_t num_sampled,
                                                const float prob_val, float *true_expected_count,
                                                float *sampled_expected_count, cudaStream_t cuda_stream);
