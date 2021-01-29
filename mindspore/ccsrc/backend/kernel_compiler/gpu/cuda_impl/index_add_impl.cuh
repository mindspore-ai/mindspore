/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_INDEXADD_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_INDEXADD_H_
enum class IndexAddErrorCode {
  kOk = 0,
  kIndexOutOfRange
};

void ValidateIndexAddInputValues(const int *index, const size_t src_axis_size, const size_t dst_axis_size,
  IndexAddErrorCode *error_code, cudaStream_t cuda_stream);

template <typename T>
void CalIndexAdd(T *dst, const int *index, const T *src, const size_t outer_size, const size_t src_axis_size,
  const size_t dst_axis_size, const size_t inner_size, const bool use_lock, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_INDEXADD_H_
