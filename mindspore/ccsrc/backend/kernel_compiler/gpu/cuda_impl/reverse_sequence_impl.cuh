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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_REVERSE_SEQUENCE_IMPL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_REVERSE_SEQUENCE_IMPL_H_
#include <cuda_runtime.h>
#include "runtime/device/gpu/cuda_common.h"

template <typename T, typename S>
void CalReverseSequence(const size_t size, const T *input, const S *seq_len, const int64_t batch_dim,
                        const int64_t seq_dim, size_t *cur_pos_arr, const size_t *input_shape_ptr,
                        size_t *intput_shape_cum_ptr, size_t shape_size, T *output, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_REVERSE_SEQUENCE_IMPL_H_
