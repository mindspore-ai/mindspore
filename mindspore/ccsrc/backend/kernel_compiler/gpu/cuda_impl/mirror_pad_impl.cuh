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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_MIRROR_PAD_IMPL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_MIRROR_PAD_IMPL_H_
#include <cuda_runtime.h>
#include "runtime/device/gpu/cuda_common.h"

// preset size of paddings
#define MAX_PADDINGS 4
#define PADDING_SIZE 2

// define constants for kernel indexing use
#define BATCH 0 * PADDING_SIZE
#define CHANNEL 1 * PADDING_SIZE
#define HEIGHT 2 * PADDING_SIZE
#define WIDTH 3 * PADDING_SIZE
#define TOP 0
#define BOTTOM 1
#define LEFT 0
#define RIGHT 1

template <typename T>
void CalMirrorPad(const size_t size, const T *input, const int old_batch, const int old_channel, const int old_height,
                  const int old_width, const int padded_height, const int padded_width, int padd_num,
                  const int64_t *paddings, int mode, T *output, cudaStream_t cuda_stream);
template <typename T>
void CalMirrorPadGrad(const size_t dx_size, const size_t dy_size, T *dy, T *interim, const int output_batch,
                      const int output_channel, const int output_height, const int output_width, const int input_height,
                      const int input_width, const int padd_dim, const int64_t *paddings, int mode, T *dx,
                      cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_MIRROR_PAD_IMPL_H_
