/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_RANDOM_CHOICE_WITH_MASK_IMPL_CUH_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_RANDOM_CHOICE_WITH_MASK_IMPL_CUH_

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "runtime/device/gpu/cuda_common.h"
#define BLOCKSIZE 256
#define MAX_DIMENSION 5

template <typename T, typename S, typename K>
void CalRandomChoiceWithMaskSmall(int input_size, int seedc, int count, K *input, S *output_index, K *output_mask,
                               cudaStream_t stream);

template <typename T, typename S>
void CalRandomChoiceWithMask(const int &input_size, const int &input_shape_size, const int &d1, const int &d2,
                             const int &d3, const int &d4, const int &d5, const int &seedc, const int &count,
                             const T *input, S *output_index, T *output_mask, S *index_buff, S *mask_buff, S *rank_buff,
                             S *Tnum_buff, S *tmp_buff, curandState *globalState, cudaStream_t stream);

int RcwmRoundUpPower2(int v);

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_RANDOM_CHOICE_WITH_MASK_IMPL_CUH_
