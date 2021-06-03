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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_EXTRACT_IMAGE_PATCHES_IMPL_CUH_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_EXTRACT_IMAGE_PATCHES_IMPL_CUH_

#include <vector>
#include "runtime/device/gpu/cuda_common.h"

template <typename T>
void CalExtractImagePatchesNHWC(size_t output_size, int64_t stride_row, int64_t stride_col, int64_t rate_row,
                                int64_t rate_col, int64_t output_cols, bool need_batch, int64_t row_stride,
                                int64_t patch_stride, int64_t other_stride, int64_t input_row_size,
                                int64_t input_col_size, int64_t row_padding_top, int64_t col_padding_left,
                                int64_t col_input_stride, int64_t row_input_stride, int64_t patch_input_stride,
                                int64_t output_depth, const T *input, T *output, cudaStream_t stream);

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_EXTRACT_IMAGE_PATCHES_IMPL_CUH_
