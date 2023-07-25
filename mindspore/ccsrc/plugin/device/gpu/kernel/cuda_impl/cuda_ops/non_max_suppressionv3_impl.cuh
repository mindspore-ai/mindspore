/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_NONMAXSUPPRESSIONV3_IMPL_CUH_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_NONMAXSUPPRESSIONV3_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

template <typename T, typename M, typename S>
CUDA_LIB_EXPORT cudaError_t DoNms(const int num_input, int *count, int *num_keep, T *scores, T *boxes_in,
                                  M iou_threshold_, M score_threshold_, int *index_buff, S max_output_size_,
                                  int box_size, unsigned int *sel_mask, bool *sel_boxes, int *output_ptr,
                                  const uint32_t &device_id, cudaStream_t cuda_stream, int *output_size);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_MATH_NONMAXSUPPRESSIONV3_IMPL_CUH_
