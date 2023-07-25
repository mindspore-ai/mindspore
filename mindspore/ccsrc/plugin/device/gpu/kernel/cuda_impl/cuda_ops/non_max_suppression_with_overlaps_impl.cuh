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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_NON_MAX_SUPPRESSION_WITH_OVERLAPS_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_NON_MAX_SUPPRESSION_WITH_OVERLAPS_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalSort(const int &inner, int *index_buff, T *score_buff, T *up_score_buff,
                                    int *valid_score_num, T *score_threshold, const uint32_t &device_id,
                                    cudaStream_t stream, int *valid_num_host);

CUDA_LIB_EXPORT cudaError_t CalPreprocess(const int num, bool *sel_boxes, bool *row_mask, const uint32_t &device_id,
                                          cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalNms(const int num, const int total_num, const T *IOU_value, T *overlaps, bool *sel_boxes,
                                   bool *row_mask, int *index_buff, const uint32_t &device_id,
                                   cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalPostprocess(const int inputsize, int *maxoutput_size, int *valid_score_num,
                                           T *score_threshold, int *index_buff, int *sel_idx, bool *sel_boxes,
                                           const uint32_t &device_id, cudaStream_t cuda_stream, int *num_output_host);

CUDA_LIB_EXPORT int NumRoundUpPower2(int v);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_NON_MAX_SUPPRESSION_WITH_OVERLAPS_IMPL_CUH_
