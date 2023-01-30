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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_NMS_WITH_MASK_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_NMS_WITH_MASK_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

template <typename T>
CUDA_LIB_EXPORT void CalSort(const int &inner, const T *data_in, T *data_out, int *index_buff, T *data_buff,
                             int box_size_, const uint32_t &device_id, cudaStream_t stream);

template <typename T>
CUDA_LIB_EXPORT void CalPreprocess(const int num, int *sel_idx, bool *sel_boxes, const T *input, T *output,
                                   int *index_buff, int box_size_, bool *row_mask, const uint32_t &device_id,
                                   cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT void CalPreprocess(const int num, int *sel_idx, int *sel_boxes, const T *input, T *output,
                                   int *index_buff, int box_size_, bool *row_mask, const uint32_t &device_id,
                                   cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT void CalNms(const int num, const float IOU_value, T *output, bool *sel_boxes, int box_size_,
                            bool *row_mask, const uint32_t &device_id, cudaStream_t cuda_stream);

// for tensorrt plugin which output tensor type can not be bool
template <typename T>
CUDA_LIB_EXPORT void CalNms(const int num, const float IOU_value, T *output, int *sel_boxes, int box_size_,
                            bool *row_mask, const uint32_t &device_id, cudaStream_t cuda_stream);

CUDA_LIB_EXPORT int NmsRoundUpPower2(int v);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_NMS_WITH_MASK_IMPL_CUH_
