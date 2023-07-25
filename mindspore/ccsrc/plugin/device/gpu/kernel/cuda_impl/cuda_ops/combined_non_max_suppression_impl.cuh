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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_COMBINED_NON_MAX_SUPPRESSION_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_COMBINED_NON_MAX_SUPPRESSION_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalSort(T *scores, int *index, T *score_threshold, int num_classes, T *boxes,
                                    float *new_boxes, float *new_scores, int batch_size, int num_boxes,
                                    float *boxes_result, int q, bool *sel, const uint32_t &device_id,
                                    cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT cudaError_t Calnms(int batch_size, int num_classes, T *iou_threshold, bool *sel, float *boxes_result,
                                   int *index, int q, int num_boxes, int *max_output_size_per_class, float *new_scores,
                                   bool *mask, const uint32_t &device_id, cudaStream_t cuda_stream);
template <typename T>
CUDA_LIB_EXPORT cudaError_t Caloutput(int batch_size, int per_detections, int *index, float *new_scores, bool *sel,
                                      float *new_boxes, T *nmsed_classes, T *nmsed_scores, T *nmsed_boxes,
                                      int *valid_detections, bool clip_boxes, int num_classes, int num_boxes, int q,
                                      const uint32_t &device_id, cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_COMBINED_NON_MAX_SUPPRESSION_IMPL_CUH_
