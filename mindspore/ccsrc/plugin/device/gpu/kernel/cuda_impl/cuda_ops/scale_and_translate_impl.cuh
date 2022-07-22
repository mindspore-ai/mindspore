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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SCALE_AND_TRANSLATE_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SCALE_AND_TRANSLATE_IMPL_CUH_
#include <string>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"
template <typename T>
CUDA_LIB_EXPORT void CalScaleAndTranslate(const size_t *thread_num, const T *image, const float *scale,
                                          const float *translation, int64_t batch, int64_t image_height,
                                          int64_t image_width, int64_t channels, int64_t output_height,
                                          int64_t output_width, std::string kernel_type, bool antialias,
                                          const float radius, const int64_t *input_shape, const int32_t *size,
                                          int32_t *spans_size, int32_t *forward_starts, float *forward_weights,
                                          float *intermediate, float *output, const uint32_t &device_id,
                                          cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT void CallScaleAndTranslateGrad(const std::string kernel_type, const T *grads, const T *original_image,
  const float radius, const int64_t *input_shape, const int32_t *size, const float *scale, const float *translate,
  const bool antialias, int32_t *spans_size, int32_t *forward_starts, int32_t *grad_starts, float *forward_weights,
  float *grad_weights, const size_t *thread_num, float *intermediate, const int64_t input_pix_per_batch,
  const int64_t intermediate_pix_per_batch, const int64_t output_pix_per_batch, float *output, int32_t *weight_size,
  const uint32_t &device_id, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SCALE_AND_TRANSLATE_IMPL_CUH_
