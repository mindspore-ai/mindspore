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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_ADAPTIVEAVGPOOL2D_IMPL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_ADAPTIVEAVGPOOL2D_IMPL_H_

#include "runtime/device/gpu/cuda_common.h"
template <typename T>
void ApplyAdaptiveAvgPool2D(const uint size, const uint input_height, const uint input_width, const uint output_height,
                            const uint output_width, T *input_data, T *output_data, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_ADAPTIVEAVGPOOL2D_IMPL_H_
