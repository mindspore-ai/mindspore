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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_IOU_IMPL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_IOU_IMPL_H_

#include "runtime/device/gpu/cuda_common.h"

#define IOU_LOCATION_NUM 2
#define IOU_DIMENSION 4

template <typename T>
void IOU(const size_t &size, const T *box1, const T *box2, T *iou_results, const size_t &mode,
         const size_t &input_len_0, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_IOU_IMPL_H_
