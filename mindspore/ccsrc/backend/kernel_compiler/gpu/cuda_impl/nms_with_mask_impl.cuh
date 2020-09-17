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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_NMS_WITH_MASK_IMPL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_NMS_WITH_MASK_IMPL_H_

#include "runtime/device/gpu/cuda_common.h"

template <typename T>
void CalSort(const int &inner, T *data_in, T *data_out, int *index_buff, T *data_buff, int box_size_,
             cudaStream_t stream);

template <typename T>
void CalPreprocess(const int num, int *sel_idx, bool *sel_boxes, T *input, T *output, int *index_buff, int box_size_,
                   bool *row_mask, cudaStream_t cuda_stream);

template <typename T>
void CalNms(const int num, const float IOU_value, T *output, bool *sel_boxes, int box_size_, bool *row_mask,
            cudaStream_t cuda_stream);

int NmsRoundUpPower2(int v);

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_NMS_WITH_MASK_IMPL_H_
