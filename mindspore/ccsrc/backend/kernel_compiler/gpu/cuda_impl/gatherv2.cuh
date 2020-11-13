/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_GATHER_GPU_CU_H
#define MINDSPORE_GATHER_GPU_CU_H
template <typename T, typename S>
void CalGatherV2StaticShape(T *input, S *indices, T *output, size_t output_dim0, size_t output_dim1, size_t output_dim2,
                            size_t input_dim1, cudaStream_t stream);

template <typename T, typename S>
void CalGatherV2DynamicShape(T *input, S *indices, T *output, size_t *input_shape_wksp, size_t input_rank,
                             size_t *indices_shape_wksp, size_t indices_rank, int64_t *axis_wksp,
                             size_t *output_shape_wksp, const int max_output_size, cudaStream_t stream);
#endif
