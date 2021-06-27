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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_SCATTER_ND_FUNCTOR_IMPL_CUH_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_SCATTER_ND_FUNCTOR_IMPL_CUH_

#include "runtime/device/gpu/cuda_common.h"

enum ScatterNdFunctorType {
  SCATTER_ND_FUNC_UPDATE = 0,
  SCATTER_ND_FUNC_ADD,
  SCATTER_ND_FUNC_SUB,
  SCATTER_ND_FUNC_INVALID_TYPE = 255
};

template <typename T, typename S>
void CalScatterNdFunctor(enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units,
                         const size_t &index_depth, const S *out_strides, const S *indices, const T *updates, T *input,
                         cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_SCATTER_ND_FUNCTOR_IMPL_CUH_
