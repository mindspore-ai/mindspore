/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ELEMENTWISE_OPS_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ELEMENTWISE_OPS_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/elementwise/eltwise_ops_type.cuh"
template <enum ElwiseOpType Op, typename In0_t, typename In1_t, typename Out_t>
CUDA_LIB_EXPORT cudaError_t BinaryOpsCudaFunc(const size_t num, const In0_t *in0, const In1_t *in1, Out_t *out,
                                              cudaStream_t cuda_stream);

template <enum ElwiseOpType Op, typename Inp_t, typename Out_t>
CUDA_LIB_EXPORT cudaError_t UnaryOpsCudaFunc(const size_t num, const Inp_t *inp, Out_t *out, cudaStream_t stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ELEMENTWISE_OPS_IMPL_CUH_
