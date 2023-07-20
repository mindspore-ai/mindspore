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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CONVOLUTION_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CONVOLUTION_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/convolution/convolution_func_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/convolution/convolution_common.cuh"

template <enum ConvolutionOpType op, typename T>
struct ConvolutionFunc {
  __host__ __forceinline__ ConvolutionFunc() {}
  __host__ __forceinline__ cudaError_t operator()(const ConvolutionCudaArgs &cuda_args, const T *input0_addr,
                                                  const T *input1_addr, T *output_addr,
                                                  cudaStream_t cuda_stream) const {
    return cudaSuccess;
  }
};

#define REGISTER_CONVOLUTION_OP_CUDA_FUNC_FLOAT_TYPE(OP)                                                          \
  template CUDA_LIB_EXPORT cudaError_t ConvolutionOpCudaFunc<OP, float>(                                          \
    const ConvolutionCudaArgs &cuda_args, const float *input0_addr, const float *input1_addr, float *output_addr, \
    cudaStream_t cuda_stream);                                                                                    \
  template CUDA_LIB_EXPORT cudaError_t ConvolutionOpCudaFunc<OP, half>(                                           \
    const ConvolutionCudaArgs &cuda_args, const half *input0_addr, const half *input1_addr, half *output_addr,    \
    cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CONVOLUTION_IMPL_CUH_
