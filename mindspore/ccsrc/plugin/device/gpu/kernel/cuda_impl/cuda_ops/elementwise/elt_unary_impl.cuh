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
 * WITHOut_t WARRANTIES OR CONDITIONS OF ANY KInp_tD, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ELEMENTWISE_UNARY_PUB_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ELEMENTWISE_UNARY_PUB_CUH_
#include <algorithm>
#include <type_traits>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/elementwise/eltwise_ops_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/elementwise/elementswise_pub_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

#define DEVICE __device__ __forceinline__
#define DEVICE_HOST __device__ __host__ __forceinline__

template <enum ElwiseOpType Op, typename Inp_t, typename Out_t>
struct UnaryFunc {
  __device__ __host__ __forceinline__ UnaryFunc() {}
  __device__ __forceinline__ Out_t operator()(Inp_t val) const { return Out_t(0.0); }
};

template <enum ElwiseOpType Op, typename Inp_t, typename Out_t>
__global__ void UnaryNoVecKernel(const UnaryFunc<Op, Inp_t, Out_t> functor, const size_t size, const Inp_t *input,
                                 Out_t *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = functor(input[pos]);
  }
  return;
}
template <enum ElwiseOpType Op, typename Inp_t, typename Out_t>
cudaError_t UnaryOpsCudaFunc(size_t num, const Inp_t *inp, Out_t *out, cudaStream_t stream) {
  UnaryFunc<Op, Inp_t, Out_t> func;
  if (num < 8 * 1024) {
    size_t thread_num = num > 1024 ? 1024 : num;
    auto num_blocks = CUDA_BLOCKS_CAL(GET_CTX_DEVICE_ID, num, thread_num);
    UnaryNoVecKernel<Op, Inp_t, Out_t><<<num_blocks, thread_num, 0, stream>>>(func, num, inp, out);
    return cudaPeekAtLastError();
  }
  return cuda::elementwise::Unary(func, static_cast<uint>(num), out, inp, stream);
}
// float
#define REGISTER_UNARY_OP_CUDA_FUNC_FLOAT_TYPE(Op)                                                                    \
  template CUDA_LIB_EXPORT cudaError_t UnaryOpsCudaFunc<Op, half, half>(const size_t num, const half *inp, half *out, \
                                                                        cudaStream_t cuda_stream);                    \
  template CUDA_LIB_EXPORT cudaError_t UnaryOpsCudaFunc<Op, float, float>(const size_t num, const float *inp,         \
                                                                          float *out, cudaStream_t cuda_stream);      \
  template CUDA_LIB_EXPORT cudaError_t UnaryOpsCudaFunc<Op, double, double>(const size_t num, const double *inp,      \
                                                                            double *out, cudaStream_t cuda_stream)

// bool
#define REGISTER_UNARY_OP_CUDA_FUNC_BOOL_TYPE(Op)                                                                     \
  template CUDA_LIB_EXPORT cudaError_t UnaryOpsCudaFunc<Op, bool, bool>(const size_t num, const bool *inp, bool *out, \
                                                                        cudaStream_t cuda_stream)

// int
#define REGISTER_UNARY_OP_CUDA_FUNC_INT_TYPE(Op)                                                                       \
  template CUDA_LIB_EXPORT cudaError_t UnaryOpsCudaFunc<Op, int8_t, int8_t>(const size_t num, const int8_t *inp,       \
                                                                            int8_t *out, cudaStream_t cuda_stream);    \
  template CUDA_LIB_EXPORT cudaError_t UnaryOpsCudaFunc<Op, int16_t, int16_t>(const size_t num, const int16_t *inp,    \
                                                                              int16_t *out, cudaStream_t cuda_stream); \
  template CUDA_LIB_EXPORT cudaError_t UnaryOpsCudaFunc<Op, int32_t, int32_t>(const size_t num, const int32_t *inp,    \
                                                                              int32_t *out, cudaStream_t cuda_stream); \
  template CUDA_LIB_EXPORT cudaError_t UnaryOpsCudaFunc<Op, int64_t, int64_t>(const size_t num, const int64_t *inp,    \
                                                                              int64_t *out, cudaStream_t cuda_stream)

// uint
#define REGISTER_UNARY_OP_CUDA_FUNC_UINT_TYPE(Op)                                                                      \
  template CUDA_LIB_EXPORT cudaError_t UnaryOpsCudaFunc<Op, uint8_t, uint8_t>(const size_t num, const uint8_t *inp,    \
                                                                              uint8_t *out, cudaStream_t cuda_stream); \
  template CUDA_LIB_EXPORT cudaError_t UnaryOpsCudaFunc<Op, uint16_t, uint16_t>(                                       \
    const size_t num, const uint16_t *inp, uint16_t *out, cudaStream_t cuda_stream);                                   \
  template CUDA_LIB_EXPORT cudaError_t UnaryOpsCudaFunc<Op, uint32_t, uint32_t>(                                       \
    const size_t num, const uint32_t *inp, uint32_t *out, cudaStream_t cuda_stream);                                   \
  template CUDA_LIB_EXPORT cudaError_t UnaryOpsCudaFunc<Op, uint64_t, uint64_t>(                                       \
    const size_t num, const uint64_t *inp, uint64_t *out, cudaStream_t cuda_stream)

// complex
#define REGISTER_UNARY_OP_CUDA_FUNC_COMPLEX_TYPE(Op)                                             \
  template CUDA_LIB_EXPORT cudaError_t UnaryOpsCudaFunc<Op, Complex<float>, Complex<float>>(     \
    const size_t num, const Complex<float> *inp, Complex<float> *out, cudaStream_t cuda_stream); \
  template CUDA_LIB_EXPORT cudaError_t UnaryOpsCudaFunc<Op, Complex<double>, Complex<double>>(   \
    const size_t num, const Complex<double> *inp, Complex<double> *out, cudaStream_t cuda_stream)
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ELEMENTWISE_UNARY_PUB_CUH_
