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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ELEMENTWISE_BINARY_PUB_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ELEMENTWISE_BINARY_PUB_CUH_
#include <algorithm>
#include <type_traits>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/elementwise/eltwise_ops_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/elementwise/elementswise_pub_impl.cuh"

#define DEVICE __device__ __forceinline__
#define DEVICE_HOST __device__ __host__ __forceinline__

template <enum ElwiseOpType Op, typename In0_t, typename In1_t, typename Out_t>
struct BinaryFunc {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ Out_t operator()(In0_t val0, In1_t val1) const { return Out_t(0.0); }
};

template <enum ElwiseOpType Op, typename In0_t, typename In1_t, typename Out_t>
__global__ void BinaryNoVecKernel(const BinaryFunc<Op, In0_t, In1_t, Out_t> functor, const size_t size,
                                  const In0_t *in0, const In0_t *in1, Out_t *out) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    out[pos] = functor(in0[pos], in1[pos]);
  }
  return;
}
template <enum ElwiseOpType Op, typename In0_t, typename In1_t, typename Out_t>
cudaError_t BinaryOpsCudaFunc(const size_t num, const In0_t *in0, const In1_t *in1, Out_t *out, cudaStream_t stream) {
  BinaryFunc<Op, In0_t, In1_t, Out_t> func;
  if (num < 8 * 1024) {
    size_t thread_num = num > 1024 ? 1024 : num;
    auto num_blocks = CUDA_BLOCKS_CAL(GET_CTX_DEVICE_ID, num, thread_num);
    BinaryNoVecKernel<<<num_blocks, thread_num, 0, stream>>>(func, num, in0, in1, out);
    return cudaPeekAtLastError();
  }
  return cuda::elementwise::Binary(func, static_cast<uint>(num), out, in0, in1, stream);
}
// float
#define REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(Op)                                              \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpsCudaFunc<Op, half, half, half>(                  \
    const size_t num, const half *in0, const half *in1, half *out, cudaStream_t cuda_stream);    \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpsCudaFunc<Op, float, float, float>(               \
    const size_t num, const float *in0, const float *in1, float *out, cudaStream_t cuda_stream); \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpsCudaFunc<Op, double, double, double>(            \
    const size_t num, const double *in0, const double *in1, double *out, cudaStream_t cuda_stream)

// bool
#define REGISTER_BINARY_OP_CUDA_FUNC_BOOL_TYPE(Op)                              \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpsCudaFunc<Op, bool, bool, bool>( \
    const size_t num, const bool *in0, const bool *in1, bool *out, cudaStream_t cuda_stream)

// int
#define REGISTER_BINARY_OP_CUDA_FUNC_INT_TYPE(Op)                                                      \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpsCudaFunc<Op, int8_t, int8_t, int8_t>(                  \
    const size_t num, const int8_t *in0, const int8_t *in1, int8_t *out, cudaStream_t cuda_stream);    \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpsCudaFunc<Op, int16_t, int16_t, int16_t>(               \
    const size_t num, const int16_t *in0, const int16_t *in1, int16_t *out, cudaStream_t cuda_stream); \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpsCudaFunc<Op, int32_t, int32_t, int32_t>(               \
    const size_t num, const int32_t *in0, const int32_t *in1, int32_t *out, cudaStream_t cuda_stream); \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpsCudaFunc<Op, int64_t, int64_t, int64_t>(               \
    const size_t num, const int64_t *in0, const int64_t *in1, int64_t *out, cudaStream_t cuda_stream)

// uint
#define REGISTER_BINARY_OP_CUDA_FUNC_UINT_TYPE(Op)                                                        \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpsCudaFunc<Op, uint8_t, uint8_t, uint8_t>(                  \
    const size_t num, const uint8_t *in0, const uint8_t *in1, uint8_t *out, cudaStream_t cuda_stream);    \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpsCudaFunc<Op, uint16_t, uint16_t, uint16_t>(               \
    const size_t num, const uint16_t *in0, const uint16_t *in1, uint16_t *out, cudaStream_t cuda_stream); \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpsCudaFunc<Op, uint32_t, uint32_t, uint32_t>(               \
    const size_t num, const uint32_t *in0, const uint32_t *in1, uint32_t *out, cudaStream_t cuda_stream); \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpsCudaFunc<Op, uint64_t, uint64_t, uint64_t>(               \
    const size_t num, const uint64_t *in0, const uint64_t *in1, uint64_t *out, cudaStream_t cuda_stream)

// complex
#define REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(Op)                                                            \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpsCudaFunc<Op, Complex<float>, Complex<float>, Complex<float>>(    \
    const size_t num, const Complex<float> *in0, const Complex<float> *in1, Complex<float> *out,                 \
    cudaStream_t cuda_stream);                                                                                   \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpsCudaFunc<Op, Complex<double>, Complex<double>, Complex<double>>( \
    const size_t num, const Complex<double> *in0, const Complex<double> *in1, Complex<double> *out,              \
    cudaStream_t cuda_stream)
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ELEMENTWISE_BINARY_PUB_CUH_
