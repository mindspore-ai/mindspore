/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/dropout_nd_impl.cuh"
#include <stdint.h>
#include "include/cuda_fp16.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/elementswise_op_impl.cuh"
constexpr uint kThreadsPerBlock = cuda::elementwise::kThreadsPerBlock;

template <typename T>
struct DropOutput {
  __device__ DropOutput() {}
  T output_{0};
  bool mask_{false};
};

template <typename T>
struct DropoutNdFunctor {
  float scale_;
  float keep_prob_;
  uint inner_size_;
  explicit DropoutNdFunctor(float scale, float keep_prob, uint inner_size)
      : scale_(scale), keep_prob_(keep_prob), inner_size_(inner_size) {}
  __device__ __forceinline__ DropOutput<T> operator()(const float rand_f, const T input_x) const {
    auto output = DropOutput<T>();
    bool drop_f = rand_f <= keep_prob_;
    if (!drop_f) {
      return output;
    }
    output.output_ = scale_ * input_x * drop_f;
    output.mask_ = drop_f;
    return output;
  }
};

template <>
struct DropoutNdFunctor<half> {
  float scale_;
  float keep_prob_;
  uint inner_size_;
  explicit DropoutNdFunctor<half>(float scale, float keep_prob, uint inner_size)
      : scale_(scale), keep_prob_(keep_prob), inner_size_(inner_size) {}
  __device__ __forceinline__ DropOutput<half> operator()(const float rand_f, const half input_x) const {
    auto output = DropOutput<half>();
    bool drop_f = rand_f <= keep_prob_;
    output.output_ = __float2half(scale_ * __half2float(input_x) * static_cast<float>(drop_f));
    output.mask_ = drop_f;
    return output;
  }
};

template <typename Func, uint vec_size, typename T>
__device__ __forceinline__ void VectorizedCall(Func func, const T *in, const float *rand_f, T *out, bool *mask,
                                               const float keep_prob, int inner_size, uint offset) {
  uint tid = threadIdx.x;
  auto index = tid * vec_size + offset;
  auto x = index / inner_size;
  auto y = index % inner_size;
  auto rand = rand_f[x];

  using VecT = cuda::elementwise::AlignVec<T, vec_size>;
  using VecBool = cuda::elementwise::AlignVec<bool, vec_size>;

  auto vec_in = reinterpret_cast<const VecT *>(in + offset);
  auto vec_out = reinterpret_cast<VecT *>(out + offset);
  auto vec_mask = reinterpret_cast<VecBool *>(mask + offset);
  VecT cache = vec_in[tid];
  VecT out1{0};
  VecBool out2{false};

  if (x == (index + vec_size) / inner_size && rand > keep_prob) {
    vec_out[tid] = out1;
    vec_mask[tid] = out2;
    return;
  }

#pragma unroll
  for (uint j = 0; j < vec_size; j++) {
    auto output_pair = func(rand, cache.elements_[j]);
    out1.elements_[j] = output_pair.output_;
    out2.elements_[j] = output_pair.mask_;
    if (++y == inner_size) {
      y = 0;
      rand = rand_f[++x];
    }
  }
  vec_out[tid] = out1;
  vec_mask[tid] = out2;
}

template <typename Func, uint vec_size, typename T>
__device__ __forceinline__ void NormalCall(Func func, const T *in, const float *rand_f, T *out, bool *mask,
                                           int inner_size, uint offset, uint remaining) {
  uint loop = UP_DIV(remaining, vec_size);
  for (uint i = threadIdx.x; i < loop; i += blockDim.x) {
#pragma unroll
    for (uint j = 0; j < vec_size; j++) {
      uint index = i * vec_size + j;
      if (index >= remaining) {
        return;
      }
      index += offset;
      auto rand = rand_f[index / inner_size];
      auto output_pair = func(rand, in[index]);
      out[index] = output_pair.output_;
      mask[index] = output_pair.mask_;
    }
  }
}

template <typename Func, uint vec_size, typename T>
__global__ void DropoutNdVectorized(Func func, const T *in, const float *rand_f, T *out, bool *mask,
                                    const float keep_prob, uint inner_size, uint num_of_elements) {
  uint elements_per_block = kThreadsPerBlock * vec_size;
  for (uint offset = elements_per_block * blockIdx.x; offset < num_of_elements;
       offset += elements_per_block * gridDim.x) {
    uint remaining = num_of_elements - offset;
    if (remaining < elements_per_block) {
      NormalCall<Func, vec_size, T>(func, in, rand_f, out, mask, inner_size, offset, remaining);
    } else {
      VectorizedCall<Func, vec_size, T>(func, in, rand_f, out, mask, keep_prob, inner_size, offset);
    }
  }
}

template <typename T>
void DropoutNDForward(const T *input, bool *mask, T *output, float *rand_f, const size_t num_count,
                      const float keep_prob, const size_t num_per_chan, const uint32_t &device_id,
                      cudaStream_t cuda_stream) {
  const float scale = 1.f / keep_prob;
  uint inner_size = (uint)(num_per_chan);
  constexpr uint vec_size = cuda::elementwise::VecSize<T>();
  const auto block_x = uint(kThreadsPerBlock);
  const uint elements_per_block = kThreadsPerBlock * vec_size;
  const auto grid_x = uint(UP_DIV(num_count, elements_per_block));
  dim3 block{block_x};
  dim3 grid{grid_x};
  DropoutNdFunctor<T> functor{scale, keep_prob, inner_size};
  DropoutNdVectorized<DropoutNdFunctor<T>, vec_size, T>
    <<<grid, block, 0, cuda_stream>>>(functor, input, rand_f, output, mask, keep_prob, inner_size, num_count);
}

template CUDA_LIB_EXPORT void DropoutNDForward(const float *input, bool *mask, float *output, float *rand_f,
                                               const size_t num_count, const float keep_prob, const size_t num_per_chan,
                                               const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void DropoutNDForward(const double *input, bool *mask, double *output, float *rand_f,
                                               const size_t num_count, const float keep_prob, const size_t num_per_chan,
                                               const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void DropoutNDForward(const half *input, bool *mask, half *output, float *rand_f,
                                               const size_t num_count, const float keep_prob, const size_t num_per_chan,
                                               const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void DropoutNDForward<int8_t>(const int8_t *input, bool *mask, int8_t *output, float *rand_f,
                                                       const size_t num_count, const float keep_prob,
                                                       const size_t num_per_chan, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void DropoutNDForward<int16_t>(const int16_t *input, bool *mask, int16_t *output,
                                                        float *rand_f, const size_t num_count, const float keep_prob,
                                                        const size_t num_per_chan, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void DropoutNDForward<int32_t>(const int32_t *input, bool *mask, int32_t *output,
                                                        float *rand_f, const size_t num_count, const float keep_prob,
                                                        const size_t num_per_chan, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void DropoutNDForward<int64_t>(const int64_t *input, bool *mask, int64_t *output,
                                                        float *rand_f, const size_t num_count, const float keep_prob,
                                                        const size_t num_per_chan, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);
